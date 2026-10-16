"""Step 2: regularise noisy heights and fill missing data.

Roofs are piecewise planar: flat faces that meet at sharp ridges, hips, valleys and eaves.
The regulariser used here, second-order Total Generalized Variation (TGV²; Bredies,
Kunisch & Pock 2010), has exactly that model built in:

    u* = argmin_u,w  1/2 * sum(lambda * (u - f)^2)  +  alpha1 * |grad u - w|_1  +  alpha0 * |E(w)|_1

* ``w`` is a slope field. The alpha1 term lets grad u follow w, so planes cost nothing.
* E(w) = symmetrised gradient of w (change of slope). Its L1 norm is small when slope is
  constant on large areas and changes abruptly at a few lines: planes with sharp creases.
* ``lambda`` is a per-pixel data weight: 1 where the DSM is trusted, 0 where it is missing
  or an outlier. There the solution is filled from the surrounding planes.

By contrast, plain TV (alpha * |grad u|_1) favours piecewise *constant* surfaces, so it
flattens slopes (on a noise-free 35 deg roof the 5th-percentile slope drops to 27-31 deg)
and smooths faces much less than TGV (benchmark: 1.6 vs 0.5 deg face-normal error).
Gaussian and median filters blur walls and creases (see scripts/benchmark_cleaning.py).

The problem is convex and is solved with the Chambolle-Pock primal-dual algorithm.
"""

from __future__ import annotations

import logging

import numpy as np
from scipy import ndimage as ndi

log = logging.getLogger(__name__)


# ----------------------------------------------------------------- discrete operators
# Forward differences with Neumann boundaries for grad, backward differences for the
# symmetrised gradient. Each div operator is the negative adjoint of its gradient
# (checked in tests), which the primal-dual algorithm needs to converge.


def _grad(u: np.ndarray) -> np.ndarray:
    g = np.zeros((2, *u.shape), u.dtype)
    g[0, :, :-1] = u[:, 1:] - u[:, :-1]  # d/dx (columns)
    g[1, :-1, :] = u[1:, :] - u[:-1, :]  # d/dy (rows)
    return g


def _div(p: np.ndarray) -> np.ndarray:
    """Negative adjoint of _grad."""
    px, py = p
    d = np.zeros(px.shape, p.dtype)
    d[:, 0] += px[:, 0]
    d[:, 1:-1] += px[:, 1:-1] - px[:, :-2]
    d[:, -1] -= px[:, -2]
    d[0, :] += py[0, :]
    d[1:-1, :] += py[1:-1, :] - py[:-2, :]
    d[-1, :] -= py[-2, :]
    return d


def _bdx(a: np.ndarray) -> np.ndarray:
    d = np.zeros_like(a)
    d[:, 0] = a[:, 0]
    d[:, 1:-1] = a[:, 1:-1] - a[:, :-2]
    d[:, -1] = -a[:, -2]
    return d


def _bdy(a: np.ndarray) -> np.ndarray:
    d = np.zeros_like(a)
    d[0, :] = a[0, :]
    d[1:-1, :] = a[1:-1, :] - a[:-2, :]
    d[-1, :] = -a[-2, :]
    return d


def _fdx(a: np.ndarray) -> np.ndarray:
    d = np.zeros_like(a)
    d[:, :-1] = a[:, 1:] - a[:, :-1]
    return d


def _fdy(a: np.ndarray) -> np.ndarray:
    d = np.zeros_like(a)
    d[:-1, :] = a[1:, :] - a[:-1, :]
    return d


def _sym_grad(w: np.ndarray) -> np.ndarray:
    """E(w) as (xx, yy, xy) using backward differences. The adjoint is _div2 (negated).

    _bdx is exactly -(forward difference)^T, so E^T q = -(Fx qxx + Fy qxy, Fx qxy + Fy qyy).
    """
    wx, wy = w
    return np.stack([_bdx(wx), _bdy(wy), 0.5 * (_bdy(wx) + _bdx(wy))])


def _div2(q: np.ndarray) -> np.ndarray:
    """Negative adjoint of _sym_grad, with the xy component counted twice."""
    qxx, qyy, qxy = q
    return np.stack([_fdx(qxx) + _fdy(qxy), _fdx(qxy) + _fdy(qyy)])


# ----------------------------------------------------------------- TGV solver


def _mask_boundary_duals(p: np.ndarray | None, q: np.ndarray | None) -> None:
    """Zero the dual variables of differences that do not exist on the grid (in place).

    With Neumann boundaries the forward difference past the last column/row is forced to 0,
    so a sloping plane would be penalised there and bent near the border. Fixing those
    duals at 0 removes the constraint, and planes then have exactly zero TGV cost. For E(w),
    an entry is kept only where both slope samples it differences are real.
    """
    if p is not None:
        p[0, :, -1] = 0
        p[1, -1, :] = 0
    if q is not None:
        q[0, :, [0, -1]] = 0
        q[1, [0, -1], :] = 0
        q[2, [0, -1], :] = 0
        q[2, :, [0, -1]] = 0


def tgv_denoise(
    f: np.ndarray,
    alpha1: float,
    alpha0: float,
    weight: np.ndarray | None = None,
    n_iter: int = 1000,
    tol: float = 1e-6,
) -> tuple[np.ndarray, dict]:
    """Solve the TGV²-L2 problem above with Chambolle-Pock.

    ``f`` is in metres on a pixel grid (gradients are per pixel, so alpha1 is in metres and
    alpha0 in metres too: both scale with the noise level). NaNs in ``f`` get data weight 0
    and are filled. ``weight`` (0..1) can mark more pixels as untrusted.

    Stops when the RMS change of u per iteration falls below ``tol`` metres.
    Returns (u, info) where info holds the iteration count, final change and convergence.
    """
    lam = np.isfinite(f).astype(np.float64)
    if weight is not None:
        lam = lam * weight
    if lam.sum() == 0:
        raise ValueError("No valid data")
    # Start from a nearest-neighbour fill so missing pixels begin at plausible heights.
    idx = ndi.distance_transform_edt(lam == 0, return_distances=False, return_indices=True)
    f0 = np.where(np.isfinite(f), f, 0.0)[tuple(idx)]
    offset = float(np.mean(f0))  # work around zero for float precision
    f0 = f0 - offset

    u = f0.copy()
    w = np.zeros((2, *f.shape))
    p = np.zeros((2, *f.shape))
    q = np.zeros((3, *f.shape))
    u_bar, w_bar = u.copy(), w.copy()
    tau = sigma = 1.0 / np.sqrt(12.0)  # tau * sigma * ||K||^2 <= 1, ||K||^2 <= 12

    change = np.inf
    it = 0
    for it in range(1, n_iter + 1):
        # dual ascent + projection onto the alpha balls
        p += sigma * (_grad(u_bar) - w_bar)
        _mask_boundary_duals(p, q=None)
        p /= np.maximum(1.0, np.sqrt((p**2).sum(0)) / alpha1)
        q += sigma * _sym_grad(w_bar)
        _mask_boundary_duals(p=None, q=q)
        qn = np.sqrt(q[0] ** 2 + q[1] ** 2 + 2 * q[2] ** 2)
        q /= np.maximum(1.0, qn / alpha0)
        # primal descent + proximal step of the weighted data term
        u_old, w_old = u, w
        u = (u + tau * _div(p) + tau * lam * f0) / (1.0 + tau * lam)
        w = w + tau * (p + _div2(q))
        # over-relaxation
        u_bar = 2 * u - u_old
        w_bar = 2 * w - w_old
        if it % 25 == 0:
            change = float(np.sqrt(np.mean((u - u_old) ** 2)))
            if change < tol:
                break
    converged = change < tol
    if not converged:
        log.warning("TGV stopped after %d iterations (RMS change %.2e m > tol)", it, change)
    return u + offset, {"iterations": it, "final_rms_change_m": change, "converged": converged}


# ----------------------------------------------------------------- missing data / outliers


def fill_biharmonic(z: np.ndarray, missing: np.ndarray) -> np.ndarray:
    """Fill ``missing`` pixels by biharmonic (thin-plate) interpolation.

    Chosen by hold-out validation on real roof pixels (scripts/benchmark_cleaning.py, part
    B): it beat nearest, IDW (GDAL) and linear (Delaunay) for every hole size, and TGV for
    holes up to 0.75 m radius. Each connected hole is solved in its own bounding box.
    """
    if not missing.any():
        return z.copy()
    from skimage.restoration import inpaint_biharmonic

    return inpaint_biharmonic(np.where(missing, 0.0, z), missing, split_into_regions=True)


def detect_outliers(
    z: np.ndarray,
    u: np.ndarray,
    valid: np.ndarray,
    k: float = 6.0,
    window_px: int = 7,
    max_blob_px: int = 25,
    background_px: int = 11,
    same_object_frac: float = 0.5,
) -> tuple[np.ndarray, float]:
    """Flag isolated spikes/pits: data ``z`` versus regularised surface ``u``.

    1. Candidates, by a local Hampel test on r = z - u: |r - median_local(r)| >
       k * max(local robust sigma, global robust sigma), robust sigma = 1.4826 * MAD.
       The local sigma stops real edges and rough vegetation (large local spread) from
       being flagged; the global floor stops near-perfect areas (MAD ~ 0) from flagging
       tiny noise.
    2. Isolation, on the raw heights. For each candidate blob, its amplitude A is its
       largest deviation from the local background (``background_px`` median of z). The
       "same object" region is every connected pixel deviating the same way by at least
       ``same_object_frac`` * A. The blob is an outlier only if that region has at most
       ``max_blob_px`` pixels.
       - A spike rises alone above a wavy roof (neighbours deviate by about 2 cm, much
         less than half the spike).
       - The corner of a real box or chimney shares its full height with the object's
         whole outline, so the region is large and the corner is kept as data. This
         matters because regularisation rounds corners slightly, so they stand out in r.

    Returns (mask, global robust sigma of r).
    """
    r = z - u
    vals = r[valid]
    sigma_g = float(1.4826 * np.median(np.abs(vals - np.median(vals))))
    rr = np.where(valid, r, 0.0)
    med = ndi.median_filter(rr, window_px)
    dev = np.abs(rr - med)
    sigma_l = 1.4826 * ndi.median_filter(dev, window_px)
    cand = valid & (dev > k * np.maximum(sigma_l, sigma_g))
    if not cand.any():
        return cand, sigma_g

    zz = np.where(valid, z, np.nanmedian(z[valid]))
    height = zz - ndi.median_filter(zz, background_px)  # signed deviation from background
    lab, n = ndi.label(cand, structure=np.ones((3, 3)))
    out = np.zeros_like(cand)
    pad = int(np.sqrt(max_blob_px)) + 2  # an isolated object cannot reach further
    for i, sl in enumerate(ndi.find_objects(lab), start=1):
        rs = slice(max(sl[0].start - pad, 0), sl[0].stop + pad)
        cs = slice(max(sl[1].start - pad, 0), sl[1].stop + pad)
        blob = lab[rs, cs] == i
        h = height[rs, cs]
        amp = h[blob][np.argmax(np.abs(h[blob]))]
        same = valid[rs, cs] & (np.sign(amp) * h >= same_object_frac * abs(amp))
        same |= blob
        reg_lab, _ = ndi.label(same, structure=np.ones((3, 3)))
        region = np.isin(reg_lab, np.unique(reg_lab[blob]))
        touches_edge = (
            region[0].any() or region[-1].any() or region[:, 0].any() or region[:, -1].any()
        )
        if region.sum() <= max_blob_px and not touches_edge:
            out[rs, cs] |= blob
    return out, sigma_g


def excess_green(rgb: np.ndarray) -> np.ndarray:
    """Excess-green vegetation index ExG = 2g - r - b on chromatic coordinates.

    r, g, b = R/(R+G+B) ... remove brightness, so shadows and sunlit foliage score alike.
    ``rgb`` is (3, rows, cols) or (rows, cols, 3).
    """
    a = np.asarray(rgb, dtype=np.float64)
    if a.shape[-1] == 3 and a.shape[0] != 3:
        a = np.moveaxis(a, -1, 0)
    s = a.sum(0) + 1e-9
    r, g, b = a / s
    return 2 * g - r - b
