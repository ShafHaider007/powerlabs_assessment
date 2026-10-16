"""Figures for each pipeline stage. Only plotting here, no analysis."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

if TYPE_CHECKING:
    from roofsolar.pipeline import TerrainResult

DPI = 130


def _extent(shape: tuple[int, int], pixel: tuple[float, float]) -> tuple[float, ...]:
    """Image extent in local metres from the AOI's top-left corner (north up)."""
    rows, cols = shape
    return (0, cols * pixel[0], rows * pixel[1], 0)


def _style(ax, title: str) -> None:
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("east (m)")
    ax.set_ylabel("south (m)")
    ax.annotate(
        "N",
        xy=(0.95, 0.95),
        xytext=(0.95, 0.83),
        xycoords="axes fraction",
        ha="center",
        fontsize=10,
        fontweight="bold",
        arrowprops=dict(arrowstyle="-|>", color="k"),
    )


def _outline(
    ax, mask: np.ndarray, pixel: tuple[float, float], color: str = "yellow", lw: float = 1.2
) -> None:
    """Draw a mask boundary using explicit pixel-centre coordinates (local metres).

    Explicit coordinates avoid matplotlib's origin/extent conventions for contour, which
    differ from imshow and silently flipped the outline vertically.
    """
    rows, cols = mask.shape
    x = (np.arange(cols) + 0.5) * pixel[0]
    y = (np.arange(rows) + 0.5) * pixel[1]
    ax.contour(x, y, mask.astype(float), levels=[0.5], colors=color, linewidths=lw)


def _aoi_outline(ax, aoi: np.ndarray, ext) -> None:
    rows, cols = aoi.shape
    _outline(ax, aoi, (ext[1] / cols, ext[2] / rows))


def _compass_wheel(fig, rect, cmap) -> None:
    """Small colour wheel showing which hue means which facing direction."""
    ax = fig.add_axes(rect, projection="polar")
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    theta = np.linspace(0, 2 * np.pi, 361)
    r = np.linspace(0.5, 1, 2)
    t, _ = np.meshgrid(theta, r)
    ax.pcolormesh(theta, r, np.degrees(t), cmap=cmap, vmin=0, vmax=360, shading="auto")
    ax.set_yticks([])
    ax.set_xticks(np.radians([0, 90, 180, 270]), ["N", "E", "S", "W"], fontsize=8)
    ax.spines["polar"].set_visible(False)


def _panel_elevation(ax, res, hs, ext):
    ax.imshow(np.where(res.aoi, hs, np.nan), cmap="gray", extent=ext)
    im = ax.imshow(res.elevation.data, cmap="terrain", alpha=0.6, extent=ext)
    _style(ax, "Elevation (m a.s.l.) over hillshade")
    return im


def _panel_slope(ax, res, ext):
    im = ax.imshow(res.slope.data, cmap="magma", vmin=0, vmax=90, extent=ext)
    _style(ax, "Slope (deg)")
    return im


def _panel_aspect(ax, res, ext, flat_deg):
    ax.imshow(
        np.where(res.aoi & np.isnan(res.aspect.data), 0.85, np.nan),
        cmap="gray",
        vmin=0,
        vmax=1,
        extent=ext,
    )
    im = ax.imshow(
        res.aspect.data, cmap="hsv", vmin=0, vmax=360, extent=ext, interpolation="nearest"
    )
    _style(ax, f"Aspect (deg from true N); grey = flat (< {flat_deg:g} deg)")
    return im


def _panel_ndsm(ax, res, ext):
    im = ax.imshow(
        res.ndsm.data,
        cmap="viridis",
        vmin=0,
        vmax=np.nanpercentile(res.ndsm.data, 99.5),
        extent=ext,
    )
    _style(ax, "Height above ground (m)")
    return im


def plot_terrain(
    res: TerrainResult, hs: np.ndarray, roof_px: np.ndarray, ground_px: np.ndarray, out: Path
) -> None:
    """Write all step-1 figures."""
    pixel = res.slope.pixel_size
    ext = _extent(res.aoi.shape, pixel)
    flat_deg = res.stats["flat_slope_threshold_deg"]

    # 01 inputs
    fig, ax = plt.subplots(1, 2, figsize=(15, 6.5))
    ax[0].imshow(res.rgb, extent=ext)
    _aoi_outline(ax[0], res.aoi, ext)
    _style(ax[0], "RGB with area of interest (yellow)")
    fig.colorbar(_panel_elevation(ax[1], res, hs, ext), ax=ax[1], shrink=0.8)
    fig.tight_layout()
    fig.savefig(out / "01_inputs_elevation.png", dpi=DPI)
    plt.close(fig)

    # 02 slope
    fig, ax = plt.subplots(figsize=(9, 7))
    fig.colorbar(_panel_slope(ax, res, ext), ax=ax, shrink=0.8, label="deg")
    fig.tight_layout()
    fig.savefig(out / "02_slope.png", dpi=DPI)
    plt.close(fig)

    # 03 aspect
    fig, ax = plt.subplots(figsize=(9, 7))
    _panel_aspect(ax, res, ext, flat_deg)
    fig.tight_layout()
    _compass_wheel(fig, [0.78, 0.12, 0.17, 0.17], "hsv")
    fig.savefig(out / "03_aspect.png", dpi=DPI)
    plt.close(fig)

    # 04 height above ground
    fig, ax = plt.subplots(figsize=(9, 7))
    fig.colorbar(_panel_ndsm(ax, res, ext), ax=ax, shrink=0.8, label="m")
    fig.tight_layout()
    fig.savefig(out / "04_height_above_ground.png", dpi=DPI)
    plt.close(fig)

    # 05 roof diagnostics: aspect rose, slope histogram, ground slope noise
    fig = plt.figure(figsize=(16, 5))
    axr = fig.add_subplot(1, 3, 1, projection="polar")
    asp = res.aspect.data[roof_px]
    asp = asp[~np.isnan(asp)]
    counts, edges = np.histogram(asp, bins=72, range=(0, 360))
    axr.bar(
        np.radians(edges[:-1] + 2.5),
        counts,
        width=np.radians(5),
        color=plt.cm.hsv(edges[:-1] / 360),
        edgecolor="k",
        linewidth=0.3,
    )
    axr.set_theta_zero_location("N")
    axr.set_theta_direction(-1)
    axr.set_yticklabels([])
    for p in res.stats["roof_candidates"]["aspect_peaks_deg"]:
        axr.annotate(
            f"{p['azimuth_deg']:.0f}°",
            xy=(np.radians(p["azimuth_deg"]), counts.max()),
            ha="center",
            fontsize=9,
            fontweight="bold",
        )
    axr.set_title("Roof-candidate aspect (true N)", pad=18)

    ax2 = fig.add_subplot(1, 3, 2)
    ax2.hist(res.slope.data[roof_px], bins=60, range=(0, 90), color="tab:orange")
    ax2.axvline(
        res.stats["roof_candidates"]["slope_deg"]["median"],
        color="k",
        ls="--",
        label=f"median {res.stats['roof_candidates']['slope_deg']['median']:.1f}°",
    )
    ax2.set(xlabel="slope (deg)", ylabel="pixels", title="Roof-candidate slope")
    ax2.legend()

    ax3 = fig.add_subplot(1, 3, 3)
    gs = res.slope.data[ground_px]
    ax3.hist(gs[~np.isnan(gs)], bins=60, range=(0, 15), color="tab:green")
    ax3.axvline(flat_deg, color="r", ls="--", label=f"flat threshold {flat_deg:g}°")
    ax3.axvline(
        res.stats["flat_ground_slope_deg"]["p95"],
        color="k",
        ls=":",
        label=f"ground p95 {res.stats['flat_ground_slope_deg']['p95']:.1f}°",
    )
    ax3.set(xlabel="slope (deg)", ylabel="pixels", title="Slope on flat ground (noise)")
    ax3.legend()
    fig.tight_layout()
    fig.savefig(out / "05_roof_diagnostics.png", dpi=DPI)
    plt.close(fig)

    # 00 summary (for README / video)
    fig, ax = plt.subplots(2, 3, figsize=(20, 12))
    ax[0, 0].imshow(res.rgb, extent=ext)
    _aoi_outline(ax[0, 0], res.aoi, ext)
    _style(ax[0, 0], "RGB + AOI")
    fig.colorbar(_panel_elevation(ax[0, 1], res, hs, ext), ax=ax[0, 1], shrink=0.7)
    fig.colorbar(_panel_ndsm(ax[0, 2], res, ext), ax=ax[0, 2], shrink=0.7)
    fig.colorbar(_panel_slope(ax[1, 0], res, ext), ax=ax[1, 0], shrink=0.7)
    _panel_aspect(ax[1, 1], res, ext, flat_deg)
    ax[1, 2].imshow(np.where(res.aoi, hs, np.nan), cmap="gray", extent=ext)
    ax[1, 2].imshow(np.where(roof_px, 1.0, np.nan), cmap="autumn", alpha=0.5, extent=ext)
    _style(ax[1, 2], "Roof candidates (preview, refined in step 3)")
    fig.tight_layout()
    fig.savefig(out / "00_summary.png", dpi=100)
    plt.close(fig)


def plot_provenance(res: TerrainResult, off_grid: np.ndarray, out: Path) -> None:
    """Show where the provider re-processed the DSM (heights off the 1 cm grid)."""
    ext = _extent(res.aoi.shape, res.slope.pixel_size)
    fig, ax = plt.subplots(1, 2, figsize=(15, 6.5))
    ax[0].imshow(res.rgb, extent=ext)
    ax[0].imshow(np.where(off_grid, 1.0, np.nan), cmap="autumn", alpha=0.45, extent=ext)
    _aoi_outline(ax[0], res.aoi, ext)
    _style(ax[0], "Re-processed DSM regions (heights not on 1 cm grid)")
    frac = (res.elevation.data * 100) % 1
    ax[1].imshow(
        np.where(res.aoi, np.minimum(frac, 1 - frac), np.nan),
        cmap="magma",
        vmin=0,
        vmax=0.5,
        extent=ext,
        interpolation="nearest",
    )
    _style(ax[1], "Distance of each height from the 1 cm grid (cm)")
    fig.tight_layout()
    fig.savefig(out / "06_data_provenance.png", dpi=DPI)
    plt.close(fig)


def _profile_coords(col, row, az_deg, length_m, pixel, step_m=0.05):
    t = np.arange(-length_m / 2, length_m / 2 + 1e-9, step_m)
    a = np.radians(az_deg)
    cols = col + t * np.sin(a) / pixel
    rows = row - t * np.cos(a) / pixel
    return t, rows, cols


def plot_cleaning(
    prev, res, hs_before, hs_after, scatter_before, scatter_after, profiles, out: Path
) -> None:
    """Write all step-2 figures: before/after maps, profiles, outliers/vegetation, noise."""
    from scipy import ndimage as ndi

    aoi = prev.aoi
    pixel = res.slope.pixel_size
    ext = _extent(aoi.shape, pixel)
    flat = res.stats["flat_slope_threshold_deg"]["after"]

    # 00 summary: before / after
    fig, ax = plt.subplots(2, 3, figsize=(20, 12))
    ax[0, 0].imshow(np.where(aoi, hs_before, np.nan), cmap="gray", extent=ext)
    _style(ax[0, 0], "Hillshade: raw DSM")
    ax[0, 1].imshow(np.where(aoi, hs_after, np.nan), cmap="gray", extent=ext)
    _style(ax[0, 1], "Hillshade: cleaned DSM (TGV²)")
    im = ax[0, 2].imshow(res.removed.data * 100, cmap="RdBu_r", vmin=-5, vmax=5, extent=ext)
    fig.colorbar(im, ax=ax[0, 2], shrink=0.7, label="cm")
    _style(ax[0, 2], "Removed by cleaning: raw - clean")
    for a, data, title in (
        (ax[1, 0], prev.slope.data, "Slope: raw"),
        (ax[1, 1], res.slope.data, "Slope: cleaned"),
    ):
        im = a.imshow(data, cmap="magma", vmin=0, vmax=90, extent=ext)
        fig.colorbar(im, ax=a, shrink=0.7, label="deg")
        _style(a, title)
    ax[1, 2].imshow(
        np.where(aoi & np.isnan(res.aspect.data), 0.85, np.nan),
        cmap="gray",
        vmin=0,
        vmax=1,
        extent=ext,
    )
    ax[1, 2].imshow(
        res.aspect.data, cmap="hsv", vmin=0, vmax=360, extent=ext, interpolation="nearest"
    )
    _style(ax[1, 2], f"Aspect: cleaned (grey = flat < {flat:g}°)")
    fig.tight_layout()
    fig.savefig(out / "00_summary.png", dpi=100)
    plt.close(fig)

    # 01 profiles
    fig = plt.figure(figsize=(20, 12))
    gsp = fig.add_gridspec(len(profiles), 2, width_ratios=[3, 1])
    axm = fig.add_subplot(gsp[:, 1])
    axm.imshow(prev.rgb, extent=ext)
    _style(axm, "Profile lines")
    for i, (col, row, az, length, label) in enumerate(profiles):
        t, rows, cols = _profile_coords(col, row, az, length, pixel[0])
        raw = ndi.map_coordinates(np.nan_to_num(prev.elevation.data), [rows, cols], order=1)
        cln = ndi.map_coordinates(np.nan_to_num(res.dsm_clean.data), [rows, cols], order=1)
        # keep samples whose 4 bilinear neighbours are all inside the AOI (no mixing with
        # the 0 placeholder outside it)
        inside = ndi.map_coordinates(aoi.astype(float), [rows, cols], order=1) > 0.999
        raw, cln = np.where(inside, raw, np.nan), np.where(inside, cln, np.nan)
        a = fig.add_subplot(gsp[i, 0])
        a.plot(t, raw, color="0.6", lw=2.2, label="raw")
        a.plot(t, cln, "r-", lw=0.9, label="cleaned")
        a.set_title(label, fontsize=10)
        a.set_ylabel("z (m)")
        a.grid(alpha=0.3)
        k = int(np.nanargmax(cln))
        ins = a.inset_axes([0.01, 0.55, 0.22, 0.42])
        sl = slice(max(k - 30, 0), k + 30)
        ins.plot(t[sl], raw[sl], color="0.6", lw=2.2)
        ins.plot(t[sl], cln[sl], "r-", lw=1)
        ins.set_title("zoom at highest point", fontsize=7)
        ins.tick_params(labelsize=6)
        axm.plot(cols * pixel[0], rows * pixel[1], lw=2)
        axm.annotate(
            label[0], (cols[-1] * pixel[0], rows[-1] * pixel[1]), color="w", fontweight="bold"
        )
        if i == 0:
            a.legend(loc="upper right")
    a.set_xlabel("distance along profile (m)")
    fig.tight_layout()
    fig.savefig(out / "01_profiles.png", dpi=100)
    plt.close(fig)

    # 02 outliers and vegetation
    fig, ax = plt.subplots(1, 2, figsize=(16, 6.5))
    ax[0].imshow(prev.rgb, extent=ext)
    _outline(ax[0], res.vegetation, pixel, color="lime", lw=0.8)
    rr, cc = np.nonzero(res.outliers)
    ax[0].scatter(
        (cc + 0.5) * pixel[0],
        (rr + 0.5) * pixel[1],
        s=14,
        facecolors="none",
        edgecolors="red",
        linewidths=1,
    )
    o = res.stats["outliers"]
    _style(ax[0], f"Vegetation (green) and {o['count']} isolated outlier px (red)")
    im = ax[1].imshow(
        np.where(res.filled, res.removed.data * 100, np.nan),
        cmap="RdBu_r",
        vmin=-30,
        vmax=30,
        extent=ext,
        interpolation="nearest",
    )
    ax[1].imshow(np.where(aoi, 0.9, np.nan), cmap="gray", vmin=0, vmax=1, extent=ext, alpha=0.25)
    fig.colorbar(im, ax=ax[1], shrink=0.8, label="raw - clean (cm)")
    _style(ax[1], "Corrections at replaced (outlier / missing) pixels")
    fig.tight_layout()
    fig.savefig(out / "02_outliers_vegetation.png", dpi=DPI)
    plt.close(fig)

    # 03 noise reduction
    s = res.stats
    fig, ax = plt.subplots(1, 2, figsize=(14, 4.8))
    g = prev.ground_px
    bins = np.linspace(0, 15, 61)
    ax[0].hist(prev.slope.data[g], bins=bins, alpha=0.5, label="raw", color="0.4")
    ax[0].hist(res.slope.data[g], bins=bins, alpha=0.6, label="cleaned", color="tab:red")
    b, a = s["open_ground_slope_deg"]["before_p50_p95"], s["open_ground_slope_deg"]["after_p50_p95"]
    ax[0].set(
        xlabel="slope (deg)",
        ylabel="pixels",
        title=f"Open flat ground slope (pure noise): p50 {b[0]}° → {a[0]}°, p95 {b[1]}° → {a[1]}°",
    )
    ax[0].legend()
    bins = np.linspace(0, max(8, float(np.max(scatter_before))), 41)
    ax[1].hist(scatter_before, bins=bins, alpha=0.5, label="raw", color="0.4")
    ax[1].hist(scatter_after, bins=bins, alpha=0.6, label="cleaned", color="tab:red")
    rs = s["roof_normal_scatter_deg"]
    ax[1].set(
        xlabel="median normal deviation within tile (deg)",
        ylabel="tiles",
        title=f"Roof-face normal scatter ({rs['tiles']} planar 1.5 m tiles): "
        f"{rs['before_median']}° → {rs['after_median']}°",
    )
    ax[1].legend()
    fig.tight_layout()
    fig.savefig(out / "03_noise_reduction.png", dpi=DPI)
    plt.close(fig)


EDGE_COLORS = {"ridge": "red", "hip": "orange", "valley": "deepskyblue", "step": "magenta"}


def _label_image(labels: np.ndarray, seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    cmap = rng.random((labels.max() + 1, 3)) * 0.75 + 0.25
    cmap[0] = 1.0
    return cmap[labels]


def _zoom_window(mask: np.ndarray, pixel, margin_m: float = 3.0):
    rr, cc = np.nonzero(mask)
    m = margin_m / pixel[0]
    return (
        (cc.min() - m) * pixel[0],
        (cc.max() + m) * pixel[0],
        (rr.max() + m) * pixel[1],
        (rr.min() - m) * pixel[1],
    )


def plot_faces(t, prev, res, trace, profiles, out: Path) -> None:
    """Write all step-3 figures: faces, crease map, reconstruction, residual, stages, profiles."""
    from scipy import ndimage as ndi

    from roofsolar import terrain as T

    pixel = res.planar.pixel_size
    ext = _extent(res.labels.shape, pixel)
    zoom = _zoom_window(res.candidate, pixel)
    labels, cand = res.labels, res.candidate

    def setzoom(a):
        a.set_xlim(zoom[0], zoom[1])
        a.set_ylim(zoom[2], zoom[3])

    def hs(z):
        filled = np.where(np.isnan(z), prev.dsm_clean.data, z)
        filled = np.where(np.isnan(filled), np.nanmin(filled), filled)
        return T.hillshade(*T.horn_gradients(filled, *pixel), azimuth_deg=315 - t.rotation_deg)

    fig, ax = plt.subplots(2, 3, figsize=(21, 14))
    # (0,0) RGB + outlines + ids
    ax[0, 0].imshow(prev.terrain.rgb, extent=ext)
    for k in res.planes:
        _outline(ax[0, 0], labels == k, pixel, color="white", lw=0.7)
        r, c = ndi.center_of_mass(labels == k)
        ax[0, 0].text(
            (c + 0.5) * pixel[0],
            (r + 0.5) * pixel[1],
            str(k),
            color="yellow",
            fontsize=9,
            fontweight="bold",
            ha="center",
            va="center",
        )
    _style(ax[0, 0], f"{len(res.planes)} roof faces on the RGB")
    # (0,1) faces coloured by aspect
    asp = np.full(labels.shape, np.nan)
    for f in res.face_table:
        asp[labels == f["id"]] = f["aspect_deg"]
    ax[0, 1].imshow(
        np.where(cand & (labels == 0), 0.6, np.nan), cmap="gray", vmin=0, vmax=1, extent=ext
    )
    ax[0, 1].imshow(asp, cmap="hsv", vmin=0, vmax=360, extent=ext, interpolation="nearest")
    _style(ax[0, 1], "Faces by aspect (grey = building, not a face)")
    # (0,2) crease map
    ax[0, 2].imshow(hs(res.roof_surface.data), cmap="gray", extent=ext)
    for e in res.edges:
        ma, mb = labels == e.a, labels == e.b
        bnd = (ma & ndi.binary_dilation(mb)) | (mb & ndi.binary_dilation(ma))
        rr, cc = np.nonzero(bnd)
        ax[0, 2].scatter(
            (cc + 0.5) * pixel[0],
            (rr + 0.5) * pixel[1],
            s=1.5,
            c=EDGE_COLORS[e.kind],
            marker="s" if e.consistent is not False else "x",
        )
    for kind, col in EDGE_COLORS.items():
        ax[0, 2].plot([], [], "s", color=col, label=f"{kind} ({res.stats['edges'][kind]})")
    ax[0, 2].legend(loc="lower left", fontsize=8)
    share = res.stats["crease_consistent_share"]
    _style(
        ax[0, 2],
        f"Edges by type; creases consistent with plane intersection: {share * 100:.0f}% of length",
    )
    # (1,0) cleaned DSM, (1,1) reconstructed, (1,2) residual
    ax[1, 0].imshow(hs(prev.dsm_clean.data), cmap="gray", extent=ext)
    _style(ax[1, 0], "Cleaned DSM (step 2): rounded ridges, smeared walls")
    ax[1, 1].imshow(hs(res.roof_surface.data), cmap="gray", extent=ext)
    _style(ax[1, 1], "Reconstructed roof: planes, sharp creases, rebuilt eaves")
    im = ax[1, 2].imshow(res.residual.data * 100, cmap="RdBu_r", vmin=-15, vmax=15, extent=ext)
    ax[1, 2].imshow(
        np.where(cand & (labels == 0), 0.3, np.nan), cmap="gray", vmin=0, vmax=1, extent=ext
    )
    fig.colorbar(im, ax=ax[1, 2], shrink=0.7, label="raw DSM - plane (cm)")
    _style(ax[1, 2], "Residual to face planes (dark grey = objects / unassigned)")
    for a in ax.ravel():
        setzoom(a)
    fig.tight_layout()
    fig.savefig(out / "00_summary.png", dpi=100)
    plt.close(fig)

    # 01 stages
    stages = [
        ("grown", trace["grown"]),
        ("merged + filtered", trace["merged"]),
        ("assigned (k-planes)", trace["assigned"]),
        ("final (+ eaves)", labels),
    ]
    fig, ax = plt.subplots(1, 4, figsize=(24, 7))
    for a, (name, lab) in zip(ax, stages, strict=True):
        img = _label_image(lab)
        img[~cand] = 0.85
        a.imshow(img, extent=ext, interpolation="nearest")
        _style(a, f"{name}: {len(np.unique(lab)) - 1} segments")
        setzoom(a)
    fig.tight_layout()
    fig.savefig(out / "01_stages.png", dpi=90)
    plt.close(fig)

    # 02 profiles: raw vs cleaned vs reconstructed
    fig, ax = plt.subplots(len(profiles), 1, figsize=(16, 4.2 * len(profiles)))
    ax = np.atleast_1d(ax)
    for a, (col, row, az, length, label) in zip(ax, profiles, strict=True):
        tt, rows, cols = _profile_coords(col, row, az, length, pixel[0])
        inside = ndi.map_coordinates(t.aoi.astype(float), [rows, cols], order=1) > 0.999
        raw = ndi.map_coordinates(np.nan_to_num(t.elevation.data), [rows, cols], order=1)
        cln = ndi.map_coordinates(np.nan_to_num(prev.dsm_clean.data), [rows, cols], order=1)
        lab = ndi.map_coordinates(labels, [rows, cols], order=0)
        # evaluate each face plane exactly at the sample point (no pixel staircase)
        xs, ys = (cols + 0.5) * pixel[0], -(rows + 0.5) * pixel[1]
        rec = np.full(tt.shape, np.nan)
        for k, pl in res.planes.items():
            m = lab == k
            rec[m] = pl.z_at(xs[m], ys[m])
        a.plot(tt, np.where(inside, raw, np.nan), color="0.65", lw=2.5, label="raw DSM")
        a.plot(tt, np.where(inside, cln, np.nan), "b-", lw=0.8, label="cleaned (step 2)")
        a.plot(
            tt,
            np.where(inside & (lab > 0), rec, np.nan),
            "r-",
            lw=1.4,
            label="reconstructed planes",
        )
        a.set_title(label, fontsize=10)
        a.set_ylabel("z (m)")
        a.grid(alpha=0.3)
        good = inside & (lab > 0)
        if good.any():  # zoom on the highest reconstructed point (a ridge)
            kk = int(np.nanargmax(np.where(good, rec, np.nan)))
            sl = slice(max(kk - 25, 0), kk + 25)
            ins = a.inset_axes([0.01, 0.52, 0.2, 0.45])
            ins.plot(tt[sl], raw[sl], color="0.65", lw=2.5)
            ins.plot(tt[sl], cln[sl], "b-", lw=0.8)
            ins.plot(tt[sl], np.where(good, rec, np.nan)[sl], "r-", lw=1.4)
            ins.set_title("zoom: highest ridge", fontsize=7)
            ins.tick_params(labelsize=6)
    ax[0].legend(loc="upper right")
    ax[-1].set_xlabel("distance along profile (m)")
    fig.tight_layout()
    fig.savefig(out / "02_profiles.png", dpi=100)
    plt.close(fig)


OBSTRUCTION_COLORS = {
    1: "orange",
    2: "red",
    3: "magenta",
    4: "green",
    5: "cyan",
    6: "lime",
    7: "yellow",
    8: "blue",
}


def plot_obstructions(t, fr, res, det, ref_path, out: Path) -> None:
    """Write step-4 figures: detections on RGB, evidence maps, usable area."""
    import json as _json

    from matplotlib.colors import ListedColormap

    from roofsolar.obstructions import CLASSES

    pixel = res.hag.pixel_size
    ext = _extent(res.classes.shape, pixel)
    zoom = _zoom_window(fr.candidate, pixel)
    faces = fr.labels > 0

    def setzoom(a):
        a.set_xlim(zoom[0], zoom[1])
        a.set_ylim(zoom[2], zoom[3])

    ref = None
    if ref_path and Path(ref_path).exists():
        ref = _json.loads(Path(ref_path).read_text())
    found = {
        o["name"]: o["found"] for o in (res.stats["reference_recall"] or {}).get("objects", [])
    }

    def markers(a):
        if not ref:
            return
        for o in ref["objects"]:
            ok = found.get(o["name"], False)
            a.plot(
                (o["col"] + 0.5) * pixel[0],
                (o["row"] + 0.5) * pixel[1],
                marker="o" if ok else "X",
                ms=11,
                mfc="none",
                mec="white" if ok else "red",
                mew=1.8,
            )

    fig, ax = plt.subplots(2, 2, figsize=(18, 17))
    # (0,0) RGB + obstruction outlines by class
    ax[0, 0].imshow(t.rgb, extent=ext)
    for code, col in OBSTRUCTION_COLORS.items():
        m = res.classes == code
        if m.any():
            _outline(ax[0, 0], m, pixel, color=col, lw=1.4)
            n = res.stats["by_class"].get(CLASSES[code], {}).get("count", 0)
            ax[0, 0].plot([], [], color=col, lw=3, label=f"{CLASSES[code]} ({n})")
    markers(ax[0, 0])
    ax[0, 0].plot([], [], "o", mfc="none", mec="white", label="reference object: found")
    ax[0, 0].plot([], [], "X", mfc="none", mec="red", label="reference object: missed")
    ax[0, 0].legend(loc="lower left", fontsize=8, framealpha=0.85)
    rc = res.stats["reference_recall"]
    _style(
        ax[0, 0],
        "Obstructions" + (f" (reference recall {rc['found']}/{rc['total']})" if rc else ""),
    )
    # (0,1) height above roof plane (raw DSM)
    im = ax[0, 1].imshow(
        np.where(fr.candidate, res.hag.data, np.nan), cmap="RdBu_r", vmin=-0.5, vmax=0.5, extent=ext
    )
    _outline(ax[0, 1], det.raised, pixel, color="k", lw=0.8)
    fig.colorbar(im, ax=ax[0, 1], shrink=0.7, label="raw DSM - roof plane (m)")
    _style(ax[0, 1], "Evidence 1: height above roof plane (black = raised > 0.15 m)")
    # (1,0) colour distance
    im = ax[1, 0].imshow(
        np.where(faces, det.colour_dist, np.nan), cmap="magma", vmin=0, vmax=10, extent=ext
    )
    _outline(ax[1, 0], det.colour, pixel, color="lime", lw=1)
    _outline(ax[1, 0], det.glass, pixel, color="cyan", lw=1)
    fig.colorbar(im, ax=ax[1, 0], shrink=0.7, label="colour distance from face tiles")
    _style(ax[1, 0], "Evidence 2: colour anomaly (lime) and glass faces (cyan)")
    # (1,1) usable area
    img = np.zeros(res.classes.shape)
    img[fr.candidate & ~faces] = 1
    img[res.keepout] = 2
    img[res.classes > 0] = 3
    img[res.usable] = 4
    cmap = ListedColormap(["white", "lightgrey", "salmon", "firebrick", "seagreen"])
    ax[1, 1].imshow(
        np.where(fr.candidate | faces, img, np.nan),
        cmap=cmap,
        vmin=-0.5,
        vmax=4.5,
        extent=ext,
        interpolation="nearest",
    )
    for lab_, col in (
        ("usable", "seagreen"),
        ("keep-out buffer", "salmon"),
        ("obstruction", "firebrick"),
        ("building, not a face", "lightgrey"),
    ):
        ax[1, 1].plot([], [], "s", color=col, ms=10, label=lab_)
    ax[1, 1].legend(loc="lower left", fontsize=9)
    s = res.stats
    _style(
        ax[1, 1],
        f"Usable roof: {s['usable_3d_m2']:.0f} m² sloped "
        f"({100 * s['usable_share_of_faces']:.0f}% of face area)",
    )
    for a in ax.ravel():
        setzoom(a)
    fig.tight_layout()
    fig.savefig(out / "00_summary.png", dpi=100)
    plt.close(fig)


VERTEX_STYLE = {
    "crease": ("o", "black", 9),
    "junction": ("*", "red", 90),
    "unresolved_junction": ("X", "darkviolet", 70),
    "step": ("s", "magenta", 14),
}


def _shade(colors: np.ndarray, normals: np.ndarray) -> np.ndarray:
    """Lambert shading for the 3D renders (light from the NW, 45 deg up)."""
    light = np.array([-0.5, 0.5, 0.707])
    light /= np.linalg.norm(light)
    k = 0.45 + 0.55 * np.clip(np.abs(normals @ light), 0, 1)
    return np.clip(colors / 255 * k[:, None], 0, 1)


def plot_model3d(res, fr, t, out: Path) -> None:
    """Plan view of the vector model (with vertex types) and two 3D renders."""
    from matplotlib.patches import Polygon as MplPolygon
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    v, f, col = res.vertices, res.triangles, res.colors
    tri = v[f]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
    shaded = _shade(col.astype(float), n)

    fig = plt.figure(figsize=(24, 9))
    ax = fig.add_subplot(1, 3, 1)
    for k, plist in res.model.polygons.items():
        idx = np.flatnonzero(res.face_id == k)
        fc = col[idx[0]] / 255 if len(idx) else (0.8, 0.8, 0.8)
        for p in plist:
            ax.add_patch(
                MplPolygon(np.array(p.exterior.coords), closed=True, fc=fc, ec="k", lw=0.6)
            )
            for r in p.interiors:
                ax.add_patch(
                    MplPolygon(np.array(r.coords), closed=True, fc="white", ec="k", lw=0.5)
                )
    for kind, (mk, c, s) in VERTEX_STYLE.items():
        pts = np.array([vv.xy for vv in res.model.vertices.values() if vv.kind == kind])
        if len(pts):
            ax.scatter(
                pts[:, 0], pts[:, 1], marker=mk, c=c, s=s, zorder=5, label=f"{kind} ({len(pts)})"
            )
    ax.set_aspect("equal")
    ax.autoscale_view()
    ax.legend(loc="lower left", fontsize=8)
    st = res.stats
    ax.set_title(
        f"Vector model: {st['faces']} faces, {st['polygon_vertices']} vertices\n"
        f"crease/junction vertices consistent to {st['crease_vertex_gap_m_max']:.1e} m",
        fontsize=10,
    )
    ax.set_xlabel("east (m)")
    ax.set_ylabel("north (m)")

    for i, (elev, azim, title) in enumerate(
        ((35, -130, "3D view from the south-west"), (35, 50, "3D view from the north-east"))
    ):
        a = fig.add_subplot(1, 3, i + 2, projection="3d")
        # edges in the face colour: avoids antialiasing seams between triangles
        a.add_collection3d(
            Poly3DCollection(tri, facecolors=shaded, edgecolors=shaded, linewidths=0.3)
        )
        lo, hi = v.min(0), v.max(0)
        a.set_xlim(lo[0], hi[0])
        a.set_ylim(lo[1], hi[1])
        a.set_zlim(lo[2], lo[2] + max(hi[0] - lo[0], hi[1] - lo[1]) * 0.5)
        a.set_box_aspect((hi[0] - lo[0], hi[1] - lo[1], max(hi[0] - lo[0], hi[1] - lo[1]) * 0.5))
        a.view_init(elev=elev, azim=azim)
        a.set_title(title, fontsize=10)
        a.set_axis_off()
    fig.tight_layout()
    fig.savefig(out / "00_summary.png", dpi=100)
    plt.close(fig)


def model_html(res, path: Path) -> None:
    """Interactive 3D viewer (plotly, loads plotly.js from its CDN)."""
    import plotly.graph_objects as go

    v, f, c = res.vertices, res.triangles, res.colors
    v0 = v - v.min(0) * np.array([1, 1, 0])  # start at (0, 0) in plan for nicer axes
    mesh = go.Mesh3d(
        x=v0[:, 0],
        y=v0[:, 1],
        z=v0[:, 2],
        i=f[:, 0],
        j=f[:, 1],
        k=f[:, 2],
        facecolor=[f"rgb({r},{g},{b})" for r, g, b in c],
        flatshading=True,
        lighting=dict(ambient=0.5, diffuse=0.8, specular=0.1),
        lightposition=dict(x=-100, y=100, z=200),
        hoverinfo="skip",
    )
    fig = go.Figure(mesh)
    fig.update_layout(
        scene=dict(
            aspectmode="data",
            xaxis_title="east (m)",
            yaxis_title="north (m)",
            zaxis_title="z (m a.s.l.)",
        ),
        title="Reconstructed roof (planar faces, walls, obstructions)",
        margin=dict(l=0, r=0, t=40, b=0),
    )
    fig.write_html(path, include_plotlyjs="cdn")


def plot_panels(res, fr, om, t, out: Path) -> None:
    """Step-6 figures: layout on the RGB, suitability, usable area, strategy comparison, 3D."""
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Polygon as MplPolygon
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    pixel = om.hag.pixel_size
    ext = _extent(fr.labels.shape, pixel)
    zoom = _zoom_window(fr.candidate, pixel)
    labels = fr.labels
    s = res.stats

    def setzoom(a):
        a.set_xlim(zoom[0], zoom[1])
        a.set_ylim(zoom[2], zoom[3])

    def draw_panels(a, fc="#1b2a6b", ec="white"):
        for p in res.panels:
            xy = np.array(p["plan"].exterior.coords)
            # plan y is north (negative down the image); figures use south-positive axes
            a.add_patch(MplPolygon(np.c_[xy[:, 0], -xy[:, 1]], closed=True, fc=fc, ec=ec, lw=0.5))

    fig, ax = plt.subplots(2, 2, figsize=(18, 17))
    ax[0, 0].imshow(t.rgb, extent=ext)
    for k in fr.planes:
        _outline(ax[0, 0], labels == k, pixel, color="yellow", lw=0.5)
    draw_panels(ax[0, 0])
    _style(
        ax[0, 0],
        f"{s['panels']} panels ({s['kwp']} kWp, ~{s['annual_kwh_estimate']:,} kWh/yr) "
        f"on {sum(r['panels'] > 0 for r in res.faces)} faces",
    )

    rel = np.full(labels.shape, np.nan)
    for r in res.faces:
        rel[labels == r["id"]] = r["relative_yield"]
    im = ax[0, 1].imshow(rel, cmap="RdYlGn", vmin=0.5, vmax=1.0, extent=ext)
    fig.colorbar(im, ax=ax[0, 1], shrink=0.7, label="annual irradiation / site optimum")
    from scipy import ndimage as ndi

    for r in res.faces:
        m = labels == r["id"]
        if m.sum() < 300:
            continue
        rr, cc = ndi.center_of_mass(m)
        ax[0, 1].text(
            (cc + 0.5) * pixel[0],
            (rr + 0.5) * pixel[1],
            f"{r['relative_yield']:.2f}" + ("" if r["suitable"] else "\n✗"),
            ha="center",
            va="center",
            fontsize=8,
            fontweight="bold",
        )
    _style(ax[0, 1], "Suitability: annual irradiation relative to the site optimum (✗ = excluded)")

    img = np.zeros(labels.shape)
    img[fr.candidate] = 1
    img[labels > 0] = 1
    img[om.keepout | (om.classes > 0) & (labels > 0)] = 2
    img[om.usable] = 3
    ax[1, 0].imshow(
        np.where(img > 0, img, np.nan),
        cmap=ListedColormap(["lightgrey", "salmon", "#a8d5a2"]),
        vmin=0.5,
        vmax=3.5,
        extent=ext,
        interpolation="nearest",
    )
    draw_panels(ax[1, 0])
    for lab_, col in (
        ("usable", "#a8d5a2"),
        ("obstruction / keep-out", "salmon"),
        ("panel", "#1b2a6b"),
    ):
        ax[1, 0].plot([], [], "s", color=col, ms=10, label=lab_)
    ax[1, 0].legend(loc="lower left")
    c = s["checks"]
    _style(
        ax[1, 0],
        f"Panels vs usable area: overlaps {c['overlaps']}, min gap {c['min_gap_m']} m, "
        f"min edge dist {c['min_edge_distance_m']} m, raster violations "
        f"{c['raster_violations']}",
    )
    for a in (ax[0, 0], ax[0, 1], ax[1, 0]):
        setzoom(a)

    suit = [r for r in res.faces if r["suitable"]]
    keys = list(s["strategy_totals"])
    xs = np.arange(len(suit))
    bw = 0.8 / len(keys)
    for i, key in enumerate(keys):
        ax[1, 1].bar(
            xs + (i - (len(keys) - 1) / 2) * bw,
            [r.get("candidates", {}).get(key, 0) for r in suit],
            bw,
            label=f"{key} (total {s['strategy_totals'][key]})",
        )
    ax[1, 1].set_xticks(xs, [f"face {r['id']}\n{r['aspect_deg']:.0f}°" for r in suit], fontsize=8)
    ax[1, 1].set_ylabel("panels")
    ax[1, 1].legend()
    ax[1, 1].set_title("Packing strategy comparison per suitable face")
    fig.tight_layout()
    fig.savefig(out / "00_summary.png", dpi=100)
    plt.close(fig)

    # 01: 3D view of the roof with panels
    m = res.model
    v, f, col = m.vertices, m.triangles, m.colors
    tri = v[f]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
    shaded = _shade(col.astype(float), n)
    planes = m.obstructions.faces.planes
    ptris = [np.array(p["corners_local"]) + planes[p["face"]].normal * 0.06 for p in res.panels]
    # Painter's algorithm limits: panels always lie on top of their roof face, so draw the
    # roof first and the panels last (the HTML viewer uses a real depth buffer instead).
    fig = plt.figure(figsize=(16, 10))
    a = fig.add_subplot(projection="3d", computed_zorder=False)
    a.add_collection3d(
        Poly3DCollection(list(tri), facecolors=shaded, edgecolors=shaded, linewidths=0.3, zorder=1)
    )
    a.add_collection3d(
        Poly3DCollection(ptris, facecolors="#1b2a6b", edgecolors="white", linewidths=0.5, zorder=2)
    )
    lo, hi = v.min(0), v.max(0)
    span = max(hi[0] - lo[0], hi[1] - lo[1])
    a.set_xlim(lo[0], hi[0])
    a.set_ylim(lo[1], hi[1])
    a.set_zlim(lo[2], lo[2] + span * 0.35)
    a.set_box_aspect((hi[0] - lo[0], hi[1] - lo[1], span * 0.35), zoom=1.3)
    a.view_init(elev=40, azim=-120)
    a.set_axis_off()
    a.set_title(f"{s['panels']} panels on the reconstructed roof (view from the south-west)")
    fig.tight_layout()
    fig.savefig(out / "01_3d_panels.png", dpi=100)
    plt.close(fig)
