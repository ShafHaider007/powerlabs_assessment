"""Step 4: rooftop obstruction detection and usable-area mask.

Two physically different kinds of obstruction need two kinds of evidence:

* **Raised objects** (dormers, chimneys, flues, vents, overhanging vegetation) stand out of
  the roof plane: height above the fitted face plane from the *raw* DSM (the cleaned DSM
  softens small objects, so it is not used here; see step 2).
* **Flush objects** (roof windows, skylights, small fittings) lie in the roof plane and are
  invisible in height: colour anomaly in the RGB, relative to the tiles of the *same face*
  (roofs can have more than one tile colour), with a robust colour covariance.
* **Glass faces** (the lantern pyramid) are planar, so step 3 made them faces, but they
  cannot carry panels: face-level test on glazing contrast and specular brightness.

Usable area = roof faces minus every obstruction grown by a keep-out buffer.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
from scipy import ndimage as ndi
from skimage.color import rgb2lab
from skimage.measure import label, regionprops
from skimage.morphology import disk
from sklearn.covariance import MinCovDet

from roofsolar.faces import Plane, pixel_coords

# class codes of the obstruction raster
CLASSES = {
    1: "dormer / structure",
    2: "chimney / small structure",
    3: "vent / small raised object",
    4: "vegetation",
    5: "glass roof (lantern)",
    6: "roof window / skylight / fitting",
    7: "dark anomaly (shadow or dark fitting)",
    8: "depression",
}


def height_above_planes(
    raw: np.ndarray,
    labels: np.ndarray,
    planes: dict[int, Plane],
    building: np.ndarray,
    pixel: tuple[float, float],
) -> np.ndarray:
    """Raw DSM minus the roof plane, for every building pixel.

    Face pixels use their own plane; other building pixels (dormers, chimneys, ...) use the
    plane of the nearest face, i.e. the roof surface the object stands on. NaN off-building.
    """
    x, y = pixel_coords(labels.shape, pixel)
    idx = ndi.distance_transform_edt(labels == 0, return_distances=False, return_indices=True)
    nearest = labels[tuple(idx)]
    out = np.full(labels.shape, np.nan)
    for k, pl in planes.items():
        m = building & (nearest == k)
        out[m] = raw[m] - pl.z_at(x[m], y[m])
    return out


def local_contrast(lightness: np.ndarray, size: int = 5) -> np.ndarray:
    """Standard deviation of lightness in a size x size window."""
    m = ndi.uniform_filter(lightness, size)
    return np.sqrt(np.maximum(ndi.uniform_filter(lightness**2, size) - m**2, 0))


def glass_faces(
    labels: np.ndarray,
    lab: np.ndarray,
    min_contrast: float,
    bright_l: float,
    min_bright_frac: float,
) -> np.ndarray:
    """Faces made of glass: glazing frames give high local contrast AND glass reflects
    specularly (many very bright pixels). Both cues are required, so a shaded face (high
    contrast, dark) or a brightly lit tile face (bright, smooth) does not qualify."""
    contrast = local_contrast(lab[..., 0])
    out = np.zeros(labels.shape, bool)
    for k in np.unique(labels[labels > 0]):
        m = labels == k
        if (
            np.median(contrast[m]) > min_contrast
            and np.mean(lab[..., 0][m] > bright_l) > min_bright_frac
        ):
            out |= m
    return out


def colour_distance(
    lab: np.ndarray,
    labels: np.ndarray,
    exclude: np.ndarray,
    sample: int = 8000,
    seed: int = 0,
    floor_std: float = 1.0,
) -> np.ndarray:
    """Mahalanobis colour distance of each face pixel from its own face's median colour.

    Location: per face (handles roofs with several tile colours). Shape: one covariance for
    all tiles, estimated with the Minimum Covariance Determinant, which ignores the windows,
    shadows and capping that contaminate the sample. Eigenvalues are floored at
    ``floor_std``^2 (1 CIELAB unit, below the just-noticeable difference dE ~ 2.3) so very
    uniform roofs stay invertible; a well-conditioned covariance is left unchanged. If MCD
    fails on degenerate colours, the plain covariance is used.
    """
    faces = (labels > 0) & ~exclude
    x = lab[faces]
    rng = np.random.default_rng(seed)
    pick = x[rng.choice(len(x), min(sample, len(x)), replace=False)]
    try:
        with warnings.catch_warnings():  # rank deficiency is handled by the floor below
            warnings.simplefilter("ignore", UserWarning)
            cov = MinCovDet(random_state=seed).fit(pick).covariance_
    except ValueError:  # degenerate (near-constant) colours
        cov = np.cov(pick, rowvar=False)
    # floor the eigenvalues (not add to them): a well-conditioned covariance is unchanged
    w, v = np.linalg.eigh(cov)
    prec = (v / np.maximum(w, floor_std**2)) @ v.T
    d = np.zeros(labels.shape)
    for k in np.unique(labels[faces]):
        m = (labels == k) & ~exclude
        dv = lab[m] - np.median(lab[m], axis=0)
        d[m] = np.sqrt(np.einsum("ij,jk,ik->i", dv, prec, dv))
    return d


def compact_blobs(
    mask: np.ndarray, min_px: int, max_px: int, max_elong: float, min_minor_px: float
) -> np.ndarray:
    """Keep compact blobs: removes line-like responses (ridge/hip capping, valley strips)."""
    lb = label(mask)
    out = np.zeros_like(mask)
    for r in regionprops(lb):
        elong = r.axis_major_length / max(r.axis_minor_length, 1e-6)
        if (
            min_px <= r.area <= max_px
            and elong <= max_elong
            and r.axis_minor_length >= min_minor_px
        ):
            out[lb == r.label] = True
    return out


@dataclass
class Detection:
    raised: np.ndarray
    sunken: np.ndarray
    colour: np.ndarray
    glass: np.ndarray
    hag: np.ndarray
    colour_dist: np.ndarray
    lab: np.ndarray


def detect(
    raw: np.ndarray,
    rgb: np.ndarray,
    labels: np.ndarray,
    planes: dict[int, Plane],
    building: np.ndarray,
    pixel: tuple[float, float],
    p: dict,
) -> Detection:
    """Run the three detectors (height, colour, glass). ``p`` is the config section."""
    px = pixel[0]
    cell = pixel[0] * pixel[1]
    faces = labels > 0
    hag = height_above_planes(raw, labels, planes, building, pixel)
    lab = rgb2lab(rgb)
    glass = glass_faces(
        labels, lab, p["glass_min_contrast"], p["glass_bright_l"], p["glass_min_bright_frac"]
    )

    open_r = int(round(p["open_radius_m"] / px))
    h = np.nan_to_num(hag)
    raised = ndi.binary_opening(building & (h > p["min_height_m"]), structure=disk(open_r))
    edge = ndi.distance_transform_edt(building) * px
    sunken = faces & (h < -p["min_height_m"]) & (edge > p["eave_band_m"])
    sunken = ndi.binary_opening(sunken, structure=disk(open_r))
    sunken = compact_blobs(sunken, int(p["min_depression_m2"] / cell), 10**9, np.inf, 0)

    cdist = colour_distance(lab, labels, glass)
    colour = faces & ~glass & (cdist > p["colour_threshold"])
    colour = ndi.binary_opening(colour, structure=disk(open_r))
    colour = compact_blobs(
        colour,
        max(1, int(round(p["colour_min_m2"] / cell))),
        int(round(p["colour_max_m2"] / cell)),
        p["colour_max_elongation"],
        p["colour_min_width_m"] / px,
    )
    return Detection(raised, sunken, colour, glass, hag, cdist, lab)


def classify(
    det: Detection, labels: np.ndarray, vegetation: np.ndarray, pixel: tuple[float, float], p: dict
) -> tuple[np.ndarray, list[dict]]:
    """Turn detections into an obstruction class raster and an object table."""
    cell = pixel[0] * pixel[1]
    cls = np.zeros(labels.shape, np.uint8)
    objects: list[dict] = []
    x, y = pixel_coords(labels.shape, pixel)

    def add(mask, code, evidence, confidence, extra=None):
        cls[mask & (cls == 0)] = code
        objects.append(
            {
                "class": CLASSES[code],
                "code": code,
                "evidence": evidence,
                "confidence": confidence,
                "area_m2": round(float(mask.sum() * cell), 2),
                "centroid_local_m": [
                    round(float(x[mask].mean()), 2),
                    round(float(y[mask].mean()), 2),
                ],
                **(extra or {}),
            }
        )

    lab_g = label(det.glass)
    for i in range(1, lab_g.max() + 1):
        add(lab_g == i, 5, "rgb (glazing contrast + specular)", "high")

    lab_r = label(det.raised, connectivity=2)
    for r in regionprops(lab_r):
        m = lab_r == r.label
        hmax = float(np.nanmax(det.hag[m]))
        area = r.area * cell
        with_rgb = bool((det.colour & m).any())
        elong = r.axis_major_length / max(r.axis_minor_length, 1e-6)
        ring = ndi.binary_dilation(m, iterations=3) & ~m
        if det.glass[ring].mean() > 0.5:
            code = 5  # raised part enclosed by glass faces (lantern apex / finial)
        elif vegetation[m].mean() > p["vegetation_frac"] or hmax > p["vegetation_min_height_m"]:
            code = 4
        elif area >= p["structure_min_m2"] or elong > p["chimney_max_elongation"]:
            code = 1  # large, or a long strip (dormer cheek / wall smear)
        elif hmax >= p["chimney_min_height_m"]:
            code = 2  # small, compact and tall
        else:
            code = 3
        add(
            m,
            code,
            "dsm+rgb" if with_rgb else "dsm",
            "high",
            {"max_height_m": round(hmax, 2), "elongation": round(float(elong), 1)},
        )

    lab_c = label(det.colour & ~det.raised)
    for r in regionprops(lab_c):
        m = lab_c == r.label
        k = np.bincount(labels[m]).argmax()
        ref = (labels == k) & ~det.colour
        d_l = float(np.median(det.lab[..., 0][m]) - np.median(det.lab[..., 0][ref]))
        code = 7 if d_l < p["dark_delta_l"] else 6
        add(m, code, "rgb", "medium" if code == 6 else "low", {"delta_lightness": round(d_l, 1)})

    lab_s = label(det.sunken & (cls == 0))
    for i in range(1, lab_s.max() + 1):
        add(lab_s == i, 8, "dsm", "medium")
    return cls, objects


def usable_mask(
    labels: np.ndarray, obstruction: np.ndarray, pixel: tuple[float, float], buffer_m: float
) -> tuple[np.ndarray, np.ndarray]:
    """Keep-out = obstructions grown by ``buffer_m``; usable = faces minus keep-out."""
    r = int(round(buffer_m / pixel[0]))
    keepout = ndi.binary_dilation(obstruction, structure=disk(r)) if r > 0 else obstruction.copy()
    keepout &= labels > 0
    return keepout, (labels > 0) & ~keepout & ~obstruction
