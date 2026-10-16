"""Step 3: roof face segmentation, plane fitting and planar roof reconstruction.

Pipeline (all on the AOI grid, coordinates in local metres: x east, y north):

1. ``candidate_mask``  raised (nDSM > wall-line height), non-vegetation, largest connected
   region, holes filled (keeps glass/lantern areas that the RGB index mislabels).
2. ``region_grow``     segments of coherent surface normals; a pixel joins a region when its
   normal is within ``angle`` of the region's *fitted plane* normal and it lies within
   ``dist`` of that plane. Comparing to the region (not the neighbour) stops the growth
   leaking across creases that the provider's smoothing turned into gradual bends.
3. ``fit_plane``       robust total-least-squares plane per segment.
4. ``merge_coplanar``  adjacent segments on the same plane are merged; slivers and tiny
   segments are dropped (their pixels are re-assigned in step 5).
5. ``assign_pixels``   every candidate pixel takes the nearby plane it is closest to
   (vertical distance), then planes are refitted (k-planes style iterations). Where two
   planes meet at a rounded crease, "closest plane" switches exactly where the planes are
   equal, i.e. on their intersection line, so sharp ridges, hips and valleys are rebuilt.
6. ``extend_eaves``    pixels of the smeared outer walls (data *below* the adjacent plane,
   near the footprint edge) are given to that plane, rebuilding the eave line.
7. ``face_adjacency``  each pair of touching faces is classified as a crease (planes meet:
   ridge / hip / valley) or a step (vertical offset), with geometric consistency checks.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage as ndi

# ------------------------------------------------------------------------------ planes


@dataclass
class Plane:
    """Plane n . p = d with unit normal n pointing up (n_z > 0), p = (x east, y north, z)."""

    normal: np.ndarray
    d: float
    rmse: float = 0.0
    n_points: int = 0

    def z_at(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        nx, ny, nz = self.normal
        return (self.d - nx * x - ny * y) / nz

    @property
    def slope_deg(self) -> float:
        return float(np.degrees(np.arccos(np.clip(self.normal[2], -1, 1))))

    def aspect_deg(self, rotation_deg: float = 0.0) -> float:
        """Downslope compass direction (deg clockwise from north), plus a grid rotation."""
        nx, ny, _ = self.normal
        return float((np.degrees(np.arctan2(nx, ny)) + rotation_deg) % 360)

    def angle_to(self, other: Plane) -> float:
        return float(np.degrees(np.arccos(np.clip(abs(self.normal @ other.normal), -1, 1))))


def fit_plane(
    points: np.ndarray, iterations: int = 3, k: float = 2.5, floor_m: float = 0.01
) -> tuple[Plane, np.ndarray]:
    """Robust total-least-squares plane through ``points`` (N, 3).

    TLS minimises perpendicular distances (smallest singular vector of the centred points),
    so it does not depend on how the plane is oriented, unlike z = ax + by + c regression,
    which minimises vertical distances and is biased on steep faces. Each iteration drops
    points further than k robust sigmas (1.4826 * MAD, floored at ``floor_m``) and refits.
    Returns (plane, inlier mask).
    """
    inl = np.ones(len(points), bool)
    plane = None
    for _ in range(iterations + 1):
        p = points[inl]
        c = p.mean(0)
        _, _, vt = np.linalg.svd(p - c, full_matrices=False)
        n = vt[2] if vt[2][2] >= 0 else -vt[2]
        r = (points - c) @ n
        med = np.median(r[inl])
        sigma = max(1.4826 * np.median(np.abs(r[inl] - med)), floor_m)
        plane = Plane(n, float(n @ c), float(np.sqrt(np.mean(r[inl] ** 2))), int(inl.sum()))
        # centre on the median residual: a biased first fit (pulled by outliers) has its
        # inliers offset from zero
        new = np.abs(r - med) <= k * sigma
        if new.sum() < 3 or np.array_equal(new, inl):
            break
        inl = new
    return plane, inl


# ------------------------------------------------------------------------------ helpers


def pixel_coords(shape: tuple[int, int], pixel: tuple[float, float]) -> tuple[np.ndarray, ...]:
    """Local metric coordinates of pixel centres: x east, y north (row 0 at the top)."""
    rows, cols = shape
    x = (np.arange(cols) + 0.5) * pixel[0]
    y = -(np.arange(rows) + 0.5) * pixel[1]
    return np.meshgrid(x, y)


def candidate_mask(
    ndsm: np.ndarray, vegetation: np.ndarray, aoi: np.ndarray, min_height_m: float
) -> np.ndarray:
    """Largest connected raised, non-vegetation region, with interior holes filled."""
    m = aoi & (np.nan_to_num(ndsm) > min_height_m) & ~vegetation
    lab, n = ndi.label(m)
    if n == 0:
        return m
    biggest = np.argmax(ndi.sum(m, lab, np.arange(1, n + 1))) + 1
    return ndi.binary_fill_holes(lab == biggest) & aoi


def local_planarity(normals: np.ndarray, size: int = 5) -> np.ndarray:
    """Angle (deg) between each normal and the mean normal of its size x size window."""
    n = np.nan_to_num(normals)
    mean = np.stack([ndi.uniform_filter(n[i], size) for i in range(3)])
    mean /= np.linalg.norm(mean, axis=0) + 1e-12
    return np.degrees(np.arccos(np.clip((n * mean).sum(0), -1, 1)))


# ------------------------------------------------------------------------------ growing

_NB8 = ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1))


def region_grow(
    z: np.ndarray,
    normals: np.ndarray,
    grow: np.ndarray,
    planarity: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    angle_deg: float,
    dist_m: float,
    seed_max_deg: float,
    min_px: int,
) -> np.ndarray:
    """Plane-aware region growing. Returns int labels (0 = unassigned).

    Seeds are taken from the most planar pixels first (planarity < ``seed_max_deg``). The
    region plane is refitted (TLS) whenever the region has doubled in size, so the test
    uses an increasingly accurate plane. Regions smaller than ``min_px`` are discarded and
    their pixels are not used as seeds again.
    """
    rows, cols = z.shape
    labels = np.zeros(z.shape, np.int32)
    tried = np.zeros(z.shape, bool)
    cos_t = np.cos(np.radians(angle_deg))
    order = np.argsort(np.where(grow, planarity, np.inf), axis=None)
    k = 0
    for s in order:
        r, c = divmod(int(s), cols)
        if not grow[r, c] or planarity[r, c] > seed_max_deg:
            break  # remaining pixels are less planar (sorted)
        if labels[r, c] or tried[r, c]:
            continue
        k += 1
        labels[r, c] = k
        region = [(r, c)]
        nrm = normals[:, r, c].copy()
        cen = np.array([x[r, c], y[r, c], z[r, c]])
        refit_at = 20
        q = deque([(r, c)])
        while q:
            rr, cc = q.popleft()
            for dr, dc in _NB8:
                a, b = rr + dr, cc + dc
                if a < 0 or b < 0 or a >= rows or b >= cols or labels[a, b] or not grow[a, b]:
                    continue
                if normals[:, a, b] @ nrm < cos_t:
                    continue
                if abs((np.array([x[a, b], y[a, b], z[a, b]]) - cen) @ nrm) > dist_m:
                    continue
                labels[a, b] = k
                region.append((a, b))
                q.append((a, b))
                if len(region) >= refit_at:
                    rr_, cc_ = np.array(region).T
                    pl, _ = fit_plane(np.c_[x[rr_, cc_], y[rr_, cc_], z[rr_, cc_]], iterations=0)
                    nrm = pl.normal
                    cen = np.array([x[rr_, cc_].mean(), y[rr_, cc_].mean(), z[rr_, cc_].mean()])
                    refit_at = 2 * len(region)
        if len(region) < min_px:
            rr_, cc_ = np.array(region).T
            labels[rr_, cc_] = 0
            tried[rr_, cc_] = True
            k -= 1
    return labels


# ------------------------------------------------------------------------------ merge / filter


def _relabel(labels: np.ndarray) -> np.ndarray:
    ids = np.unique(labels[labels > 0])
    lut = np.zeros(labels.max() + 1, np.int32)
    lut[ids] = np.arange(1, len(ids) + 1)
    return lut[labels]


def fit_all(labels: np.ndarray, z: np.ndarray, x: np.ndarray, y: np.ndarray) -> dict[int, Plane]:
    planes = {}
    for k in np.unique(labels[labels > 0]):
        m = labels == k
        planes[int(k)], _ = fit_plane(np.c_[x[m], y[m], z[m]])
    return planes


def neighbours(labels: np.ndarray, gap_px: int) -> set[tuple[int, int]]:
    """Pairs of labels within ``gap_px`` pixels of each other (a gap may be a sliver/hole)."""
    pairs = set()
    for k in np.unique(labels[labels > 0]):
        near = ndi.binary_dilation(labels == k, iterations=gap_px)
        for j in np.unique(labels[near]):
            if j > k:
                pairs.add((int(k), int(j)))
    return pairs


def merge_coplanar(
    labels: np.ndarray,
    z: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    angle_deg: float,
    dist_m: float,
    gap_px: int,
    max_ratio: float = 1.3,
    floor_m: float = 0.005,
) -> np.ndarray:
    """Agglomerative merging of neighbouring segments that lie on one plane.

    A pair is mergeable when (1) the normals differ by < ``angle_deg``, (2) the plane
    fitted to the union has RMSE < ``dist_m`` and (3) that RMSE is at most ``max_ratio``
    times the pooled RMSE of the two separate planes: one plane explains both nearly as
    well as two. The single best pair (lowest union RMSE) is merged, its plane refitted on
    the actual merged pixels, and the search repeats. Merging one pair at a time avoids
    chains (A~B and B~C merging A with C although A and C are not coplanar), which
    union-find would allow.
    """
    labels = labels.copy()
    pts = {
        int(k): np.c_[x[labels == k], y[labels == k], z[labels == k]]
        for k in np.unique(labels[labels > 0])
    }
    planes = {k: fit_plane(p)[0] for k, p in pts.items()}
    adj: dict[int, set[int]] = {k: set() for k in pts}
    for a, b in neighbours(labels, gap_px):
        adj[a].add(b)
        adj[b].add(a)
    cache: dict[tuple[int, int], float | None] = {}

    def union_rmse(a: int, b: int) -> float | None:
        key = (min(a, b), max(a, b))
        if key not in cache:
            pa, pb = planes[a], planes[b]
            val = None
            if pa.angle_to(pb) < angle_deg:
                u, _ = fit_plane(np.vstack([pts[a], pts[b]]))
                pooled = np.sqrt(
                    (pa.n_points * pa.rmse**2 + pb.n_points * pb.rmse**2)
                    / max(pa.n_points + pb.n_points, 1)
                )
                if u.rmse < dist_m and u.rmse <= max_ratio * max(pooled, floor_m):
                    val = u.rmse
            cache[key] = val
        return cache[key]

    while True:
        best = None
        for a in adj:
            for b in adj[a]:
                if a < b:
                    r = union_rmse(a, b)
                    if r is not None and (best is None or r < best[0]):
                        best = (r, a, b)
        if best is None:
            return _relabel(labels)
        _, a, b = best
        labels[labels == b] = a
        pts[a] = np.vstack([pts[a], pts.pop(b)])
        planes[a] = fit_plane(pts[a])[0]
        planes.pop(b)
        adj[a] = (adj[a] | adj.pop(b)) - {a, b}
        for k in adj:
            if b in adj[k]:
                adj[k].discard(b)
                if k != a:
                    adj[k].add(a)
        cache = {key: v for key, v in cache.items() if a not in key and b not in key}


def drop_small(
    labels: np.ndarray, min_px: int, core_erosion_px: int, min_core_frac: float
) -> np.ndarray:
    """Remove segments that are too small or too thin (slivers along blurred creases).

    Thinness test: fraction of the segment surviving an erosion of ``core_erosion_px``.
    """
    out = labels.copy()
    for k in np.unique(labels[labels > 0]):
        m = labels == k
        core = ndi.binary_erosion(m, iterations=core_erosion_px).sum()
        if m.sum() < min_px or core / m.sum() < min_core_frac:
            out[m] = 0
    return _relabel(out)


def drop_poor_fit(
    labels: np.ndarray, z: np.ndarray, x: np.ndarray, y: np.ndarray, max_rmse_m: float
) -> np.ndarray:
    """Remove segments whose robust plane RMSE exceeds ``max_rmse_m``.

    Real faces on this DSM fit their plane to about 1-4 cm (the measured waviness is about
    2.5 cm). A worse fit means the segment mixes a face with something non-planar, such as
    roof windows, a dormer or canopy, so it is not accepted as a face.
    """
    out = labels.copy()
    for k, pl in fit_all(labels, z, x, y).items():
        if pl.rmse > max_rmse_m:
            out[labels == k] = 0
    return _relabel(out)


# ------------------------------------------------------------------------------ assignment


def assign_pixels(
    labels: np.ndarray,
    z: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    cand: np.ndarray,
    radius_px: int,
    tol_m: float,
    iterations: int,
) -> tuple[np.ndarray, dict[int, Plane]]:
    """k-planes refinement restricted to nearby faces.

    Each candidate pixel takes the face, among faces within ``radius_px`` of it, whose plane
    is vertically closest to the data, if that distance is <= ``tol_m``; otherwise it stays
    unassigned (an object on the roof, a wall, vegetation). Planes are then refitted from
    their new pixels, and this is repeated.
    """
    planes = fit_all(labels, z, x, y)
    for _ in range(iterations):
        best = np.full(z.shape, np.inf)
        new = np.zeros_like(labels)
        for k, pl in planes.items():
            near = ndi.distance_transform_edt(labels != k) <= radius_px
            near &= cand
            res = np.abs(z - pl.z_at(x, y))
            take = near & (res < best) & (res <= tol_m)
            best[take] = res[take]
            new[take] = k
        # keep only the part of each face connected to its previous core
        for k in planes:
            m = new == k
            lab, n = ndi.label(m)
            if n > 1:
                keep = np.unique(lab[m & (labels == k)])
                new[m & ~np.isin(lab, keep[keep > 0])] = 0
        labels = new
        planes = fit_all(labels, z, x, y)
    return labels, planes


def extend_eaves(
    labels: np.ndarray,
    z: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    cand: np.ndarray,
    planes: dict[int, Plane],
    band_px: int,
) -> np.ndarray:
    """Give smeared outer-wall pixels to the adjacent face (rebuilds the eave line).

    Only unassigned candidate pixels within ``band_px`` of the footprint edge, whose
    nearest face is also within ``band_px`` and whose data lies *below* that face's plane,
    are taken: below = the edge was smeared
    downwards by the provider's smoothing. Data *above* the plane is something on the roof
    and is left for obstruction detection.
    """
    edge_dist = ndi.distance_transform_edt(cand)
    band = cand & (labels == 0) & (edge_dist <= band_px)
    if not band.any() or not planes:
        return labels
    dist, idx = ndi.distance_transform_edt(labels == 0, return_indices=True)
    nearest = labels[tuple(idx)]
    out = labels.copy()
    for k, pl in planes.items():
        m = band & (nearest == k) & (dist <= band_px)  # only from an adjacent face
        below = z[m] <= pl.z_at(x[m], y[m]) + 0.02
        rr, cc = np.nonzero(m)
        out[rr[below], cc[below]] = k
    return out


# ------------------------------------------------------------------------------ adjacency


@dataclass
class Edge:
    a: int
    b: int
    kind: str  # ridge | hip | valley | step
    length_m: float
    gap_m: float  # median |plane_a - plane_b| on the shared boundary
    line_angle_deg: float | None = None  # boundary direction vs plane intersection (plan)
    line_offset_m: float | None = None  # mean plan distance of boundary to intersection
    line_sigma_m: float | None = None  # expected uncertainty of the intersection position
    consistent: bool | None = None
    extra: dict = field(default_factory=dict)


def face_adjacency(
    labels: np.ndarray,
    planes: dict[int, Plane],
    x: np.ndarray,
    y: np.ndarray,
    pixel: float,
    crease_gap_m: float,
    max_angle_deg: float = 5.0,
    min_offset_m: float = 0.15,
) -> list[Edge]:
    """Classify each touching pair of faces and check geometric consistency.

    * gap: on the shared boundary the two planes should give the same height if the faces
      meet at a crease. Median gap < ``crease_gap_m``: crease, otherwise step.
    * crease type: the planes intersect in a 3D line with direction n_a x n_b. Convex
      (each plane, extended over the other face, lies above it) means ridge (horizontal
      line) or hip (sloping); concave means valley.
    * consistency: the pixel boundary should run along the intersection line: angle
      between them (PCA direction of boundary points vs line direction) and mean offset.
      Consistent when angle <= ``max_angle_deg`` and offset <= max(``min_offset_m``,
      3 * expected line uncertainty), so nearly parallel planes are judged fairly.
    """
    edges = []
    centroid = {k: (x[labels == k].mean(), y[labels == k].mean()) for k in planes}
    for a, b in sorted(neighbours(labels, 1)):
        ma, mb = labels == a, labels == b
        bnd = (ma & ndi.binary_dilation(mb)) | (mb & ndi.binary_dilation(ma))
        if bnd.sum() < 3:
            continue
        bx, by = x[bnd], y[bnd]
        pa, pb = planes[a], planes[b]
        gap = float(np.median(np.abs(pa.z_at(bx, by) - pb.z_at(bx, by))))
        length = float(bnd.sum() * pixel / 2)  # boundary is two pixels thick
        if gap >= crease_gap_m:
            edges.append(Edge(a, b, "step", length, gap))
            continue
        # convexity: plane b evaluated over face a's centroid (and vice versa)
        ca, cb = centroid[a], centroid[b]
        above = (pb.z_at(*ca) - pa.z_at(*ca)) + (pa.z_at(*cb) - pb.z_at(*cb))
        direction = np.cross(pa.normal, pb.normal)
        direction /= np.linalg.norm(direction)
        horizontal = abs(direction[2]) < np.sin(np.radians(5))
        kind = ("ridge" if horizontal else "hip") if above > 0 else "valley"
        # intersection line in plan: points where z_a = z_b; take one point on it
        n1, n2 = pa.normal, pb.normal
        A = np.array([[n1[0] / n1[2] - n2[0] / n2[2], n1[1] / n1[2] - n2[1] / n2[2]]])
        rhs = np.array([pa.d / n1[2] - pb.d / n2[2]])
        p0 = np.linalg.lstsq(A, rhs, rcond=None)[0]  # minimum-norm point on the plan line
        dplan = direction[:2] / (np.linalg.norm(direction[:2]) + 1e-12)
        pts = np.c_[bx, by]
        c = pts.mean(0)
        _, _, vt = np.linalg.svd(pts - c, full_matrices=False)
        ang = float(np.degrees(np.arccos(np.clip(abs(vt[0] @ dplan), -1, 1))))
        normal2d = np.array([-dplan[1], dplan[0]])
        offset = float(np.mean(np.abs((pts - p0) @ normal2d)))
        # Where z_a = z_b moves by sigma_z / |grad z_a - grad z_b| for a height error
        # sigma_z, so nearly parallel planes have a poorly defined intersection.
        ga = -pa.normal[:2] / pa.normal[2]
        gb = -pb.normal[:2] / pb.normal[2]
        sigma_z = np.sqrt((pa.rmse**2 + pb.rmse**2) / 2)
        sigma_line = float(sigma_z / max(np.linalg.norm(ga - gb), 1e-6))
        ok = ang <= max_angle_deg and offset <= max(min_offset_m, 3 * sigma_line)
        edges.append(Edge(a, b, kind, length, gap, ang, offset, sigma_line, bool(ok)))
    return edges


# ------------------------------------------------------------------------------ driver


@dataclass
class SegmentParams:
    max_grow_slope_deg: float = 70.0
    seed_max_planarity_deg: float = 3.0
    grow_angle_deg: float = 10.0
    grow_dist_m: float = 0.10
    min_face_m2: float = 1.0
    core_erosion_m: float = 0.3
    min_core_frac: float = 0.15
    merge_angle_deg: float = 5.0
    merge_rmse_m: float = 0.04
    merge_max_ratio: float = 1.3
    merge_gap_m: float = 0.5
    max_face_rmse_m: float = 0.05
    assign_radius_m: float = 1.0
    assign_tol_m: float = 0.25
    assign_iterations: int = 3
    eave_band_m: float = 1.0


def segment_roof(
    z: np.ndarray,
    normals: np.ndarray,
    slope: np.ndarray,
    cand: np.ndarray,
    pixel: tuple[float, float],
    p: SegmentParams,
    trace: dict | None = None,
) -> tuple[np.ndarray, dict[int, Plane]]:
    """Run stages 2-6 on a cleaned DSM. Returns (face labels, planes by label).

    If ``trace`` is a dict, the intermediate label maps are stored in it (for figures).
    """
    x, y = pixel_coords(z.shape, pixel)
    px = pixel[0]
    grow = cand & (np.nan_to_num(slope, nan=90) < p.max_grow_slope_deg)
    planarity = local_planarity(normals)
    min_px = int(round(p.min_face_m2 / (pixel[0] * pixel[1])))
    labels = region_grow(
        np.nan_to_num(z),
        np.nan_to_num(normals),
        grow,
        planarity,
        x,
        y,
        p.grow_angle_deg,
        p.grow_dist_m,
        p.seed_max_planarity_deg,
        min_px,
    )
    if trace is not None:
        trace["grown"] = labels.copy()
    core_px = int(round(p.core_erosion_m / px))
    radius_px = int(round(p.assign_radius_m / px))

    def tidy(lab: np.ndarray, gap_px: int) -> np.ndarray:
        lab = merge_coplanar(
            lab, z, x, y, p.merge_angle_deg, p.merge_rmse_m, gap_px, p.merge_max_ratio
        )
        lab = drop_small(lab, min_px, core_px, p.min_core_frac)
        return drop_poor_fit(lab, z, x, y, p.max_face_rmse_m)

    labels = tidy(labels, int(round(p.merge_gap_m / px)))
    if trace is not None:
        trace["merged"] = labels.copy()
    labels, _ = assign_pixels(labels, z, x, y, cand, radius_px, p.assign_tol_m, p.assign_iterations)
    # Assignment can make two faces meet on one plane, leave crumbs or create a poor fit;
    # tidy again, then let the surviving faces take back the pixels that fit them.
    labels = tidy(labels, 1)
    labels, _ = assign_pixels(labels, z, x, y, cand, radius_px, p.assign_tol_m, 1)
    labels = drop_small(labels, min_px, core_px, p.min_core_frac)
    planes = fit_all(labels, z, x, y)
    if trace is not None:
        trace["assigned"] = labels.copy()
    labels = extend_eaves(labels, z, x, y, cand, planes, int(round(p.eave_band_m / px)))
    return labels, planes


def planar_surface(
    labels: np.ndarray, planes: dict[int, Plane], pixel: tuple[float, float]
) -> np.ndarray:
    """Reconstructed roof heights: each labelled pixel takes its face plane (NaN elsewhere)."""
    x, y = pixel_coords(labels.shape, pixel)
    out = np.full(labels.shape, np.nan)
    for k, pl in planes.items():
        m = labels == k
        out[m] = pl.z_at(x[m], y[m])
    return out
