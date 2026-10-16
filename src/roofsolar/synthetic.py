"""Synthetic roof scenes with known ground truth, for benchmarks and tests.

A hip roof over a rectangle is exactly ``z = eave + tan(pitch) * d``, where ``d`` is the
distance to the nearest footprint edge: every face rises at the same pitch from its eave,
and ridges and hips form where two distances are equal.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage as ndi


@dataclass
class Scene:
    z: np.ndarray  # true heights (m)
    footprint: np.ndarray  # building mask
    crease: np.ndarray  # pixels on ridges / hips (where the face changes)
    wall: np.ndarray  # pixels next to the footprint boundary (eave + wall)
    chimney: np.ndarray  # chimney mask
    pixel: float


def hip_roof_scene(
    size_m: float = 30.0,
    pixel: float = 0.1,
    length_m: float = 18.0,
    width_m: float = 12.0,
    rotation_deg: float = 44.0,
    eave_m: float = 5.0,
    pitch_deg: float = 35.0,
    chimney_m: tuple[float, float, float] = (0.8, 0.8, 1.2),
    ground_m: float = 0.0,
) -> Scene:
    """Rotated hip roof with vertical walls and one chimney on flat ground."""
    n = int(round(size_m / pixel))
    c = (np.arange(n) + 0.5) * pixel - size_m / 2
    x, y = np.meshgrid(c, -c)  # x east, y north (row 0 = north)
    a = np.radians(rotation_deg)
    # local building axes: u along the length, v across
    u = x * np.cos(a) - y * np.sin(a)
    v = x * np.sin(a) + y * np.cos(a)
    du = length_m / 2 - np.abs(u)
    dv = width_m / 2 - np.abs(v)
    d = np.minimum(du, dv)
    footprint = d > 0
    z = np.where(footprint, eave_m + np.tan(np.radians(pitch_deg)) * d, ground_m)

    # Creases: where the two nearest-edge distances (or the two halves) are about equal.
    tol = 1.5 * pixel
    crease = footprint & ((np.abs(du - dv) < tol) | ((np.abs(v) < tol) & (du > dv)))
    edge = footprint ^ ndi.binary_erosion(footprint)
    wall = ndi.binary_dilation(edge, iterations=int(round(0.3 / pixel)))

    # Chimney on one face, halfway between ridge and eave.
    cw, cl, ch = chimney_m
    cu, cv = length_m / 4, width_m / 4
    chimney = (np.abs(u - cu) < cl / 2) & (np.abs(v - cv) < cw / 2)
    z = np.where(chimney, z[chimney].max() + ch, z)
    return Scene(z, footprint, crease, wall, chimney, pixel)


def correlated_noise(
    shape: tuple[int, int], sigma_m: float, corr_px: float, rng: np.random.Generator
) -> np.ndarray:
    """Gaussian noise with std ``sigma_m`` and a Gaussian correlation length of ``corr_px``.

    Photogrammetric DSM noise is spatially correlated (lag-1 autocorrelation of 0.77 was
    measured in step 1); corr_px = 1 gives about that value.
    """
    white = rng.normal(size=shape)
    if corr_px > 0:
        white = ndi.gaussian_filter(white, corr_px)
    return white / white.std() * sigma_m


def quantize(z: np.ndarray, step_m: float) -> np.ndarray:
    return np.round(z / step_m) * step_m


def realistic_dsm(scene: Scene, rng: np.random.Generator) -> np.ndarray:
    """Degrade a sharp scene the way the provided DSM is degraded (measured in step 2).

    * Building regions were re-processed by the provider: creases and walls are blurred
      (walls drop over 0.5-0.8 m, i.e. a Gaussian of ~2.5 px) and faces are wavy
      (plane residual ~2.2 cm with autocorrelation matching ~2.5 px correlated noise).
    * Elsewhere: 1 cm noise with ~1 px correlation, stored in 1 cm steps.
    """
    building = ndi.binary_dilation(scene.footprint, iterations=int(round(2.0 / scene.pixel)))
    blurred = ndi.gaussian_filter(scene.z, 2.5)
    roof_noise = correlated_noise(scene.z.shape, 0.025, 2.5, rng)
    ground_noise = correlated_noise(scene.z.shape, 0.01, 1.0, rng)
    return np.where(building, blurred + roof_noise, quantize(scene.z + ground_noise, 0.01))
