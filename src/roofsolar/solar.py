"""First-order annual irradiation on tilted planes, to rank roof faces for PV.

Purpose: *relative* comparison of faces (how much less a NE face receives than a SW face
of the same pitch), not an energy yield certificate (that would use PVGIS / measured data).

Model, hourly over one year:
* sun position: NOAA approximation (declination, equation of time, hour angle);
* horizontal irradiance: a two-state sky. A fraction of hours is clear (clearness index
  Kt_clear), the rest overcast (Kt_overcast); G = Kt * G0 * cos(zenith) and the diffuse
  fraction comes from the Erbs et al. (1982) correlation for each state. A single constant
  Kt would make every hour equally hazy (diffuse fraction ~76 %), which flatters north
  faces; the two states reproduce a realistic UK diffuse fraction (~50 %). Irradiance on
  a plane is linear in its components, so the year is the weighted sum of both states;
* plane of array: beam * cos(incidence) + isotropic sky diffuse (Liu-Jordan) + ground
  reflection (albedo). No horizon or near shading.
"""

from __future__ import annotations

import numpy as np

SOLAR_CONSTANT = 1361.0  # W/m2


def sun_position(
    lat_deg: float, lon_deg: float, day_of_year: np.ndarray, utc_hour: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Solar zenith and azimuth (deg, azimuth clockwise from north) - NOAA approximation."""
    g = 2 * np.pi / 365 * (day_of_year - 1 + (utc_hour - 12) / 24)
    eot = 229.18 * (
        0.000075
        + 0.001868 * np.cos(g)
        - 0.032077 * np.sin(g)
        - 0.014615 * np.cos(2 * g)
        - 0.040849 * np.sin(2 * g)
    )  # minutes
    decl = (
        0.006918
        - 0.399912 * np.cos(g)
        + 0.070257 * np.sin(g)
        - 0.006758 * np.cos(2 * g)
        + 0.000907 * np.sin(2 * g)
        - 0.002697 * np.cos(3 * g)
        + 0.00148 * np.sin(3 * g)
    )
    solar_time = utc_hour * 60 + eot + 4 * lon_deg  # minutes
    ha = np.radians(solar_time / 4 - 180)
    lat = np.radians(lat_deg)
    cos_z = np.sin(lat) * np.sin(decl) + np.cos(lat) * np.cos(decl) * np.cos(ha)
    zen = np.arccos(np.clip(cos_z, -1, 1))
    az = (
        np.degrees(np.arctan2(np.sin(ha), np.cos(ha) * np.sin(lat) - np.tan(decl) * np.cos(lat)))
        + 180
    )
    return np.degrees(zen), az % 360


def erbs_diffuse_fraction(kt: np.ndarray) -> np.ndarray:
    """Erbs, Klein & Duffie (1982): diffuse fraction of global horizontal irradiance."""
    kt = np.asarray(kt, float)
    return np.where(
        kt <= 0.22,
        1 - 0.09 * kt,
        np.where(
            kt <= 0.8, 0.9511 - 0.1604 * kt + 4.388 * kt**2 - 16.638 * kt**3 + 12.336 * kt**4, 0.165
        ),
    )


class AnnualIrradiation:
    """Precomputed hourly sun/sky for one year; evaluates any plane quickly."""

    def __init__(
        self,
        lat_deg: float,
        lon_deg: float,
        clear_kt: float = 0.70,
        overcast_kt: float = 0.25,
        clear_fraction: float = 0.42,
        albedo: float = 0.2,
        step_h: float = 1.0,
    ):
        hours = np.arange(0, 365 * 24, step_h) + step_h / 2
        day, hour = np.floor(hours / 24) + 1, hours % 24
        zen, az = sun_position(lat_deg, lon_deg, day, hour)
        up = zen < 89.5
        self.zen, self.az = np.radians(zen[up]), np.radians(az[up])
        g0h = SOLAR_CONSTANT * (1 + 0.033 * np.cos(2 * np.pi * day[up] / 365)) * np.cos(self.zen)
        self.ghi = np.zeros_like(g0h)
        self.dhi = np.zeros_like(g0h)
        for kt, w in ((clear_kt, clear_fraction), (overcast_kt, 1 - clear_fraction)):
            g = kt * g0h
            self.ghi += w * g
            self.dhi += w * g * erbs_diffuse_fraction(np.array(kt))
        self.dni = (self.ghi - self.dhi) / np.maximum(np.cos(self.zen), 0.05)
        self.albedo = albedo
        self.step_h = step_h

    @property
    def diffuse_fraction(self) -> float:
        return float(self.dhi.sum() / self.ghi.sum())

    def plane(self, tilt_deg: float, azimuth_deg: float) -> float:
        """Annual plane-of-array irradiation (kWh/m2) for a plane facing ``azimuth_deg``."""
        b, g = np.radians(tilt_deg), np.radians(azimuth_deg)
        cos_i = np.cos(self.zen) * np.cos(b) + np.sin(self.zen) * np.sin(b) * np.cos(self.az - g)
        poa = (
            self.dni * np.clip(cos_i, 0, None)
            + self.dhi * (1 + np.cos(b)) / 2
            + self.ghi * self.albedo * (1 - np.cos(b)) / 2
        )
        return float(poa.sum() * self.step_h / 1000)

    def optimum(self) -> tuple[float, float, float]:
        """(tilt, azimuth, kWh/m2) of the best fixed plane (grid search)."""
        best = (0.0, 180.0, self.plane(0, 180))
        for tilt in range(0, 71, 2):
            for az in range(150, 211, 5):
                v = self.plane(tilt, az)
                if v > best[2]:
                    best = (float(tilt), float(az), v)
        return best
