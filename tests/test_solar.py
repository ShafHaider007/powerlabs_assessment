"""Tests for the solar geometry and irradiation model."""

import numpy as np
import pytest

from roofsolar.solar import AnnualIrradiation, erbs_diffuse_fraction, sun_position

LAT, LON = 52.285, -0.94


def _noon_zenith(day):
    # solar noon at this longitude is about 12:04 UTC (+ equation of time); search it
    hours = np.linspace(11, 13.5, 2001)
    zen, _ = sun_position(LAT, LON, np.full_like(hours, day), hours)
    i = np.argmin(zen)
    return zen[i], hours[i]


def test_noon_zenith_equinox_and_solstices():
    z, _ = _noon_zenith(80)  # ~ 21 March
    assert z == pytest.approx(LAT, abs=0.7)
    z, _ = _noon_zenith(172)  # ~ 21 June
    assert z == pytest.approx(LAT - 23.44, abs=0.3)
    z, _ = _noon_zenith(355)  # ~ 21 December
    assert z == pytest.approx(LAT + 23.44, abs=0.3)


def test_sun_due_south_at_solar_noon():
    _, h = _noon_zenith(172)
    _, az = sun_position(LAT, LON, np.array([172.0]), np.array([h]))
    assert az[0] == pytest.approx(180, abs=1)


def test_erbs_limits():
    assert erbs_diffuse_fraction(0.1) == pytest.approx(0.991)
    assert erbs_diffuse_fraction(0.9) == pytest.approx(0.165)


def test_irradiation_ordering_and_magnitude():
    m = AnnualIrradiation(LAT, LON)
    ghi = m.plane(0, 180)
    assert 850 < ghi < 1250  # UK horizontal irradiation is ~950-1100 kWh/m2/yr
    s, e, w, n = (m.plane(35, a) for a in (180, 90, 270, 0))
    assert s > e > n and abs(e - w) / e < 0.02  # east ~ west by symmetry
    tilt, az, best = m.optimum()
    assert 30 <= tilt <= 42 and 170 <= az <= 190  # UK optimum tilt is ~35-38 deg
    assert 0.4 < m.diffuse_fraction < 0.6
