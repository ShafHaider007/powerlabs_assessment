"""Tests for the TGV regulariser: operator maths, plane preservation, crease sharpness, filling."""

import numpy as np
import pytest

from roofsolar.cleaning import (
    _div,
    _div2,
    _grad,
    _sym_grad,
    detect_outliers,
    excess_green,
    fill_biharmonic,
    tgv_denoise,
)


@pytest.fixture
def rng():
    return np.random.default_rng(0)


def test_div_is_negative_adjoint_of_grad(rng):
    u, p = rng.normal(size=(17, 23)), rng.normal(size=(2, 17, 23))
    assert np.sum(_grad(u) * p) == pytest.approx(-np.sum(u * _div(p)))


def test_div2_is_negative_adjoint_of_sym_grad(rng):
    w, q = rng.normal(size=(2, 17, 23)), rng.normal(size=(3, 17, 23))
    eq = _sym_grad(w)
    inner = np.sum(eq[0] * q[0]) + np.sum(eq[1] * q[1]) + 2 * np.sum(eq[2] * q[2])
    assert inner == pytest.approx(-np.sum(w * _div2(q)))


def hip(n=80, pitch=0.07):
    """Ridge roof: two planes meeting at a sharp ridge in the middle column."""
    x = np.arange(n)
    row = 100 + pitch * (n / 2 - np.abs(x - n / 2))
    return np.tile(row, (n, 1))


def test_plane_is_unchanged():
    yy, xx = np.mgrid[:60, :60]
    z = 100 + 0.03 * xx - 0.05 * yy
    u, _ = tgv_denoise(z, alpha1=0.05, alpha0=0.1, n_iter=2000, tol=1e-8)
    np.testing.assert_allclose(u, z, atol=1e-4)


def test_noise_reduced_and_ridge_kept_sharp(rng):
    truth = hip()
    noisy = truth + rng.normal(0, 0.01, truth.shape)
    u, _ = tgv_denoise(noisy, alpha1=0.02, alpha0=0.04, n_iter=2000)
    inner = (slice(5, -5), slice(5, -5))
    assert np.sqrt(np.mean((u - truth)[inner] ** 2)) < 0.4 * 0.01
    ridge = truth.shape[1] // 2
    # Error at the crease stays at noise level (Gaussian smoothing would cut it by
    # pitch * sigma ~ several cm).
    assert np.abs(u[inner[0], ridge] - truth[inner[0], ridge]).mean() < 0.01


def test_missing_block_filled_by_extending_planes():
    truth = hip()
    f = truth.copy()
    f[30:50, 30:50] = np.nan  # hole across the ridge
    u, _ = tgv_denoise(f, alpha1=0.02, alpha0=0.04, n_iter=4000, tol=1e-7)
    assert np.abs(u - truth)[30:50, 30:50].max() < 0.05


def test_all_nan_raises():
    with pytest.raises(ValueError):
        tgv_denoise(np.full((5, 5), np.nan), 0.1, 0.2)


def test_biharmonic_fill_reproduces_plane_and_keeps_data():
    yy, xx = np.mgrid[:40, :40]
    z = 10 + 0.05 * xx + 0.02 * yy
    missing = np.zeros(z.shape, bool)
    missing[10:20, 15:25] = True
    filled = fill_biharmonic(z, missing)
    np.testing.assert_allclose(filled, z, atol=1e-6)
    assert fill_biharmonic(z, np.zeros_like(missing)) is not z


def test_outliers_found_but_edges_and_large_objects_kept(rng):
    z = hip() + rng.normal(0, 0.005, (80, 80))
    z[:, :5] += 3.0  # a tall wall/step along one side (real edge)
    z[60:70, 60:70] += 1.0  # 1 m^2 box (real object, 100 px)
    z[20, 20] += 0.5  # spike
    z[40, 55] -= 0.4  # pit
    u, _ = tgv_denoise(z, 0.05, 0.2, n_iter=3000)
    mask, sigma = detect_outliers(z, u, np.ones(z.shape, bool), k=6, max_blob_px=25)
    assert mask[20, 20] and mask[40, 55]
    assert not mask[58:72, 58:72].any()  # corners of the real box are not outliers
    assert not mask[:, :8].any()  # nor the wall
    assert mask.sum() <= 4
    assert 0 < sigma < 0.02


def test_excess_green_separates_grass_from_roof():
    grass = np.array([60, 140, 50])
    roof = np.array([90, 95, 110])
    img = np.stack([np.full((4, 4), v) for v in grass] + [])
    assert excess_green(img)[0, 0] > 0.1
    roof_img = np.stack([np.full((4, 4), v) for v in roof])
    assert excess_green(roof_img)[0, 0] < 0.02
    # channel-last input gives the same answer
    assert excess_green(np.moveaxis(img, 0, -1))[0, 0] == excess_green(img)[0, 0]
