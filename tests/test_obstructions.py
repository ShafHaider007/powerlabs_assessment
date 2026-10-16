"""Tests for obstruction detection on a synthetic roof with known objects."""

import numpy as np
import pytest
import yaml
from scipy import ndimage as ndi

from roofsolar.config import PROJECT_ROOT
from roofsolar.faces import fit_all, pixel_coords
from roofsolar.obstructions import (
    CLASSES,
    classify,
    compact_blobs,
    detect,
    height_above_planes,
    usable_mask,
)
from roofsolar.synthetic import hip_roof_scene

PX = (0.1, 0.1)
P = yaml.safe_load((PROJECT_ROOT / "configs" / "default.yaml").read_text())["obstructions"]


@pytest.fixture(scope="module")
def scene():
    sc = hip_roof_scene()
    rng = np.random.default_rng(0)
    raw = sc.z + rng.normal(0, 0.01, sc.z.shape)
    labels = sc.faces.copy()
    x, y = pixel_coords(labels.shape, PX)
    planes = fit_all(labels, sc.z, x, y)
    # RGB: grey tiles with mild texture, green ground
    rgb = np.empty((*sc.z.shape, 3))
    rgb[:] = (95, 95, 110)
    rgb += rng.normal(0, 3, rgb.shape)
    rgb[~sc.footprint] = (70, 140, 60)
    # a flush roof window (bright, bluish) on face 3, away from the chimney and creases
    rr, cc = np.nonzero(ndi.binary_erosion(labels == 3, iterations=25))
    r0, c0 = rr[len(rr) // 3], cc[len(cc) // 3]
    window = np.zeros(labels.shape, bool)
    window[r0 : r0 + 8, c0 : c0 + 12] = True
    rgb[window] = (170, 190, 220)
    # a thin bright line (ridge capping) on face 4: must not be detected
    rr, cc = np.nonzero(ndi.binary_erosion(labels == 4, iterations=25))
    line = np.zeros(labels.shape, bool)
    line[rr[len(rr) // 2], cc[len(cc) // 2] - 40 : cc[len(cc) // 2] + 40] = True
    rgb[line] = (150, 150, 160)
    rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    building = sc.footprint
    return sc, raw, rgb, labels, planes, building, window, line


def test_height_above_planes_zero_on_faces_and_tall_on_chimney(scene):
    sc, raw, _, labels, planes, building, *_ = scene
    hag = height_above_planes(raw, labels, planes, building, PX)
    assert abs(np.nanmedian(hag[labels > 0])) < 0.005
    assert np.nanmax(hag[sc.chimney]) > 1.0
    assert np.isnan(hag[~building]).all()


def test_detect_and_classify(scene):
    sc, raw, rgb, labels, planes, building, window, line = scene
    det = detect(raw, rgb, labels, planes, building, PX, P)
    cls, objects = classify(det, labels, np.zeros(labels.shape, bool), PX, P)
    # chimney found as a raised, compact, tall object
    assert (cls[sc.chimney] == 2).mean() > 0.8
    # roof window found from colour; the thin capping line is not
    assert (cls[window] == 6).mean() > 0.6
    assert not (cls[line] > 0).any()
    assert not det.glass.any()
    names = {o["class"] for o in objects}
    assert CLASSES[2] in names and CLASSES[6] in names


def test_glass_face_detected(scene):
    sc, raw, rgb, labels, planes, building, *_ = scene
    rgb2 = rgb.copy()
    rng = np.random.default_rng(1)
    glass = labels == 1
    # glazing: bright panes with dark frames every 0.5 m
    pane = np.where(
        (np.indices(labels.shape)[0] % 5 == 0) | (np.indices(labels.shape)[1] % 5 == 0), 40, 235
    )
    for ch in range(3):
        rgb2[..., ch] = np.where(
            glass, np.clip(pane + rng.normal(0, 5, labels.shape), 0, 255), rgb2[..., ch]
        )
    det = detect(raw, rgb2, labels, planes, building, PX, P)
    assert det.glass[glass].all() and not det.glass[labels == 3].any()


def test_compact_blobs_rejects_lines():
    m = np.zeros((50, 50), bool)
    m[10:16, 10:16] = True  # compact
    m[30, 5:45] = True  # line
    m[40:42, 5:30] = True  # strip, elongation > 4
    out = compact_blobs(m, 4, 1000, 4.0, 2)
    assert out[10:16, 10:16].all() and not out[30].any() and not out[40:42].any()


def test_usable_mask_buffer():
    labels = np.ones((50, 50), np.int32)
    obs = np.zeros((50, 50), bool)
    obs[24:26, 24:26] = True
    keepout, usable = usable_mask(labels, obs, PX, 0.3)
    assert not usable[obs].any()
    assert not usable[24, 22] and usable[24, 20]  # 0.2 m away blocked, 0.4 m away free
    assert keepout[24, 22]
