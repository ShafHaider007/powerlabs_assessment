# Progress Log

One entry per step. Each entry covers what was done, outputs, decisions and why, and assumptions. This log feeds the technical write-up.

| Step | Status |
|---|---|
| 1. Load data, then elevation, slope, aspect | ✅ Done (awaiting approval) |
| 2. Clean noise, fill missing data | ⏳ Next |
| 3. Roof face segmentation and plane fitting | ⬜ |
| 4. Obstruction detection | ⬜ |
| 5. 3D roof model | ⬜ |
| 6. Solar panel placement | ⬜ |
| 7. Validation, README, write-up, video | ⬜ |

## Assumptions (running list)
- Standard panel size is about 1.7 m × 1.0 m (to be confirmed).
- `dsm_clipped.tif` is the area of interest (the property); everything outside it is ignored.
- Heights are metres above sea level in EPSG:32630; the vertical and horizontal units are both metres.

---

## Step 1: Data loading, elevation, slope, aspect

**Run:** `python scripts/run_pipeline.py --steps 1` (under 1 s)
**Validate:** `python scripts/validate_terrain_gdal.py`
**Code:** `src/roofsolar/raster_io.py`, `src/roofsolar/terrain.py`, `src/roofsolar/pipeline.py::run_terrain`
**Outputs:** `outputs/step01_terrain/`: `elevation.tif`, `slope.tif`, `aspect.tif`, `normals.tif` (3 bands), `dtm.tif`, `ndsm.tif`, `stats.json`, figures `00_summary.png` … `06_data_provenance.png`

### 1.1 What the data really is (findings from exploration)
| Finding | Evidence | Consequence |
|---|---|---|
| All three rasters share one grid: 0.1 m, EPSG:32630, north-up | gdalinfo; checked in code (`check_grid`, `window_of`) | No resampling is needed (resampling would change the data) |
| `dsm_clipped` = `dsm_full` cropped to a polygon, at offset (40, 36) px | max abs diff on valid pixels = **0.0 m** | Derivatives are computed on the **full** DSM and then cropped, so AOI-boundary pixels have real neighbours |
| The clip polygon is the **property parcel** (house, lawn, part of a tree, hedges), not the roof | 54 % NoData, in 2 regions both touching the border, **0 interior holes** | NoData means "outside the AOI", not missing measurements; roof selection happens in step 3 |
| The RGB and DSM are co-registered | DSM contours drawn over the RGB follow the roof edges | RGB can be used directly for obstruction detection (step 4) |
| **The DSM is a composite.** Raw areas store heights in exact 1 cm steps; areas around buildings were re-processed by the provider | 0.0 % off-grid heights outside these regions vs 90 % inside; inside is about 3× smoother at pixel scale (median Laplacian 0.007 vs 0.020) | Roofs are already smoothed: pixel noise is low, but they are wavy at the metre scale (plane residual 1.85 cm vs 1.03 cm) and ridges are likely rounded. This supports reconstructing faces as planes and sharp ridges as **plane intersections** (step 3). Diagnostic only; not used to find buildings |
| A tree about 24 m tall overhangs the roof's west corner | nDSM max 24.1 m | Occlusion edge case for steps 3, 4 and 6 |
| Main roof: hip roof, four faces 90° apart, pitch about 35° | aspect peaks 44°, 134°, 224°, 316° (true N); median slope 35.3° | Sanity check that the derivatives are right |

### 1.2 Method and why

**Slope and aspect: Horn (1981) 3×3 operator**
```
a b c      dz/dx = ((c+2f+i) − (a+2d+g)) / (8·dx)
d e f      dz/dy = ((a+2b+c) − (g+2h+i)) / (8·dy)      (north up)
g h i      slope = atan(√(gx² + gy²))      aspect = atan2(−gx, −gy) mod 360
```
- **Why Horn:**
  - It is a Sobel operator: a central difference smoothed across its direction with (1, 2, 1) weights. For independent noise σ, its gradient error is σ·√12/(8d) ≈ **0.43σ/d**, compared with 0.71σ/d for a 2-point central difference (`np.gradient`). That is about 40 % less noise. Proved in `test_horn_is_less_noisy_than_central_difference`.
  - It is the industry standard (GDAL, QGIS, ArcGIS), so the output can be checked against a reference.
- **Rejected alternatives:**
  - Zevenbergen–Thorne (4-neighbour, 2nd order): more noise-sensitive.
  - `np.gradient`: noisier.
  - Fitting a local plane per pixel over a larger window: smooths more but blurs ridges. Noise is handled deliberately in step 2 instead.
- **Aspect is the downslope direction**, clockwise from north (0 = N, 90 = E, 180 = S). That is the direction a solar panel on the face would point.
- **Note:** Horn never uses the centre pixel, so a single NaN blanks its 8 neighbours but not itself (tested).

**Edge cases handled**
| Edge case | Handling | Why |
|---|---|---|
| NoData (-9999) | converted to NaN on read; any NaN in the 3×3 window gives NaN output | never invent gradients next to missing data |
| AOI boundary | derivatives computed on the full DSM, then cropped | boundary pixels get real neighbours instead of a fake cliff |
| Outer frame of the raster (the clip touches the full raster's east edge) | **linear extrapolation** padding (2z₀ − z₁) | a unit test caught that repeating edge pixels halves the gradient on the outer ring; extrapolation is exact for planes |
| Flat surfaces (aspect undefined) | aspect = NaN where slope < threshold | on flat ground the "direction" is pure noise |
| Grid north ≠ true north | aspect rotated by the grid convergence **+1.626°** | UTM grid north is true north only on the central meridian (3°W); the site is at 0.94°W, 52.28°N. Computed numerically (step 1 km up the grid, then take the geodesic bearing); matches (λ−λ₀)·sin φ = 1.630° |
| Geographic CRS, rotated or south-up grids | rejected with a clear error (`check_grid`) | the maths assumes metres and north-up |
| Clipped DSM not a subset of the full DSM | rejected (max diff > 0.1 mm raises) | the full-grid derivative trick would be invalid |
| Misaligned grids (sub-pixel offsets) | rejected (`window_of`) | would need resampling |

**Flat threshold (data-driven): 9.88°**
- Measured slope on open flat ground: nDSM < 0.3 m, at least 1 m from any raised object (that buffer removes wall bases and hedge edges, which pushed p99 to 61°).
- The threshold is the p95 of that distribution: the slope that noise alone produces 95 % of the time. The value is stable from a 1 m to a 2 m buffer (9.88° → 9.47°). A clean lawn patch gives p95 = 7.9°.
- **Why measure it:** DSM noise is spatially correlated (lag-1 autocorrelation 0.77), so the formula for independent noise overestimates slope noise (11° vs 3.2° measured median).
- **Why so high:** at 10 cm pixels, 1 cm of noise is a 1:10 height-to-distance ratio, which is about 6°. This is a pixel-scale effect that step 2 reduces.
- It is far below the roof pitch (35°), so no roof face is lost.

**Height above ground (nDSM): morphological ground filter**
1. Down-sample to 1 m using the block minimum.
2. Grey opening with a 31 m square (wider than the house, about 26 m), which removes buildings and trees.
3. Gaussian σ = 2 m, then bilinear interpolation back to 10 cm. nDSM = DSM − DTM.
- **Why:** it separates roof from ground and trees by height, which steps 3 and 4 need. It is a simplified version of the progressive morphological filter (Zhang et al., 2003).
- **Limitation:** within half a window (about 15 m) of the raster edge, sloping ground is biased low by up to (w/2)·tan(slope). Here the ground slope is about 0.2°, so the bias is a few cm.

**Noise estimate:** least-squares plane fitted to each 1.5 m tile; noise = p10 of the residual std (the flattest tiles are planar, so their residual is noise) → **1.1 cm** overall.

### 1.3 Validation
- **gdaldem (GDAL reference):** 636,000 pixels compared.
  - Slope |diff| max **3.8×10⁻⁶°**; aspect |diff| max **2.3×10⁻⁵°** → PASS.
  - The first run showed differences up to 0.26° in aspect on near-flat pixels. Traced to gdaldem doing its arithmetic in **float32** on absolute heights (about 110 m, which rounds at about 3×10⁻⁵ m). Proved by offsetting the DSM by −100 m: the differences dropped about 10,000×. Our float64 result is the more precise one.
- **58 unit tests** with known answers:
  - planes at 4 slopes × 10 aspects (including the 0/360 wrap) are exact to 1e-9
  - non-square pixels
  - NaN propagation
  - edge modes
  - normals
  - noise ratio of Horn vs central difference
  - grid convergence (0 on the central meridian, 1.63° here)
  - ground filter removes a building
  - noise estimate recovers a known σ
  - raster I/O: CRS and orientation rejection, window/crop round trip, half-pixel rejection, NaN round trip
  - the real data is aligned
- **Physical sanity:** four aspect peaks exactly 90° apart (a rectangular hip roof), pitch 35°, ground at about 106.8 m, tree 24 m.

### 1.4 Key numbers (`stats.json`)
| | |
|---|---|
| AOI | 215,590 px = 2,155.9 m² |
| Elevation (AOI) | 106.47 to 131.16 m (median 107.40) |
| Ground (DTM) | 106.39 to 107.32 m |
| Height above ground | up to 24.1 m (tree); roof about 3 to 13 m |
| Grid convergence | +1.626° |
| DSM noise | 1.1 cm overall; 1.0 cm raw areas, 1.85 cm re-processed areas |
| Flat threshold | 9.88° (auto, p95 of open-ground slope) |
| Roof candidates (preview) | 592 m² planimetric; slope median 35.3° |
| Roof aspect peaks | 224.0° (SW, 26 %), 134.1° (SE, 19 %), 43.9° (NE, 19 %), 315.8° (NW, 18 %) |

### 1.5 Limitations and open points for later steps
- Slope at 10 cm is noisy on flat ground (median 3.4°); step 2 smooths it.
- The roof-candidate mask (nDSM ≥ 2.5 m and slope 15–60°) is only a preview: it includes hedges and the tree. Step 3 does the real segmentation.
- Re-processed roof regions are probably smoothed, with rounded ridges; step 3 should rebuild sharp edges from plane intersections.
- The tree overhanging the west corner hides part of the roof.
