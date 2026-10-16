# Progress Log

One entry per step. Each entry covers what was done, outputs, decisions and why, and assumptions. This log feeds the technical write-up.

| Step | Status |
|---|---|
| 1. Load data, then elevation, slope, aspect | ✅ Done |
| 2. Clean noise, fill missing data | ✅ Done (awaiting approval) |
| 3. Roof face segmentation and plane fitting | ⏳ Next |
| 4. Obstruction detection | ⬜ |
| 5. 3D roof model | ⬜ |
| 6. Solar panel placement | ⬜ |
| 7. Validation, README, write-up, video | ⬜ |

## Assumptions (running list)
- Standard panel size is about 1.7 m × 1.0 m (to be confirmed).
- `dsm_clipped.tif` is the area of interest (the property); everything outside it is ignored.
- Heights are metres above sea level in EPSG:32630; the vertical and horizontal units are both metres.
- Roofs are piecewise planar (the TGV² prior in step 2 and plane fitting in step 3 rely on this).
- Noise model for tuning: measured from this DSM (step 2 §2.1); parameters may need retuning for other data.

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

---

## Step 2: Noise regularisation and missing data

**Run:** `python scripts/run_pipeline.py --steps 1 2` (about 40 s; step 2 needs step 1 and re-runs it if not given)
**Evidence:** `python scripts/benchmark_cleaning.py [--noise simple|realistic] [--part A B]`
**Code:** `src/roofsolar/cleaning.py` (TGV solver, fill, outliers, ExG), `src/roofsolar/synthetic.py`, `pipeline.py::run_cleaning`
**Outputs:** `outputs/step02_cleaning/`: `dsm_clean.tif`, `slope.tif`, `aspect.tif`, `normals.tif`, `removed.tif`, `outliers.tif`, `filled.tif`, `vegetation.tif`, `stats.json`, figures `00`–`03`, `benchmark/`

### 2.1 What is actually wrong with this DSM (research before choosing a method)
| Check | Finding | Consequence |
|---|---|---|
| NaN holes inside the AOI | **none** (step 1) | "missing data" has to be found elsewhere |
| Spikes (|z − median₅| map) | large deviations only at **real structure**: walls, hedges, the tree, ridge and hip lines. No isolated spikes on roof faces | a median filter would shave ridges (they show up in its residual) |
| Roof cross-sections (3 profiles) | walls drop over **0.5–0.8 m** (about 75°, not vertical); ridges and valleys rounded over 0.5–1 m; faces wavy | the provider's smoothing blurred edges; **no local filter can undo that**, only plane intersection (step 3) |
| Roof structure | a **ring hip roof**: outer faces slope outward, inner faces slope inward to valleys around a central glass lantern; dormers on the NE side | relevant for step 3 segmentation |
| Waviness on roof faces | plane residual about 2.2 cm in 2.1 m planar tiles; autocorrelation 0.94 / 0.78 / 0.56 / 0.08 at lags 1 / 2 / 3 / 5 px = Gaussian-correlated noise with **σ_c ≈ 2.5 px (0.25 m)** | "noise" is metre-scale waviness, not pixel noise (pixel-level noise in building regions is only about 0.14 cm) |
| Tree over the roof? | ExG vegetation mask: the tree touches but **does not cover** the roof | no canopy gap to fill on this roof |
| Glass lantern | scores as vegetation in ExG (green reflections) | step 4 must not trust RGB vegetation on glass |

### 2.2 Method
1. **Fill missing pixels** (NaN) by **biharmonic** (thin-plate) inpainting.
2. **Regularise with TGV²** (Bredies, Kunisch & Pock 2010) on the full grid:
   `min ½Σλ(u−f)² + α₁‖∇u − w‖₁ + α₀‖ε(w)‖₁`
   - `w` is a slope field; ε(w) is its symmetrised gradient (change of slope).
   - The L1 norm on the change of slope makes the solution **piecewise planar with sharp creases**, which is exactly the model of a roof.
   - Solved with the **Chambolle–Pock primal-dual algorithm** (convex, so the result does not depend on the starting point).
   - λ = 0 marks untrusted pixels, which are then filled from the surrounding planes.
3. **Outliers:** local Hampel test on r = z − u, plus an amplitude-relative isolation test. Outlier pixels are treated as missing, re-filled, and step 2 runs again.
4. **Recompute slope, aspect and normals** with the step-1 code on the cleaned surface.
5. **Vegetation layer:** ExG = 2g − r − b on chromatic coordinates, with an Otsu threshold. Canopy heights are not roof heights.

**Design rule:** the cleaned DSM is for *geometry* (normals, segmentation, planes). Obstruction detection (step 4) must use the **raw** DSM against the fitted planes, so small real objects (vents) that regularisation softens are never lost. The outlier mask is exported for step 4.

### 2.3 Why these choices (evidence)

**A. Smoothing benchmark** on a synthetic rotated hip roof (with walls and a chimney) where the truth is known. Errors are measured per region.
- First noise model (1 cm, 1 px correlation, quantised): **TGV face-normal error 0.50° vs 2.54° raw**. The best other method is median (1.92°), but it rounds the walls (12.9 cm).
- On the real data that model proved wrong: the effect was much smaller than predicted. So the noise model was **rebuilt from measurements** (blur σ 2.5 px, 2.5 cm waviness with 2.5 px correlation, quantised ground) and the parameters re-tuned.

| realistic model | face normal error | crease RMSE | wall RMSE |
|---|---|---|---|
| raw | 3.80° | 10.99 cm | 164.4 cm |
| Gaussian (best) | 3.70° | 11.40 | 165.5 |
| median (best) | 3.65° | 11.22 | 164.4 |
| bilateral (best) | 4.16° | 11.03 | 164.4 |
| TV (best) | 3.76° | 11.08 | 164.4 |
| **TGV α₁ 0.05, α₀ 0.2** | **1.17°** | 11.32 | 164.4 |

- Selection rule: lowest (face-normal error + crease RMSE) **with walls no worse than the input**.
- Crease and wall errors are about the same for every method: the blur cannot be undone locally, which motivates plane intersection in step 3.
- TV flattens slopes (noise-free 35° roof: p5 slope 27–31°) and smooths faces much less than TGV.

**Parameters confirmed on the real roof** (40 planar 1.5 m tiles; open ground):

| setting | roof normal scatter | ground slope p50 / p95 | ridge lowered | time |
|---|---|---|---|---|
| raw | 3.07° | 3.44° / 9.88° | n/a | n/a |
| α₁ 0.02 / α₀ 0.04 | 2.17° | 2.56° / 7.75° | 0.6–0.8 cm | 12 s |
| **α₁ 0.05 / α₀ 0.2** | **1.21°** | **1.73° / 6.51°** | 1.0–1.7 cm | 18 s |
| α₁ 0.1 / α₀ 0.4 | 0.73° | 1.58° / 5.84° | 1.7–2.8 cm | 31 s |

α₁ 0.05 / α₀ 0.2 halves the noise for about 1.5 cm of ridge loss. Going further doubles the ridge loss for little extra gain.

**B. Fill benchmark (hold-out on real roof pixels).** 25 holes per radius were cut into the real roof, filled, and compared with the original heights. Median RMSE in cm [p90]:

| method | r = 0.3 m | r = 0.75 m | r = 1.5 m |
|---|---|---|---|
| nearest | 7.74 [9.45] | 15.92 [19.49] | 31.00 [39.91] |
| IDW (GDAL fillnodata) | 3.55 [5.30] | 9.65 [12.47] | 22.14 [29.97] |
| linear (Delaunay) | 0.51 [0.99] | 4.58 [13.12] | 12.82 [30.20] |
| **biharmonic** | **0.21 [0.68]** | **1.96 [3.89]** | **7.68 [12.68]** |
| TGV (α₁ 0.05, α₀ 0.2) | 1.47 [2.33] | 3.15 [8.32] | 8.01 [15.17] |

- This **contradicted my expectation** that TGV would win.
- The provider-smoothed roof is gently curved, not perfectly planar, and biharmonic interpolation matches that best.
- So the pipeline fills with biharmonic and regularises with TGV.
- Large holes that cross creases (obstruction footprints) are better handled by plane reconstruction (steps 3–4).

**C. Outlier detector, validated by planting 150 spikes/pits** (1–4 px, ±0.15–2 m) in the real DSM:

| version | recall (all) | recall (roof) | false flags on clean data |
|---|---|---|---|
| Hampel on r only | 96 % | n/a | 425 px |
| + fixed isolation threshold | 66 % | 46 % | 12 px |
| **+ isolation relative to spike amplitude** | **95 %** | **100 % (37/37)** | 189 px (27 roof) |

- **Why an isolation test:** TGV rounds the *corners* of real boxes, so corners stand out in r (unit test). Requiring the "same object" region to be small separates a spike (alone) from a chimney corner (part of the whole outline).
- **Why relative to amplitude:** the roof's 2.5 cm waviness connected spikes to large regions under a fixed threshold.
- The 27 roof flags include a **row of 20–30 cm pits along the SW eave** (the dotted line visible in the raw hillshade), which is a real artifact, now removed.

### 2.4 Edge cases handled
| Edge case | Handling |
|---|---|
| NaN / missing pixels | biharmonic fill, then regularisation (TGV also supports λ = 0) |
| AOI boundary | regularised on the full grid, then cropped |
| Image border bending planes in TGV | dual variables of non-existent differences fixed at 0, so a plane is exactly zero-cost (error went from 4 cm to 0.04 mm; test) |
| Operator correctness | adjoint tests ⟨∇u, p⟩ = −⟨u, div p⟩ and ⟨εw, q⟩ = −⟨w, div₂ q⟩ (caught a sign bug that made the solver diverge) |
| Float precision | TGV runs on heights minus their mean |
| Convergence | stops at RMS change < 1e-6 m/iteration; logs a warning if `tgv_max_iter` is reached (real data: 1,325 iterations, converged) |
| Real edges and vegetation vs spikes | local robust σ (Hampel) |
| Near-perfect areas (MAD ≈ 0) | global σ floor |
| Corners of real objects | isolation test on raw heights |
| Spikes on wavy roofs | amplitude-relative isolation |
| Large real objects | blobs > 0.25 m² are never outliers |
| 1 cm quantisation on ground | smoothed out (spiky slope histogram becomes smooth) |
| Small real roof objects | cleaned DSM is not used for obstruction detection (rule above) |

### 2.5 Results (`stats.json`)
| | before | after |
|---|---|---|
| Roof-face normal scatter (40 tiles) | 3.04° | **1.25°** |
| Open-ground slope p50 / p95 | 3.44° / 9.88° | **1.73° / 6.51°** |
| Flat threshold (auto) | 9.88° | 6.51° |
| Isolated outliers replaced | | 189 px (27 roof, 108 vegetation, 54 other); max 0.41 m |
| Removed by regularisation | | robust σ 1.31 cm; roof p99 \|Δz\| 5.1 cm |
| Missing pixels in AOI | 0 | 0 |
| Vegetation | | 54 % of AOI (lawn, hedges, tree); 368 m² tall vegetation |
| TGV | | 2 passes × 1,325 iterations, 36 s |

### 2.6 Limitations
- Blurred walls and ridges are not restored here; step 3 rebuilds them from plane intersections.
- TGV parameters are tuned to this DSM's measured noise; another sensor or provider would need the benchmark re-run (it is scripted).
- ExG mislabels the glass lantern as vegetation.
- ExG is RGB-only; there is no NIR band for NDVI.
- 36 s runtime (pure NumPy). Acceptable here; a GPU or numba port would be the next step for large areas.
