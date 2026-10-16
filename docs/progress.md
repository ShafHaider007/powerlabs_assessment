# Progress Log

One entry per step. Each entry covers what was done, outputs, decisions and why, and assumptions. This log feeds the technical write-up.

| Step | Status |
|---|---|
| 1. Load data, then elevation, slope, aspect | ✅ Done |
| 2. Clean noise, fill missing data | ✅ Done |
| 3. Roof face segmentation and plane fitting | ✅ Done |
| 4. Obstruction detection | ✅ Done |
| 5. 3D roof model | ✅ Done (awaiting approval) |
| 6. Solar panel placement | ⏳ Next |
| 7. Validation, README, write-up, video | ⬜ |

## Assumptions (running list)
- Standard panel size is about 1.7 m × 1.0 m (to be confirmed).
- `dsm_clipped.tif` is the area of interest (the property); everything outside it is ignored.
- Heights are metres above sea level in EPSG:32630; the vertical and horizontal units are both metres.
- Roofs are piecewise planar (the TGV² prior in step 2 and plane fitting in step 3 rely on this).
- Noise model for tuning: measured from this DSM (step 2 §2.1); parameters may need retuning for other data.
- Obstruction keep-out buffer is 0.3 m (measured DSM smear width; also a typical installation clearance).
- Missing a skylight or flue under a panel costs more than losing some panel area, so obstruction detection favours recall.

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

---

## Step 3: Roof faces, plane fitting, planar reconstruction

**Run:** `python scripts/run_pipeline.py --steps 1 2 3` (step 3 itself takes about 2 s)
**Evidence:** `python scripts/validate_faces.py` (synthetic accuracy and real-data stability)
**Code:** `src/roofsolar/faces.py`, `pipeline.py::run_faces`
**Outputs:** `outputs/step03_faces/`
- `faces.tif` (labels), `candidate.tif`
- `planar.tif` (plane heights), `roof_surface.tif` (planes plus cleaned DSM on non-face building pixels)
- `residual.tif` (raw − plane, the input for step 4)
- `faces.json` (per face: area, slope, aspect, RMSE, plane in UTM `z = aE + bN + c`)
- `edges.json` (adjacency with type and consistency), `stats.json`
- figures `00_summary`, `01_stages`, `02_profiles`; `validation/validation.json`

### 3.1 Findings that shaped the method
| Finding | Evidence | Consequence |
|---|---|---|
| True wall line is at about 1.5 m nDSM | smeared walls: steepest point at nDSM 1.4–2.2 m, eave tops at about 3.1 m (profiles) | footprint threshold 1.5 m, not 2.5 m |
| Glass lantern labelled vegetation by ExG | step 2 | fill holes in the building mask |
| Shaded tree canopy at NW not caught by ExG | mask leak | a roof is **what is planar**: canopy cannot form a well-fitting plane, so it drops out |
| Real faces: pitch 35–40°, 4 orientations, plane RMSE 1.2–4.4 cm | segment statistics | sliver test: real faces keep 40–78 % after 0.3 m erosion, slivers about 0 %. RMSE limit 5 cm (2× waviness) |
| Roof structure | faces on RGB | main hip roof (ridge 10.5 m) + south **ring wing** around a **4-sided glass pyramid lantern** + west wing with dormers; the main SW face continues as the ring's inner face (one plane, 2.3 cm RMSE) |

### 3.2 Method (`faces.segment_roof`)
1. **Candidate mask:** nDSM > 1.5 m and not vegetation; the largest connected region; holes filled.
2. **Plane-aware region growing** on the cleaned normals.
   - Seeds come from the most planar pixels (normal within 3° of its 5×5 mean).
   - A pixel joins when its normal is within **10°** of the *region's fitted plane* normal and it lies within **10 cm** of that plane.
   - Comparing with the region, not the neighbour, stops leakage across the provider-rounded creases, where normals bend gradually.
   - 10° ≈ 8× the cleaned normal scatter (1.25°); 10 cm ≈ 4× the waviness.
3. **Robust total-least-squares planes:** smallest singular vector of the centred points (orientation-independent, unbiased on steep faces, unlike z = ax + by + c regression; tested at 60°). Iteratively drop points beyond 2.5 robust σ **from the median residual**.
4. **Tidy:**
   - agglomerative coplanar merge: angle < 5°, union RMSE < 4 cm and ≤ 1.3 × pooled RMSE
   - drop slivers (< 15 % survive a 0.3 m erosion) and segments < 1 m²
   - drop poor fits (RMSE > 5 cm, i.e. mixed segments such as the roof-window patch)
5. **k-planes assignment:** each building pixel takes the vertically closest plane among faces within 1 m, if within 25 cm (more = an object; chimneys ≥ 0.3 m), then refit; 3 iterations, tidy, one more pass. **On a rounded crease the closest-plane switch happens exactly where the two planes are equal, i.e. on their intersection line, so sharp ridges, hips and valleys are rebuilt.**
6. **Eave reconstruction:** unassigned pixels in the 1 m band inside the footprint edge whose data is *below* an **adjacent** face's plane (smeared downwards) are given to that face, extending it to the wall line. Data *above* the plane is left for obstruction detection.
7. **Adjacency and consistency:**
   - Each touching pair is a crease (planes agree within 10 cm on the boundary) or a step.
   - Crease type comes from convexity: each plane, extended over the other face, lies above it means ridge (horizontal intersection) or hip; otherwise valley.
   - **Consistency:** the boundary must run along the planes' 3D intersection line: angle < 5° and offset ≤ max(15 cm, 3·σ_line).
   - σ_line = σ_z / |∇z_a − ∇z_b| is the uncertainty of the intersection position (error propagation), so nearly parallel planes are judged fairly.

Reconstructed surface = each face pixel takes its plane height.

### 3.3 Problems found and fixed while building it
| Problem | How it showed up | Fix |
|---|---|---|
| Robust plane fit kept outliers | unit test with 20 % gross outliers: 0.4° error, no points rejected | threshold centred on the **median residual** (a biased first fit offsets the inliers from 0); now exact, all outliers rejected |
| Roof-window patch accepted as a face | face RMSE 8.5 cm, on RGB = roof windows between dormers | RMSE > 5 cm means not a face (pixels go back to neighbours or to step 4) |
| Fixed consistency threshold unfair to near-parallel planes | 17–19 and 17–20 offsets of 0.4–0.6 m with only 7–9° between planes | uncertainty-aware threshold (σ_line) |
| **Faces lost by chained merging** | stability test: in 1 of 10 runs two large faces (35 m², 16 m²) vanished; a plane changed by **74.7°** | union-find chained A~B~C although A and C were not coplanar (5–6: parallel, 33 cm apart), and the blob then failed the RMSE test. Replaced by **greedy agglomerative merging** (best pair first, re-evaluated on the real union) plus the relative RMSE test. Worst plane change is now **0.16°** |
| Eave extension from a distant face | same run: edge pixels given to an opposite-facing face | only from a face within the eave band |

### 3.4 Validation
**A. Synthetic truth** (hip roof degraded like the real DSM, run through step 2 and step 3; 3 noise seeds):

| median \|error\| | raw | cleaned (step 2) | **reconstructed (step 3)** |
|---|---|---|---|
| faces | 1.9 cm | 1.2 cm | **0.2–0.5 cm** |
| creases (ridges/hips) | 6.8 cm | 6.8 cm | **0.3–0.8 cm** |
| walls/eaves inside footprint | 146 cm | 146 cm | **0.4–0.9 cm** |

- 4 of 4 faces every seed; plane error ≤ 0.2°; label accuracy 99.7 %; footprint IoU 0.957.
- Step 2 could not repair blurred edges, but plane reconstruction does: creases are about 10× better, and eaves go from metres to under 1 cm.

**B. Real-data stability** (10 runs with 1 mm perturbations, 25× below the waviness):
- 24–26 faces per run; pixel agreement ≥ 95.5 % (mean 97.2 %)
- **large-face planes change ≤ 0.16°** (median 0.007°)
- The face count varies only through small faces at clutter and one **genuinely ambiguous pair**: two NE regions whose single-plane RMSE is 1.21× the two-plane RMSE (in 9 of 10 runs they merge into one NE face).
- Region growing is order-dependent, which is why float32 vs float64 inputs alone gave 23 vs 25 faces.

**C. Edge restoration on real data:**
- On crease boundaries, plane height minus raw DSM: **ridges +3.5 cm, hips +2.1 cm, valleys −4.3 cm**.
- These are exactly the signs expected if the provider's smoothing rounded the creases: the reconstruction puts back the sharp apex and the sharp V.
- Profiles (`02_profiles.png`) show sharp ridges above the rounded data, and eaves extended to the wall line.

### 3.5 Results
| | |
|---|---|
| Faces | 25 (7 ridges, 21 hips, 13 valleys, 5 steps) |
| Face area | 457 m² plan, **559 m² true (sloped)**; 88 % of the 522 m² building |
| Not a face | 65 m²: dormers, chimneys, lantern apex, tree leak, wall smear (for step 4) |
| Face plane RMSE | median 2.6 cm, max 4.7 cm |
| Crease consistency | **81 % of 150 m** of creases; median boundary-to-intersection offset 5.9 cm (about half a pixel) |
| Largest faces | 84.3 m² (35°, SW 224°), 83.3 m² (36°, SE 135°), 49.7 m² (36°, NE 44°), 43.5 m² (37°, SW 224°) |
| Runtime | 1.8 s |

### 3.6 Limitations
- About 19 % of crease length is not consistent. It is concentrated in the cluttered west wing (dormers, tree, shade), the glass lantern (unreliable matching on glass) and nearly parallel planes.
- Small faces near clutter are less stable than large faces.
- One profile running along a ridge crosses a junction with a small bump, giving a 2-pixel artifact.
- The roof outline is still the pixel boundary of the footprint threshold. Straight, regularised outline edges are done when building polygons for the 3D model (step 5).
- Region growing is order-dependent; this is mitigated by the merge/assignment stages and quantified by the stability test.
- Dormer roofs are mostly left as "not a face" (too small or cluttered). They are not usable for standard panels anyway.

---

## Step 4: Obstruction detection and usable area

**Run:** `python scripts/run_pipeline.py --steps 1 2 3 4` (step 4 itself takes 0.6 s)
**Evidence:** `python scripts/validate_obstructions.py` (threshold sensitivity); recall against `data/reference/obstructions_reference.json` (9 hand-labelled objects) is reported in `stats.json`
**Code:** `src/roofsolar/obstructions.py`, `pipeline.py::run_obstructions`
**Outputs:** `outputs/step04_obstructions/`
- `obstructions.tif` (class codes), `keepout.tif`, `usable.tif`
- `height_above_roof.tif`, `colour_distance.tif`
- `obstructions.json` (every object: class, evidence, confidence, area, height; usable area per face), `stats.json`
- `00_summary.png`, `validation/sensitivity.json`

### 4.1 Two kinds of obstruction, two kinds of evidence
| Kind | Examples | Evidence | Why |
|---|---|---|---|
| Raised | dormers, chimneys, flues, vents, overhanging tree | **raw** DSM minus the face plane (non-face pixels use the nearest face's plane) | they stand out of the roof plane. The raw DSM is used because the cleaned DSM softens small objects (step 2 rule) |
| Flush | roof windows, skylights, small fittings | RGB colour anomaly vs the tiles of the **same face** | they lie in the plane and are invisible in height |
| Glass faces | the lantern pyramid | face-level glazing test | planar, so step 3 made them faces, but they cannot carry panels |

### 4.2 Method and measured parameters
- **Height:** raw-DSM noise on faces σ = 4.2 cm, so the threshold is **0.15 m (3.5σ)**. A 0.1 m opening removes 1-pixel specks. Below-plane data in the 1 m eave band is the smear rebuilt in step 3, not an obstruction.
- **Colour:**
  - CIELAB (lightness separated from colour).
  - Mahalanobis distance from the face's median colour, with one robust covariance (**Minimum Covariance Determinant**, which ignores the windows, shadows and capping in the sample) for all tiles.
  - Per-face location, because one global model flagged a whole brown-tiled roof section (17–20 m²) as anomalous.
  - Threshold 5.
  - Then a compactness filter (area 0.04–6 m², elongation ≤ 4, width ≥ 0.2 m), so that ridge/hip capping (lighter tiles in long thin lines) and valley strips are ignored while compact windows and vents are kept.
- **Glass faces:** median 5×5 lightness std > 6 (glazing frames; tiles ≤ 5.3, lantern ≥ 8.2) **and** more than 10 % of pixels with L > 80 (specular reflection; lantern p90 = 86–94, tiles ≤ 82). Both cues are required, so a shaded face (high contrast, dark) and a sunlit face (bright, smooth) are excluded. Exactly the 4 lantern faces qualify; the brown face does not.
- **Classification of raised objects:**
  - vegetation: canopy share > 30 % or > 4 m above the roof
  - dormer/structure: ≥ 2.5 m², or elongated (dormer cheek / wall smear strips)
  - chimney/small structure: compact and ≥ 0.4 m high
  - vent: otherwise
  - lantern part: enclosed by glass faces (the apex finial)
- **Colour-only objects:** "roof window / skylight / fitting" (medium confidence); darker than the face means "dark anomaly (shadow or dark fitting)" (low confidence).
- **Keep-out:** every obstruction grown by **0.3 m**. The DSM smear around raised objects decays to noise (median < 1 cm) at **0.3–0.4 m** (measured at 0.1 m steps). **Usable = faces − obstructions − keep-out.**

### 4.3 Decisions and the evidence behind them
| Question | Finding | Decision |
|---|---|---|
| Shadow or dark object? | the dark flue (ΔL −44.6, colour shift 8.5) is **colour-identical** to cast shadows (ΔL −26 to −47, shift 1–17). Shadows here turn bluer (lit by sky), so chromaticity does not separate them. A caster-distance rule also failed: the flue is 1.4 m from a dormer, like many shadows | the classes cannot be separated with these data, so **asymmetric costs** decide: a false keep-out loses a little area, a missed flue or skylight under a panel is a failed installation. Keep **every** compact colour anomaly as keep-out, labelled by confidence. Cost: 5.4 % of face area (12.6 m² from low-confidence dark anomalies) |
| Split merged NE dormers? | provider smoothing bridges the ~1 m gaps between dormers even above 0.5 m; watershed on height cores still merged them and broke the wall smear into dozens of fragments | keep connected components and report "dormer / structure" groups honestly. The keep-out mask, which is what the brief needs, is unaffected |
| Dormers "missed"? | dormer roofs were segmented as small faces in step 3 (correct geometry; not raised relative to their own plane) | left as faces; they are too small for standard panels and step 6's minimum-area rule rejects them |
| Robust covariance on a uniform roof | the end-to-end test (uniform synthetic colour) crashed MCD (singular covariance) | eigenvalue floor of 1 CIELAB unit (below the JND dE ≈ 2.3), fallback to the plain covariance. A first version **added** the floor to the variances, which shrank all distances and cut recall from 9/9 to 7/9; flooring eigenvalues leaves a well-conditioned covariance unchanged |

### 4.4 Validation
**Recall against the hand-labelled reference** (9 objects read from the orthophoto: 2 roof windows, 2 skylights, 1 chimney, 1 lantern, 3 fittings): **9/9**.
- Precision cannot be measured without full labels. Visual inspection shows that most of the extra colour detections are cast shadows and staining; they are kept as low-confidence keep-outs by design.

**Sensitivity** (`validate_obstructions.py`, one parameter at a time):

| parameter | values → recall / usable share |
|---|---|
| colour threshold | 4 → 9/9, 77 % · **5 → 9/9, 81 %** · 6 → 7/9, 84 % · 7 → 6/9, 85 % |
| height threshold | 0.10 → 9/9, 78 % · **0.15 → 9/9, 81 %** · 0.20 → 9/9, 82 % · 0.30 → 9/9, 83 % |
| keep-out buffer | 0 → 91 % · 0.2 → 85 % · **0.3 → 81 %** · 0.5 → 73 % (recall unaffected) |

- The colour threshold of 5 is the highest value with full recall.
- The height threshold is robust.
- The buffer is the main area driver, which is why it is set from the measured smear width rather than guessed.

**Unit tests (synthetic roof):**
- the chimney is found as a compact tall object
- a painted roof window is found from colour
- a thin bright capping line is *not* found
- a glazed face is found as glass and a tiled face is not
- the compactness filter and keep-out buffer behave as designed

The end-to-end test covers steps 1→4.

### 4.5 Results (`stats.json`)
| | |
|---|---|
| Objects | 58: glass lantern (2 parts, 33.1 m²), 5 dormer/structure groups (29.9 m²), 7 chimneys/small structures, 1 vent, 4 vegetation, 18 roof windows/skylights/fittings, 21 dark anomalies |
| Face area | 457 m² plan |
| Obstructions on faces | 42.7 m² plan |
| Obstructions + keep-out | 87.8 m² plan |
| **Usable roof** | **369.5 m² plan = 458.6 m² sloped, 81 % of face area** |

### 4.6 Limitations
- Shadows vs dark fittings cannot be separated from one RGB image; the conservative choice costs about 5 % of area.
- Adjacent dormers merge into groups (DSM smoothing).
- The class labels for some west-wing structures (chimney vs dormer part) are uncertain; the keep-out is unaffected.
- The reference is small (9 objects) and hand-labelled to ±1 m, so recall is indicative, not a statistical estimate.
- No shading or irradiance analysis: shadows reduce yield but are not treated as geometric obstructions (beyond the conservative keep-out).

---

## Step 5: 3D roof model

**Run:** `python scripts/run_pipeline.py --steps 1 2 3 4 5` (step 5 itself takes 0.1 s)
**Code:** `src/roofsolar/model3d.py`, `pipeline.py::run_model3d`
**Outputs:** `outputs/step05_model3d/`
- `roof_model.ply/.obj/.glb` (coloured mesh, local metres) + `model_metadata.json` (origin in CRS)
- `roof_faces_3d.geojson` (3D face polygons in EPSG:32630, for GIS)
- `roof_model.html` (interactive viewer), `stats.json`, `00_summary.png`

### 5.1 Method: a vector roof model with exact geometric consistency
1. **Face polygons:** polygonise the face labels. They form a *partition*, so neighbours share edges exactly.
2. **Noding** (found necessary, see 5.2), then **topology-preserving coverage simplification** (GEOS `coverage_simplify`, tolerance 0.15 m ≈ 1.5 px). Each shared edge is simplified once for both neighbours, so there are no gaps or overlaps.
3. **3D-consistent vertices:**
   - **Crease vertex** (2 faces meeting in a ridge, hip or valley): moved in plan onto the faces' **intersection line** z_a(x, y) = z_b(x, y). That line is linear: g·(x, y) = h with g = (n_xa/n_za − n_xb/n_zb, n_ya/n_za − n_yb/n_zb), h = d_a/n_za − d_b/n_zb. On it both planes give the same z.
   - **Junction** (≥ 3 crease faces): the least-squares common point of their planes, solving (NᵀN)p = Nᵀd. This is **exact** for 3 planes.
   - Moves are limited to 0.3 m (≈ 3 px, within the boundary uncertainty from step 3); beyond that the vertex is left and reported.
   - **Step** vertices (planes do not meet) keep each face's own z; a vertical wall closes the gap.
4. **Mesh:**
   - Each polygon is triangulated by **constrained Delaunay** (holes kept) and every vertex is lifted with its face's plane.
   - Walls: outer edges go down to the ground (DTM from step 1), found by probing 0.15 m outside each edge; step edges get a wall between the two planes.
   - Raised obstructions from step 4 become prisms up to their 90th-percentile raw height.
5. **Exports:**
   - Meshes are written in **local metres plus an origin**, because float32 mesh formats cannot store UTM coordinates (about 6.4·10⁵ m) to cm precision.
   - The GeoJSON is in full CRS coordinates (float64 JSON).

### 5.2 Problems found and fixed
| Problem | Evidence | Fix |
|---|---|---|
| No vertex was shared by ≥ 3 faces, so junctions were never recognised | 15 pixel corners touch 3 faces, but **0** polygon vertices did | the polygonizer drops **collinear** vertices: at a T-junction the point is a vertex of two faces but not of the third, whose edge runs straight through. **Noding** (`shapely.snap` of each polygon to its neighbours' boundaries) inserts it, giving 13 shared junction vertices |
| Crease-gap metric reported up to 36 cm | gaps included faces meeting in a *step* at that vertex, and compared a different pair than the one snapped | measure only between the faces that meet in creases (the snapped pair, or all junction planes). Junctions where the plane solution was too far are a separate category (unresolved) |
| `planarity_max_m` was hard-coded to 0 | a claim, not a measurement | measured on the mesh: max distance of any roof vertex from its face plane |

### 5.3 Validation
| Check | Result |
|---|---|
| Simplification | 4974 → 880 vertices; max face-area change 1.5 %; zero overlap; all polygons valid |
| Vertex types | 289 crease, 7 junction, 72 step, 5 unresolved junction, 507 single (eaves, object edges) |
| **Crease/junction consistency** | max height disagreement **3.8e-13 m** (floating-point zero) |
| Snap moves | median 4.7 cm, max 28.6 cm |
| **Planarity** (measured) | max vertex-to-plane distance **1.4e-14 m** |
| Unresolved junctions | 5; the faces' planes meet 0.4–1.4 m away (one near-parallel triple, condition 7·10⁵); remaining gaps 9, 32, 42, 10, 30 cm. All are at small clutter faces near dormers or the lantern |
| Model vs step-3 raster | label agreement 99.7 %; height difference median 0 |
| Model vs raw DSM | median \|Δz\| 2.8 cm (noise level); RMSE 10.9 cm (dominated by the smeared edges the model rebuilds) |
| Roof area | mesh 560.8 m² vs raster 559.2 m² |
| Analytic test | synthetic hip roof: 4 faces, exactly 2 junctions, ridge ends within **5 cm** of the analytic 3D positions, crease gaps < 1e-9 m |

### 5.4 Limitations
- The outline follows the simplified pixel boundary (wavy by a few cm); it is not regularised to straight, orthogonal building edges.
- Objects are simple flat-topped prisms (LOD1-style). Dormers are not modelled with their own roofs, except where step 3 made dormer roofs faces.
- Building pixels that are neither a face nor a raised object (wall smear, tree leak) are left as holes rather than invented.
- The model is not a closed solid (holes and object prisms are not stitched), so no volume is reported.
