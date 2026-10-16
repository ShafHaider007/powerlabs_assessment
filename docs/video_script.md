# Video Demonstration: Script and Storyboard (≤ 5 minutes)

**Before recording:**
- Run `python scripts/run_pipeline.py` once so all outputs exist.
- Open these in tabs: `outputs/step06_panels/overview.png`, `outputs/step06_panels/roof_with_panels.html`, the step figures in `outputs/step0*/`, and the repo in an editor.
- Speak slowly, in short sentences. Each section has one main message (in bold).

| Time | Show on screen | Say (talking points) |
|---|---|---|
| **0:00–0:25** Intro | `overview.png` | "This is my solution to the roof reconstruction and solar panel task. From a DSM and an RGB image, it rebuilds the roof as planar faces, finds obstructions, builds a 3D model, and places 45 standard panels, about 19 kilowatt-peak. **Every step is driven by measurements of this data and is validated.**" |
| **0:25–0:50** Run it | terminal: `python scripts/run_pipeline.py`; README | "The whole pipeline is one command and runs in under a minute. It is a Python package with 110 tests, one module per step, and every parameter in one config file." |
| **0:50–1:25** Step 1 | `step01_terrain/00_summary.png`, then `06_data_provenance.png` | "Slope and aspect use Horn's method, the same as GDAL. I cross-checked against gdaldem: they agree to a millionth of a degree. Aspect is corrected to true north. I also found that the DSM is a composite: **the provider smoothed the building areas**, so roof edges are blurred. That finding shaped the next steps." |
| **1:25–2:05** Step 2 | `step02_cleaning/benchmark/benchmark_smoothing_profiles_realistic.png`, `03_noise_reduction.png` | "To remove noise I used TGV, a filter that assumes flat planes with sharp creases, which is what a roof is. I compared it with Gaussian, median, bilateral and TV on a synthetic roof with the noise I measured on this DSM. **TGV was the only one that cleaned the faces without damaging the edges.** On the real roof, the normal noise dropped from 3 to 1.25 degrees." |
| **2:05–2:50** Step 3 | `step03_faces/01_stages.png`, `00_summary.png`, `02_profiles.png` | "Faces are found by region growing on surface normals, then robust plane fitting and merging. Each pixel then takes its closest plane. On a rounded ridge, that switch happens exactly where the two planes intersect, **so the sharp ridges and eaves are rebuilt.** On a synthetic test, ridge errors went from 7 cm to under 1 cm. A stability test also found a merging bug, which I fixed." |
| **2:50–3:25** Step 4 | `step04_obstructions/00_summary.png` | "Raised objects like dormers and chimneys come from height above the roof plane. Flush roof windows are invisible in height, so I use colour, compared with the tiles of the same face. The glass lantern is detected by its contrast and reflections. **All 9 objects I labelled by hand were found.** Unclear dark spots are kept out on purpose: losing a little area is better than a panel over a flue." |
| **3:25–3:55** Step 5 | `roof_with_panels.html` (rotate the model), `step05_model3d/00_summary.png` | "The 3D model is vector-based. Where two faces meet, I move the vertex onto the line where their planes intersect, so **neighbouring faces agree exactly**: the measured gap is about 10 to the minus 13 metres. It exports to PLY, GLB and GeoJSON." |
| **3:55–4:40** Step 6 | `step06_panels/00_summary.png`, `01_3d_panels.png` | "Suitability comes from a simple solar model: south-east and south-west faces get 95 percent of the optimum, north-east and north-west only 64, so those are excluded. Panels are placed in each face's own plane, so sizes are true sloped distances. Dynamic programming gives the best row layout. **Every panel is checked automatically: no overlaps, gaps and setbacks kept, and nothing on an obstruction.**" |
| **4:40–5:00** Close | `docs/technical_report.md` (limitations section) | "Main limitations: no shading analysis yet, and the energy figure is a first-order estimate. Everything is documented, with the evidence, in the technical report. Thank you." |

## Likely interview questions (with short answers)

- **Why Horn and not np.gradient?** Lower noise (0.43σ/d vs 0.71σ/d), and it is the GIS standard, so I could cross-check exactly with GDAL.
- **Why TGV?** Its prior is piecewise planar: smooth faces with sharp creases. It won on a synthetic roof with the measured noise model and was confirmed on the real roof. TV flattens slopes; Gaussian blurs edges.
- **How were the TGV parameters chosen?** By grid search on the realistic synthetic roof, under the rule "walls no worse than the input", then confirmed on real roof tiles (noise halved, ridges lowered by only about 1.5 cm).
- **Why total least squares?** It minimises perpendicular distance, so it does not depend on orientation and is unbiased on steep faces.
- **How are sharp ridges rebuilt?** By closest-plane assignment, then snapping vertices onto plane intersection lines in the 3D model.
- **Why did the face count change between runs?** Region growing depends on order, and one pair of faces is genuinely ambiguous (single-plane RMSE only 1.21× the two-plane RMSE). Large faces change by at most 0.16°.
- **Why keep dark anomalies?** A shadow and a dark flue look the same in one RGB image. The costs are asymmetric, so I favour recall (about 5 % of area).
- **Why only 45 panels on a big roof?** The outer faces of the ring roof are about 3 m of usable slope deep after setbacks, so only 2 landscape rows fit, and NE/NW faces are excluded. The DP shows 45 is the optimum for row layouts. The setback is the most sensitive assumption (0.2 m gives 54).
- **What would you do next?** Shading by ray-casting on the 3D mesh, PVGIS irradiance, regularised building outlines, and a faster TGV.
