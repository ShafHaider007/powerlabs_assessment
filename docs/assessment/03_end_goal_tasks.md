# 3. What We Need To Do (End Goal)

> Source: `docs/Technical Assessment.pdf`, Section 3 (wording copied from the PDF)

## From the PDF
> Develop a workflow that addresses the following:
> - Calculate elevation, slope, and aspect/azimuth.
> - Identify roof faces and reconstruct a geometrically consistent roof surface.
> - Regularize noisy elevation values and handle missing data through appropriate
>   interpolation or reconstruction.
> - Identify rooftop obstructions, such as chimneys, skylights, and roof windows, and
>   exclude them from usable areas.
> - Generate a 3D representation of the reconstructed roof.
> - Identify suitable roof areas and place solar panels while respecting geometric
>   constraints, including available usable area, sufficient space for standard-sized panels,
>   compact panel placement, and avoidance of rooftop obstructions such as chimneys,
>   windows, and skylights.
>
> You are free to choose your own algorithms and implementation approach.

## Requirement checklist
| # | Requirement | Done means |
|---|---|---|
| R1 | Elevation, slope, aspect/azimuth | Rasters (and figures) for elevation, slope (°) and aspect (° from north) |
| R2 | Roof faces and a consistent surface | Each face segmented with a fitted plane; faces meet cleanly at ridges, hips and valleys |
| R3 | Regularize noise, fill missing data | DSM denoised; `-9999` holes filled by interpolation or plane reconstruction |
| R4 | Obstructions | Chimneys, skylights and roof windows detected and removed (with a buffer) from usable area |
| R5 | 3D representation | 3D mesh or model of the reconstructed roof (interactive view and/or exported file) |
| R6 | Solar panel placement | Standard-size panels, fully inside usable area, no overlap, clear of obstructions, compact packing |

## Panel constraints called out in the PDF
- Available usable area
- Enough space for **standard-sized panels** (assumption: about 1.7 m × 1.0 m, to be stated in the write-up)
- **Compact** placement (tight grid, as many panels as possible)
- Avoid chimneys, windows and skylights

## Freedom
The choice of algorithms is ours, but every choice must be justified (see `05_evaluation_criteria.md`).
