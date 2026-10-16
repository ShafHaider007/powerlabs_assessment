# 2. Input Data

> Source: `docs/Technical Assessment.pdf`, Section 2 (wording copied from the PDF)

## From the PDF
> You will be provided with:
> - DSM (Digital Surface Model)
> - RGB Aerial Imagery

## Files actually provided (`data/raw/`, checked with gdalinfo)

| File | Size (px) | Type | Pixel size | CRS | Value range | NoData |
|---|---|---|---|---|---|---|
| `data/raw/dsm_full.tif` | 797 × 802 | Float32, 1 band | 0.1 m | WGS 84 / UTM zone 30N (EPSG:32630) | 105.91 to 131.16 m | not set |
| `data/raw/dsm_clipped.tif` | 757 × 623 | Float32, 1 band | 0.1 m | EPSG:32630 | 106.47 to 131.16 m | -9999 |
| `data/raw/rgb.tif` | 797 × 802 | Byte, 3 bands (R,G,B) | 0.1 m | EPSG:32630 | 1 to 255 | 0 |

**Origins (top-left corner, UTM metres):**
- `dsm_full.tif` and `rgb.tif`: (640495.0, 5794761.9), so the two grids line up pixel for pixel.
- `dsm_clipped.tif`: (640499.0, 5794758.3). That is an offset of +40 columns and +36 rows inside the full grid.

## Notes
- DSM heights are in metres. The 10 cm resolution is high enough to see chimneys and skylights.
- The `-9999` pixels in `dsm_clipped.tif` are the missing data the assessment asks us to fill.
- UTM 30N at northing ≈ 5.79e6 puts the site at about 52°N (likely the UK), so south-facing roof faces get the most sun.
- Slope/aspect convention to use: azimuth measured clockwise from north (0° = N, 90° = E, 180° = S).
