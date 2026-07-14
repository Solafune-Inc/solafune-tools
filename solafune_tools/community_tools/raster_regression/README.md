# Raster Regression Tools

## Description

Utilities for competitions scored on **zip archives of GeoTIFF tiles** (raster
regression), such as *Precipitation Nowcasting From Space*. The existing
`competition_tools` validators cover JSON (detection/segmentation) and CSV
submissions; this module fills the raster gap with three things every raster-regression
participant ends up re-implementing:

1. **`validate_submission_zip`** — pre-upload submission checking (tile count/names vs.
   the sample submission, raster shape, NaN/Inf, negative values) with
   `quality_checker`-style error codes. Catches format rejects *before* they burn a
   daily submission slot.
2. **`blend_submission_zips`** — pixel-wise (weighted) averaging of N submission
   archives. Ensembling at the artifact level works across pipelines and frameworks
   with no re-inference; blending two diverse submissions typically beats both inputs.
3. **`IsotonicCalibrator` / `calibrate_submission_zip`** — dependency-free isotonic
   (PAVA) output calibration fit on out-of-fold predictions, with optional per-group
   (e.g. per-satellite) calibrators. Corrects the systematic tail under-prediction of
   MSE-trained models on skewed targets.

A small `band_count_report` helper is included to locate dataset tiles whose band
count deviates from nominal (e.g. GOES tiles missing their visible channels), since
GeoTIFFs carry no record of *which* bands are absent and silent positional
misalignment is easy to hit.

## Impact to the data and Example of the dataset

These tools operate on prediction artifacts and datasets of any raster regression
competition (single- or multi-band GeoTIFF tiles, any grid size):

- Validation prevents malformed uploads (wrong tile set, NaN pixels, negative
  precipitation) from wasting quota or scoring as garbage.
- Blending and calibration are direct, generic score improvements: in the
  Precipitation Nowcasting From Space competition, a 50/50 blend of two diverse
  submissions and per-satellite isotonic calibration each improved leaderboard RMSE
  for the authors' pipeline.

## Usage Documentation

Refer to this [link](/docs/community_tools/raster_regression.md) for the usage of this tool.

## Attribution

Isotonic regression via pool-adjacent-violators follows Barlow, Bartholomew, Bremner &
Brunk (1972), *Statistical Inference under Order Restrictions*. Original implementation
for this module; developed with assistance from Claude (Anthropic) during the
Precipitation Nowcasting From Space competition.

Contact: open an issue on this repository.
