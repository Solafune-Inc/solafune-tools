# Raster Regression Tools — Usage

Tools for competitions scored on zip archives of GeoTIFF tiles (raster regression).
Everything below needs only `numpy`, `rasterio`, and `tqdm` (already in the base
requirements).

```python
from solafune_tools.community_tools.raster_regression import (
    validate_submission_zip,
    return_error_message,
    band_count_report,
    blend_submission_zips,
    IsotonicCalibrator,
    calibrate_submission_zip,
)
```

## 1. Validate a submission before uploading

Check your prediction archive against the competition's sample submission. This catches
the mistakes that either get a submission rejected or silently scored as garbage:
missing/misnamed tiles, wrong grid shape, NaN/Inf pixels, negative values.

```python
code, report = validate_submission_zip(
    "my_submission.zip",
    reference_zip_path="sample_submission.zip",  # tile names + shapes come from here
    value_min=0.0,                               # rain rate can't be negative
)
print(return_error_message(code))   # "Valid" when code == 0
print(report["stats"])              # n_tiles, value min/max/mean, all-zero tile count
```

Error codes (mirrors the `competition_tools.quality_checker` convention):

| code | meaning |
|------|---------|
| 0    | valid |
| 30   | cannot open the zip archive |
| 31   | no `.tif` files inside |
| 32   | tile count mismatch vs. reference |
| 33   | tile names don't match reference |
| 34   | unreadable tile |
| 35   | unexpected raster shape |
| 36   | NaN/Inf pixels |
| 37   | values outside the allowed range |

On failure, `report` lists the offending files (capped at 20 per problem).
`report["warnings"]` carries non-fatal observations (e.g. many all-zero tiles).

Without a reference archive, pass `expected_shape=(41, 41)` (or `(bands, H, W)`) to pin
shapes explicitly.

## 2. Blend submissions (zip-level ensembling)

Averaging diverse predictions is the most reliable free score improvement in regression
competitions, and doing it on the submission artifacts means no re-inference and no
pipeline coupling — you can blend this week's model with last week's.

```python
blend_submission_zips(
    ["model_a.zip", "model_b.zip"],
    out_path="blend_ab.zip",
)                                          # equal weights

blend_submission_zips(
    ["model_a.zip", "model_b.zip", "model_c.zip"],
    out_path="blend_abc.zip",
    weights=[2, 1, 1],                     # normalized automatically
    clip_min=0.0,                          # keep physical validity after averaging
)
```

Tiles are matched by base file name, geo metadata is copied from the first archive, and
tiles stream one at a time so memory use stays flat.

**Tip:** blending helps most when the inputs are *diverse* (different architectures,
feature sets, or training configs) and roughly comparable in score. Two strong-but-
different submissions usually beat either one alone; ten near-copies of the same model
do not add much.

## 3. Isotonic output calibration

Models trained with MSE-family losses on skewed targets (rainfall, biomass, cost)
systematically under-predict the high tail. Isotonic calibration learns a monotone
correction from *held-out* predictions and applies it as a post-process.

```python
# 1) Fit on out-of-fold predictions (never on training-set predictions!)
cal = IsotonicCalibrator(n_bins=64).fit(oof_pred, oof_true)   # any-shape arrays

# 2) Apply to arrays...
test_pred_calibrated = cal.transform(test_pred)

# ...or directly to a submission archive
calibrate_submission_zip("my_submission.zip", "my_submission_cal.zip", cal)

# 3) Persist for reproducibility
cal.save("calibration.json")
cal = IsotonicCalibrator.load("calibration.json")
```

When your data mixes sources with different error profiles (e.g. three geostationary
satellites), fit one calibrator per source and route tiles by file name:

```python
cals = {
    "himawari": IsotonicCalibrator().fit(oof_pred_him, oof_true_him),
    "goes":     IsotonicCalibrator().fit(oof_pred_goes, oof_true_goes),
    "meteosat": IsotonicCalibrator().fit(oof_pred_met, oof_true_met),
}
calibrate_submission_zip(
    "my_submission.zip", "calibrated.zip",
    calibrator=cals,
    group_fn=lambda name: next(s for s in cals if s in name.lower()),
)
```

Implementation notes: predictions are quantile-binned (`n_bins`), bin means are made
monotone with the pool-adjacent-violators algorithm, and the mapping is linearly
interpolated between bin centers with flat extrapolation outside the fitted range.
No scikit-learn dependency.

## 4. Find tiles with missing bands

Multi-band competition tiles sometimes ship with fewer bands than nominal, and the
GeoTIFF format does not record *which* bands are absent — naive `array[channel_i]`
indexing then reads the wrong band for every channel of that tile. Scan for such tiles
once and decide how to handle them (exclude, special-case, or query the host):

```python
deviant = band_count_report("evaluation_dataset/", expected_bands=16)
for path, count, shape in deviant:
    print(path, count, shape)
```

## Provenance

Extracted and generalized from the authors' pipeline in the *Precipitation Nowcasting
From Space* competition, where zip-level blending and per-satellite isotonic
calibration each improved leaderboard RMSE. Original code; PAVA follows Barlow et al.
(1972). Developed with assistance from Claude (Anthropic).
