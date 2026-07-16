"""Isotonic output calibration for raster regression predictions.

Regression models trained with MSE-family losses on skewed targets (rainfall, biomass,
cost, ...) are systematically miscalibrated: they under-predict the high tail and
over-predict near zero. Isotonic regression learns a monotone mapping from predicted
to observed values on held-out (out-of-fold) data and applies it as a post-process —
a few lines that reliably buy RMSE without touching the model.

The fit uses the classic pool-adjacent-violators algorithm (PAVA; Barlow et al., 1972,
"Statistical Inference under Order Restrictions") on quantile-binned predictions, so it
needs only numpy and is O(n log n) in the number of held-out samples.
"""

import json
import os
import zipfile
from io import BytesIO
from typing import Callable, Dict, Optional, Union

import numpy as np
from tqdm import tqdm

from rasterio.io import MemoryFile

from solafune_tools.community_tools.raster_regression.validation import _tif_names


def _pava(y: np.ndarray, w: np.ndarray):
    """Weighted pool-adjacent-violators: least-squares non-decreasing fit to y.

    Returns the pooled block values and block weights; expanding blocks back to
    the original positions is the caller's job (block weights are sums of the
    input weights they absorbed).
    """
    level_y = y.astype("float64").copy()
    level_w = w.astype("float64").copy()
    n = len(level_y)
    i = 0
    while i < n - 1:
        if level_y[i] <= level_y[i + 1] + 1e-15:
            i += 1
            continue
        # merge violating neighbors into one block at their weighted mean
        wsum = level_w[i] + level_w[i + 1]
        level_y[i] = (level_y[i] * level_w[i] + level_y[i + 1] * level_w[i + 1]) / wsum
        level_w[i] = wsum
        level_y = np.delete(level_y, i + 1)
        level_w = np.delete(level_w, i + 1)
        n -= 1
        # a merge can violate the previous pair — step back
        if i > 0:
            i -= 1
    return level_y, level_w


class IsotonicCalibrator:
    """Monotone (isotonic) mapping fit on held-out predictions vs. ground truth.

    Usage::

        cal = IsotonicCalibrator(n_bins=64).fit(oof_pred, oof_true)
        test_pred_calibrated = cal.transform(test_pred)
        cal.save("calibration.json")   # later: IsotonicCalibrator.load(...)

    Fit on *out-of-fold* predictions only — calibrating on training-set predictions
    just memorizes residual noise.
    """

    def __init__(self, n_bins: int = 64):
        self.n_bins = int(n_bins)
        self.x_: Optional[np.ndarray] = None  # bin-mean predictions (knots)
        self.y_: Optional[np.ndarray] = None  # isotonic bin-mean targets

    def fit(self, y_pred: np.ndarray, y_true: np.ndarray) -> "IsotonicCalibrator":
        """Fit the mapping from flattened predictions to flattened targets."""
        p = np.asarray(y_pred, dtype="float64").ravel()
        t = np.asarray(y_true, dtype="float64").ravel()
        if p.shape != t.shape:
            raise ValueError("y_pred and y_true must have the same number of elements.")
        if p.size < self.n_bins * 2:
            raise ValueError(f"Need at least {self.n_bins * 2} samples to fit {self.n_bins} bins.")

        edges = np.unique(np.quantile(p, np.linspace(0.0, 1.0, self.n_bins + 1)))
        if len(edges) < 3:  # near-constant predictions: single global shift
            self.x_ = np.array([p.min(), p.max() + 1e-9])
            self.y_ = np.array([t.mean(), t.mean()])
            return self
        idx = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, len(edges) - 2)

        counts = np.bincount(idx, minlength=len(edges) - 1).astype("float64")
        keep = counts > 0
        mean_p = np.bincount(idx, weights=p, minlength=len(edges) - 1)[keep] / counts[keep]
        mean_t = np.bincount(idx, weights=t, minlength=len(edges) - 1)[keep] / counts[keep]

        level_y, level_w = _pava(mean_t, counts[keep])
        # expand pooled levels back onto the kept bins
        y_iso = np.empty_like(mean_t)
        pos = 0
        remaining = counts[keep].copy()
        for ly, lw in zip(level_y, level_w):
            acc = 0.0
            while pos < len(y_iso) and acc < lw - 1e-9:
                y_iso[pos] = ly
                acc += remaining[pos]
                pos += 1
        self.x_, self.y_ = mean_p, y_iso
        return self

    def transform(self, y_pred: np.ndarray) -> np.ndarray:
        """Apply the fitted mapping (flat extrapolation beyond the fitted range)."""
        if self.x_ is None:
            raise RuntimeError("Calibrator is not fitted.")
        p = np.asarray(y_pred, dtype="float64")
        return np.interp(p, self.x_, self.y_).astype(y_pred.dtype if
                                                     np.issubdtype(np.asarray(y_pred).dtype, np.floating)
                                                     else "float64")

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump({"n_bins": self.n_bins, "x": self.x_.tolist(), "y": self.y_.tolist()}, f)

    @classmethod
    def load(cls, path: str) -> "IsotonicCalibrator":
        with open(path) as f:
            d = json.load(f)
        cal = cls(n_bins=d["n_bins"])
        cal.x_ = np.asarray(d["x"], dtype="float64")
        cal.y_ = np.asarray(d["y"], dtype="float64")
        return cal


def calibrate_submission_zip(
    in_zip_path: str,
    out_zip_path: str,
    calibrator: Union[IsotonicCalibrator, Dict[str, IsotonicCalibrator]],
    group_fn: Optional[Callable[[str], str]] = None,
    clip_min: Optional[float] = 0.0,
    clip_max: Optional[float] = None,
    progress: bool = True,
) -> str:
    """Apply isotonic calibration to every tile of a submission archive.

    Args:
        in_zip_path: Submission archive to calibrate.
        out_zip_path: Calibrated archive to write.
        calibrator: A fitted :class:`IsotonicCalibrator`, or a dict of them keyed by
            group name for per-group calibration (e.g. one per satellite/sensor).
        group_fn: Maps a tile base name to its group key. Required when
            ``calibrator`` is a dict. Tiles whose group has no calibrator pass
            through unchanged.
        clip_min / clip_max: Clip applied after calibration (default floor 0.0).
        progress: Show a tqdm progress bar.

    Returns:
        ``out_zip_path``.
    """
    grouped = isinstance(calibrator, dict)
    if grouped and group_fn is None:
        raise ValueError("group_fn is required when calibrator is a dict.")

    with zipfile.ZipFile(in_zip_path) as zin, \
            zipfile.ZipFile(out_zip_path, "w", zipfile.ZIP_DEFLATED) as zout:
        names = sorted(_tif_names(zin))
        if not names:
            raise ValueError(f"No .tif files found in {in_zip_path}")
        for name in tqdm(names, desc="calibrate", disable=not progress):
            with MemoryFile(BytesIO(zin.read(name))) as mem:
                with mem.open() as src:
                    arr = src.read().astype("float64")
                    profile = src.profile.copy()
            cal = calibrator.get(group_fn(os.path.basename(name))) if grouped else calibrator
            if cal is not None:
                arr = cal.transform(arr)
            if clip_min is not None or clip_max is not None:
                arr = np.clip(arr, clip_min, clip_max)
            dtype = profile.get("dtype", "float32")
            with MemoryFile() as mem:
                with mem.open(**profile) as dst:
                    dst.write(arr.astype(dtype))
                zout.writestr(name, mem.read())
    return out_zip_path
