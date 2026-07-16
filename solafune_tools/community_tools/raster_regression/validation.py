"""Validation helpers for raster (GeoTIFF-zip) regression competition submissions.

Raster regression competitions (e.g. Precipitation Nowcasting From Space) are scored on
zip archives of GeoTIFF tiles rather than JSON/CSV files, so the existing
``competition_tools.quality_checker`` validators do not apply. These helpers catch the
failure modes that silently burn daily submission quota or corrupt scores: missing or
misnamed tiles, wrong grid shape, NaN/Inf pixels, and out-of-range values.

Error codes follow the ``quality_checker`` convention (0 = valid, one code per failure
mode) so they can be dispatched the same way.
"""

import os
import zipfile
from io import BytesIO
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import rasterio
    from rasterio.io import MemoryFile
except ImportError as _err:  # pragma: no cover
    raise ImportError("raster_regression tools require rasterio: pip install rasterio") from _err

ERROR_MESSAGES = {
    0: "Valid",
    30: "Cannot open the zip archive.",
    31: "No .tif files found inside the archive.",
    32: "Tile count mismatch against the reference archive.",
    33: "Tile file names do not match the reference archive.",
    34: "One or more tiles could not be read as GeoTIFF.",
    35: "One or more tiles have an unexpected raster shape.",
    36: "One or more tiles contain NaN or Inf pixels.",
    37: "One or more tiles contain values outside the allowed range.",
}


def return_error_message(error_code: int) -> str:
    """Map a raster-submission error code to a human-readable message."""
    return ERROR_MESSAGES.get(error_code, "Unknown error, contact the developer for more information.")


def _tif_names(zf: zipfile.ZipFile) -> List[str]:
    """All .tif/.tiff member paths in the archive (directories excluded)."""
    return [
        n for n in zf.namelist()
        if n.lower().endswith((".tif", ".tiff")) and not n.endswith("/")
    ]


def _read_member(zf: zipfile.ZipFile, name: str) -> np.ndarray:
    """Read one archive member as a (bands, H, W) float64 array."""
    with MemoryFile(BytesIO(zf.read(name))) as mem:
        with mem.open() as src:
            return src.read().astype("float64")


def validate_submission_zip(
    pred_zip_path: str,
    reference_zip_path: Optional[str] = None,
    expected_shape: Optional[Tuple[int, ...]] = None,
    value_min: Optional[float] = 0.0,
    value_max: Optional[float] = None,
    max_report_items: int = 20,
) -> Tuple[int, Dict]:
    """Validate a prediction zip before uploading it.

    Args:
        pred_zip_path: Path to the submission archive to check.
        reference_zip_path: Optional sample-submission archive. When given, the
            prediction must contain exactly the same tile file names (paths are
            compared by base name, so an extra top-level folder is fine).
        expected_shape: Optional required raster shape. Give ``(H, W)`` to require
            single-band tiles of that grid, or ``(bands, H, W)`` to also pin the
            band count. When omitted and a reference is given, each tile is checked
            against the shape of the reference tile of the same name.
        value_min: Lowest allowed pixel value (default 0.0 — negative rain/cost/mass
            predictions are almost always a bug). Pass ``None`` to skip.
        value_max: Highest allowed pixel value. Pass ``None`` to skip.
        max_report_items: Cap on offending file names listed per problem.

    Returns:
        (error_code, report) — ``error_code`` 0 means uploadable. ``report`` carries
        the offending file names per check, plus ``stats`` (tile count, global
        min/max/mean, all-zero tile count) and non-fatal ``warnings``.
    """
    report: Dict = {"stats": {}, "warnings": []}

    try:
        zf = zipfile.ZipFile(pred_zip_path)
    except (OSError, zipfile.BadZipFile):
        return 30, report

    with zf:
        names = sorted(_tif_names(zf))
        if not names:
            return 31, report
        report["stats"]["n_tiles"] = len(names)

        ref_shapes: Dict[str, Tuple[int, ...]] = {}
        if reference_zip_path is not None:
            try:
                ref = zipfile.ZipFile(reference_zip_path)
            except (OSError, zipfile.BadZipFile):
                return 30, report
            with ref:
                ref_names = sorted(_tif_names(ref))
                if len(names) != len(ref_names):
                    report["expected_n_tiles"] = len(ref_names)
                    return 32, report
                pred_base = {os.path.basename(n): n for n in names}
                missing = [os.path.basename(n) for n in ref_names
                           if os.path.basename(n) not in pred_base]
                if missing:
                    report["missing_files"] = missing[:max_report_items]
                    return 33, report
                if expected_shape is None:
                    for n in ref_names:
                        with MemoryFile(BytesIO(ref.read(n))) as mem:
                            with mem.open() as src:
                                ref_shapes[os.path.basename(n)] = (src.count, src.height, src.width)

        unreadable, bad_shape, non_finite, out_of_range = [], [], [], []
        gmin, gmax, gsum, gcount, all_zero = np.inf, -np.inf, 0.0, 0, 0

        for n in names:
            try:
                arr = _read_member(zf, n)
            except Exception:
                unreadable.append(n)
                continue

            shape_ok = True
            if expected_shape is not None:
                want = tuple(expected_shape)
                if len(want) == 2:
                    shape_ok = arr.shape[0] == 1 and arr.shape[1:] == want
                else:
                    shape_ok = arr.shape == want
            elif ref_shapes:
                shape_ok = arr.shape == ref_shapes.get(os.path.basename(n), arr.shape)
            if not shape_ok:
                bad_shape.append((n, arr.shape))

            if not np.isfinite(arr).all():
                non_finite.append(n)
                continue

            amin, amax = float(arr.min()), float(arr.max())
            if (value_min is not None and amin < value_min) or \
               (value_max is not None and amax > value_max):
                out_of_range.append((n, amin, amax))
            gmin, gmax = min(gmin, amin), max(gmax, amax)
            gsum += float(arr.sum())
            gcount += arr.size
            if amax == 0.0 and amin == 0.0:
                all_zero += 1

        report["stats"].update({
            "value_min": None if gcount == 0 else gmin,
            "value_max": None if gcount == 0 else gmax,
            "value_mean": None if gcount == 0 else gsum / gcount,
            "all_zero_tiles": all_zero,
        })
        if all_zero:
            report["warnings"].append(
                f"{all_zero} tiles are entirely zero — expected for dry scenes, "
                "but verify if the count looks high."
            )

        for code, items, key in (
            (34, unreadable, "unreadable"),
            (35, bad_shape, "bad_shape"),
            (36, non_finite, "non_finite"),
            (37, out_of_range, "out_of_range"),
        ):
            if items:
                report[key] = items[:max_report_items]
                return code, report

    return 0, report


def band_count_report(
    data_dir: str,
    expected_bands: int,
    pattern: str = ".tif",
) -> List[Tuple[str, int, Tuple[int, ...]]]:
    """List dataset tiles whose band count differs from ``expected_bands``.

    Some competition tiles ship with fewer bands than nominal (e.g. GOES tiles missing
    their visible channels), and the GeoTIFF itself carries no record of *which* bands
    are absent — naive channel indexing then reads the wrong band for every channel of
    the tile. Run this once over a dataset directory to find such tiles so you can
    exclude or special-case them.

    Args:
        data_dir: Directory scanned recursively for raster files.
        expected_bands: Nominal band count (e.g. 16 for full-disk geostationary stacks).
        pattern: File-name suffix filter.

    Returns:
        List of ``(path, band_count, (bands, H, W))`` for every deviating tile.
    """
    deviant = []
    for root, _dirs, files in os.walk(data_dir):
        for f in sorted(files):
            if not f.lower().endswith(pattern):
                continue
            path = os.path.join(root, f)
            try:
                with rasterio.open(path) as src:
                    if src.count != expected_bands:
                        deviant.append((path, src.count, (src.count, src.height, src.width)))
            except Exception:
                deviant.append((path, -1, ()))
    return deviant
