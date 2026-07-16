import json
import os
import shutil
import tempfile
import unittest
import zipfile
from io import BytesIO

import numpy as np
import rasterio
from rasterio.io import MemoryFile

from solafune_tools.community_tools.raster_regression import (
    IsotonicCalibrator,
    band_count_report,
    blend_submission_zips,
    calibrate_submission_zip,
    return_error_message,
    validate_submission_zip,
)


def _tif_bytes(arr: np.ndarray) -> bytes:
    """Encode a (bands, H, W) float32 array as an in-memory GeoTIFF."""
    arr = arr.astype("float32")
    with MemoryFile() as mem:
        with mem.open(
            driver="GTiff", count=arr.shape[0], height=arr.shape[1],
            width=arr.shape[2], dtype="float32",
        ) as dst:
            dst.write(arr)
        return mem.read()


def _make_zip(path: str, tiles: dict) -> str:
    """Write {name: (bands,H,W) array} into a zip of GeoTIFFs."""
    with zipfile.ZipFile(path, "w") as zf:
        for name, arr in tiles.items():
            zf.writestr(name, _tif_bytes(np.asarray(arr)))
    return path


def _read_zip_tile(path: str, name: str) -> np.ndarray:
    with zipfile.ZipFile(path) as zf:
        with MemoryFile(BytesIO(zf.read(name))) as mem:
            with mem.open() as src:
                return src.read()


class RasterRegressionBase(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        rng = np.random.RandomState(0)
        self.tiles = {
            f"tile_{i:02d}.tif": rng.rand(1, 8, 8) * 10 for i in range(4)
        }
        self.pred_zip = _make_zip(os.path.join(self.dir, "pred.zip"), self.tiles)
        self.ref_zip = _make_zip(
            os.path.join(self.dir, "ref.zip"),
            {n: np.zeros((1, 8, 8)) for n in self.tiles},
        )

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)


class TestValidation(RasterRegressionBase):
    def test_valid_submission(self):
        code, report = validate_submission_zip(self.pred_zip, self.ref_zip)
        self.assertEqual(code, 0)
        self.assertEqual(report["stats"]["n_tiles"], 4)
        self.assertEqual(return_error_message(code), "Valid")

    def test_bad_zip(self):
        bad = os.path.join(self.dir, "not_a_zip.zip")
        with open(bad, "wb") as f:
            f.write(b"hello")
        code, _ = validate_submission_zip(bad)
        self.assertEqual(code, 30)

    def test_empty_zip(self):
        empty = os.path.join(self.dir, "empty.zip")
        with zipfile.ZipFile(empty, "w") as zf:
            zf.writestr("readme.txt", "no rasters here")
        code, _ = validate_submission_zip(empty)
        self.assertEqual(code, 31)

    def test_count_mismatch(self):
        subset = {k: v for k, v in list(self.tiles.items())[:2]}
        small = _make_zip(os.path.join(self.dir, "small.zip"), subset)
        code, report = validate_submission_zip(small, self.ref_zip)
        self.assertEqual(code, 32)
        self.assertEqual(report["expected_n_tiles"], 4)

    def test_name_mismatch(self):
        renamed = dict(self.tiles)
        renamed["wrong_name.tif"] = renamed.pop("tile_00.tif")
        bad = _make_zip(os.path.join(self.dir, "renamed.zip"), renamed)
        code, report = validate_submission_zip(bad, self.ref_zip)
        self.assertEqual(code, 33)
        self.assertIn("tile_00.tif", report["missing_files"])

    def test_folder_prefix_is_accepted(self):
        nested = {f"submission/{n}": a for n, a in self.tiles.items()}
        z = _make_zip(os.path.join(self.dir, "nested.zip"), nested)
        code, _ = validate_submission_zip(z, self.ref_zip)
        self.assertEqual(code, 0)

    def test_shape_mismatch(self):
        bad_tiles = dict(self.tiles)
        bad_tiles["tile_00.tif"] = np.zeros((1, 4, 4))
        z = _make_zip(os.path.join(self.dir, "badshape.zip"), bad_tiles)
        code, report = validate_submission_zip(z, expected_shape=(8, 8))
        self.assertEqual(code, 35)
        self.assertEqual(report["bad_shape"][0][0], "tile_00.tif")

    def test_nan_detected(self):
        bad_tiles = dict(self.tiles)
        arr = np.ones((1, 8, 8))
        arr[0, 3, 3] = np.nan
        bad_tiles["tile_01.tif"] = arr
        z = _make_zip(os.path.join(self.dir, "nan.zip"), bad_tiles)
        code, report = validate_submission_zip(z)
        self.assertEqual(code, 36)
        self.assertIn("tile_01.tif", report["non_finite"])

    def test_negative_detected(self):
        bad_tiles = dict(self.tiles)
        bad_tiles["tile_02.tif"] = np.full((1, 8, 8), -1.0)
        z = _make_zip(os.path.join(self.dir, "neg.zip"), bad_tiles)
        code, report = validate_submission_zip(z)
        self.assertEqual(code, 37)
        self.assertEqual(report["out_of_range"][0][0], "tile_02.tif")

    def test_all_zero_warning(self):
        code, report = validate_submission_zip(self.ref_zip)
        self.assertEqual(code, 0)
        self.assertEqual(report["stats"]["all_zero_tiles"], 4)
        self.assertTrue(report["warnings"])

    def test_band_count_report(self):
        d = os.path.join(self.dir, "dataset")
        os.makedirs(d)
        full = np.zeros((16, 8, 8), dtype="float32")
        short = np.zeros((12, 8, 8), dtype="float32")
        for name, arr in (("full.tif", full), ("short.tif", short)):
            with rasterio.open(
                os.path.join(d, name), "w", driver="GTiff", count=arr.shape[0],
                height=8, width=8, dtype="float32",
            ) as dst:
                dst.write(arr)
        deviant = band_count_report(d, expected_bands=16)
        self.assertEqual(len(deviant), 1)
        self.assertTrue(deviant[0][0].endswith("short.tif"))
        self.assertEqual(deviant[0][1], 12)


class TestEnsemble(RasterRegressionBase):
    def test_equal_blend_is_mean(self):
        tiles_b = {n: a + 2.0 for n, a in self.tiles.items()}
        zb = _make_zip(os.path.join(self.dir, "b.zip"), tiles_b)
        out = blend_submission_zips(
            [self.pred_zip, zb], os.path.join(self.dir, "blend.zip"), progress=False,
        )
        got = _read_zip_tile(out, "tile_00.tif")
        want = (self.tiles["tile_00.tif"] + tiles_b["tile_00.tif"]) / 2.0
        np.testing.assert_allclose(got, want.astype("float32"), rtol=1e-6)

    def test_weighted_blend(self):
        tiles_b = {n: np.zeros_like(a) for n, a in self.tiles.items()}
        zb = _make_zip(os.path.join(self.dir, "b.zip"), tiles_b)
        out = blend_submission_zips(
            [self.pred_zip, zb], os.path.join(self.dir, "wblend.zip"),
            weights=[3.0, 1.0], progress=False,
        )
        got = _read_zip_tile(out, "tile_00.tif")
        want = 0.75 * self.tiles["tile_00.tif"]
        np.testing.assert_allclose(got, want.astype("float32"), rtol=1e-6)

    def test_clip_floor(self):
        neg = {n: a - 100.0 for n, a in self.tiles.items()}
        za = _make_zip(os.path.join(self.dir, "na.zip"), neg)
        zb = _make_zip(os.path.join(self.dir, "nb.zip"), neg)
        out = blend_submission_zips(
            [za, zb], os.path.join(self.dir, "clip.zip"), progress=False,
        )
        self.assertGreaterEqual(_read_zip_tile(out, "tile_00.tif").min(), 0.0)

    def test_mismatched_tiles_raise(self):
        other = _make_zip(
            os.path.join(self.dir, "other.zip"), {"different.tif": np.zeros((1, 8, 8))},
        )
        with self.assertRaises(ValueError):
            blend_submission_zips(
                [self.pred_zip, other], os.path.join(self.dir, "x.zip"), progress=False,
            )

    def test_single_zip_raises(self):
        with self.assertRaises(ValueError):
            blend_submission_zips([self.pred_zip], os.path.join(self.dir, "x.zip"))


class TestCalibration(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def test_recovers_monotone_distortion(self):
        # Model predicts sqrt of the truth (tail under-prediction, as with MSE on
        # skewed targets); isotonic calibration must approximately invert it.
        rng = np.random.RandomState(1)
        true = rng.gamma(shape=1.0, scale=3.0, size=200_000)
        pred = np.sqrt(true)
        cal = IsotonicCalibrator(n_bins=64).fit(pred, true)
        test_true = rng.gamma(shape=1.0, scale=3.0, size=50_000)
        test_pred = np.sqrt(test_true)
        rmse_before = np.sqrt(np.mean((test_pred - test_true) ** 2))
        rmse_after = np.sqrt(np.mean((cal.transform(test_pred) - test_true) ** 2))
        self.assertLess(rmse_after, rmse_before * 0.35)

    def test_mapping_is_monotone(self):
        rng = np.random.RandomState(2)
        pred = rng.rand(10_000)
        true = -pred + rng.rand(10_000) * 5  # anti-correlated noise stresses PAVA
        cal = IsotonicCalibrator(n_bins=32).fit(pred, true)
        self.assertTrue(np.all(np.diff(cal.y_) >= -1e-12))

    def test_save_load_roundtrip(self):
        rng = np.random.RandomState(3)
        pred = rng.rand(5_000) * 4
        cal = IsotonicCalibrator(n_bins=16).fit(pred, pred * 2)
        path = os.path.join(self.dir, "cal.json")
        cal.save(path)
        loaded = IsotonicCalibrator.load(path)
        x = rng.rand(100) * 4
        np.testing.assert_allclose(cal.transform(x), loaded.transform(x))
        with open(path) as f:
            self.assertEqual(json.load(f)["n_bins"], 16)

    def test_constant_predictions(self):
        pred = np.full(1000, 2.0)
        cal = IsotonicCalibrator(n_bins=8).fit(pred, np.full(1000, 5.0))
        np.testing.assert_allclose(cal.transform(np.array([2.0])), [5.0])

    def test_calibrate_zip_grouped(self):
        tiles = {
            "sat_a_tile0.tif": np.full((1, 4, 4), 1.0),
            "sat_b_tile0.tif": np.full((1, 4, 4), 1.0),
        }
        zin = _make_zip(os.path.join(self.dir, "in.zip"), tiles)
        rng = np.random.RandomState(4)
        pred = rng.rand(2_000) * 2
        cal_double = IsotonicCalibrator(n_bins=16).fit(pred, pred * 2)
        out = calibrate_submission_zip(
            zin, os.path.join(self.dir, "out.zip"),
            calibrator={"sat_a": cal_double},
            group_fn=lambda name: name.split("_tile")[0],
            progress=False,
        )
        a = _read_zip_tile(out, "sat_a_tile0.tif")
        b = _read_zip_tile(out, "sat_b_tile0.tif")
        np.testing.assert_allclose(a, np.full((1, 4, 4), 2.0), rtol=1e-2)
        np.testing.assert_allclose(b, np.full((1, 4, 4), 1.0))  # no calibrator: unchanged


if __name__ == "__main__":
    unittest.main()
