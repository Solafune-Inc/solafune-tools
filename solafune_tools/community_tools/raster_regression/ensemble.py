"""Zip-level ensembling for raster regression submissions.

Averaging the predictions of independently trained models is one of the most reliable
score improvements available in regression competitions, and doing it directly on
submission archives means it works across teams' pipelines, frameworks, and even
between your own past submissions — no re-inference needed. A plain 50/50 average of
two diverse submissions typically beats both inputs.
"""

import os
import zipfile
from io import BytesIO
from typing import List, Optional, Sequence

import numpy as np
from tqdm import tqdm

from rasterio.io import MemoryFile

from solafune_tools.community_tools.raster_regression.validation import _tif_names


def blend_submission_zips(
    zip_paths: Sequence[str],
    out_path: str,
    weights: Optional[Sequence[float]] = None,
    clip_min: Optional[float] = 0.0,
    clip_max: Optional[float] = None,
    progress: bool = True,
) -> str:
    """Blend N submission archives into one by (weighted) pixel-wise averaging.

    Tiles are matched across archives by base file name; every archive must contain
    the same tile set. Geo metadata (CRS/transform/dtype) is copied from the first
    archive. Tiles are processed one at a time, so memory stays flat regardless of
    archive size.

    Args:
        zip_paths: Two or more submission archives to blend.
        out_path: Path of the blended archive to write (created/overwritten).
        weights: Optional per-archive weights; normalized to sum to 1. Default equal.
        clip_min: Clip blended values below this (default 0.0). ``None`` disables.
        clip_max: Clip blended values above this. ``None`` disables.
        progress: Show a tqdm progress bar.

    Returns:
        ``out_path``.

    Raises:
        ValueError: On fewer than two archives, weight/archive count mismatch, or
            tile-set mismatch between archives.
    """
    if len(zip_paths) < 2:
        raise ValueError("Need at least two archives to blend.")
    if weights is None:
        w = np.full(len(zip_paths), 1.0 / len(zip_paths))
    else:
        if len(weights) != len(zip_paths):
            raise ValueError("weights and zip_paths must have the same length.")
        w = np.asarray(weights, dtype="float64")
        if w.sum() <= 0:
            raise ValueError("weights must sum to a positive value.")
        w = w / w.sum()

    archives = [zipfile.ZipFile(p) for p in zip_paths]
    try:
        member_maps: List[dict] = []
        for p, zf in zip(zip_paths, archives):
            m = {os.path.basename(n): n for n in _tif_names(zf)}
            if not m:
                raise ValueError(f"No .tif files found in {p}")
            member_maps.append(m)
        base_names = sorted(member_maps[0])
        for p, m in zip(zip_paths[1:], member_maps[1:]):
            if sorted(m) != base_names:
                raise ValueError(f"Tile set in {p} does not match {zip_paths[0]}")

        with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as out:
            iterator = tqdm(base_names, desc="blend", disable=not progress)
            for name in iterator:
                blended = None
                profile = None
                for wi, zf, m in zip(w, archives, member_maps):
                    with MemoryFile(BytesIO(zf.read(m[name]))) as mem:
                        with mem.open() as src:
                            arr = src.read().astype("float64")
                            if profile is None:
                                profile = src.profile.copy()
                    blended = wi * arr if blended is None else blended + wi * arr
                if clip_min is not None or clip_max is not None:
                    blended = np.clip(blended, clip_min, clip_max)
                profile.update(count=blended.shape[0])
                dtype = profile.get("dtype", "float32")
                with MemoryFile() as mem:
                    with mem.open(**profile) as dst:
                        dst.write(blended.astype(dtype))
                    out.writestr(name, mem.read())
    finally:
        for zf in archives:
            zf.close()
    return out_path
