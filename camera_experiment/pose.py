"""Estimate camera pose in screen-width units using EXIF focal length."""
from __future__ import annotations

from math import hypot
from pathlib import Path

import cv2
import numpy as np
from PIL import ExifTags, Image
from typed_errs import Result

from camera_experiment.geometry import Failure, Matrix, attempt


def estimate_pose(path: Path, photo_size: tuple[int, int], aspect: float,
                  quad: Matrix) -> Result[dict[str, object], Failure]:
    """Assume centered principal point, zero distortion, and the supplied screen aspect."""
    return attempt("EXIF pose estimate", lambda: _estimate_pose(path, photo_size, aspect, quad))


def _estimate_pose(path: Path, photo_size: tuple[int, int], aspect: float,
                   quad: Matrix) -> dict[str, object]:
    with Image.open(path) as image:
        metadata = image.getexif().get_ifd(ExifTags.IFD.Exif)
        equivalent_mm = float(metadata[41989])
    if equivalent_mm <= 0:
        raise ValueError("EXIF has no usable 35mm-equivalent focal length")
    width, height = photo_size
    focal = equivalent_mm * hypot(width, height) / hypot(36, 24)
    intrinsics = np.array([[focal, 0, width / 2], [0, focal, height / 2], [0, 0, 1]], np.float64)
    world = np.array([[0, 0, 0], [1, 0, 0], [1, 1 / aspect, 0], [0, 1 / aspect, 0]], np.float64)
    ok, rotation, translation = cv2.solvePnP(world, quad, intrinsics, np.zeros(5), flags=cv2.SOLVEPNP_IPPE)
    if not ok or translation[2, 0] <= 0:
        raise ValueError("no forward-facing planar pose")
    matrix, _ = cv2.Rodrigues(rotation)
    projected, _ = cv2.projectPoints(world, rotation, translation, intrinsics, np.zeros(5))
    error = float(np.sqrt(np.mean(np.square(projected.reshape(4, 2) - quad).sum(1))))
    return {"method": "EXIF 35mm focal length + planar solvePnP",
            "calibrated": False, "units": "normalization frame width",
            "screen_axes": "origin top-left, x right, y down, z completes right-handed frame",
            "assumptions": ["centered principal point", "zero lens distortion", "output aspect equals physical screen aspect"],
            "focal_35mm": equivalent_mm, "intrinsics": intrinsics.tolist(),
            "rotation_screen_to_camera": matrix.tolist(),
            "translation_screen_to_camera": translation.reshape(3).tolist(),
            "camera_center_in_screen": (-matrix.T @ translation).reshape(3).tolist(),
            "reprojection_rmse_px": error}
