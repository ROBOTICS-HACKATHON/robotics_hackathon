"""Opt-in camera parameters and checks shared by the touch commands."""

import argparse
import json
import math
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Generic, TypeVar

import numpy as np
from numpy.typing import NDArray

T = TypeVar("T")


@dataclass(frozen=True)
class Ok(Generic[T]):
    value: T


@dataclass(frozen=True)
class Error:
    message: str


def attempt(operation: Callable[[], T]) -> Ok[T] | Error:
    """Contain exceptions at an I/O or numerical boundary."""
    try:
        return Ok(operation())
    except Exception as error:  # noqa: BLE001 - this boundary converts every operation failure into a result.
        return Error(str(error))


@dataclass(frozen=True)
class CameraParameters:
    image_size: tuple[int, int]
    matrix: NDArray[np.float64]
    distortion: NDArray[np.float64]


def load_parameters(path: Path) -> Ok[dict[str, CameraParameters]] | Error:
    """Read parameters keyed by the camera specification used on the CLI."""
    def read() -> dict[str, CameraParameters]:
        document = json.loads(path.read_text())
        parameters = {}
        for spec, entry in document.items():
            width, height = entry["image_size"]
            fx, fy, cx, cy = (float(entry[key]) for key in ("fx", "fy", "cx", "cy"))
            distortion = np.asarray(entry.get("distortion", [0.0] * 5), dtype=float)
            if (not isinstance(width, int) or not isinstance(height, int) or width <= 0 or height <= 0
                    or not all(math.isfinite(x) for x in (fx, fy, cx, cy)) or fx <= 0 or fy <= 0
                    or not 0 <= cx < width or not 0 <= cy < height
                    or distortion.ndim != 1 or distortion.size not in (4, 5, 8, 12, 14)
                    or not np.isfinite(distortion).all()):
                raise ValueError(f"invalid calibration for {spec}")
            matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=float)
            parameters[spec] = CameraParameters((width, height), matrix, distortion)
        return parameters

    result = attempt(read)
    return Error(f"cannot load camera calibration {path}: {result.message}") if isinstance(result, Error) else result


@dataclass(frozen=True)
class VisionOptions:
    calibration: Path | None = None
    focal_px: tuple[float, ...] = ()
    screen_height_mm: float | None = None
    check_geometry: bool = False
    min_ray_angle_deg: float = 12.0
    max_height_mm: float = 300.0
    paired_capture: bool = False
    max_pair_delay_ms: float = 150.0
    stable_frames: int = 0
    stable_px: float = 3.0
    diagnostics: bool = False
    setup_id: str | None = None
    tip_offset_mm: float | None = None


def add_vision_arguments(parser: argparse.ArgumentParser) -> None:
    """Add independent experiments; no option changes defaults when omitted."""
    group = parser.add_argument_group("opt-in tip vision experiments")
    group.add_argument("--vision-calibration", help="JSON camera intrinsics and distortion, keyed by camera spec")
    group.add_argument("--vision-focal-px", help="override focal length in pixels, one value per camera, e.g. 900,1100")
    group.add_argument("--screen-height-mm", type=float, help="measured marker-frame height; overrides taught aspect ratio")
    group.add_argument("--vision-tip-offset-mm", type=float, help="marker height at contact; overrides taught marker offset")
    group.add_argument("--vision-check-geometry", action="store_true", help="reject weak ray angles, behind-camera and implausible points")
    group.add_argument("--vision-min-ray-angle-deg", type=float, default=12.0)
    group.add_argument("--vision-max-height-mm", type=float, default=300.0)
    group.add_argument("--vision-paired-capture", action="store_true", help="read cameras concurrently and check host receive-time separation")
    group.add_argument("--vision-max-pair-delay-ms", type=float, default=150.0)
    group.add_argument("--vision-stable-frames", type=int, default=0, help="require N stationary observations before using the tip (0 = off)")
    group.add_argument("--vision-stable-px", type=float, default=3.0)
    group.add_argument("--vision-diagnostics", action="store_true", help="show calibration, remembered markers and rejection reasons")
    group.add_argument("--vision-setup-id", help="separate learned corrections by setup; change after moving a phone")


def vision_options(args: argparse.Namespace) -> Ok[VisionOptions] | Error:
    """Validate CLI values without exceptions escaping argument conversion."""
    def parse() -> VisionOptions:
        focal = tuple(float(x) for x in args.vision_focal_px.split(",")) if args.vision_focal_px else ()
        if len(focal) > 2 or any(not math.isfinite(x) or x <= 0 for x in focal):
            raise ValueError("--vision-focal-px needs one or two positive finite values")
        if args.vision_calibration and focal:
            raise ValueError("choose --vision-calibration or --vision-focal-px")
        for name in ("screen_mm", "screen_height_mm", "vision_max_height_mm", "vision_max_pair_delay_ms", "vision_stable_px"):
            value = getattr(args, name)
            if value is not None and (not math.isfinite(value) or value <= 0):
                raise ValueError(f"--{name.replace('_', '-')} must be positive and finite")
        if not math.isfinite(args.vision_min_ray_angle_deg) or not 0 < args.vision_min_ray_angle_deg < 90:
            raise ValueError("--vision-min-ray-angle-deg must be between 0 and 90")
        if args.vision_stable_frames != 0 and not 2 <= args.vision_stable_frames <= 10:
            raise ValueError("--vision-stable-frames must be 0 or between 2 and 10")
        if args.vision_setup_id is not None and not args.vision_setup_id.strip():
            raise ValueError("--vision-setup-id must not be empty")
        if args.vision_tip_offset_mm is not None and not math.isfinite(args.vision_tip_offset_mm):
            raise ValueError("--vision-tip-offset-mm must be finite")
        calibration = Path(args.vision_calibration) if args.vision_calibration else None
        return VisionOptions(calibration=calibration, focal_px=focal, screen_height_mm=args.screen_height_mm,
                             check_geometry=args.vision_check_geometry, min_ray_angle_deg=args.vision_min_ray_angle_deg,
                             max_height_mm=args.vision_max_height_mm, paired_capture=args.vision_paired_capture,
                             max_pair_delay_ms=args.vision_max_pair_delay_ms, stable_frames=args.vision_stable_frames,
                             stable_px=args.vision_stable_px, diagnostics=args.vision_diagnostics,
                             setup_id=args.vision_setup_id, tip_offset_mm=args.vision_tip_offset_mm)

    return attempt(parse)
