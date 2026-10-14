"""Locate a bright laptop display and normalize its perspective from pixels."""
from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from enum import Enum, auto
from typing import Literal, TypeVar, cast

import cv2
import numpy as np
from numpy.typing import NDArray
from typed_errs import Err, Nothing, Ok, Option, Result, Some

from touch.see import DICT, find_screen

Frame = NDArray[np.uint8]
Matrix = NDArray[np.float64]
T = TypeVar("T")


class Failure(Enum):
    OPERATION = auto()
    SCREEN_NOT_FOUND = auto()
    INTERRUPTED = auto()


def attempt(label: str, operation: Callable[[], T]) -> Result[T, Failure]:
    """Keep exceptions at the external operation boundary."""
    try:
        return Ok(operation())
    except KeyboardInterrupt:
        return Err(Failure.INTERRUPTED, context_msg=f"{label}: interrupted")
    except Exception as error:  # noqa: BLE001 - third-party operation boundary
        return Err(Failure.OPERATION, context_msg=f"{label}: {type(error).__name__}: {error}")


async def async_attempt(label: str, operation: Callable[[], Awaitable[T]]) -> Result[T, Failure]:
    """Keep asynchronous service failures at their external boundary."""
    try:
        return Ok(await operation())
    except (KeyboardInterrupt, asyncio.CancelledError):
        return Err(Failure.INTERRUPTED, context_msg=f"{label}: interrupted")
    except Exception as error:  # noqa: BLE001 - async third-party operation boundary
        return Err(Failure.OPERATION, context_msg=f"{label}: {type(error).__name__}: {error}")


@dataclass
class Screen:
    image: Frame
    quad: Matrix
    photo_to_screen: Matrix
    screen_to_photo: Matrix
    frame_kind: Literal["display", "aruco-outer"] = "display"


@dataclass(frozen=True)
class LocatedScreen:
    quad: Matrix
    kind: Literal["display", "aruco-outer"]


def identify(frame: Frame, mode: Literal["auto", "markers", "display"] = "auto") -> Result[LocatedScreen, Failure]:
    """Prefer the exact corner-marker frame used by the arm's click-test page."""
    if mode != "display":
        detected = attempt("detect corner markers", lambda: _markers(frame))
        if isinstance(detected, Err):
            return detected
        quad, ids = detected.value
        if isinstance(quad, Some):
            return Ok(LocatedScreen(quad.value, "aruco-outer"))
        if mode == "markers":
            return Err(Failure.SCREEN_NOT_FOUND,
                       context_msg=f"need markers 0,1,2,3; saw {ids}. Enable Camera markers and include all four corners")
    found = locate(frame)
    return Ok(LocatedScreen(found.value, "display")) if isinstance(found, Ok) else found


def _markers(frame: Frame) -> tuple[Option[Matrix], list[int]]:
    quad, ids = find_screen(frame)
    return (Some(quad.astype(np.float64)) if quad is not None else Nothing(), ids)


def marker_aspect(frame: Frame, quad: Matrix) -> Result[float, Failure]:
    """Recover the frame aspect from the known square corner markers."""
    return attempt("measure marker frame aspect", lambda: _marker_aspect(frame, quad))


def _marker_aspect(frame: Frame, quad: Matrix) -> float:
    corners, ids, _ = cv2.aruco.ArucoDetector(DICT, cv2.aruco.DetectorParameters()).detectMarkers(
        cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    if ids is None:
        raise ValueError("markers disappeared during aspect measurement")
    unit = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], np.float32)
    transform = cv2.getPerspectiveTransform(quad.astype(np.float32), unit)
    ratios: list[float] = []
    for marker_id, corner in zip(ids.flatten(), corners):
        if int(marker_id) not in (0, 1, 2, 3):
            continue
        points = cv2.perspectiveTransform(corner.astype(np.float64), transform).reshape(4, 2)
        horizontal = (np.linalg.norm(points[1] - points[0]) + np.linalg.norm(points[2] - points[3])) / 2
        vertical = (np.linalg.norm(points[3] - points[0]) + np.linalg.norm(points[2] - points[1])) / 2
        ratios.append(float(vertical / horizontal))
    aspect = float(np.median(ratios))
    if len(ratios) != 4 or not np.isfinite(aspect) or aspect <= 0:
        raise ValueError("cannot recover frame aspect from four square markers")
    return aspect


def locate(frame: Frame) -> Result[Matrix, Failure]:
    """Return corners in TL, TR, BR, BL order; tuned for these bright displays."""
    return attempt("locate display", lambda: _locate(frame)).and_then(
        lambda candidates: Ok(candidates[0]) if candidates else
        Err(Failure.SCREEN_NOT_FOUND, context_msg="no bright convex display found")
    )


def _locate(frame: Frame, smooth: bool = False) -> list[Matrix]:
    scale = min(1.0, 1200 / frame.shape[1])
    small = cv2.resize(frame, (round(frame.shape[1] * scale), round(frame.shape[0] * scale)))
    if smooth:
        small = cv2.GaussianBlur(small, (5, 5), 0)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    brightness = max(30, min(100, round(float(np.percentile(hsv[:, :, 2], 80)) * .55)))
    mask = cv2.inRange(hsv, (0, 0, brightness), (179, 150, 255))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates: list[Matrix] = []
    height, width = small.shape[:2]
    for contour in contours:
        polygon = cv2.approxPolyDP(contour, (.03 if smooth else .02) * cv2.arcLength(contour, True), True)
        if len(polygon) != 4 or not cv2.isContourConvex(polygon):
            continue
        if cv2.contourArea(polygon) < width * height * .08:
            continue
        points = polygon.reshape(4, 2).astype(np.float64)
        # Screen candidates must be enclosed, unlike the table at the photo edge.
        if np.any(points[:, 0] < 2) or np.any(points[:, 1] < 2):
            continue
        if np.any(points[:, 0] > width - 3) or np.any(points[:, 1] > height - 3):
            continue
        ordered = np.array([points[np.argmin(points.sum(1))],
                            points[np.argmin(np.diff(points, axis=1))],
                            points[np.argmax(points.sum(1))],
                            points[np.argmax(np.diff(points, axis=1))]])
        horizontal = np.linalg.norm(ordered[1] - ordered[0])
        vertical = np.linalg.norm(ordered[3] - ordered[0])
        if 1.2 < horizontal / vertical < 2.3:
            candidates.append(ordered / scale)
    if not candidates and not smooth:
        return _locate(frame, smooth=True)
    return sorted(candidates, key=lambda q: abs(float(cv2.contourArea(q.astype(np.float32)))),
                  reverse=True)


def rectify(frame: Frame, quad: Matrix, size: tuple[int, int],
            kind: Literal["display", "aruco-outer"] = "display") -> Result[Screen, Failure]:
    """Use screen-edge coordinates: normalized u=x/width, v=y/height."""
    return attempt("rectify screen frame", lambda: _rectify(frame, quad, size, kind))


def _rectify(frame: Frame, quad: Matrix, size: tuple[int, int],
             kind: Literal["display", "aruco-outer"]) -> Screen:
    width, height = size
    destination = np.array([[0, 0], [width, 0], [width, height], [0, height]], np.float32)
    transform = cast(Matrix, cv2.getPerspectiveTransform(quad.astype(np.float32), destination))
    return Screen(cast(Frame, cv2.warpPerspective(frame, transform, size)), quad, transform,
                  np.linalg.inv(transform), kind)


def map_points(points: Matrix, transform: Matrix) -> Result[Matrix, Failure]:
    """Project screen clicks back into photo pixels with the saved homography."""
    return attempt("project points", lambda: cast(Matrix, cv2.perspectiveTransform(
        points.reshape(-1, 1, 2), transform).reshape(-1, 2)))
