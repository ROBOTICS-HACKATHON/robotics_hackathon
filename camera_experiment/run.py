"""Compare camera normalization and the existing CAPTCHA solver on saved photos."""
from __future__ import annotations

import argparse
import asyncio
import io
import json
import os
import sys
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TypeVar, cast

import cv2
import numpy as np
from captchasolver.client import VLM_MODEL, WorkersAI
from captchasolver.grid import solve_grid, tile_centers
from captchasolver.screen import (
    Layout,
    annotate,
    build_actions,
    find_layout,
    read_target,
)
from PIL import Image
from typed_errs import Err, Ok, Result

from camera_experiment.geometry import (
    Failure,
    Frame,
    Screen,
    attempt,
    locate,
    map_points,
    rectify,
)
from camera_experiment.pose import estimate_pose

T = TypeVar("T")
Record = dict[str, object]
VARIANTS = ("original", "blur", "jpeg", "lowres", "noise", "dark", "combined")


@dataclass(frozen=True)
class Options:
    images: Path
    out: Path
    width: int
    height: int
    variants: list[str]
    seed: int
    solve: bool
    model: str


def degrade(frame: Frame, variant: str, seed: int) -> Result[Frame, Failure]:
    """Apply reproducible damage before detection, including the screen boundary."""
    return attempt("degrade photo", lambda: _degrade(frame, variant, seed))


def _degrade(frame: Frame, variant: str, seed: int) -> Frame:
    image = frame.copy()
    height, width = image.shape[:2]
    if variant in ("lowres", "combined"):
        image = cv2.resize(image, (640, round(height * 640 / width)), interpolation=cv2.INTER_AREA)
    if variant in ("blur", "combined"):
        sigma = 4 if variant == "blur" else 1.4
        image = cv2.GaussianBlur(image, (0, 0), sigma)
    if variant in ("noise", "combined"):
        noise = np.random.default_rng(seed).normal(0, 14, image.shape)
        image = np.clip(image.astype(float) + noise, 0, 255).astype(np.uint8)
    if variant in ("dark", "combined"):
        image = np.clip(image.astype(float) * .48, 0, 255).astype(np.uint8)
    if variant in ("jpeg", "combined"):
        ok, data = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 12])
        if not ok:
            raise ValueError("JPEG encoding failed")
        image = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("JPEG decoding failed")
    return cast(Frame, image)


def photo_layout(frame: Frame) -> Result[Layout, Failure]:
    """Locate saturated blue blocks despite camera exposure and white balance."""
    result = attempt("photo dialog", lambda: _photo_layout(frame))
    if isinstance(result, Err):
        return result
    if not result.value:
        return Err(Failure.OPERATION, context_msg="no blue CAPTCHA banner and Verify pair")
    return Ok(result.value[0])


def _photo_layout(frame: Frame) -> list[Layout]:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    brightness = max(20, min(65, round(float(np.percentile(hsv[:, :, 2], 80)) * .3)))
    mask = cv2.inRange(hsv, (90, 120, brightness), (135, 255, 255))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = [cv2.boundingRect(contour) for contour in contours
             if cv2.contourArea(contour) > frame.shape[0] * frame.shape[1] * .0003]
    layouts: list[Layout] = []
    for x, y, w, h in sorted(boxes, key=lambda box: box[2] * box[3], reverse=True):
        if not 2.5 < w / h < 4.5 or w < frame.shape[1] * .08:
            continue
        # The demo has a white gap of about 2% of the banner width above its grid.
        gy = y + h + round(w * .02)
        if gy + w >= frame.shape[0]:
            continue
        buttons = [(bx, by, bw, bh) for bx, by, bw, bh in boxes
                   if .15 * w < bw < .4 * w and 1.5 < bw / bh < 3.5
                   and x + .55 * w < bx < x + w
                   and gy + .95 * w < by < gy + 1.3 * w]
        if buttons:
            bx, by, bw, bh = min(buttons, key=lambda box: box[1])
            layouts.append(Layout((x, y, w, h), (x, gy, w, w), (bx + bw // 2, by + bh // 2)))
    return layouts


def save_image(path: Path, frame: Frame) -> Result[bool, Failure]:
    result = attempt(f"write {path}", lambda: cv2.imwrite(str(path), frame))
    if isinstance(result, Ok) and not result.value:
        return Err(Failure.OPERATION, context_msg=f"could not write {path}")
    return result


def save_json(path: Path, record: object) -> Result[int, Failure]:
    return attempt(f"write {path}", lambda: path.write_text(json.dumps(record, indent=2) + "\n"))


def png(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, "PNG")
    return buffer.getvalue()


async def async_attempt(label: str, operation: Callable[[], Awaitable[T]]) -> Result[T, Failure]:
    try:
        return Ok(await operation())
    except (KeyboardInterrupt, asyncio.CancelledError):
        return Err(Failure.INTERRUPTED, context_msg=f"{label}: interrupted")
    except Exception as error:  # noqa: BLE001 - async third-party operation boundary
        return Err(Failure.OPERATION, context_msg=f"{label}: {type(error).__name__}: {error}")


async def solve(screen: Screen, layout: Layout, model: str) -> Result[Record, Failure]:
    return await async_attempt("Workers AI solve", lambda: _solve(screen, layout, model))


async def _solve(screen: Screen, layout: Layout, model: str) -> Record:
    async with WorkersAI(concurrency=3) as ai:
        image = Image.fromarray(cv2.cvtColor(screen.image, cv2.COLOR_BGR2RGB))
        target = await read_target(ai, image, layout, model=model)
        x, y, w, h = layout.grid
        tiles = await solve_grid(ai, png(image.crop((x, y, x + w, y + h))), target,
                                 mode="ask", model=model)
    width, height = image.size
    centers = tile_centers((w, h), 3, 3)
    clicks = [[x + centers[t][0], y + centers[t][1]] for t in tiles]
    verify = list(layout.verify) if layout.verify else None
    result: Record = {"target": target, "image_size": [width, height], "grid_box": list(layout.grid),
              "tiles": tiles, "click": clicks, "verify": verify,
              "click_norm": [[round(cx / width, 6), round(cy / height, 6)] for cx, cy in clicks],
              "verify_norm": [round(verify[0] / width, 6), round(verify[1] / height, 6)]
              if verify else None}
    result["actions"] = build_actions(result)
    return cast(Record, result)


async def process(path: Path, variant: str, args: Options) -> Result[Record, Failure]:
    folder = args.out / path.stem / variant
    cleared = attempt("invalidate previous plan", lambda: [
        (folder / name).unlink(missing_ok=True)
        for name in ("actions.json", "plan.png", "challenge.png", "geometry.json")])
    if isinstance(cleared, Err):
        return cleared
    loaded = attempt("read photo", lambda: cv2.imread(str(path)))
    if isinstance(loaded, Err):
        return loaded
    if loaded.value is None:
        return Err(Failure.OPERATION, context_msg=f"unreadable photo: {path}")
    damaged = degrade(cast(Frame, loaded.value), variant, args.seed)
    if isinstance(damaged, Err):
        return damaged
    frame = damaged.value
    quad = locate(frame)
    if isinstance(quad, Err):
        return quad
    normalized = rectify(frame, quad.value, (args.width, args.height))
    if isinstance(normalized, Err):
        return normalized
    screen = normalized.value
    created = attempt("create output directory", lambda: folder.mkdir(parents=True, exist_ok=True))
    if isinstance(created, Err):
        return created
    record: Record = {"source": str(path), "variant": variant, "photo_size": [frame.shape[1], frame.shape[0]],
                      "screen_size": [args.width, args.height], "screen_quad": screen.quad.tolist(),
                      "photo_to_screen": screen.photo_to_screen.tolist(),
                      "screen_to_photo": screen.screen_to_photo.tolist(), "metric_pose": "uncalibrated"}
    pose = estimate_pose(path, (frame.shape[1], frame.shape[0]), args.width / args.height, screen.quad)
    record["pose_estimate"] = pose.value if isinstance(pose, Ok) else pose.context_msg
    for name, image in (("photo.jpg", frame), ("screen.png", screen.image)):
        saved = save_image(folder / name, image)
        if isinstance(saved, Err):
            return saved
    pil = attempt("screen RGB", lambda: Image.fromarray(cv2.cvtColor(screen.image, cv2.COLOR_BGR2RGB)))
    if isinstance(pil, Err):
        return pil
    baseline = attempt("original screenshot locator", lambda: asdict(find_layout(pil.value)))
    record["screenshot_locator"] = baseline.value if isinstance(baseline, Ok) else baseline.context_msg
    layout = photo_layout(screen.image)
    if isinstance(layout, Err):
        record["status"] = "no_dialog"
        record["reason"] = layout.context_msg
    else:
        found = layout.value
        record["layout"] = asdict(found)
        x, y, w, _ = found.header
        bottom = min(args.height, found.grid[1] + found.grid[3] + round(w * .2))
        record["challenge_box"] = [x, y, w, bottom - y]
        crop = save_image(folder / "challenge.png", screen.image[y:bottom, x:x + w])
        if isinstance(crop, Err):
            return crop
        solved = await solve(screen, found, args.model) if args.solve else Ok({"status": "not_requested"})
        if isinstance(solved, Err):
            if solved.error == Failure.INTERRUPTED:
                return solved
            record["status"] = "solver_error"
            record["reason"] = solved.context_msg
        elif args.solve:
            plan = solved.value
            actions = cast(list[Record], plan["actions"])
            clicks = np.array([[action["x"], action["y"]] for action in actions], dtype=np.float64)
            projected = map_points(clicks, screen.screen_to_photo)
            if isinstance(projected, Err):
                return projected
            for action, point in zip(actions, projected.value):
                action["photo_x"], action["photo_y"] = point.tolist()
            plan["coordinate_frame"] = "full normalized screen, never the challenge crop"
            record["solver"] = plan
            record["status"] = "solved"
            overlay = attempt("annotate plan", lambda: cv2.cvtColor(
                np.asarray(annotate(pil.value, found, plan)), cv2.COLOR_RGB2BGR))
            if isinstance(overlay, Err):
                return overlay
            saved = save_image(folder / "plan.png", cast(Frame, overlay.value))
            if isinstance(saved, Err):
                return saved
            saved_plan = save_json(folder / "actions.json", plan)
            if isinstance(saved_plan, Err):
                return saved_plan
        else:
            record["status"] = "layout_only"
    saved_record = save_json(folder / "geometry.json", record)
    return Ok(record) if isinstance(saved_record, Ok) else saved_record


async def run(args: Options) -> Result[int, Failure]:
    paths = attempt("list photos", lambda: sorted(args.images.glob("*.jpg")))
    if isinstance(paths, Err):
        return paths
    if not paths.value:
        return Err(Failure.OPERATION, context_msg="no JPG photos in input directory")
    records: list[Record] = []
    for path in paths.value:
        for variant in args.variants:
            result = await process(path, variant, args)
            if isinstance(result, Err):
                if result.error == Failure.INTERRUPTED:
                    return result
                record: Record = {"source": str(path), "variant": variant, "status": "geometry_error",
                                  "reason": result.context_msg}
            else:
                record = result.value
            records.append(record)
            print(path.name, variant, record["status"], flush=True)
            saved = save_json(args.out / "report.json", {"seed": args.seed, "model": args.model,
                                                        "records": records})
            if isinstance(saved, Err):
                return saved
    return Ok(0)


def main() -> Result[int, Failure]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("images", type=Path)
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / "results")
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1200)
    parser.add_argument("--variants", choices=VARIANTS, nargs="+", default=list(VARIANTS))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--solve", action="store_true")
    parser.add_argument("--model", default=VLM_MODEL)
    parsed = parser.parse_args()
    args = Options(cast(Path, parsed.images), cast(Path, parsed.out), cast(int, parsed.width),
                   cast(int, parsed.height), cast(list[str], parsed.variants), cast(int, parsed.seed),
                   cast(bool, parsed.solve), cast(str, parsed.model))
    if args.width < 100 or args.height < 100:
        return Err(Failure.OPERATION, context_msg="screen width and height must be at least 100")
    if args.solve and not all(os.environ.get(key) for key in ("CLOUDFLARE_ACCOUNT_ID", "CLOUDFLARE_API_TOKEN")):
        return Err(Failure.OPERATION, context_msg="load CLOUDFLARE_ACCOUNT_ID and CLOUDFLARE_API_TOKEN before --solve")
    created = attempt("create output directory", lambda: args.out.mkdir(parents=True, exist_ok=True))
    if isinstance(created, Err):
        return created
    outcome = attempt("run comparison", lambda: asyncio.run(run(args)))
    return outcome.value if isinstance(outcome, Ok) else outcome


if __name__ == "__main__":
    try:
        outcome = main()
        if isinstance(outcome, Err):
            print(outcome.context_msg, file=sys.stderr)
            sys.exit(130 if outcome.error == Failure.INTERRUPTED else 1)
        sys.exit(outcome.value)
    except (Exception, KeyboardInterrupt) as error:  # noqa: BLE001 - last-resort process boundary
        print(f"unexpected process boundary failure: {type(error).__name__}: {error}", file=sys.stderr)
        sys.exit(1)
