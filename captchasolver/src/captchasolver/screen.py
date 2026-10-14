"""Full-page screenshot of the challenge -> target, grid location, click coordinates.

The dialog is located by its blue header (#1a73e8) and Verify button, so no DOM
access is needed: this works on a camera/screen capture too, as long as the
dialog is roughly axis-aligned and unscaled-looking.
"""
from __future__ import annotations

import io
import re
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw

from .client import VLM_MODEL, WorkersAI, vlm_query
from .grid import solve_grid, tile_centers

BLUE = (26, 115, 232)
INSTRUCTION_Q = (
    "This is a captcha instruction banner. Which kind of object must be selected? "
    "Reply with ONLY that object as a plural noun, e.g. cars."
)


class LayoutNotFound(ValueError):
    pass


@dataclass
class Layout:
    header: tuple[int, int, int, int]            # x, y, w, h
    grid: tuple[int, int, int, int]              # x, y, w, h
    verify: tuple[int, int] | None               # click point


def _near(arr: np.ndarray, rgb: tuple[int, int, int], tol: int = 24) -> np.ndarray:
    return np.all(np.abs(arr - np.array(rgb)) < tol, axis=-1)


def find_layout(img: Image.Image) -> Layout:
    """Locate the dialog at any resolution: every threshold is relative, none in pixels.

    The banner is the widest blue block, so its width is the reference ("peak");
    banner rows hold >= 40% of it, the narrower Verify button holds 8-40%.
    """
    a = np.asarray(img.convert("RGB")).astype(int)
    blue = _near(a, BLUE)
    per_row = blue.sum(1)
    peak = int(per_row.max())
    if peak < max(12, 0.02 * a.shape[1]):
        raise LayoutNotFound("blue instruction banner not found in screenshot")

    wide = np.where(per_row >= 0.4 * peak)[0]
    # first contiguous run of wide blue rows = the banner
    breaks = np.where(np.diff(wide) > 1)[0]
    y0, y1 = int(wide[0]), int(wide[breaks[0]] if breaks.size else wide[-1])
    cols = np.where(blue[y0:y1 + 1].sum(0) > (y1 - y0) // 2)[0]
    x0, x1 = int(cols.min()), int(cols.max())
    header = (x0, y0, x1 - x0 + 1, y1 - y0 + 1)

    # grid: starts at the first non-white row under the banner, square, banner-wide
    side = x1 - x0 + 1
    below = a[y1 + 1:, x0:x1 + 1]
    non_white = (np.abs(below - 255).max(-1) > 40).mean(1) > 0.5
    # Skip the banner's anti-aliased edge (blurred by scaling): wait for the white gap
    # under it, then take the first content row after the gap.
    gaps = np.where(~non_white)[0]
    if gaps.size == 0:
        raise LayoutNotFound("grid not found below the banner")
    start = np.where(non_white[gaps[0]:])[0]
    if start.size == 0:
        raise LayoutNotFound("grid not found below the banner")
    gy = y1 + 1 + int(gaps[0] + start[0])
    if side < 30 or gy + side > a.shape[0]:
        raise LayoutNotFound("grid is too small or runs off the bottom (scrolled / cropped?)")
    grid = (x0, gy, side, side)

    verify = None
    vrows = np.where((per_row > 0.08 * peak) & (per_row < 0.4 * peak))[0]
    vrows = vrows[vrows > gy + side]
    if vrows.size:
        vcols = np.where(blue[vrows.min():vrows.max() + 1].sum(0) > 0)[0]
        verify = (int((vcols.min() + vcols.max()) // 2), int((vrows.min() + vrows.max()) // 2))
    return Layout(header, grid, verify)


def _png(img: Image.Image, max_side: int | None = None) -> bytes:
    """PNG bytes; optionally shrunk so 4K screens don't mean slow, costly requests."""
    if max_side and max(img.size) > max_side:
        img = img.copy()
        img.thumbnail((max_side, max_side), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


async def read_target(ai: WorkersAI, img: Image.Image, layout: Layout, *,
                      model: str = VLM_MODEL) -> str:
    x, y, w, h = layout.header
    raw = await vlm_query(ai, _png(img.crop((x, y, x + w, y + h)), 640), INSTRUCTION_Q,
                          model=model, max_tokens=12)
    target = re.sub(r"[^a-z ]", "", raw.lower()).strip()
    if not target:
        raise ValueError(f"could not read the target from the banner (model said {raw!r})")
    return target


async def solve_screenshot(ai: WorkersAI, png: bytes, *, target: str | None = None,
                           rows: int = 3, cols: int = 3, model: str = VLM_MODEL,
                           mode: str = "ask") -> tuple[dict, Image.Image, Layout]:
    img = Image.open(io.BytesIO(png)).convert("RGB")
    layout = find_layout(img)
    target = target or await read_target(ai, img, layout, model=model)
    gx, gy, gw, gh = layout.grid
    tiles = await solve_grid(ai, _png(img.crop((gx, gy, gx + gw, gy + gh)), 768), target,
                             rows=rows, cols=cols, mode=mode, model=model)
    centers = tile_centers((gw, gh), rows, cols)
    W, H = img.size
    click = [[gx + centers[i][0], gy + centers[i][1]] for i in tiles]
    result = {
        "target": target,
        "image_size": [W, H],
        "grid_box": list(layout.grid),
        "tiles": tiles,
        "click": click,                       # pixels of THIS screenshot
        "verify": list(layout.verify) if layout.verify else None,
        # resolution-independent (0..1): multiply by the real screen size to click
        "click_norm": [[round(x / W, 5), round(y / H, 5)] for x, y in click],
        "verify_norm": ([round(layout.verify[0] / W, 5), round(layout.verify[1] / H, 5)]
                        if layout.verify else None),
    }
    result["actions"] = build_actions(result)
    return result, img, layout


def build_actions(result: dict) -> list[dict]:
    """The click script a robot (or Playwright) follows: tiles first, Verify last.

    Coordinates come both in pixels of the screenshot and normalised (0..1); executors
    should use the normalised ones so any screen size / DPI scale lands on the same spot.
    """
    acts = [{"action": "click", "role": "tile", "tile": t, "x": x, "y": y,
             "x_norm": nx, "y_norm": ny}
            for t, (x, y), (nx, ny) in zip(result["tiles"], result["click"], result["click_norm"])]
    if result["verify"]:
        (x, y), (nx, ny) = result["verify"], result["verify_norm"]
        acts.append({"action": "click", "role": "verify", "x": x, "y": y, "x_norm": nx, "y_norm": ny})
    return acts


def annotate(img: Image.Image, layout: Layout, result: dict, rows: int = 3,
             cols: int = 3) -> Image.Image:
    """Debug overlay: grid cells numbered, chosen tiles in green, Verify point."""
    out = img.copy()
    d = ImageDraw.Draw(out)
    gx, gy, gw, gh = layout.grid
    d.rectangle((gx, gy, gx + gw, gy + gh), outline="magenta", width=3)
    for i in range(rows * cols):
        r, c = divmod(i, cols)
        x, y = gx + c * gw // cols, gy + r * gh // rows
        color = "lime" if i in result["tiles"] else "red"
        d.rectangle((x, y, x + gw // cols, y + gh // rows), outline=color, width=4 if color == "lime" else 1)
        d.text((x + 8, y + 8), str(i), fill=color)
    for cx, cy in result["click"]:
        d.ellipse((cx - 10, cy - 10, cx + 10, cy + 10), outline="lime", width=4)
    if result["verify"]:
        vx, vy = result["verify"]
        d.ellipse((vx - 10, vy - 10, vx + 10, vy + 10), outline="orange", width=4)
    return out
