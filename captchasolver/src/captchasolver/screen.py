"""Screenshot or camera photo of the challenge -> target, grid location, click coordinates.

The dialog is located from its blue banner and Verify button, so no DOM access is needed. The finder
is built to survive a *photo of a screen* that has already been straightened to screen size:
  * the banner blue is found relative to the photo itself (white balance, brightness and contrast
    move the exact colour), not as a fixed RGB value;
  * the grid is located from edges and the dialog's proportions, not from a pure-white gap;
  * a few degrees of residual tilt are estimated and removed (positions are mapped back);
  * anything that fails a geometry cross-check is REFUSED (LayoutNotFound) rather than guessed,
    because a robot must never press positions from a half-found layout.
"""
from __future__ import annotations

import io
import math
import re
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw

from .client import VLM_MODEL, WorkersAI, vlm_query
from .grid import solve_grid, tile_centers

INSTRUCTION_Q = (
    "This is a captcha instruction banner. Which kind of object must be selected? "
    "Reply with ONLY that object as a plural noun, e.g. cars."
)

# Proportions of the demo dialog as fractions of the banner width (= the grid's side), measured on a
# real screenshot: banner 517 wide, grid 517x517 starting 10 px under it, Verify starting 22 px under the grid.
GRID_GAP = 0.020
VERIFY_GAP = 0.043
MAX_TILT = 10.0   # degrees


class LayoutNotFound(ValueError):
    """The dialog was not found, or what was found is not plausible. `code` is machine-readable:
    no_dialog | layout_inconsistent | cropped."""

    def __init__(self, message: str, code: str = "no_dialog"):
        super().__init__(message)
        self.code = code


@dataclass
class Layout:
    header: tuple[int, int, int, int]            # x, y, w, h
    grid: tuple[int, int, int, int]              # x, y, w, h
    verify: tuple[int, int] | None               # click point


def _blue_mask(a: np.ndarray) -> np.ndarray:
    """Pixels that are 'the banner blue' IN THIS PHOTO.

    A pixel's blueness is how much bluer than red/green it is. The threshold comes from the photo:
    half the typical blueness of its clearly-blue pixels, so a dim, washed-out or tinted photo still works.
    """
    a16 = a.astype(np.int16)
    blueness = a16[..., 2] - np.maximum(a16[..., 0], a16[..., 1])
    strong = blueness[blueness > 25]
    if strong.size < 50:
        raise LayoutNotFound("No blue instruction banner found in the image.")
    return blueness > 0.5 * float(np.percentile(strong, 90))


def _first_block(idx: np.ndarray, max_gap: int) -> tuple[int, int]:
    """First run of indices, bridging gaps of up to `max_gap` (blur and text break rows apart)."""
    breaks = np.where(np.diff(idx) > max_gap + 1)[0]
    return int(idx[0]), int(idx[breaks[0]] if breaks.size else idx[-1])


def _find_banner(mask: np.ndarray) -> tuple[int, int, int, int]:
    H, W = mask.shape
    per_row = mask.sum(1)
    peak = int(per_row.max())
    if peak < max(12, 0.02 * W):
        raise LayoutNotFound("No blue instruction banner found in the image.")
    wide = np.where(per_row >= 0.4 * peak)[0]
    y0, y1 = _first_block(wide, max(2, int(0.01 * H)))
    cols = np.where(mask[y0:y1 + 1].sum(0) > (y1 - y0) // 2)[0]
    if cols.size == 0:
        raise LayoutNotFound("No blue instruction banner found in the image.")
    return int(cols.min()), y0, int(cols.max()), y1


def estimate_tilt(img: Image.Image) -> float:
    """Degrees the dialog is rotated clockwise, from the straightness of the banner's top edge."""
    a = np.asarray(img.convert("RGB"))
    mask = _blue_mask(a)
    x0, y0, x1, _ = _find_banner(mask)
    side = x1 - x0 + 1
    xs = np.arange(x0 + int(0.1 * side), x1 - int(0.1 * side))
    ylo, yhi = max(0, y0 - int(0.2 * side)), min(a.shape[0], y0 + int(0.2 * side))
    sub = mask[ylo:yhi, xs]
    ok = sub.any(0)
    if ok.sum() < 20:
        return 0.0
    slope = np.polyfit(xs[ok], sub.argmax(0)[ok] + ylo, 1)[0]
    return math.degrees(math.atan(slope))


def _rotate_point(x: float, y: float, angle: float, size: tuple[int, int], back: bool) -> tuple[float, float]:
    """Map a point between the original image and the one produced by img.rotate(angle) (counter-
    clockwise about the centre). back=True maps rotated -> original."""
    cx, cy = size[0] / 2, size[1] / 2
    t = math.radians(angle)
    c, s = math.cos(t), math.sin(t)
    dx, dy = x - cx, y - cy
    if back:
        return cx + dx * c - dy * s, cy + dx * s + dy * c
    return cx + dx * c + dy * s, cy - dx * s + dy * c


def find_layout(img: Image.Image) -> Layout:
    """Locate the dialog in an UPRIGHT image (see `locate` for tilted ones) at any resolution."""
    a = np.asarray(img.convert("RGB"))
    H, W = a.shape[:2]
    mask = _blue_mask(a)
    x0, y0, x1, y1 = _find_banner(mask)
    bw, bh = x1 - x0 + 1, y1 - y0 + 1
    side = bw
    if not 2.0 <= bw / bh <= 5.5 or side < max(30, 0.04 * W):
        raise LayoutNotFound("The blue block found does not look like the instruction banner.", "layout_inconsistent")
    header = (x0, y0, bw, bh)

    # Grid top = the strongest horizontal edge just under the banner (white gap -> tile border).
    gray = a[:, x0:x1 + 1].astype(np.float32).mean(2)
    edge = np.abs(np.diff(gray, axis=0)).mean(1)               # edge[i]: between row i and i+1
    lo, hi = y1 + 1 + int(0.008 * side), min(H - 2, y1 + 1 + int(0.045 * side))
    if hi <= lo:
        raise LayoutNotFound("The grid is not visible under the banner.", "cropped")
    gy = lo + int(np.argmax(edge[lo:hi])) + 1
    if gy + side > H:
        raise LayoutNotFound("The grid runs off the bottom of the image (cropped or scrolled).", "cropped")
    grid = (x0, gy, side, side)

    # Verify = the narrower blue block under the grid, on the right. It must sit where the dialog's
    # proportions say it should, otherwise the whole layout is distrusted.
    r0, r1 = gy + int(0.985 * side), min(H, gy + int(1.30 * side))
    c0, c1 = x0 + int(0.45 * side), min(W, x1 + int(0.04 * side))
    sub = mask[r0:r1, c0:c1]
    verify = None
    if sub.size and sub.sum(1).max() >= 0.08 * side:
        rows = np.where(sub.sum(1) >= 0.5 * sub.sum(1).max())[0]
        vy0, vy1 = _first_block(rows, max(2, int(0.01 * H)))
        vcols = np.where(sub[vy0:vy1 + 1].sum(0) >= 0.5 * (vy1 - vy0 + 1))[0]
        if vcols.size:
            vy0, vy1 = r0 + vy0, r0 + vy1
            vx = c0 + (int(vcols.min()) + int(vcols.max())) // 2
            expected_top = gy + side + VERIFY_GAP * side
            if abs(vy0 - expected_top) > 0.05 * side or vx < x0 + 0.5 * side:
                raise LayoutNotFound("The Verify button is not where this dialog's layout says it should be.",
                                     "layout_inconsistent")
            verify = (vx, (vy0 + vy1) // 2)
    if verify is None:
        raise LayoutNotFound("The Verify button was not found under the grid.", "layout_inconsistent")
    return Layout(header, grid, verify)


def locate(img: Image.Image) -> tuple[Image.Image, Layout, float]:
    """(upright image, layout in its coordinates, tilt in degrees). Straightens up to MAX_TILT."""
    tilt = estimate_tilt(img)
    if abs(tilt) > MAX_TILT:
        raise LayoutNotFound(f"The dialog is tilted {tilt:.0f} degrees; straighten the photo first.",
                             "layout_inconsistent")
    if abs(tilt) < 0.35:
        return img, find_layout(img), 0.0
    upright = img.rotate(tilt, resample=Image.BICUBIC, fillcolor=(255, 255, 255))
    return upright, find_layout(upright), round(tilt, 2)


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
    """Returns (result, upright image, layout). Positions in `result` are in the coordinates of the
    image as given. If the dialog was tilted, the image and layout returned are the straightened
    ones (for overlays) and `result["tilt_deg"]` says by how much."""
    orig = Image.open(io.BytesIO(png)).convert("RGB")
    img, layout, tilt = locate(orig)
    target = target or await read_target(ai, img, layout, model=model)
    gx, gy, gw, gh = layout.grid
    tiles = await solve_grid(ai, _png(img.crop((gx, gy, gx + gw, gy + gh)), 768), target,
                             rows=rows, cols=cols, mode=mode, model=model)
    centers = tile_centers((gw, gh), rows, cols)
    W, H = orig.size

    def back(x: float, y: float) -> list[int]:       # upright -> original image coordinates
        if tilt:
            x, y = _rotate_point(x, y, tilt, orig.size, back=True)
        return [int(round(x)), int(round(y))]
    click = [back(gx + centers[i][0], gy + centers[i][1]) for i in tiles]
    verify = back(*layout.verify) if layout.verify else None
    result = {
        "target": target,
        "image_size": [W, H],
        "tilt_deg": tilt,
        "grid_box": list(layout.grid),            # in the straightened image (same as the photo if tilt is 0)
        "tiles": tiles,
        "click": click,                           # pixels of the image as given
        "verify": verify,
        # resolution-independent (0..1): multiply by the real screen size to click
        "click_norm": [[round(x / W, 5), round(y / H, 5)] for x, y in click],
        "verify_norm": [round(verify[0] / W, 5), round(verify[1] / H, 5)] if verify else None,
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
    """Debug overlay on the (straightened) image: grid cells numbered, chosen tiles in green, Verify point."""
    out = img.copy()
    d = ImageDraw.Draw(out)
    gx, gy, gw, gh = layout.grid
    d.rectangle((gx, gy, gx + gw, gy + gh), outline="magenta", width=3)
    centers = tile_centers((gw, gh), rows, cols)
    for i in range(rows * cols):
        r, c = divmod(i, cols)
        x, y = gx + c * gw // cols, gy + r * gh // rows
        on = i in result["tiles"]
        color = "lime" if on else "red"
        d.rectangle((x, y, x + gw // cols, y + gh // rows), outline=color, width=4 if on else 1)
        d.text((x + 8, y + 8), str(i), fill=color)
        if on:
            cx, cy = gx + centers[i][0], gy + centers[i][1]
            d.ellipse((cx - 10, cy - 10, cx + 10, cy + 10), outline="lime", width=4)
    if layout.verify:
        vx, vy = layout.verify
        d.ellipse((vx - 10, vy - 10, vx + 10, vy + 10), outline="orange", width=4)
    return out
