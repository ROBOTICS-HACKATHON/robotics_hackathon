"""'Click all squares containing X' captchas."""
from __future__ import annotations

import asyncio
import io
import re
from dataclasses import dataclass
from typing import Literal

from PIL import Image

from .client import VLM_MODEL, WorkersAI, data_uri, vlm_query

DETECT_MODEL = "@cf/moondream/moondream3.1-9B-A2B"  # open-vocabulary `detect`
CLASSIFY_MODEL = "@cf/microsoft/resnet-50"          # ImageNet labels, per-tile fallback


@dataclass
class Box:
    x_min: float
    y_min: float
    x_max: float
    y_max: float


def to_pixels(boxes: list[Box], size: tuple[int, int]) -> list[Box]:
    """Moondream's coordinate convention is undocumented on Cloudflare, so detect it.

    All values <= 1 -> normalized, scale up. Values within the image -> already pixels.
    Anything else -> raise rather than silently click the wrong squares.
    """
    w, h = size
    top = max((max(b.x_min, b.y_min, b.x_max, b.y_max) for b in boxes), default=0.0)
    if top <= 1.0:
        return [Box(b.x_min * w, b.y_min * h, b.x_max * w, b.y_max * h) for b in boxes]
    if top <= max(w, h) * 1.05:
        return boxes
    raise ValueError(f"unrecognised box coordinates (max {top}) for {w}x{h} image")


def tiles_from_boxes(boxes: list[Box], size: tuple[int, int], rows: int, cols: int,
                     *, min_overlap: float = 0.15) -> list[int]:
    """Row-major tile indices hit by a pixel-space box.

    A tile counts if the box covers >= min_overlap of the tile OR >= min_overlap of
    the box lies in the tile (so small objects fully inside one tile still count,
    but a thin sliver of a big object does not).
    """
    w, h = size
    tw, th = w / cols, h / rows
    hit: set[int] = set()
    for b in boxes:
        box_area = max((b.x_max - b.x_min) * (b.y_max - b.y_min), 1e-9)
        for r in range(rows):
            for c in range(cols):
                ix = min(b.x_max, (c + 1) * tw) - max(b.x_min, c * tw)
                iy = min(b.y_max, (r + 1) * th) - max(b.y_min, r * th)
                if ix <= 0 or iy <= 0:
                    continue
                inter = ix * iy
                if max(inter / (tw * th), inter / box_area) >= min_overlap:
                    hit.add(r * cols + c)
    return sorted(hit)


def tile_centers(size: tuple[int, int], rows: int, cols: int) -> list[tuple[int, int]]:
    """Pixel centre of each tile (row-major), i.e. where to click."""
    w, h = size
    return [(int((c + 0.5) * w / cols), int((r + 0.5) * h / rows))
            for r in range(rows) for c in range(cols)]


async def _detect(ai: WorkersAI, image: bytes, target: str, model: str) -> list[Box]:
    result = await ai.run(model, json={
        "task": "detect", "image": data_uri(image), "target": target,
        "reasoning": False,
    })
    return [Box(o["x_min"], o["y_min"], o["x_max"], o["y_max"])
            for o in result.get("objects", [])]


def singular(word: str) -> str:
    w = word.strip().lower()
    if w.endswith("ies"):
        return w[:-3] + "y"
    if w.endswith(("ses", "xes", "ches", "shes")):
        return w[:-2]
    return w[:-1] if w.endswith("s") and not w.endswith("ss") else w


async def _tile_classify(ai: WorkersAI, tile: bytes, target: str, min_score: float) -> bool:
    result = await ai.run(CLASSIFY_MODEL, image=tile)
    # ImageNet labels look like "traffic light, traffic signal, stoplight"
    pat = re.compile(rf"\b{re.escape(singular(target))}s?\b")
    return any(pat.search(p["label"].lower()) and p["score"] >= min_score for p in result)


def parse_yes_no(text: str) -> bool:
    """First standalone yes/no wins, so 'Yes, there is a cat.' and 'No.' both parse."""
    m = re.search(r"\b(yes|no)\b", text.lower())
    return bool(m) and m.group(1) == "yes"


async def _tile_ask(ai: WorkersAI, tile: bytes, target: str, model: str) -> bool:
    answer = await vlm_query(
        ai, tile, f"Does this image contain {target}? Answer with only yes or no.",
        model=model, max_tokens=8)
    return parse_yes_no(answer)


def _crops(img: Image.Image, rows: int, cols: int) -> list[bytes]:
    w, h = img.size
    out = []
    for r in range(rows):
        for c in range(cols):
            buf = io.BytesIO()
            img.crop((c * w // cols, r * h // rows,
                      (c + 1) * w // cols, (r + 1) * h // rows)).save(buf, "PNG")
            out.append(buf.getvalue())
    return out


async def solve_grid(ai: WorkersAI, image: bytes, target: str, *, rows: int = 3,
                     cols: int = 3, mode: Literal["detect", "ask", "classify"] = "detect",
                     model: str | None = None, min_score: float = 0.2) -> list[int]:
    """Return row-major indices of tiles containing `target`.

    detect:   one Moondream `detect` call on the whole image, boxes mapped to tiles
              (fastest; for ONE picture cut into squares).
    ask:      Moondream yes/no per tile, in parallel (rows*cols calls; for grids
              where each square is an independent photo; open vocabulary).
    classify: ResNet-50 per tile, in parallel (cheap, ImageNet classes only).
    """
    img = Image.open(io.BytesIO(image)).convert("RGB")
    if mode == "detect":
        boxes = to_pixels(await _detect(ai, image, target, model or DETECT_MODEL), img.size)
        return tiles_from_boxes(boxes, img.size, rows, cols)
    return await solve_tiles(ai, _crops(img, rows, cols), target, mode=mode,
                             model=model or VLM_MODEL, min_score=min_score)


async def solve_tiles(ai: WorkersAI, tiles: list[bytes], target: str, *,
                      mode: Literal["ask", "classify"] = "ask",
                      model: str = VLM_MODEL, min_score: float = 0.2) -> list[int]:
    """Same as solve_grid for tiles that already arrive as separate images."""
    if mode == "ask":
        flags = await asyncio.gather(*[_tile_ask(ai, t, target, model) for t in tiles])
    else:
        flags = await asyncio.gather(*[_tile_classify(ai, t, target, min_score) for t in tiles])
    return [i for i, ok in enumerate(flags) if ok]
