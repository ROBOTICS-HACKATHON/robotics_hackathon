"""Solve a folder of PNGs and return structured click instructions."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from PIL import Image

from .client import VLM_MODEL, WorkersAI
from .grid import solve_grid, solve_tiles, tile_centers
from .screen import annotate, solve_screenshot


def _natural(p: Path) -> list:
    # tile2.png before tile10.png
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", p.name)]


def list_pngs(directory: str | Path) -> list[Path]:
    files = sorted(Path(directory).glob("*.png"), key=_natural)
    if not files:
        raise FileNotFoundError(f"no .png files in {directory}")
    return files


async def solve_dir(ai: WorkersAI, directory: str | Path, target: str | None = None, *,
                    layout: str = "screenshot", grid: tuple[int, int] = (3, 3),
                    mode: str = "ask", model: str = VLM_MODEL,
                    debug_dir: str | Path | None = None) -> dict[str, Any]:
    """Solve every PNG in `directory`.

    layout="screenshot": full-page capture; finds the dialog, reads the target from
                         the banner (unless given) -> tiles + click points in
                         screenshot pixels + the Verify button.
    layout="grid":       each PNG is already just the grid (target required).
    layout="tiles":      each PNG is ONE tile (target required) -> files to click.
    """
    if layout != "screenshot" and not target:
        raise ValueError(f"--target is required for layout={layout!r}")
    files = list_pngs(directory)
    out: dict[str, Any] = {"mode": mode, "model": model, "directory": str(directory),
                           "layout": layout}

    if layout == "tiles":
        picked = await solve_tiles(ai, [f.read_bytes() for f in files], target,
                                   mode=mode, model=model)
        out["target"] = target
        out["click"] = [files[i].name for i in picked]
        out["results"] = [{"file": f.name, "click": i in picked} for i, f in enumerate(files)]
        return out

    rows, cols = grid
    out["grid"] = f"{rows}x{cols}"
    out["results"] = []
    for f in files:
        if layout == "screenshot":
            try:
                res, img, lay = await solve_screenshot(
                    ai, f.read_bytes(), target=target, rows=rows, cols=cols,
                    model=model, mode=mode)
            except Exception as e:  # one bad screenshot shouldn't sink the folder
                out["results"].append({"file": f.name, "error": f"{type(e).__name__}: {e}"})
                continue
            if debug_dir:
                Path(debug_dir).mkdir(parents=True, exist_ok=True)
                annotate(img, lay, res, rows, cols).save(Path(debug_dir) / f"{f.stem}.debug.png")
            out["results"].append({"file": f.name, **res})
        else:
            data = f.read_bytes()
            tiles = await solve_grid(ai, data, target, rows=rows, cols=cols,
                                     mode=mode, model=model)
            centers = tile_centers(Image.open(f).size, rows, cols)
            out["results"].append({"file": f.name, "target": target, "tiles": tiles,
                                   "click": [list(centers[i]) for i in tiles]})
    return out
