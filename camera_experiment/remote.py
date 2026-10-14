"""Classify normalized grid crops through the deployed solver API."""
from __future__ import annotations

import os
from typing import cast

import httpx
from typed_errs import Result

from camera_experiment.geometry import Failure, async_attempt

DEFAULT_ENDPOINT = "https://captchasolver.alvaro-camposmndz.workers.dev"


async def grid_tiles(endpoint: str, image: bytes, target: str, model: str) -> Result[list[int], Failure]:
    """Return validated zero-based tile choices from the deployed API."""
    return await async_attempt("deployed solver API", lambda: _grid_tiles(endpoint, image, target, model))


async def _grid_tiles(endpoint: str, image: bytes, target: str, model: str) -> list[int]:
    async with httpx.AsyncClient(timeout=120) as client:
        response = await client.post(
            endpoint.rstrip("/") + "/api/solve",
            headers={"X-CF-Account-Id": os.environ["CLOUDFLARE_ACCOUNT_ID"],
                     "X-CF-Api-Token": os.environ["CLOUDFLARE_API_TOKEN"]},
            files={"image": ("grid.png", image, "image/png")},
            data={"layout": "grid", "grid": "3x3", "target": target, "model": model},
        )
        _ = response.raise_for_status()
        body = cast(object, response.json())
    if not isinstance(body, dict):
        raise TypeError("solver API returned no object")
    values = cast(dict[str, object], body).get("tiles")
    if not isinstance(values, list):
        raise TypeError("solver API returned no tile list")
    tiles: list[int] = []
    for tile in cast(list[object], values):
        if type(tile) is not int or not 0 <= tile < 9:
            raise ValueError("solver API returned an invalid tile index")
        tiles.append(tile)
    if len(set(tiles)) != len(tiles):
        raise ValueError("solver API returned duplicate tiles")
    return sorted(tiles)
