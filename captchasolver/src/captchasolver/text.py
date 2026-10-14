"""Distorted-text captchas via Moondream's `query` task."""
from __future__ import annotations

import asyncio
import re
from collections import Counter

from .client import VLM_MODEL, WorkersAI, vlm_query

QUESTION = (
    "This image is a text captcha. Read the characters exactly as shown, "
    "respecting upper/lower case. Reply with ONLY the characters: "
    "no spaces, quotes, or explanation."
)


def clean(raw: str, length: int | None = None) -> str:
    text = re.sub(r"[^A-Za-z0-9]", "", raw)
    return text[:length] if length else text


async def solve_text(ai: WorkersAI, image: bytes, *, model: str = VLM_MODEL,
                     length: int | None = None, votes: int = 1) -> str:
    """Return the captcha text. `votes` > 1 runs parallel samples and majority-votes."""
    temp = 0.3 if votes > 1 else 0
    raws = await asyncio.gather(*[
        vlm_query(ai, image, QUESTION, model=model, temperature=temp)
        for _ in range(votes)])
    return Counter(clean(r, length) for r in raws).most_common(1)[0][0]
