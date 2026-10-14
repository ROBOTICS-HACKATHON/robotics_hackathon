"""`captchasolver-probe`: check which vision models actually answer on this account."""
from __future__ import annotations

import asyncio
import io

from PIL import Image, ImageDraw

from .client import WorkersAI, data_uri

MOONDREAM = "@cf/moondream/moondream3.1-9B-A2B"
CANDIDATES = [
    "@cf/google/gemma-4-26b-a4b-it",
    "@cf/meta/llama-4-scout-17b-16e-instruct",
    "@cf/meta/llama-3.2-11b-vision-instruct",  # may need the one-time license "agree"
]


def _sample() -> bytes:
    img = Image.new("RGB", (240, 240), "white")
    ImageDraw.Draw(img).text((70, 110), "HELLO 42", fill="black")
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


async def _main() -> None:
    uri = data_uri(_sample())
    chat = {"messages": [{"role": "user", "content": [
        {"type": "text", "text": "What text is shown? Reply with only the text."},
        {"type": "image_url", "image_url": {"url": uri}}]}], "max_tokens": 32}
    tests = [(m, chat) for m in CANDIDATES] + [
        (MOONDREAM, {"task": "query", "image": uri, "question": "What text is shown?"})]
    async with WorkersAI() as ai:
        print(f"account={ai.account_id[:6]}...\n")
        for model, payload in tests:
            try:
                status, body = await ai.raw(model, payload)
            except Exception as e:  # noqa: BLE001 - diagnostics, show everything
                status, body = -1, repr(e)
            print(f"--- {model}: HTTP {status}\n{body[:600]}\n")


def main() -> None:
    asyncio.run(_main())


if __name__ == "__main__":
    main()
