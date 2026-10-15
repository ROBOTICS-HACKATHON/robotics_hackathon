"""Drive the team's demo site (image-grid challenge) with Playwright."""
from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass

from playwright.async_api import Page, TimeoutError as PWTimeout, async_playwright

from .client import WorkersAI
from .grid import solve_tiles

DEFAULT_URL = "https://robot-captcha-demo.pages.dev/"
CHALLENGE = "#image-challenge"


class SiteDidNotAnswer(RuntimeError):
    """After Verify the site showed neither acceptance nor rejection. That is an ERROR, never a FAIL:
    a slow site (or a slow container) must not be able to turn a correct answer into a failure."""


ACCEPTED = f'{CHALLENGE}[data-state="success"]'
REJECTED = f'{CHALLENGE}[data-state="error"]'


async def _outcome(page: Page, timeout_ms: int = 30000) -> bool:
    """After Verify: True = the site accepted the selection, False = it rejected it.

    Three signals, first one wins: the success state, the error state ("Incorrect selection"), or the
    navigation to /verified/ (which the site only starts after accepting). A wrong answer answers at once,
    so the long timeout only matters for slow accepts. If nothing shows within it, raise SiteDidNotAnswer.
    """
    async def watch(coro, verdict):
        try:
            await coro
        except PWTimeout:
            return None
        except Exception:                 # e.g. 'context destroyed': the page is navigating; re-check below
            return None
        return verdict
    pending = {
        asyncio.ensure_future(watch(page.wait_for_url("**/verified/**", timeout=timeout_ms, wait_until="commit"), True)),
        asyncio.ensure_future(watch(page.wait_for_selector(ACCEPTED, state="attached", timeout=timeout_ms), True)),
        asyncio.ensure_future(watch(page.wait_for_selector(REJECTED, state="attached", timeout=timeout_ms), False)),
    }
    verdict = None
    try:
        while pending and verdict is None:
            done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
            verdicts = [t.result() for t in done if t.result() is not None]
            verdict = verdicts[0] if verdicts else None
    finally:
        for t in pending:
            t.cancel()
    if verdict is not None:
        return verdict
    # Last, definitive look at where the page ended up.
    if "/verified" in page.url:
        return True
    try:
        if await page.locator(ACCEPTED).count():
            return True
        if await page.locator(REJECTED).count():
            return False
    except Exception:
        pass
    raise SiteDidNotAnswer(f"The site neither accepted nor rejected the answer within {timeout_ms // 1000} s.")



@dataclass
class Result:
    index: int
    subject: str
    picked: list[int]
    solved: bool
    attempts: int


async def _open_image_challenge(page: Page, url: str, index: int) -> None:
    """Open image challenge `index` (the dropdown's `image:<index>` entry)."""
    await page.goto(url)
    await page.select_option("#challenge-number", f"image:{index}")
    await page.uncheck("#fullscreen")  # deterministic viewport; fullscreen can fail headless
    await page.click('button.choice[data-challenge="image"]')
    await page.click("#human-checkbox")
    await page.wait_for_selector(f"{CHALLENGE} .image-tile", state="visible")
    await page.wait_for_function(
        "[...document.querySelectorAll('#image-challenge .image-tile img')]"
        ".every(i => i.complete && i.naturalWidth > 0)")


async def solve_image_challenge(page: Page, ai: WorkersAI, *, mode: str = "ask",
                                max_attempts: int = 2) -> tuple[str, list[int], bool, int]:
    tiles = page.locator(f"{CHALLENGE} .image-tile")
    subject = (await page.locator(f"{CHALLENGE} .image-subject").inner_text()).strip()
    picked: list[int] = []
    for attempt in range(1, max_attempts + 1):
        # What the user sees: screenshot each tile rather than trusting img src.
        shots = [await tiles.nth(i).screenshot() for i in range(await tiles.count())]
        picked = await solve_tiles(ai, shots, subject, mode=mode)
        for i in picked:
            await tiles.nth(i).click()
        await page.click(f"{CHALLENGE} .verify-button")
        if await _outcome(page):          # raises SiteDidNotAnswer if the site never answers: an error, not a fail
            return subject, picked, True, attempt
        await page.click(f"{CHALLENGE} .reset-button")
    return subject, picked, False, max_attempts


async def snap(url: str, indices: list[int], out_dir: str) -> None:
    """Save each challenge's whole grid as one PNG (no solving, no API calls)."""
    from pathlib import Path
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    async with async_playwright() as pw:
        browser = await pw.chromium.launch()
        for index in indices:
            page = await (await browser.new_context()).new_page()
            await _open_image_challenge(page, url, index)
            tiles = page.locator(f"{CHALLENGE} .image-tile")
            boxes = [await tiles.nth(i).bounding_box() for i in range(await tiles.count())]
            x0 = min(b["x"] for b in boxes); y0 = min(b["y"] for b in boxes)
            x1 = max(b["x"] + b["width"] for b in boxes)
            y1 = max(b["y"] + b["height"] for b in boxes)
            subject = (await page.locator(f"{CHALLENGE} .image-subject").inner_text()).strip()
            path = Path(out_dir) / f"grid-{index + 1:02d}-{subject}.png"
            await page.screenshot(path=str(path), clip={
                "x": x0, "y": y0, "width": x1 - x0, "height": y1 - y0})
            print(path)
        await browser.close()


async def run(url: str, indices: list[int], *, mode: str, headless: bool,
              max_attempts: int) -> list[Result]:
    results = []
    async with WorkersAI() as ai, async_playwright() as pw:
        browser = await pw.chromium.launch(headless=headless)
        for index in indices:
            ctx = await browser.new_context()  # fresh sessionStorage per challenge
            page = await ctx.new_page()
            await _open_image_challenge(page, url, index)
            subject, picked, ok, n = await solve_image_challenge(
                page, ai, mode=mode, max_attempts=max_attempts)
            results.append(Result(index, subject, picked, ok, n))
            print(f"challenge {index + 1:>2} [{subject:<9}] picked={picked} "
                  f"{'PASS' if ok else 'FAIL'} (attempts={n})", flush=True)
            await ctx.close()
        await browser.close()
    return results


def main() -> None:
    p = argparse.ArgumentParser(prog="captchasolver-web")
    p.add_argument("--url", default=DEFAULT_URL)
    p.add_argument("--challenge", type=int, action="append",
                   help="image challenge number, 1-based; repeat for several. Default: the first 10 "
                        "(use `captchasolver test` for large runs)")
    p.add_argument("--mode", choices=["ask", "classify"], default="ask")
    p.add_argument("--attempts", type=int, default=2)
    p.add_argument("--snap", metavar="DIR",
                   help="only save each challenge's grid as a PNG in DIR (no solving)")
    p.add_argument("--headed", action="store_true", help="show the browser window")
    a = p.parse_args()
    indices = [c - 1 for c in a.challenge] if a.challenge else list(range(10))
    if a.snap:
        return asyncio.run(snap(a.url, indices, a.snap))
    res = asyncio.run(run(a.url, indices, mode=a.mode, headless=not a.headed,
                          max_attempts=a.attempts))
    print(f"\n{sum(r.solved for r in res)}/{len(res)} solved")


if __name__ == "__main__":
    main()
