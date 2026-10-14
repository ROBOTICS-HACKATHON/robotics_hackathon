"""`captchasolver test`: run the solver over many challenges on the demo site and report.

Mirrors the site's session panel (count -- default 100 --, split, shuffle) but drives each
challenge on a fresh page, because the site's own session cannot skip a failed challenge.

Pipelines:
  screenshot  full-viewport screenshot -> find dialog -> read banner -> click with the MOUSE at
              the returned coordinates. This is what the robot does, so it tests everything.
  dom         per-tile element screenshots + DOM clicks (isolates the vision model).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

from playwright.async_api import Page, TimeoutError as PWTimeout, async_playwright

from .client import VLM_MODEL, WorkersAI, describe_exc
from .screen import annotate, solve_screenshot
from .web import CHALLENGE, DEFAULT_URL, _open_image_challenge, solve_image_challenge

PRESSED_JS = ("els => els.map((e, i) => e.getAttribute('aria-pressed') === 'true' ? i : -1)"
              ".filter(i => i >= 0)")


@dataclass
class Record:
    index: int
    label: str
    split: str
    subject: str = ""            # what the page says (ground truth for the banner)
    target: str = ""             # what the solver read
    tiles: list[int] | None = None
    clicked_ok: bool | None = None   # pressed tiles == intended tiles (actuation check)
    passed: bool = False
    seconds: float = 0.0
    error: str = ""
    screenshot: str = ""         # files pipeline: where the captured PNG was saved
    fail_image: str = ""         # what the page looked like after the failure (failed_runs/)


def split_of(label: str) -> str:
    return "test" if label.startswith("test-") else "train"


def select_challenges(options: list[tuple[int, str]], count: int, split: str,
                      seed: int) -> list[tuple[int, str]]:
    pool = [(i, lbl) for i, lbl in options if split == "all" or split_of(lbl) == split]
    random.Random(seed).shuffle(pool)
    return pool[:count]


def summarize(records: list[Record]) -> dict:
    n = len(records)
    done = [r for r in records if not r.error]
    by_subject: dict[str, list[bool]] = defaultdict(list)
    for r in done:
        by_subject[r.subject].append(r.passed)
    checked = [r for r in done if r.target]
    clicks = [r for r in done if r.clicked_ok is not None]
    return {
        "total": n,
        "passed": sum(r.passed for r in records),
        "failed": sum(not r.passed and not r.error for r in records),
        "accuracy": round(sum(r.passed for r in records) / n, 4) if n else 0.0,
        "errors": sum(bool(r.error) for r in records),
        "banner_read_accuracy": (round(sum(r.target == r.subject for r in checked) / len(checked), 4)
                                 if checked else None),
        "click_accuracy": (round(sum(bool(r.clicked_ok) for r in clicks) / len(clicks), 4)
                           if clicks else None),
        "mean_seconds": round(sum(r.seconds for r in records) / n, 2) if n else 0.0,
        "by_subject": {k: f"{sum(v)}/{len(v)}" for k, v in sorted(by_subject.items())},
    }


def failure_cause(r: Record) -> str:
    """Why a record did not pass, judged only from what we observed (no answer key)."""
    if r.error:
        return f"error: {r.error}"
    if r.passed:
        return ""
    if r.target and r.target != r.subject:
        return f"banner misread: read {r.target!r}, page says {r.subject!r}"
    if r.clicked_ok is False:
        return f"clicks missed the intended tiles {r.tiles}"
    return f"wrong tile selection: picked {r.tiles}"


_CODES = {"green": "32", "red": "31", "yellow": "33", "cyan": "36", "dim": "2", "bold": "1"}


def paint(text: str, style: str, on: bool) -> str:
    return f"\033[{_CODES[style]}m{text}\033[0m" if on else text


def use_color(choice: str) -> bool:
    if choice == "always":
        return True
    if choice == "never" or os.environ.get("NO_COLOR"):
        return False
    return sys.stdout.isatty()


def format_line(r: Record, k: int, n: int, color: bool) -> str:
    if r.error:
        mark, tag, style = "💥", "ERROR", "yellow"
    elif r.passed:
        mark, tag, style = "✅", "PASS ", "green"
    else:
        mark, tag, style = "❌", "FAIL ", "red"
    head = paint(f"{mark} {tag}", style, color)
    detail = r.error if r.error else f"picked={r.tiles}"
    line = (f"{head} {paint(f'[{k:>3}/{n}]', 'dim', color)} {r.label:<16} "
            f"{r.subject:<9} {paint(detail, 'dim' if r.passed else style, color)}")
    if not r.passed and not r.error:  # errors already carry their reason inline
        line += "\n" + paint(f"         ↳ {failure_cause(r)}", style, color)
    if r.fail_image:
        p = Path(r.fail_image)
        line += paint(f"\n         📷 {p.parent.name}/{p.name}", "dim", color)
    return line


def format_summary(records: list[Record], s: dict, color: bool) -> str:
    n = s["total"]
    bad = [r for r in records if not r.passed]
    zero = lambda v: "dim" if v == 0 else None  # noqa: E731
    out = ["", paint("═" * 14 + " Results " + "═" * 14, "bold", color)]
    out.append("   ".join([
        paint(f"✅ {s['passed']} passed", "green" if s["passed"] else "dim", color),
        paint(f"❌ {s['failed']} failed", zero(s["failed"]) or "red", color),
        paint(f"💥 {s['errors']} error{'s' if s['errors'] != 1 else ''}", zero(s["errors"]) or "yellow", color),
        paint(f"({n} total · {s['accuracy']:.1%})", "bold", color)]))
    out.append("")
    if s["banner_read_accuracy"] is not None:
        read = [r for r in records if not r.error and r.target]
        miss = sum(r.target != r.subject for r in read)
        clicks = [r for r in records if r.clicked_ok is not None]
        cmiss = sum(not r.clicked_ok for r in clicks)
        out.append(f"  banner read     {len(read) - miss}/{len(read)} correct   "
                   + paint(f"{miss} misread", "red" if miss else "dim", color))
        out.append(f"  click accuracy  {len(clicks) - cmiss}/{len(clicks)} on target  "
                   + paint(f"{cmiss} off-target", "red" if cmiss else "dim", color))
    out.append(f"  mean time       {s['mean_seconds']}s per challenge")
    out.append("")
    subj: dict[str, list[Record]] = defaultdict(list)
    for r in records:
        if not r.error:
            subj[r.subject].append(r)
    if subj:
        out.append(paint("By subject", "bold", color))
    for name, rs in sorted(subj.items()):
        ok = sum(r.passed for r in rs)
        full = ok == len(rs)
        out.append(f"  {'✅' if full else '❌'} {name:<10}"
                   + paint(f"{ok}/{len(rs)}", "green" if full else "red", color))
    if subj:
        out.append("")
    if not bad:
        out.append(paint("✅ No failures.", "green", color))
    else:
        out.append(paint(f"Failures ({len(bad)})", "red", color))
        for r in bad:
            mark = "💥" if r.error else "❌"
            out.append(f"  {mark} {r.label:<16} {r.subject or '?':<9} "
                       + paint(failure_cause(r), "yellow" if r.error else "red", color))
    return "\n".join(out)


async def _pressed(page: Page) -> list[int]:
    return await page.eval_on_selector_all(f"{CHALLENGE} .image-tile", PRESSED_JS)


async def _outcome(page: Page, timeout_ms: int = 5000) -> bool:
    """After Verify: True on redirect to /verified/, False on the error state."""
    ok = asyncio.ensure_future(page.wait_for_url("**/verified/**", timeout=timeout_ms))
    bad = asyncio.ensure_future(page.wait_for_selector(
        f'{CHALLENGE}[data-state="error"]', timeout=timeout_ms))
    done, pending = await asyncio.wait({ok, bad}, return_when=asyncio.FIRST_COMPLETED)
    for t in pending:
        t.cancel()
    first = next(iter(done))
    try:
        first.result()
    except PWTimeout:
        return False
    return first is ok


async def _record_outcome(page: Page, rec: Record, fail_dir: Path | None, plan=None) -> None:
    """On a FAIL save two images in failed_runs/: the page as it looked after the failed Verify
    (NNN-label.png: tiles selected, the site's error message) and, when we have it, the solver's
    plan drawn on the screenshot it saw (NNN-label-plan.png). On a pass, delete stale ones."""
    if fail_dir is None:
        return
    base = fail_dir / f"{rec.index:03d}-{rec.label}"
    state, plan_png = base.with_suffix(".png"), Path(f"{base}-plan.png")
    if rec.passed:
        for old in (state, plan_png, Path(f"{base}-error.png")):
            old.unlink(missing_ok=True)
        return
    fail_dir.mkdir(parents=True, exist_ok=True)
    await page.screenshot(path=str(state))
    rec.fail_image = str(state)
    if plan:
        annotate(*plan).save(plan_png)


async def _save_error_state(page: Page | None, rec: Record, fail_dir: Path | None) -> None:
    """Best effort: when a challenge crashed, the page at that moment is the best clue."""
    if fail_dir is None or page is None:
        return
    try:
        fail_dir.mkdir(parents=True, exist_ok=True)
        path = fail_dir / f"{rec.index:03d}-{rec.label}-error.png"
        await page.screenshot(path=str(path))
        rec.fail_image = str(path)
    except Exception:  # noqa: BLE001 - the page may already be gone
        pass


async def _run_screenshot(page: Page, ai: WorkersAI, rec: Record, *, viewport: tuple[int, int],
                          mode: str, model: str, fail_dir: Path | None) -> None:
    vw, vh = viewport
    png = await page.screenshot()                      # physical px = css px * dpr
    res, img, lay = await solve_screenshot(ai, png, mode=mode, model=model)
    rec.target, rec.tiles = res["target"], res["tiles"]
    # click via NORMALISED coordinates, so any resolution / DPI scale lands in the same place
    for nx, ny in res["click_norm"]:
        await page.mouse.click(nx * vw, ny * vh)
    rec.clicked_ok = sorted(await _pressed(page)) == sorted(res["tiles"])
    if res["verify_norm"] is None:
        raise RuntimeError("Verify button not found in screenshot")
    await page.mouse.click(res["verify_norm"][0] * vw, res["verify_norm"][1] * vh)
    rec.passed = await _outcome(page)
    await _record_outcome(page, rec, fail_dir, plan=(img, lay, res))


def default_examples_dir() -> Path:
    """<repo>/captchasolver/examples when running from a checkout, else ./examples."""
    root = Path(__file__).resolve().parents[2]
    return root / "examples" if (root / "pyproject.toml").exists() else Path("examples")


async def execute_actions(page: Page, doc: dict) -> tuple[bool, bool]:
    """Follow a solver JSON's `actions` literally. Returns (passed, tiles_clicked_as_planned).

    Uses only the normalised coordinates and the CURRENT viewport size, so it works at any
    resolution. Nothing from the solver's internals is consulted: the JSON is the whole plan.
    """
    vw, vh = await page.evaluate("[window.innerWidth, window.innerHeight]")
    planned: list[int] = []
    clicked_ok: bool | None = None
    for act in doc["actions"]:
        if act["action"] != "click":
            raise ValueError(f"unsupported action {act['action']!r}")
        if act["role"] == "verify":
            clicked_ok = sorted(await _pressed(page)) == sorted(planned)
        await page.mouse.click(act["x_norm"] * vw, act["y_norm"] * vh)
        if act["role"] == "tile":
            planned.append(act["tile"])
    if clicked_ok is None:
        raise RuntimeError("the JSON has no verify action")
    return await _outcome(page), clicked_ok


async def _run_files(page: Page, ai: WorkersAI, rec: Record, *, url: str, out_dir: Path,
                     mode: str, model: str, fail_dir: Path | None) -> None:
    """screenshot -> examples/NNN-label.png -> solver JSON on disk -> act on the JSON."""
    stem = out_dir / f"{rec.index:03d}-{rec.label}"
    vw, vh = await page.evaluate("[window.innerWidth, window.innerHeight]")
    png = await page.screenshot()
    stem.with_suffix(".png").write_bytes(png)
    rec.screenshot = str(stem.with_suffix(".png"))
    res, img, lay = await solve_screenshot(ai, png, mode=mode, model=model)
    rec.target, rec.tiles = res["target"], res["tiles"]
    doc = {"challenge": {"index": rec.index, "label": rec.label, "url": url},
           "viewport": [vw, vh], "model": model, **res}
    stem.with_suffix(".json").write_text(json.dumps(doc, indent=2) + "\n")
    # act from the saved file, exactly as a robot reading it later would
    rec.passed, rec.clicked_ok = await execute_actions(
        page, json.loads(stem.with_suffix(".json").read_text()))
    await _record_outcome(page, rec, fail_dir, plan=(img, lay, res))


async def _run_dom(page: Page, ai: WorkersAI, rec: Record, *, mode: str, fail_dir: Path | None) -> None:
    _, picked, ok, _ = await solve_image_challenge(page, ai, mode=mode, max_attempts=1)
    rec.target, rec.tiles, rec.passed = rec.subject, picked, ok
    await _record_outcome(page, rec, fail_dir)


async def run_tests(url: str, *, count: int = 100, split: str = "all", seed: int = 0,
                    concurrency: int = 4, pipeline: str = "screenshot",
                    viewport: tuple[int, int] = (1914, 1080), dpr: float = 1.0,
                    mode: str = "ask", model: str = VLM_MODEL, headless: bool = True,
                    fail_dir: str | None = None, out_dir: str | None = None,
                    fail_images: bool = True, progress=None,
                    note=print) -> tuple[list[Record], dict]:
    odir = Path(out_dir) if out_dir else default_examples_dir()
    # failed runs always land in <examples>/failed_runs unless told otherwise
    fdir = (Path(fail_dir) if fail_dir else odir / "failed_runs") if fail_images else None
    if pipeline == "files":
        odir.mkdir(parents=True, exist_ok=True)
    async with WorkersAI() as ai, async_playwright() as pw:
        browser = await pw.chromium.launch(headless=headless)
        page = await browser.new_page()
        await page.goto(url)
        raw = await page.eval_on_selector_all(
            '#challenge-number option[value^="image:"]',
            "els => els.map(e => [e.value, e.textContent.trim()])")
        await page.close()
        options = [(int(v.split(":")[1]), lbl) for v, lbl in raw]
        chosen = select_challenges(options, count, split, seed)
        if len(chosen) < count:
            note(f"note: only {len(chosen)} challenges available for split={split!r} "
                 f"(asked for {count})")
        sem, records, counter = asyncio.Semaphore(concurrency), [], [0]

        async def one(index: int, label: str) -> None:
            rec = Record(index, label, split_of(label))
            page = None
            async with sem:
                t0 = time.perf_counter()
                ctx = await browser.new_context(viewport={"width": viewport[0], "height": viewport[1]},
                                                device_scale_factor=dpr)
                try:
                    page = await ctx.new_page()
                    await _open_image_challenge(page, url, index)
                    rec.subject = (await page.locator(f"{CHALLENGE} .image-subject").inner_text()).strip()
                    if pipeline == "screenshot":
                        await _run_screenshot(page, ai, rec, viewport=viewport, mode=mode,
                                              model=model, fail_dir=fdir)
                    elif pipeline == "files":
                        await _run_files(page, ai, rec, url=url, out_dir=odir, mode=mode, model=model,
                                         fail_dir=fdir)
                    else:
                        await _run_dom(page, ai, rec, mode=mode, fail_dir=fdir)
                except Exception as e:  # keep the run going; the error is part of the result
                    rec.error = describe_exc(e)
                    await _save_error_state(page, rec, fdir)
                finally:
                    rec.seconds = round(time.perf_counter() - t0, 2)
                    try:
                        await ctx.close()
                    except Exception:  # noqa: BLE001 - cleanup only; the record is already final
                        pass
            counter[0] += 1
            if progress:
                progress(rec, counter[0], len(chosen))
            records.append(rec)

        await asyncio.gather(*[one(i, lbl) for i, lbl in chosen])
        await browser.close()
    records.sort(key=lambda r: r.index)
    return records, summarize(records)


async def run_replay(directory: str, *, url: str | None = None, concurrency: int = 4,
                     dpr: float = 1.0, headless: bool = True, progress=None,
                     note=print) -> tuple[list[Record], dict]:
    """Re-run saved solver JSONs: open each challenge and follow its actions. No model calls."""
    docs = []
    for path in sorted(Path(directory).glob("*.json")):
        try:
            doc = json.loads(path.read_text())
        except ValueError:
            continue
        if "challenge" in doc and "actions" in doc:
            docs.append(doc)
        else:
            note(f"skipping {path.name}: no 'challenge' + 'actions' (not made by `test --pipeline files`)")
    if not docs:
        raise FileNotFoundError(f"no replayable .json files in {directory}")
    fdir = Path(directory) / "failed_runs"
    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=headless)
        sem, records, counter = asyncio.Semaphore(concurrency), [], [0]

        async def one(doc: dict) -> None:
            ch = doc["challenge"]
            rec = Record(ch["index"], ch["label"], split_of(ch["label"]), target=doc["target"],
                         tiles=[a["tile"] for a in doc["actions"] if a["role"] == "tile"])
            page = None
            async with sem:
                t0 = time.perf_counter()
                vw, vh = doc["viewport"]
                ctx = await browser.new_context(viewport={"width": vw, "height": vh}, device_scale_factor=dpr)
                try:
                    page = await ctx.new_page()
                    await _open_image_challenge(page, url or ch["url"], ch["index"])
                    rec.subject = (await page.locator(f"{CHALLENGE} .image-subject").inner_text()).strip()
                    rec.passed, rec.clicked_ok = await execute_actions(page, doc)
                    await _record_outcome(page, rec, fdir)
                except Exception as e:
                    rec.error = describe_exc(e)
                    await _save_error_state(page, rec, fdir)
                finally:
                    rec.seconds = round(time.perf_counter() - t0, 2)
                    try:
                        await ctx.close()
                    except Exception:  # noqa: BLE001 - cleanup only; the record is already final
                        pass
            counter[0] += 1
            if progress:
                progress(rec, counter[0], len(docs))
            records.append(rec)

        await asyncio.gather(*[one(d) for d in docs])
        await browser.close()
    records.sort(key=lambda r: r.index)
    return records, summarize(records)


def replay_main() -> None:
    p = argparse.ArgumentParser(prog="captchasolver replay",
                                description="Follow saved solver JSONs on the live site (no model calls).")
    p.add_argument("directory", nargs="?", default=str(default_examples_dir()))
    p.add_argument("--url", help="override the URL stored in each JSON")
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--dpr", type=float, default=1.0)
    p.add_argument("--color", choices=["auto", "always", "never"], default="auto")
    p.add_argument("--headed", action="store_true")
    a = p.parse_args()
    color = use_color(a.color)
    records, summary = asyncio.run(run_replay(
        a.directory, url=a.url, concurrency=a.concurrency, dpr=a.dpr, headless=not a.headed,
        progress=lambda r, k, n: print(format_line(r, k, n, color), flush=True),
        note=lambda m: print(paint(m, "yellow", color), flush=True)))
    print(format_summary(records, summary, color))
    saved = [r.fail_image for r in records if r.fail_image]
    if saved:
        print(paint(f"\n📷 {len(saved)} failure image(s) in {Path(saved[0]).parent}", "dim", color))


def _viewport(text: str) -> tuple[int, int]:
    w, h = text.lower().split("x")
    return int(w), int(h)


def main() -> None:
    p = argparse.ArgumentParser(prog="captchasolver test", description=__doc__.split("\n\n")[0])
    p.add_argument("--count", type=int, default=100, help="challenges to run (default 100)")
    p.add_argument("--split", choices=["all", "train", "test"], default="all")
    p.add_argument("--seed", type=int, default=0, help="shuffle seed (same seed = same challenges)")
    p.add_argument("--concurrency", type=int, default=4, help="parallel browser pages")
    p.add_argument("--pipeline", choices=["screenshot", "dom", "files"], default="screenshot",
                   help="files: save screenshot + solver JSON to --out-dir, then act on the JSON")
    p.add_argument("--out-dir", help="where the files pipeline saves PNG+JSON (default: examples/)")
    p.add_argument("--viewport", type=_viewport, default=(1914, 1080), metavar="WxH")
    p.add_argument("--dpr", type=float, default=1.0, help="device pixel ratio, e.g. 1.5 or 2")
    p.add_argument("--mode", choices=["ask", "classify"], default="ask")
    p.add_argument("--model", default=VLM_MODEL)
    p.add_argument("--url", default=DEFAULT_URL)
    p.add_argument("--report", help="write full JSON results here")
    p.add_argument("--save-failures", metavar="DIR",
                   help="where failure images go (default: <examples>/failed_runs)")
    p.add_argument("--no-fail-images", action="store_true", help="don't save failure images")
    p.add_argument("--color", choices=["auto", "always", "never"], default="auto")
    p.add_argument("--headed", action="store_true")
    a = p.parse_args()
    color = use_color(a.color)

    records, summary = asyncio.run(run_tests(
        a.url, count=a.count, split=a.split, seed=a.seed, concurrency=a.concurrency,
        pipeline=a.pipeline, viewport=a.viewport, dpr=a.dpr, mode=a.mode, model=a.model,
        headless=not a.headed, fail_dir=a.save_failures, out_dir=a.out_dir,
        fail_images=not a.no_fail_images,
        progress=lambda r, k, n: print(format_line(r, k, n, color), flush=True),
        note=lambda m: print(paint(m, 'yellow', color), flush=True)))

    print(format_summary(records, summary, color))
    saved = [r.fail_image for r in records if r.fail_image]
    if saved:
        print(paint(f"\n📷 {len(saved)} failure image(s) in {Path(saved[0]).parent}", "dim", color))
    if a.report:
        Path(a.report).write_text(json.dumps({
            "config": {k: v for k, v in vars(a).items() if k != "report"} | {"viewport": list(a.viewport)},
            "summary": summary,
            "records": [asdict(r) | {"cause": failure_cause(r)} for r in records]}, indent=2) + "\n")
        print(paint(f"report: {a.report}", "dim", color))


if __name__ == "__main__":
    main()
