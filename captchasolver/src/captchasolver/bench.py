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
from contextlib import AsyncExitStack
from dataclasses import asdict, dataclass
from pathlib import Path

from playwright.async_api import Page, TimeoutError as PWTimeout, async_playwright

from . import diag
from .client import VLM_MODEL, WorkersAI, describe_exc
from .screen import annotate, solve_screenshot
from .web import (ACCEPTED, CHALLENGE, DEFAULT_URL, REJECTED, SiteDidNotAnswer, _open_image_challenge, _outcome,
                  solve_image_challenge)

# Chromium in a container: /dev/shm is tiny there, so keep its shared memory on disk instead (harmless elsewhere).
CHROMIUM_ARGS = ["--disable-dev-shm-usage"]
CHALLENGE_TIMEOUT_S = 240     # one challenge: open, screenshot, solve (10 model calls), click, wait for the answer
MAX_ATTEMPTS = 3              # a challenge whose browser died is retried on a fresh browser
MAX_BROWSER_RESTARTS = 4      # more than this in one run and Chromium is plainly not viable here: stop, say why


class BrowserKeepsCrashing(RuntimeError):
    """Chromium died again and again. Grinding through the remaining challenges would only waste time."""


def browser_died(e: BaseException) -> bool:
    """The browser/page went away under us (crash, OOM kill, restart), as opposed to a normal failure."""
    text = f"{type(e).__name__} {e}"
    return "TargetClosedError" in text or "has been closed" in text or "Target crashed" in text


async def _close(thing, timeout: float = 10.0) -> None:
    """Close a context/browser without ever letting a dead or hung one block the run."""
    if thing is None:
        return
    try:
        await asyncio.wait_for(thing.close(), timeout)
    except Exception:  # noqa: BLE001 - cleanup only
        pass


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


async def _save_after(page: Page, path: Path) -> str:
    """Screenshot the page once the site has answered (the success page after an accept, the error state
    after a reject). Waits for a redirect in flight to finish, so it is not a blank page. Never raises."""
    try:
        await page.wait_for_load_state("load", timeout=8000)
    except Exception:  # noqa: BLE001 - a slow load still deserves a screenshot
        pass
    try:
        await page.screenshot(path=str(path))
    except Exception:  # noqa: BLE001
        return ""
    return page.url


def bundle_zip(records: list, out_dir: Path, zip_path: Path, config: dict, summary: dict) -> Path:
    """One folder per challenge (before.png, solver.json, plan.png, after.png) + report.json + README.txt."""
    import zipfile
    parts = {".png": "before.png", ".json": "solver.json", ".plan.png": "plan.png", ".after.png": "after.png"}
    zip_path = Path(zip_path).expanduser()
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for r in sorted(records, key=lambda r: r.index):
            stem = f"{r.index:03d}-{r.label}"
            for suffix, name in parts.items():
                f = out_dir / f"{stem}{suffix}"
                if f.exists():
                    z.write(f, f"{stem}/{name}")
        z.writestr("report.json", json.dumps({
            "config": config, "summary": summary,
            "records": [asdict(r) | {"cause": failure_cause(r)} for r in sorted(records, key=lambda r: r.index)]}, indent=2))
        z.writestr("README.txt", (
            "captchasolver example run\n\n"
            "One folder per challenge:\n"
            "  before.png   what the solver saw (the full-screen screenshot)\n"
            "  solver.json  what it decided: target, tiles, click positions and the ordered `actions`;\n"
            "               `outcome` records what the site answered after those clicks\n"
            "  plan.png     the plan drawn on before.png (green = picked tiles, orange = Verify)\n"
            "  after.png    the page after the site answered (success page, or the error state)\n"
            "report.json    summary and one record per challenge\n"))
    return zip_path


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
    annotate(img, lay, res).save(stem.with_suffix(".plan.png"))      # what the solver planned, drawn on its screenshot
    # act from the saved file, exactly as a robot reading it later would
    rec.passed, rec.clicked_ok = await execute_actions(
        page, json.loads(stem.with_suffix(".json").read_text()))
    final_url = await _save_after(page, stem.with_suffix(".after.png"))   # the page once the site has answered
    doc["outcome"] = {"passed": rec.passed, "tiles_clicked_as_planned": rec.clicked_ok, "final_url": final_url}
    stem.with_suffix(".json").write_text(json.dumps(doc, indent=2) + "\n")
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
                    fail_images: bool = True, progress=None, note=print,
                    ai: WorkersAI | None = None) -> tuple[list[Record], dict]:
    """Run the demo-site challenges. Pass `ai` to use caller-supplied credentials (the web
    server does); otherwise a client is built from the environment. Cancel-safe: cancelling the
    task closes the browser and any client created here."""
    odir = Path(out_dir) if out_dir else default_examples_dir()
    # failed runs always land in <examples>/failed_runs unless told otherwise
    fdir = (Path(fail_dir) if fail_dir else odir / "failed_runs") if fail_images else None
    if pipeline == "files":
        odir.mkdir(parents=True, exist_ok=True)
    async with AsyncExitStack() as stack:
        if ai is None:
            ai = await stack.enter_async_context(WorkersAI())
        pw = await stack.enter_async_context(async_playwright())
        lock, current, restarts = asyncio.Lock(), {"browser": None}, [0]

        async def get_browser():
            """The shared browser, relaunched if it died (crash, OOM kill, container hiccup)."""
            async with lock:
                b = current["browser"]
                if b is None or not b.is_connected():
                    if b is not None:
                        restarts[0] += 1
                        if restarts[0] > MAX_BROWSER_RESTARTS:
                            if restarts[0] == MAX_BROWSER_RESTARTS + 1:
                                note("Chromium keeps crashing, so the run is stopping early.")
                            raise BrowserKeepsCrashing(
                                "Chromium keeps crashing here: open /api/diagnostics?stress=1 to see why, or lower "
                                f"'Parallel pages'. (It crashed {restarts[0] - 1} times; {diag.hints()}.)")
                        note(f"Chromium closed unexpectedly: restarting it ({diag.hints()}).")
                    b = current["browser"] = await pw.chromium.launch(headless=headless, args=CHROMIUM_ARGS, timeout=20000)
                return b

        browser = await get_browser()
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

        async def challenge(page, rec: Record, index: int) -> None:
            await _open_image_challenge(page, url, index)
            rec.subject = (await page.locator(f"{CHALLENGE} .image-subject").inner_text()).strip()
            if pipeline == "screenshot":
                await _run_screenshot(page, ai, rec, viewport=viewport, mode=mode, model=model, fail_dir=fdir)
            elif pipeline == "files":
                await _run_files(page, ai, rec, url=url, out_dir=odir, mode=mode, model=model, fail_dir=fdir)
            else:
                await _run_dom(page, ai, rec, mode=mode, fail_dir=fdir)

        async def one(index: int, label: str) -> None:
            async with sem:
                t0 = time.perf_counter()
                for attempt in range(1, MAX_ATTEMPTS + 1):
                    rec = Record(index, label, split_of(label))     # a retry starts from a clean record
                    ctx = page = None
                    try:
                        ctx = await (await get_browser()).new_context(
                            viewport={"width": viewport[0], "height": viewport[1]}, device_scale_factor=dpr)
                        page = await ctx.new_page()
                        await asyncio.wait_for(challenge(page, rec, index), CHALLENGE_TIMEOUT_S)
                        break
                    except asyncio.TimeoutError:                    # nothing may hang a whole run
                        rec.error = f"TimeoutError: the challenge did not finish within {CHALLENGE_TIMEOUT_S} s"
                        await _save_error_state(page, rec, fdir)
                        break
                    except Exception as e:  # keep the run going; the error is part of the result
                        if browser_died(e) and attempt < MAX_ATTEMPTS:
                            note(f"Chromium died during {label}; trying it again ({attempt}/{MAX_ATTEMPTS - 1}).")
                            continue
                        rec.error = describe_exc(e, 400)
                        await _save_error_state(page, rec, fdir)
                        break
                    finally:
                        await _close(ctx)
                rec.seconds = round(time.perf_counter() - t0, 2)
            counter[0] += 1
            if progress:
                progress(rec, counter[0], len(chosen))
            records.append(rec)

        await asyncio.gather(*[one(i, lbl) for i, lbl in chosen])
        await _close(current["browser"])
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
                    rec.error = describe_exc(e, 400)
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
    saved = [r.fail_image for r in records if r.fail_image and Path(r.fail_image).exists()]
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
    p.add_argument("--zip", metavar="PATH", help="write a zip of this run's screenshots and JSON (implies --pipeline "
                                                  "files, and keeps the files out of examples/)")
    p.add_argument("--color", choices=["auto", "always", "never"], default="auto")
    p.add_argument("--headed", action="store_true")
    a = p.parse_args()
    color = use_color(a.color)
    tmp = None
    if a.zip:
        import tempfile
        a.pipeline = "files"
        if not a.out_dir:
            tmp = tempfile.TemporaryDirectory(prefix="captchasolver-zip-")
            a.out_dir = tmp.name

    records, summary = asyncio.run(run_tests(
        a.url, count=a.count, split=a.split, seed=a.seed, concurrency=a.concurrency,
        pipeline=a.pipeline, viewport=a.viewport, dpr=a.dpr, mode=a.mode, model=a.model,
        headless=not a.headed, fail_dir=a.save_failures, out_dir=a.out_dir,
        fail_images=not a.no_fail_images,
        progress=lambda r, k, n: print(format_line(r, k, n, color), flush=True),
        note=lambda m: print(paint(m, 'yellow', color), flush=True)))

    print(format_summary(records, summary, color))
    if a.zip:
        config = {k: v for k, v in vars(a).items() if k not in ("zip", "report", "out_dir")} | {"viewport": list(a.viewport)}
        done = bundle_zip(records, Path(a.out_dir), Path(a.zip), config, summary)
        print(paint(f"\n📦 {done}  ({done.stat().st_size // 1024} KB, {len(records)} challenge folders)", "bold", color))
        if tmp:
            tmp.cleanup()
    saved = [r.fail_image for r in records if r.fail_image and Path(r.fail_image).exists()]
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
