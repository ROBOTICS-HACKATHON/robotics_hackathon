"""`captchasolver compare`: benchmark several Workers AI vision models on the same challenges.

1. Preflight: every model reads a small test image on YOUR account. Request quirks (thinking
   switches, temperature) are discovered by trying variants. Models that fail are skipped, with the reason.
2. Each surviving model runs the same seeded set of challenges through the full pipeline.
3. Ranked table + which challenges every model failed.
"""
from __future__ import annotations

import argparse
import asyncio
import io
import json
import time
from dataclasses import asdict, dataclass

from PIL import Image, ImageDraw, ImageFont

from .bench import Record, _viewport, paint, run_tests, use_color
from .client import MODEL_CONFIG, ModelConfig, WorkersAI, describe_exc, save_model_config, vlm_query
from .web import DEFAULT_URL

# (model id, short note, $ per M input tokens from Cloudflare's docs; None = not confirmed)
CANDIDATES: list[tuple[str, str, float | None]] = [
    ("@cf/moonshotai/kimi-k2.5", "frontier-scale, native vision, reasoning", 0.60),
    ("@cf/google/gemma-4-26b-a4b-it", "DEFAULT: newest Gemma, MoE 26B/4B active, vision+thinking", 0.10),
    ("@cf/mistralai/mistral-small-3.1-24b-instruct", "24B, 'state-of-the-art vision understanding'", 0.351),
    ("@cf/meta/llama-4-scout-17b-16e-instruct", "previous default, natively multimodal MoE", 0.27),
    ("@cf/meta/llama-3.2-11b-vision-instruct", "older, small vision baseline", 0.0485),
]
PRICE_IN = {m: p for m, _, p in CANDIDATES}

# request variants tried in order until one answers correctly
VARIANTS = [
    ModelConfig({"chat_template_kwargs": {"enable_thinking": False}}, True),
    ModelConfig({"chat_template_kwargs": {"enable_thinking": False}}, False),
    ModelConfig({}, True),
    ModelConfig({}, False),
]


@dataclass
class Preflight:
    model: str
    ok: bool
    answer: str = ""
    seconds: float = 0.0
    variant: int | None = None
    error: str = ""


def _probe_image() -> bytes:
    img = Image.new("RGB", (360, 160), "white")
    try:
        font = ImageFont.load_default(size=56)
    except TypeError:  # very old Pillow
        font = ImageFont.load_default()
    ImageDraw.Draw(img).text((20, 45), "HELLO 42", fill="black", font=font)
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


def _correct(answer: str) -> bool:
    a = answer.lower()
    return "hello" in a and "42" in a


async def preflight_one(ai: WorkersAI, model: str) -> Preflight:
    img, last = _probe_image(), ""
    t0 = time.perf_counter()
    for i, cfg in enumerate(VARIANTS):
        MODEL_CONFIG[model] = cfg
        try:
            ans = await vlm_query(ai, img, "What text is shown? Reply with only the text.",
                                  model=model, max_tokens=32)
        except Exception as e:  # noqa: BLE001 - any failure just means "try the next variant"
            last = describe_exc(e, 200)
            continue
        if _correct(ans):
            return Preflight(model, True, ans.strip()[:60], round(time.perf_counter() - t0, 2), i)
        last = f"wrong answer {ans.strip()[:60]!r}"
    MODEL_CONFIG.pop(model, None)
    return Preflight(model, False, seconds=round(time.perf_counter() - t0, 2), error=last)


def analyse(results: dict[str, list[Record]]) -> dict:
    """Cross-model view: which challenges nobody solved, how many anyone solved."""
    labels = {r.label for rs in results.values() for r in rs}
    passed = {m: {r.label for r in rs if r.passed} for m, rs in results.items()}
    solved = set().union(*passed.values()) if passed else set()
    return {"challenges": len(labels), "solved_by_any": len(solved),
            "failed_by_all": sorted(labels - solved)}


def rank(summaries: dict[str, dict]) -> list[str]:
    return sorted(summaries, key=lambda m: (-summaries[m]["accuracy"], summaries[m]["mean_seconds"]))


def format_table(summaries: dict[str, dict], skipped: list[Preflight], analysis: dict,
                 color: bool) -> str:
    out = ["", paint("═" * 16 + " Model comparison " + "═" * 16, "bold", color)]
    head = f"  {'#':<3}{'model':<46}{'passed':>9}{'acc':>8}{'banner':>8}{'s/chal':>8}{'$/M in':>8}"
    out.append(paint(head, "dim", color))
    medals = ["🥇", "🥈", "🥉"]
    for i, m in enumerate(rank(summaries)):
        s = summaries[m]
        banner = f"{s['banner_read_accuracy']:.0%}" if s["banner_read_accuracy"] is not None else "-"
        price = PRICE_IN.get(m)
        row = (f"{s['passed']}/{s['total']}".rjust(9) + f"{s['accuracy']:.1%}".rjust(8) + banner.rjust(8)
               + f"{s['mean_seconds']:.1f}".rjust(8) + (f"${price}" if price is not None else "?").rjust(8))
        style = "green" if s["accuracy"] >= 0.9 else "yellow" if s["accuracy"] >= 0.6 else "red"
        out.append(f"  {(medals[i] if i < 3 else '  '):<3}{m.removeprefix('@cf/'):<46}" + paint(row, style, color))
    for p in skipped:
        out.append(f"  ⏭️  {p.model.removeprefix('@cf/'):<46}" + paint(f"skipped: {p.error}", "yellow", color))
    out.append("")
    out.append(f"  solved by at least one model: {analysis['solved_by_any']}/{analysis['challenges']}")
    fa = analysis["failed_by_all"]
    if fa:
        out.append(paint(f"  ❌ failed by EVERY model ({len(fa)}): " + ", ".join(fa[:12])
                         + (" …" if len(fa) > 12 else ""), "red", color))
        out.append(paint("     (worth a look: ambiguous images, or something systematic)", "dim", color))
    else:
        out.append(paint("  ✅ every challenge was solved by at least one model", "green", color))
    return "\n".join(out)


async def run_compare(models: list[str], *, preflight_only: bool, color: bool, report: str | None,
                      **kw) -> int:
    say = lambda m="": print(m, flush=True)  # noqa: E731
    say(paint("Preflight — can each model read an image on this account?", "bold", color))
    async with WorkersAI() as ai:
        pre = await asyncio.gather(*[preflight_one(ai, m) for m in models])
    for p in pre:
        if p.ok:
            say(f"  ✅ {p.model.removeprefix('@cf/'):<46} {p.seconds:>5.1f}s  read {p.answer!r}  "
                + paint(f"(variant {p.variant}: thinking={'off' if 'chat_template_kwargs' in VARIANTS[p.variant].extra else 'default'}"
                        f"{'' if VARIANTS[p.variant].send_temperature else ', no temperature'})", "dim", color))
        else:
            say(f"  ❌ {p.model.removeprefix('@cf/'):<46} " + paint(p.error, "yellow", color))
    ready = [p.model for p in pre if p.ok]
    say(paint(f"  settings saved to {save_model_config()}", "dim", color))
    if preflight_only or not ready:
        if not ready:
            say(paint("No model passed preflight.", "red", color))
        return 0 if ready else 1

    results: dict[str, list[Record]] = {}
    summaries: dict[str, dict] = {}
    for m in ready:
        say(f"\n{paint('▶ ' + m.removeprefix('@cf/'), 'bold', color)}")
        mark = lambda r, k, n: print("💥" if r.error else "✅" if r.passed else "❌",  # noqa: E731
                                     end="" if k < n else "\n", flush=True)
        records, summaries[m] = await run_tests(model=m, progress=mark, note=say, **kw)
        results[m] = records
        s = summaries[m]
        say(f"  {s['passed']}/{s['total']} passed ({s['accuracy']:.1%}) · {s['failed']} failed · "
            f"{s['errors']} errors · {s['mean_seconds']}s/challenge")

    analysis = analyse(results)
    say(format_table(summaries, [p for p in pre if not p.ok], analysis, color))
    if report:
        with open(report, "w") as f:
            json.dump({"preflight": [asdict(p) for p in pre], "summaries": summaries,
                       "analysis": analysis,
                       "records": {m: [asdict(r) for r in rs] for m, rs in results.items()}}, f, indent=2)
        say(paint(f"report: {report}", "dim", color))
    return 0


def main() -> None:
    p = argparse.ArgumentParser(prog="captchasolver compare", description=__doc__.split("\n\n")[0])
    p.add_argument("--models", help="comma-separated model ids (default: the 5 built-in candidates)")
    p.add_argument("--count", type=int, default=30, help="challenges PER MODEL (default 30)")
    p.add_argument("--split", choices=["all", "train", "test"], default="all")
    p.add_argument("--seed", type=int, default=0, help="same seed = every model sees the same challenges")
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--pipeline", choices=["screenshot", "dom"], default="screenshot")
    p.add_argument("--viewport", type=_viewport, default=(1914, 1080), metavar="WxH")
    p.add_argument("--dpr", type=float, default=1.0)
    p.add_argument("--url", default=DEFAULT_URL)
    p.add_argument("--preflight-only", action="store_true", help="only check which models respond")
    p.add_argument("--report", help="write full JSON results here")
    p.add_argument("--color", choices=["auto", "always", "never"], default="auto")
    a = p.parse_args()
    models = [m.strip() for m in a.models.split(",")] if a.models else [m for m, _, _ in CANDIDATES]
    raise SystemExit(asyncio.run(run_compare(
        models, preflight_only=a.preflight_only, color=use_color(a.color), report=a.report,
        url=a.url, count=a.count, split=a.split, seed=a.seed, concurrency=a.concurrency,
        pipeline=a.pipeline, viewport=a.viewport, dpr=a.dpr, mode="ask")))


if __name__ == "__main__":
    main()
