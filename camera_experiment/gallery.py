"""Run all photo variants through the deployed solver and open an offline gallery."""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import cast
from urllib.parse import quote

from PIL import Image, ImageOps
from typed_errs import Err, Ok, Result, Some

from camera_experiment.geometry import Failure, attempt
from camera_experiment.remote import DEFAULT_ENDPOINT
from camera_experiment.run import VARIANTS, Options, Record, run


def read_json(path: Path) -> Result[Record, Failure]:
    return attempt(f"read {path}", lambda: cast(Record, json.loads(path.read_text())))


def verdict(record: Record, expected: dict[str, Record]) -> tuple[bool, str]:
    """Separate a returned plan from an exact match to the known photo labels."""
    truth = expected.get(Path(cast(str, record["source"])).name, {})
    status = record["status"]
    if status == "no_dialog" and truth.get("visible") is False:
        return True, "Correctly found no CAPTCHA"
    if status in ("layout_only", "solved") and truth.get("visible") is False:
        return False, "False positive: expected no CAPTCHA"
    if status == "layout_only":
        return True, "Screen and click locations found"
    if status == "solved":
        plan = cast(Record, record["solver"])
        if truth.get("visible") is True:
            if plan["target"] != truth["target"] or plan["tiles"] != truth["tiles"]:
                return False, f"Expected {truth['target']} {truth['tiles']}; got {plan['target']} {plan['tiles']}"
            return True, "Exact tile match"
        return True, "Solver returned a plan; labels not available"
    return False, cast(str, record.get("reason", status))


def show_progress(record: Record, expected: dict[str, Record]) -> Result[None, Failure]:
    success, label = verdict(record, expected)
    return attempt("print result", lambda: print(
        "🤓" if success else "🥀", Path(cast(str, record["source"])).name,
        record["variant"], label, flush=True))


def thumbnail(source: Path, output: Path) -> Result[str, Failure]:
    return attempt(f"preview {source.name}", lambda: _thumbnail(source, output))


def _thumbnail(source: Path, output: Path) -> str:
    with Image.open(source) as original:
        image = ImageOps.exif_transpose(original).convert("RGB")
        image.thumbnail((1000, 800))
        image.save(output, "JPEG", quality=85)
    return quote("thumbs/" + output.name)


def render_gallery(out: Path, expected: dict[str, Record], endpoint: str) -> Result[Path, Failure]:
    report = read_json(out / "report.json")
    if isinstance(report, Err):
        return report
    created = attempt("create previews", lambda: (out / "thumbs").mkdir(exist_ok=True))
    if isinstance(created, Err):
        return created
    assets = out / "assets"
    copied = attempt("copy report font", lambda: (
        assets.mkdir(exist_ok=True), shutil.copyfile(
            Path(__file__).parent.parent / "captcha/public/fonts/dm-sans.woff2",
            assets / "dm-sans.woff2")))
    if isinstance(copied, Err):
        return copied
    items: list[Record] = []
    for record in cast(list[Record], report.value["records"]):
        source = Path(cast(str, record["source"]))
        variant = cast(str, record["variant"])
        folder = out / source.stem / variant
        images: dict[str, str] = {}
        candidates = {"photo": source, "damaged": folder / "photo.jpg", "screen": folder / "screen.png",
                      "crop": folder / "challenge.png", "plan": folder / "plan.png",
                      "source_plan": folder / "source-plan.jpg"}
        if record["status"] == "geometry_error":
            candidates = {"photo": source}
        for view, path in candidates.items():
            exists = attempt("find preview", path.is_file)
            if isinstance(exists, Err):
                return exists
            if not exists.value:
                continue
            preview = thumbnail(path, out / "thumbs" / f"{source.stem}-{variant}-{view}.jpg")
            if isinstance(preview, Err):
                return preview
            images[view] = preview.value
        success, label = verdict(record, expected)
        plan = cast(Record, record.get("solver", {}))
        links: dict[str, str] = {}
        for name in ("geometry.json", "targets.json", "actions.json"):
            exists = attempt("find JSON", lambda name=name, folder=folder: (folder / name).is_file())
            if isinstance(exists, Err):
                return exists
            if exists.value:
                links[name] = quote(f"{source.stem}/{variant}/{name}")
        items.append({"source": source.name, "variant": variant, "success": success,
                      "label": label, "status": record["status"], "images": images, "links": links,
                      "frame": cast(Record, record.get("coordinate_frame", {})).get("kind", "not found"),
                      "target": plan.get("target", ""), "tiles": plan.get("tiles", []),
                      "actions": plan.get("actions", record.get("targets", []))})
    template = attempt("read gallery template", lambda: Path(__file__).with_name("gallery.html").read_text())
    if isinstance(template, Err):
        return template
    payload = attempt("encode gallery", lambda: json.dumps(
        {"items": items, "endpoint": endpoint, "model": report.value["model"],
         "generated": datetime.now(UTC).isoformat(timespec="seconds")}, ensure_ascii=False).replace("<", "\\u003c"))
    if isinstance(payload, Err):
        return payload
    destination = out / "index.html"
    saved = attempt("write gallery", lambda: destination.write_text(template.value.replace("__DATA__", payload.value)))
    return Ok(destination) if isinstance(saved, Ok) else saved


async def compare(args: Options) -> Result[Path, Failure]:
    labels = read_json(Path(__file__).with_name("expected.json"))
    if isinstance(labels, Err):
        return labels
    expected = cast(dict[str, Record], labels.value)
    result = await run(args, lambda record: show_progress(record, expected))
    if isinstance(result, Err):
        return result
    return render_gallery(args.out, expected, DEFAULT_ENDPOINT)


def main() -> Result[int, Failure]:
    parser = argparse.ArgumentParser(description=__doc__)
    _ = parser.add_argument("images", nargs="?", type=Path, default=Path.home() / "Downloads/asodkasodkasd")
    _ = parser.add_argument("--out", type=Path, default=Path(__file__).parent / "results-all")
    _ = parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    _ = parser.add_argument("--no-solve", action="store_true", help="only normalize and locate clicks, without model calls")
    _ = parser.add_argument("--no-open", action="store_true", help="write the report without launching the browser")
    parsed = parser.parse_args()
    if not cast(bool, parsed.no_solve) and not all(os.environ.get(key) for key in ("CLOUDFLARE_ACCOUNT_ID", "CLOUDFLARE_API_TOKEN")):
        return Err(Failure.OPERATION, context_msg="Set CLOUDFLARE_ACCOUNT_ID and CLOUDFLARE_API_TOKEN, or put their exports in ~/.config/captchasolver/credentials.env. Use --no-solve for coordinates only.")
    out = cast(Path, parsed.out)
    created = attempt("create results directory", lambda: out.mkdir(parents=True, exist_ok=True))
    if isinstance(created, Err):
        return created
    options = Options(cast(Path, parsed.images), out, 1920, 1200, cast(list[str], parsed.variants),
                      42, not cast(bool, parsed.no_solve), "@cf/google/gemma-4-26b-a4b-it",
                      endpoint=Some(DEFAULT_ENDPOINT))
    result = attempt("run all photos", lambda: asyncio.run(compare(options)))
    if isinstance(result, Err):
        return result
    if isinstance(result.value, Err):
        return result.value
    path = result.value.value
    shown = attempt("print report location", lambda: print(f"Report: {path.resolve()}", flush=True))
    if isinstance(shown, Err):
        return shown
    if not cast(bool, parsed.no_open):
        opened = attempt("open report; open the printed path manually if needed", lambda: subprocess.run(
            ["xdg-open", str(path.resolve())], check=True))
        if isinstance(opened, Err):
            return opened
    report = read_json(out / "report.json")
    if isinstance(report, Err):
        return report
    return Ok(1 if any(record["status"] == "solver_error"
                       for record in cast(list[Record], report.value["records"])) else 0)


if __name__ == "__main__":
    try:
        outcome = main()
        if isinstance(outcome, Err):
            print("🥀", outcome.context_msg, file=sys.stderr)
            sys.exit(130 if outcome.error == Failure.INTERRUPTED else 1)
        sys.exit(outcome.value)
    except (Exception, KeyboardInterrupt) as error:  # noqa: BLE001 - last-resort process boundary
        print(f"🥀 unexpected process boundary failure: {type(error).__name__}: {error}", file=sys.stderr)
        sys.exit(1)
