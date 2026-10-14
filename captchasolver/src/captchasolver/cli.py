from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from PIL import Image

from . import WorkersAI, solve_grid, solve_text, tile_centers
from .batch import solve_dir
from .capture import grab_screen
from .screen import annotate, solve_screenshot


async def _run(args: argparse.Namespace) -> None:
    if args.kind == "grab":
        data = grab_screen()
        Path(args.out).write_bytes(data)
        print(f"saved {args.out}: {Image.open(args.out).size[0]}x{Image.open(args.out).size[1]}")
        return
    if args.kind == "live":
        shot = grab_screen()
        if args.save:
            Path(args.save).write_bytes(shot)
        async with WorkersAI() as ai:
            kw = {"model": args.model} if args.model else {}
            res, img, lay = await solve_screenshot(ai, shot, target=args.target, **kw)
        if args.debug:
            annotate(img, lay, res).save(args.debug)
        print(json.dumps(res, indent=2))
        return
    if args.kind == "dir":
        grid = tuple(int(x) for x in args.grid.split("x"))
        async with WorkersAI() as ai:
            kw = {"model": args.model} if args.model else {}
            res = await solve_dir(ai, args.directory, args.target, layout=args.layout,
                                  grid=grid, mode=args.mode, debug_dir=args.debug_dir, **kw)
        text = json.dumps(res, indent=2)
        if args.out:
            Path(args.out).write_text(text + "\n")
        print(text)
        return
    image = Path(args.image).read_bytes()
    async with WorkersAI() as ai:
        if args.kind == "text":
            kw = {"model": args.model} if args.model else {}
            print(await solve_text(ai, image, length=args.length, votes=args.votes, **kw))
            return
        rows, cols = (int(x) for x in args.grid.split("x"))
        kw = {"model": args.model} if args.model else {}
        tiles = await solve_grid(ai, image, args.target, rows=rows, cols=cols,
                                 mode=args.mode, **kw)
        centers = tile_centers(Image.open(args.image).size, rows, cols)
        print(json.dumps({"tiles": tiles, "click": [centers[i] for i in tiles]}))


def main() -> None:
    import sys
    # `captchasolver probe` / `captchasolver web ...` forward to the other entry points
    if len(sys.argv) > 1 and sys.argv[1] in ("probe", "web", "test", "compare", "replay"):
        cmd = sys.argv.pop(1)
        if cmd == "replay":
            from .bench import replay_main
            return replay_main()
        if cmd == "compare":
            from .compare import main as compare_main
            return compare_main()
        if cmd == "test":
            from .bench import main as test_main
            return test_main()
        if cmd == "probe":
            from .probe import main as probe_main
            return probe_main()
        from .web import main as web_main
        return web_main()
    p = argparse.ArgumentParser(prog="captchasolver")
    sub = p.add_subparsers(dest="kind", required=True)
    t = sub.add_parser("text", help="distorted text captcha")
    t.add_argument("image")
    t.add_argument("--length", type=int)
    t.add_argument("--votes", type=int, default=1)
    t.add_argument("--model", help="override the Workers AI model id")
    g = sub.add_parser("grid", help="click-the-squares captcha")
    g.add_argument("image")
    g.add_argument("--target", required=True, help='e.g. "traffic light"')
    g.add_argument("--grid", default="3x3")
    g.add_argument("--mode", choices=["detect", "ask", "classify"], default="detect")
    g.add_argument("--model", help="override the detect model id")
    gr = sub.add_parser("grab", help="capture the screen to a PNG (tests capture only)")
    gr.add_argument("out", nargs="?", default="screen.png")
    lv = sub.add_parser("live", help="capture the screen now and solve the challenge on it")
    lv.add_argument("--target", help="default: read from the banner")
    lv.add_argument("--model", help="override the Workers AI model id")
    lv.add_argument("--save", help="also keep the captured screenshot here")
    lv.add_argument("--debug", help="write an annotated overlay PNG here")
    d = sub.add_parser("dir", help="folder of PNGs -> JSON of what to click")
    d.add_argument("directory")
    d.add_argument("--target", help='e.g. "cats"; default: read from the banner '
                                    '(screenshot layout only)')
    d.add_argument("--layout", choices=["screenshot", "grid", "tiles"], default="screenshot",
                   help="screenshot: full page with the dialog (default); "
                        "grid: PNG is just the grid; tiles: each PNG is one tile")
    d.add_argument("--grid", default="3x3", help="RxC tile layout (default 3x3)")
    d.add_argument("--debug-dir", help="write annotated overlays here (screenshot layout)")
    d.add_argument("--mode", choices=["ask", "classify"], default="ask")
    d.add_argument("--model", help="override the Workers AI model id")
    d.add_argument("--out", help="also write the JSON to this file")
    asyncio.run(_run(p.parse_args()))


if __name__ == "__main__":
    main()
