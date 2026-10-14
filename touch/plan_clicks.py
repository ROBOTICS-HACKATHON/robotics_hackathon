"""Photo of the screen in, "click here next" out, as relative screen coordinates (u, v from 0 to 1).

This is what the Playwright steps do on the site (open the card, tick the checkbox, select tiles, Verify),
but driven only by what a camera sees. It looks at ONE photo and says what to click next:

  python plan_clicks.py --image photo.jpg                # raw camera photo of the screen (4 markers visible)
  python plan_clicks.py --image screen.jpg --straight    # already straightened (outputs/screen.jpg from ./lr see)
  python plan_clicks.py --camera /dev/video4             # take the photo now
  python plan_clicks.py --image photo.jpg --tiles 1,3,8  # no solver call: use these tiles (offline planning)

Output (stdout, JSON; also --out file.json, and --annotate plan.png draws the clicks on the photo):

  {"state": "grid", "done": false, "target": "bicycles",
   "clicks": [{"seq": 1, "role": "tile",   "tile": 1, "u": 0.5, "v": 0.3708, "x": 640, "y": 297},
              {"seq": 2, "role": "tile",   "tile": 3, "u": 0.3966, "v": 0.5393, ...}],
   "tiles_wanted": [1, 3, 8], "tiles_selected": [], "size": [1280, 800], "note": "..."}

What it returns for each page (the same stateless rule the robot loop uses):
  choice     the reCAPTCHA card            checkbox  the "I'm not a robot" box   (taught positions)
  grid       the tiles that still differ from the solver's answer. A tap toggles a tile, so tiles that are
             already right are not listed. When the selection is already right it returns Verify only.
             (--full also appends Verify after the tile clicks, as Playwright would do in one go.)
  grid+error "Incorrect selection" is showing: nothing to click, done = false, state "grid_error"
  verified   "Access granted": done = true (--next returns the "Start again" button for the next challenge)
  unknown    nothing to click: take another photo (loading, arm in the way, page changing)

Coordinates are relative to the area between the outer corners of the 4 markers (what see.py straightens
to), the same frame the arm is calibrated in: pass (u, v) straight to `./lr tap u v` or Tapper.tap(u, v).

Needs for choice / checkbox / verified: `./lr captcha --learn-layout` once (touch/site_layout.json), or
--demo-layout for the committed screenshots of the site at a 1280x800 viewport (only valid for that).
The grid needs the solver (CLOUDFLARE_* in the environment) unless --tiles is given.
"""

import argparse
import json
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pagestate import LAYOUT_PATH, PAGES, SiteLayout, check_presses_against_grid, classify, read_selection, selected_set  # noqa: E402

HERE = Path(__file__).resolve().parent


def demo_layout():
    """Pages taught from the committed screenshots (tests/shots) of the site at a 1280x800 viewport."""
    shots = HERE / "tests" / "shots"
    pos = json.loads((shots / "positions.json").read_text())
    lay = SiteLayout({}, "/nonexistent/demo_layout.json")
    for name in PAGES:
        lay.teach(name, cv2.imread(str(shots / f"{name}.png")), pos[name])
    return lay


def _click(seq, role, u, v, size, tile=None):
    c = {"seq": seq, "role": role}
    if tile is not None:
        c["tile"] = int(tile)
    c.update(u=round(float(u), 5), v=round(float(v), 5), x=int(round(u * size[0])), y=int(round(v * size[1])))
    return c


def plan_clicks(img, layout=None, solver=None, target=None, model=None, tiles=None, full=False, next_challenge=False):
    """The next clicks for this photo. `solver(img, target, model)` returns the /api/robot/solve answer (or raises
    solve.SolverError); `tiles` skips the solver. Returns a JSON-able dict (never raises for page problems)."""
    h, w = img.shape[:2]
    size = [w, h]
    out = {"state": "unknown", "done": False, "clicks": [], "size": size, "note": ""}
    st = classify(img, layout)
    out["state"] = st.kind
    out["page"] = str(st)

    if st.kind in ("choice", "checkbox"):
        spot = layout.tap(st.kind) if layout else None
        if spot is None:
            out["note"] = f"'{st.kind}' page, but its tap position is not taught (./lr captcha --learn-layout)"
        else:
            out["clicks"] = [_click(1, st.kind, *spot, size)]
            out["note"] = {"choice": "click the reCAPTCHA card, then photograph again",
                           "checkbox": "click the checkbox, wait for the images, then photograph again"}[st.kind]
        return out

    if st.kind == "verified":
        out["done"] = True
        out["note"] = "Access granted"
        if next_challenge:
            spot = layout.tap("verified") if layout else None
            if spot:
                out["clicks"] = [_click(1, "start_again", *spot, size)]
                out["note"] += "; click 'Start again' for the next challenge"
        return out

    if st.kind != "grid":
        out["note"] = "no known page in this photo: take another one (loading, arm in the way, markers hidden?)"
        return out

    grid = st.grid
    if st.error:
        out["state"] = "grid_error"
        out["note"] = "the site shows 'Incorrect selection'; nothing to click (reload or reset the dialog)"
        return out

    have = selected_set(read_selection(img, grid))
    out["tiles_selected"] = sorted(have)
    if tiles is not None:
        want, order = set(tiles), list(tiles)
        verify = (grid.verify[0] / w, grid.verify[1] / h)
        uv = {t: grid.tile_center_uv(t) for t in want}
        out["target"] = target or ""
    else:
        from solve import SolverError
        try:
            res = solver(img, target, model)
            worst, ok = check_presses_against_grid(res["presses"], grid)
            if not ok:
                raise SolverError("layout_mismatch", f"solver tiles are {worst:.2f} tiles away from the dialog in this photo",
                                  retry=True)
        except SolverError as e:
            out.update(error=e.error, retry=e.retry, note=f"solver: {e.error}: {e.detail}")
            return out
        order = [p["tile"] for p in res["presses"] if p["role"] == "tile"]
        want = set(order)
        uv = {p["tile"]: (p["x_norm"], p["y_norm"]) for p in res["presses"] if p["role"] == "tile"}
        vp = res["presses"][-1]
        verify = (vp["x_norm"], vp["y_norm"])
        out["target"] = res.get("target", "")

    out["tiles_wanted"] = sorted(want)
    todo = [t for t in order if t not in have] + sorted(have - want)      # tick the missing, untick the extra
    clicks = []
    for t in todo:
        u, v = uv.get(t) or grid.tile_center_uv(t)
        clicks.append(_click(len(clicks) + 1, "tile" if t in want else "untick", u, v, size, t))
    if not todo:
        clicks.append(_click(1, "verify", *verify, size))
        out["note"] = "the right tiles are selected: click Verify"
    elif full:
        clicks.append(_click(len(clicks) + 1, "verify", *verify, size))
        out["note"] = "tile clicks, then Verify (--full)"
    else:
        out["note"] = "click these tiles, photograph again and rerun: Verify is returned once the selection is right"
    out["clicks"] = clicks
    return out


def annotate(img, plan):
    out = img.copy()
    for c in plan["clicks"]:
        col = (0, 200, 0) if c["role"] in ("verify", "start_again") else (0, 0, 255)
        cv2.circle(out, (c["x"], c["y"]), 18, col, 3)
        cv2.putText(out, str(c["seq"]), (c["x"] + 22, c["y"] - 14), cv2.FONT_HERSHEY_SIMPLEX, 1.0, col, 2)
    cv2.putText(out, f"{plan['state']}: {plan['note']}"[:110], (12, out.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (0, 0, 0), 2)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--image", help="a photo of the screen (raw camera photo, or straightened with --straight)")
    src.add_argument("--camera", help="take the photo now: /dev/videoN, ipwebcam:IP, ...")
    p.add_argument("--straight", action="store_true", help="--image is already straightened (no markers needed)")
    p.add_argument("--size", default="1280x800", help="straightened size, same aspect as the screen")
    p.add_argument("--tiles", help="e.g. 1,3,8: skip the solver and plan with these tiles")
    p.add_argument("--target", help="what to select, if known (skips reading the banner)")
    p.add_argument("--model")
    p.add_argument("--full", action="store_true", help="append Verify after the tile clicks")
    p.add_argument("--next", action="store_true", dest="next_challenge", help="on 'Access granted', return 'Start again'")
    p.add_argument("--layout", default=str(LAYOUT_PATH))
    p.add_argument("--demo-layout", action="store_true", help="use the committed 1280x800 screenshots as taught pages")
    p.add_argument("--out", help="also write the JSON here")
    p.add_argument("--annotate", help="write the photo with the clicks drawn on it")
    p.add_argument("--delay", type=float, default=0, help="seconds to switch to the page before a --camera photo")
    a = p.parse_args()

    w, h = map(int, a.size.lower().split("x"))
    if a.image and a.straight:
        img = cv2.imread(a.image)
        if img is None:
            raise SystemExit(f"Can't read {a.image}")
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA) if img.shape[:2] != (h, w) else img
    else:
        from see import ScreenCamera
        import time
        cam = ScreenCamera(a.image or a.camera, w, h)
        if a.delay > 0 and a.camera:
            print(f"Photo in {a.delay:.0f} s: switch to the page", file=sys.stderr, flush=True)
            time.sleep(a.delay)
        img = cam.screen_image(allow_remembered=False)
        if img is None:
            raise SystemExit("Can't see all 4 corner markers in the photo (tick 'Camera markers' on the site; "
                             "whole screen in view, no glare).")

    layout = demo_layout() if a.demo_layout else SiteLayout.load(a.layout)
    solver = None
    if not a.tiles:
        from solve import SolverError, solve_screen, solver_config
        try:
            cfg = solver_config()
        except SolverError as e:
            cfg = None
            cfg_err = e.detail

        def solver(im, target, model):
            if cfg is None:
                raise SolverError("bad_config", cfg_err)
            return solve_screen(im, cfg, target, model, (w, h))
    tiles = [int(t) for t in a.tiles.split(",") if t != ""] if a.tiles else None
    plan = plan_clicks(img, layout, solver, a.target, a.model, tiles, a.full, a.next_challenge)

    text = json.dumps(plan, indent=1)
    print(text)
    if a.out:
        Path(a.out).write_text(text)
    if a.annotate:
        cv2.imwrite(a.annotate, annotate(img, plan))
    raise SystemExit(0 if plan["clicks"] or plan["done"] else 2)


if __name__ == "__main__":
    main()
