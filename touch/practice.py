"""Practice session: the robot teaches itself to tap accurately.

For each target it taps, sees where it landed (the page reports it), corrects its joint
pose and tries again until it's within --tol pixels, then moves on. Every attempt is
experience: the learned model is refit after each target, so first tries get better as
the session goes on. A reward per attempt also tunes how deep it presses.

  python practice.py --port /dev/ttyACM0 --targets 30 --tol 15
"""

import argparse
import time
from pathlib import Path

import numpy as np

from common import countdown, load_calib
from correction import calib_id
from learner import LearnedModel, ZoneBandit, draw_heatmap, refine, zone_centers, zone_of, zone_report, zone_weights
from tap import Tapper
from touchserver import TouchServer, save_log

p = argparse.ArgumentParser()
p.add_argument("--port", required=True)
p.add_argument("--id", default="my_follower")
p.add_argument("--robot", default="so101")
p.add_argument("--targets", type=int, default=30, help="how many targets to practise")
p.add_argument("--tol", type=float, default=15, help="good enough = within this many pixels")
p.add_argument("--tries", type=int, default=5, help="max attempts per target")
p.add_argument("--speed", type=float, default=1.0)
p.add_argument("--max-load", type=int, default=0, help="force guard (0 = off)")
p.add_argument("--http-port", type=int, default=8765)
p.add_argument("--cameras", default="", help="tip vision cameras: 'none', one, or two comma-separated")
p.add_argument("--tip-color", default="pink")
p.add_argument("--screen-mm", type=float, default=300.0, help="marker-to-marker screen width (for mm heights)")
p.add_argument("--delay", type=float, default=5, help="seconds to switch to the page before starting")
p.add_argument("--tap-from", type=float, default=0.85,
               help="stop on your recorded path this far down (0..1), then go straight down: lower = longer straight tap")
a = p.parse_args()

calib = load_calib()
cal = calib["points"]
lo = [min(q["u"] for q in cal), min(q["v"] for q in cal)]
hi = [max(q["u"] for q in cal), max(q["v"] for q in cal)]
rng = np.random.default_rng()
centers = zone_centers(calib)

srv = TouchServer(a.http_port, mode="test", msg="Practice: the robot is teaching itself")
countdown(a.delay, "switch to the page on the touch screen (full-screen, F11)")

results = []
vision = None
if a.cameras:
    from tipvision import TipVision
    vision = TipVision(a.cameras, a.tip_color, width_mm=a.screen_mm, cid=calib_id(calib))
with Tapper(a.port, a.id, a.robot, speed=a.speed, max_load=a.max_load, correct=False, learned=False,
            vision=vision, tap_from=a.tap_from) as t:
    if vision is not None:
        print(vision.describe())
    t.stop_on_touch(srv)  # stop pressing the moment the page reports the touch
    exp = t.exp
    bandit = ZoneBandit(exp, centers)
    print(f"Starting with {len(exp)} touches of past experience for this calibration.")
    # practise hard zones more: pick a zone by difficulty, then a random spot inside it
    weights = zone_weights(zone_report(exp, centers, a.tol))
    print("practice share per zone: " + ", ".join(f"{z}:{100 * w:.0f}%" for z, w in weights.items()))
    zones = rng.choice(list(weights), size=a.targets, p=list(weights.values()))
    targets = []
    for z in zones:
        for _ in range(500):
            uv = rng.uniform(lo, hi).round(3)
            if zone_of(*uv, centers) == z:
                break
        targets.append(uv)
    for i, (u, v) in enumerate(targets, 1):
        model = LearnedModel(calib, exp)  # refit from everything learned so far
        t.model.touch_pose = model.touch_pose
        print(f"\ntarget {i}/{len(targets)}  ({u:.3f}, {v:.3f})  zone {zone_of(u, v, centers)}")
        att = refine(t, srv, model, exp, bandit, (u, v), a.tol, a.tries, rng)
        ok = att[-1]["err_px"] is not None and att[-1]["err_px"] <= a.tol
        results.append({"target": [float(u), float(v)], "attempts": att, "success": ok})
        print(f"   -> {'HIT' if ok else 'gave up'} after {len(att)} tr{'y' if len(att) == 1 else 'ies'}")

# ---- learning report ----
first = [r["attempts"][0]["err_px"] for r in results]
tries = [len(r["attempts"]) for r in results]
print("\n== Learning curve (first-try miss, before any retry) ==")
for s in range(0, len(results), 10):
    blk = [e for e in first[s:s + 10] if e is not None]
    nt = sum(e is None for e in first[s:s + 10])
    print(f"targets {s + 1:3d}-{min(s + 10, len(results)):3d}: "
          + (f"mean {np.mean(blk):5.1f}px" if blk else "  -  ")
          + f", avg tries {np.mean(tries[s:s + 10]):.1f}" + (f", {nt} first-try no-touch" if nt else ""))
hits = sum(r["success"] for r in results)
print(f"reached {a.tol:.0f}px on {hits}/{len(results)} targets, {np.mean(tries):.1f} tries on average")
print("\n== Per zone (all practice so far with this calibration) ==")
hm = Path(__file__).resolve().parent.parent / "outputs" / "heatmap.png"
print(draw_heatmap(zone_report(exp, centers, a.tol), bandit, hm))
log = save_log("practice", {"calib_id": calib_id(calib), "settings": vars(a), "results": results,
                            "zones": exp.data.get("zones"), "zone_report": zone_report(exp, centers, a.tol)})
draw_heatmap(zone_report(exp, centers, a.tol), bandit, str(log).replace(".json", "_heatmap.png"))
print(f"Heatmap: {hm}")
print(f"Experience: {len(exp)} touches saved. Log: {log}")
if vision is not None:
    print(vision.describe())
srv.set(target=None, msg=f"Practice done: {hits}/{len(results)} targets within {a.tol:.0f}px")
time.sleep(1.5)
srv.stop()
