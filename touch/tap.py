"""Tap points on the screen.

  python tap.py 0.25 0.40 0.75 0.40      # tap (u,v) pairs, 0..1 screen coords
  python tap.py --json '[[0.25,0.4],[0.75,0.4]]'
  python tap.py --test                    # tap every calibration dot to check accuracy
  python tap.py --home                    # just go to the home pose

Use from Python (e.g. from your agent):
  from tap import Tapper
  with Tapper(port="/dev/ttyACM0") as t:
      t.tap_many([(0.25, 0.4), (0.75, 0.4)])
"""

import argparse
import json
import time

import numpy as np

from common import ScreenModel, connect_follower, follow_path, load_calib, move_to


class Tapper:
    def __init__(self, port, robot_id="my_follower", robot_type="so101",
                 lift=0.3, press=0.04, speed=1.0, via_home=True, smooth=0.08, own_path=True):
        """
        own_path: replay your recorded (smoothed) leader paths if the calibration has them
        smooth: smoothing strength in seconds (0 = raw recording, 0.15 = very smooth)
        via_home: go back to the home pose after every tap (stylus never drags across the screen)
        lift:  how high the stylus hovers right before/after a tap (0..1, fraction of the way to home)
        press: how far past the taught touch point to push (raise if taps don't register)
        speed: >1 faster, <1 slower
        """
        calib = load_calib()
        self.model = ScreenModel(calib, smooth_s=smooth)
        self.own_path = own_path and self.model.has_paths
        self.lift, self.press, self.speed, self.via_home = lift, press, speed, via_home
        self.robot = connect_follower(port, robot_id, robot_type)
        self.at_home = False

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.go_home()
        self.robot.disconnect()

    def go_home(self, seconds=1.5):
        if not self.at_home:
            move_to(self.robot, self.model.home, seconds / self.speed)
            self.at_home = True

    def tap(self, u, v):
        if not (0 <= u <= 1 and 0 <= v <= 1):
            raise ValueError(f"({u}, {v}) is outside the screen")
        touch = self.model.touch_pose(u, v)
        if self.own_path:
            return self._tap_own_path(u, v, touch)
        above = touch + (self.model.home - touch) * self.lift
        pressed = touch + (touch - above) * self.press / max(self.lift, 1e-6)
        if self.via_home:
            self.go_home()  # start every tap from home, above the screen
        self.at_home = False
        move_to(self.robot, above, 0.7 / self.speed)    # over the target
        move_to(self.robot, pressed, 0.35 / self.speed)  # down
        time.sleep(0.12)
        move_to(self.robot, above, 0.3 / self.speed)     # straight back up
        if self.via_home:
            self.go_home(0.7)                            # and back home before the next one

    def _tap_own_path(self, u, v, touch):
        """home -> your smoothed path down -> small press -> your smoothed path back up -> home"""
        self.go_home()
        self.at_home = False
        down, t_down = self.model.path("approach", u, v)
        up, t_up = self.model.path("retreat", u, v)
        follow_path(self.robot, down, t_down / self.speed)
        pressed = touch + (touch - down[-8]) * self.press * 4  # push a little further along the last bit
        move_to(self.robot, pressed, 0.1)
        time.sleep(0.12)
        follow_path(self.robot, up, t_up / self.speed)
        self.at_home = True

    def tap_many(self, points):
        for u, v in points:
            self.tap(u, v)


def measure_summary(results):
    """Errors between where we aimed and where the page saw the touch."""
    px = [r["error_px"] for r in results if r.get("error_px") is not None]
    frac = [r["error_frac"] for r in results if r.get("error_frac") is not None]
    missed = sum(1 for r in results if not r["touches"])
    s = {"taps": len(results), "no_touch": missed}
    if frac:
        s.update(mean_error_pct=100 * float(np.mean(frac)), max_error_pct=100 * float(np.max(frac)))
    if px:
        s.update(mean_error_px=float(np.mean(px)), max_error_px=float(np.max(px)),
                 within_25px=sum(e <= 25 for e in px), within_50px=sum(e <= 50 for e in px))
        dx = [r["offset_px"][0] for r in results if r.get("offset_px")]
        dy = [r["offset_px"][1] for r in results if r.get("offset_px")]
        s.update(mean_offset_px=[float(np.mean(dx)), float(np.mean(dy))])  # same-direction miss = systematic
    return s


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("coords", nargs="*", type=float, help="u1 v1 u2 v2 ...")
    p.add_argument("--json", help='list of [u, v] pairs')
    p.add_argument("--test", action="store_true", help="tap every calibration target and measure")
    p.add_argument("--random", type=int, default=0, metavar="N", help="tap N random points and measure")
    p.add_argument("--no-measure", action="store_true", help="tests without the page reporting touches")
    p.add_argument("--http-port", type=int, default=8765)
    p.add_argument("--home", action="store_true")
    p.add_argument("--port", required=True)
    p.add_argument("--id", default="my_follower")
    p.add_argument("--robot", default="so101")
    p.add_argument("--press", type=float, default=0.04)
    p.add_argument("--lift", type=float, default=0.3)
    p.add_argument("--speed", type=float, default=1.0)
    p.add_argument("--no-home", action="store_true", help="don't return home between taps (faster)")
    p.add_argument("--smooth", type=float, default=0.08, help="path smoothing in seconds (0 = raw)")
    p.add_argument("--simple-path", action="store_true", help="ignore your recorded paths")
    a = p.parse_args()

    if a.home:
        pts = []
    elif a.test:
        pts = [(d["u"], d["v"]) for d in load_calib()["points"]]
    elif a.random:
        rng = np.random.default_rng()
        pts = [tuple(x) for x in rng.uniform(0.1, 0.9, size=(a.random, 2)).round(3)]
    elif a.json:
        pts = json.loads(a.json)
    else:
        if len(a.coords) % 2:
            raise SystemExit("Give coordinates in u v pairs")
        pts = list(zip(a.coords[::2], a.coords[1::2]))

    srv = None
    if (a.test or a.random) and not a.no_measure:
        from touchserver import TouchServer, save_log
        srv = TouchServer(a.http_port, mode="test", msg="Tap test: blue dots show where the robot touched")
        print("Reload the page on the touch screen. Starting in 3 s...")
        time.sleep(3)

    results = []
    with Tapper(a.port, a.id, a.robot, a.lift, a.press, a.speed, via_home=not a.no_home,
                smooth=a.smooth, own_path=not a.simple_path) as t:
        err = ", ".join(f"{e:.1f}" for e in t.model.fit_error[:5])
        print(f"Calibration fit error per joint (degrees): {err}")
        if t.own_path:
            m = t.model
            print(f"Replaying your recorded paths: down {m.path_time['approach'] / a.speed:.1f}s, "
                  f"up {m.path_time['retreat'] / a.speed:.1f}s, shake removed ~{m.shake['approach']:.2f} deg")
        else:
            print("Using calculated paths (no recorded paths in the calibration)")
        for i, (u, v) in enumerate(pts, 1):
            if srv:
                srv.set(step=i, target=[u, v], msg=f"Tap {i}/{len(pts)}: aiming at the orange ring")
                srv.drain()
            t.tap(u, v)
            line = f"tap {i}: aim u={u:.3f} v={v:.3f}"
            if srv:
                time.sleep(0.3)  # give the page time to report
                got = srv.all_touches()
                r = {"target": [u, v], "touches": got}
                if got:
                    g = got[0]
                    du, dv = g["u"] - u, g["v"] - v
                    r["error_frac"] = float(np.hypot(du, dv))
                    if g.get("w") and g.get("h"):
                        r["offset_px"] = [du * g["w"], dv * g["h"]]
                        r["error_px"] = float(np.hypot(*r["offset_px"]))
                        line += f"  landed u={g['u']:.3f} v={g['v']:.3f}  off by {r['error_px']:.0f}px"
                    else:
                        line += f"  landed u={g['u']:.3f} v={g['v']:.3f}"
                    if len(got) > 1:
                        line += f"  ({len(got)} touches: bounced or dragged)"
                else:
                    line += "  NO TOUCH registered (press harder: --press 0.08)"
                results.append(r)
            print(line, flush=True)

    if srv:
        summ = measure_summary(results)
        log = save_log("test", {"settings": vars(a), "summary": summ, "results": results})
        print("\n== Accuracy ==")
        if "mean_error_px" in summ:
            print(f"mean error {summ['mean_error_px']:.0f}px, worst {summ['max_error_px']:.0f}px, "
                  f"within 25px: {summ['within_25px']}/{summ['taps']}, within 50px: {summ['within_50px']}/{summ['taps']}")
            ox, oy = summ["mean_offset_px"]
            print(f"average miss: {abs(ox):.0f}px {'right' if ox >= 0 else 'left'}, "
                  f"{abs(oy):.0f}px {'down' if oy >= 0 else 'up'}  "
                  "(if this is most of the error, the misses are systematic: recalibrate)")
        print(f"taps with no touch: {summ['no_touch']}")
        print(f"Log saved: {log}")
        srv.set(step=0, target=None, msg=(f"Done: mean error {summ['mean_error_px']:.0f}px"
                                          if "mean_error_px" in summ else "Done"))
        time.sleep(1.5)
        srv.stop()
