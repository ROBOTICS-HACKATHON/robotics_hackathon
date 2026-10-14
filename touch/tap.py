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

from common import ScreenModel, connect_follower, load_calib, move_to


class Tapper:
    def __init__(self, port, robot_id="my_follower", robot_type="so101",
                 lift=0.3, press=0.04, speed=1.0, via_home=True):
        """
        via_home: go back to the home pose after every tap (stylus never drags across the screen)
        lift:  how high the stylus hovers right before/after a tap (0..1, fraction of the way to home)
        press: how far past the taught touch point to push (raise if taps don't register)
        speed: >1 faster, <1 slower
        """
        calib = load_calib()
        self.model = ScreenModel(calib)
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

    def tap_many(self, points):
        for u, v in points:
            self.tap(u, v)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("coords", nargs="*", type=float, help="u1 v1 u2 v2 ...")
    p.add_argument("--json", help='list of [u, v] pairs')
    p.add_argument("--test", action="store_true", help="tap every calibration dot")
    p.add_argument("--home", action="store_true")
    p.add_argument("--port", required=True)
    p.add_argument("--id", default="my_follower")
    p.add_argument("--robot", default="so101")
    p.add_argument("--press", type=float, default=0.04)
    p.add_argument("--lift", type=float, default=0.3)
    p.add_argument("--speed", type=float, default=1.0)
    p.add_argument("--no-home", action="store_true", help="don't return home between taps (faster)")
    a = p.parse_args()

    if a.home:
        pts = []
    elif a.test:
        pts = [(d["u"], d["v"]) for d in load_calib()["points"]]
    elif a.json:
        pts = json.loads(a.json)
    else:
        if len(a.coords) % 2:
            raise SystemExit("Give coordinates in u v pairs")
        pts = list(zip(a.coords[::2], a.coords[1::2]))

    with Tapper(a.port, a.id, a.robot, a.lift, a.press, a.speed, via_home=not a.no_home) as t:
        err = ", ".join(f"{e:.1f}" for e in t.model.fit_error[:5])
        print(f"Calibration fit error per joint (degrees): {err}")
        for i, (u, v) in enumerate(pts, 1):
            print(f"tap {i}: u={u:.3f} v={v:.3f}")
            t.tap(u, v)
