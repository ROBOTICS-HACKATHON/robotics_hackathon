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
                 lift=0.3, press=0.04, speed=1.0):
        """
        lift:  how far the stylus backs off between taps (0..1, fraction of the way to home)
        press: how far past the taught touch point to push (raise if taps don't register)
        speed: >1 faster, <1 slower
        """
        calib = load_calib()
        self.model = ScreenModel(calib)
        self.lift, self.press, self.speed = lift, press, speed
        self.robot = connect_follower(port, robot_id, robot_type)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.go_home()
        self.robot.disconnect()

    def go_home(self):
        move_to(self.robot, self.model.home, 1.5 / self.speed)

    def tap(self, u, v):
        if not (0 <= u <= 1 and 0 <= v <= 1):
            raise ValueError(f"({u}, {v}) is outside the screen")
        touch = self.model.touch_pose(u, v)
        above = touch + (self.model.home - touch) * self.lift
        pressed = touch + (touch - above) * self.press / max(self.lift, 1e-6)
        move_to(self.robot, above, 0.8 / self.speed)
        move_to(self.robot, pressed, 0.35 / self.speed)
        time.sleep(0.12)
        move_to(self.robot, above, 0.3 / self.speed)

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
    a = p.parse_args()

    if a.test:
        pts = [(d["u"], d["v"]) for d in load_calib()["points"]]
    elif a.json:
        pts = json.loads(a.json)
    else:
        if len(a.coords) % 2:
            raise SystemExit("Give coordinates in u v pairs")
        pts = list(zip(a.coords[::2], a.coords[1::2]))

    with Tapper(a.port, a.id, a.robot, a.lift, a.press, a.speed) as t:
        err = ", ".join(f"{e:.1f}" for e in t.model.fit_error[:5])
        print(f"Calibration fit error per joint (degrees): {err}")
        if not a.home:
            for i, (u, v) in enumerate(pts, 1):
                print(f"tap {i}: u={u:.3f} v={v:.3f}")
                t.tap(u, v)
