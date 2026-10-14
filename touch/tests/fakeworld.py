"""A scripted stand-in for "touch screen + camera + arm + solver", built from the real screenshots in shots/.

The world is a tiny state machine over the site's pages. The arm taps (u, v); the world finds what is there
(using positions.json, measured from the real page) and changes state like the site does. The camera shows
the matching real screenshot. Only the selections we have screenshots of can be shown: {}, {0,4}, {0,4,8}, all.
"""

import json
from pathlib import Path

import cv2
import numpy as np

SHOTS = Path(__file__).resolve().parent / "shots"
POS = json.loads((SHOTS / "positions.json").read_text())
IMG = {p.stem: cv2.imread(str(p)) for p in SHOTS.glob("*.png")}
SELECTION_SHOT = {frozenset(): "grid_empty", frozenset({0, 4}): "grid_tiles_0_4",
                  frozenset({0, 4, 8}): "grid_three_selected", frozenset(range(9)): "grid_all_selected"}
CORRECT = frozenset({0, 4, 8})      # the "right answer" of this fake challenge
HIT = 0.04                          # a tap within this (u, v) distance of an element presses it


def near(u, v, target):
    return np.hypot(u - target[0], v - target[1]) < HIT


class World:
    def __init__(self, page="choice", ignore_tiles=(), double_touch=None, blind=0, drop_taps=0):
        self.page = page
        self.sel = set()
        self.error = False
        self.taps = []                      # every tap the arm made: (u, v, what)
        self.ignore_tiles = set(ignore_tiles)   # taps on these tiles never register
        self.double_touch = double_touch or {}  # {tile: other tile it also toggles}
        self.blind = blind                  # next N looks return None (markers not visible)
        self.drop_taps = drop_taps          # the first N tile taps are lost entirely
        self.verify_pressed = 0
        self.arm_moves = 0

    # --- the arm
    def tap(self, u, v):
        self.arm_moves += 1
        what = "nothing"
        if self.page == "choice" and near(u, v, POS["choice"]):
            what, self.page = "choice card", "checkbox"
        elif self.page == "checkbox" and near(u, v, POS["checkbox"]):
            what, self.page = "checkbox", "grid"
        elif self.page == "verified" and near(u, v, POS["verified"]):
            what, self.page, self.sel, self.error = "start again", "choice", set(), False
        elif self.page == "grid":
            for i, t in enumerate(POS["tiles"]):
                if near(u, v, t):
                    what = f"tile {i}"
                    if self.drop_taps > 0:
                        self.drop_taps -= 1
                    elif i not in self.ignore_tiles:
                        for k in (i, self.double_touch.get(i)):
                            if k is not None:
                                self.sel ^= {k}
                    break
            else:
                if near(u, v, POS["verify"]):
                    what = "verify"
                    self.verify_pressed += 1
                    if frozenset(self.sel) == CORRECT:
                        self.page = "verified"
                    else:
                        self.error = True
        self.taps.append((round(u, 4), round(v, 4), what))

    def go_home(self):
        pass

    # --- the camera
    def look(self):
        if self.blind > 0:
            self.blind -= 1
            return None
        if self.page == "grid":
            if self.error:
                return IMG["grid_error"].copy()
            return IMG[SELECTION_SHOT[frozenset(self.sel)]].copy()
        return IMG[self.page].copy()

    # --- the solver (what /api/robot/solve returns, from the true positions)
    def presses(self, tiles, shift=0.0, W=1280, H=800):
        out = [{"seq": k + 1, "role": "tile", "tile": t, "x_norm": POS["tiles"][t][0] + shift,
                "y_norm": POS["tiles"][t][1], "x": 0, "y": 0} for k, t in enumerate(tiles)]
        out.append({"seq": len(out) + 1, "role": "verify", "x_norm": POS["verify"][0], "y_norm": POS["verify"][1],
                    "x": 0, "y": 0})
        return {"ok": True, "target": "bicycles", "tiles": list(tiles), "presses": out, "model": "fake"}


class Clock:
    """Time that only moves when the code sleeps: tests run instantly."""

    def __init__(self):
        self.t = 1000.0

    def now(self):
        return self.t

    def sleep(self, s):
        self.t += s
