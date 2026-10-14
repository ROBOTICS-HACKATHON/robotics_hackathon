"""Learning from experience: every measured touch teaches the arm how to hit the screen.

Three parts:
  * Experience  - every attempt: the joint pose the arm was sent to, where the page saw
                  the touch, and the reward. Stored per calibration in experience.json.
  * LearnedModel - joint-space model (u, v) -> joint pose, refit from calibration points
                  PLUS all experience, with nearby examples weighted more (locally
                  weighted regression). This is the arm's own "how do I hit (u, v)".
  * PressBandit - reward-driven choice of how deep to press (epsilon-greedy bandit):
                  options that register cleanly and accurately get picked more.
And `refine()`: the per-target feedback loop that retries until the tap is accurate.
"""

import json
import time
from pathlib import Path

import numpy as np

from common import _features

HERE = Path(__file__).resolve().parent
EXP_FILE = HERE / "experience.json"
PRESS_OPTIONS = [0.02, 0.04, 0.07, 0.1]


def reward(err_px, touched, n_touches, force_stop, tol_px):
    """Score of one attempt: 1 = perfect, 0 = at the tolerance edge, negative = bad."""
    if not touched:
        return -1.0
    r = 1.0 - err_px / tol_px
    if n_touches > 1:
        r -= 0.3  # bounced / dragged
    if force_stop:
        r -= 0.2  # pushed too hard
    return float(max(r, -1.0))


class Experience:
    def __init__(self, cid):
        self.cid = cid
        allx = json.loads(EXP_FILE.read_text()) if EXP_FILE.exists() else {}
        self.data = allx.get(cid, {"touches": [], "bandit": {}})

    def add(self, q, landed, target, r, press, extra=None):
        self.data["touches"].append({"q": [round(float(x), 3) for x in q], "uv": [float(landed[0]), float(landed[1])],
                                     "target": [float(target[0]), float(target[1])], "reward": r,
                                     "press": press, "t": time.time(), **(extra or {})})

    def save(self):
        allx = json.loads(EXP_FILE.read_text()) if EXP_FILE.exists() else {}
        allx[self.cid] = self.data
        EXP_FILE.write_text(json.dumps(allx))

    def __len__(self):
        return len(self.data["touches"])


class LearnedModel:
    """(u, v) -> joint pose that touches there. Calibration points + experience, locally weighted."""

    def __init__(self, calib, exp, width=0.18, ridge=1e-3):
        pts = calib["points"]
        uv = [[p["u"], p["v"]] for p in pts]
        q = [p["q"] for p in pts]
        w = [3.0] * len(pts)  # taught points count a bit more
        for t in exp.data["touches"]:
            uv.append(t["uv"])
            q.append(t["q"])
            w.append(1.0)
        self.uv, self.q, self.w = np.array(uv, float), np.array(q, float), np.array(w, float)
        self.home = np.array(calib["home"], float)
        self.width, self.ridge = width, ridge
        F = _features(self.uv[:, 0], self.uv[:, 1])
        W = self.w[:, None]
        self.global_coef = np.linalg.lstsq(F * np.sqrt(W), self.q * np.sqrt(W), rcond=None)[0]

    def touch_pose(self, u, v):
        f = _features(np.array([u]), np.array([v]))[0]
        d2 = ((self.uv - [u, v]) ** 2).sum(1)
        w = self.w * np.exp(-d2 / (2 * self.width ** 2))
        if w.sum() < 1.0 or len(self.uv) < 12:  # not enough nearby experience: global fit
            q = f @ self.global_coef
        else:
            F = _features(self.uv[:, 0], self.uv[:, 1])
            A = F.T @ (F * w[:, None]) + self.ridge * w.sum() * np.eye(6)
            B = F.T @ (self.q * w[:, None]) + self.ridge * w.sum() * self.global_coef
            q = f @ np.linalg.solve(A, B)  # local fit, pulled toward the global one
        q[-1] = self.home[-1]
        return q

    def jacobian(self, u, v, h=0.01):
        """How the joints change per unit of u and v around (u, v)  -> shape (6, 2)."""
        du = (self.touch_pose(u + h, v) - self.touch_pose(u - h, v)) / (2 * h)
        dv = (self.touch_pose(u, v + h) - self.touch_pose(u, v - h)) / (2 * h)
        return np.stack([du, dv], 1)


class PressBandit:
    """Picks the press depth; learns from rewards which one works best (epsilon-greedy)."""

    def __init__(self, exp, eps=0.15):
        self.exp, self.eps = exp, eps
        b = exp.data.setdefault("bandit", {})
        for p in PRESS_OPTIONS:
            b.setdefault(str(p), {"n": 0, "mean": 0.0})
        self.b = b

    def choose(self, rng):
        if rng.random() < self.eps or any(v["n"] == 0 for v in self.b.values()):
            untried = [k for k, v in self.b.items() if v["n"] == 0]
            return float(rng.choice(untried or list(self.b)))
        return float(max(self.b, key=lambda k: self.b[k]["mean"]))

    def best(self):
        tried = {k: v for k, v in self.b.items() if v["n"] > 0}
        return float(max(tried, key=lambda k: tried[k]["mean"])) if tried else 0.04

    def update(self, press, r):
        s = self.b[str(press)]
        s["n"] += 1
        s["mean"] += (r - s["mean"]) / s["n"]

    def describe(self):
        return ", ".join(f"press {k}: {v['mean']:+.2f} ({v['n']}x)" for k, v in self.b.items())


def refine(tapper, server, model, exp, bandit, target, tol_px, max_tries, rng, learn=True, gain=0.8, log=print):
    """Feedback loop for one target: tap, see where it landed, correct the JOINT pose, retry
    until within tol_px. Every attempt is stored as experience and updates the model."""
    u, v = target
    q = model.touch_pose(u, v)
    J = model.jacobian(u, v)
    attempts = []
    for k in range(1, max_tries + 1):
        press = bandit.choose(rng)
        tapper.press = press
        server.drain()
        server.set(target=[u, v], msg=f"target ({u:.2f},{v:.2f}) try {k}/{max_tries}")
        tapper.tap_pose(u, v, q)
        q_used = tapper.last_touch_q if tapper.last_touch_q is not None else q
        time.sleep(0.3)
        got = server.all_touches()
        if got:
            g = got[0]
            w, h = g.get("w") or 1000, g.get("h") or 1000
            miss = np.array([g["u"] - u, g["v"] - v])
            err_px = float(np.hypot(miss[0] * w, miss[1] * h))
            r = reward(err_px, True, len(got), tapper.last_force_stop, tol_px)
            if learn:
                exp.add(q_used, (g["u"], g["v"]), (u, v), r, press, {"try": k, "err_px": err_px})
                if getattr(tapper, "vision", None) is not None and tapper.last_hover_obs:
                    tapper.vision.learn(tapper.last_hover_obs, (g["u"], g["v"]))
        else:
            err_px, r = None, reward(0, False, 0, tapper.last_force_stop, tol_px)
        bandit.update(press, r)
        lv = getattr(tapper, "last_vision", None)
        attempts.append({"try": k, "press": press, "err_px": err_px, "reward": r,
                         "touches": len(got), "force_stop": tapper.last_force_stop,
                         "vision_pred": lv.get("landing") if lv else None,
                         "vision_height": lv.get("height") if lv else None})
        log(f"   try {k}: press {press:<4} " + (f"miss {err_px:5.1f}px" if err_px is not None else "NO TOUCH  ")
            + f"  reward {r:+.2f}" + ("  [force stop]" if tapper.last_force_stop else ""))
        if err_px is not None and err_px <= tol_px:
            break
        if err_px is not None:
            # where we landed tells us how the arm really moves: step the joints toward the target
            q = q_used + gain * (J @ -miss)  # correct from the pose that was really used
            q[-1] = model.home[-1]
        else:
            # no touch: go a little deeper next time along the approach direction (towards the screen)
            q = q_used + 0.15 * (q_used - model.home) * 0.1
    exp.save()
    if getattr(tapper, "vision", None) is not None and tapper.vision.enabled:
        tapper.vision.save()
    return attempts
