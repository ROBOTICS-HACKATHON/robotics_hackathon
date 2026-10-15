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


NEAR_MM = 15.0  # a missed tap this close to the glass (camera-measured) still earns partial credit


def reward(err_px, touched, n_touches, force_stop, tol_px, gap_mm=None):
    """Score of one attempt: 1 = perfect, 0 = at the tolerance edge, negative = bad.
    No touch: -1, or (with 2 cameras) up to -0.4 the closer the tip stopped above the glass,
    so the press-depth choice learns from near misses too."""
    if not touched:
        if gap_mm is None:
            return -1.0
        return float(-1.0 + 0.6 * max(0.0, 1.0 - max(gap_mm, 0.0) / NEAR_MM))
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

    def __init__(self, calib, exp, width=0.18, ridge=1e-3, half_life=100):
        pts = calib["points"]
        uv = [[p["u"], p["v"]] for p in pts]
        q = [p["q"] for p in pts]
        w = [3.0] * len(pts)  # taught points count a bit more
        touches = exp.data["touches"]
        n = len(touches)
        for i, t in enumerate(touches):
            uv.append(t["uv"])
            q.append(t["q"])
            # recent touches count more (a touch `half_life` touches ago counts half), so the model follows
            # slow drift and small moves of the arm or screen instead of averaging them away; never below 0.15
            w.append(max(0.15, 0.5 ** ((n - 1 - i) / half_life)) if half_life else 1.0)
        self.uv, self.q, self.w = np.array(uv, float), np.array(q, float), np.array(w, float)
        self.home = np.array(calib["home"], float)
        self.width, self.ridge = width, ridge
        F = _features(self.uv[:, 0], self.uv[:, 1])
        W = self.w[:, None]
        self.global_coef = np.linalg.lstsq(F * np.sqrt(W), self.q * np.sqrt(W), rcond=None)[0]
        n0 = len(pts)
        self.calib_coef = np.linalg.lstsq(F[:n0], self.q[:n0], rcond=None)[0]

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
        # never stray far from the calibration-only fit (protects against wild extrapolation)
        f0 = _features(np.array([u]), np.array([v]))[0] @ self.calib_coef
        q[:5] = np.clip(q[:5], f0[:5] - 8.0, f0[:5] + 8.0)
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


SLOW_OPTIONS = [1.0, 0.6]  # approach speed factor: 0.6 = slower, gentler on the servos
ARMS = [(p, s) for p in PRESS_OPTIONS for s in SLOW_OPTIONS]


def zone_centers(calib):
    """The 9 calibration targets: each one is the centre of its own screen zone (1..9)."""
    return np.array([p.get("target") or [p["u"], p["v"]] for p in calib["points"]], float)


def zone_of(u, v, centers):
    return int(np.argmin(((centers - [u, v]) ** 2).sum(1))) + 1


class ZoneBandit:
    """Reward-driven settings PER ZONE: how deep to press and how fast to come down.

    Zones differ (servo load, reach, angle), so each zone learns its own best option.
    A zone with little data borrows from the other zones (shrinkage), so it starts sensible
    and becomes its own as evidence comes in. Epsilon-greedy exploration.
    """

    def __init__(self, exp, centers, eps=0.15, prior=3.0):
        self.exp, self.centers, self.eps, self.prior = exp, centers, eps, prior
        self.z = exp.data.setdefault("zones", {})

    @staticmethod
    def key(arm):
        return f"{arm[0]}|{arm[1]}"

    def zone(self, u, v):
        return zone_of(u, v, self.centers)

    def _global(self, k):
        n = sum(z.get(k, {}).get("n", 0) for z in self.z.values())
        m = sum(z.get(k, {}).get("n", 0) * z.get(k, {}).get("mean", 0) for z in self.z.values()) / n if n else 0.0
        return n, m

    def value(self, zone, arm):
        k = self.key(arm)
        s = self.z.get(str(zone), {}).get(k, {"n": 0, "mean": 0.0})
        ng, mg = self._global(k)
        if s["n"] + ng == 0:
            return None
        return (s["n"] * s["mean"] + self.prior * mg) / (s["n"] + self.prior) if ng else s["mean"]

    def choose(self, zone, rng):
        untried = [a for a in ARMS if self._global(self.key(a))[0] == 0]
        if untried:
            return untried[int(rng.integers(len(untried)))]
        if rng.random() < self.eps:
            return ARMS[int(rng.integers(len(ARMS)))]
        return self.best(zone)

    def best(self, zone):
        vals = [(self.value(zone, a), a) for a in ARMS]
        vals = [x for x in vals if x[0] is not None]
        return max(vals)[1] if vals else (0.04, 1.0)

    def update(self, zone, arm, r):
        s = self.z.setdefault(str(zone), {}).setdefault(self.key(arm), {"n": 0, "mean": 0.0})
        s["n"] += 1
        s["mean"] += (r - s["mean"]) / s["n"]


def zone_report(exp, centers, tol_px=15):
    """Per-zone accuracy from all recorded attempts."""
    att = exp.data.get("attempts", [])
    rep = {}
    for z in range(1, len(centers) + 1):
        a = [x for x in att if x["zone"] == z]
        first = [x for x in a if x["try"] == 1]
        fe = [x["err_px"] for x in first if x["err_px"] is not None]
        ae = [x["err_px"] for x in a if x["err_px"] is not None]
        nt = sum(1 for x in a if not x["touched"])
        rep[z] = {"attempts": len(a), "first_tries": len(first),
                  "first_err": float(np.mean(fe)) if fe else None,
                  "all_err": float(np.mean(ae)) if ae else None,
                  "no_touch": nt / len(a) if a else None,
                  "hit_rate": (sum(e <= tol_px for e in fe) / len(fe)) if fe else None}
        # one number for "how hard is this zone": typical first-try miss, no-touches count as big misses
        if a:
            rep[z]["difficulty"] = (rep[z]["first_err"] if fe else (rep[z]["all_err"] or 40.0)) \
                + 60.0 * (rep[z]["no_touch"] or 0)
        else:
            rep[z]["difficulty"] = None
    return rep


def zone_weights(rep):
    """How often to practise each zone: harder zones more, every zone at least a bit."""
    d = {z: r["difficulty"] for z, r in rep.items()}
    known = [x for x in d.values() if x is not None]
    base = np.mean(known) if known else 1.0
    w = {z: (1.5 if x is None else 0.5 + x / max(base, 1e-6)) for z, x in d.items()}
    s = sum(w.values())
    return {z: v / s for z, v in w.items()}


def draw_heatmap(rep, bandit, path, title="Tap accuracy per zone (first try)"):
    """3x3 heatmap image: green = accurate, red = hard. Also returns a text table."""
    import cv2
    W, H, top = 1200, 840, 70
    img = np.full((H, W, 3), 255, np.uint8)
    cv2.putText(img, title, (24, 46), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (40, 40, 40), 2, cv2.LINE_AA)
    known = [r["difficulty"] for r in rep.values() if r["difficulty"] is not None]
    lo, hi = (min(known), max(known)) if known else (0, 1)
    cw, ch = W // 3, (H - top) // 3
    lines = []
    for z, r in rep.items():
        row, col = (z - 1) // 3, (z - 1) % 3
        x0, y0 = col * cw, top + row * ch
        if r["difficulty"] is None:
            color = (225, 225, 225)
        else:
            t = 0 if hi == lo else (r["difficulty"] - lo) / (hi - lo)
            color = (int(120 * (1 - t) + 60 * t), int(200 * (1 - t) + 70 * t), int(90 * (1 - t) + 230 * t))
        cv2.rectangle(img, (x0 + 6, y0 + 6), (x0 + cw - 6, y0 + ch - 6), color, -1)
        best = bandit.best(z) if bandit else None
        txt = [f"zone {z}",
               f"first try: {r['first_err']:.0f}px" if r["first_err"] is not None else "first try: -",
               f"hit<=15px: {100 * r['hit_rate']:.0f}%" if r["hit_rate"] is not None else "hit: -",
               f"no touch: {100 * r['no_touch']:.0f}%" if r["no_touch"] is not None else "no touch: -",
               f"best: press {best[0]}, {'slow' if best[1] < 1 else 'normal'}" if best else "",
               f"{r['attempts']} attempts"]
        for i, s in enumerate(txt):
            cv2.putText(img, s, (x0 + 22, y0 + 48 + i * 38), cv2.FONT_HERSHEY_SIMPLEX,
                        0.9 if i == 0 else 0.75, (20, 20, 20), 2 if i == 0 else 1, cv2.LINE_AA)
        lines.append(f"zone {z}: first try "
                     + (f"{r['first_err']:5.1f}px" if r["first_err"] is not None else "   -   ")
                     + (f", hit {100 * r['hit_rate']:3.0f}%" if r["hit_rate"] is not None else "")
                     + (f", no touch {100 * r['no_touch']:3.0f}%" if r["no_touch"] is not None else "")
                     + (f", best press {best[0]} {'slow' if best[1] < 1 else 'normal'}" if best else "")
                     + f"  ({r['attempts']} attempts)")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), img)
    return "\n".join(lines)


def refine(tapper, server, model, exp, bandit, target, tol_px, max_tries, rng, learn=True, gain=0.8, log=print):
    """Feedback loop for one target: tap, see where it landed, correct the JOINT pose, retry
    until within tol_px. Every attempt is stored as experience and updates the model."""
    u, v = target
    q = tapper.safety.clamp(model.touch_pose(u, v), tapper.base_model.touch_pose(u, v))
    J = model.jacobian(u, v)
    zone = bandit.zone(u, v)
    base_speed = tapper.speed
    attempts = []
    for k in range(1, max_tries + 1):
        arm = bandit.choose(zone, rng)
        press, slow = arm
        tapper.press, tapper.speed = press, base_speed * slow
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
            err_px = None
            r = reward(0, False, 0, tapper.last_force_stop, tol_px, getattr(tapper, "last_press_height", None))
        bandit.update(zone, arm, r)
        if learn:
            exp.data.setdefault("attempts", []).append(
                {"zone": zone, "try": k, "err_px": err_px, "touched": err_px is not None, "press": press,
                 "slow": slow, "reward": r, "target": [float(u), float(v)], "t": time.time()})
        lv = getattr(tapper, "last_vision", None)
        attempts.append({"try": k, "zone": zone, "press": press, "slow": slow, "err_px": err_px, "reward": r,
                         "touches": len(got), "force_stop": tapper.last_force_stop,
                         "vision_pred": lv.get("landing") if lv else None,
                         "vision_height": lv.get("height_mm", lv.get("height")) if lv else None,
                         "press_height_mm": getattr(tapper, "last_press_height", None)})
        log(f"   try {k} (zone {zone}): press {press:<4} {'slow' if slow < 1 else 'norm'} "
            + (f"miss {err_px:5.1f}px" if err_px is not None else
               ("NO TOUCH  " if getattr(tapper, "last_press_height", None) is None else
                f"NO TOUCH, tip {tapper.last_press_height:4.1f}mm above glass"))
            + f"  reward {r:+.2f}" + ("  [force stop]" if tapper.last_force_stop else "")
            + (("  [cam: " + ("tip not seen" if not lv or lv.get("landing") is None else
                ("REJECTED (implausible)" if lv.get("rejected") else
                 f"predicted miss {np.hypot((lv['landing'][0] - u) * (w if got else 1000), (lv['landing'][1] - v) * (h if got else 1000)):.0f}px"))
                + ("" if getattr(tapper.vision, "ready", False) else ", learning") + "]")
               if getattr(tapper, "vision", None) is not None and tapper.vision.enabled else ""))
        if err_px is not None and err_px <= tol_px:
            break
        if err_px is not None:
            # where we landed tells us how the arm really moves: step the joints toward the target
            # correct from the pose that was really used, in limited steps, never far from calibration
            q = tapper.safety.step(q_used, gain * (J @ -miss), tapper.base_model.touch_pose(u, v))
            q[-1] = model.home[-1]
        else:
            gap = getattr(tapper, "last_press_height", None)
            hov_q = getattr(tapper, "last_hover_q", None)
            hov_h = None if (lv or {}).get("inconsistent") else (lv or {}).get("height_mm")
            if gap is not None and hov_q is not None and hov_h is not None and hov_h - gap > 3:
                # cameras: the tip stopped `gap` mm short. Joints per mm from the hover->press move,
                # then go that much deeper (+2 mm so it really touches). safety.step limits the step.
                per_mm = (q_used - hov_q) / (hov_h - gap)
                q = tapper.safety.step(q_used, per_mm * (gap + 2.0), tapper.base_model.touch_pose(u, v))
            else:
                # no camera height: go a little deeper next time along the approach direction
                q = tapper.safety.step(q_used, 0.015 * (q_used - model.home), tapper.base_model.touch_pose(u, v))
    tapper.speed = base_speed
    exp.save()
    if getattr(tapper, "vision", None) is not None and tapper.vision.enabled:
        tapper.vision.save()
    return attempts


if __name__ == "__main__":
    import argparse

    from common import load_calib
    from correction import calib_id

    ap = argparse.ArgumentParser()
    ap.add_argument("--heatmap", default="outputs/heatmap.png")
    ap.add_argument("--tol", type=float, default=15)
    a = ap.parse_args()
    calib = load_calib()
    exp = Experience(calib_id(calib))
    centers = zone_centers(calib)
    if not exp.data.get("attempts"):
        raise SystemExit("No practice data for this calibration yet: run ./lr practice first.")
    print(draw_heatmap(zone_report(exp, centers, a.tol), ZoneBandit(exp, centers), a.heatmap))
    print(f"Heatmap image: {a.heatmap}")
