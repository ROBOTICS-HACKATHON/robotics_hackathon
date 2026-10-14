"""Adaptive aim correction, learned from measured tap tests.

Every `./lr tap --test` / `--random N` run logs where the robot aimed and where
the page saw the touch. This fits the systematic part of that miss and corrects
future aims:   aim = target + correction(target)

It grows with the data: a constant shift for a few taps, an affine (shift +
stretch + rotation) for more, a smooth quadratic warp for many. Leave-one-out
validation decides which one is actually better, so it never overfits noise.
Only tests made with the CURRENT calibration are used.

  python correction.py            # fit from all logs, save, print report
  python correction.py --reset    # delete the correction
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CORR_FILE = HERE / "screen_correction.json"
LOG_DIR = HERE / "logs"
OUTLIER = 0.25  # misses larger than 25% of the screen are treated as bad touches (palm, bounce)


def calib_id(calib):
    """Fingerprint of the touch calibration (targets + poses), so old tests aren't mixed in."""
    key = json.dumps([[round(p["u"], 4), round(p["v"], 4), [round(x, 2) for x in p["q"]]] for p in calib["points"]])
    return hashlib.sha1(key.encode()).hexdigest()[:12]


def _feats(uv, kind):
    u, v = uv[:, 0], uv[:, 1]
    one = np.ones_like(u)
    if kind == "shift":
        return one[:, None]
    if kind == "affine":
        return np.stack([one, u, v], 1)
    return np.stack([one, u, v, u * v, u * u, v * v], 1)


KINDS = [("shift", 1), ("affine", 3), ("quadratic", 6)]


def _fit(landed, aimed, kind, ridge=1e-3):
    """Fit aim - landed as a function of landed (what we must add to hit a spot)."""
    F = _feats(landed, kind)
    R = aimed - landed
    A = F.T @ F + ridge * len(F) * np.eye(F.shape[1])
    A[0, 0] -= ridge * len(F)  # don't shrink the constant shift
    return np.linalg.solve(A, F.T @ R)


def collect(cid):
    """(aimed, landed) pairs from all test logs made with calibration `cid`."""
    aimed, landed, runs = [], [], 0
    calib_logs = sorted(LOG_DIR.glob("calib_*.json"))
    for f in sorted(LOG_DIR.glob("test_*.json")):
        d = json.loads(f.read_text())
        test_cid = d.get("calib_id")
        if test_cid is None:  # older log: belongs to the latest calibration made before it
            before = [c for c in calib_logs if c.name.split("_", 1)[1] < f.name.split("_", 1)[1]]
            test_cid = calib_id(json.loads(before[-1].read_text())) if before else None
        if test_cid != cid:
            continue
        runs += 1
        for r in d.get("results", []):
            if not r.get("touches"):
                continue
            a = r.get("command", r["target"])  # where the arm was actually sent
            t = r["touches"][0]
            aimed.append(a)
            landed.append([t["u"], t["v"]])
    return np.array(aimed, float).reshape(-1, 2), np.array(landed, float).reshape(-1, 2), runs


def fit(cid, verbose=True):
    aimed, landed, runs = collect(cid)
    keep = np.hypot(*(landed - aimed).T) < OUTLIER if len(aimed) else np.array([], bool)
    aimed, landed = aimed[keep], landed[keep]
    n = len(aimed)
    if n < 3:
        if verbose:
            print(f"Only {n} usable taps for this calibration: run ./lr tap --random 20 first.")
        return None

    def loo(kind, k):
        """Leave-one-out: error we'd get on a NEW spot after correcting."""
        if n <= k + 1:
            return np.inf
        errs = []
        for i in range(n):
            m = np.arange(n) != i
            c = _fit(landed[m], aimed[m], kind)
            # to hit target T we'd aim at T + corr(T); the remaining miss at this tap is approx:
            pred_aim = landed[i] + _feats(landed[i:i + 1], kind) @ c
            errs.append(np.hypot(*(pred_aim[0] - aimed[i])))
        return float(np.mean(errs))

    raw = float(np.mean(np.hypot(*(landed - aimed).T)))
    scores = {kind: loo(kind, k) for kind, k in KINDS}
    best = min(scores, key=scores.get)
    use = best if scores[best] < raw else None
    model = {"calib_id": cid, "kind": use, "n": n, "runs": runs,
             "coef": _fit(landed, aimed, use).tolist() if use else None,
             "error_before": raw, "error_after": scores[best] if use else raw, "scores": scores}
    if verbose:
        mean_off = (landed - aimed).mean(0)
        print(f"{n} taps from {runs} test run(s) with this calibration")
        print(f"average miss now: {100 * raw:.2f}% of screen  "
              f"(systematic part: {100 * mean_off[0]:+.2f}% right, {100 * mean_off[1]:+.2f}% down)")
        for kind, s in scores.items():
            print(f"  {kind:9s} correction -> expected miss {100 * s:.2f}%" if np.isfinite(s)
                  else f"  {kind:9s} correction -> needs more taps")
        print(f"Using: {use or 'no correction (it would not help yet)'}")
    return model


class Correction:
    """Applied by tap.py. Does nothing if there's no fit for the current calibration."""

    def __init__(self, calib):
        self.cid = calib_id(calib)
        self.model = None
        if CORR_FILE.exists():
            m = json.loads(CORR_FILE.read_text())
            if m.get("calib_id") == self.cid and m.get("kind"):
                self.model = m

    def aim(self, u, v):
        if not self.model:
            return u, v
        t = np.array([[u, v]], float)
        a = t + _feats(t, self.model["kind"]) @ np.array(self.model["coef"])
        return float(np.clip(a[0, 0], 0, 1)), float(np.clip(a[0, 1], 0, 1))

    def describe(self):
        if not self.model:
            return "no aim correction (run ./lr tap --random 20 to learn one)"
        m = self.model
        return (f"aim correction: {m['kind']} from {m['n']} taps, expected miss "
                f"{100 * m['error_before']:.2f}% -> {100 * m['error_after']:.2f}% of screen")


def refit_and_save(calib, verbose=True):
    m = fit(calib_id(calib), verbose)
    if m:
        CORR_FILE.write_text(json.dumps(m, indent=1))
    return m


if __name__ == "__main__":
    from common import load_calib

    p = argparse.ArgumentParser()
    p.add_argument("--reset", action="store_true")
    a = p.parse_args()
    if a.reset:
        CORR_FILE.unlink(missing_ok=True)
        print("Correction removed.")
    else:
        refit_and_save(load_calib())
