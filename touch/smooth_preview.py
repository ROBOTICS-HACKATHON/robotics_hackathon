"""Simulate how smoothly a tap moves, without the arm: old playback vs new, through a simple servo model.

Takes your recorded path for one calibration target, plays it the way tap.py does (most of the way
down, stop at the hover point, down to the glass, back up), and models each servo chasing its
commands with a limited acceleration. Plots joint speed over time and reports the roughness
(peak acceleration and jerk). Compare settings before trying them on the arm:

  python smooth_preview.py                         # old (LeRobot defaults) vs new (SERVO_ACCEL=50)
  python smooth_preview.py --accel 30 --smooth 0.15 --speed 0.8
  -> outputs/smoothness.png
"""

import argparse
import json
from pathlib import Path

import numpy as np

from common import CALIB_FILE, TEMPO, ScreenModel, path_timing

DEG_PER_UNIT = 8.79  # Feetech STS3215 acceleration register unit, deg/s^2
HZ = 50


def commands(model, u, v, speed, new, rng):
    """Commanded joint poses with their send times, like tap.py with the hover check."""
    touch = model.touch_pose(u, v)
    down, t_down = model.path("approach", u, v, touch)
    up, t_up = model.path("retreat", u, v, touch)
    hover = touch + (model.home - touch) * 0.15
    k = int(0.75 * (len(down) - 1))
    up = up + (1 - np.linspace(0, 1, len(up)))[:, None] * (touch - up[0])  # lift starts where it is (as tap.py)
    segs = [(down[:k + 1], 0.75 * t_down / speed, 0.0),
            (np.array([down[k], hover]), 0.4 / speed, 0.0),  # move_to the hover point
            (np.array([hover, hover]), 0.6, 0.0),  # settle + look
            (np.array([hover, touch]), 0.8 / speed, 0.0), (np.array([touch, touch]), 0.3, 0.0),
            (up, t_up / speed, 0.0)]
    t_now, out = 0.0, []
    for P, secs, slow in segs:
        x = np.linspace(0, 1, len(P))
        if new:  # real-clock pacing + minimum-jerk start/stop
            secs *= TEMPO
            prog = path_timing(slow, ease=True)
            n = max(2, int(secs * HZ))
            for i in range(1, n + 1):
                tt = i / n
                out.append((t_now + tt * secs, np.array([np.interp(prog(tt), x, P[:, j]) for j in range(P.shape[1])])))
            t_now += secs
        else:  # old: constant speed along the path, fixed sleeps + whatever the USB read/write takes
            n = max(2, int(secs * HZ))
            for i in range(n):
                t = i / (n - 1)
                t_now += 1 / HZ + rng.uniform(0.004, 0.022)  # bus time through WSL varies
                out.append((t_now, np.array([np.interp(t, x, P[:, j]) for j in range(P.shape[1])])))
    return out


def servo_sim(cmds, accel_deg, vmax=180.0, dt=0.002, wn=35.0):
    """Each joint follows its latest command like a damped spring (stiff, no overshoot of its own),
    with the servo's speed and acceleration limits."""
    T = np.arange(0, cmds[-1][0] + 0.8, dt)
    pos = cmds[0][1][:5].copy()
    vel = np.zeros(5)
    ci, target = 0, cmds[0][1][:5]
    P, V = [], []
    for t in T:
        while ci + 1 < len(cmds) and cmds[ci + 1][0] <= t:
            ci += 1
            target = cmds[ci][1][:5]
        acc = wn * wn * (target - pos) - 2 * wn * vel  # critically damped position loop
        vel = np.clip(vel + np.clip(acc, -accel_deg, accel_deg) * dt, -vmax, vmax)
        pos = pos + vel * dt
        P.append(pos.copy())
        V.append(vel.copy())
    return T, np.array(P), np.array(V)


def draw(results, out, w=1100, ph=300):
    """Joint speed plots, one panel per run (OpenCV only: no matplotlib in the image)."""
    import cv2
    cols = [(180, 119, 31), (14, 127, 255), (44, 160, 44), (40, 39, 214)]  # BGR
    names = ["shoulder pan", "shoulder lift", "elbow", "wrist flex"]
    tmax = max(r[1][-1] for r in results)
    vmax = max(np.abs(r[2][:, :4]).max() for r in results) * 1.1
    img = np.full((ph * len(results) + 40, w, 3), 255, np.uint8)
    L, R = 70, w - 20
    for k, (name, T, V, pa, jerk) in enumerate(results):
        top, bot = k * ph + 50, (k + 1) * ph - 10
        mid = (top + bot) // 2
        X = lambda t: int(L + (R - L) * t / tmax)
        Y = lambda v: int(mid - (bot - top) / 2 * v / vmax)
        cv2.rectangle(img, (L, top), (R, bot), (200, 200, 200), 1)
        cv2.line(img, (L, mid), (R, mid), (220, 220, 220), 1)
        for s in range(0, int(tmax) + 1):
            cv2.putText(img, f"{s}s", (X(s) - 8, bot + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (90, 90, 90), 1)
        cv2.putText(img, f"{vmax:.0f} deg/s", (2, top + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (90, 90, 90), 1)
        cv2.putText(img, f"-{vmax:.0f}", (14, bot), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (90, 90, 90), 1)
        cv2.putText(img, f"{name}   peak accel {pa:,.0f} deg/s^2   jerkiness {jerk:,.0f}", (L, top - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        step = max(1, len(T) // 2000)
        for j in range(4):
            pts = np.array([[X(t), Y(v)] for t, v in zip(T[::step], V[::step, j])], np.int32)
            cv2.polylines(img, [pts], False, cols[j], 2, cv2.LINE_AA)
    for j, n in enumerate(names):
        cv2.putText(img, n, (L + 150 * j, img.shape[0] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, cols[j], 2, cv2.LINE_AA)
    cv2.imwrite(out, img)


def roughness(T, V):
    A = np.gradient(V, T, axis=0)
    J = np.gradient(A, T, axis=0)
    return float(np.abs(A).max()), float(np.sqrt((J ** 2).sum(1).mean()))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--accel", type=int, default=80, help="new SERVO_ACCEL to try")
    ap.add_argument("--smooth", type=float, default=0.08, help="path smoothing (tap --smooth)")
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--target", type=int, default=5, help="which calibration target (1-9)")
    ap.add_argument("--calib", default=str(CALIB_FILE))
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent.parent / "outputs" / "smoothness.png"))
    a = ap.parse_args()
    calib = json.loads(Path(a.calib).read_text())
    pt = calib["points"][a.target - 1]
    u, v = (pt.get("target") or [pt["u"], pt["v"]])
    rng = np.random.default_rng(0)
    runs = {
        "before (LeRobot accel 254, old playback)": (ScreenModel(calib, smooth_s=0.08), False, 254),
        f"after (SERVO_ACCEL={a.accel}, smooth {a.smooth}, speed {a.speed})": (ScreenModel(calib, smooth_s=a.smooth), True, a.accel),
    }
    results = []
    for name, (model, new, acc) in runs.items():
        T, P, V = servo_sim(commands(model, u, v, a.speed if new else 1.0, new, rng), acc * DEG_PER_UNIT)
        pa, jerk = roughness(T, V)
        results.append((name, T, V, pa, jerk))
        print(f"{name:55s} peak accel {pa:8,.0f} deg/s^2   jerkiness {jerk:10,.0f}   duration {T[-1]:.1f}s")
    Path(a.out).parent.mkdir(exist_ok=True)
    draw(results, a.out)
    print(f"plot: {a.out}")
