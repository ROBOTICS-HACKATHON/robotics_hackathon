"""Shared helpers for the touch-screen tools.

Screen coordinates are always (u, v) in 0..1:
  u = 0 left edge, 1 right edge; v = 0 top edge, 1 bottom edge
measured between the outer corners of the 4 ArUco markers on the page.
"""

import json
import time
from pathlib import Path

import numpy as np

JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
CALIB_FILE = Path(__file__).resolve().parent / "screen_calib.json"


def connect_follower(port, robot_id, robot_type="so101", hold=True):
    """hold=True keeps the motors stiff after disconnect so the arm doesn't drop onto the screen."""
    from lerobot.robots.so_follower import SO100Follower, SO100FollowerConfig, SO101Follower, SO101FollowerConfig

    cls, cfg = (SO100Follower, SO100FollowerConfig) if robot_type == "so100" else (SO101Follower, SO101FollowerConfig)
    robot = cls(cfg(port=port, id=robot_id, disable_torque_on_disconnect=not hold))
    robot.connect()
    return robot


def connect_leader(port, leader_id, robot_type="so101"):
    from lerobot.teleoperators.so_leader import SO100Leader, SO100LeaderConfig, SO101Leader, SO101LeaderConfig

    cls, cfg = (SO100Leader, SO100LeaderConfig) if robot_type == "so100" else (SO101Leader, SO101LeaderConfig)
    leader = cls(cfg(port=port, id=leader_id))
    leader.connect()
    return leader


def read_joints(robot):
    obs = robot.get_observation()
    return np.array([obs[f"{j}.pos"] for j in JOINTS], dtype=float)


def send_joints(robot, q):
    robot.send_action({f"{j}.pos": float(v) for j, v in zip(JOINTS, q)})


def move_to(robot, target, seconds=1.0, hz=50):
    """Smooth joint-space move from the current pose to `target`."""
    start = read_joints(robot)
    steps = max(1, int(seconds * hz))
    for i in range(1, steps + 1):
        s = i / steps
        s = s * s * (3 - 2 * s)  # ease in/out
        send_joints(robot, start + (target - start) * s)
        time.sleep(1 / hz)


def load_calib(path=CALIB_FILE):
    if not Path(path).exists():
        raise SystemExit(f"No screen calibration at {path}. Run: ./lr touch-calib")
    return json.loads(Path(path).read_text())


def _features(u, v):
    u, v = np.asarray(u, float), np.asarray(v, float)
    return np.stack([np.ones_like(u), u, v, u * v, u * u, v * v], axis=-1)


N_SAMPLES = 80  # every recorded path is resampled to this many poses


def smooth_path(rec, sigma_s=0.08, trim_deg=1.0, hz=50):
    """Clean one recorded leader path.

    rec: list of [time, q0..q5]. Steps: resample to an even 50 Hz, remove hand
    shake with a Gaussian filter of `sigma_s` seconds, cut the still parts at the
    start/end (you waiting at home), resample to N_SAMPLES.
    Returns (path (N_SAMPLES, 6), duration in seconds).
    """
    a = np.asarray(rec, float)
    t, q = a[:, 0], a[:, 1:]
    tu = np.arange(t[0], t[-1] + 1e-9, 1 / hz)
    qu = np.stack([np.interp(tu, t, q[:, j]) for j in range(q.shape[1])], 1)

    sig = sigma_s * hz
    if sig > 0 and len(qu) > 3:
        r = int(3 * sig) + 1
        k = np.exp(-0.5 * (np.arange(-r, r + 1) / sig) ** 2)
        k /= k.sum()
        pad = np.pad(qu, ((r, r), (0, 0)), mode="edge")
        qu = np.stack([np.convolve(pad[:, j], k, mode="valid") for j in range(qu.shape[1])], 1)

    # cut the still parts at the start/end; "still" = within 5% of the whole movement
    thr = max(trim_deg, 0.05 * np.ptp(qu[:, :5], axis=0).max())
    moved = np.abs(qu - qu[0])[:, :5].max(1) > thr
    s = max(0, int(np.argmax(moved)) - 1) if moved.any() else 0
    moved_end = np.abs(qu - qu[-1])[:, :5].max(1) > thr
    e = len(qu) - max(0, int(np.argmax(moved_end[::-1])) - 1) if moved_end.any() else len(qu)
    qu = qu[s:e] if e - s > 2 else qu

    duration = max(len(qu) - 1, 1) / hz
    x, xn = np.linspace(0, 1, len(qu)), np.linspace(0, 1, N_SAMPLES)
    return np.stack([np.interp(xn, x, qu[:, j]) for j in range(qu.shape[1])], 1), duration


class ScreenModel:
    """Maps a screen point (u, v) to the joint angles that touch it, and to the
    paths down to it and back up to home.

    Fits a smooth quadratic surface per joint through the taught points. Paths are
    fitted the same way, sample by sample, so a point between targets gets a
    blend of the recorded paths of its neighbours.
    """

    def __init__(self, calib, smooth_s=0.08):
        self.pts = pts = calib["points"]
        if len(pts) < 6:
            raise SystemExit("Need at least 6 taught points (use the 3x3 grid).")
        self.u = np.array([p["u"] for p in pts])
        self.v = np.array([p["v"] for p in pts])
        q = np.array([p["q"] for p in pts])
        F = _features(self.u, self.v)
        self.coef, *_ = np.linalg.lstsq(F, q, rcond=None)
        self.home = np.array(calib["home"], float)
        self.fit_error = np.abs(F @ self.coef - q).max(axis=0)
        self.has_paths = all(len(p.get("approach", [])) > 3 and len(p.get("retreat", [])) > 3 for p in pts)
        if self.has_paths:
            self.set_smoothing(smooth_s)

    def set_smoothing(self, smooth_s):
        """Re-clean the recorded paths with a different smoothing (no recalibration needed)."""
        F = _features(self.u, self.v)
        self.path_coef, self.path_time, self.shake = {}, {}, {}
        for kind in ("approach", "retreat"):
            cleaned = [smooth_path(p[kind], smooth_s) for p in self.pts]
            raw = [smooth_path(p[kind], 0) for p in self.pts]
            Y = np.array([c[0] for c in cleaned]).reshape(len(self.pts), -1)
            self.path_coef[kind], *_ = np.linalg.lstsq(F, Y, rcond=None)
            self.path_time[kind] = float(np.median([c[1] for c in cleaned]))
            # how much shake was removed (degrees, mean over joints and samples)
            self.shake[kind] = float(np.mean([np.abs(c[0] - r[0])[:, :5].mean() for c, r in zip(cleaned, raw)]))

    def touch_pose(self, u, v):
        q = _features(u, v) @ self.coef
        q[-1] = self.home[-1]  # keep the gripper as it was at home
        return q

    def path(self, kind, u, v):
        """Smoothed path for 'approach' (home -> touch) or 'retreat' (touch -> home).
        Returns (poses (N, 6), duration_s)."""
        P = (_features(u, v) @ self.path_coef[kind]).reshape(N_SAMPLES, -1)
        touch = self.touch_pose(u, v)
        start, end = (self.home, touch) if kind == "approach" else (touch, self.home)
        s = np.linspace(0, 1, N_SAMPLES)[:, None]
        P = P + (1 - s) * (start - P[0]) + s * (end - P[-1])  # start/end exactly at home and touch
        P[:, -1] = self.home[-1]  # gripper stays closed on the stylus
        return P, self.path_time[kind]


def follow_path(robot, P, seconds, hz=50):
    """Play a path of poses evenly over `seconds`."""
    steps = max(2, int(seconds * hz))
    x, xn = np.linspace(0, 1, len(P)), np.linspace(0, 1, steps)
    for t in xn:
        send_joints(robot, np.array([np.interp(t, x, P[:, j]) for j in range(P.shape[1])]))
        time.sleep(1 / hz)
