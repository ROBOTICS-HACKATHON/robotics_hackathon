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


class ScreenModel:
    """Maps a screen point (u, v) to the joint angles that touch it.

    Fits a smooth quadratic surface per joint through the taught points.
    """

    def __init__(self, calib):
        pts = calib["points"]
        if len(pts) < 6:
            raise SystemExit("Need at least 6 taught points (use the 3x3 grid).")
        u = np.array([p["u"] for p in pts])
        v = np.array([p["v"] for p in pts])
        q = np.array([p["q"] for p in pts])
        self.coef, *_ = np.linalg.lstsq(_features(u, v), q, rcond=None)
        self.home = np.array(calib["home"], float)
        self.fit_error = np.abs(_features(u, v) @ self.coef - q).max(axis=0)

    def touch_pose(self, u, v):
        q = _features(u, v) @ self.coef
        q[-1] = self.home[-1]  # keep the gripper as it was at home
        return q
