"""Teach the arm where the screen is.

Open touch/calib.html full-screen on the touch screen first. It shows the
4 corner markers and 9 numbered dots. Then run this, drive the follower
with the leader arm, and press Enter at each step.
"""

import argparse
import json
import queue
import threading
import time

import numpy as np

from common import CALIB_FILE, connect_follower, connect_leader, read_joints

p = argparse.ArgumentParser()
p.add_argument("--follower-port", required=True)
p.add_argument("--leader-port", required=True)
p.add_argument("--follower-id", default="my_follower")
p.add_argument("--leader-id", default="my_leader")
p.add_argument("--robot", default="so101")
p.add_argument("--grid", type=int, default=3, help="dots per side (must match calib.html)")
args = p.parse_args()

fr = np.linspace(0.15, 0.85, args.grid)
targets = [(float(u), float(v)) for v in fr for u in fr]  # row by row, like calib.html

robot = connect_follower(args.follower_port, args.follower_id, args.robot)
leader = connect_leader(args.leader_port, args.leader_id, args.robot)

keys = queue.Queue()


def read_keys():
    while True:
        keys.put(input())


threading.Thread(target=read_keys, daemon=True).start()


def wait_for_enter(msg):
    print(f"\n>>> {msg}\n    Move with the leader arm, then press Enter.", flush=True)
    while True:
        robot.send_action(leader.get_action())  # follower mirrors the leader
        try:
            keys.get_nowait()
            time.sleep(0.3)  # let it settle
            return read_joints(robot)
        except queue.Empty:
            time.sleep(1 / 50)


try:
    home = wait_for_enter("HOME: hold the stylus a few cm above the middle of the screen")
    points = []
    for i, (u, v) in enumerate(targets, 1):
        q = wait_for_enter(f"Dot {i}/{len(targets)}: touch dot {i} lightly with the stylus tip")
        points.append({"u": u, "v": v, "q": q.tolist()})
        print(f"    saved dot {i}")
    CALIB_FILE.write_text(json.dumps({"home": home.tolist(), "points": points}, indent=1))
    print(f"\nSaved {CALIB_FILE}. Check it with: ./lr tap --test")
finally:
    robot.disconnect()
    leader.disconnect()
