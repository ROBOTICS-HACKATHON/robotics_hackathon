"""Re-set only the HOME pose (where the arm waits between taps), keeping the rest of the calibration.

Drive the follower with the leader arm to a spot well above the screen, then press Enter.
"""

import argparse
import json
import queue
import threading
import time

import numpy as np

from common import CALIB_FILE, JOINTS, connect_follower, connect_leader, load_calib

p = argparse.ArgumentParser()
p.add_argument("--follower-port", required=True)
p.add_argument("--leader-port", required=True)
p.add_argument("--follower-id", default="my_follower")
p.add_argument("--leader-id", default="my_leader")
p.add_argument("--robot", default="so101")
args = p.parse_args()

calib = load_calib()
robot = connect_follower(args.follower_port, args.follower_id, args.robot)
leader = connect_leader(args.leader_port, args.leader_id, args.robot)

keys = queue.Queue()


def read_keys():
    try:
        while True:
            keys.put(input())
    except EOFError:
        pass


threading.Thread(target=read_keys, daemon=True).start()

print(">>> Move the stylus to the new HOME: clearly above the screen, roughly over its middle.")
print("    The arm comes back here between every tap. Press Enter to save.", flush=True)
try:
    while True:
        cmd = leader.get_action()
        robot.send_action(cmd)
        if not keys.empty():
            break
        time.sleep(1 / 50)
    home = np.array([cmd[f"{j}.pos"] for j in JOINTS], dtype=float)
    # keep the gripper as it was calibrated so the stylus grip doesn't change
    home[-1] = calib["home"][-1]
    calib["home"] = home.tolist()
    CALIB_FILE.write_text(json.dumps(calib, indent=1))
    print(f"Saved new home in {CALIB_FILE}. Try: ./lr tap --test")
finally:
    robot.disconnect()
    leader.disconnect()
