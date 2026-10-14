"""Show the robot what "back and forth" (the tap) and "up" (along the screen) mean:  ./lr teach-dirs

You drive the follower with the leader arm, touch the screen, then move the way you mean, pressing
Enter at each step. The robot stores those moves in its own joint terms (touch/directions.json):

  back : straight away from the glass   -> every tap goes in and out along this line
  up   : up along the screen            -> taps lean this way a little to cancel the arm sagging

Do it after ./lr touch-calib (or any time). Re-run if the screen or the arm moves.
"""

import argparse
import json
import queue
import threading
import time
from pathlib import Path

import numpy as np

from common import JOINTS, connect_follower, connect_leader

DIR_FILE = Path(__file__).resolve().parent / "directions.json"

p = argparse.ArgumentParser()
p.add_argument("--follower-port", required=True)
p.add_argument("--leader-port", required=True)
p.add_argument("--follower-id", default="my_follower")
p.add_argument("--leader-id", default="my_leader")
p.add_argument("--robot", default="so101")
p.add_argument("--cm", type=float, default=3.0, help="how far you move in each step (roughly)")
args = p.parse_args()

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


def follow_until_enter(msg):
    print("\n>>> " + msg + "\n    Press Enter when you are there (keep the arm still).", flush=True)
    while True:
        cmd = leader.get_action()
        robot.send_action(cmd)
        if not keys.empty():
            keys.get()
            return np.array([cmd[f"{j}.pos"] for j in JOINTS], dtype=float)
        time.sleep(1 / 50)


try:
    c = args.cm
    t1 = follow_until_enter("1/4  TOUCH the middle of the screen with the stylus (lightly, like a tap).")
    b = follow_until_enter(f"2/4  Pull STRAIGHT BACK away from the glass, about {c:.0f} cm. This is the tap direction.")
    t2 = follow_until_enter("3/4  TOUCH the middle of the screen again.")
    up = follow_until_enter(f"4/4  Move UP along the screen (toward its top edge), about {c:.0f} cm, "
                            "staying at the same distance from the glass.")
    d = {"back": (b - t1)[:5].round(3).tolist(), "up": (up - t2)[:5].round(3).tolist(), "cm": c,
         "touch": t1[:5].round(3).tolist(), "time": time.strftime("%Y-%m-%d %H:%M:%S")}
    DIR_FILE.write_text(json.dumps(d, indent=1))
    nb, nu = np.linalg.norm(d["back"]), np.linalg.norm(d["up"])
    print(f"\nSaved {DIR_FILE}: back = {nb:.1f} deg of joint motion for ~{c:.0f} cm, up = {nu:.1f} deg.")
    if nb < 2 or nu < 2:
        print("   (very small: did you move enough? You can run ./lr teach-dirs again.)")
    print("Taps now go in and out along your 'back' direction. Try: ./lr tap --test")
finally:
    robot.disconnect()
    leader.disconnect()
