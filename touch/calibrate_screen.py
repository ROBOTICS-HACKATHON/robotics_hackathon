"""Teach the arm where the screen is.

This starts a small web server. On the touch screen, open the address it
prints (http://<this computer>:8765/) full-screen. Then drive the follower with
the leader arm, for each target: home -> touch the target -> back up to home.
The page reports the exact touch spot, and the whole movement is recorded so
taps later replay (a smoothed version of) your own path. Enter also works.
"""

import argparse
import json
import queue
import threading
import time

import numpy as np

from common import CALIB_FILE, JOINTS, ForceGuard, connect_follower, connect_leader, read_joints
from touchserver import TouchServer, save_log

p = argparse.ArgumentParser()
p.add_argument("--follower-port", required=True)
p.add_argument("--leader-port", required=True)
p.add_argument("--follower-id", default="my_follower")
p.add_argument("--leader-id", default="my_leader")
p.add_argument("--robot", default="so101")
p.add_argument("--grid", type=int, default=3, help="dots per side")
p.add_argument("--http-port", type=int, default=8765)
p.add_argument("--max-load", type=int, default=0,
               help="force guard: follower stops pushing when motor load jumps this much (0 = off)")
p.add_argument("--lock", default="roll", choices=["roll", "wrist", "none"],
               help="what is held at its home angle after HOME: roll = wrist roll + gripper (wrist flex free), "
                    "wrist = wrist flex + roll + gripper, none = nothing")
args = p.parse_args()

fr = np.linspace(0.15, 0.85, args.grid)
targets = [(float(u), float(v)) for v in fr for u in fr]  # row by row, like calib.html

keys = queue.Queue()  # lines typed in the terminal
server = TouchServer(args.http_port, keys=keys, total=len(targets), grid=args.grid, mode="calib")
state = server.state


def read_keys():
    try:
        while True:
            keys.put(input())
    except EOFError:
        pass


threading.Thread(target=read_keys, daemon=True).start()

robot = connect_follower(args.follower_port, args.follower_id, args.robot)
leader = connect_leader(args.leader_port, args.leader_id, args.robot)


def drain(q):
    while not q.empty():
        q.get_nowait()


touches = server.touches


def pose(cmd):
    # Save what we COMMANDED, not where the follower ended up: the follower sags a bit
    # under gravity, and replaying the commanded pose reproduces the same sag.
    return np.array([cmd[f"{j}.pos"] for j in JOINTS], dtype=float)


LOCKED = {}  # joint -> fixed value, set after HOME
guard = ForceGuard(robot, limit=args.max_load, lag_s=1.0)
hold_until = 0.0


def mirror_once(path):
    global hold_until
    cmd = leader.get_action()
    cmd.update(LOCKED)  # wrist held at its home angle, so every target is reached the same way
    if time.monotonic() < hold_until or guard.check():
        # pushing too hard on the screen: the follower holds still instead of following deeper
        if time.monotonic() >= hold_until:
            print("    !! too much force on the screen: follower stopped. Lift the leader a little.", flush=True)
            hold_until = time.monotonic() + 0.4
        robot.send_action({f"{j}.pos": float(x) for j, x in zip(JOINTS, read_joints(robot))})
    else:
        robot.send_action(cmd)  # follower mirrors the leader
    q = pose(cmd)
    if path is not None:
        path.append([round(time.monotonic(), 3)] + q.round(3).tolist())
    return q


def wait_step(step, msg, use_touch, path=None):
    """Mirror the leader until the page reports a touch (or Enter).
    Records every pose into `path`. Returns (pose, (u, v) or None)."""
    state.update(step=step, msg=msg)
    how = ("touch the target (saved automatically) or press Enter" if use_touch
           else "press Enter (in the terminal, or Enter/Space on the page)")
    print(f">>> {msg}\n    Move with the leader arm, then {how}.", flush=True)
    drain(touches)
    drain(keys)
    while True:
        q = mirror_once(path)
        if use_touch and not touches.empty():
            t = touches.get_nowait()
            return q, (t["u"], t["v"], t.get("w"), t.get("h"))
        if not keys.empty():
            keys.get_nowait()
            return q, None
        time.sleep(1 / 50)


def record_return(step, home, touch, path, still=1.5, still_s=0.4):
    """After a touch: record the way back up until the arm is near home and held still (or Enter)."""
    state.update(step=step, msg="Saved! Now lift straight up and go back to home, then hold still")
    print("    now go back up to home and hold still (or press Enter)", flush=True)
    drain(keys)
    # "near home" = within 40% of the home-to-touch distance (max 8 degrees)
    near = max(1.5, min(8.0, 0.4 * np.abs(touch[:5] - home[:5]).max()))
    while True:
        q = mirror_once(path)
        if not keys.empty():
            keys.get_nowait()
            return
        recent = [r[1:6] for r in path if r[0] > time.monotonic() - still_s]
        settled = len(recent) > 5 and np.ptp(np.array(recent), axis=0).max() < still
        if np.abs(q[:5] - home[:5]).max() < near and settled:
            return
        time.sleep(1 / 50)


try:
    home, _ = wait_step(0, "HOME: hold the stylus a few cm ABOVE the middle of the screen, "
                           "wrist angled so the stylus points down at the screen", use_touch=False)
    if args.lock in ("roll", "wrist"):
        LOCKED.update({"wrist_roll.pos": float(home[4]), "gripper.pos": float(home[5])})
        if args.lock == "wrist":
            LOCKED["wrist_flex.pos"] = float(home[3])
        names = ", ".join(k.split(".")[0] for k in LOCKED)
        print(f"    LOCKED at home: {names}  (moving those on the leader does nothing now)", flush=True)
        if args.lock == "roll":
            print("    Wrist flex is free: try to keep the stylus at about the SAME angle for every target\n"
                  "    (very different wrist angles make taps between targets less accurate).", flush=True)
    points = []
    for i, (u, v) in enumerate(targets, 1):
        approach, retreat = [], []
        q, landed = wait_step(i, f"Target {i}/{len(targets)}: from home, touch target {i} with the stylus",
                              use_touch=True, path=approach)
        target_uv = (u, v)
        frame_px = None
        if landed:
            u, v, *frame_px = landed
        print(f"    saved target {i} at u={u:.3f} v={v:.3f}" + ("" if landed else " (dot centre)"))
        record_return(i, home, q, retreat)
        points.append({"u": u, "v": v, "q": q.tolist(), "approach": approach, "retreat": retreat,
                       "target": target_uv, "touched": bool(landed), "frame_px": frame_px})
        print(f"    recorded path: down {approach[-1][0] - approach[0][0]:.1f}s, "
              f"up {retreat[-1][0] - retreat[0][0]:.1f}s")
    calib = {"home": home.tolist(), "points": points, "wrist_locked": args.lock == "wrist",
             "locked": [k.split(".")[0] for k in LOCKED]}
    spread = float(np.ptp([p["q"][3] for p in points]))
    print(f"    wrist flex varied {spread:.0f} deg across the targets"
          + ("  (OK)" if spread <= 15 else "  (a lot: in-between taps may miss more; practice will compensate)"))
    CALIB_FILE.write_text(json.dumps(calib))
    log = save_log("calib", calib)
    print(f"Session log: {log}")
    state.update(step=len(targets) + 1, msg="Done! Calibration saved.")
    print(f"\nSaved {CALIB_FILE}. Check it with: ./lr tap --test")
    time.sleep(1.5)  # let the page show "Done"
finally:
    robot.disconnect()
    leader.disconnect()
    server.stop()
