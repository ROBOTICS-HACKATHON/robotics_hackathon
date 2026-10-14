"""Teach the arm where the screen is.

This starts a small web server. On the touch screen, open the address it
prints (http://<this computer>:8765/) full-screen. Then drive the follower with
the leader arm: each time the stylus touches a target, the page reports the
exact spot and the arm's pose is saved automatically. Enter also works.
"""

import argparse
import json
import mimetypes
import queue
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

from common import CALIB_FILE, JOINTS, connect_follower, connect_leader

HERE = Path(__file__).resolve().parent

p = argparse.ArgumentParser()
p.add_argument("--follower-port", required=True)
p.add_argument("--leader-port", required=True)
p.add_argument("--follower-id", default="my_follower")
p.add_argument("--leader-id", default="my_leader")
p.add_argument("--robot", default="so101")
p.add_argument("--grid", type=int, default=3, help="dots per side")
p.add_argument("--http-port", type=int, default=8765)
args = p.parse_args()

fr = np.linspace(0.15, 0.85, args.grid)
targets = [(float(u), float(v)) for v in fr for u in fr]  # row by row, like calib.html

# ---------- shared state between the web page and the robot loop ----------
touches = queue.Queue()  # (u, v) from the page
keys = queue.Queue()  # lines typed in the terminal
state = {"step": 0, "total": len(targets), "grid": args.grid, "msg": "Starting..."}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):  # keep the terminal quiet
        pass

    def _send(self, code, body, ctype="application/json"):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = self.path.split("?")[0]
        if path == "/state":
            return self._send(200, json.dumps(state).encode())
        name = "calib.html" if path in ("/", "") else path.lstrip("/")
        f = (HERE / name).resolve()
        if f.parent != HERE or not f.is_file():
            return self._send(404, b"not found", "text/plain")
        self._send(200, f.read_bytes(), mimetypes.guess_type(f.name)[0] or "application/octet-stream")

    def do_POST(self):
        if self.path != "/touch":
            return self._send(404, b"{}")
        data = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        touches.put((float(data["u"]), float(data["v"])))
        self._send(200, b"{}")


server = ThreadingHTTPServer(("0.0.0.0", args.http_port), Handler)
threading.Thread(target=server.serve_forever, daemon=True).start()


def read_keys():
    try:
        while True:
            keys.put(input())
    except EOFError:
        pass


threading.Thread(target=read_keys, daemon=True).start()

print(f"\nOn the touch screen, open:  http://localhost:{args.http_port}/   (full-screen, F11)")
print("  (separate tablet? use this computer's network address instead of localhost)\n")

robot = connect_follower(args.follower_port, args.follower_id, args.robot)
leader = connect_leader(args.leader_port, args.leader_id, args.robot)


def drain(q):
    while not q.empty():
        q.get_nowait()


def wait_step(step, msg, use_touch):
    """Mirror the leader until the page reports a touch (or Enter). Returns (pose, (u, v) or None)."""
    state.update(step=step, msg=msg)
    how = "touch the target (saved automatically) or press Enter" if use_touch else "press Enter"
    print(f">>> {msg}\n    Move with the leader arm, then {how}.", flush=True)
    drain(touches)
    drain(keys)
    while True:
        cmd = leader.get_action()
        robot.send_action(cmd)  # follower mirrors the leader
        landed, done = None, False
        if use_touch and not touches.empty():
            landed, done = touches.get_nowait(), True
        elif not keys.empty():
            keys.get_nowait()
            done = True
        if done:
            # Save what we COMMANDED, not where the follower ended up: the follower sags a bit
            # under gravity, and replaying the commanded pose reproduces the same sag.
            return np.array([cmd[f"{j}.pos"] for j in JOINTS], dtype=float), landed
        time.sleep(1 / 50)


try:
    home, _ = wait_step(0, "HOME: hold the stylus a few cm ABOVE the middle of the screen", use_touch=False)
    points = []
    for i, (u, v) in enumerate(targets, 1):
        q, landed = wait_step(i, f"Target {i}/{len(targets)}: touch target {i} with the stylus", use_touch=True)
        if landed:
            u, v = landed
        points.append({"u": u, "v": v, "q": q.tolist()})
        print(f"    saved target {i} at u={u:.3f} v={v:.3f}" + ("" if landed else " (dot centre)"))
        time.sleep(0.4)  # ignore a bounce from the same touch
    CALIB_FILE.write_text(json.dumps({"home": home.tolist(), "points": points}, indent=1))
    state.update(step=len(targets) + 1, msg="Done! Calibration saved.")
    print(f"\nSaved {CALIB_FILE}. Check it with: ./lr tap --test")
    time.sleep(1.5)  # let the page show "Done"
finally:
    robot.disconnect()
    leader.disconnect()
    server.shutdown()
