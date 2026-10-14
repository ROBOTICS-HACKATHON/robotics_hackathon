"""Visual servoing: move the stylus tip onto the target using the 2-camera 3D view.

Coordinates are millimetres in the SCREEN frame, measured from the screen markers:
  x = right along the screen, y = down along the screen, z = height above the glass.
The target (the orange ring on the page) is at (u * W, v * H, 0): known exactly, nothing to detect.

How a joint move shifts the tip is LEARNED FROM THE ARM'S OWN MOVEMENT, not computed from a model:
  * probing: wiggle each arm joint by a couple of degrees at the hover point and watch the tip
    move in 3D -> a 3x4 matrix J (mm of tip motion per degree of each joint)
  * every later correction move updates J (Broyden update), so it keeps improving
J is stored per screen zone (the arm bends differently over different parts of the screen) in
touch/servo_jacobian.json, keyed by the calibration, so probing happens only once per zone.

Servo step at the hover point:
  predicted landing = tip now + J @ (the taught move from hover down to the touch pose)
  error = target - predicted landing  (x, y only; height is handled by the press)
  joint change = damped least squares of J for that error, limited by the safety envelope
Repeat until the predicted landing is within tol_mm, then go down as taught.
"""

import json
import time
from pathlib import Path

import numpy as np

from common import move_to, read_joints, settle

HERE = Path(__file__).resolve().parent
JAC_FILE = HERE / "servo_jacobian.json"
ARM = [0, 1, 2, 3]  # shoulder_pan, shoulder_lift, elbow_flex, wrist_flex (roll + gripper stay locked)


def screen_mm(vision, u, v, z=0.0):
    return np.array([u * vision.width_mm, v * vision.width_mm / vision.aspect, z], float)


class TipServo:
    def __init__(self, vision, cid, probe_deg=2.0, tol_mm=1.5, max_iters=4, damping=2.0, log=print):
        self.vision, self.cid, self.log = vision, cid, log
        self.probe_deg, self.tol_mm, self.max_iters, self.damping = probe_deg, tol_mm, max_iters, damping
        allj = json.loads(JAC_FILE.read_text()) if JAC_FILE.exists() else {}
        self.jac = {k: np.array(v, float) for k, v in allj.get(str(cid), {}).items()}
        self.last = None  # info about the last servo run (for logs)

    def save(self):
        allj = json.loads(JAC_FILE.read_text()) if JAC_FILE.exists() else {}
        allj[str(self.cid)] = {k: v.round(4).tolist() for k, v in self.jac.items()}
        JAC_FILE.write_text(json.dumps(allj))

    # ---- measuring ----
    def measure(self, looks=2):
        """Tip position in mm (screen frame), averaged over a couple of looks; None if unreliable."""
        pts = []
        for _ in range(looks):
            obs = self.vision.observe()
            self.last_obs = obs
            t3 = self.vision.tip3d(obs)
            if t3 is not None and t3["gap_mm"] <= self.vision.max_gap:
                u, v = t3["uv"]
                pts.append(screen_mm(self.vision, u, v, t3["height_mm"]))
        if not pts:
            return None
        pts = np.array(pts)
        if len(pts) > 1 and np.abs(pts[0] - pts[1]).max() > 4:  # the two looks disagree: arm still moving
            return None
        return pts.mean(0)

    def _goto(self, robot, q, guard):
        move_to(robot, q, 0.35)
        why = settle(robot, q, timeout=0.6, guard=guard)[1]
        time.sleep(0.12)  # let it stop shaking before the pictures
        return why

    # ---- learning how the arm moves the tip ----
    def probe(self, robot, q0, P0, guard):
        """Wiggle each joint a little and watch the tip: columns of J (mm per degree)."""
        J = np.zeros((3, len(ARM)))
        for c, j in enumerate(ARM):
            got = None
            for sign in (+1, -1):  # if one direction goes too close to the glass, try the other
                q = q0.copy()
                q[j] += sign * self.probe_deg
                if self._goto(robot, q, guard):
                    break
                P = self.measure()
                if P is not None and P[2] > 4:
                    got = (P - P0) / (sign * self.probe_deg)
                    break
            self._goto(robot, q0, guard)
            if got is None:
                return None
            J[:, c] = got
        return J

    @staticmethod
    def broyden(J, dq, dP):
        """Update J from one observed move (dq degrees -> dP mm)."""
        n = float(dq @ dq)
        if n < 0.25:  # too small a move to learn from (noise)
            return J
        return J + np.outer(dP - J @ dq, dq) / n

    # ---- servo ----
    def servo(self, robot, q_hover, q_touch, target_uv, zone, guard, safety=None, base=None):
        """base: the plain-calibration HOVER pose (the safety envelope keeps q near it)."""
        return self._servo(robot, q_hover, q_touch, target_uv, zone, guard, safety, base)

    def _servo(self, robot, q_hover, q_touch, target_uv, zone, guard, safety=None, base=None):
        """At the hover pose, move so the predicted landing hits the target. Returns the corrected
        (hover, touch) poses (same taught descent), or the inputs unchanged if it can't see well."""
        info = {"zone": zone, "iters": 0, "probed": False, "err_mm": None, "start_err_mm": None}
        self.last = info
        goal = screen_mm(self.vision, *target_uv)
        descent = (q_touch - q_hover)[ARM]
        if self._goto(robot, q_hover, guard):
            return q_hover, q_touch
        P = self.measure()
        if P is None:
            info["why"] = "tip not measured"
            return q_hover, q_touch
        key = str(zone)
        J = self.jac.get(key)
        if J is None:
            self.log(f"      servo: learning how the arm moves the tip in zone {zone} (a few small wiggles)")
            J = self.probe(robot, q_hover, P, guard)
            info["probed"] = True
            if J is None:
                info["why"] = "probe failed"
                return q_hover, q_touch
            P2 = self.measure()
            P = P if P2 is None else P2
        q = q_hover.copy()
        for it in range(self.max_iters + 1):
            land = P + J @ descent  # where the tip would touch down from here
            err = goal[:2] - land[:2]
            e = float(np.hypot(*err))
            if it == 0:
                info["start_err_mm"] = round(e, 1)
            info.update(err_mm=round(e, 1), tip_mm=P.round(1).tolist(), landing_mm=land.round(1).tolist(),
                        target_mm=goal.round(1).tolist(), iters=it)
            if e <= self.tol_mm or it == self.max_iters:
                break
            Jxy = J[:2]
            dq = Jxy.T @ np.linalg.solve(Jxy @ Jxy.T + self.damping * np.eye(2), err)  # damped least squares
            q_new = q.copy()
            q_new[ARM] += dq
            if safety is not None:  # limited steps, never far from what you taught
                q_new = safety.step(q, q_new - q, base)
            if self._goto(robot, q_new, guard):
                break
            P_new = self.measure()
            if P_new is None:
                break
            J = self.broyden(J, (q_new - q)[ARM], P_new - P)
            q, P = q_new, P_new
        self.jac[key] = J
        self.save()
        shift = q - q_hover
        return q, q_touch + shift
