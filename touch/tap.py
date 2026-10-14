"""Tap points on the screen.

  python tap.py 0.25 0.40 0.75 0.40      # tap (u,v) pairs, 0..1 screen coords
  python tap.py --json '[[0.25,0.4],[0.75,0.4]]'
  python tap.py --test                    # tap every calibration dot to check accuracy
  python tap.py --home                    # just go to the home pose

Use from Python (e.g. from your agent):
  from tap import Tapper
  with Tapper(port="/dev/ttyACM0") as t:
      t.tap_many([(0.25, 0.4), (0.75, 0.4)])
"""

import argparse
import json
import time

import numpy as np

from common import (ForceGuard, SafetyEnvelope, countdown, ScreenModel, connect_follower, follow_path, load_calib, move_to,
                    read_joints, set_torque_limit, settle)


class Tapper:
    def __init__(self, port, robot_id="my_follower", robot_type="so101",
                 lift=0.3, press=0.04, speed=1.0, via_home=True, smooth=0.08, own_path=True, correct=True, dwell=0.25,
                 max_load=0, torque=100, learned=True, vision=None, hover=0.15, vision_gain=0.8, servo=True,
                 tap_from=0.85):
        """
        vision: a TipVision (0, 1 or 2 cameras); hover-check before each tap when it has learned enough
        hover: how high to hover for the check (fraction of the way from the touch pose to home)
        max_load: force guard, how much the motor load may jump on contact (0..1000, 0 = off)
        torque: cap on shoulder/elbow/wrist strength in percent (100 = normal)
        dwell: seconds the stylus stays on the screen (longer = more reliable touches)
        correct: apply the aim correction learned from measured tests (see correction.py)
        own_path: replay your recorded (smoothed) leader paths if the calibration has them
        smooth: smoothing strength in seconds (0 = raw recording, 0.15 = very smooth)
        via_home: go back to the home pose after every tap (stylus never drags across the screen)
        lift:  how high the stylus hovers right before/after a tap (0..1, fraction of the way to home)
        press: how far past the taught touch point to push (raise if taps don't register)
        speed: >1 faster, <1 slower
        """
        calib = load_calib()
        self.model = ScreenModel(calib, smooth_s=smooth)
        self.base_model = ScreenModel(calib, smooth_s=smooth)  # plain calibration, never replaced
        self.safety = SafetyEnvelope(calib)
        self.own_path = own_path and self.model.has_paths
        from correction import Correction, calib_id
        from learner import Experience, LearnedModel, ZoneBandit, zone_centers
        self.exp = Experience(calib_id(calib))
        self.learned = None
        if learned and len(self.exp) >= 5 and self.own_path:
            # the arm's own learned "how to hit (u, v)" replaces the screen-space offset
            self.learned = LearnedModel(calib, self.exp)
            self.model.touch_pose = self.learned.touch_pose
            correct = False
        self.zones = (ZoneBandit(self.exp, zone_centers(calib))
                      if learned and self.exp.data.get("zones") else None)
        self.correction = Correction(calib) if correct else Correction({"points": []})
        self.last_command = None
        self.last_touch_q = None
        self.vision, self.hover_frac, self.vision_gain = vision, hover, vision_gain
        self.tap_from = tap_from  # where on your recorded path down it stops above the target (0.85 = last 15% is the tap)
        # 2 cameras: steer the tip onto the target in mm, learning from the arm's own moves (servo.py)
        self.servo = None
        if servo and vision is not None and getattr(vision, "n", 0) >= 2:
            from servo import TipServo
            self.servo = TipServo(vision, calib_id(calib))
        self.zone_centers = zone_centers(calib)
        self.last_hover_obs, self.last_vision = None, None
        self.dwell = dwell
        self.last_settle_err = None
        self.last_force_stop, self.last_peak_load, self.last_stop = False, 0.0, None
        self.lift, self.press, self.speed, self.via_home = lift, press, speed, via_home
        self.base_speed = speed
        self.robot = connect_follower(port, robot_id, robot_type)
        self.at_home = False
        self.guard = ForceGuard(self.robot, limit=max_load)
        self.torque = torque
        if torque < 100:
            set_torque_limit(self.robot, torque)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.go_home()
        if self.torque < 100:
            set_torque_limit(self.robot, 100)
        self.robot.disconnect()

    def go_home(self, seconds=1.5):
        if not self.at_home:
            move_to(self.robot, self.model.home, seconds / self.speed)
            self.at_home = True

    def tap(self, u, v):
        if not (0 <= u <= 1 and 0 <= v <= 1):
            raise ValueError(f"({u}, {v}) is outside the screen")
        if self.zones is not None:  # per-zone settings learned from rewards in practice
            self.press, slow = self.zones.best(self.zones.zone(u, v))
            self.speed = self.base_speed * slow
        u, v = self.correction.aim(u, v)  # aim slightly off to cancel the learned miss
        self.last_command = (u, v)
        touch = self.model.touch_pose(u, v)
        if self.own_path:
            return self._tap_own_path(u, v, touch)
        above = touch + (self.model.home - touch) * self.lift
        pressed = touch + (touch - above) * self.press / max(self.lift, 1e-6)
        if self.via_home:
            self.go_home()  # start every tap from home, above the screen
        self.at_home = False
        move_to(self.robot, above, 0.7 / self.speed)    # over the target
        move_to(self.robot, pressed, 0.35 / self.speed)  # down
        time.sleep(0.12)
        move_to(self.robot, above, 0.3 / self.speed)     # straight back up
        if self.via_home:
            self.go_home(0.7)                            # and back home before the next one

    def _tap_own_path(self, u, v, touch):
        """home -> your smoothed path down -> small press -> your smoothed path back up -> home.
        The force guard stops the push the moment the motors strain against the glass."""
        base = self.base_model.touch_pose(u, v)
        touch = self.safety.clamp(touch, base)  # never far from what you taught
        self.go_home()
        self.at_home = False
        down, t_down = self.model.path("approach", u, v, touch)
        up, t_up = self.model.path("retreat", u, v, touch)
        self.last_touch_q = touch
        g = self.guard
        g.peak = 0.0
        self.last_hover_obs, self.last_vision = None, None
        self.last_hover_q, self.last_press_height = None, None
        self.tap_log, self._t0 = [], time.time()
        self._ev("start", touch_cmd=touch)
        if self.vision is not None and self.vision.enabled:
            # most of your path, then hover just above the target, look, correct, then straight down
            k = int(0.75 * (len(down) - 1))
            why = follow_path(self.robot, down[:k + 1], 0.75 * t_down / self.speed, guard=g, guard_from=1.1)
            if not why and self.servo is not None:
                from learner import zone_of
                hover0 = touch + (self.model.home - touch) * self.hover_frac
                btouch = self.base_model.touch_pose(u, v)
                bhover = btouch + (self.model.home - btouch) * self.hover_frac
                h2, t2 = self.servo.servo(self.robot, hover0, touch, (u, v), zone_of(u, v, self.zone_centers),
                                          g, self.safety, bhover)
                info = self.servo.last or {}
                if info.get("err_mm") is not None:  # it saw the tip and steered: use it
                    touch = self.safety.clamp(t2, btouch)
                    self.last_hover_obs = getattr(self.servo, "last_obs", None)
                    W, H = self.vision.width_mm, self.vision.width_mm / self.vision.aspect
                    lm = info["landing_mm"]
                    self.last_vision = {"landing": (lm[0] / W, lm[1] / H), "height_mm": info["tip_mm"][2],
                                        "mode": "3d", "servo": info}
                else:
                    touch, why = self._hover_check(u, v, touch)
            elif not why:
                touch, why = self._hover_check(u, v, touch)
            if not why:
                hover = touch + (self.model.home - touch) * self.hover_frac
                self.last_hover_q = hover
                self._ev("above_target", cmd=hover)
                why = follow_path(self.robot, np.array([hover, touch]), 0.8 / self.speed, guard=g, guard_from=0.3)
                self._ev("down", cmd=touch, why=why)
            approach_dir = touch - (touch + (self.model.home - touch) * self.hover_frac)
            self.last_touch_q = touch
        else:
            # TAP, not drag: your path down to just above the target, STOP there (the arm catches up, no
            # sideways speed left), then straight down onto the target. Never arrives while still moving sideways.
            k = int(self.tap_from * (len(down) - 1))
            hover = down[k]
            self.last_hover_q = hover
            why = follow_path(self.robot, down[:k + 1], self.tap_from * t_down / self.speed, guard=g, guard_from=1.1)
            if not why:
                why = settle(self.robot, hover, timeout=0.6, guard=g)[1]
            self._ev("above_target", cmd=hover, why=why)
            if not why:
                why = follow_path(self.robot, np.array([hover, touch]), max(0.5, 0.35 * t_down) / self.speed,
                                  guard=g, guard_from=0.3)
            self._ev("down", cmd=touch, why=why)
            approach_dir = (touch - hover) * (0.4 / max(1e-6, 1 - self.tap_from))  # same press depth as before
        self.last_settle_err = None
        if not why:
            # not touched yet: push a little further, and make sure the stylus really arrives
            pressed = touch + approach_dir * self.press
            why = follow_path(self.robot, np.array([touch, pressed]), 0.15, guard=g, guard_from=0.0,
                              reset_guard=False)
            if not why:
                self.last_settle_err, why = settle(self.robot, pressed, timeout=self.dwell + 0.8, guard=g)
        self.last_stop = why
        self._ev("pressed", why=why)
        self.last_force_stop, self.last_peak_load = why == "force", g.peak
        if why != "touch" and self.vision is not None and self.vision.enabled:
            # no touch registered (yet): the cameras measure how far above the glass the tip stopped.
            # Taken during the dwell we wait anyway, so it costs no time.
            t0 = time.time()
            est = self.vision.estimate(self.vision.observe())
            if est.get("mode") == "3d" and not est.get("inconsistent") and est.get("height_mm") is not None:
                self.last_press_height = float(est["height_mm"])
            time.sleep(max(0.0, self.dwell - (time.time() - t0)))
            why_wait = 0.0
        else:
            why_wait = 0.05 if why == "touch" else self.dwell
        # touched already: lift off almost right away (don't keep pressing); otherwise hold for dwell
        time.sleep(why_wait)
        # lift STRAIGHT back up the line it came down (no sliding on the glass), then your path home
        lift_to = self.last_hover_q if self.last_hover_q is not None else touch + (self.model.home - touch) * self.hover_frac
        self._ev("lift_start")
        follow_path(self.robot, np.array([read_joints(self.robot), lift_to]), 0.35 / self.speed)
        settle(self.robot, lift_to, timeout=0.3)
        self._ev("lifted", cmd=lift_to)
        follow_path(self.robot, up, t_up / self.speed, start=read_joints(self.robot))
        self._ev("home")
        self.at_home = True

    def _ev(self, name, cmd=None, **kw):
        """One line of the per-tap motion log: time, where the arm was told to be, where it really is."""
        try:
            act = read_joints(self.robot)
        except Exception:
            act = None
        e = {"ev": name, "t": round(time.time() - getattr(self, "_t0", time.time()), 3),
             "act": None if act is None else [round(float(x), 2) for x in act]}
        if cmd is not None:
            e["cmd"] = [round(float(x), 2) for x in cmd]
            if act is not None:
                e["err_deg"] = round(float(np.abs(np.asarray(cmd)[:5] - act[:5]).max()), 2)
        for k, x in kw.items():
            e[k] = [round(float(y), 2) for y in x] if isinstance(x, np.ndarray) else x
        getattr(self, "tap_log", []).append(e)

    def _jacobian(self, u, v, h=0.01):
        tp = self.model.touch_pose
        return np.stack([(tp(u + h, v) - tp(u - h, v)) / (2 * h), (tp(u, v + h) - tp(u, v - h)) / (2 * h)], 1)

    def _hover_check(self, u, v, touch, iters=2, tol=0.004):
        """Hover above the target, see where the tip would land, correct the joint pose.
        Only corrects once the vision model has learned enough; always records what it saw."""
        J = self._jacobian(u, v)
        why = None
        for it in range(iters + 1):
            hover = touch + (self.model.home - touch) * self.hover_frac
            move_to(self.robot, hover, 0.4 / self.speed)
            why = settle(self.robot, hover, timeout=0.6, guard=self.guard)[1]
            if why:
                break
            time.sleep(0.15)  # let the arm stop shaking before the picture
            obs = self.vision.observe()
            est = self.vision.estimate(obs)
            self.last_hover_obs, self.last_vision = obs, est
            if not (self.vision.ready and est["landing"] is not None) or it == iters:
                break
            err = np.array(est["landing"]) - [u, v]
            if np.hypot(*err) < tol:
                break
            if np.hypot(*err) > 0.08:  # camera says >8% of the screen off: don't trust it
                est["rejected"] = True
                break
            # aim the joints so the tip lands on target, in small limited steps
            touch = self.safety.step(touch, self.vision_gain * (J @ -err), self.base_model.touch_pose(u, v))
            touch[-1] = self.model.home[-1]
        return touch, why

    def stop_on_touch(self, server):
        """While a TouchServer is running (tests, practice), stop pressing as soon as it reports a touch."""
        self.guard.touch_event = server.touched

    def tap_pose(self, u, v, q):
        """Tap with an exact joint pose q for screen point (u, v) (used by the learning loop)."""
        if not self.own_path:
            raise SystemExit("Learning needs a calibration with recorded paths: run ./lr touch-calib")
        self._tap_own_path(u, v, np.asarray(q, float))

    def tap_many(self, points):
        for u, v in points:
            self.tap(u, v)


def measure_summary(results):
    """Errors between where we aimed and where the page saw the touch."""
    px = [r["error_px"] for r in results if r.get("error_px") is not None]
    frac = [r["error_frac"] for r in results if r.get("error_frac") is not None]
    missed = sum(1 for r in results if not r["touches"])
    s = {"taps": len(results), "no_touch": missed}
    if frac:
        s.update(mean_error_pct=100 * float(np.mean(frac)), max_error_pct=100 * float(np.max(frac)))
    if px:
        s.update(mean_error_px=float(np.mean(px)), max_error_px=float(np.max(px)),
                 within_25px=sum(e <= 25 for e in px), within_50px=sum(e <= 50 for e in px))
        dx = [r["offset_px"][0] for r in results if r.get("offset_px")]
        dy = [r["offset_px"][1] for r in results if r.get("offset_px")]
        s.update(mean_offset_px=[float(np.mean(dx)), float(np.mean(dy))])  # same-direction miss = systematic
    return s


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("coords", nargs="*", type=float, help="u1 v1 u2 v2 ...")
    p.add_argument("--json", help='list of [u, v] pairs')
    p.add_argument("--test", action="store_true", help="tap every calibration target and measure")
    p.add_argument("--random", type=int, default=0, metavar="N", help="tap N random points and measure")
    p.add_argument("--no-measure", action="store_true", help="tests without the page reporting touches")
    p.add_argument("--http-port", type=int, default=8765)
    p.add_argument("--delay", type=float, default=5, help="seconds to switch to the page before starting")
    p.add_argument("--home", action="store_true")
    p.add_argument("--port", required=True)
    p.add_argument("--id", default="my_follower")
    p.add_argument("--robot", default="so101")
    p.add_argument("--press", type=float, default=0.04)
    p.add_argument("--lift", type=float, default=0.3)
    p.add_argument("--speed", type=float, default=1.0)
    p.add_argument("--no-home", action="store_true", help="don't return home between taps (faster)")
    p.add_argument("--smooth", type=float, default=0.08, help="path smoothing in seconds (0 = raw)")
    p.add_argument("--simple-path", action="store_true", help="ignore your recorded paths")
    p.add_argument("--max-load", type=int, default=0,
                   help="force guard: stop pushing when motor load jumps this much (0..1000, 0 = off)")
    p.add_argument("--torque", type=int, default=100, help="cap arm strength in percent while tapping")
    p.add_argument("--dwell", type=float, default=0.25, help="seconds the stylus stays on the screen")
    p.add_argument("--tap-from", type=float, default=0.85,
                   help="stop on your recorded path this far down (0..1), then go straight down onto the target")
    p.add_argument("--cameras", default="", help="tip vision cameras: 'none', one, or two comma-separated")
    p.add_argument("--tip-color", default="pink")
    p.add_argument("--screen-mm", type=float, default=300.0, help="marker-to-marker screen width (for mm heights)")
    p.add_argument("--no-vision", action="store_true", help="tap without the camera hover-check")
    p.add_argument("--no-learned", action="store_true", help="ignore the learned model (use calibration only)")
    p.add_argument("--no-correct", action="store_true", help="don't apply the learned aim correction")
    p.add_argument("--no-learn", action="store_true", help="don't update the correction after a test")
    a = p.parse_args()

    if a.home:
        pts = []
    elif a.test:
        pts = [(d["u"], d["v"]) for d in load_calib()["points"]]
    elif a.random:
        # random spots inside the calibrated area (outside it the model has to guess)
        cal = load_calib()["points"]
        lo = [min(p["u"] for p in cal), min(p["v"] for p in cal)]
        hi = [max(p["u"] for p in cal), max(p["v"] for p in cal)]
        rng = np.random.default_rng()
        pts = [tuple(x) for x in rng.uniform(lo, hi, size=(a.random, 2)).round(3)]
    elif a.json:
        pts = json.loads(a.json)
    else:
        if len(a.coords) % 2:
            raise SystemExit("Give coordinates in u v pairs")
        pts = list(zip(a.coords[::2], a.coords[1::2]))

    srv = None
    if (a.test or a.random) and not a.no_measure:
        from touchserver import TouchServer, save_log
        srv = TouchServer(a.http_port, mode="test", msg="Tap test: blue dots show where the robot touched")
        countdown(a.delay, "switch to the page on the touch screen (reload it, full-screen F11)")

    results = []
    vision = None
    if a.cameras and not a.no_vision:
        from correction import calib_id as _cid
        from tipvision import TipVision
        vision = TipVision(a.cameras, a.tip_color, width_mm=a.screen_mm, cid=_cid(load_calib()))
    with Tapper(a.port, a.id, a.robot, a.lift, a.press, a.speed, via_home=not a.no_home, tap_from=a.tap_from,
                smooth=a.smooth, own_path=not a.simple_path, correct=not a.no_correct,
                dwell=a.dwell, max_load=a.max_load, torque=a.torque, learned=not a.no_learned,
                vision=vision) as t:
        if vision is not None:
            print(vision.describe())
        print("force guard: " + (f"on (stops at a load jump of {a.max_load})" if t.guard.ok else "OFF"))
        if srv:
            t.stop_on_touch(srv)
            print("stop-on-touch: on (stops pressing the moment the page reports the touch)")
        print(f"learned model: {len(t.exp)} past touches, press {t.press}" if t.learned else t.correction.describe())
        err = ", ".join(f"{e:.1f}" for e in t.model.fit_error[:5])
        print(f"Calibration fit error per joint (degrees): {err}")
        if t.model.wrist_spread > 15:
            print(f"note: the wrist flex varied {t.model.wrist_spread:.0f} deg between calibration targets; taps "
                  "between targets may miss more (practice compensates, or recalibrate with a steadier wrist).")
        if t.own_path:
            m = t.model
            print(f"Replaying your recorded paths: down {m.path_time['approach'] / a.speed:.1f}s, "
                  f"up {m.path_time['retreat'] / a.speed:.1f}s, shake removed ~{m.shake['approach']:.2f} deg")
        else:
            print("Using calculated paths (no recorded paths in the calibration)")
        for i, (u, v) in enumerate(pts, 1):
            if srv:
                srv.set(step=i, target=[u, v], msg=f"Tap {i}/{len(pts)}: aiming at the orange ring")
                srv.drain()
            t.tap(u, v)
            line = f"tap {i}: aim u={u:.3f} v={v:.3f}"
            if srv:
                time.sleep(0.3)  # give the page time to report
                got = srv.all_touches()
                traces = srv.all_traces()
                r = {"target": [u, v], "command": list(t.last_command), "touches": got,
                     "contacts": traces, "motion": t.tap_log,
                     "drag_px": max((c.get("drag_px", 0) for c in traces), default=None),
                     "settle_err_deg": t.last_settle_err, "force_stop": t.last_force_stop,
                     "peak_load_jump": t.last_peak_load, "stopped_by": t.last_stop,
                     "vision": t.last_vision, "hover_obs": t.last_hover_obs,
                     "press_height_mm": t.last_press_height}
                if got and not a.no_learn and t.vision is not None and t.last_hover_obs:
                    t.vision.learn(t.last_hover_obs, (got[0]["u"], got[0]["v"]))
                if got and not a.no_learn and t.last_touch_q is not None:
                    t.exp.add(t.last_touch_q, (got[0]["u"], got[0]["v"]), (u, v), None, t.press, {"source": "test"})
                if not a.no_learn:
                    from learner import reward, zone_centers, zone_of
                    _e = (np.hypot((got[0]["u"] - u) * (got[0].get("w") or 1000), (got[0]["v"] - v) * (got[0].get("h") or 1000))
                          if got else None)
                    t.exp.data.setdefault("attempts", []).append(
                        {"zone": zone_of(u, v, zone_centers(load_calib())), "try": 1, "err_px": _e,
                         "touched": bool(got), "press": t.press, "slow": t.speed / t.base_speed,
                         "reward": reward(_e or 0, bool(got), len(got), t.last_force_stop, 15),
                         "target": [u, v], "t": time.time(), "source": "test"})
                if got:
                    g = got[0]
                    du, dv = g["u"] - u, g["v"] - v
                    r["error_frac"] = float(np.hypot(du, dv))
                    if g.get("w") and g.get("h"):
                        r["offset_px"] = [du * g["w"], dv * g["h"]]
                        r["error_px"] = float(np.hypot(*r["offset_px"]))
                        line += f"  landed u={g['u']:.3f} v={g['v']:.3f}  off by {r['error_px']:.0f}px"
                    else:
                        line += f"  landed u={g['u']:.3f} v={g['v']:.3f}"
                    if len(got) > 1:
                        line += f"  ({len(got)} touches: bounced or dragged)"
                else:
                    se = t.last_settle_err
                    line += "  NO TOUCH registered" + (" (force guard stopped it: try --max-load 300)" if t.last_force_stop else "") + (
                        f" (arm stopped {se:.1f} deg short: blocked by the screen or too weak)" if se and se > 1.5
                        else " (arm reached its pose: press deeper with --press 0.08)")
                if r.get("drag_px") is not None:
                    c = max(r["contacts"], key=lambda c: c.get("drag_px", 0))
                    line += f"  drag {r['drag_px']:.0f}px in {c.get('ms')}ms" + ("  <- DRAGGED" if r["drag_px"] > 15 else "")
                results.append(r)
            if t.last_force_stop:
                line += "  [force guard stopped the push]"
            if t.last_vision and t.last_vision.get("landing"):
                lv = t.last_vision
                line += (f"  [cams: predicted ({lv['landing'][0]:.3f},{lv['landing'][1]:.3f})"
                         + (f", height {lv['height']:.2f}" if lv.get("height") is not None else "") + "]")
            print(line, flush=True)

    if srv:
        summ = measure_summary(results)
        from correction import calib_id, refit_and_save
        calib = load_calib()
        log = save_log("test", {"settings": vars(a), "calib_id": calib_id(calib),
                                "correction": t.correction.model, "summary": summ, "results": results})
        print("\n== Accuracy ==")
        if "mean_error_px" in summ:
            print(f"mean error {summ['mean_error_px']:.0f}px, worst {summ['max_error_px']:.0f}px, "
                  f"within 25px: {summ['within_25px']}/{summ['taps']}, within 50px: {summ['within_50px']}/{summ['taps']}")
            ox, oy = summ["mean_offset_px"]
            print(f"average miss: {abs(ox):.0f}px {'right' if ox >= 0 else 'left'}, "
                  f"{abs(oy):.0f}px {'down' if oy >= 0 else 'up'}  "
                  "(if this is most of the error, the misses are systematic: recalibrate)")
        print(f"taps with no touch: {summ['no_touch']}")
        print(f"Log saved: {log}")
        if not a.no_learn:
            t.exp.save()
            if t.vision is not None and t.vision.enabled:
                t.vision.save()
                print(t.vision.describe())
            print(f"\nSaved {len(t.exp)} touches of experience for the learned model.")
            if not t.learned:
                print("== Updating aim correction from all tests with this calibration ==")
                refit_and_save(calib)
        srv.set(step=0, target=None, msg=(f"Done: mean error {summ['mean_error_px']:.0f}px"
                                          if "mean_error_px" in summ else "Done"))
        time.sleep(1.5)
        srv.stop()
