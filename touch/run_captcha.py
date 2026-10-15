"""The whole loop on the real site:   ./lr captcha [--count N]

  look (straightened photo, arm at home)
    -> what page is it?  choice / checkbox: tap the taught spot,  grid: ask the solver
    -> tap the tiles the solver picked, one by one, then go home
    -> look again: exactly those tiles selected?  if not, tap ONLY the tiles that differ (max --max-fix rounds)
    -> tap Verify (once)
    -> look until the site answers: "Access granted" = pass, red "Incorrect selection" = fail

Safety rules (a second tap on a tile toggles it back, so nothing is ever tapped "just to be sure"):
  * the solver's tile positions must agree with the dialog this program finds itself, or nothing is tapped;
  * Verify is never pressed unless the camera confirms the selection (two steady looks);
  * Verify is pressed once (--verify-retries 0), a failed or unclear result stops the run;
  * a camera that cannot see the whole screen or the dialog means "look again", never "tap anyway".

Setup: `./lr touch-calib` (arm), `./lr captcha --learn-layout` (once: teaches the 3 pages around the grid),
the site open full-screen on the touch screen with "Camera markers" ticked, CLOUDFLARE_* set in lr.conf.
Everything is saved in outputs/run/<time>/ : every photo, the solver's answer, the plan, the result.

  ./lr captcha --dry-run          camera + solver only: shows what it WOULD tap, no arm
  ./lr captcha --learn-layout     teach the choice / checkbox / verified pages
  ./lr captcha-train [--count N]  training: starts sessions on the Training page by itself, retries a wrong
                                  answer on the same challenge until solved (N = challenges to solve), and (with the robot link, TOUCH_WEB) learns from every tap
"""

import argparse
import json
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pagestate import (LAYOUT_PATH, PAGES, GridNotFound, PageState, SiteLayout, check_presses_against_grid,  # noqa: E402
                       classify, find_grid, has_error_text, read_selection, selected_set)
from solve import SolverError, draw  # noqa: E402
from verify_selection import check_selection, observe_stable  # noqa: E402


class RunError(RuntimeError):
    """Stop this challenge (and the run) without tapping anything further."""


@dataclass
class Config:
    count: int = 1
    tries: int = 3               # new photos when the solver says "try again"
    max_fix: int = 2             # rounds of "tap only the differing tiles"
    verify_retries: int = 0      # extra Verify taps if the page did not react at all
    verify_wait: float = 8.0     # seconds with no reaction before a Verify retry is considered
    settle: float = 1.0          # seconds after the arm is home before looking
    page_timeout: float = 90.0   # reaching the grid (choice -> checkbox -> grid)
    outcome_timeout: float = 30.0
    retap_after: float = 6.0     # same page still showing this long after a tap: tap its button again
    max_taps_per_page: int = 3
    target: str = None
    model: str = None
    dry_run: bool = False
    stop_on_fail: bool = True
    size: tuple = (1280, 800)
    training: bool = False       # start sessions itself, keep going after fails, never stop on a stale error


@dataclass
class Result:
    n: int
    challenge: str = ""          # the site's name for it (robot link), e.g. "3/100 · Train · dogs-5"
    attempt: int = 1             # which try on this same challenge
    outcome: str = "error"       # pass | fail | pass_unconfirmed | unknown | dry_run | error
    seconds: float = 0.0
    target: str = ""
    tiles: list = field(default_factory=list)
    fixes: int = 0
    detail: str = ""
    folder: str = ""


class Run:
    """One run of the loop. `look`, `solve` and `arm` are injected, so the logic is testable without
    hardware (see tests/test_run_captcha.py)."""

    def __init__(self, look, solve, arm, layout, cfg, out_dir, log=None, sleep=time.sleep, clock=time.time, web=None):
        self.look, self.solve, self.arm, self.layout, self.cfg = look, solve, arm, layout, cfg
        self.web = web             # WebTouches: the site reports pass/fail (and touches) itself
        self.failed_answers = set()  # tile sets already marked wrong on the current challenge
        self.attempt_of = ("", 0)    # (challenge, how many tries on it so far)
        self.out = Path(out_dir)
        self.say = log or (lambda m: print(m, flush=True))
        self.sleep, self.clock = sleep, clock
        self.folder = None
        self.shots = 0
        self._last = None          # the most recent photo, for saving

    # ------------------------------------------------------------------ small helpers
    def save(self, name, img):
        if self.folder is not None and img is not None:
            self.shots += 1
            cv2.imwrite(str(self.folder / f"{self.shots:02d}_{name}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 92])

    def snap(self, wait=0.0):
        if wait:
            self.sleep(wait)
        self._last = self.look()
        return self._last

    def tap(self, u, v, why):
        self.say(f"   tap {why}: u={u:.4f} v={v:.4f}")
        self.arm.tap(float(u), float(v))

    def home(self):
        self.arm.go_home()

    # ------------------------------------------------------------------ 1. get to the grid
    def reach_grid(self):
        """Look, and tap whatever page we are on until the image grid is up and steady. Returns (img, state).
        With the robot link the site itself says which screen it shows and where its buttons are, so nothing
        has to be taught; the camera is still used for the grid (the solver needs the photo). Each page seen
        that way is also learned from the camera, for runs without the link."""
        cfg, t0 = self.cfg, self.clock()
        taps, last = {}, (None, 0.0)          # taps per page kind; (kind, time) of the last page tap
        blind = 0
        while self.clock() - t0 < cfg.page_timeout:
            img = self.snap()
            if img is None:
                blind += 1
                if blind >= 5:
                    raise RunError("I can't see the whole screen (all 4 markers). Is 'Camera markers' ticked on "
                                   "the site, the camera in place, and the arm out of the way?")
                self.sleep(0.7)
                continue
            blind = 0
            kind, spot, why = self._site_page(img)
            if kind is None:                         # no robot link (or it is quiet): the camera decides
                st = classify(img, self.layout)
                self.say(f"   page: {st}")
                kind = st.kind
                if kind == "grid":
                    if st.error and cfg.training:
                        self.say("   wrong answer on screen: solving the same challenge again")
                    elif st.error:
                        self.save("page_grid_error", img)
                        raise RunError("The grid still shows 'Incorrect selection' from before. Reload the page "
                                       "(or tap its reset button) and start again.")
                    img2 = self.snap(0.7)               # steady? (images loaded, animations done)
                    st2 = classify(img2, self.layout) if img2 is not None else None
                    if (st2 is not None and st2.kind == "grid" and (cfg.training or not st2.error)
                            and _steady(img, img2)):
                        self.save("page_grid", img2)
                        return img2, st2
                    continue
                if kind in PAGES:
                    spot = self.layout.tap(kind) if self.layout else None
                    if cfg.training and kind == "choice" and self.layout and self.layout.start_tap():
                        spot = self.layout.start_tap()      # Training page: start a whole session
                    why = f"on the {kind} page"
            elif kind == "grid":
                try:                                 # the site says grid: the camera finds its geometry
                    st = _grid_state(img)
                except GridNotFound as e:
                    self.say(f"   site: grid, camera: not yet ({e})")
                    self.sleep(0.7)
                    continue
                if st.error and cfg.training:
                    self.say("   wrong answer on screen: solving the same challenge again")
                elif st.error:
                    self.save("page_grid_error", img)
                    raise RunError("The grid still shows 'Incorrect selection' from before. Reload the page "
                                   "(or tap its reset button) and start again.")
                img2 = self.snap(0.7)
                if img2 is not None and _steady(img, img2):
                    try:
                        st2 = _grid_state(img2)
                    except GridNotFound:
                        continue
                    self.save("page_grid", img2)
                    return img2, st2
                continue
            elif kind in ("loading", "wait"):
                self.sleep(0.7)
                continue
            if kind not in PAGES:
                self.sleep(0.8)                      # unknown: loading, or the page is changing
                continue
            if cfg.dry_run:
                raise RunError(f"Page is '{kind}', not the grid; --dry-run never taps. Open the grid first.")
            k0, when = last
            if k0 == kind and self.clock() - when < cfg.retap_after:
                self.sleep(0.7)                      # tapped a moment ago: give the page time
                continue
            if taps.get(kind, 0) >= cfg.max_taps_per_page:
                raise RunError(f"Tapped the '{kind}' page {taps[kind]} times and it is still showing. "
                               + ("The tap is probably missing the button (check the aim: ./lr practice --web)."
                                  if self.web is not None else "Check the taught position (./lr captcha --learn-layout)."))
            if spot is None:
                raise RunError(f"I don't know where to tap on the '{kind}' page. "
                               "Run ./lr captcha --learn-layout first (or use the robot link: --web).")
            self.save(f"page_{kind}", img)
            self.tap(*spot, why)
            self.home()
            taps[kind] = taps.get(kind, 0) + 1
            last = (kind, self.clock())
            self.sleep(cfg.settle)
        raise RunError(f"The grid did not appear within {cfg.page_timeout:.0f} s.")

    def _site_page(self, img):
        """(kind, (u, v) to tap, description) from the site itself, or (None, None, None) without the link."""
        if self.web is None:
            return None, None, None
        site = self.web.page()
        if site is None:
            return None, None, None
        kind = site.get("kind")
        tg = lambda n: self.web.target(site, n)  # noqa: E731
        if not site.get("fullscreen") and not getattr(self, "_warned_fs", False):
            self._warned_fs = True
            self.say("   note: the site is not full-screen; taps may land off (calibration assumes full-screen)")
        if kind == "choice":
            start, card = tg("start"), tg("card")
            spot, why = ((start, "on 'Start session'") if self.cfg.training and start else (card, "on the reCAPTCHA card"))
            self._learn_page("choice", img, card, start)
        elif kind == "checkbox":
            spot, why = tg("checkbox"), "on 'I'm not a robot'"
            self._learn_page("checkbox", img, spot)
        elif kind == "verified":
            spot, why = tg("restart"), "on 'Start again'"
            self._learn_page("verified", img, spot)
        elif kind in ("grid", "loading"):
            return kind, None, None
        else:
            return None, None, None              # other / slider: let the camera decide
        self.say(f"   page (site): {kind}")
        return kind, spot, why

    def _learn_page(self, kind, img, tap, start=None):
        """Remember what this page looks like on camera (and where to tap), once per run, for runs without the link."""
        if self.layout is None or tap is None or kind in getattr(self, "_learned_pages", set()):
            return
        self._learned_pages = getattr(self, "_learned_pages", set()) | {kind}
        try:
            self.layout.teach(kind, img, tap)
            if start is not None:
                self.layout.pages[kind]["start"] = [round(start[0], 5), round(start[1], 5)]
            self.layout.save()
            self.say(f"   learned the '{kind}' page from the site (camera signature + tap spot saved)")
        except Exception as e:  # never stop a run over this
            self.say(f"   (could not save the '{kind}' page: {e})")

    # ------------------------------------------------------------------ 2. ask the solver
    def ask_solver(self, img, state):
        cfg, last = self.cfg, None
        for attempt in range(1, cfg.tries + 1):
            if attempt > 1:
                self.home()
                img = self.snap(cfg.settle)
                if img is None:
                    continue
                try:
                    state = _grid_state(img)
                except GridNotFound as e:
                    last = SolverError("no_dialog", str(e), retry=True)
                    continue
            self.say(f"   asking the solver (try {attempt}) ...")
            t = self.clock()
            try:
                res = self.solve(img, cfg.target, cfg.model)
                worst, ok = check_presses_against_grid(res["presses"], state.grid)
                if not ok:
                    raise SolverError("layout_mismatch", f"the solver's tiles are up to {worst:.2f} tiles away from "
                                      "the dialog found in this photo", retry=True)
            except SolverError as e:
                self.say(f"   solver: {e.error}: {e.detail}")
                last = e
                if not e.retry:
                    raise RunError(f"Solver refused for good ({e.error}): {e.detail}")
                continue
            self.say(f"   solver: target '{res['target']}', tiles {res['tiles']}  ({self.clock() - t:.1f} s)")
            self.save("solver_sent", img)
            self.save("solver_plan", draw(img, res["presses"]))
            if self.folder is not None:
                (self.folder / "solver.json").write_text(json.dumps(res, indent=1))
            return res, state, img
        raise RunError(f"No usable answer from the solver after {cfg.tries} tries ({last}).")

    # ------------------------------------------------------------------ 3-6. tap, check, verify
    def run_selection(self, res, state, img):
        """Make the selection equal to the solver's tiles. Returns (fixes, grid used)."""
        want = {p["tile"] for p in res["presses"] if p["role"] == "tile"}
        press = {p["tile"]: (p["x_norm"], p["y_norm"]) for p in res["presses"] if p["role"] == "tile"}
        have = selected_set(read_selection(img, state.grid))
        if have:
            self.say(f"   note: tiles {sorted(have)} were already selected; only the difference is tapped")
        order = [p["tile"] for p in res["presses"] if p["role"] == "tile" and p["tile"] not in have]
        order += sorted(have - want)                  # ticked but not wanted: tap to untick
        for tile in order:
            u, v = press.get(tile) or state.grid.tile_center_uv(tile)
            self.tap(u, v, f"tile {tile}")
        self.home()
        fixes, grid = 0, state.grid
        for rnd in range(self.cfg.max_fix + 1):
            self.sleep(self.cfg.settle)
            chk = observe_stable(lambda: self.snap(), want, sleep=self.sleep)
            if chk is None:
                raise RunError("I can't read which tiles are selected (dialog not steady or not visible). "
                               "Not pressing Verify.")
            self.save(f"selection_{rnd}", self._last)
            self.say(f"   {chk}")
            grid = chk.grid
            if chk.ok:
                return fixes, grid
            if rnd == self.cfg.max_fix:
                raise RunError(f"{chk}  Still wrong after {fixes} fix round(s): not pressing Verify.")
            fixes += 1
            for tile in chk.to_tap:
                u, v = grid.tile_center_uv(tile)
                self.tap(u, v, f"fix tile {tile} ({'untick' if tile in chk.extra else 'tick'})")
            self.home()
        return fixes, grid

    def press_verify(self, res, grid):
        vp = res["presses"][-1]
        W, H = grid.size
        vx, vy = vp["x_norm"] * W, vp["y_norm"] * H
        off = float(np.hypot(vx - grid.verify[0], vy - grid.verify[1])) / grid.box[2]
        if off > 0.15:
            raise RunError(f"The solver's Verify position is {off:.2f} grid-widths from the Verify button seen "
                           "in the photo. Not pressing it.")
        self.tap(vp["x_norm"], vp["y_norm"], "VERIFY")
        self.home()

    # ------------------------------------------------------------------ 7. the site's answer
    def wait_outcome(self, res, grid):
        cfg, t0 = self.cfg, self.clock()
        retries, gone, taught = cfg.verify_retries, 0, "verified" in (self.layout.pages if self.layout else {})
        last_img = None
        self.sleep(cfg.settle)
        while self.clock() - t0 < cfg.outcome_timeout:
            got = self.web.result() if self.web is not None else None
            if got is not None and got.get("state") in ("pass", "fail"):
                self.save(f"result_{got['state']}_site", self.snap())
                return got["state"], f"the site reported {got['state'].upper()} {got.get('challenge') or ''}".strip()
            img = self.snap()
            if img is None:
                self.sleep(0.7)
                continue
            last_img = img
            st = classify(img, self.layout)
            if st.kind == "grid":
                gone = 0
                if st.error and self.web is None:
                    self.save("result_fail", img)
                    return "fail", "the site says: Incorrect selection"
                if retries and self.clock() - t0 >= cfg.verify_wait:
                    chk = check_selection(img, {p["tile"] for p in res["presses"] if p["role"] == "tile"})
                    if chk.ok:
                        retries -= 1
                        self.say("   the page did not react to Verify: tapping it once more")
                        self.press_verify(res, chk.grid)
                        t0 = self.clock()
            elif st.kind == "verified":
                self.save("result_pass", img)
                return "pass", "Access granted"
            else:
                gone += 1
                if gone >= 3 and not taught:         # no way to recognise the success page: say so honestly
                    self.save("result_unconfirmed", img)
                    return "pass_unconfirmed", "the dialog is gone and no error shown (teach the 'verified' page to confirm)"
            self.sleep(0.8)
        self.save("result_timeout", last_img)
        return "unknown", f"the site did not answer within {cfg.outcome_timeout:.0f} s (this is NOT a fail)"

    def _reuse_answer(self, state):
        """Retry after a wrong answer: the solver was right, so the taps missed. Keep its answer (no new solver
        call) as long as the grid is where it was; run_selection then taps only the tiles that differ."""
        res = getattr(self, "last_answer", None)
        if not res:
            return None
        worst, ok = check_presses_against_grid(res["presses"], state.grid)
        if not ok:
            self.say("   the grid moved since the last try: asking the solver again")
            return None
        self.say(f"   same challenge: keeping the solver's answer {res['tiles']}, fixing the taps")
        return res

    def _which_challenge(self, r):
        """Same challenge as the last try (after a wrong answer) or a new one? Wrong answers are only remembered
        for the challenge they were given on."""
        name = _challenge_name(self.web)
        prev, n = self.attempt_of
        # with the robot link the site names the challenge; without it, a wrong answer leaves the same grid up
        same = (name == prev) if name else getattr(self, "_last_outcome", None) == "fail"
        if same:
            n += 1
        else:
            n, self.failed_answers, self.last_answer = 1, set(), None
        self.attempt_of = (name, n)
        r.challenge, r.attempt = name, n
        if n > 1:
            self.say(f"   try {n} on this challenge" + (f" ({name})" if name else "")
                     + (f"; wrong before: {[sorted(x) for x in self.failed_answers]}" if self.failed_answers else ""))

    # ------------------------------------------------------------------ one challenge, and the loop
    def solve_one(self, n):
        t0 = self.clock()
        stamp = time.strftime("%Y%m%d-%H%M%S")
        self.folder = self.out / f"{stamp}-{n:02d}"
        self.folder.mkdir(parents=True, exist_ok=True)
        self.shots = 0
        r = Result(n, folder=str(self.folder))
        self.say(f"\n=== challenge {n} ===")
        try:
            img, st = self.reach_grid()
            self._which_challenge(r)
            res = self._reuse_answer(st) if r.attempt > 1 else None
            if res is None:
                res, st, img = self.ask_solver(img, st)
            self.last_answer = res
            r.target, r.tiles = res["target"], res["tiles"]
            if self.web is not None:
                self.web.drain_results()             # only this challenge's result counts
            if self.cfg.dry_run:
                r.outcome, r.detail = "dry_run", f"would tap tiles {res['tiles']} then Verify; plan saved in {self.folder}"
            else:
                r.fixes, grid = self.run_selection(res, st, img)
                self.press_verify(res, grid)
                r.outcome, r.detail = self.wait_outcome(res, grid)
        except RunError as e:
            r.outcome, r.detail = "error", str(e)
        except Exception as e:  # arm, camera, anything unexpected: stop, and the arm goes home on exit
            r.outcome, r.detail = "error", f"{type(e).__name__}: {e}"
        r.seconds = round(self.clock() - t0, 1)
        self._last_outcome = r.outcome
        if r.outcome == "fail" and r.tiles:
            self.failed_answers.add(frozenset(r.tiles))
        elif r.outcome in ("pass", "pass_unconfirmed"):
            self.failed_answers = set()
        if hasattr(self.arm, "end_challenge"):      # learning arm: save the taps, refit the model
            taps = self.arm.end_challenge(r.outcome)
            if taps:
                (self.folder / "taps.json").write_text(json.dumps(taps, indent=1))
                errs = [t["err_px"] for t in taps if t["err_px"] is not None]
                self.say(f"   taps: {len(taps)}, landed {len(errs)}" + (f", miss median {np.median(errs):.0f}px"
                                                                         f", max {max(errs):.0f}px" if errs else ""))
        (self.folder / "result.json").write_text(json.dumps(asdict(r), indent=1))
        with open(self.out / "results.jsonl", "a") as f:
            f.write(json.dumps(asdict(r)) + "\n")
        icon = {"pass": "PASS", "fail": "FAIL"}.get(r.outcome, r.outcome.upper())
        self.say(f"=== {icon} in {r.seconds} s: {r.detail}")
        return r

    def run(self):
        self.out.mkdir(parents=True, exist_ok=True)
        results, errors_in_row = [], 0
        # training: --count = challenges SOLVED (a wrong answer is retried on the same challenge, so it can take
        # several tries each); capped at 6 tries per wanted challenge so it always ends
        limit = self.cfg.count * 6 if self.cfg.training else self.cfg.count
        for n in range(1, limit + 1):
            r = self.solve_one(n)
            results.append(r)
            errors_in_row = errors_in_row + 1 if r.outcome in ("error", "unknown") else 0
            if self.cfg.training:                     # training: fails are data; stop only when stuck
                if sum(x.outcome in ("pass", "pass_unconfirmed") for x in results) >= self.cfg.count:
                    break
                if errors_in_row >= 3:
                    self.say("3 challenges in a row went wrong (not just a wrong answer): stopping.")
                    break
                self.home()
                continue
            if r.outcome not in ("pass", "pass_unconfirmed", "dry_run"):
                if r.outcome != "fail" or self.cfg.stop_on_fail:
                    break
        passed = sum(r.outcome in ("pass", "pass_unconfirmed") for r in results)
        if self.cfg.training and results:
            fails = sum(r.outcome == "fail" for r in results)
            self.say(f"\nTraining: {passed} solved, {fails} wrong answers (retried), {len(results) - passed - fails} errors.")
            tries = [r.attempt for r in results if r.outcome in ("pass", "pass_unconfirmed")]
            if tries:
                self.say(f"Tries needed per solved challenge: " + ", ".join(map(str, tries))
                         + f"  ({sum(t == 1 for t in tries)}/{len(tries)} first try)")
        self.say(f"\nDone: {passed}/{len(results)} passed" + (f" (of {self.cfg.count} planned)"
                                                          if len(results) < self.cfg.count else "")
                 + f".  Details: {self.out}")
        return results


class LearningArm:
    """Wraps the Tapper: after every tap, the site says where the stylus really landed. Each tap becomes
    experience for the learned model (the same data practice collects), so solving CAPTCHAs is practice too.
    The model is refit after every challenge, so the next challenge already aims better."""

    def __init__(self, tapper, web, tol_px=15.0, log=print):
        from learner import LearnedModel, reward, zone_of
        from common import load_calib
        self.t, self.web, self.tol, self.say = tapper, web, tol_px, log
        self._reward, self._zone_of, self._LM = reward, zone_of, LearnedModel
        self.calib = load_calib()
        self.taps = []

    def tap(self, u, v):
        t = self.t
        self.web.drain()
        zone = self._zone_of(u, v, t.zone_centers)
        arm = t.zones.best(zone) if t.zones is not None else None
        t.last_touch_q = None
        t.tap(u, v)
        got = self.web.all_touches()
        rec = {"target": [float(u), float(v)], "zone": zone, "press": t.press, "touches": len(got), "err_px": None,
               "landed": None, "el": None, "t": time.time()}
        if got:
            g = got[0]
            w, h = g.get("w") or 1000, g.get("h") or 1000
            err = float(np.hypot((g["u"] - u) * w, (g["v"] - v) * h))
            r = self._reward(err, True, len(got), t.last_force_stop, self.tol)
            rec.update(err_px=round(err, 1), landed=[g["u"], g["v"]], el=g.get("el"), reward=r)
            if getattr(t, "vision", None) is not None and t.last_hover_obs:
                t.vision.learn(t.last_hover_obs, (g["u"], g["v"]))   # the cameras learn where the tip really was
            if t.last_touch_q is not None:
                t.exp.add(t.last_touch_q, (g["u"], g["v"]), (u, v), r, t.press,
                          {"src": "captcha", "err_px": err, "el": g.get("el")})
            self.say(f"      landed {err:4.0f}px from the aim" + (f" on {g['el']}" if g.get("el") else ""))
        else:
            r = self._reward(0, False, 0, t.last_force_stop, self.tol)
            rec["reward"] = r
            self.say("      the site saw NO touch")
        if t.zones is not None and arm is not None:
            t.zones.update(zone, arm, r)
        t.exp.data.setdefault("attempts", []).append(
            {"zone": zone, "try": 1, "err_px": rec["err_px"], "touched": bool(got), "press": t.press, "slow": 1.0,
             "reward": r, "target": [float(u), float(v)], "t": time.time(), "src": "captcha"})
        self.taps.append(rec)

    def go_home(self):
        self.t.go_home()

    def end_challenge(self, outcome):
        taps, self.taps = self.taps, []
        for x in taps:
            x["outcome"] = outcome
        self.t.exp.save()
        if getattr(self.t, "vision", None) is not None and self.t.vision.enabled:
            self.t.vision.save()
        if self.t.learned is not None:          # refit: the next challenge aims with what was just learned
            self.t.learned = self._LM(self.calib, self.t.exp)
            self.t.model.touch_pose = self.t.learned.touch_pose
        return taps


def _challenge_name(web):
    pg = web.page() if web is not None else None
    return (pg or {}).get("challenge") or ""


def _grid_state(img):
    g = find_grid(img)
    return PageState("grid", g, has_error_text(img, g))


def _steady(a, b, tol=6.0):
    """Two looks a moment apart show the same thing (mean absolute difference, on a small copy)."""
    sa, sb = (cv2.resize(x, (160, 100), interpolation=cv2.INTER_AREA).astype(np.float32) for x in (a, b))
    return float(np.abs(sa - sb).mean()) < tol


# ---------------------------------------------------------------------------------- command line

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--count", type=int, default=1, help="challenges to solve in a row (continues only after a pass)")
    p.add_argument("--camera", default=os.environ.get("CAMERA", "/dev/video4"))
    p.add_argument("--cam-mode", default="auto")
    p.add_argument("--size", default="1280x800")
    p.add_argument("--out", default=str(Path(__file__).resolve().parent.parent / "outputs" / "run"))
    p.add_argument("--delay", type=float, default=5, help="seconds to switch to the page before starting")
    p.add_argument("--layout", default=str(LAYOUT_PATH))
    p.add_argument("--learn-layout", action="store_true", help="teach the choice / checkbox / verified pages, then exit")
    p.add_argument("--only", choices=PAGES + ("start",), nargs="+",
                   help="with --learn-layout: teach just these pages ('start' = the Start session button)")
    p.add_argument("--training", action="store_true",
                   help="training: start sessions on the Training page, keep going after wrong answers")
    p.add_argument("--web", action="store_true",
                   help="robot link (TOUCH_WEB, ROBOT_KEY): the site reports results and where every tap landed")
    p.add_argument("--http-port", type=int, default=8765)
    p.add_argument("--dry-run", action="store_true", help="camera + solver only: show the plan, never move the arm")
    p.add_argument("--tries", type=int, default=3)
    p.add_argument("--max-fix", type=int, default=2)
    p.add_argument("--verify-retries", type=int, default=0)
    p.add_argument("--continue-on-fail", action="store_true")
    p.add_argument("--target", help="what to select, if known (skips reading the banner)")
    p.add_argument("--model")
    # arm
    p.add_argument("--port")
    p.add_argument("--id", default="my_follower")
    p.add_argument("--robot", default="so101")
    p.add_argument("--press", type=float, default=0.04)
    p.add_argument("--lift", type=float, default=0.3)
    p.add_argument("--speed", type=float, default=1.0)
    p.add_argument("--smooth", type=float, default=0.08)
    p.add_argument("--dwell", type=float, default=0.25)
    p.add_argument("--max-load", type=int, default=0)
    p.add_argument("--torque", type=int, default=100)
    p.add_argument("--simple-path", action="store_true")
    p.add_argument("--no-learned", action="store_true")
    p.add_argument("--no-correct", action="store_true")
    p.add_argument("--cameras", default="", help="tip vision while tapping: the phone cameras, comma-separated ('none' = off)")
    p.add_argument("--tip-color", default="pink")
    p.add_argument("--screen-mm", type=float, default=300.0)
    a = p.parse_args()

    from see import open_screen_camera
    w, h = map(int, a.size.lower().split("x"))
    cam = open_screen_camera(a.camera, w, h, a.cam_mode)
    if hasattr(cam, "describe"):
        print("Screen " + cam.describe())

    if a.learn_layout:
        from learn_layout import learn
        learn(cam, a.layout, a.out, a.delay, a.http_port, a.only)
        return

    layout = SiteLayout.load(a.layout)
    if not layout.pages and not a.web:
        print("note: no taught pages yet (./lr captcha --learn-layout). Fine if the grid is already open; "
              "otherwise the robot can't tap the choice page / checkbox.")
    elif layout.missing() and not a.web:
        print(f"note: pages not taught yet: {', '.join(layout.missing())}")

    from solve import solve_screen, solver_config
    try:
        cfg_s = solver_config()      # --dry-run asks the solver too: it only skips the arm
    except SolverError as e:
        raise SystemExit(e.detail)

    def look():
        return cam.screen_image(tries=6, allow_remembered=False)

    def solve(img, target, model):
        return solve_screen(img, cfg_s, target, model, (w, h))

    cfg = Config(count=a.count, tries=a.tries, max_fix=a.max_fix, verify_retries=a.verify_retries,
                 target=a.target, model=a.model, dry_run=a.dry_run,
                 stop_on_fail=not (a.continue_on_fail or a.training), size=(w, h), training=a.training)
    if a.training and not a.web and layout.start_tap() is None:
        print("note: 'Start session' not taught (./lr captcha --learn-layout --only start): on the Training page "
              "the robot taps the single-challenge card instead.")
    web = None
    if a.web and not a.dry_run:
        from webtouch import WebTouches
        web = WebTouches(msg="CAPTCHA " + ("training" if a.training else "run"))

    if a.dry_run:
        class NoArm:
            def tap(self, u, v):
                raise RuntimeError("dry run: the arm must not move")

            def go_home(self):
                pass
        arm_ctx = _Null(NoArm())
    else:
        if not a.port:
            raise SystemExit("--port is required (./lr passes it).")
        from tap import Tapper
        vision = None
        if a.cameras and a.cameras != "none":
            from common import load_calib
            from correction import calib_id
            from tipvision import TipVision
            vision = TipVision(a.cameras, a.tip_color, width_mm=a.screen_mm, cid=calib_id(load_calib()))
            print(vision.describe())
        arm_ctx = Tapper(a.port, a.id, a.robot, a.lift, a.press, a.speed, smooth=a.smooth,
                         own_path=not a.simple_path, correct=not a.no_correct, dwell=a.dwell,
                         max_load=a.max_load, torque=a.torque, learned=not a.no_learned, vision=vision)

    if a.delay > 0:
        print(f"Starting in {a.delay:.0f} s: switch to the page on the touch screen (full-screen, F11)", flush=True)
        time.sleep(a.delay)
    with arm_ctx as arm:
        if web is not None:
            print(f"Learning from every tap: {len(arm.exp)} touches of experience so far.")
            arm = LearningArm(arm, web)
        try:
            results = Run(look, solve, arm, layout, cfg, a.out, web=web).run()
        finally:
            if web is not None:
                web.stop()
                print(f"Experience: {len(arm.t.exp)} touches saved (./lr heatmap to see accuracy per zone).")
    raise SystemExit(0 if results and all(r.outcome in ("pass", "pass_unconfirmed", "dry_run") for r in results) else 1)


class _Null:
    def __init__(self, x):
        self.x = x

    def __enter__(self):
        return self.x

    def __exit__(self, *exc):
        return False


if __name__ == "__main__":
    main()
