"""The loop, end to end, against the scripted world (real screenshots, no robot, no network)."""

import json

import pytest

import pagestate as ps
from fakeworld import CORRECT, IMG, POS, Clock, World
from run_captcha import Config, Run
from solve import SolverError


def taught(pages=ps.PAGES):
    lay = ps.SiteLayout({}, "/nonexistent/site_layout.json")
    for n in pages:
        lay.teach(n, IMG[n], POS[n])
    return lay


def make(world, tmp_path, tiles=tuple(sorted(CORRECT)), layout=None, solver=None, **cfg):
    clock = Clock()
    calls = []

    def solve(img, target, model):
        calls.append(1)
        if solver:
            return solver(len(calls), world)
        return world.presses(tiles)

    cfg = Config(**cfg)
    run = Run(world.look, solve, world, layout if layout is not None else taught(), cfg, tmp_path,
              log=lambda m: None, sleep=clock.sleep, clock=clock.now)
    run.calls = calls
    return run


def whats(world):
    return [w for _, _, w in world.taps]


def test_full_flow_from_the_first_page_to_access_granted(tmp_path):
    w = World("choice")
    run = make(w, tmp_path)
    [r] = run.run()
    assert r.outcome == "pass" and r.fixes == 0
    assert whats(w) == ["choice card", "checkbox", "tile 0", "tile 4", "tile 8", "verify"]
    assert w.verify_pressed == 1 and r.tiles == [0, 4, 8] and r.target == "bicycles"
    saved = {p.name for p in (tmp_path / r.folder.split("/")[-1]).iterdir()}
    assert "solver.json" in saved and "result.json" in saved and any("solver_plan" in n for n in saved)
    assert json.loads((tmp_path / "results.jsonl").read_text().splitlines()[0])["outcome"] == "pass"


def test_starts_directly_on_the_grid(tmp_path):
    w = World("grid")
    [r] = make(w, tmp_path).run()
    assert r.outcome == "pass" and whats(w) == ["tile 0", "tile 4", "tile 8", "verify"]


def test_a_lost_tap_is_fixed_by_tapping_only_that_tile(tmp_path):
    w = World("grid", drop_taps=0, ignore_tiles=())
    # tile 8's first tap is lost: simulate with a world that drops the 3rd tile tap only
    orig = w.tap
    state = {"n": 0}

    def tap(u, v):
        if ps.np.hypot(u - POS["tiles"][8][0], v - POS["tiles"][8][1]) < 0.04 and state["n"] == 0:
            state["n"] += 1
            w.taps.append((u, v, "tile 8 (lost)"))
            return
        orig(u, v)
    w.tap = tap
    [r] = make(w, tmp_path).run()
    assert r.outcome == "pass" and r.fixes == 1
    assert whats(w) == ["tile 0", "tile 4", "tile 8 (lost)", "tile 8", "verify"]
    assert w.verify_pressed == 1


def test_an_extra_tile_is_unticked_and_nothing_else_is_touched(tmp_path):
    w = World("grid", double_touch={4: 8})     # touching tile 4 also toggles tile 8
    [r] = make(w, tmp_path, tiles=(0, 4)).run()
    assert w.sel == {0, 4} and r.fixes == 1
    assert whats(w) == ["tile 0", "tile 4", "tile 8", "verify"]      # the third tap is the untick
    assert r.outcome == "fail"                                         # the (fake) answer needs {0,4,8}: site says no


def test_never_presses_verify_when_the_selection_cannot_be_fixed(tmp_path):
    w = World("grid", ignore_tiles=(8,))
    [r] = make(w, tmp_path, max_fix=2).run()
    assert r.outcome == "error" and "not pressing Verify" in r.detail
    assert w.verify_pressed == 0
    assert whats(w).count("tile 8") == 3          # first attempt + 2 fix rounds, then it gives up


def test_a_wrong_answer_is_reported_as_fail_and_the_run_stops(tmp_path):
    w = World("grid")
    run = make(w, tmp_path, tiles=(0, 4), count=3)
    results = run.run()
    assert [r.outcome for r in results] == ["fail"] and w.verify_pressed == 1


def test_count_goes_on_after_a_pass_via_start_again(tmp_path):
    w = World("choice")
    results = make(w, tmp_path, count=2).run()
    assert [r.outcome for r in results] == ["pass", "pass"]
    assert whats(w).count("start again") == 1 and w.verify_pressed == 2


def test_no_taps_when_the_solvers_tiles_disagree_with_the_dialog(tmp_path):
    w = World("grid")
    run = make(w, tmp_path, solver=lambda n, world: world.presses((0, 4, 8), shift=0.06), tries=2)
    [r] = run.run()
    assert r.outcome == "error" and "No usable answer" in r.detail and not w.taps and len(run.calls) == 2


def test_solver_retry_takes_a_new_photo_then_works(tmp_path):
    def solver(n, world):
        if n == 1:
            raise SolverError("no_dialog", "arm in the way", retry=True)
        return world.presses((0, 4, 8))
    w = World("grid")
    run = make(w, tmp_path, solver=solver)
    [r] = run.run()
    assert r.outcome == "pass" and len(run.calls) == 2


def test_solver_refusal_for_good_stops_before_any_tap(tmp_path):
    def solver(n, world):
        raise SolverError("bad_credentials", "token rejected", retry=False)
    w = World("grid")
    [r] = make(w, tmp_path, solver=solver).run()
    assert r.outcome == "error" and "bad_credentials" in r.detail and not w.taps


def test_blind_camera_means_no_taps(tmp_path):
    w = World("grid", blind=50)
    [r] = make(w, tmp_path).run()
    assert r.outcome == "error" and "4 markers" in r.detail and not w.taps


def test_unknown_page_waits_then_gives_up_without_tapping(tmp_path):
    w = World("choice")
    w.look = lambda: IMG["grid_empty"][::-1].copy()      # nonsense: upside-down grid
    [r] = make(w, tmp_path, page_timeout=10).run()
    assert r.outcome == "error" and "did not appear" in r.detail and not w.taps


def test_does_not_know_where_to_tap_without_taught_pages(tmp_path):
    w = World("choice")
    lay = ps.SiteLayout({}, "/nonexistent")
    lay.pages = {"choice": {"sig": taught().pages["choice"]["sig"]}}   # recognised but no tap position
    [r] = make(w, tmp_path, layout=lay).run()
    assert r.outcome == "error" and "learn-layout" in r.detail and not w.taps


def test_dry_run_never_moves_the_arm(tmp_path):
    w = World("grid")
    [r] = make(w, tmp_path, dry_run=True).run()
    assert r.outcome == "dry_run" and not w.taps and r.tiles == [0, 4, 8]
    w2 = World("choice")
    [r2] = make(w2, tmp_path, dry_run=True).run()
    assert r2.outcome == "error" and not w2.taps


def test_old_error_state_is_not_built_upon(tmp_path):
    w = World("grid")
    w.error = True
    [r] = make(w, tmp_path).run()
    assert r.outcome == "error" and "Incorrect selection" in r.detail and not w.taps


def test_checkbox_is_not_tapped_twice_while_the_page_loads(tmp_path):
    """After tapping the checkbox the page may show it for a moment longer: wait, don't tap again."""
    w = World("checkbox")
    shown = {"n": 0}
    real_look = w.look

    def slow_look():
        if w.page == "grid" and shown["n"] < 2:        # the grid takes a couple of looks to appear
            shown["n"] += 1
            return IMG["checkbox"].copy()
        return real_look()
    w.look = slow_look
    [r] = make(w, tmp_path).run()
    assert r.outcome == "pass" and whats(w).count("checkbox") == 1


def test_unconfirmed_pass_is_labelled_honestly_without_the_verified_page_taught(tmp_path):
    w = World("grid")
    lay = taught(("choice", "checkbox"))
    [r] = make(w, tmp_path, layout=lay).run()
    assert r.outcome == "pass_unconfirmed"


def test_silence_after_verify_is_unknown_not_fail(tmp_path):
    w = World("grid")
    orig = w.tap

    def tap(u, v):
        orig(u, v)
        if w.verify_pressed:
            w.page, w.error = "grid", False            # page "does nothing": still the same grid
    w.tap = tap
    [r] = make(w, tmp_path, outcome_timeout=12).run()
    assert r.outcome == "unknown" and "NOT a fail" in r.detail
    assert w.verify_pressed == 1                       # and Verify was not pressed again


def test_optional_verify_retry_presses_once_more_only_if_nothing_happened(tmp_path):
    w = World("grid")
    orig = w.tap

    def tap(u, v):
        if ps.np.hypot(u - POS["verify"][0], v - POS["verify"][1]) < 0.04 and w.verify_pressed == 0:
            w.verify_pressed += 1
            w.taps.append((u, v, "verify (lost)"))
            return
        orig(u, v)
    w.tap = tap
    [r] = make(w, tmp_path, verify_retries=1, verify_wait=3).run()
    assert r.outcome == "pass" and whats(w)[-2:] == ["verify (lost)", "verify"]
