import json
import subprocess
import sys
from pathlib import Path

import cv2
import pytest

import pagestate as ps
from fakeworld import IMG, POS, World
from plan_clicks import demo_layout, plan_clicks
from solve import SolverError

TOUCH = Path(__file__).resolve().parent.parent
LAYOUT = demo_layout()


def uv(c):
    return c["u"], c["v"]


def close(a, b, tol=0.012):
    return abs(a[0] - b[0]) < tol and abs(a[1] - b[1]) < tol


def fake_solver(tiles):
    return lambda img, target, model: World().presses(tiles)


def test_choice_and_checkbox_pages_give_the_taught_click():
    for page in ("choice", "checkbox"):
        p = plan_clicks(IMG[page], LAYOUT)
        assert p["state"] == page and len(p["clicks"]) == 1 and p["clicks"][0]["role"] == page
        assert close(uv(p["clicks"][0]), POS[page], 1e-4)


def test_empty_grid_lists_every_wanted_tile_in_solver_order_with_pixels_and_fractions():
    p = plan_clicks(IMG["grid_empty"], LAYOUT, fake_solver((8, 0, 4)))
    assert [c["tile"] for c in p["clicks"]] == [8, 0, 4] and all(c["role"] == "tile" for c in p["clicks"])
    for c in p["clicks"]:
        assert close(uv(c), POS["tiles"][c["tile"]], 1e-4)
        assert c["x"] == round(c["u"] * 1280) and c["y"] == round(c["v"] * 800)
    assert p["tiles_wanted"] == [0, 4, 8] and p["tiles_selected"] == [] and p["size"] == [1280, 800]
    assert "Verify" in p["note"]


def test_full_appends_verify_like_playwright_would():
    p = plan_clicks(IMG["grid_empty"], LAYOUT, fake_solver((0, 4, 8)), full=True)
    assert [c["role"] for c in p["clicks"]] == ["tile", "tile", "tile", "verify"]
    assert close(uv(p["clicks"][-1]), POS["verify"], 1e-4)


def test_only_the_missing_tile_is_returned_when_most_are_ticked():
    p = plan_clicks(IMG["grid_tiles_0_4"], LAYOUT, fake_solver((0, 4, 8)))
    assert [c["tile"] for c in p["clicks"]] == [8] and p["tiles_selected"] == [0, 4]


def test_extra_tile_is_returned_as_an_untick():
    p = plan_clicks(IMG["grid_three_selected"], LAYOUT, fake_solver((0, 4)))
    assert [(c["role"], c["tile"]) for c in p["clicks"]] == [("untick", 8)]


def test_verify_only_once_the_selection_is_right():
    p = plan_clicks(IMG["grid_three_selected"], LAYOUT, fake_solver((0, 4, 8)))
    assert [c["role"] for c in p["clicks"]] == ["verify"] and close(uv(p["clicks"][0]), POS["verify"], 1e-4)


def test_tiles_option_needs_no_solver_and_uses_the_photos_own_geometry():
    p = plan_clicks(IMG["grid_empty"], LAYOUT, solver=None, tiles=[1, 3, 8], full=True)
    assert [c["tile"] for c in p["clicks"][:3]] == [1, 3, 8]
    for c in p["clicks"][:3]:
        assert close(uv(c), POS["tiles"][c["tile"]], 0.012)
    assert close(uv(p["clicks"][3]), POS["verify"], 0.012)


def test_error_state_verified_and_unknown_return_no_clicks():
    p = plan_clicks(IMG["grid_error"], LAYOUT, fake_solver((0,)))
    assert p["state"] == "grid_error" and p["clicks"] == [] and not p["done"]
    p = plan_clicks(IMG["verified"], LAYOUT)
    assert p["done"] and p["clicks"] == []
    p = plan_clicks(IMG["verified"], LAYOUT, next_challenge=True)
    assert p["done"] and p["clicks"][0]["role"] == "start_again" and close(uv(p["clicks"][0]), POS["verified"], 1e-4)
    p = plan_clicks(IMG["grid_empty"][::-1].copy(), LAYOUT)
    assert p["state"] == "unknown" and p["clicks"] == [] and "another" in p["note"]


def test_solver_problems_are_reported_not_raised_and_nothing_is_clicked():
    def refuse(img, target, model):
        raise SolverError("no_dialog", "arm in the way", retry=True)
    p = plan_clicks(IMG["grid_empty"], LAYOUT, refuse)
    assert p["clicks"] == [] and p["error"] == "no_dialog" and p["retry"] is True
    far = lambda img, t, m: World().presses((0, 4, 8), shift=0.06)
    p = plan_clicks(IMG["grid_empty"], LAYOUT, far)
    assert p["clicks"] == [] and p["error"] == "layout_mismatch"


def test_untaught_choice_page_says_so():
    p = plan_clicks(IMG["choice"], ps.SiteLayout({"choice": {"sig": LAYOUT.pages["choice"]["sig"]},
                                                  "checkbox": LAYOUT.pages["checkbox"],
                                                  "verified": LAYOUT.pages["verified"]}, "/x"))
    assert p["clicks"] == [] and "learn-layout" in p["note"]


def run_cli(tmp_path, image, *args):
    return subprocess.run([sys.executable, str(TOUCH / "plan_clicks.py"), "--image", str(image), *args],
                          capture_output=True, text=True, cwd=str(tmp_path), timeout=60)


def test_cli_from_a_raw_camera_photo_with_margins(tmp_path):
    photo = tmp_path / "photo.png"       # the screen with white all round, as a camera sees it (markers detectable)
    cv2.imwrite(str(photo), cv2.copyMakeBorder(IMG["grid_empty"], 90, 90, 120, 120, cv2.BORDER_CONSTANT, value=(255,) * 3))
    ann, out = tmp_path / "plan.png", tmp_path / "plan.json"
    p = run_cli(tmp_path, photo, "--tiles", "1,3,8", "--full", "--demo-layout", "--annotate", str(ann), "--out", str(out))
    assert p.returncode == 0, p.stderr
    plan = json.loads(p.stdout)
    assert plan == json.loads(out.read_text()) and ann.exists()
    assert [c["role"] for c in plan["clicks"]] == ["tile", "tile", "tile", "verify"]
    for c in plan["clicks"][:3]:
        assert close(uv(c), POS["tiles"][c["tile"]], 0.012)


def test_cli_with_a_straightened_image_and_no_clicks_exit_code(tmp_path):
    p = run_cli(tmp_path, TOUCH / "tests" / "shots" / "choice.png", "--straight", "--demo-layout")
    assert p.returncode == 0 and json.loads(p.stdout)["state"] == "choice"
    blank = tmp_path / "blank.png"
    cv2.imwrite(str(blank), IMG["grid_empty"][::-1].copy())
    p = run_cli(tmp_path, blank, "--straight", "--demo-layout")
    assert p.returncode == 2 and json.loads(p.stdout)["state"] == "unknown"
