import numpy as np
import pytest

import solve
from solve import SolverError, check_presses, solve_screen, solver_config

GOOD = {"ok": True, "target": "flowers", "tiles": [2, 4], "presses": [
    {"seq": 1, "role": "tile", "tile": 2, "x_norm": 0.58, "y_norm": 0.37},
    {"seq": 2, "role": "tile", "tile": 4, "x_norm": 0.49, "y_norm": 0.53},
    {"seq": 3, "role": "verify", "x_norm": 0.59, "y_norm": 0.82}]}
IMG = np.full((800, 1280, 3), 255, np.uint8)
CFG = solve.SolverConfig("https://x/api/robot/solve", {"X-CF-Account-Id": "a", "X-CF-Api-Token": "t"})


def test_config_reads_env_and_appends_the_endpoint_path():
    c = solver_config(env={"SOLVER_URL": "https://s.dev/", "CLOUDFLARE_ACCOUNT_ID": "acc", "CLOUDFLARE_API_TOKEN": "tok"})
    assert c.url == "https://s.dev/api/robot/solve"
    assert c.headers["X-CF-Account-Id"] == "acc" and c.headers["X-CF-Api-Token"] == "tok"


def test_config_names_what_is_missing_without_leaking_values():
    with pytest.raises(SolverError) as e:
        solver_config(env={"SOLVER_URL": "https://s.dev"})
    assert e.value.error == "bad_config" and "CLOUDFLARE_API_TOKEN" in e.value.detail and not e.value.retry


@pytest.mark.parametrize("mutate", [
    lambda r: r.update(presses=[]),
    lambda r: r.update(presses=r["presses"][:1]),                            # no verify
    lambda r: r["presses"].reverse(),                                        # verify first
    lambda r: r["presses"][0].update(x_norm=1.4),                            # outside the screen
    lambda r: r["presses"][1].update(tile=2),                                # repeated tile
    lambda r: r["presses"][0].pop("tile"),
    lambda r: r["presses"][2].update(role="tile", tile=7),                   # no verify at the end
])
def test_malformed_answers_are_refused(mutate):
    import copy
    r = copy.deepcopy(GOOD)
    mutate(r)
    with pytest.raises(SolverError) as e:
        check_presses(r)
    assert e.value.error == "bad_response"


def test_good_answer_passes_and_is_returned(monkeypatch):
    seen = {}

    def fake_post(url, jpg, fields, headers, timeout):
        seen.update(url=url, fields=fields, headers=headers, jpg=jpg)
        return 200, GOOD
    monkeypatch.setattr(solve, "post_image", fake_post)
    assert solve_screen(IMG, CFG, target="flowers") == GOOD
    assert seen["fields"] == {"screen_width": 1280, "screen_height": 800, "target": "flowers"}
    assert seen["jpg"][:2] == b"\xff\xd8"


@pytest.mark.parametrize("reply,error,retry", [
    ((200, {"ok": False, "error": "no_dialog", "detail": "x", "retry": True}), "no_dialog", True),
    ((401, {"ok": False, "error": "bad_credentials", "detail": "x", "retry": False}), "bad_credentials", False),
    ((502, "<html>bad gateway</html>"), "bad_response", True),
])
def test_server_refusals_keep_their_code_and_retry_flag(monkeypatch, reply, error, retry):
    monkeypatch.setattr(solve, "post_image", lambda *a, **k: reply)
    with pytest.raises(SolverError) as e:
        solve_screen(IMG, CFG)
    assert (e.value.error, e.value.retry) == (error, retry)


def test_network_failure_is_retryable(monkeypatch):
    def boom(*a, **k):
        raise OSError("no route")
    monkeypatch.setattr(solve, "post_image", boom)
    with pytest.raises(SolverError) as e:
        solve_screen(IMG, CFG)
    assert e.value.error == "network" and e.value.retry
