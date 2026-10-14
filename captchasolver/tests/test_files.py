import json

import pytest

from captchasolver import bench
from captchasolver.screen import build_actions


RESULT = {"tiles": [2, 6], "click": [[1116, 409], [772, 753]],
          "click_norm": [[0.58307, 0.37870], [0.40335, 0.69722]],
          "verify": [1134, 890], "verify_norm": [0.59248, 0.82407]}


def test_build_actions_tiles_then_verify_last():
    acts = build_actions(RESULT)
    assert [(a["role"], a.get("tile")) for a in acts] == [("tile", 2), ("tile", 6), ("verify", None)]
    assert all(a["action"] == "click" for a in acts)
    assert acts[0]["x_norm"] == 0.58307 and acts[0]["x"] == 1116   # both unit systems present


def test_build_actions_without_verify_button():
    acts = build_actions({**RESULT, "verify": None, "verify_norm": None})
    assert [a["role"] for a in acts] == ["tile", "tile"]


def test_default_examples_dir_is_the_repo_examples_folder():
    d = bench.default_examples_dir()
    assert d.name == "examples" and (d.parent / "pyproject.toml").exists()


async def test_replay_skips_json_without_actions_and_fails_loudly_when_empty(tmp_path):
    (tmp_path / "plain.json").write_text(json.dumps({"target": "cats", "tiles": [1]}))   # from `dir`, no actions
    (tmp_path / "broken.json").write_text("{nope")
    notes = []
    with pytest.raises(FileNotFoundError, match="no replayable"):
        await bench.run_replay(str(tmp_path), note=notes.append)
    assert any("plain.json" in n for n in notes)


class FakePage:
    def __init__(self, pressed_after):
        self.clicks, self.pressed_after = [], pressed_after

    async def evaluate(self, js):
        return [1000, 500]

    async def eval_on_selector_all(self, sel, js):
        return self.pressed_after

    class mouse:  # noqa: N801 - mimic page.mouse
        pass


async def test_execute_actions_uses_normalised_coords_of_current_viewport(monkeypatch):
    page = FakePage(pressed_after=[2, 6])
    clicks = []

    async def click(x, y):
        clicks.append((x, y))
    page.mouse = type("M", (), {"click": staticmethod(click)})()

    async def passed(_page, timeout_ms=5000):
        return True
    monkeypatch.setattr(bench, "_outcome", passed)
    ok, clicked_ok = await bench.execute_actions(page, {"actions": build_actions(RESULT)})
    assert ok and clicked_ok
    assert clicks[0] == pytest.approx((583.07, 189.35))      # 0.58307*1000, 0.37870*500
    assert clicks[-1] == pytest.approx((592.48, 412.035))    # verify is clicked last


async def test_execute_actions_flags_tiles_that_did_not_get_pressed(monkeypatch):
    page = FakePage(pressed_after=[2])                        # only one of the two tiles registered

    async def click(x, y):
        pass
    page.mouse = type("M", (), {"click": staticmethod(click)})()

    async def failed(_page, timeout_ms=5000):
        return False
    monkeypatch.setattr(bench, "_outcome", failed)
    ok, clicked_ok = await bench.execute_actions(page, {"actions": build_actions(RESULT)})
    assert not ok and not clicked_ok


async def test_execute_actions_rejects_unknown_actions_and_missing_verify():
    page = FakePage([])
    with pytest.raises(ValueError, match="unsupported"):
        await bench.execute_actions(page, {"actions": [{"action": "drag", "role": "tile"}]})
    page.mouse = type("M", (), {"click": staticmethod(lambda x, y: __import__("asyncio").sleep(0))})()
    with pytest.raises(RuntimeError, match="no verify"):
        await bench.execute_actions(page, {"actions": build_actions({**RESULT, "verify": None, "verify_norm": None})})


class ShotPage:
    async def screenshot(self, path=None):
        from pathlib import Path
        Path(path).write_bytes(b"png")


async def test_failed_runs_images_saved_on_fail_and_removed_when_it_later_passes(tmp_path):
    from captchasolver.bench import Record, _record_outcome, format_line
    fdir = tmp_path / "failed_runs"
    rec = Record(22, "cats-5", "train", subject="cats", target="cats", tiles=[1])
    await _record_outcome(ShotPage(), rec, fdir)                    # fail, no plan available
    assert (fdir / "022-cats-5.png").exists() and rec.fail_image.endswith("022-cats-5.png")
    assert "failed_runs/022-cats-5.png" in format_line(rec, 1, 1, color=False)
    (fdir / "022-cats-5-plan.png").write_bytes(b"x")                # leftovers from an older run
    (fdir / "022-cats-5-error.png").write_bytes(b"x")
    (fdir / "023-other.png").write_bytes(b"x")                      # a different challenge: untouched
    rec2 = Record(22, "cats-5", "train", passed=True)
    await _record_outcome(ShotPage(), rec2, fdir)                   # now it passes
    assert sorted(p.name for p in fdir.iterdir()) == ["023-other.png"]
    assert rec2.fail_image == ""


async def test_failure_images_can_be_switched_off(tmp_path):
    from captchasolver.bench import Record, _record_outcome
    rec = Record(1, "a", "train")
    await _record_outcome(ShotPage(), rec, None)
    assert rec.fail_image == "" and not list(tmp_path.iterdir())
