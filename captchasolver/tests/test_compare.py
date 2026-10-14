import pytest

from captchasolver import client
from captchasolver.bench import Record
from captchasolver.client import MODEL_CONFIG, WorkersAIError, chat_payload, find_answer
from captchasolver.compare import analyse, format_table, preflight_one, rank
from captchasolver.grid import parse_yes_no


@pytest.fixture(autouse=True)
def clean_config():
    MODEL_CONFIG.clear()
    yield
    MODEL_CONFIG.clear()


class FakeAI:
    """Programmable stand-in: `behave(model, payload)` returns a result dict or raises."""
    def __init__(self, behave):
        self.behave, self.calls = behave, []

    async def run(self, model, *, json=None, image=None):
        self.calls.append(json)
        return self.behave(model, json)


def chat(text):
    return {"choices": [{"message": {"content": text}}], "response": text}


def test_parse_yes_no():
    assert parse_yes_no("Yes") and parse_yes_no("yes, there is a cat.") and parse_yes_no("Yes.")
    assert not parse_yes_no("No") and not parse_yes_no("No, it is a dog.") and not parse_yes_no("")
    assert not parse_yes_no("unclear")  # neither word -> don't click
    assert not parse_yes_no("Nothing here")  # 'no' only counts as a whole word


def test_find_answer_handles_null_content_from_reasoning_models():
    with pytest.raises(ValueError):
        find_answer({"choices": [{"message": {"content": None, "reasoning": "hmm"}}]})
    assert find_answer(chat("HELLO 42")) == "HELLO 42"


def test_reasoning_models_get_token_headroom_and_config_applies():
    p = chat_payload("@cf/moonshotai/kimi-k2.5", b"x", "q", max_tokens=8, temperature=0)
    assert p["max_tokens"] >= 1024 and p["temperature"] == 0
    plain = chat_payload("@cf/meta/llama-4-scout-17b-16e-instruct", b"x", "q", max_tokens=8, temperature=0)
    assert plain["max_tokens"] == 8
    MODEL_CONFIG["@cf/moonshotai/kimi-k2.5"] = client.ModelConfig(
        {"chat_template_kwargs": {"enable_thinking": False}}, send_temperature=False)
    p = chat_payload("@cf/moonshotai/kimi-k2.5", b"x", "q", max_tokens=8, temperature=0)
    assert "temperature" not in p and p["chat_template_kwargs"] == {"enable_thinking": False}


async def test_preflight_picks_first_variant_that_works():
    ai = FakeAI(lambda m, j: chat("HELLO 42"))
    r = await preflight_one(ai, "@cf/meta/llama-4-scout-17b-16e-instruct")
    assert r.ok and r.variant == 0


async def test_preflight_falls_back_when_thinking_switch_is_rejected():
    def behave(model, j):
        if "chat_template_kwargs" in j:
            raise WorkersAIError("400 unknown field")
        return chat("hello 42.")
    ai = FakeAI(behave)
    r = await preflight_one(ai, "@cf/google/gemma-4-26b-a4b-it")
    assert r.ok and r.variant == 2          # plain request with temperature
    assert MODEL_CONFIG["@cf/google/gemma-4-26b-a4b-it"].extra == {}


async def test_preflight_drops_temperature_when_that_is_the_problem():
    def behave(model, j):
        if "temperature" in j:
            raise WorkersAIError("temperature not allowed")
        return chat("HELLO 42")
    r = await preflight_one(FakeAI(behave), "@cf/moonshotai/kimi-k2.5")
    assert r.ok and r.variant == 1 and MODEL_CONFIG[r.model].send_temperature is False


async def test_preflight_reports_failure_reason_and_cleans_up():
    def behave(model, j):
        raise WorkersAIError("403 license not accepted")
    r = await preflight_one(FakeAI(behave), "@cf/meta/llama-3.2-11b-vision-instruct")
    assert not r.ok and "403" in r.error and r.model not in MODEL_CONFIG
    wrong = await preflight_one(FakeAI(lambda m, j: chat("I cannot see an image")), "@cf/x/y")
    assert not wrong.ok and "wrong answer" in wrong.error


def _s(acc, sec, passed=0, total=10, banner=1.0):
    return {"accuracy": acc, "mean_seconds": sec, "passed": passed, "total": total,
            "banner_read_accuracy": banner, "failed": total - passed, "errors": 0}


def test_rank_by_accuracy_then_speed():
    s = {"a": _s(0.8, 5), "b": _s(0.9, 9), "c": _s(0.9, 3)}
    assert rank(s) == ["c", "b", "a"]


def test_cross_model_analysis_and_table():
    R = lambda label, ok: Record(0, label, "train", passed=ok)  # noqa: E731
    results = {"@cf/a/one": [R("x", True), R("y", False), R("z", False)],
               "@cf/b/two": [R("x", False), R("y", False), R("z", True)]}
    an = analyse(results)
    assert an == {"challenges": 3, "solved_by_any": 2, "failed_by_all": ["y"]}
    skipped = [__import__("captchasolver.compare", fromlist=["Preflight"]).Preflight(
        "@cf/c/three", False, error="HTTP 403")]
    text = format_table({"@cf/a/one": _s(0.9, 2, 9), "@cf/b/two": _s(0.5, 4, 5)}, skipped, an, color=False)
    assert "🥇" in text and text.index("a/one") < text.index("b/two")
    assert "skipped: HTTP 403" in text and "failed by EVERY model (1): y" in text
    assert "\033[" not in text


def test_default_model_is_gemma4():
    assert client.VLM_MODEL == "@cf/google/gemma-4-26b-a4b-it"
    from captchasolver import grid, screen, text
    import inspect
    for fn in (text.solve_text, screen.solve_screenshot, grid.solve_tiles):
        assert inspect.signature(fn).parameters["model"].default == client.VLM_MODEL


def test_model_config_persists_between_runs(tmp_path):
    path = tmp_path / "cfg.json"
    MODEL_CONFIG["@cf/google/gemma-4-26b-a4b-it"] = client.ModelConfig(
        {"chat_template_kwargs": {"enable_thinking": False}}, send_temperature=False)
    client.save_model_config(path)
    MODEL_CONFIG.clear()
    client.load_model_config(path)
    cfg = MODEL_CONFIG["@cf/google/gemma-4-26b-a4b-it"]
    assert cfg.extra == {"chat_template_kwargs": {"enable_thinking": False}} and not cfg.send_temperature
    MODEL_CONFIG.clear()
    client.load_model_config(tmp_path / "missing.json")   # no cache file: silently plain
    (tmp_path / "bad.json").write_text("{not json")
    client.load_model_config(tmp_path / "bad.json")       # corrupt cache: silently plain
    assert MODEL_CONFIG == {}
