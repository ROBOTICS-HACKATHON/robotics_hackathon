"""A run must survive a dead browser, never hang, and say so when it cannot continue."""
import asyncio
import time

import pytest
from fastapi.testclient import TestClient

from captchasolver import bench
from captchasolver.server import api as appmod
from captchasolver.server import runs as runs_mod
from captchasolver.server.api import create_app

ACCOUNT, TOKEN = "a" * 32, "t" * 40
CREDS = {"X-CF-Account-Id": ACCOUNT, "X-CF-Api-Token": TOKEN}


def test_browser_death_is_told_apart_from_ordinary_errors():
    class TargetClosedError(Exception):
        pass
    assert bench.browser_died(TargetClosedError("Mouse.click: Target page, context or browser has been closed"))
    assert bench.browser_died(RuntimeError("Page.screenshot: Target crashed"))
    assert bench.browser_died(Exception("BrowserContext.close: browser has been closed"))
    for ordinary in (TimeoutError("x"), ValueError("no dialog"), RuntimeError("The site neither accepted nor rejected")):
        assert not bench.browser_died(ordinary)


async def test_closing_a_hung_or_dead_browser_never_blocks_the_run():
    class Hung:
        async def close(self):
            await asyncio.sleep(60)

    class Dead:
        async def close(self):
            raise RuntimeError("Target page, context or browser has been closed")
    t0 = time.perf_counter()
    await bench._close(Hung(), timeout=0.2)
    await bench._close(Dead())
    await bench._close(None)
    assert time.perf_counter() - t0 < 2


@pytest.fixture
def client(monkeypatch):
    appmod._ready.add("@cf/google/gemma-4-26b-a4b-it")
    monkeypatch.setenv("CAPTCHASOLVER_ALLOWED_HOSTS", "demo.example.com")
    monkeypatch.setenv("CAPTCHASOLVER_RATE_LIMITS", "off")

    async def ok(ai, model):
        return None
    monkeypatch.setattr(appmod, "verify_credentials", ok)
    with TestClient(create_app()) as c:
        yield c


def events(text):
    import json
    out = []
    for frame in text.split("\n\n"):
        lines = [l for l in frame.splitlines() if not l.startswith(":")]
        if lines:
            d = dict(l.split(": ", 1) for l in lines)
            out.append((d["event"], json.loads(d["data"])))
    return out


def test_a_run_that_never_finishes_is_stopped_with_a_clear_message(client, monkeypatch):
    async def hangs(url, *, progress=None, note=None, ai=None, **kw):
        rec = bench.Record(0, "cats-1", "train", subject="cats", target="cats", passed=True, clicked_ok=True)
        progress(rec, 1, 5)
        await asyncio.sleep(3600)
    monkeypatch.setattr(bench, "run_tests", hangs)
    monkeypatch.setattr(runs_mod, "MAX_RUN_SECONDS", 0.4)
    rid = client.post("/api/tests", json={"url": "https://demo.example.com/"}, headers=CREDS).json()["id"]
    ev = events(client.get(f"/api/tests/{rid}/events").text)
    assert ev[-1] == ("end", {"status": "error"})
    assert any(k == "error" and "longer than" in d["message"] for k, d in ev)
    state = client.get(f"/api/tests/{rid}").json()
    assert state["status"] == "error" and state["summary"]["total"] == 1      # what finished is kept
    # and the single run slot is free again, so the user is not stuck behind it
    monkeypatch.setattr(bench, "run_tests", hangs)
    assert client.post("/api/tests", json={"url": "https://demo.example.com/"}, headers=CREDS).status_code == 202


def test_health_reports_uptime_so_a_restart_is_visible(client):
    h = client.get("/api/health").json()
    assert isinstance(h["uptime_s"], int) and 0 <= h["uptime_s"] < 600


def test_default_parallelism_is_calmer(client):
    from captchasolver.server.api import TestParams
    assert TestParams().concurrency == 3
