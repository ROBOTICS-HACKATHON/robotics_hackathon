import asyncio
import io
import json
import zipfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from captchasolver import bench, grid, screen
from captchasolver.client import MODEL_CONFIG
from captchasolver.compare import Preflight
from captchasolver.server import api as appmod
from captchasolver.server.api import create_app

EXAMPLE = Path(__file__).parent.parent / "examples" / "1.png"
ACCOUNT = "a" * 32
TOKEN = "t" * 40
CREDS = {"X-CF-Account-Id": ACCOUNT, "X-CF-Api-Token": TOKEN}


@pytest.fixture
def client(monkeypatch):
    appmod._ready.add("@cf/google/gemma-4-26b-a4b-it")      # skip the live preflight
    monkeypatch.setenv("CAPTCHASOLVER_ALLOWED_HOSTS", "demo.example.com")
    for var in ("SOLVE", "PREFLIGHT", "TESTS"):              # the real limits have their own tests
        monkeypatch.setenv(f"CAPTCHASOLVER_RATE_{var}", "100000")

    async def creds_ok(ai, model):
        return None
    monkeypatch.setattr(appmod, "verify_credentials", creds_ok)
    with TestClient(create_app()) as c:
        yield c


def png(size=(40, 30), color="white") -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, "PNG")
    return buf.getvalue()


def files(data=None, name="x.png"):
    return {"image": (name, data if data is not None else png(), "image/png")}


# ---------------------------------------------------------------- meta + security headers
def test_health_models_ping_and_page(client):
    assert client.get("/ping").text == "ok"
    assert client.head("/ping").status_code == 200 and client.head("/").status_code == 200   # health probes
    h = client.get("/api/health").json()
    assert h["ok"] and h["default_model"].endswith("gemma-4-26b-a4b-it") and h["allowed_hosts"] == ["demo.example.com"]
    ms = client.get("/api/models").json()
    assert sum(m["default"] for m in ms) == 1
    r = client.get("/")
    assert r.status_code == 200 and "default-src 'self'" in r.headers["content-security-policy"]
    assert r.headers["x-content-type-options"] == "nosniff"
    assert client.get("/api/health").headers["cache-control"] == "no-store"
    assert client.get("/api/openapi.json").json()["info"]["title"] == "captchasolver"


# ---------------------------------------------------------------- credentials
@pytest.mark.parametrize("headers,code", [
    ({}, 401),
    ({"X-CF-Account-Id": ACCOUNT}, 401),
    ({"X-CF-Account-Id": "../../evil", "X-CF-Api-Token": TOKEN}, 400),      # would land in a URL path
    ({"X-CF-Account-Id": "g" * 32, "X-CF-Api-Token": TOKEN}, 400),
    ({"X-CF-Account-Id": ACCOUNT, "X-CF-Api-Token": "short"}, 400),
    ({"X-CF-Account-Id": ACCOUNT, "X-CF-Api-Token": "x" * 30 + " ;rm -rf"}, 400),
])
def test_bad_or_missing_credentials(client, headers, code):
    r = client.post("/api/solve", files=files(), headers=headers)
    assert r.status_code == code
    assert TOKEN not in r.text


def test_model_id_is_validated_because_it_ends_up_in_a_url(client):
    for bad in ("../../x", "@cf/a/b/../../c", "gpt-4", "@cf/only-one-part"):
        r = client.post("/api/solve", files=files(), data={"model": bad}, headers=CREDS)
        assert r.status_code == 400, bad


# ---------------------------------------------------------------- uploads
def test_unreadable_and_oversized_uploads(client, monkeypatch):
    r = client.post("/api/solve", files=files(b"not an image"), headers=CREDS)
    assert r.status_code == 400 and "readable" in r.json()["detail"]
    monkeypatch.setattr(appmod, "MAX_UPLOAD", 100)
    assert client.post("/api/solve", files=files(png((300, 300))), headers=CREDS).status_code == 413


def test_decompression_bomb_is_rejected(client, monkeypatch):
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 100)
    r = client.post("/api/solve", files=files(png((200, 200))), headers=CREDS)
    assert r.status_code == 413


def test_target_is_sanitised_and_grid_layout_needs_one(client):
    assert client.post("/api/solve", files=files(), data={"target": "cats; DROP"}, headers=CREDS).status_code == 400
    r = client.post("/api/solve", files=files(), data={"layout": "grid"}, headers=CREDS)
    assert r.status_code == 400 and "needs a target" in r.json()["detail"]


def test_no_dialog_in_screenshot_is_a_422_not_a_500(client):
    r = client.post("/api/solve", files=files(png((800, 600))), headers=CREDS)
    assert r.status_code == 422 and "dialog" in r.json()["detail"]


# ---------------------------------------------------------------- solving (fake model)
@pytest.mark.skipif(not EXAMPLE.exists(), reason="examples/1.png missing")
def test_solve_screenshot_returns_actions_and_never_echoes_the_token(client, monkeypatch):
    calls = []

    async def fake(ai, image, question, **kw):
        calls.append(kw.get("model"))
        if "banner" in question:
            return "flowers"
        return "yes" if len(calls) % 3 == 0 else "no"
    monkeypatch.setattr(grid, "vlm_query", fake)
    monkeypatch.setattr(screen, "vlm_query", fake)
    r = client.post("/api/solve", files=files(EXAMPLE.read_bytes()), headers=CREDS)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["target"] == "flowers" and body["layout"] == "screenshot" and body["image_size"] == [1914, 1080]
    assert body["actions"][-1]["role"] == "verify" and body["verify_norm"]
    assert all(m.endswith("gemma-4-26b-a4b-it") for m in calls)      # default model is used
    assert TOKEN not in r.text and ACCOUNT not in r.text


def test_solve_grid_layout_maps_tiles_to_click_points(client, monkeypatch):
    async def fake(ai, image, question, **kw):
        return "yes"
    monkeypatch.setattr(grid, "vlm_query", fake)
    r = client.post("/api/solve", files=files(png((300, 300))), data={"layout": "grid", "target": "Cats"},
                    headers=CREDS)
    assert r.status_code == 200, r.text
    b = r.json()
    assert b["target"] == "cats" and b["tiles"] == list(range(9)) and b["verify"] is None
    assert b["click"][4] == [150, 150] and b["actions"][-1]["role"] == "tile"


def test_upstream_cloudflare_errors_become_502(client, monkeypatch):
    from captchasolver.client import WorkersAIError

    async def boom(ai, image, question, **kw):
        raise WorkersAIError(f"model: 403 license; leaked {TOKEN}")
    monkeypatch.setattr(grid, "vlm_query", boom)
    r = client.post("/api/solve", files=files(png((300, 300))), data={"layout": "grid", "target": "cats"},
                    headers=CREDS)
    assert r.status_code == 502 and "403" in r.json()["detail"]
    assert TOKEN not in r.text                       # scrubbed even if upstream text contained it


def test_model_is_preflighted_once_per_process(client, monkeypatch):
    appmod._ready.discard("@cf/mistralai/mistral-small-3.1-24b-instruct")
    seen = []

    async def fake_pre(ai, model):
        seen.append(model)
        return Preflight(model, True, "HELLO 42", 0.1, 0)

    async def yes(ai, image, question, **kw):
        return "yes"
    monkeypatch.setattr(appmod, "preflight_one", fake_pre)
    monkeypatch.setattr(appmod, "save_model_config", lambda: None)
    monkeypatch.setattr(grid, "vlm_query", yes)
    data = {"layout": "grid", "target": "cats", "model": "@cf/mistralai/mistral-small-3.1-24b-instruct"}
    for _ in range(2):
        assert client.post("/api/solve", files=files(png((300, 300))), data=data, headers=CREDS).status_code == 200
    assert seen == ["@cf/mistralai/mistral-small-3.1-24b-instruct"]


def test_failed_preflight_is_a_clear_502(client, monkeypatch):
    appmod._ready.discard("@cf/x/never")

    async def fake_pre(ai, model):
        return Preflight(model, False, error="HTTP 403 license")
    monkeypatch.setattr(appmod, "preflight_one", fake_pre)
    r = client.post("/api/solve", files=files(png((300, 300))),
                    data={"layout": "grid", "target": "cats", "model": "@cf/x/never"}, headers=CREDS)
    assert r.status_code == 502 and "did not answer" in r.json()["detail"]


# ---------------------------------------------------------------- test runs
def fake_run_tests_factory(n=3, delay=0.0, boom=False):
    async def fake(url, *, progress=None, note=None, ai=None, fail_dir=None, **kw):
        note("note: fake run")
        recs = []
        for i in range(n):
            await asyncio.sleep(delay)
            rec = bench.Record(i, f"cats-{i}", "train", subject="cats", target="cats" if i else "dogs",
                               tiles=[i], clicked_ok=True, passed=(i == 0))
            if not rec.passed:
                (Path(fail_dir) / f"{i:03d}-cats-{i}.png").write_bytes(b"\x89PNG state")
                (Path(fail_dir) / f"{i:03d}-cats-{i}-plan.png").write_bytes(b"\x89PNG plan")
            recs.append(rec)
            progress(rec, i + 1, n)
            if boom and i == 1:
                raise RuntimeError("browser exploded")
        return recs, bench.summarize(recs)
    return fake


def sse_events(text: str):
    out = []
    for frame in text.split("\n\n"):
        lines = [l for l in frame.splitlines() if not l.startswith(":")]
        if lines:
            d = dict(l.split(": ", 1) for l in lines)
            out.append((d["event"], json.loads(d["data"])))
    return out


def test_run_streams_events_serves_images_and_exports(client, monkeypatch):
    monkeypatch.setattr(bench, "run_tests", fake_run_tests_factory(3))
    body = {"count": 3, "url": "https://demo.example.com/"}
    r = client.post("/api/tests", json=body, headers=CREDS)
    assert r.status_code == 202, r.text
    run_id = r.json()["id"]
    events = sse_events(client.get(f"/api/tests/{run_id}/events").text)
    kinds = [k for k, _ in events]
    assert kinds[0] == "start" and kinds[-1] == "end" and kinds.count("record") == 3 and "summary" in kinds
    assert TOKEN not in json.dumps(events) and ACCOUNT not in json.dumps(events)
    rec1 = [d for k, d in events if k == "record"][1]["record"]
    assert rec1["passed"] is False and rec1["cause"] == "wrong tile selection: picked [1]"
    rec0 = [d for k, d in events if k == "record"][0]["record"]
    assert rec0["passed"] is True and rec0["cause"] == ""      # passes carry no failure reason
    assert rec1["images"]["state"].endswith("/images/001-cats-1.png") and rec1["images"]["plan"]

    state = client.get(f"/api/tests/{run_id}").json()
    assert state["status"] == "done" and state["done"] == 3 and state["summary"]["passed"] == 1
    img = client.get(rec1["images"]["state"])
    assert img.status_code == 200 and img.content.startswith(b"\x89PNG")
    z = zipfile.ZipFile(io.BytesIO(client.get(f"/api/tests/{run_id}/failed_runs.zip").content))
    assert "failed_runs/001-cats-1.png" in z.namelist() and "report.json" in z.namelist()
    rep = client.get(f"/api/tests/{run_id}/report.json").json()
    assert rep["summary"]["total"] == 3 and TOKEN not in json.dumps(rep) and "token" not in json.dumps(rep["config"]).lower()
    assert len(run_id) == 32 and int(run_id, 16) >= 0            # unguessable: the id is the capability


def test_events_can_resume_after_a_reconnect(client, monkeypatch):
    monkeypatch.setattr(bench, "run_tests", fake_run_tests_factory(2))
    run_id = client.post("/api/tests", json={"url": "https://demo.example.com/"}, headers=CREDS).json()["id"]
    full = sse_events(client.get(f"/api/tests/{run_id}/events").text)
    resumed = sse_events(client.get(f"/api/tests/{run_id}/events", headers={"Last-Event-ID": "2"}).text)
    assert [k for k, _ in resumed] == [k for k, _ in full][2:]


def test_error_in_run_is_reported_not_raised(client, monkeypatch):
    monkeypatch.setattr(bench, "run_tests", fake_run_tests_factory(3, boom=True))
    run_id = client.post("/api/tests", json={"url": "https://demo.example.com/"}, headers=CREDS).json()["id"]
    events = sse_events(client.get(f"/api/tests/{run_id}/events").text)
    assert ("error", {"message": "RuntimeError: browser exploded"}) in events
    state = client.get(f"/api/tests/{run_id}").json()
    assert state["status"] == "error" and state["summary"]["total"] == 2     # partial results kept


def test_hosts_are_allow_listed_and_inputs_validated(client):
    for url in ("https://evil.example.org/", "http://169.254.169.254/", "file:///etc/passwd", "https://demo.example.com.evil.org/"):
        r = client.post("/api/tests", json={"url": url}, headers=CREDS)
        assert r.status_code == 400, url
    assert client.post("/api/tests", json={"url": "https://demo.example.com/", "count": 5000}, headers=CREDS).status_code == 422
    assert client.post("/api/tests", json={"url": "https://demo.example.com/", "viewport": "10x10"}, headers=CREDS).status_code == 400
    assert client.post("/api/tests", json={"url": "https://demo.example.com/", "pipeline": "files"}, headers=CREDS).status_code == 422
    assert client.post("/api/tests", json={"url": "https://demo.example.com/"}).status_code == 401


def test_only_one_run_at_a_time_and_cancel_keeps_partial_results(client, monkeypatch):
    monkeypatch.setattr(bench, "run_tests", fake_run_tests_factory(50, delay=0.05))
    body = {"url": "https://demo.example.com/"}
    run_id = client.post("/api/tests", json=body, headers=CREDS).json()["id"]
    assert client.post("/api/tests", json=body, headers=CREDS).status_code == 409
    import time as _t
    _t.sleep(0.2)
    assert client.post(f"/api/tests/{run_id}/cancel").status_code == 200
    events = sse_events(client.get(f"/api/tests/{run_id}/events").text)
    assert events[-1] == ("end", {"status": "cancelled"})
    state = client.get(f"/api/tests/{run_id}").json()
    assert state["status"] == "cancelled" and 0 < state["done"] < 50
    assert client.post("/api/tests", json=body, headers=CREDS).status_code == 202      # slot is free again


def test_unknown_run_and_path_traversal(client):
    assert client.get("/api/tests/nope").status_code == 404
    assert client.get("/api/tests/nope/images/001-a.png").status_code == 404


def test_image_names_cannot_escape_the_run_directory(client, monkeypatch):
    monkeypatch.setattr(bench, "run_tests", fake_run_tests_factory(1))
    run_id = client.post("/api/tests", json={"url": "https://demo.example.com/"}, headers=CREDS).json()["id"]
    client.get(f"/api/tests/{run_id}/events")
    for name in ("..%2F..%2Fetc%2Fpasswd", "001-a.png%00.txt", "..%5C..%5Cx.png", ".env"):
        assert client.get(f"/api/tests/{run_id}/images/{name}").status_code in (404, 422), name


def test_old_runs_are_evicted_and_their_files_deleted(client, monkeypatch):
    monkeypatch.setattr(bench, "run_tests", fake_run_tests_factory(1))
    ids = []
    for _ in range(6):
        rid = client.post("/api/tests", json={"url": "https://demo.example.com/"}, headers=CREDS).json()["id"]
        client.get(f"/api/tests/{rid}/events")
        ids.append(rid)
    runs = client.app.state.runs
    assert len(runs.runs) <= 4 and ids[0] not in runs.runs and ids[-1] in runs.runs


# ---------------------------------------------------------------- public deployment guard rails
def test_runs_cannot_be_listed(client, monkeypatch):
    monkeypatch.setattr(bench, "run_tests", fake_run_tests_factory(1))
    client.post("/api/tests", json={"url": "https://demo.example.com/"}, headers=CREDS)
    assert client.get("/api/tests").status_code in (404, 405)       # others' ids must not be discoverable


def test_rate_limit_per_visitor_with_retry_after(client, monkeypatch):
    monkeypatch.setenv("CAPTCHASOLVER_RATE_SOLVE", "2")
    send = lambda ip: client.post("/api/solve", files=files(png((300, 300))), data={"layout": "grid"},  # noqa: E731
                                  headers={**CREDS, "X-Client-Ip": ip})
    assert [send("1.1.1.1").status_code for _ in range(2)] == [400, 400]    # allowed (then fails validation)
    r = send("1.1.1.1")
    assert r.status_code == 429 and int(r.headers["retry-after"]) >= 1
    assert "Too many requests" in r.json()["detail"]
    assert send("2.2.2.2").status_code == 400                        # a different visitor has their own budget


def test_rate_limit_window_slides(monkeypatch):
    lim = appmod.RateLimiter()
    t = [1000.0]
    monkeypatch.setattr(appmod.time, "monotonic", lambda: t[0])
    assert lim.retry_after("v", "b", 2, 60) is None and lim.retry_after("v", "b", 2, 60) is None
    wait = lim.retry_after("v", "b", 2, 60)
    assert wait and 1 <= wait <= 61
    t[0] += 61
    assert lim.retry_after("v", "b", 2, 60) is None                  # the window moved on


def test_test_runs_have_a_tighter_default_limit(client, monkeypatch):
    monkeypatch.delenv("CAPTCHASOLVER_RATE_TESTS")
    monkeypatch.setattr(bench, "run_tests", fake_run_tests_factory(1))
    codes = []
    for _ in range(8):
        r = client.post("/api/tests", json={"url": "https://demo.example.com/"}, headers=CREDS)
        if r.status_code == 202:
            client.get(f"/api/tests/{r.json()['id']}/events")        # let it finish so the slot frees up
        codes.append(r.status_code)
    assert codes[:6] == [202] * 6 and codes[6:] == [429, 429]


async def test_credential_check_maps_cloudflare_rejections():
    """Uses Cloudflare's real error shapes; each cause gets its own status and its own advice."""
    import httpx
    from captchasolver.client import WorkersAI, WorkersAIError
    from fastapi import HTTPException

    def ai_with(run_status, run_errors, token_result=None):
        def handler(request):
            if "/tokens/verify" in request.url.path:
                if token_result is None:
                    return httpx.Response(401, json={"success": False, "errors": [{"code": 1000}]})
                return httpx.Response(200, json={"success": True, "result": token_result})
            if run_status == 200:
                return httpx.Response(200, json={"success": True, "result": {"response": "OK"}})
            return httpx.Response(run_status, json={"success": False, "errors": run_errors})
        return WorkersAI(ACCOUNT, TOKEN, transport=httpx.MockTransport(handler), backoff=0, retries=0)

    async def failure(ai):
        with pytest.raises(HTTPException) as e:
            await appmod.verify_credentials(ai, "@cf/google/gemma-4-26b-a4b-it")
        return e.value

    ok = ai_with(200, [])
    await appmod.verify_credentials(ok, "@cf/google/gemma-4-26b-a4b-it")
    AUTH = [{"code": 10000, "message": "Authentication error"}]
    e = await failure(ai_with(401, AUTH, {"status": "expired"}))
    assert e.status_code == 401 and "has expired" in e.detail
    e = await failure(ai_with(401, AUTH, {"status": "disabled"}))
    assert e.status_code == 401 and "revoked" in e.detail
    e = await failure(ai_with(401, AUTH, {"status": "active"}))
    assert e.status_code == 401 and "valid, but Cloudflare did not accept it for THIS account" in e.detail
    e = await failure(ai_with(401, AUTH, None))
    assert e.status_code == 401 and "does not recognise this token" in e.detail
    # a licence problem is NOT a credentials problem, even though Cloudflare answers 403 (your earlier Llama 3.2 output)
    e = await failure(ai_with(403, [{"code": 5016, "message": "AiError: Model Agreement: Prior to using this model, you must submit the prompt 'agree'."}]))
    assert e.status_code == 400 and "licence" in e.detail and "Model Agreement" in e.detail
    # a used-up allowance is not a credentials problem either
    e = await failure(ai_with(429, [{"code": 4006, "message": "you have used up your daily free allocation of 10,000 neurons"}]))
    assert e.status_code == 429 and "not a credentials problem" in e.detail and "10,000 neurons" in e.detail
    # a recognised token that may not run Workers AI
    e = await failure(ai_with(403, [{"code": 9109, "message": "Unauthorized to access requested resource"}]))
    assert e.status_code == 403 and "Workers AI permission" in e.detail and "Unauthorized to access" in e.detail
    # anything else: say what happened
    e = await failure(ai_with(500, [{"code": 3000, "message": "internal"}]))
    assert e.status_code == 502
    for client_ in (ok,):
        await client_.aclose()


def test_run_does_not_start_when_credentials_are_rejected(client, monkeypatch):
    from fastapi import HTTPException

    async def rejected(ai, model):
        raise HTTPException(401, "Cloudflare rejected these credentials.")
    monkeypatch.setattr(appmod, "verify_credentials", rejected)
    started = []
    monkeypatch.setattr(bench, "run_tests", lambda *a, **k: started.append(1))
    r = client.post("/api/tests", json={"url": "https://demo.example.com/"}, headers=CREDS)
    assert r.status_code == 401 and not started and not client.app.state.runs.runs


# ---------------------------------------------------------------- robot endpoint
def camera_photo(deg=3, tint=(1.1, 1.0, 0.9)) -> bytes:
    """The reference screenshot, tinted, slightly tilted and JPEG-compressed: a stand-in for a straightened camera photo."""
    import numpy as np
    img = Image.open(EXAMPLE).convert("RGB")
    img = Image.fromarray((np.asarray(img).astype(float) * tint).clip(0, 255).astype("uint8"))
    img = img.rotate(deg, resample=Image.BICUBIC, fillcolor=(255, 255, 255))
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=70)
    return buf.getvalue()


@pytest.fixture
def robot_fake(monkeypatch):
    async def fake(ai, image, question, **kw):
        if "banner" in question:
            return "flowers"
        return "yes" if hashlib_md5(image) % 3 == 0 else "no"
    monkeypatch.setattr(grid, "vlm_query", fake)
    monkeypatch.setattr(screen, "vlm_query", fake)


def hashlib_md5(b: bytes) -> int:
    import hashlib
    return hashlib.md5(b).digest()[0]


def robot_post(client, data, **form):
    return client.post("/api/robot/solve", files={"image": ("photo.jpg", data, "image/jpeg")}, data=form, headers=CREDS)


@pytest.mark.skipif(not EXAMPLE.exists(), reason="examples/1.png missing")
def test_robot_endpoint_returns_ordered_presses_in_screen_pixels(client, robot_fake):
    from captchasolver.grid import tile_centers
    r = robot_post(client, camera_photo(), screen_width="1920", screen_height="1080")
    assert r.status_code == 200, r.text
    b = r.json()
    assert b["ok"] and b["target"] == "flowers" and b["screen"] == {"width": 1920, "height": 1080}
    p = b["presses"]
    assert [x["seq"] for x in p] == list(range(1, len(p) + 1))
    assert p[-1]["role"] == "verify" and all(x["role"] == "tile" for x in p[:-1])     # Verify is pressed last
    assert [x["tile"] for x in p[:-1]] == b["tiles"] == sorted(b["tiles"])
    # where the tiles truly are in the TILTED photo, scaled to the 1920x1080 screen
    ref = screen.find_layout(Image.open(EXAMPLE).convert("RGB"))
    gx, gy, gw, _ = ref.grid
    for press in p[:-1]:
        dx, dy = tile_centers((gw, gw), 3, 3)[press["tile"]]
        tx, ty = screen._rotate_point(gx + dx, gy + dy, 3, (1914, 1080), back=False)
        assert abs(press["x"] - tx * 1920 / 1914) < 20 and abs(press["y"] - ty) < 20, press
    vx, vy = screen._rotate_point(*ref.verify, 3, (1914, 1080), back=False)
    assert abs(p[-1]["x"] - vx * 1920 / 1914) < 15 and abs(p[-1]["y"] - vy) < 15
    assert TOKEN not in r.text and ACCOUNT not in r.text


@pytest.mark.skipif(not EXAMPLE.exists(), reason="examples/1.png missing")
def test_robot_screen_size_defaults_to_the_image_and_target_can_be_forced(client, robot_fake):
    b = robot_post(client, camera_photo(0), target="Roses").json()
    assert b["screen"] == {"width": 1914, "height": 1080} and b["target"] == "roses"
    assert all(0 <= x["x_norm"] <= 1 and 0 <= x["y_norm"] <= 1 for x in b["presses"])


def test_robot_errors_share_one_shape_and_say_whether_to_retry(client, robot_fake, monkeypatch):
    def check(resp, status, code, retry):
        assert resp.status_code == status, resp.text
        b = resp.json()
        assert b["ok"] is False and b["error"] == code and b["retry"] is retry and b["detail"]
        assert TOKEN not in resp.text
    check(robot_post(client, png((800, 600))), 422, "no_dialog", True)                       # nothing to find: new photo
    check(client.post("/api/robot/solve", files={"image": ("x.jpg", png(), "image/jpeg")}), 401, "bad_credentials", False)
    check(client.post("/api/robot/solve", files={"image": ("x.jpg", png(), "image/jpeg")},
                      headers={"X-CF-Account-Id": "nope", "X-CF-Api-Token": TOKEN}), 400, "bad_request", False)
    check(robot_post(client, b"not an image"), 400, "bad_image", False)
    check(robot_post(client, png(), model="../../etc"), 400, "bad_request", False)
    check(robot_post(client, png(), screen_width="5"), 422, "bad_request", False)            # validation, same shape
    check(client.post("/api/robot/solve", headers=CREDS), 422, "bad_request", False)         # no image at all
    monkeypatch.setattr(appmod, "MAX_UPLOAD", 100)
    check(robot_post(client, png((300, 300))), 413, "bad_image", False)


@pytest.mark.skipif(not EXAMPLE.exists(), reason="examples/1.png missing")
def test_robot_refuses_instead_of_pressing_when_nothing_matches_or_models_fail(client, monkeypatch):
    from captchasolver.client import WorkersAIError

    async def none_match(ai, image, question, **kw):
        return "flowers" if "banner" in question else "no"
    monkeypatch.setattr(grid, "vlm_query", none_match)
    monkeypatch.setattr(screen, "vlm_query", none_match)
    r = robot_post(client, camera_photo(0))
    assert r.status_code == 422 and r.json()["error"] == "no_matches" and r.json()["retry"] is True

    async def boom(ai, image, question, **kw):
        raise WorkersAIError(f"model: 429 slow down {TOKEN}")
    monkeypatch.setattr(grid, "vlm_query", boom)
    monkeypatch.setattr(screen, "vlm_query", boom)
    r = robot_post(client, camera_photo(0))
    assert r.status_code == 502 and r.json()["error"] == "model_error" and TOKEN not in r.text

    async def unreadable(ai, image, question, **kw):
        return "???" if "banner" in question else "yes"
    monkeypatch.setattr(grid, "vlm_query", unreadable)
    monkeypatch.setattr(screen, "vlm_query", unreadable)
    assert robot_post(client, camera_photo(0)).json()["error"] == "unreadable_target"


@pytest.mark.skipif(not EXAMPLE.exists(), reason="examples/1.png missing")
def test_robot_endpoint_is_never_rate_limited(client, robot_fake, monkeypatch):
    monkeypatch.setenv("CAPTCHASOLVER_RATE_SOLVE", "1")
    monkeypatch.setenv("CAPTCHASOLVER_RATE_TESTS", "1")
    photo = camera_photo(0)
    assert all(robot_post(client, photo).status_code == 200 for _ in range(6))
    # ...while the general endpoint with the same setting still is
    ok = [client.post("/api/solve", files=files(png((300, 300))), data={"layout": "grid"}, headers=CREDS).status_code for _ in range(3)]
    assert ok[-1] == 429


def test_rate_limits_can_be_switched_off_for_everything(client, monkeypatch):
    monkeypatch.setenv("CAPTCHASOLVER_RATE_SOLVE", "1")
    assert client.get("/api/health").json()["rate_limits"] is True
    monkeypatch.setenv("CAPTCHASOLVER_RATE_LIMITS", "off")
    assert client.get("/api/health").json()["rate_limits"] is False
    codes = {client.post("/api/solve", files=files(png((300, 300))), data={"layout": "grid"}, headers=CREDS).status_code for _ in range(5)}
    assert 429 not in codes
    monkeypatch.setenv("CAPTCHASOLVER_RATE_LIMITS", "ON")            # anything but 'off' keeps them on
    assert client.get("/api/health").json()["rate_limits"] is True


# ---------------------------------------------------------------- credential problems are told apart end to end
@pytest.mark.skipif(not EXAMPLE.exists(), reason="examples/1.png missing")
@pytest.mark.parametrize("status,errors,expect_status,code,retry,words", [
    (401, [{"code": 10000, "message": "Authentication error"}], 401, "bad_credentials", False, "has expired"),
    (403, [{"code": 5016, "message": "AiError: Model Agreement: submit 'agree'"}], 400, "model_not_available", False, "licence"),
    (429, [{"code": 4006, "message": "you have used up your daily free allocation of 10,000 neurons"}], 429, "rate_limited", True, "not a credentials problem"),
    (403, [{"code": 9109, "message": "Unauthorized to access requested resource"}], 403, "bad_credentials", False, "Workers AI permission"),
])
def test_robot_endpoint_tells_the_robot_whether_retrying_can_help(client, monkeypatch, status, errors, expect_status, code, retry, words):
    from captchasolver import client as client_mod
    from captchasolver.client import WorkersAIError

    async def refuse(ai, image, question, **kw):
        raise WorkersAIError("model: refused", status, errors)

    async def expired(ai):
        return "expired"
    monkeypatch.setattr(grid, "vlm_query", refuse)
    monkeypatch.setattr(screen, "vlm_query", refuse)
    monkeypatch.setattr(client_mod, "token_status", expired)
    r = robot_post(client, camera_photo(0))
    assert r.status_code == expect_status, r.text
    b = r.json()
    assert b["ok"] is False and b["error"] == code and b["retry"] is retry and words in b["detail"]
    assert TOKEN not in r.text


def test_settings_test_connection_returns_plain_english_advice(client, monkeypatch):
    from captchasolver.compare import Preflight

    async def quota(ai, model):
        return Preflight(model, False, error="WorkersAIError: ... 429 ...", kind="quota",
                         advice="Cloudflare is rate limiting you or your allowance is used up. This is not a credentials problem.")
    monkeypatch.setattr(appmod, "preflight_one", quota)
    b = client.post("/api/preflight", json={}, headers=CREDS).json()
    assert b["ok"] is False and b["kind"] == "quota" and "not a credentials problem" in b["advice"]


def test_a_known_cause_becomes_the_right_status_when_a_model_fails_its_first_check(client, monkeypatch):
    from captchasolver.compare import Preflight
    appmod._ready.discard("@cf/x/never")

    async def expired(ai, model):
        return Preflight(model, False, error="x", kind="auth", advice="The API token has expired. Create a new one.")
    monkeypatch.setattr(appmod, "preflight_one", expired)
    r = client.post("/api/solve", files=files(png((300, 300))),
                    data={"layout": "grid", "target": "cats", "model": "@cf/x/never"}, headers=CREDS)
    assert r.status_code == 401 and "has expired" in r.json()["detail"]
