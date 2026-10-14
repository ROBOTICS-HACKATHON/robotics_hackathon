"""FastAPI app: JSON API (+ OpenAPI docs at /api/docs) and the single-page UI at /.

Security model
- The app may be PUBLIC (no login): anyone can use it, but only with their own Cloudflare credentials.
  So expensive endpoints are rate limited per visitor, a test run only starts after a quick check that
  the credentials work, and runs cannot be listed (the unguessable run id is the capability).
- Cloudflare credentials arrive per request in `X-CF-Account-Id` / `X-CF-Api-Token`. They are never
  stored or logged; a test run holds them in memory only while it runs.
- Who may reach this server is decided in front of it (Cloudflare Access, checked by the Worker).
- The test runner may only visit hosts in CAPTCHASOLVER_ALLOWED_HOSTS, so the server cannot be
  used to browse or scan arbitrary sites. /api/solve takes uploaded images only.
"""
from __future__ import annotations

import argparse
import asyncio
import io
import os
import re
import time
from collections import deque
from pathlib import Path
from urllib.parse import urlparse

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response, StreamingResponse
from PIL import Image
from pydantic import BaseModel, Field

from .. import __version__ as _v  # noqa: F401  (import check)
from .. import diag
from ..client import (MODEL_CONFIG, VLM_MODEL, WorkersAI, WorkersAIError, describe_exc, explain_auth_failure,
                      explain_error, save_model_config)
from ..compare import CANDIDATES, preflight_one
from ..grid import solve_grid, tile_centers
from ..screen import LayoutNotFound, build_actions, solve_screenshot
from .runs import RunManager

STATIC = Path(__file__).parent / "static"
STARTED = time.time()
MAX_UPLOAD = 15 * 1024 * 1024
MAX_PIXELS = 60_000_000
DEFAULT_HOSTS = "skibidisigma67rizz.pages.dev"
ACCOUNT_RE = re.compile(r"^[0-9a-f]{32}$")
TOKEN_RE = re.compile(r"^[A-Za-z0-9_.\-]{20,300}$")
MODEL_RE = re.compile(r"^@(cf|hf)/[A-Za-z0-9._\-]+/[A-Za-z0-9._\-]+$")
TARGET_RE = re.compile(r"^[A-Za-z][A-Za-z ]{0,38}$")
VIEWPORT_RE = re.compile(r"^(\d{3,4})x(\d{3,4})$")

Image.MAX_IMAGE_PIXELS = MAX_PIXELS


def allowed_hosts() -> set[str]:
    return {h.strip().lower() for h in os.environ.get("CAPTCHASOLVER_ALLOWED_HOSTS", DEFAULT_HOSTS).split(",") if h.strip()}


# ---------------------------------------------------------------------------- credentials
def make_ai(account_id: str | None, token: str | None) -> WorkersAI:
    if not account_id or not token:
        raise HTTPException(401, "Missing Cloudflare credentials: set them in Settings "
                                 "(headers X-CF-Account-Id and X-CF-Api-Token).")
    if not ACCOUNT_RE.match(account_id.strip().lower()):
        raise HTTPException(400, "The Cloudflare account ID should be 32 hex characters.")
    if not TOKEN_RE.match(token.strip()):
        raise HTTPException(400, "The API token has an unexpected format.")
    return WorkersAI(account_id.strip().lower(), token.strip())


def creds(x_cf_account_id: str | None = Header(None), x_cf_api_token: str | None = Header(None)):
    return x_cf_account_id, x_cf_api_token


def scrub(text: str, token: str | None) -> str:
    return text.replace(token, "***") if token else text


# ---------------------------------------------------------------------------- rate limiting
class RateLimiter:
    """Sliding window per (visitor, bucket), in memory. One container serves everyone, so this is
    shared state. Visitors are identified by the IP the Worker passes in `X-Client-Ip`."""

    def __init__(self) -> None:
        self.hits: dict[tuple[str, str], deque[float]] = {}

    def retry_after(self, visitor: str, bucket: str, limit: int, window: float) -> int | None:
        now = time.monotonic()
        q = self.hits.setdefault((visitor, bucket), deque())
        while q and now - q[0] > window:
            q.popleft()
        if len(q) >= limit:
            return max(1, int(window - (now - q[0])) + 1)
        q.append(now)
        if len(self.hits) > 5000:                      # forget visitors that went quiet
            for k in [k for k, v in self.hits.items() if not v or now - v[-1] > 3600]:
                del self.hits[k]
        return None


RATES = {  # bucket: (env var, default limit, window seconds)
    "solve": ("CAPTCHASOLVER_RATE_SOLVE", 30, 60),
    "preflight": ("CAPTCHASOLVER_RATE_PREFLIGHT", 20, 60),
    "tests": ("CAPTCHASOLVER_RATE_TESTS", 6, 3600),
}


def rate_limits_enabled() -> bool:
    """CAPTCHASOLVER_RATE_LIMITS=off turns every limit off (e.g. a live demo where the robot calls often)."""
    return os.environ.get("CAPTCHASOLVER_RATE_LIMITS", "on").strip().lower() != "off"


def rate_limit(bucket: str):
    env, default, window = RATES[bucket]

    async def dep(request: Request) -> None:
        if not rate_limits_enabled():
            return
        visitor = request.headers.get("x-client-ip") or (request.client.host if request.client else "?")
        wait = request.app.state.limiter.retry_after(visitor, bucket, int(os.environ.get(env, default)), window)
        if wait is not None:
            raise HTTPException(429, f"Too many requests. Try again in {wait} s.", headers={"Retry-After": str(wait)})
    return dep


async def verify_credentials(ai: WorkersAI, model: str) -> None:
    """One tiny text-only request, so a run (which launches Chromium) never starts on bad credentials."""
    if "moondream" in model:
        return
    try:
        await ai.run(model, json={"messages": [{"role": "user", "content": "Reply with OK."}], "max_tokens": 8})
    except WorkersAIError as e:
        kind, advice = await explain_auth_failure(ai, e)
        raise HTTPException(STATUS_FOR_KIND.get(kind, 502), advice if kind != "other" else describe_exc(e, 300))


STATUS_FOR_KIND = {"auth": 401, "forbidden": 403, "licence": 400, "quota": 429}
KIND_FOR_STATUS = {401: "auth", 403: "forbidden", 400: "licence", 429: "quota"}


# ---------------------------------------------------------------------------- model readiness
_ready: set[str] = set(MODEL_CONFIG)   # models whose request settings are known (from the cache)
_locks: dict[str, "asyncio.Lock"] = {}


async def ensure_model_ready(ai: WorkersAI, model: str) -> None:
    """Discover a model's working request settings once (thinking off, temperature...)."""
    import asyncio
    if "moondream" in model or model in _ready:
        return
    lock = _locks.setdefault(model, asyncio.Lock())
    async with lock:
        if model in _ready:
            return
        pre = await preflight_one(ai, model)
        if not pre.ok:
            if pre.kind in STATUS_FOR_KIND:        # we know why: say so, with the right status
                raise HTTPException(STATUS_FOR_KIND[pre.kind], pre.advice)
            raise HTTPException(502, f"{model} did not answer a test image: {pre.error}")
        _ready.add(model)
        try:
            save_model_config()
        except OSError:
            pass


# ---------------------------------------------------------------------------- schemas
class TestParams(BaseModel):
    count: int = Field(100, ge=1, le=200)
    split: str = Field("all", pattern="^(all|train|test)$")
    seed: int = Field(0, ge=0, le=2**31 - 1)
    concurrency: int = Field(3, ge=1, le=8)
    pipeline: str = Field("screenshot", pattern="^(screenshot|dom)$")
    viewport: str = "1914x1080"
    dpr: float = Field(1.0, ge=1.0, le=3.0)
    model: str = VLM_MODEL
    url: str = f"https://{DEFAULT_HOSTS.split(',')[0]}/"


class PreflightParams(BaseModel):
    model: str = VLM_MODEL


def jsonable_errors(exc: RequestValidationError) -> list:
    return [{"loc": list(e.get("loc", ())), "msg": e.get("msg", "")} for e in exc.errors()]


def _check_model(model: str) -> str:
    if not MODEL_RE.match(model):
        raise HTTPException(400, "model must look like @cf/vendor/name")
    return model


def _check_url(url: str) -> str:
    u = urlparse(url)
    if u.scheme not in ("https", "http") or not u.hostname or u.hostname.lower() not in allowed_hosts():
        raise HTTPException(400, f"Tests may only run against: {', '.join(sorted(allowed_hosts()))}")
    return url


def _read_image(data: bytes) -> None:
    try:
        with Image.open(io.BytesIO(data)) as im:
            im.verify()
    except Image.DecompressionBombError:
        raise HTTPException(413, f"Image is larger than {MAX_PIXELS // 1_000_000} megapixels.")
    except Exception:
        raise HTTPException(400, "That file is not a readable PNG/JPEG image.")


async def _chromium_selftest(stress: bool) -> dict:
    """Launch Chromium the way a run does, make a page, screenshot it. Never raises: the answer is the result."""
    from playwright.async_api import async_playwright
    from ..bench import CHROMIUM_ARGS
    from ..web import DEFAULT_URL, _open_image_challenge
    t0, steps = time.perf_counter(), []
    try:
        async with async_playwright() as pw:
            browser = await pw.chromium.launch(args=CHROMIUM_ARGS, timeout=20000)
            steps.append(f"launched {browser.version} in {time.perf_counter() - t0:.1f}s")
            ctx = await browser.new_context(viewport={"width": 1914, "height": 1080})
            page = await ctx.new_page()
            await page.set_content("<h1>hello</h1>")
            await page.screenshot()
            steps.append("blank page + screenshot ok")
            await ctx.close()
            if stress:
                host = next(iter(sorted(allowed_hosts())))

                async def worker(i: int) -> None:
                    c = await browser.new_context(viewport={"width": 1914, "height": 1080})
                    try:
                        p = await c.new_page()
                        await _open_image_challenge(p, f"https://{host}/", i)
                        for _ in range(3):
                            await p.screenshot()
                    finally:
                        await c.close()
                await asyncio.wait_for(asyncio.gather(*[worker(i) for i in range(3)]), 40)
                steps.append("3 demo-site pages in parallel + 9 screenshots ok")
            await browser.close()
        return {"ok": True, "seconds": round(time.perf_counter() - t0, 1), "steps": steps}
    except Exception as e:  # noqa: BLE001 - reporting is the whole point
        return {"ok": False, "seconds": round(time.perf_counter() - t0, 1), "steps": steps,
                "error": describe_exc(e, 600)}


def grid_only_result(tiles: list[int], size: tuple[int, int], rows: int, cols: int, target: str) -> dict:
    """Same shape as solve_screenshot's result, for an image that is just the grid."""
    w, h = size
    centers = tile_centers(size, rows, cols)
    click = [list(centers[i]) for i in tiles]
    res = {"target": target, "image_size": [w, h], "grid_box": [0, 0, w, h], "tiles": tiles,
           "click": click, "verify": None,
           "click_norm": [[round(x / w, 5), round(y / h, 5)] for x, y in click], "verify_norm": None}
    res["actions"] = build_actions(res)
    return res


# ---------------------------------------------------------------------------- app
def create_app() -> FastAPI:
    app = FastAPI(title="captchasolver", version="0.1.0", docs_url="/api/docs",
                  openapi_url="/api/openapi.json", redoc_url=None)
    runs = RunManager(max_concurrent=int(os.environ.get("CAPTCHASOLVER_MAX_RUNS", "1")))
    app.state.runs = runs
    app.state.limiter = RateLimiter()

    @app.middleware("http")
    async def headers(request: Request, call_next):
        resp = await call_next(request)
        resp.headers["X-Content-Type-Options"] = "nosniff"
        resp.headers["Referrer-Policy"] = "no-referrer"
        if request.url.path.startswith("/api/") and "event-stream" not in resp.headers.get("content-type", ""):
            resp.headers["Cache-Control"] = "no-store"
        if request.url.path == "/":
            resp.headers["Content-Security-Policy"] = (
                "default-src 'self'; img-src 'self' data: blob:; style-src 'self' 'unsafe-inline'; "
                "script-src 'self' 'unsafe-inline'; connect-src 'self'; frame-ancestors 'none'")
        return resp

    @app.api_route("/ping", methods=["GET", "HEAD"], include_in_schema=False)
    async def ping():
        return Response("ok", media_type="text/plain")

    @app.api_route("/", methods=["GET", "HEAD"], include_in_schema=False)
    async def index():
        return HTMLResponse((STATIC / "index.html").read_text())

    @app.get("/favicon.ico", include_in_schema=False)
    async def favicon():
        return Response(status_code=204)

    @app.get("/api/health", tags=["meta"])
    async def health():
        """Service info and limits. Needs no Cloudflare credentials."""
        return {"ok": True, "default_model": VLM_MODEL, "allowed_hosts": sorted(allowed_hosts()),
                "rate_limits": rate_limits_enabled(), "uptime_s": int(time.time() - STARTED),
                "limits": {"max_upload_bytes": MAX_UPLOAD, "max_count": 200, "max_concurrency": 8,
                           "max_concurrent_runs": runs.max_concurrent}}

    diag_lock = asyncio.Lock()

    @app.get("/api/diagnostics", tags=["meta"])
    async def diagnostics(stress: bool = False):
        """What this server can give Chromium (memory, /dev/shm, /tmp) and whether Chromium actually runs here.

        `?stress=1` also opens the demo site in 3 pages at once and screenshots them, like a real run does,
        and reports peak memory and any OOM kills. Takes up to ~40 s. One at a time."""
        if diag_lock.locked():
            raise HTTPException(429, "A diagnostics run is already in progress.")
        async with diag_lock:
            before = diag.snapshot()
            result = await _chromium_selftest(stress)
            after = diag.snapshot()
        oom = (after["memory"]["oom_kill"] or 0) - (before["memory"]["oom_kill"] or 0)
        return {"uptime_s": int(time.time() - STARTED), "before": before, "after": after,
                "oom_kills_during_test": oom, "chromium": result, "hints": diag.hints()}

    @app.get("/api/models", tags=["meta"])
    async def models():
        """Vision models worth trying (see `captchasolver compare`)."""
        return [{"id": m, "note": n, "usd_per_m_input": p, "default": m == VLM_MODEL}
                for m, n, p in CANDIDATES]

    @app.post("/api/preflight", tags=["meta"], dependencies=[Depends(rate_limit("preflight"))])
    async def preflight(body: PreflightParams, c=Depends(creds)):
        """Check that a model can read an image with these credentials (also tunes its settings)."""
        model = _check_model(body.model)
        ai = make_ai(*c)
        try:
            pre = await preflight_one(ai, model)
            if pre.ok:
                _ready.add(model)
                try:
                    save_model_config()
                except OSError:
                    pass
            return {"ok": pre.ok, "model": model, "answer": pre.answer, "seconds": pre.seconds,
                    "error": scrub(pre.error, c[1]), "kind": pre.kind, "advice": scrub(pre.advice, c[1])}
        finally:
            await ai.aclose()

    @app.post("/api/solve", tags=["solve"], dependencies=[Depends(rate_limit("solve"))])
    async def solve(image: UploadFile = File(..., description="PNG/JPEG screenshot"),
                    layout: str = Form("screenshot", pattern="^(screenshot|grid)$"),
                    target: str | None = Form(None), grid: str = Form("3x3", pattern=r"^[2-5]x[2-5]$"),
                    model: str = Form(VLM_MODEL), c=Depends(creds)):
        """Screenshot (or just the grid) in; target, picked tiles and a click script out."""
        model = _check_model(model)
        if target is not None and target.strip():
            target = target.strip().lower()
            if not TARGET_RE.match(target):
                raise HTTPException(400, "target should be a short word or phrase, e.g. 'traffic lights'")
        else:
            target = None
        data = await image.read(MAX_UPLOAD + 1)
        if len(data) > MAX_UPLOAD:
            raise HTTPException(413, f"Image is larger than {MAX_UPLOAD // (1024 * 1024)} MB.")
        _read_image(data)
        rows, cols = (int(x) for x in grid.split("x"))
        if layout == "grid" and not target:
            raise HTTPException(400, "layout=grid needs a target (there is no banner to read).")
        ai = make_ai(*c)
        t0 = time.perf_counter()
        try:
            await ensure_model_ready(ai, model)
            if layout == "screenshot":
                res, _, _ = await solve_screenshot(ai, data, target=target, rows=rows, cols=cols, model=model)
            else:
                w, h = Image.open(io.BytesIO(data)).size
                tiles = await solve_grid(ai, data, target, rows=rows, cols=cols, mode="ask", model=model)
                res = grid_only_result(tiles, (w, h), rows, cols, target)
        except LayoutNotFound as e:
            raise HTTPException(422, f"Could not find the captcha dialog: {e}")
        except WorkersAIError as e:
            raise HTTPException(502, scrub(describe_exc(e, 300), c[1]))
        except ValueError as e:
            raise HTTPException(422, scrub(str(e), c[1]))
        finally:
            await ai.aclose()
        return {**res, "layout": layout, "model": model, "seconds": round(time.perf_counter() - t0, 2)}

    # ----- robot: photo of the screen in, press positions out
    def robot_error(status: int, code: str, detail: str, retry: bool) -> JSONResponse:
        """One error shape for every failure, so the robot can decide: take a new photo, or stop."""
        return JSONResponse({"ok": False, "error": code, "detail": detail, "retry": retry}, status_code=status)

    def robot_upstream(kind: str, status: int, detail: str) -> JSONResponse:
        """A failure on Cloudflare's side, with what the robot should do about it."""
        code, retry = {"auth": ("bad_credentials", False), "forbidden": ("bad_credentials", False),
                       "licence": ("model_not_available", False), "quota": ("rate_limited", True)}.get(kind, ("model_error", True))
        return robot_error(status, code, detail, retry)

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, exc: RequestValidationError):
        if request.url.path.startswith("/api/robot/"):
            first = exc.errors()[0] if exc.errors() else {}
            where = ".".join(str(x) for x in first.get("loc", ())[1:])
            return robot_error(422, "bad_request", f"{where}: {first.get('msg', 'invalid request')}", False)
        return JSONResponse({"detail": jsonable_errors(exc)}, status_code=422)

    @app.post("/api/robot/solve", tags=["robot"])
    async def robot_solve(image: UploadFile = File(..., description="Photo of the screen, already normalised to the screen's size"),
                          screen_width: int | None = Form(None, ge=200, le=10000, description="Screen width in px (default: the image's)"),
                          screen_height: int | None = Form(None, ge=200, le=10000, description="Screen height in px (default: the image's)"),
                          target: str | None = Form(None), model: str = Form(VLM_MODEL), c=Depends(creds)):
        """The robot's endpoint. Send the photo (already straightened to the screen's size); get back the
        ordered positions to press, in SCREEN pixels: the tiles first, then Verify.

        Not rate limited. Every failure is `{"ok": false, "error": <code>, "detail": ..., "retry": bool}`;
        `retry: true` means a new photo may succeed. Codes: bad_credentials, bad_request, bad_image,
        no_dialog, layout_inconsistent, cropped, unreadable_target, no_matches, model_error, rate_limited,
        model_not_available. `bad_credentials` also covers a token Cloudflare rejects (retrying will not help)."""
        try:
            model = _check_model(model)
            if target is not None and target.strip():
                target = target.strip().lower()
                if not TARGET_RE.match(target):
                    raise HTTPException(400, "target should be a short word or phrase, e.g. 'traffic lights'")
            else:
                target = None
            data = await image.read(MAX_UPLOAD + 1)
            if len(data) > MAX_UPLOAD:
                raise HTTPException(413, f"Image is larger than {MAX_UPLOAD // (1024 * 1024)} MB.")
            _read_image(data)
            ai = make_ai(*c)
        except HTTPException as e:
            code = {401: "bad_credentials", 400: "bad_request", 413: "bad_image"}.get(e.status_code, "bad_request")
            if e.status_code == 400 and "image" in str(e.detail).lower():
                code = "bad_image"
            return robot_error(e.status_code, code, str(e.detail), retry=False)
        t0 = time.perf_counter()
        try:
            await ensure_model_ready(ai, model)
            res, _, _ = await solve_screenshot(ai, data, target=target, model=model)
        except HTTPException as e:                      # a model that fails its first check
            return robot_upstream(KIND_FOR_STATUS.get(e.status_code, "other"), e.status_code, scrub(str(e.detail), c[1]))
        except LayoutNotFound as e:
            return robot_error(422, e.code, str(e), retry=True)
        except WorkersAIError as e:
            kind, advice = await explain_auth_failure(ai, e)
            return robot_upstream(kind, STATUS_FOR_KIND.get(kind, 502), scrub(advice if kind != "other" else describe_exc(e, 300), c[1]))
        except ValueError as e:
            return robot_error(422, "unreadable_target", scrub(str(e), c[1]), retry=True)
        finally:
            await ai.aclose()
        if not res["tiles"]:
            return robot_error(422, "no_matches", f"No tile looked like '{res['target']}'. Not pressing Verify blindly.", retry=True)
        sw, sh = screen_width or res["image_size"][0], screen_height or res["image_size"][1]
        presses = [{"seq": n + 1, "role": a["role"], **({"tile": a["tile"]} if a["role"] == "tile" else {}),
                    "x": round(a["x_norm"] * sw), "y": round(a["y_norm"] * sh),
                    "x_norm": a["x_norm"], "y_norm": a["y_norm"]}
                   for n, a in enumerate(res["actions"])]
        return {"ok": True, "target": res["target"], "tiles": res["tiles"], "screen": {"width": sw, "height": sh},
                "presses": presses, "tilt_deg": res["tilt_deg"], "model": model,
                "seconds": round(time.perf_counter() - t0, 2)}

    # ----- test runs
    @app.post("/api/tests", status_code=202, tags=["tests"], dependencies=[Depends(rate_limit("tests"))])
    async def start_test(body: TestParams, c=Depends(creds)):
        """Start a run against the demo site. Follow it at /api/tests/{id}/events."""
        m = VIEWPORT_RE.match(body.viewport)
        if not m or not (800 <= int(m[1]) <= 3840 and 700 <= int(m[2]) <= 2160):
            raise HTTPException(400, "viewport must be WxH, between 800x700 and 3840x2160")
        params = body.model_dump() | {"viewport": [int(m[1]), int(m[2])]}
        _check_model(params["model"])
        _check_url(params["url"])
        ai = make_ai(*c)
        try:
            await verify_credentials(ai, params["model"])
            await ensure_model_ready(ai, params["model"])
            run = runs.start(params, ai)
        except RuntimeError as e:
            await ai.aclose()
            raise HTTPException(409, str(e))
        except BaseException:
            await ai.aclose()
            raise
        return {"id": run.id, "events": f"/api/tests/{run.id}/events", "state": f"/api/tests/{run.id}"}

    def _run(run_id: str):
        run = runs.get(run_id)
        if not run:
            raise HTTPException(404, "No such run (results are kept in memory only and expire).")
        return run

    @app.get("/api/tests/{run_id}", tags=["tests"])
    async def test_state(run_id: str):
        return _run(run_id).state()

    @app.get("/api/tests/{run_id}/events", tags=["tests"])
    async def test_events(run_id: str, request: Request, after: int = 0):
        """Server-sent events: start, note, record, summary, end. Resumable with Last-Event-ID."""
        run = _run(run_id)
        last = request.headers.get("last-event-id")
        start = int(last) if last and last.isdigit() else after
        return StreamingResponse(runs.stream(run, start), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    @app.post("/api/tests/{run_id}/cancel", tags=["tests"])
    async def cancel_test(run_id: str):
        run = _run(run_id)
        runs.cancel(run)
        return {"id": run.id, "status": run.status}

    @app.get("/api/tests/{run_id}/images/{name}", tags=["tests"])
    async def test_image(run_id: str, name: str):
        run = _run(run_id)
        if not re.fullmatch(r"\d{3}-[A-Za-z0-9._\-]+\.png", name):
            raise HTTPException(404, "No such image")
        path = run.dir / "failed_runs" / name
        if not path.is_file():
            raise HTTPException(404, "No such image")
        return FileResponse(path, media_type="image/png")

    @app.get("/api/tests/{run_id}/report.json", tags=["tests"])
    async def test_report(run_id: str):
        return JSONResponse(_run(run_id).report(),
                            headers={"Content-Disposition": f'attachment; filename="report-{run_id}.json"'})

    @app.get("/api/tests/{run_id}/failed_runs.zip", tags=["tests"])
    async def test_zip(run_id: str):
        return Response(_run(run_id).failures_zip(), media_type="application/zip",
                        headers={"Content-Disposition": f'attachment; filename="failed_runs-{run_id}.zip"'})

    return app


app = create_app()


def main() -> None:
    import uvicorn
    p = argparse.ArgumentParser(prog="captchasolver serve", description="Run the web UI + API.")
    p.add_argument("--host", default="127.0.0.1", help="use 0.0.0.0 inside a container")
    p.add_argument("--port", type=int, default=8000)
    a = p.parse_args()
    uvicorn.run("captchasolver.server:app", host=a.host, port=a.port, log_level="info")
