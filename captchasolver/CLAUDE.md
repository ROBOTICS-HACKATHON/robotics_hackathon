# captchasolver

Python tool that solves image-grid captchas ("select all images with X") with vision models on
Cloudflare Workers AI and emits a click script (JSON `actions`) for a robot arm or Playwright.
Part of a robotics hackathon repo. The robot code in `../touch` (`solve.py`, `run_captcha.py`) calls `POST /api/robot/solve`, so
keep that endpoint's shape stable; the rest of the repo (LeRobot setup, `lr`, Docker) does not use this package's code.

## Scope: do not extend beyond this

- Only the team's own demo site (default URL in `web.py`) and test captchas. Never point this at a
  third-party site or at real reCAPTCHA/Turnstile/hCaptcha/GeeTest widgets.
- The solver must work from pixels only. The demo site ships its answer key in client-side JS;
  never read or use it for solving. Failure reasons are derived from observations, not the key.

## Commands

```bash
pip install -e '.[dev]' && playwright install chromium
pytest -q                              # offline, no credentials needed
captchasolver serve                    # web UI + API on localhost:8000 (pip install -e '.[server]')
captchasolver locate photo.jpg         # offline: is the dialog found in this photo? (no credentials)
cd cloudflare && npm install && npm test   # Worker tests
captchasolver test --count 10          # needs CLOUDFLARE_ACCOUNT_ID / CLOUDFLARE_API_TOKEN
captchasolver compare --preflight-only # check models on the account; caches settings
```

No linter is configured. Match the existing style (type hints, short docstrings, `from __future__ import annotations`).

## Architecture

`screen.find_layout` (colour/shape, all thresholds relative) → `screen.read_target` (banner via
`vlm_query`) → `grid.solve_grid` / `solve_tiles` (parallel per-tile yes/no) →
`screen.build_actions` (tiles first, Verify last). `bench.py` drives the demo site with
Playwright (`test`, `replay`); `compare.py` ranks models; `capture.py` grabs the screen per platform.

## Robot flow (the product's real input)

Real input is a **screenshot of the screen** (`CAMERA=screen`; the robot side cuts the page out at its 4 corner markers and
scales it, see `../touch/see.py`), sent to `POST /api/robot/solve`; the robot presses the returned `presses` (tiles then Verify).
A camera photo of the screen is also supported, which is why the finder must stay camera-tolerant:

- `screen.find_layout`/`locate` must stay camera-tolerant: adaptive blue (blueness vs red/green, threshold
  from the photo), grid top from the strongest edge under the banner, tilt estimated from the banner's top
  edge and removed (`locate` returns the upright image and `tilt`; positions are mapped back with
  `_rotate_point`). Never go back to fixed RGB or pure-white-gap logic: it failed on any tint or dimming.
- **Refuse, never guess.** A robot pressing from a half-found layout is the worst outcome. Keep the
  cross-checks (banner aspect, Verify where the proportions say, grid inside the image, tilt <= 10°) and
  the `LayoutNotFound.code` values (`no_dialog`, `layout_inconsistent`, `cropped`) that the endpoint maps
  to `retry: true`. Loosening a check needs a test showing it cannot turn a refusal into a wrong press.
- `tests/test_camera.py` simulates camera conditions on `examples/1.png` and judges by what the robot
  presses (Verify + 9 tile centres vs where they truly are in the degraded image, tolerances 18/40 px).
  When you judge a tilted/skewed photo, compare against the point's **transformed** position, not the
  original one: that mistake once produced false alarms that looked like a bug.
- `/api/robot/solve` is exempt from rate limits and has ONE error shape (`ok:false, error, detail, retry`).
  `no_matches` is an error on purpose: do not press Verify with zero tiles selected.
- Real-photo failures are the thing to collect: run `captchasolver locate` on them (offline, free) and add
  the failing case as a test. The simulation does not cover moiré, reflections or lens distortion.

## Web app and Cloudflare deployment

- `server/api.py` (FastAPI) reuses the solver; `server/runs.py` keeps test runs **in memory only**
  (no persistence: that is a product decision). `server/static/index.html` is one dependency-free
  page; build DOM with `textContent`/`el()`, never `innerHTML` with server data.
- Cloudflare credentials arrive per request in `X-CF-Account-Id` / `X-CF-Api-Token`, are validated by
  format (they end up in URL paths), and must never be stored, logged or echoed. A run keeps its
  client only while running. Scrub tokens from upstream error text.
- Anything user-supplied that reaches a URL, a browser or a prompt is validated (model id, target,
  account id, test URL). The test runner may only visit `CAPTCHASOLVER_ALLOWED_HOSTS`: do not loosen
  this, it keeps the server from being an open browser/scanner.
- The deployment is **public by default** (`AUTH_MODE=public` in `wrangler.jsonc`; Cloudflare Zero
  Trust is not available to the team). Public means: anyone may use it with their own credentials, so keep
  the self-protection: per-visitor rate limits (`RateLimiter`, keyed on the Worker's `X-Client-Ip`),
  `verify_credentials` before a run launches Chromium, one run at a time, host allow-list, unguessable run
  ids and no listing endpoint. Do not add features that spend the host's money or reach other sites.
  The live demo ships with `RATE_LIMITS: "off"` (user's choice, robot calls often, keys are theirs); the
  switch is `CAPTCHASOLVER_RATE_LIMITS=off`.
- `cloudflare/` is a Worker (`src/index.ts`) in front of a Cloudflare Container built from the
  `Dockerfile`. `src/access.ts` decides access: `public` only when `AUTH_MODE` is exactly that, else it
  verifies the Access JWT and fails closed when unconfigured; keep it pure (only `jose`) so `npm test`
  runs it in plain Node. The Worker strips cookies/Access headers
  before forwarding (`forward.ts`). Do not deploy, and never set `ALLOW_UNAUTHENTICATED_DEV=true`
  outside `wrangler dev`.
- Module is `server/api.py`, not `app.py`: `from .app import app` would shadow the submodule.
- Verify UI changes in a real browser (Playwright against `captchasolver serve` with a faked
  `vlm_query`). The page has a strict CSP, so Playwright's string `wait_for_function` is blocked;
  poll with `evaluate` instead.

## Conventions

- All vision calls go through `client.vlm_query`; Workers AI response shapes vary, so parse with
  `client.find_answer`. Chat vs Moondream payloads are chosen there.
- Executors click with the normalised coordinates (`x_norm`/`y_norm`) of the *current* viewport.
  Never hard-code pixel sizes; thresholds in `find_layout` are fractions of the banner width.
- Format exceptions with `client.describe_exc`. Never `str(e).splitlines()[0]` (empty messages
  such as `httpx.ReadTimeout` crash it). One failed challenge must never abort a run.
- Tests are offline: monkeypatch `grid.vlm_query` / `screen.vlm_query`, use `httpx.MockTransport`
  for the client, and fake page objects. Browser mechanics can be checked against the live demo
  site with a fake model (no credentials needed).

## Gotchas learned the hard way

- Moondream returns `result: {}` on the user's account. Do not make it a default.
- Reasoning models (Gemma 4, Kimi) need a `max_tokens` floor and sometimes a thinking-off switch.
  `compare` preflight discovers the working variant and caches it in
  `~/.cache/captchasolver/model_config.json` (`CAPTCHASOLVER_CONFIG` overrides). Default model is
  `@cf/google/gemma-4-26b-a4b-it` with thinking off.
- The test runner's default URL must be the demo site's `/training/` page: the root now is a Minesweeper page without the challenge picker
  (`#challenge-number`): pointed at the root, a run reports "only 0 challenges available" and tests nothing.
- The site's Challenge dropdown values are `image:<n>` (they were `0-9` once). The harness reads
  them from the page, never hard-code the list. A session cannot skip a failed challenge, so
  `test` picks challenges itself instead of using the site's Start session.
- Resampled screenshots blur the banner's edge. Find the grid by waiting for the white gap under
  the banner, not by the first non-white row.
- WSL capture must go through `powershell.exe` (Windows screen, DPI-aware), not WSLg.

- **Never turn "no answer yet" into FAIL.** After Verify the site answers with one of: the success state,
  "Incorrect selection" (immediate), or a navigation to `/verified/` (only after it accepted; slow under
  load). `web._outcome` waits for any of the three (30 s) and raises `SiteDidNotAnswer` (an ERROR) if none
  appears. A 5 s timeout once labelled 17 of 100 correct answers as failures. Their saved images were
  blank white pages: a blank "failure" image means a navigation was in flight, i.e. the site accepted.
  Judge a failure image by whether it shows the dialog with the error message.

- **A run must survive its browser and must never hang.** `bench.run_tests` relaunches Chromium if it dies
  (`get_browser`), retries the challenge (`browser_died`, `MAX_ATTEMPTS`), caps every challenge
  (`CHALLENGE_TIMEOUT_S`) and closes contexts through `_close` (a dead or hung context must not block the
  run); the server caps a whole run (`MAX_RUN_SECONDS`). The UI polls `/api/tests/{id}` as a safety net and
  reports a vanished run (the container restarted: runs are in memory). `tests/test_live.py` (opt in with
  `CAPTCHASOLVER_LIVE=1`) kills Chromium mid-run on the real demo site. The cause of the one real-world
  `TargetClosedError` was never reproduced (72/72 parallel runs were fine at a 64 MB /dev/shm), so do not
  claim a root cause; Chromium runs with `--disable-dev-shm-usage` as a precaution only. For a container where it
  keeps dying, use `GET /api/diagnostics?stress=1` (`diag.py` reads cgroup memory/OOM, /dev/shm, /tmp; the
  endpoint runs a real Chromium self-test). `run_tests` stops after `MAX_BROWSER_RESTARTS` with
  `BrowserKeepsCrashing` rather than retrying 100 challenges. Locally, a 2 CPU / 6 GB / 64 MB-shm container
  peaked at 570 MB with 3 parallel pages, so memory alone is an unlikely cause.
  The real Cloudflare container reported `dev_shm: null`, no cgroup memory files, 2 CPUs and ~7 GB of /tmp;
  Chromium started and ran 3 pages fine there when fresh. `diag.py` therefore falls back to /proc/meminfo and
  /proc/vmstat. The image runs under `tini` (PID 1) because uvicorn does not reap Chromium's orphaned
  children (we measured 2 zombies per launch without it, 0 with it). Keep error texts short enough for the UI
  but put the actionable part FIRST: `describe_exc` truncates and once cut off the advice.

- **Never lump 401/403 into "bad credentials".** `WorkersAIError` keeps Cloudflare's status and error codes;
  `client.explain_error` separates auth (401 / code 10000), licence (403 / code 5016 'Model Agreement'),
  quota/rate limit (429 / 'daily free allocation'), forbidden (token valid, lacks Workers AI permission) and
  other, and `explain_auth_failure` asks Cloudflare's token-verify endpoints whether a rejected token is
  expired, disabled, active-but-wrong-account, or unknown. A used-up allowance and an unaccepted licence are
  not credentials problems, and the robot is told `retry` accordingly. Keep this classification when adding
  new upstream calls.

- The `files` pipeline saves, per challenge, `before` / `solver.json` / `.plan.png` / `.after.png`, and records
  the site's `outcome` in the JSON. `.after.png` goes through `_save_after`, which waits for a redirect in
  flight (otherwise it is a blank page). `--zip` bundles one run (`bundle_zip`). When adding code to a CLI,
  check WHICH `main` you are editing: `main` and `replay_main` print similar summaries and an edit once landed
  in the wrong one.

## Git hygiene

Also ignored: `cloudflare/node_modules`, `.wrangler`, `.dev.vars`. Commit `cloudflare/package-lock.json`.

`examples/` generated files (`NNN-*.png`, `*.json`, `failed_runs/`) and report JSONs are
git-ignored. `examples/1.png` is the one tracked reference screenshot. Credentials only in env vars.
