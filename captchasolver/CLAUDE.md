# captchasolver

Python tool that solves image-grid captchas ("select all images with X") with vision models on
Cloudflare Workers AI and emits a click script (JSON `actions`) for a robot arm or Playwright.
Part of a robotics hackathon repo; this directory is self-contained and the rest of the repo
(LeRobot setup, `lr`, Docker) is unrelated.

## Scope: do not extend beyond this

- Only the team's own demo site (default URL in `web.py`) and test captchas. Never point this at a
  third-party site or at real reCAPTCHA/Turnstile/hCaptcha/GeeTest widgets.
- The solver must work from pixels only. The demo site ships its answer key in client-side JS;
  never read or use it for solving. Failure reasons are derived from observations, not the key.

## Commands

```bash
pip install -e '.[dev]' && playwright install chromium
pytest -q                              # offline, no credentials needed
captchasolver test --count 10          # needs CLOUDFLARE_ACCOUNT_ID / CLOUDFLARE_API_TOKEN
captchasolver compare --preflight-only # check models on the account; caches settings
```

No linter is configured. Match the existing style (type hints, short docstrings, `from __future__ import annotations`).

## Architecture

`screen.find_layout` (colour/shape, all thresholds relative) → `screen.read_target` (banner via
`vlm_query`) → `grid.solve_grid` / `solve_tiles` (parallel per-tile yes/no) →
`screen.build_actions` (tiles first, Verify last). `bench.py` drives the demo site with
Playwright (`test`, `replay`); `compare.py` ranks models; `capture.py` grabs the screen per platform.

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
- The site's Challenge dropdown values are `image:<n>` (they were `0-9` once). The harness reads
  them from the page, never hard-code the list. A session cannot skip a failed challenge, so
  `test` picks challenges itself instead of using the site's Start session.
- Resampled screenshots blur the banner's edge. Find the grid by waiting for the white gap under
  the banner, not by the first non-white row.
- WSL capture must go through `powershell.exe` (Windows screen, DPI-aware), not WSLg.

## Git hygiene

`examples/` generated files (`NNN-*.png`, `*.json`, `failed_runs/`) and report JSONs are
git-ignored. `examples/1.png` is the one tracked reference screenshot. Credentials only in env vars.
