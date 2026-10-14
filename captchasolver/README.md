# captchasolver

Solves "select all images with ___" captchas using vision models on
[Cloudflare Workers AI](https://developers.cloudflare.com/workers-ai/), and turns the answer
into a click script that a robot arm (or a browser) can follow.

```
screenshot ──► find the dialog ──► read the banner ──► ask the model about each tile
                                                              │
        clicks on screen ◄── actions[] in a JSON ◄────────────┘
```

> **Scope.** This was built for our team's own demo/test captcha site (default URL in
> `src/captchasolver/web.py`) as part of a robotics project. Only run it against captchas you
> own or are explicitly authorised to test. It is not meant for third-party sites or for
> Turnstile / reCAPTCHA / hCaptcha / GeeTest widgets on pages you don't control.

## How it works

1. **Locate the dialog by colour and shape** (blue banner, square 3×3 grid, Verify button).
   All thresholds are relative, so any resolution, DPI scale or window position works.
2. **Read the target** ("cats") from the banner with a vision call (or pass `--target`).
3. **Ask a yes/no question per tile**, in parallel (`Does this image contain cats?`).
4. **Emit JSON** with the picked tiles and an ordered `actions` list (tiles first, Verify last).
   Coordinates are given in screenshot pixels *and* normalised 0–1, so an executor can
   multiply by the real screen size.

The solver only looks at pixels. It never reads the site's DOM or scripts to find answers.

## Install

Python 3.10+.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e '.[dev]'
playwright install chromium        # only needed for web/test/replay/compare
```

Credentials come from the environment (Cloudflare dashboard → *Workers AI* → API token):

```bash
export CLOUDFLARE_ACCOUNT_ID=...
export CLOUDFLARE_API_TOKEN=...
```

## Quick start

```bash
# 1. Check that the default model works on your account (also saves per-model settings)
captchasolver compare --models @cf/google/gemma-4-26b-a4b-it --preflight-only

# 2. Solve a screenshot of the challenge → JSON with click actions
captchasolver dir examples                    # every PNG in the folder
captchasolver live --debug overlay.png        # capture the screen now and solve it

# 3. Try it against the demo site
captchasolver test --count 10
```

## Commands

| Command | What it does |
|---|---|
| `captchasolver dir DIR` | Solve every PNG in `DIR`. `--layout screenshot` (default, full-page capture), `grid` (PNG is only the grid) or `tiles` (one tile per PNG). `--debug-dir` writes overlays. |
| `captchasolver live` | Capture the screen now (Wayland/X11, macOS, Windows, WSL) and solve it. |
| `captchasolver grab out.png` | Only capture the screen. Good first check on a new machine. |
| `captchasolver text img.png` | Distorted-text captcha (`--length`, `--votes`). |
| `captchasolver grid img.png --target X` | Single grid image, one call per tile. |
| `captchasolver test` | Run many challenges on the demo site and print a report (see below). |
| `captchasolver replay [DIR]` | Re-run saved solver JSONs on the site. No model calls, no cost. |
| `captchasolver compare` | Benchmark several vision models on identical challenges. |
| `captchasolver probe` | Send one raw request to a few models and print the raw responses (debugging). |
| `captchasolver web` | Quick single runs and `--snap DIR` grid captures. |
| `captchasolver locate PHOTO...` | Check that the dialog is found in photos or screenshots. Offline: no model, no credentials. |
| `captchasolver serve` | Web UI + HTTP API on localhost (`pip install -e '.[server]'`). See below. |

`web`, `probe`, `test`, `compare`, `replay` and `serve` also exist as standalone `captchasolver-<name>` commands.

### Output

```json
{
  "target": "flowers",
  "image_size": [1914, 1080],
  "grid_box": [686, 323, 517, 517],
  "tiles": [2, 4, 6],
  "click": [[1116, 409], [944, 581], [772, 753]],
  "verify": [1134, 890],
  "click_norm": [[0.58307, 0.3787], [0.49321, 0.53796], [0.40335, 0.69722]],
  "verify_norm": [0.59248, 0.82407],
  "actions": [
    {"action": "click", "role": "tile", "tile": 2, "x": 1116, "y": 409, "x_norm": 0.58307, "y_norm": 0.3787},
    {"action": "click", "role": "verify", "x": 1134, "y": 890, "x_norm": 0.59248, "y_norm": 0.82407}
  ]
}
```

Tiles are numbered row-major from 0 (top-left). Executors should use the `*_norm` values.

## Testing against the demo site

```bash
captchasolver test                              # 100 shuffled challenges, 4 in parallel
captchasolver test --count 20 --split test      # the held-out test split
captchasolver test --pipeline files             # save PNG + JSON to examples/, then act on the JSON
captchasolver replay                            # replay those JSONs for free
```

- `--seed` makes the challenge selection reproducible. `--viewport WxH` and `--dpr` test other
  screen sizes and pixel ratios.
- **Pipelines:** `screenshot` (default) takes a full-screen screenshot, solves it and clicks with
  the real mouse, i.e. what the robot does. `dom` tests only the vision model. `files` is
  `screenshot` plus saved artefacts.
- **Report:** a headline of passed / failed / errors, banner-read accuracy, click accuracy,
  per-subject results, and a **Failures** list with the reason for each (banner misread, clicks
  missed, wrong tile selection, or the error). Use `--report out.json` for the full data.
- **What a Cloudflare container looks like:** it has no `/dev/shm` and no cgroup memory files (the diagnostics
  show `dev_shm: null` and fall back to the VM's own `/proc` numbers), so Chromium runs with
  `--disable-dev-shm-usage`. The image uses `tini` as its first process so Chromium's orphaned helper
  processes are reaped instead of piling up as zombies.
- **If Chromium keeps dying** (`Target page, context or browser has been closed`), open
  `/api/diagnostics?stress=1` on the deployment. It reports the container's real memory limit and peak, OOM
  kills, `/dev/shm` and `/tmp` space, and runs Chromium the way a test run does (3 pages in parallel on the demo
  site). Each restart during a run also writes a note with those numbers into the log, and after 4 restarts the
  run stops early with the reason instead of crawling. The robot endpoint and the Solve page do not use
  Chromium, so they are unaffected.
- **A run does not hang or break because of one bad challenge.** If Chromium dies mid-run it is restarted and
  the affected challenge is retried (up to 3 attempts, with a note in the log); each challenge has a 4-minute
  limit and a whole run in the web app has a 1-hour limit (`CAPTCHASOLVER_MAX_RUN_SECONDS`). The Tests page
  polls the run, so if the solver restarts (which wipes its in-memory runs) it says "this run is gone"
  instead of waiting forever. Do not `wrangler deploy` while a run is going: a deploy replaces the container
  immediately and ends it. `GET /api/health` shows `uptime_s`, so a recent restart is visible.
- **How a result is decided:** after Verify the harness waits (up to 30 s) for the site's own answer:
  accepted (success state or the redirect to `/verified/`) or rejected ("Incorrect selection"). If the site says
  neither, the challenge is a 💥 **error** (`SiteDidNotAnswer`), never a ❌ fail, so a slow site or container
  cannot make correct answers look wrong.
- **Failure images:** every failure writes `examples/failed_runs/NNN-label.png` (the page after
  the failed Verify) and `NNN-label-plan.png` (the solver's plan drawn on its screenshot).
  Crashes write `NNN-label-error.png`. Disable with `--no-fail-images`.
- Generated files in `examples/` are git-ignored. `examples/1.png` is the reference screenshot.

## Web UI and API

A single-page UI (Solve · Tests · API · Settings) and a JSON API, served by the same FastAPI app.

```bash
pip install -e '.[server]'
captchasolver serve            # http://127.0.0.1:8000  (localhost only; API docs at /api/docs)
```

- **Solve:** drop in or paste a screenshot, get the target, the picked tiles drawn on the image
  and the click script (copy or download the JSON).
- **Tests:** the same options as `captchasolver test` (default 100 challenges, 3 pages in parallel), live progress,
  per-challenge failure reasons, a failure viewer (page state next to the solver's plan), and
  downloads (`report.json`, `failed_runs.zip`). **Nothing is stored.** Runs live in server memory
  (the last few, for 30 minutes) and failure images in a temp folder that is deleted with them.
- **Settings:** your Cloudflare account ID and API token. They stay **in your browser**
  (this tab, or this device if you tick "Remember"), travel in the `X-CF-Account-Id` and
  `X-CF-Api-Token` headers, and are never stored or logged server-side. A running test holds them in
  memory only until it ends.

### API

| Endpoint | |
|---|---|
| `POST /api/robot/solve` | **the robot's endpoint**: photo in, ordered `presses` out (see "Robot flow"); never rate limited |
| `POST /api/solve` | multipart `image` (+ `layout` `screenshot`\|`grid`, `target`, `grid`, `model`) → target, tiles, `actions` |
| `POST /api/preflight` | check that a model can read an image with your credentials |
| `GET /api/models`, `GET /api/health` | candidate models; limits and allowed test hosts |
| `POST /api/tests` | start a run → `{id, events, state}` |
| `GET /api/tests/{id}/events` | server-sent events: `start`, `note`, `record`, `summary`, `end` (resumable with `Last-Event-ID`) |
| `GET /api/tests/{id}` · `POST .../cancel` | state and records · stop, keeping partial results (the long random id is the access key; runs cannot be listed) |
| `GET /api/tests/{id}/report.json` · `failed_runs.zip` · `images/{name}` | downloads |

```bash
curl -X POST http://127.0.0.1:8000/api/solve \
  -H "X-CF-Account-Id: $CLOUDFLARE_ACCOUNT_ID" -H "X-CF-Api-Token: $CLOUDFLARE_API_TOKEN" \
  -F image=@screenshot.png
```

Guard rails: uploads are capped (15 MB, 60 MP), account ID / model / target are validated (they end
up in URLs and prompts), and **the test runner only visits hosts in
`CAPTCHASOLVER_ALLOWED_HOSTS`** (default: the demo site), so the server cannot be used to browse
or probe other sites. Only one run at a time (`CAPTCHASOLVER_MAX_RUNS`). Expensive endpoints are rate limited per visitor.

## Deploy to Cloudflare

`cloudflare/` is a Worker that fronts the app, which runs as a
[Cloudflare Container](https://developers.cloudflare.com/containers/) (the `Dockerfile`: FastAPI +
headless Chromium, 1.4 GB). Needs a Workers plan with Containers, Docker **with `buildx`** on the
machine that deploys (Arch: `sudo pacman -S docker-buildx`), and Node 20+.

```bash
cd cloudflare
npm install
npx wrangler login
npx wrangler deploy            # builds the image, pushes it, deploys the Worker (takes a few minutes)
```

You get a `https://captchasolver.<your-subdomain>.workers.dev` URL. Open it, go to **Settings** and
enter your Cloudflare account ID and an API token with Workers AI access.

**Who can use it** is set by `AUTH_MODE` in `cloudflare/wrangler.jsonc`:

- `"public"` (the default here): anyone with the link can open the app, but it only does anything with
  *their own* Cloudflare credentials, which they type into Settings. Because it is open, the server
  protects itself: per-visitor rate limits (solve 30/min, preflight 20/min, test runs 6/hour; tune with
  `CAPTCHASOLVER_RATE_SOLVE|PREFLIGHT|TESTS`; the committed config sets `RATE_LIMITS: "off"` for the live demo), a quick check that the credentials work before a test
  run launches Chromium, one run at a time, the test runner restricted to `ALLOWED_TEST_HOSTS`, and no
  way to list other people's runs.
- `"access"`: only people holding a valid Cloudflare Access token. Create a self-hosted Access
  application for the Worker's hostname, then put its **AUD tag** and your **team domain**
  (`<team>.cloudflareaccess.com`) in `ACCESS_AUD` / `ACCESS_TEAM_DOMAIN` and deploy again. With
  `"access"` and those values empty, every request gets a 503 (it fails closed). Scripts use an Access
  *service token* (`CF-Access-Client-Id` / `CF-Access-Client-Secret`).

Things to know:
- Only enter a token that is limited to Workers AI and that you can revoke. In public mode, other
  people use the same URL, so treat it like any shared site.
- One container instance serves everyone and sleeps after 30 idle minutes. The first request after that
  takes a few seconds, and results held in memory are gone (by design: nothing is stored).
- Set `ALLOWED_TEST_HOSTS` (comma-separated) in `wrangler.jsonc` if the demo site moves.
- `npx wrangler dev` runs it locally (needs Docker). Copy `.dev.vars.example` to `.dev.vars` to skip
  login locally. **Never deploy with `ALLOW_UNAUTHENTICATED_DEV=true`.**
- Worker tests: `npm test` and `npm run typecheck`.

## Robot flow: camera photo in, press positions out

```
camera photo ─► (you) straighten to the screen's size ─► POST /api/robot/solve ─► presses[] ─► robot presses them
```

Your side straightens the photo (finds the screen's corners and warps it to the screen's width and height).
`POST /api/robot/solve` takes that image and returns the positions to press **in screen pixels**: the
tiles first, then Verify. It is not rate limited.

```bash
curl -X POST "$URL/api/robot/solve" \
  -H "X-CF-Account-Id: $CLOUDFLARE_ACCOUNT_ID" -H "X-CF-Api-Token: $CLOUDFLARE_API_TOKEN" \
  -F image=@photo.jpg -F screen_width=1920 -F screen_height=1080
```
```json
{ "ok": true, "target": "flowers", "tiles": [2, 4, 6], "screen": {"width": 1920, "height": 1080},
  "tilt_deg": 0.0, "model": "@cf/google/gemma-4-26b-a4b-it", "seconds": 2.1,
  "presses": [
    {"seq": 1, "role": "tile",   "tile": 2, "x": 1120, "y": 409, "x_norm": 0.58307, "y_norm": 0.3787},
    {"seq": 2, "role": "tile",   "tile": 4, "x": 947,  "y": 581, "x_norm": 0.49321, "y_norm": 0.53796},
    {"seq": 3, "role": "tile",   "tile": 6, "x": 775,  "y": 753, "x_norm": 0.40335, "y_norm": 0.69722},
    {"seq": 4, "role": "verify", "x": 1137, "y": 890, "x_norm": 0.59248, "y_norm": 0.82407} ] }
```

- `screen_width` / `screen_height` are optional: positions are `x_norm × screen size`, defaulting to the
  image's own size. Press in `seq` order.
- **Every failure has one shape**, `{"ok": false, "error": <code>, "detail": "...", "retry": true|false}`.
  `retry: true` means a new photo may well succeed (move on to the next frame); `false` means fix the call.

| `error` | Meaning | `retry` |
|---|---|---|
| `no_dialog` | no captcha dialog in the photo | yes |
| `layout_inconsistent` | something blue was found but it does not fit the dialog's geometry (arm over the button, glare, a blue object, tilt over 10°) | yes |
| `cropped` | the dialog runs off the edge of the photo | yes |
| `unreadable_target` | the banner text could not be read | yes |
| `no_matches` | no tile matched; Verify is deliberately **not** pressed blindly | yes |
| `model_error` | Cloudflare/the model failed or timed out | yes |
| `rate_limited` | Cloudflare is rate limiting you, or the Workers AI allowance is used up | yes, after a wait |
| `model_not_available` | the model needs its licence accepted on your account | no |
| `bad_credentials` | missing credentials, **or Cloudflare rejected the token** (the `detail` says whether it is expired, revoked, unknown, or lacks the Workers AI permission) | no |
| `bad_request`, `bad_image` | malformed input | no |

### How it copes with a photo of a screen

The dialog finder adapts to the photo (it does not look for an exact colour), removes up to 10° of tilt
(positions are mapped back to your photo), and **refuses instead of guessing** whenever the geometry does
not add up: banner and Verify must sit where the dialog's proportions say. Measured on simulated camera
conditions applied to the reference screenshot, judged by what the robot presses (Verify and the nine tile
centres): tints, dimming, low contrast, strong blur, noise, JPEG, vignetting, side lighting, up to 9° of
tilt and up to 6% perspective skew all stayed within about 1 px on Verify and 18 px on tiles; a tile is
about 170 px wide and Verify about 57 px tall. Across 300 random mixtures the result was 298 correct,
2 refused, **0 confidently wrong**. An arm over Verify or the banner, or big blue objects nearby, are refused.

That was **simulated**: it does not cover moiré, screen reflections, lens barrel distortion or strongly
off-axis shots. Before trusting a setup, check real photos for free (no model, no credentials):

```bash
captchasolver locate photos/*.jpg --overlay-dir overlays   # ✓/✗ per photo, grid + Verify drawn on it
```

Keep `examples/` photos you tried; `tests/test_camera.py` shows how to turn a failure into a regression test.

### Rate limits

The default limits (solve 30/min, preflight 20/min, test runs 6/hour per visitor) protect a shared
deployment. `/api/robot/solve` is exempt. To turn **all** limits off, set `RATE_LIMITS` to `"off"` in
`cloudflare/wrangler.jsonc` (that is the committed setting, for the live demo) or run with
`CAPTCHASOLVER_RATE_LIMITS=off`. Without limits anyone with the URL can use your container's CPU
(they still need their own Cloudflare key), so switch them back on if the link spreads.

## Models

Default: `@cf/google/gemma-4-26b-a4b-it`. Change it per run with `--model`, or permanently via
`VLM_MODEL` in `src/captchasolver/client.py`.

`captchasolver compare` preflights each candidate on *your* account (reads a test image,
trying request variants such as thinking off / no temperature), skips the ones that don't answer,
then runs the survivors on the same seeded challenges and ranks them. What preflight learns is
cached in `~/.cache/captchasolver/model_config.json` (override with `CAPTCHASOLVER_CONFIG`).

## Platform notes

- **Wayland:** capture uses `grim` (falls back to `gnome-screenshot` / `spectacle`).
- **WSL:** capture asks `powershell.exe` for the *Windows* screen, DPI-aware, primary display.
  Run `captchasolver grab shot.png` first and check the size.
- **macOS / Windows / X11:** Pillow `ImageGrab` / `screencapture`.

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `result: {}` from a model | The model returns nothing on this account (seen with Moondream). Use another model. |
| `ReadTimeout` / `no response after 3 attempts` | Slow model under load. Lower `--concurrency` (try 2). |
| `403 ... Model Agreement` | Llama 3.2 Vision needs a one-time licence acceptance in your account. |
| `LayoutNotFound` | No dialog in the screenshot, or the grid runs off screen. Use a taller viewport (≥ 900 px). |
| "Credentials rejected" | Settings → Test connection now says which: token expired / revoked / not recognised, valid but not for this account or lacking the Workers AI permission, a model licence not accepted, or the Workers AI allowance used up (that one is not a credentials problem). Credentials are kept per browser tab unless you tick "Remember", so a new tab or window has none. To test outside the app: `captchasolver-probe`. |
| UI says `502 ... did not answer a test image` | The model failed its first check with your credentials. Use Settings → Test connection; common causes are a wrong token, no Workers AI permission, or an unaccepted model licence. |
| Reasoning model returns an empty answer | Run `compare --preflight-only` so the thinking-off setting is discovered and cached. |

## Development

```bash
pytest -q        # all offline: no network, no credentials, no browser needed for most tests
cd cloudflare && npm test      # Worker: public/Access modes, JWT verification, header forwarding
```

```
src/captchasolver/
  client.py    Workers AI REST client, vlm_query(), per-model request config, response parsing
  grid.py      tile logic: boxes→tiles, per-tile yes/no, classify, centres
  text.py      distorted-text captchas
  screen.py    find the dialog in a screenshot, read the banner, build the actions JSON
  capture.py   screen capture per platform (incl. WSL)
  batch.py     `dir`: folders of PNGs
  web.py       Playwright driver for the demo site's image challenge
  bench.py     `test` / `replay`: runs, report formatting, failure images
  compare.py   `compare`: preflight + model ranking
  probe.py     raw model probe
  cli.py       entry point that ties the commands together
  server/      FastAPI app (api.py), in-memory test runs (runs.py), the UI (static/index.html)
cloudflare/    Worker + container config that serves the UI/API (public or behind Cloudflare Access)
Dockerfile     image for the web app (FastAPI + Chromium)
tests/         unit tests (fake model + fake pages)
examples/      1.png reference screenshot; generated data is git-ignored
```

## Limitations

- Image-grid challenges only. No slider, checkbox/Turnstile or GeeTest solving.
- Expects the demo site's layout (blue banner, square grid, Verify at the bottom right); the proportions
  are constants in `screen.py`. Other designs need `find_layout` changes.
- The photo must already be straightened to the screen's size (tilt up to 10° and some perspective skew
  are tolerated; beyond that it refuses).
- 3×3 grids by default (`--grid RxC` for others); assumes roughly equal tiles.
- Only the primary monitor is captured.
