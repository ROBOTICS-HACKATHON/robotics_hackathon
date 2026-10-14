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

`web`, `probe`, `test`, `compare` and `replay` also exist as standalone `captchasolver-<name>` commands.

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
- **Failure images:** every failure writes `examples/failed_runs/NNN-label.png` (the page after
  the failed Verify) and `NNN-label-plan.png` (the solver's plan drawn on its screenshot).
  Crashes write `NNN-label-error.png`. Disable with `--no-fail-images`.
- Generated files in `examples/` are git-ignored. `examples/1.png` is the reference screenshot.

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
| Reasoning model returns an empty answer | Run `compare --preflight-only` so the thinking-off setting is discovered and cached. |

## Development

```bash
pytest -q        # all offline: no network, no credentials, no browser needed for most tests
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
tests/         unit tests (fake model + fake pages)
examples/      1.png reference screenshot; generated data is git-ignored
```

## Limitations

- Image-grid challenges only. No slider, checkbox/Turnstile or GeeTest solving.
- Expects the demo site's layout (blue banner, square grid). Other designs need `find_layout`
  changes.
- 3×3 grids by default (`--grid RxC` for others); assumes roughly equal tiles.
- Only the primary monitor is captured.
