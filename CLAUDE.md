# Robotics hackathon: session handoff

Context for continuing this project in Claude Code inside WSL. Written 2026-10-14 ~17:00 (Vienna).
Everything below was set up in an earlier chat session; items marked **verify** were not confirmed on the real hardware.
Corrected 2026-10-15: the screen is read from a screenshot (not a camera), the force guard is off by default, the learning loop has run on
the real arm (numbers below), and the captcha loop is built and calls the solver API.

## Goal

An SO-101 robot arm solves (self-made) CAPTCHAs on a touch screen:
a screenshot of the screen (`CAMERA=screen`; a camera also works) → the solver API picks answers → arm taps them with a stylus.
Team of 5; roles and the interface rules are in `PLAN.md` (CAPTCHA page, robot, agent, integration, test/demo).
Our CAPTCHAs are made by a teammate. Never retry taps on a real CAPTCHA (a second tap toggles a tile back).

## Machine and environment

- Windows on ARM laptop, **WSL2 Ubuntu 24.04**, user `mehras`, Windows user `mehra`.
- Repo: `~/robotics_hackathon` (GitHub `ROBOTICS-HACKATHON/robotics_hackathon`). Old non-Docker setup: `.venv/` + `lerobot/` clone (both git-ignored; keep until Docker is confirmed working, then they can be deleted).
- Everything runs through Docker via `./lr` (image `lerobot`, built with `./lr build`; CPU flavour on the laptop). Code under `touch/` is mounted live (`/repo`), so no rebuild is needed after editing it.
- User is in `dialout` and `docker` groups. Hugging Face user: `mehrasmeydani`.
- USB devices reach WSL only via **usbipd** (run in Windows PowerShell as admin: `usbipd list`, `usbipd bind --busid X`, `usbipd attach --wsl --busid X`). WSL cannot see Windows webcams; phone cameras go over Wi-Fi.
- **Settings:** `lr.conf` (WSL). A one-line hook makes it also read `/mnt/c/Users/mehra/Downloads/robot-settings/lr.local.conf` (Windows side, overrides `lr.conf`) — **verify the hook line was added** (`tail -1 lr.conf`).

## Hardware state (last known)

- Arms: SO-101 follower `/dev/ttyACM0` (id `my_follower`), leader `/dev/ttyACM1` (id `my_leader`), both calibrated (LeRobot calibration in `~/.cache/huggingface/lerobot/calibration`). Previously usbipd BUSIDs 3-1 / 3-2 (COM14 / COM15). **Arms were unplugged at handoff; BUSIDs may have changed.** Attach arms first (follower first) so ACM numbering stays; update `USB_BUSIDS` in `lr.conf`.
- The demo reads the screen from a screenshot (`CAMERA=screen` + `touch/screencap.ps1` running in Windows PowerShell), so no camera is needed. Camera notes, if you use one: **RealSense D435i, BUSID 3-1** (the old D455 was returned). Colour stream probably `/dev/video4` — **verify** with `./lr cams` (pick the normal colour image, not depth/IR-dots). Alternative: Samsung S24+ with IP Webcam app (`CAMERA=ipwebcam:<phone-ip>`).
- Touch screen: slightly angled (not 90°, flat was rejected for safety). Same screen as the terminal, hence `DELAY=5` countdowns.
- Training server (separate machine, SSH): Bazzite Linux, AMD RX 6800 XT, Podman, ROCm image works (torch 2.11+rocm7.2). Earlier image lacked the `training` extra; the repo `docker/Dockerfile` (`./lr build` → auto-detects ROCm) fixes that. Training of an ACT policy on a pen-in-cup dataset was started but never confirmed — low priority now.

## `./lr` commands (see `./lr help`, `HOWTO.md`, `touch/README.md`)

Setup: `build`, `login`, `attach`, `ports`, `cams`, `cam-test [CAMERA]`, `snap`, `calibrate follower|leader`, `teleop`.
Captcha loop: `solve` (screenshot → solver, prints the plan, never taps), `plan` (one image → JSON of what to click next), `captcha` (the full loop; `--learn-layout`, `--dry-run`).
Other: `heatmap`, `tip-test`, `tip-live`, `teach-dirs` (writes `directions.json`, which `tap.py` does not read), `smooth-preview`, `share`.
Recording/training (original LeRobot flow): `test`, `record`, `train`, `logs`, `run`.
Touch screen:
- `touch-calib` — teach the screen: serves `touch/calib.html` at `http://localhost:8765/`. Set home (Enter in terminal, or Enter/Space on the page), then the **wrist locks**; touch each highlighted target, lift back home, hold still. Records full leader paths. The force guard is off by default (`--max-load 0`; `lr` never passes it).
- `set-home` — re-teach only the home pose.
- `tap u v ...` / `tap --test` / `tap --random N` — tap screen points (u,v in 0..1). Tests are measured via the page (touches reported back), logged to `touch/logs/`, and feed the learning.
- `practice --targets 40 --tol 15` — self-teaching loop: retry each target until within tol px, learns from every touch.
- `see` — screenshot (`CAMERA=screen`) or camera → the page, cut out at the 4 markers and straightened to `outputs/screen.jpg` (pixel x,y → u=x/1280, v=y/800).
- `fit [--reset]` — screen-space aim correction (fallback before practice has data).

## How the touch system works (files in `touch/`)

- `common.py` — robot I/O, `ScreenModel` (quadratic per-joint fit (u,v)→joints; recorded leader paths smoothed (Gaussian, trim pauses) and blended), `follow_path`, `settle`, `ForceGuard` (motor `Present_Load` jump → stop pushing), `set_torque_limit`, `countdown`.
- `calibrate_screen.py` — calibration (records commanded poses, not sagging follower positions; wrist lock; force guard; writes `screen_calib.json` + `logs/calib_*.json`).
- `tap.py` — `Tapper`: home → smoothed own path down (slow final 25%) → small press → settle → lift from actual pose → home. Stops on: page-reported touch (`stop_on_touch`), force guard. Uses learned model if `experience.json` has ≥5 touches; otherwise `correction.py`.
- `learner.py` — `Experience` (`experience.json`, per calibration id), `LearnedModel` (locally weighted regression in joint space from calibration + all touches), `ZoneBandit` (per-zone epsilon-greedy choice of press depth × approach speed from rewards; `PressBandit` is the older single-zone version), `zone_report` / `draw_heatmap`, `refine()` (per-target closed loop using the model Jacobian).
- `practice.py` — runs `refine` over random targets inside the calibrated area, prints a learning curve.
- `correction.py` — screen-space shift/affine/quadratic correction chosen by leave-one-out; `calib_id()` fingerprint.
- `touchserver.py` — tiny HTTP server: serves page, `/state`, `/touch` (u,v,w,h), `/key`; `touched` Event.
- `screencap.ps1` — runs in Windows PowerShell (you start it), saves a screenshot of the whole screen ~4×/s to `outputs/screen_live.jpg`; `lr` reads it when `CAMERA=screen` and refuses if it is older than 3 s.
- `see.py` — any source: the screenshot file above, or a camera (`/dev/videoN`, index, `ipwebcam:IP`, `droidcam:IP`, http snapshot/stream, rtsp) or an image file; ArUco 4x4_50 markers 0–3 at the corners are found wherever the page is in the image; homography warp to 1280×800; remembers last screen position if a marker is hidden (`outputs/last_screen.json`); `--check`, `--delay`.
- `calib.html` + `markers.js` (markers inline; any CAPTCHA page must include `markers.js`). `set_home.py`.
- Captcha loop: `run_captcha.py` (the loop), `solve.py` (solver client), `pagestate.py` (which page is on screen, pixels only), `verify_selection.py` (which tiles are ticked), `learn_layout.py` (teach the choice/checkbox/verified pages), `plan_clicks.py` (one image → what to click next).
- Tip vision (optional, `VISION=off` by default): `tipvision.py`, `servo.py`. `smooth_preview.py` simulates tap smoothness. `teach_dirs.py` records directions that `tap.py` does not use.
- **Learning loop on the real arm (15 Oct, 100 targets, tolerance 15 px, no tip cameras; `media/training_logs.json`):** 70/100 targets reached 15 px, 343 attempts. First-try miss averaged 40 px (36 px over the first 50 targets, 44 px over the last 50), so first tries did NOT improve; the retry loop is what gets the hits. (An earlier simulation predicted ~19–28 px → ~9–13 px; the real arm did not match it.)
  The force guard was tried on the arm and turned off by default because it misread normal arm movement as pushing. Phone-camera warp and the occlusion fallback are still only simulation-tested.

## What the real logs showed (before the latest fixes)

- Run 1 (old calib): mean miss 88 px, mostly up/down. Cause: wrist angle differed a lot between calibration targets (−15°…+32°), so blended poses missed → wrist lock added.
- Run 2 (calib at 15:36, still **without** wrist lock, wrist spread 57°): test 65 px; random 66 px with 8/20 no-touch (mostly mid-screen → arm lifted before arriving → slow end + settle added); with screen-space correction 40 px, 12/20 within 50 px.
- Screen frame ~1960×1296 px in the page.

## Next steps (in order)

1. Screen source: start `touch/screencap.ps1` in Windows PowerShell, set `CAMERA=screen` in `lr.conf`, run `./lr see` (show the page full-screen during the countdown; `outputs/screen.jpg` should be just the page).
   With a camera instead: `usbipd attach --wsl --busid 3-1` (Windows), `./lr cams`, set `CAMERA=/dev/videoN`, `./lr cam-test`.
3. Plug arms back in, find their BUSIDs, attach (follower first), check `ls /dev/ttyACM*`, update `USB_BUSIDS`.
4. `./lr touch-calib` (must print "WRIST LOCKED"), then `./lr practice --targets 40 --tol 15` ×2–3, then `./lr tap --random 20`.
5. Analyse `touch/logs/*.json` + `touch/experience.json`: first-try miss trend, no-touch count, `stopped_by`, `peak_load_jump` (tune `--max-load` if the force guard trips on normal taps or never trips), `settle_err_deg`.
6. Integration (Phase 3 in `PLAN.md`) is built: `./lr captcha` (see "Full captcha loop" below). Open: re-teach the pages (`./lr captcha --learn-layout`) after any change to the site's look, and measure a real end-to-end pass rate (`outputs/run/results.jsonl`). Stop-on-touch does not work in the captcha loop (the site does not report touches).

## Gotchas

- `usbipd` re-attach needed after replugging or `wsl --shutdown`; attach order decides ttyACM numbers.
- `/dev/video*` numbers change after replug; re-run `./lr cams`.
- Only one program can use the camera at a time (close `ffplay` live view before `see`).
- Port 8765 must be free (one `touch-calib`/`tap --test`/`practice` at a time).
- Windows-edited files may have CRLF line endings; the override hook strips `\r`.
- Don't commit `touch/screen_calib.json`, `touch/experience.json`, `touch/logs/`, `touch/screen_correction.json`, `outputs/` (machine-specific; already in `.gitignore`).
- Git pushes often get rejected because teammates pushed: `git pull --rebase origin main && git push`.

## Full captcha loop (added later)

`./lr captcha` (`touch/run_captcha.py`) runs: see → page state → solver API (`/api/robot/solve`, `touch/solve.py`) → tap tiles →
read the selection (`touch/verify_selection.py`) → fix only differing tiles → Verify once → read pass/fail (`touch/pagestate.py`).
Needs `CLOUDFLARE_ACCOUNT_ID` / `CLOUDFLARE_API_TOKEN` in `lr.conf`, "Camera markers" ticked on the site, and
`./lr captcha --learn-layout` once (writes the git-ignored `touch/site_layout.json`). Docs: "The whole loop on the real site" in `touch/README.md`.
Offline tests: `.venv/bin/python -m pytest touch/tests -q` (screenshots in `touch/tests/shots/`). **Those screenshots show an earlier design of the site** (no header bar,
page called "Access website"); the current `/training/` page sits ~78 px lower, so the page classifier returns "unknown" on it and the opt-in live test
(`SITE_E2E=1`, `SITE_URL=https://robot-captcha-demo.pages.dev/training/`) fails at the first page until the shots are re-captured (`touch/tests/capture_site.py`).
The loop calls the real solver API; `--dry-run` does too (it only skips the arm). Careful order on real hardware: `./lr solve`, `./lr captcha --dry-run`, then `./lr captcha`.
