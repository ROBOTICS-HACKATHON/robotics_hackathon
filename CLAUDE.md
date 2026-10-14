# Robotics hackathon: session handoff

Context for continuing this project in Claude Code inside WSL. Written 2026-10-14 ~17:00 (Vienna).
Everything below was set up in an earlier chat session; items marked **verify** were not confirmed on the real hardware.

## Goal

An SO-101 robot arm solves (self-made) CAPTCHAs on a touch screen:
camera sees the screen → agent picks answers → arm taps them with a stylus.
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
- Camera now: **RealSense D435i, BUSID 3-1** (the old D455 was returned). Colour stream probably `/dev/video4` — **verify** with `./lr cams` (pick the normal colour image, not depth/IR-dots). Alternative: Samsung S24+ with IP Webcam app (`CAMERA=ipwebcam:<phone-ip>`).
- Touch screen: slightly angled (not 90°, flat was rejected for safety). Same screen as the terminal, hence `DELAY=5` countdowns.
- Training server (separate machine, SSH): Bazzite Linux, AMD RX 6800 XT, Podman, ROCm image works (torch 2.11+rocm7.2). Earlier image lacked the `training` extra; the repo `docker/Dockerfile` (`./lr build` → auto-detects ROCm) fixes that. Training of an ACT policy on a pen-in-cup dataset was started but never confirmed — low priority now.

## `./lr` commands (see `./lr help`, `HOWTO.md`, `touch/README.md`)

Setup: `build`, `login`, `attach`, `ports`, `cams`, `cam-test [CAMERA]`, `snap`, `calibrate follower|leader`, `teleop`.
Recording/training (original LeRobot flow): `test`, `record`, `train`, `logs`, `run`.
Touch screen:
- `touch-calib` — teach the screen: serves `touch/calib.html` at `http://localhost:8765/`. Set home (Enter in terminal, or Enter/Space on the page), then the **wrist locks**; touch each highlighted target, lift back home, hold still. Records full leader paths. Force guard active.
- `set-home` — re-teach only the home pose.
- `tap u v ...` / `tap --test` / `tap --random N` — tap screen points (u,v in 0..1). Tests are measured via the page (touches reported back), logged to `touch/logs/`, and feed the learning.
- `practice --targets 40 --tol 15` — self-teaching loop: retry each target until within tol px, learns from every touch.
- `see` — camera → straightened screenshot `outputs/screen.jpg` (pixel x,y → u=x/1280, v=y/800).
- `fit [--reset]` — screen-space aim correction (fallback before practice has data).

## How the touch system works (files in `touch/`)

- `common.py` — robot I/O, `ScreenModel` (quadratic per-joint fit (u,v)→joints; recorded leader paths smoothed (Gaussian, trim pauses) and blended), `follow_path`, `settle`, `ForceGuard` (motor `Present_Load` jump → stop pushing), `set_torque_limit`, `countdown`.
- `calibrate_screen.py` — calibration (records commanded poses, not sagging follower positions; wrist lock; force guard; writes `screen_calib.json` + `logs/calib_*.json`).
- `tap.py` — `Tapper`: home → smoothed own path down (slow final 25%) → small press → settle → lift from actual pose → home. Stops on: page-reported touch (`stop_on_touch`), force guard. Uses learned model if `experience.json` has ≥5 touches; otherwise `correction.py`.
- `learner.py` — `Experience` (`experience.json`, per calibration id), `LearnedModel` (locally weighted regression in joint space from calibration + all touches), `PressBandit` (epsilon-greedy press depth from rewards), `refine()` (per-target closed loop using the model Jacobian).
- `practice.py` — runs `refine` over random targets inside the calibrated area, prints a learning curve.
- `correction.py` — screen-space shift/affine/quadratic correction chosen by leave-one-out; `calib_id()` fingerprint.
- `touchserver.py` — tiny HTTP server: serves page, `/state`, `/touch` (u,v,w,h), `/key`; `touched` Event.
- `see.py` — any camera source (`/dev/videoN`, index, `ipwebcam:IP`, `droidcam:IP`, http snapshot/stream, rtsp, image file); ArUco 4x4_50 markers 0–3 at the corners; homography warp; remembers last screen position if a marker is hidden (`outputs/last_screen.json`); `--check`, `--delay`.
- `calib.html` + `markers.js` (markers inline; any CAPTCHA page must include `markers.js`). `set_home.py`.
- Simulation-tested only: learning loop (first-try miss ~19–28 px → ~9–13 px over 40 targets), force guard, stop-on-touch, phone-camera warp (0.12% error), occlusion fallback. **Real-hardware behaviour of all of these is unverified.**

## What the real logs showed (before the latest fixes)

- Run 1 (old calib): mean miss 88 px, mostly up/down. Cause: wrist angle differed a lot between calibration targets (−15°…+32°), so blended poses missed → wrist lock added.
- Run 2 (calib at 15:36, still **without** wrist lock, wrist spread 57°): test 65 px; random 66 px with 8/20 no-touch (mostly mid-screen → arm lifted before arriving → slow end + settle added); with screen-space correction 40 px, 12/20 within 50 px.
- Screen frame ~1960×1296 px in the page.

## Next steps (in order)

1. Make sure the latest code is in the repo (`grep -c LOCKED touch/calibrate_screen.py` > 0, `ls touch/practice.py touch/learner.py`). If missing: `unzip -o /mnt/c/Users/mehra/Downloads/calib-fix.zip` then commit.
2. Camera: `usbipd attach --wsl --busid 3-1` (Windows), `./lr cams`, set `CAMERA=/dev/videoN`, `./lr cam-test`, `./lr see` (show `touch/calib.html` full-screen during the countdown).
3. Plug arms back in, find their BUSIDs, attach (follower first), check `ls /dev/ttyACM*`, update `USB_BUSIDS`.
4. `./lr touch-calib` (must print "WRIST LOCKED"), then `./lr practice --targets 40 --tol 15` ×2–3, then `./lr tap --random 20`.
5. Analyse `touch/logs/*.json` + `touch/experience.json`: first-try miss trend, no-touch count, `stopped_by`, `peak_load_jump` (tune `--max-load` if the force guard trips on normal taps or never trips), `settle_err_deg`.
6. Integration (Phase 3 in `PLAN.md`): loop `see` → agent (vision model returns `{"taps": [[u,v],...]}`, last = Verify) → `Tapper.tap_many` → `see` to read ✓/✗. Teammate's CAPTCHA page must include `markers.js`; optionally report touches like `calib.html` does so stop-on-touch works in the demo too.

## Gotchas

- `usbipd` re-attach needed after replugging or `wsl --shutdown`; attach order decides ttyACM numbers.
- `/dev/video*` numbers change after replug; re-run `./lr cams`.
- Only one program can use the camera at a time (close `ffplay` live view before `see`).
- Port 8765 must be free (one `touch-calib`/`tap --test`/`practice` at a time).
- Windows-edited files may have CRLF line endings; the override hook strips `\r`.
- Don't commit `touch/screen_calib.json`, `touch/experience.json`, `touch/logs/`, `touch/screen_correction.json`, `outputs/` (machine-specific; already in `.gitignore`).
- Git pushes often get rejected because teammates pushed: `git pull --rebase origin main && git push`.
