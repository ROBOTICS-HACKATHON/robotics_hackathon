# Touch screen: arm taps CAPTCHA answers

```
camera ──► ./lr see ──► straightened screenshot ──► agent picks tiles (u, v) ──► ./lr tap u v ...
```

Positions on the screen are always **u, v from 0 to 1**: u goes left to right, v goes top to bottom. They're measured between the outer corners of the 4 black markers.

## Setup (once, and again if the arm or screen moves)

1. **Stylus:** fix a **capacitive stylus** in the gripper (rubber or mesh tip; tape it so it can't slip). A plastic gripper tip won't register on a touch screen.
2. **Fix everything down:** tape the arm base and the screen to the table. If either one moves, redo step 4.
3. **Camera (any device):** it only has to see the whole screen with all 4 corner markers, no glare.
   Position, angle, resolution don't matter and nothing is calibrated to it, so you can swap cameras or
   share one with another team. With a **phone**: install *IP Webcam* (Android) or *DroidCam*
   (Android/iOS), put the phone on the same Wi-Fi, start the camera in the app, note the address it shows,
   then in `lr.conf` set e.g. `CAMERA=ipwebcam:192.168.1.23` (or `droidcam:...`, or any snapshot/stream URL)
   and run `./lr cam-test`. If the arm hides a marker, `see` reuses the last position it saw.
4. **Teach the screen:** run
   ```bash
   ./lr touch-calib
   ```
   On the touch screen, open the address it prints (`http://localhost:8765/` when the touch screen is plugged into this computer) full-screen (F11).
   - Hold the stylus a few cm **above** the middle of the screen with the leader arm (home), wrist angled so the
     stylus points down at the screen, press Enter. The **wrist roll and gripper are then held** at that pose
     (`WRIST_LOCK=roll`, default); wrist flex stays free, so try to keep the stylus at about the same angle for
     every target. `WRIST_LOCK=wrist` also holds wrist flex (most consistent), `none` holds nothing.
   - For each **highlighted target**: move down and touch it, then lift straight up back to home and **hold still** for a moment. The next target lights up by itself.
   - Your whole movement is recorded. Taps later replay a **smoothed** version of your own path, so move the way you want the robot to move: up and over, never sliding on the glass.
   - Touches outside the highlighted target are ignored.
5. **Check and measure:** keep the page open, then run
   ```bash
   ./lr tap --test         # taps the 9 calibration targets
   ./lr tap --random 20    # taps 20 random spots: the honest accuracy test
   ```
   The orange ring shows where the robot aims and blue dots show where it touched. At the end you get the
   mean and worst error in pixels, how many taps landed within 25 and 50 px, and the average miss direction.
   If most of the error is in one direction, the misses are systematic: recalibrate. If it's scattered, it's noise.

**Practice (the robot teaches itself):**

```bash
./lr practice --targets 40 --tol 15    # ~5-10 min; run it a few times
```

For each target it taps, sees where it landed, corrects its **joint pose** and retries until it is within
`--tol` pixels, then moves on. Every touch becomes experience (`touch/experience.json`, per calibration):
the arm's model of "which joint angles hit (u, v)" is refit from all of it, with nearby examples weighted most,
so **first tries keep getting better**. Each attempt also gets a reward (accuracy, minus penalties for no
touch, double touches or force stops) that tunes how deep it presses. The report shows the learning curve.
With 2 cameras (`VISION=auto`), a tap that doesn't register is measured too: the cameras see how far above
the glass the tip stopped. A near miss (say 2 mm short) scores much better than 15 mm short, so each zone's
press-depth choice learns from it, and the next try goes deeper by exactly that gap (+2 mm) instead of a
fixed small step. The log shows "NO TOUCH, tip 4.2mm above glass".

**Steering onto the target in mm (2 cameras, `servo.py`):** the orange ring is the target; its spot on the glass
is known exactly from the page, so only the tip is tracked: x right, y down along the screen, z height (mm).
At the hover point the arm measures where the tip would land, then steers the joints until that landing is
within 1.5 mm of the target, and goes down as taught. How each joint moves the tip is learned from the arm's
own movement: the first time in each zone it wiggles each joint 2 degrees and watches the tip (a few seconds),
and every correction after that refines it (`touch/servo_jacobian.json`, per calibration). All moves stay
inside the safety envelope. If the cameras disagree or can't see the tip, it falls back to the normal hover
check. In `./lr tip-live`, click the map to place a target and see the live x / y / z distance to it.
After practice, `./lr tap` and the agent automatically use the learned model (`--no-learned` to compare).
Don't retry on a real CAPTCHA (a second tap toggles the tile back): practise on the test page, then demo.

**Per-zone learning + heatmap:** the screen is split into 9 zones, one per calibration target. Each zone keeps
its own statistics and learns its own best press depth and approach speed from the rewards (zones differ:
some strain the servos more). Practice spends more time on the hard zones. `./lr heatmap` (also shown after
each practice) draws accuracy per zone; `./lr tap` and the agent then use each zone's best settings.

**Self-correcting aim (simpler, used until there's experience):** after every measured test, the robot learns its systematic miss (shift, stretch,
bend) from all tests made with the current calibration, and aims slightly off to cancel it. Just repeat
`./lr tap --random 20` a couple of times; each run prints the expected miss before and after correction.
`./lr fit` refits by hand, `./lr fit --reset` removes it, `--no-correct` taps without it. Recalibrating
starts a fresh correction automatically.

**Logs:** every calibration and test run is saved to `touch/logs/` (`calib_<time>.json`, `test_<time>.json`)
with all touches, so you can compare runs (e.g. different `--smooth` or `--speed`).

## Camera hover-check (0, 1 or 2 cameras)

Put a small bright **sticker near the stylus tip** (pink by default; green/orange/blue also work) where the
cameras can see it. Then in `lr.conf`:

```bash
CAMERA=/dev/video4            # first camera (best: from above / straight on)
CAMERA2=ipwebcam:192.168.1.23 # optional second camera (best: lower, flatter, from the front)
VISION=auto                   # off = tap without cameras (default)
TIP_COLOR=pink
```

`./lr tip-test` checks the setup: markers, sticker found, each camera's viewing angle, and whether the
pair is different enough to measure height (aim for 25 deg or more between them).

Before each tap the arm hovers just above the target, the cameras see where the tip points, and the arm
corrects its aim, so the **first tap lands right** (important for the real CAPTCHA, where you can't retry).
Each camera's viewing-angle shift is learned from measured taps (`tap --test/--random`, `practice`); until
5 are learned it only watches. With 2 cameras it also estimates the tip's height (1 = normal hover) and
doesn't care if the hover height varies. Simulated with a front-low + top pair: landing predicted to
~1-3 px (2 cams) vs ~3-7 px (top cam alone) vs ~18 px (front-low cam alone). `--no-vision` taps without it.

**Live test without the arm:** `./lr tip-live`, then open `http://localhost:8766/` in your browser. Both camera
feeds (screen outlined green, tip circled), plus live: height of the tip above the glass in mm, where it is over
the screen, and whether the two cameras agree. Move the stylus by hand to test. Set `SCREEN_WIDTH_MM` (marker
to marker) in `lr.conf` for real millimetres.

The live page also has:
- **Frame rate / phone resolution / quality** controls (IP Webcam phones are set remotely; 1280x720 is plenty).
- **Save photo:** full-size frames + what it detected, in `outputs/snapshots/`. `./lr share` copies photos and
  logs to `Downloads/robot-share` for analysis.
- **Click the pen tip** in a picture to teach its colour (saved in `touch/tip_color.json`, used by tap and
  practice too; "reset taught colour" undoes it). "Pause" freezes the picture, "show what it thinks is the pen"
  tints the matching pixels.
- **Flip cam 1/2** for an upside-down camera (remembered).
- **Teach distance:** hold the tip still and press "touching" or "this far: N mm" (2+ of each, at different
  places). It learns the screen shape, each phone's lens and how high the marker sits at contact
  (`touch/cam_geometry.json`). Needed whenever the screen is a different shape than the default (1.51:1).

## Protecting the screen

- **Stop on touch (tests and practice):** the page reports a touch the instant the glass registers it, and the
  arm stops pressing right then and lifts off. No pushing after contact.
- **Force guard (OFF by default: it misread normal arm movement as pushing; enable with `--max-load 300`
  only after checking `peak_load_jump` in the test logs):** the motors report how hard they work. When the stylus pushes into the
  glass that load jumps, and the arm immediately stops pushing and lifts off. During calibration the
  follower stops following the leader deeper ("too much force" message). Tune with `--max-load`
  (lower = gentler, default 200 for taps; `0` turns it off). Test logs record `peak_load_jump` for each tap,
  so you can see how hard real taps push.
- **Optional strength cap:** `./lr tap --torque 60 ...` limits shoulder/elbow/wrist to 60% while tapping.
  Too low and the arm can't hold itself up: lower it step by step and watch.
- **Mechanical (the most reliable):** mount the stylus so it can slide back a little in the gripper
  (foam, a rubber band, or a spring behind it) and use a soft rubber-tipped stylus. Add a screen protector.

## Your CAPTCHA page

Add one line so the camera can find the screen:

```html
<script src="markers.js"></script>
```

Copy `touch/markers.js` next to your page. Keep the page background light and leave the corners free.

## Using it

```bash
./lr see                       # -> outputs/screen.jpg, straightened view of the screen
./lr tap 0.25 0.40 0.75 0.40   # tap two points
./lr tap --home                # park the arm
```

A point at pixel **(x, y)** in `screen.jpg` (1280×800) is **u = x / 1280, v = y / 800**.

From your agent's Python code (run inside the container: `./lr shell`, then `cd /repo/touch`):

```python
from see import ScreenCamera
from tap import Tapper

cam = ScreenCamera("/dev/video4")
img = cam.screen_image()                 # straightened screenshot (numpy BGR), None if markers not seen
# ... agent looks at img, returns chosen tiles as pixels (x, y) ...
points = [(x / img.shape[1], y / img.shape[0]) for x, y in chosen]

with Tapper(port="/dev/ttyACM0") as arm:   # goes home when done
    arm.tap_many(points)
```

The arm covers the screen while tapping, so call `see` **before** tapping, with the arm at home.

## The whole loop on the real site (`./lr captcha`)

```
see -> page? -> choice/checkbox: tap | grid: solver API -> tap tiles -> see: right tiles? -> Verify -> see: pass/fail
```

Site: `https://skibidisigma67rizz.pages.dev/`, solver: `https://captchasolver.alvaro-camposmndz.workers.dev/` (`POST /api/robot/solve`).

**One-time setup**

1. `lr.conf`: set `CLOUDFLARE_ACCOUNT_ID` and `CLOUDFLARE_API_TOKEN` (a Workers AI token of yours; never commit it). `SOLVER_URL` is preset.
2. On the site tick **Camera markers** (off by default; the camera needs them) and open it full-screen on the touch screen.
3. Calibrate the arm (`./lr touch-calib`, optionally `./lr practice`).
4. `./lr captcha --learn-layout`: show the choice page, the "I'm not a robot" box and the "Access granted" page, one after the other.
   For each, the camera takes a photo and you click on it (browser, `http://localhost:8765/`) where the robot must tap.
   It saves `touch/site_layout.json` (machine-specific, git-ignored). Redo it if the screen or browser zoom changes.
   Shortcut to the "Access granted" page: *Training settings -> Practice mode -> Single tile*, click the highlighted tile.

**Check before moving the arm**

```bash
./lr solve                  # camera -> solver -> prints the plan, never taps
./lr captcha --dry-run      # the loop's reading and solving, arm stays still (grid must be open)
```

**Just the coordinates (no arm): `./lr plan`**

`touch/plan_clicks.py` is the Playwright steps done from a photo. Give it one photo, it says what to click next as `u, v`:

```bash
./lr plan                                   # takes a camera photo now
./lr plan --image /work/outputs/photo.jpg   # a raw photo of the screen (4 markers visible); add --straight if already straightened
./lr plan --image photo.jpg --tiles 1,3,8 --full --annotate /work/outputs/plan.png   # no solver call, draws the clicks
```

It prints JSON: `state` (choice / checkbox / grid / grid_error / verified / unknown), `done`, and `clicks` with `role`, `tile`,
`u`, `v` (and `x`, `y` in the 1280x800 frame). Only tiles that still differ from the solver's answer are listed; once the
selection is right it returns Verify only (`--full` appends Verify after the tiles in one go). Pass `u v` to `./lr tap`.
Choice / checkbox positions come from `--learn-layout` (or `--demo-layout` for the committed 1280x800 screenshots).

**Run**

```bash
./lr captcha                # one challenge, starting from whichever page is showing
./lr captcha --count 5      # five in a row (after a pass it taps "Start again"; stops at the first fail)
```

Every run saves its photos, the solver's answer, the plan and the result in `outputs/run/<time>/` and appends a line to
`outputs/run/results.jsonl` (outcome, seconds): the scoreboard data.

**What it will and won't do** (a second tap on a tile toggles it back, so nothing is tapped "to be sure")

- The solver's tile positions must match the dialog found in the photo, otherwise nothing is tapped.
- After the tile taps the arm goes home and the camera reads which tiles show the blue check. If the set differs, only the
  differing tiles are tapped (max 2 rounds, `--max-fix`). Still wrong: it stops **without pressing Verify**.
- Verify is pressed once. The result is read from the page: "Access granted" = pass, red "Incorrect selection" = fail,
  nothing within 30 s = `unknown` (never counted as fail). `--verify-retries 1` allows one more Verify tap if the page did not react at all.
- Camera can't see all 4 markers, page unknown, solver says "try again": it looks again (new photo), up to `--tries`.
- Only the image grid (reCAPTCHA card) is handled. The slider needs a drag; Turnstile is a single checkbox tap and is not wired in yet.
- Stop-on-touch is not available here (the site doesn't report touches to us); the arm relies on its calibration.

**Offline tests** (no robot, no network): `.venv/bin/python -m pytest touch/tests -q`. They run the loop against real screenshots
of the site (`touch/tests/shots/`, regenerate with `touch/tests/capture_site.py`). `SITE_E2E=1 ... pytest touch/tests/test_e2e_site.py`
runs the loop against the live site in a headless browser (a mouse click stands in for the arm).

## Tuning

| Problem | Fix |
|---|---|
| Taps don't register | The test output says why: "stopped short" = blocked or weak; "reached its pose" = press deeper with `--press 0.08`; "force guard stopped it" = `--max-load 300`. Also try `--dwell 0.4` |
| Taps miss by a bit everywhere | Redo `touch-calib` and touch the dot centres more carefully |
| Taps miss more near the edges | `touch-calib` with a 4×4 grid: open `calib.html?grid=4`, add `--grid 4` in `lr` |
| Too slow / too fast | `--speed 1.5` or `--speed 0.7` |
| Jerky movement | `SERVO_ACCEL` in `lr.conf` (default 80; lower = softer, too low lags and overshoots). Try settings without the arm first: `./lr smooth-preview --accel 60 --speed 0.8` draws joint speeds old vs new in `outputs/smoothness.png` |
| Arm drags across the screen | Recalibrate and lift higher on your way to and from each target: the arm copies your path |
| Path still shaky / too rounded | `--smooth 0.15` smoother, `--smooth 0.04` closer to your raw movement |
| Want to move the home pose | `./lr set-home` (keeps the targets, but recorded paths still start from the old home) |
| `see` can't find markers | Look at `outputs/camera_debug.jpg`: all 4 markers visible, not too small, no glare |
