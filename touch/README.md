# Touch screen: arm taps CAPTCHA answers

```
camera ──► ./lr see ──► straightened screenshot ──► agent picks tiles (u, v) ──► ./lr tap u v ...
```

Positions on the screen are always **u, v from 0 to 1**: u goes left to right, v goes top to bottom. They're measured between the outer corners of the 4 black markers.

## Setup (once, and again if the arm or screen moves)

1. **Stylus:** fix a **capacitive stylus** in the gripper (rubber or mesh tip; tape it so it can't slip). A plastic gripper tip won't register on a touch screen.
2. **Fix everything down:** tape the arm base and the screen to the table. If either one moves, redo step 4.
3. **Camera:** mount it so it sees the whole screen, including all 4 corner markers, with no glare.
4. **Teach the screen:** run
   ```bash
   ./lr touch-calib
   ```
   On the touch screen, open the address it prints (`http://localhost:8765/` when the touch screen is plugged into this computer) full-screen (F11).
   - Hold the stylus a few cm **above** the middle of the screen with the leader arm (home), press Enter.
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

**Self-correcting aim:** after every measured test, the robot learns its systematic miss (shift, stretch,
bend) from all tests made with the current calibration, and aims slightly off to cancel it. Just repeat
`./lr tap --random 20` a couple of times; each run prints the expected miss before and after correction.
`./lr fit` refits by hand, `./lr fit --reset` removes it, `--no-correct` taps without it. Recalibrating
starts a fresh correction automatically.

**Logs:** every calibration and test run is saved to `touch/logs/` (`calib_<time>.json`, `test_<time>.json`)
with all touches, so you can compare runs (e.g. different `--smooth` or `--speed`).

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

## Tuning

| Problem | Fix |
|---|---|
| Taps don't register | `./lr tap --press 0.08 ...` pushes deeper; check the stylus tip touches flat |
| Taps miss by a bit everywhere | Redo `touch-calib` and touch the dot centres more carefully |
| Taps miss more near the edges | `touch-calib` with a 4×4 grid: open `calib.html?grid=4`, add `--grid 4` in `lr` |
| Too slow / too jerky | `--speed 1.5` or `--speed 0.7` |
| Arm drags across the screen | Recalibrate and lift higher on your way to and from each target: the arm copies your path |
| Path still shaky / too rounded | `--smooth 0.15` smoother, `--smooth 0.04` closer to your raw movement |
| Want to move the home pose | `./lr set-home` (keeps the targets, but recorded paths still start from the old home) |
| `see` can't find markers | Look at `outputs/camera_debug.jpg`: all 4 markers visible, not too small, no glare |
