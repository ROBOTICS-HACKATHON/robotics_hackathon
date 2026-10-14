# Touch screen: arm taps CAPTCHA answers

```
camera ──► ./lr see ──► straightened screenshot ──► agent picks tiles (u, v) ──► ./lr tap u v ...
```

Positions on the screen are always **u, v from 0 to 1**: u goes left to right, v goes top to bottom. They're measured between the outer corners of the 4 black markers.

## Setup (once, and again if the arm or screen moves)

1. **Stylus:** fix a **capacitive stylus** in the gripper (rubber or mesh tip; tape it so it can't slip). A plastic gripper tip won't register on a touch screen.
2. **Fix everything down:** tape the arm base and the screen to the table. If either one moves, redo step 4.
3. **Camera:** mount it so it sees the whole screen, including all 4 corner markers, with no glare.
4. **Teach the screen:** open `touch/calib.html` full-screen (F11) on the touch screen, then run:
   ```bash
   ./lr touch-calib
   ```
   Use the leader arm to drive the stylus to **home** (a few cm above the screen's middle), press Enter, then **lightly touch dots 1–9**, pressing Enter at each one.
5. **Check:**
   ```bash
   ./lr tap --test
   ```
   The arm taps all 9 dots, and each dot turns red when it's hit. The page also shows where every touch landed.

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
| Arm drags across the screen | `--lift 0.5` (backs off further between taps) |
| `see` can't find markers | Look at `outputs/camera_debug.jpg`: all 4 markers visible, not too small, no glare |
