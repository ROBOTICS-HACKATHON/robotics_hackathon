# Team plan: robot arm solves CAPTCHAs

```
CAPTCHA page ──► camera ──► ./lr see ──► agent ──► taps (u, v) ──► arm ──► screen
                            (straight     (picks the answers)
                             screenshot)
```

The arm doesn't need AI training for this. It's calibrated once (`./lr touch-calib`) and then calculates where to tap.
The intelligence is the **agent**: a vision model that looks at the screenshot and picks the answers.

## Who does what

| Part | Builds | Delivers |
|---|---|---|
| **CAPTCHA page** | CAPTCHAs shown full-screen on the touch screen | A page that follows the rules below |
| **Robot** | Calibration, `see`, `tap`, accuracy | Taps that land within a few mm (`touch/README.md`) |
| **Agent** | Screenshot → vision model → answers | JSON list of points to tap |

## Rules everyone follows

### 1. Coordinates
Every position is **`u, v` from 0 to 1** (u left→right, v top→bottom), measured between the outer corners of the 4 corner markers.
Pixels are only used inside one part.

### 2. CAPTCHA page
- Include `<script src="markers.js"></script>` (copy `touch/markers.js`). Use a light background and keep the 4 corners empty.
- Full-screen, fixed layout: no scrolling, nothing moving around.
- **Tap targets at least ~2 cm** (about 100 px). The arm is accurate to millimetres, not pixels.
- **Taps only:** select tiles, then a **Verify** button. No dragging or sliders (maybe later).
- Show the result clearly: big ✓ or ✗, for the camera and the audience.
- Log every attempt (pass/fail, seconds taken) for a scoreboard.
- Also build an **accuracy test page**: it shows one random dot, records where the touch landed, and shows the error.

Good CAPTCHA types: 3×3 image grid ("select all with X"), "tap in order 1-2-3", "tap the odd one out", "type the code" on an on-screen keypad.

### 3. Agent
- Input: `outputs/screen.jpg`, the straightened screen at 1280×800. The agent sees **only what the camera sees**, never the page's code.
- Output:
  ```json
  {"taps": [[0.21, 0.35], [0.50, 0.35], [0.85, 0.92]], "reason": "tiles with traffic lights, then Verify"}
  ```
  The last tap is the Verify button. Pixel (x, y) → `u = x/1280, v = y/800`.
- It can be built and tested on saved screenshots, with no robot needed.

## Phases

1. **In parallel (~1–2 h):** robot calibrated and `./lr tap --test` hits all 9 dots · first 3×3 CAPTCHA + accuracy page · agent working on screenshots
2. **Measure the arm (~30 min):** 30 random dots on the accuracy page, aim for ≥95% hits on 2 cm targets
3. **Connect it all (~1 h):** loop `see → agent → tap → see (read ✓/✗)` → next CAPTCHA, retry once on failure
   - Built: `./lr captcha` (`touch/run_captcha.py`), see "The whole loop on the real site" in `touch/README.md`. It uses the team's solver API (`/api/robot/solve`) as the agent.
4. **Improve the demo:** more CAPTCHA types, faster taps (`--speed 1.5`), live scoreboard, dragging as a stretch goal
