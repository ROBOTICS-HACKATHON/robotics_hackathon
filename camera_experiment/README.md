# Camera photo experiment

Uses the ten photos in `/home/veya/Downloads/asodkasodkasd` without modifying them.
Five have an open CAPTCHA; five are useful negative controls. Detection and solving
use pixels only. `expected.json` contains human photo labels used only for evaluation.

Setup from the repository root:

```sh
uv venv camera_experiment/.venv
uv --directory camera_experiment pip install --python .venv/bin/python -r requirements.txt
```

Run geometry and all seven image-quality variants without credentials:

```sh
mise run camera:compare /home/veya/Downloads/asodkasodkasd
```

Run the real solver with credentials loaded in the shell:

```sh
export CLOUDFLARE_ACCOUNT_ID=your-account-id
export CLOUDFLARE_API_TOKEN=your-token
camera_experiment/.venv/bin/captchasolver compare --models @cf/google/gemma-4-26b-a4b-it --preflight-only
mise run camera:compare /home/veya/Downloads/asodkasodkasd --solve --out camera_experiment/results-live
```

For a smaller run use `--variants original`. For a model comparison use
`--model @cf/mistralai/mistral-small-3.1-24b-instruct --out camera_experiment/results-mistral`.
Separate output directories keep model runs independent.

Evaluate saved results:

```sh
mise run camera:report camera_experiment/results-live/report.json camera_experiment/results-mistral/report.json
```

## Images and JSON

Each source/variant directory contains:

- `photo.jpg`: degraded input, still in camera coordinates.
- `screen.png`: detected laptop display, straightened to 1920×1200 by default.
- `challenge.png`: useful content only, from the instruction banner through Verify.
- `geometry.json`: corners, both homographies, approximate camera pose, layout and solver status.
- `actions.json` and `plan.png`: solver click plan and overlay, only after a successful model run.

`report.json` includes failures and negative controls. `solved` means the model returned
a plan; `comparison.json` checks whether its target and tile selection were correct.
Rerunning invalidates previous action files before processing so failed runs cannot
leave an old click plan behind. Generated images and reports in `results*/` are ignored.

## Arm coordinate contract

`actions[].x_norm` and `y_norm` refer to the **whole straightened display**, with
origin top-left, u right and v down. They never refer to `challenge.png` or the raw
camera photo. Pass `(u, v)` to the calibrated arm. `photo_x` and `photo_y` project
the same click back into `photo.jpg` through `screen_to_photo`.

The defaults assume a 16:10 screen. Set `--width` and `--height` to the screen's
actual aspect ratio. Output resolution can differ from physical pixel resolution.
No ArUco markers are present in these photos. These corners describe the visible
display, whereas `touch/see.py` uses the outer marker frame. Arm calibration must
use the same frame; the experiment does not command the robot.

The car photo ending `142116069` already has tiles selected. The JSON describes
the desired selected tiles, not a state-aware toggle sequence. Do not replay that
plan on an already-selected challenge without clearing or reconciling its state.

## Pose and quality limits

`pose_estimate` uses the photo's EXIF 35mm-equivalent focal length with OpenCV
planar solvePnP. Camera position is in screen-width units, with a centered principal
point and zero distortion assumed. It is not a calibrated metric pose. Reprojection
error measures fit to the detected corners, not physical position accuracy.
Screen normalization and clicks use the homography, independently of this estimate.

Variants damage the entire camera input before detection: Gaussian blur (sigma 4),
JPEG quality 12, downsampling to 640 pixels wide, Gaussian noise (sigma 14), exposure
multiplier 0.48, and all of those combined (blur sigma 1.4). Noise seed defaults to 42.
Detection expects an enclosed, bright, low-saturation rectangular laptop display.
It is a photo experiment, not a universal monitor detector.

Both tested models missed the first dog behind debug labels on the original dog grid.
Both were exact on 4/5 original challenges. Gemma was exact on 4/5 with blur or
darkness, 3/5 with JPEG compression, low resolution or noise, and 0/5 with combined
damage. The two photographed training overlays agree with recovered tile centers
within 0.004 on each normalized axis, as recorded in `verification.json`.
Very degraded images can also lose the display or dialog; those produce no action
file. The tracked `comparison.json` records measured results and `preview.jpg`
shows the original plans. No robot tapping or server-side Verify was performed.
