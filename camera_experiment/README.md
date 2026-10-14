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
- `screen.png`: straightened marker frame when present, otherwise the detected laptop display.
- `challenge.png`: useful content only, from the instruction banner through Verify.
- `geometry.json`: corners, both homographies, approximate camera pose, layout and solver status.
- `targets.json`: all nine tile centers and Verify, with normalized and photo coordinates. No model or credentials needed.
- `actions.json` and `plan.png`: solver click plan and overlay, only after a successful model run.
- `source-plan.jpg`: those same clicks projected onto the original source photograph.

`report.json` includes failures and negative controls. `solved` means the model returned
a plan; `comparison.json` checks whether its target and tile selection were correct.
Rerunning invalidates previous action files before processing so failed runs cannot
leave an old click plan behind. Generated images and reports in `results*/` are ignored.

## Arm coordinate contract

`actions[].x_norm` and `y_norm` refer to the **whole straightened frame**, with
origin top-left, u right and v down. They never refer to `challenge.png` or the raw
camera photo. The machine-readable `coordinate_frame.kind` distinguishes
`aruco-outer`, matching the click-test frame, from `display`, which describes
the whole visible display. A consumer must use a calibration in the same frame.
`photo_x` and `photo_y` project the point into the degraded `photo.jpg`;
`source_x` and `source_y` project it into the original input, even for downsampling.
`screen_to_photo` and `screen_to_source` describe these mappings explicitly.
The inverse transforms are saved too. `screen_offset_source_px` gives the
frame's top-left in the source image; `challenge_box` gives the useful crop's
offset and size in normalized-image pixels. Perspective needs the complete
homography, not just an x/y offset.

Default mode `--frame auto` prefers markers 0–3 using the existing `touch/see.py`
detector. The marker squares recover the frame's aspect, so the normalized image
is 1920 pixels wide and its height follows that aspect. `--height` overrides it.
Without markers, the fallback still assumes a 16:10 laptop display and defaults
to 1920×1200. The ten supplied photographs have no ArUco markers and use that
display frame. Use `--frame markers` to reject images without all four markers;
use `--frame display` to force the original experiment detector.

For a new picture, enable "Camera markers" on the CAPTCHA website and include all
four corners in the photo. A single PNG or JPG works as input:

```sh
mise run camera:compare /path/to/photo.jpg --frame markers --variants original
```

This writes offsets, transforms, the straightened images and `targets.json`.
Add `--solve` to choose target tiles and emit `actions.json`. No arm connection is
used. Saved pictures describe the captured page state; they are not live commands.

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
