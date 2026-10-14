# robotics_hackathon

Astro CAPTCHA playground, run with Bun from the repository root.

```sh
bun install
mise run dev
```

Open http://localhost:4321. Build with `mise run build`.

Choose one of 400 photo puzzles (200 image grids and 200 sliders), or the checkbox-only
Turnstile mock. Then click the verification checkbox. The challenge
opens over a blurred page. "Open fullscreen" uses browser fullscreen when available.
Solving navigates to `/verified/` and keeps the completed challenge locked until
"Start again" is clicked.
If the browser cannot save completion, the finished task stays visible with a
retry button. Fullscreen failures fall back to the current window.

"High contrast" uses black borders and controls, larger text and selection marks,
and stronger image contrast. The preference is saved in the browser.

"Training overlay" adds target boxes, center markers, screen coordinates,
and distance measurements over the challenge. Coordinates start at the viewport's
top-left corner and use CSS pixels; normalized coordinates range from 0 to 1.
The slider shows the piece's horizontal offset from the gap. Markers update when
the window changes size or the slider moves, and do not intercept clicks.
These are screen measurements, not physical distances or camera coordinates.
Expand "Training settings" to show only markers, coordinates, or distances,
and focus the overlay on one target. Labels use solid backgrounds for contrast.

Practice modes isolate a checkbox click, a chosen tile click, or a slider drag.
Tile and slider practice open directly, skipping the first checkbox. Full flow
keeps the complete challenge. The chosen mode and tile are remembered per tab.

Change prompts, image paths, and `correct` values in `src/data/image-challenges.ts`.
Put your images in `public/images/`; the grid expects nine tiles. The included
photos are downloaded from the web; credits are in `public/images/SOURCES.md`
and the `grid/` and `slider/` source lists. Reset keeps the chosen variant.

`sliderChallenges` in `src/data/slider-challenges.ts` controls the puzzle photo, gap position,
piece size, and alignment tolerance. Positions and sizes are fractions of the image.
Drag and release the slider to verify; arrow keys work too.

Use the challenge ID to repeat a setup while recording robot demonstrations.
Choose "Random" to pick a variant each time a challenge opens. "Start again"
keeps Random selected; resetting an open puzzle keeps its current variant.
The default 80/20 split has 320 training puzzles and 80 held-out test puzzles.
Test photos are separate from training photos at that default ratio. The training
percentage is configurable; custom percentages split tasks and can share photos
across the resulting pools.
Use the training side panel to choose a split, task type, and count. A session
advances automatically after each success and ends after the chosen number of tasks.
The default count is 100 tasks, suitable for consecutive screenshot-based solving.
Sessions default to image grids. Sliders are an optional bonus; choose Mixed to
include both kinds. Each kind has 160 training tasks and 40 test tasks by default.
Stopping or closing the challenge cancels the session. Sessions sample without
replacement and do not mix the training and test splits.
Photos are served as AVIF, converted locally with ImageMagick at quality 55.
The site supplies the tasks and answer labels; it does not record camera frames
or robot actions. Those recordings are needed to train the robot policy.

This is a local mock, not an authentication system. No external CAPTCHA service is used.
