# robotics_hackathon

Astro CAPTCHA playground, run with Bun from the repository root.

```sh
bun install
mise run dev
```

Open http://localhost:4321. Build with `mise run build`.

Choose one of ten numbered image grids or slider puzzles, or the checkbox-only
Turnstile mock. Then click the verification checkbox. The challenge
opens over a blurred page. "Open fullscreen" uses browser fullscreen when available.
Solving navigates to `/verified/` and keeps the completed challenge locked until
"Start again" is clicked.

"High contrast" uses black borders and controls, larger text and selection marks,
and stronger image contrast. The preference is saved in the browser.

"Training overlay" adds target boxes, center markers, screen coordinates,
and distance measurements over the challenge. Coordinates start at the viewport's
top-left corner and use CSS pixels; normalized coordinates range from 0 to 1.
The slider shows the piece's horizontal offset from the gap. Markers update when
the window changes size or the slider moves, and do not intercept clicks.
These are screen measurements, not physical distances or camera coordinates.

Change prompts, image paths, and `correct` values in `src/data/image-challenges.ts`.
Put your images in `public/images/`; the grid expects nine tiles. The included
photos are downloaded from the web; credits are in `public/images/SOURCES.md`
and the `grid/` and `slider/` source lists. Reset keeps the chosen variant.

`sliderChallenges` in `src/data/slider-challenges.ts` controls the puzzle photo, gap position,
piece size, and alignment tolerance. Positions and sizes are fractions of the image.
Drag and release the slider to verify; arrow keys work too.

Use the challenge number to repeat a setup while recording robot demonstrations.
The site supplies the tasks and answer labels; it does not record camera frames
or robot actions. Those recordings are needed to train the robot policy.

This is a local mock, not an authentication system. No external CAPTCHA service is used.
