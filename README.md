# robotics_hackathon

Astro CAPTCHA playground, run with Bun from the repository root.

```sh
bun install
mise run dev
```

Open http://localhost:4321. Build with `mise run build`.

Choose a challenge, then click the fake "I'm not a robot" checkbox. The challenge
opens over a blurred page. "Open fullscreen" uses browser fullscreen when available.
Solving navigates to `/verified/` and keeps the completed challenge locked until
"Start again" is clicked.

Change the prompt, image paths, and `correct` values in `src/data/challenges.ts`.
Put your images in `public/images/`; the grid expects nine tiles. The included
photos are downloaded from Pexels; credits are in `public/images/SOURCES.md`.
Images stay in place until reset.

This is a local mock, not an authentication system. No external CAPTCHA service is used.
