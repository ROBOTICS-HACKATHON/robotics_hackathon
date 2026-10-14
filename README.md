# robotics_hackathon

SO-101 robot arm tools, a CAPTCHA playground, and a screen-solving agent.

- [`captcha/`](captcha/README.md): Astro CAPTCHA playground and photo assets.
- [`cad/`](cad/README.md): customizable printable stylus mount for the SO-101.
- [`touch/`](touch/README.md): screen calibration, camera capture, and arm tapping.
- [`captchasolver/`](captchasolver/README.md): screenshot solver and click actions.
- [`HOWTO.md`](HOWTO.md): record, train, and run the robot through `./lr` and Docker.
- [`PLAN.md`](PLAN.md): team workflow and screen-coordinate conventions.

Run the CAPTCHA website from the repository root:

```sh
cd captcha
bun install
cd ..
mise run dev
```

Open http://localhost:4321. Use `mise run build` to build and `mise run preview`
to serve the build. The website's source and assets live in `captcha/`.
