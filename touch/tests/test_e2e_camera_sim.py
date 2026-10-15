"""Opt-in: the loop against the REAL site, seen through a SIMULATED CAMERA (perspective, blur, noise, dim light, a
cluttered background) and the real marker detection + straightening of see.py. Ground truth comes from the page:
which tiles were really selected at the moment Verify was clicked.

  SITE_E2E=1 .venv/bin/python -m pytest touch/tests/test_e2e_camera_sim.py -q -s
"""

import os

import cv2
import numpy as np
import pytest

pytest.importorskip("playwright")
if not os.environ.get("SITE_E2E"):
    pytest.skip("set SITE_E2E=1 to run against the live site", allow_module_level=True)

from playwright.sync_api import sync_playwright

import pagestate as ps
from fakeworld import IMG, POS, Clock, World
from run_captcha import Config, Run
from see import find_screen, straighten

URL = os.environ.get("SITE_URL", "https://robot-captcha-demo.pages.dev/training/")
W, H, INSET = 1280, 800, 18

# (camera frame size, where the screen's corners land, blur sigma, brightness, noise sigma)
SCENES = {
    "head_on_bright": ((1920, 1080), [(420, 130), (1500, 130), (1500, 940), (420, 940)], 0.8, 1.0, 3),
    "angled_left": ((1920, 1080), [(300, 170), (1640, 100), (1700, 990), (250, 920)], 1.5, 0.85, 6),
    "from_below_dim": ((1920, 1080), [(380, 60), (1560, 90), (1620, 1010), (330, 960)], 2.0, 0.65, 8),
    "small_and_far": ((1280, 720), [(380, 160), (900, 140), (920, 560), (360, 580)], 1.2, 0.9, 5),
}


def camera_frame(screenshot, scene, seed):
    (fw, fh), quad, blur, bright, noise = SCENES[scene]
    rng = np.random.default_rng(seed)
    bg = np.full((fh, fw, 3), (90, 110, 130), np.uint8)
    bg += rng.integers(0, 25, bg.shape, dtype=np.uint8)
    src = np.float32([[0, 0], [W, 0], [W, H], [0, H]])
    M = cv2.getPerspectiveTransform(src, np.float32(quad))
    warped = cv2.warpPerspective(screenshot, M, (fw, fh), flags=cv2.INTER_AREA)
    mask = cv2.warpPerspective(np.full(screenshot.shape[:2], 255, np.uint8), M, (fw, fh))
    f = np.where(mask[..., None] > 0, warped, bg).astype(np.float32)
    f = cv2.GaussianBlur(f, (0, 0), blur) * bright + rng.normal(0, noise, f.shape)
    return np.clip(f, 0, 255).astype(np.uint8), np.float32(quad)


class Rig:
    def __init__(self, page, scene, seed=0):
        self.page, self.scene, self.seed = page, scene, seed
        self.at_verify = None
        self.err_px = []

    def look(self):
        png = self.page.screenshot()
        shot = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_COLOR)
        frame, true_quad = camera_frame(shot, self.scene, self.seed)
        self.seed += 1
        quad, _ = find_screen(frame)
        if quad is None:
            return None
        # how well were the screen corners found? (in the camera frame, px)
        self.err_px.append(float(np.abs(quad - true_quad_outer(true_quad)).max()))
        return straighten(frame, quad, W, H)

    def tap(self, u, v):
        x, y = INSET + u * (W - 2 * INSET), INSET + v * (H - 2 * INSET)
        if abs(u - POS["verify"][0]) < 0.04 and abs(v - POS["verify"][1]) < 0.04:
            self.at_verify = self.selected()
        self.page.mouse.click(x, y)
        self.page.wait_for_timeout(250)

    def selected(self):
        return {i for i in range(9) if self.page.get_attribute(f"#image-challenge .image-tile >> nth={i}", "aria-pressed") == "true"}

    def go_home(self):
        pass


def true_quad_outer(quad):
    """The marker outer corners sit INSET px inside the page, so the screen quad the detector finds is a bit smaller."""
    q = np.float32(quad)
    c = q.mean(0)
    return q + (c - q) * (2 * INSET / W)


@pytest.mark.parametrize("scene", list(SCENES))
def test_loop_through_a_simulated_camera(scene, tmp_path):
    lay = ps.SiteLayout({}, tmp_path / "l.json")
    for n in ps.PAGES:
        lay.teach(n, IMG[n], POS[n])
    wanted = (1, 3, 8)
    with sync_playwright() as p:
        b = p.chromium.launch(executable_path=os.environ.get("CHROMIUM", "/usr/bin/chromium"), args=["--no-sandbox"])
        page = b.new_page(viewport={"width": W, "height": H})
        page.goto(URL, wait_until="networkidle")
        page.check("#screen-markers-toggle")
        page.select_option("#challenge-number", "image:4")
        rig, clock, world = Rig(page, scene), Clock(), World()

        def sleep(s):
            page.wait_for_timeout(int(s * 1000))
            clock.sleep(s)

        run = Run(rig.look, lambda img, t, m: world.presses(wanted), rig, lay, Config(settle=0.5), tmp_path,
                  log=print, sleep=sleep, clock=clock.now)
        [r] = run.run()
        b.close()
    print(f"{scene}: {r.outcome} in {r.seconds}s, fixes={r.fixes}, worst corner error {max(rig.err_px):.1f}px in the camera frame")
    assert r.outcome in ("pass", "fail"), r.detail        # the site answered: the pipeline got all the way
    assert rig.at_verify == set(wanted), f"page really had {rig.at_verify} selected when Verify was clicked"
