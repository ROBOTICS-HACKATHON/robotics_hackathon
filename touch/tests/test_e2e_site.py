"""Opt-in: the whole loop against the REAL site in a headless browser (needs network + playwright + chromium).

  SITE_E2E=1 .venv/bin/python -m pytest touch/tests/test_e2e_site.py -q -s

The "arm" is a mouse click at the (u, v) it is told to tap, the "camera" is a screenshot cropped to the
markers' outer corners, exactly like see.py's output. The solver is faked (a fixed answer), so the site may
say pass or fail: both prove the flow works; what must hold is that every tap landed where intended.
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
from run_captcha import Config, Run
from fakeworld import IMG, POS, Clock, World

URL = os.environ.get("SITE_URL", "https://skibidisigma67rizz.pages.dev/")
W, H, INSET = 1280, 800, 18


class Browser:
    def __init__(self, page):
        self.page, self.taps = page, []

    def look(self):
        png = self.page.screenshot()
        img = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_COLOR)[INSET:H - INSET, INSET:W - INSET]
        return cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)

    def tap(self, u, v):
        x, y = INSET + u * (W - 2 * INSET), INSET + v * (H - 2 * INSET)
        self.taps.append((u, v))
        self.page.mouse.click(x, y)
        self.page.wait_for_timeout(250)

    def go_home(self):
        pass


def test_real_site_flow(tmp_path):
    lay = ps.SiteLayout({}, tmp_path / "l.json")
    for n in ps.PAGES:
        lay.teach(n, IMG[n], POS[n])           # taught from the committed screenshots of this same site
    with sync_playwright() as p:
        b = p.chromium.launch(executable_path=os.environ.get("CHROMIUM", "/usr/bin/chromium"), args=["--no-sandbox"])
        page = b.new_page(viewport={"width": W, "height": H})
        page.goto(URL, wait_until="networkidle")
        page.check("#screen-markers-toggle")
        page.select_option("#challenge-number", "image:4")
        br = Browser(page)
        world = World()
        clock = Clock()
        real_sleep = page.wait_for_timeout

        def sleep(s):
            real_sleep(int(s * 1000))
            clock.sleep(s)

        run = Run(br.look, lambda img, t, m: world.presses((1, 3, 8)), br, lay, Config(settle=0.5), tmp_path,
                  log=print, sleep=sleep, clock=clock.now)
        [r] = run.run()
        print(r)
        assert r.outcome in ("pass", "fail"), r.detail
        assert page.is_visible("text=Access granted") if r.outcome == "pass" else page.is_visible("text=Incorrect selection")
        b.close()
