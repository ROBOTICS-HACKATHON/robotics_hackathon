"""Capture screenshots of the real demo site in every state the robot meets, as the camera pipeline would
deliver them (straightened to the area between the outer corners of the 4 markers, resized to 1280x800),
plus `positions.json`: where the things the robot taps really are, as (u, v). Test fixtures only.

  .venv/bin/python touch/tests/capture_site.py            # writes touch/tests/shots/*.png + positions.json

Needs `playwright` and a system chromium (CHROMIUM=/usr/bin/chromium). The robot itself never touches the
page's code; the positions come from the DOM here only so the tests have ground truth to compare against.
The site needs "Camera markers" ticked (it is off by default); this script ticks it.
Fixtures are committed so the offline tests need no browser.
"""

import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
from playwright.sync_api import sync_playwright

URL = os.environ.get("SITE_URL", "https://robot-captcha-demo.pages.dev/training/")
OUT = Path(__file__).resolve().parent / "shots"
W, H = 1280, 800
MARKER_INSET = 18  # the site draws the markers 18 px from the viewport corners
POS = {}


def shot(page, name):
    """Viewport -> what `see.py` would produce: crop to the markers' outer corners, resize to 1280x800."""
    png = page.screenshot()
    img = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_COLOR)
    img = img[MARKER_INSET:H - MARKER_INSET, MARKER_INSET:W - MARKER_INSET]
    img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(OUT / f"{name}.png"), img)
    print("saved", name)


def uv(page, selector):
    """Centre of an element as (u, v) in the straightened frame."""
    b = page.locator(selector).first.bounding_box()
    x, y = b["x"] + b["width"] / 2, b["y"] + b["height"] / 2
    return [round((x - MARKER_INSET) / (W - 2 * MARKER_INSET), 5), round((y - MARKER_INSET) / (H - 2 * MARKER_INSET), 5)]


def start(b, mode=None):
    page = b.new_page(viewport={"width": W, "height": H})
    page.goto(URL, wait_until="networkidle")
    page.check("#screen-markers-toggle")   # "Camera markers": off by default, the robot needs it ON
    if mode:
        page.evaluate("document.querySelectorAll('details').forEach(d => d.open = true)")
        page.select_option("#practice-mode", mode)
    return page


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        b = p.chromium.launch(executable_path=os.environ.get("CHROMIUM", "/usr/bin/chromium"),
                              args=["--no-sandbox"])

        # --- the normal route: choice -> checkbox -> grid -> (wrong answer) error
        page = start(b)
        page.select_option("#challenge-number", "image:4")
        shot(page, "choice")
        POS["choice"] = uv(page, '[data-challenge="image"]')
        page.click('[data-challenge="image"]')
        page.wait_for_selector("#human-checkbox", state="visible")
        page.wait_for_timeout(600)
        shot(page, "checkbox")
        POS["checkbox"] = uv(page, "#human-checkbox .checkbox-box")
        page.click("#human-checkbox")
        page.wait_for_selector("#image-challenge .image-tile", state="visible")
        page.wait_for_timeout(1500)
        shot(page, "grid_empty")
        POS["tiles"] = [uv(page, f"#image-challenge .image-tile >> nth={i}") for i in range(9)]
        POS["verify"] = uv(page, "#image-challenge .verify-button")
        for i in (0, 4, 8):
            page.click(f"#image-challenge .image-tile >> nth={i}")
        page.wait_for_timeout(300)
        shot(page, "grid_three_selected")
        for i in (1, 2, 3, 5, 6, 7):
            page.click(f"#image-challenge .image-tile >> nth={i}")
        page.wait_for_timeout(300)
        shot(page, "grid_all_selected")
        for i in (1, 2, 3, 5, 6, 7, 8):
            page.click(f"#image-challenge .image-tile >> nth={i}")   # toggles those off: only 0 and 4 stay on
        page.wait_for_timeout(300)
        shot(page, "grid_tiles_0_4")
        page.click("#image-challenge .verify-button")
        page.wait_for_timeout(800)
        shot(page, "grid_error")
        page.close()

        # --- a pass: the site's own "tile practice" mode (click the highlighted tile) ends on the success page
        page = start(b, "tile")
        page.click('[data-challenge="image"]')
        page.wait_for_timeout(1500)
        page.click("#image-challenge .image-tile[data-practice-target]")
        page.wait_for_url("**/verified/**")
        page.wait_for_timeout(800)
        shot(page, "verified")
        POS["verified"] = uv(page, "text=Start again")
        page.close()
        b.close()
    (OUT / "positions.json").write_text(json.dumps(POS, indent=1))
    print("positions:", POS)


if __name__ == "__main__":
    sys.exit(main())
