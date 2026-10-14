"""--learn-layout: fake camera shows each page, a fake 'user' clicks on the picker page in the browser."""

import json
import socket
import threading
import time
import urllib.request

import pagestate as ps
from fakeworld import IMG, POS
from learn_layout import learn


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class Cam:
    def __init__(self):
        self.page = None

    def screen_image(self, allow_remembered=True):
        return IMG[self.page].copy()


def test_teaches_the_three_pages_from_clicks_on_the_photo(tmp_path, monkeypatch):
    cam, port, asked = Cam(), free_port(), []

    def fake_input(prompt=""):
        cam.page = ps.PAGES[len(asked)]
        asked.append(cam.page)
        return ""
    monkeypatch.setattr("builtins.input", fake_input)

    def user():
        for name in ps.PAGES:
            u, v = POS[name]
            for _ in range(200):                                    # wait until the picker for this page is up
                try:
                    info = json.load(urllib.request.urlopen(f"http://127.0.0.1:{port}/info", timeout=1))
                    if ps.TAP_PAGES[name] in info["title"]:
                        break
                except Exception:
                    time.sleep(0.05)
            urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{port}/pick", method="POST",
                                                          data=json.dumps({"u": u, "v": v}).encode()), timeout=2)
            time.sleep(0.3)

    t = threading.Thread(target=user, daemon=True)
    t.start()
    path = tmp_path / "site_layout.json"
    layout = learn(cam, path, tmp_path / "out", delay=0, port=port)
    t.join(5)
    assert asked == list(ps.PAGES) and layout.missing() == []
    again = ps.SiteLayout.load(path)
    for name in ps.PAGES:
        assert again.tap(name) == (round(POS[name][0], 5), round(POS[name][1], 5))
        assert (tmp_path / "out" / f"layout_{name}_tap.png").exists()
    # and the taught pages are then recognised
    assert [ps.classify(IMG[n], again).kind for n in ps.PAGES] == list(ps.PAGES)
