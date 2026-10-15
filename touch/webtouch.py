"""Robot link to the real website: where did the stylus land, and did the challenge pass?

The site (captcha/, deployed on Cloudflare Pages) reports every touch and every challenge result to its
own API (/api/touches); this reads them. It has the same interface as TouchServer (drain, all_touches,
set, touched), so practice and the CAPTCHA runner use either one.

  TOUCH_WEB=https://<your-project>.pages.dev   ROBOT_KEY=<the secret you set on Cloudflare>   (lr.conf)

Open the site once with ?robot=<ROBOT_KEY> on the touch screen so the page sends touches.
"""

import json
import os
import queue
import threading
import time
import urllib.error
import urllib.request


class WebTouches:
    def __init__(self, base=None, key=None, poll=0.15, accept_mouse=False, **state):
        self.base = (base or os.environ.get("TOUCH_WEB", "")).rstrip("/")
        self.key = key or os.environ.get("ROBOT_KEY", "")
        if not self.base or not self.key:
            raise SystemExit("Robot link needs TOUCH_WEB and ROBOT_KEY (lr.conf).")
        self.poll, self.accept_mouse = poll, accept_mouse
        self.touches = queue.Queue()      # dicts: u, v, w, h, t, el, page, challenge
        self.results = queue.Queue()      # dicts: state (pass|fail), challenge, t
        self.keys = queue.Queue()         # (TouchServer compatibility; the site has no Enter key)
        self.touched = threading.Event()
        self.state = {"msg": "", **state}
        self.errors, self._warned, self._stop = 0, 0.0, threading.Event()
        self.after = self._get("/api/touches")["latest"]   # only events from now on
        self._state_dirty = threading.Event()
        self.set(**self.state)
        threading.Thread(target=self._loop, daemon=True).start()
        print(f"Robot link: {self.base}  (open {self.base}/training/?robot=<ROBOT_KEY> once on the touch screen;"
              f" practice page: {self.base}/practice/)")

    # ---------------------------------------------------------------- http
    def _req(self, path, body=None):
        data = None if body is None else json.dumps(body).encode()
        r = urllib.request.Request(self.base + path, data=data, method="GET" if body is None else "POST",
                                   headers={"x-robot-key": self.key, "content-type": "application/json",
                                            "user-agent": "robot-link/1"})
        with urllib.request.urlopen(r, timeout=5) as resp:
            return json.loads(resp.read() or b"{}")

    def _get(self, path):
        try:
            return self._req(path)
        except urllib.error.HTTPError as e:
            msg = {401: "wrong ROBOT_KEY", 503: "API not set up (D1 binding DB + secret ROBOT_KEY)",
                   404: "no /api on this site (deploy the robot-link version)"}.get(e.code, f"HTTP {e.code}")
            raise SystemExit(f"Robot link {self.base}: {msg}")
        except OSError as e:
            raise SystemExit(f"Robot link {self.base} not reachable: {e}")

    def _loop(self):
        while not self._stop.is_set():
            try:
                if self._state_dirty.is_set():
                    self._state_dirty.clear()
                    self._req("/api/state", self.state)
                d = self._req(f"/api/touches?after={self.after}")
                for ev in d.get("events", []):
                    self.after = max(self.after, int(ev["id"]))
                    self._take(ev)
                self.errors = 0
            except Exception as e:  # network hiccup: keep trying, say so now and then
                self.errors += 1
                if time.time() - self._warned > 10:
                    print(f"   (robot link: {type(e).__name__}: {e}; retrying)", flush=True)
                    self._warned = time.time()
            self._stop.wait(self.poll)

    def _take(self, ev):
        now = time.time()
        if ev["type"] == "touch":
            if ev.get("pointer") == "mouse" and not self.accept_mouse:
                return
            if ev.get("u") is None or ev.get("v") is None:
                return
            self.touches.put({"u": float(ev["u"]), "v": float(ev["v"]), "w": ev.get("w"), "h": ev.get("h"), "t": now,
                              "el": ev.get("el"), "page": ev.get("page"), "phase": ev.get("phase"),
                              "challenge": ev.get("challenge"), "id": ev["id"]})
            self.touched.set()
        elif ev["type"] == "result":
            self.results.put({"state": ev.get("state"), "challenge": ev.get("challenge"), "t": now, "id": ev["id"]})

    # ---------------------------------------------------------------- TouchServer interface
    def set(self, **kw):
        self.state.update(kw)
        self._state_dirty.set()

    def drain(self):
        self.touched.clear()
        while not self.touches.empty():
            self.touches.get_nowait()

    def all_touches(self, wait=1.5, extra=0.25):
        """Touches since the last drain. The site needs a moment (page -> Cloudflare -> here), so wait up to
        `wait` s for the first one, then `extra` s for any more (a bounce or drag gives two)."""
        end = time.time() + wait
        while self.touches.empty() and time.time() < end:
            time.sleep(0.05)
        if not self.touches.empty():
            time.sleep(extra)
        out = []
        while not self.touches.empty():
            out.append(self.touches.get_nowait())
        return out

    def drain_results(self):
        while not self.results.empty():
            self.results.get_nowait()

    def result(self, timeout=0.0):
        """The next challenge result (pass/fail) the page reported, or None."""
        try:
            return self.results.get(timeout=timeout) if timeout > 0 else self.results.get_nowait()
        except queue.Empty:
            return None

    def stop(self):
        try:
            self._req("/api/state", {**self.state, "target": None})
        except Exception:
            pass
        self._stop.set()


if __name__ == "__main__":   # quick check:  python webtouch.py   (touch the screen, Ctrl+C to stop)
    w = WebTouches(msg="Link test: touch the screen")
    print("Touch the screen on the site; Ctrl+C to stop.")
    try:
        while True:
            for t in w.all_touches(wait=1.0):
                print(f"  touch u={t['u']:.3f} v={t['v']:.3f}  on {t['el'] or '-'}  ({t['page']})")
            r = w.result()
            if r:
                print(f"  result: {r['state'].upper()}  {r['challenge'] or ''}")
    except KeyboardInterrupt:
        w.stop()
