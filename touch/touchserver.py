"""Tiny web server shared by calibration and tap tests.

Serves calib.html + markers.js, publishes a `state` dict the page polls,
and collects the touches the page reports (exact landing spot in u, v).
"""

import json
import mimetypes
import queue
import threading
import time
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
LOG_DIR = HERE / "logs"


class TouchServer:
    def __init__(self, port=8765, keys=None, **state):
        self.touches = queue.Queue()  # dicts: u, v, w, h (frame size in px), t
        self.keys = keys if keys is not None else queue.Queue()  # Enter/Space pressed on the page
        self.touched = threading.Event()
        self.traces = queue.Queue()  # full contacts from the page: {"pts": [[u, v, ms], ...], "ms": duration}  # set the moment any touch arrives (used to stop pressing)
        self.state = {"step": 0, "msg": "Starting...", "mode": "calib", **state}
        srv = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _send(self, code, body, ctype="application/json"):
                self.send_response(code)
                self.send_header("Content-Type", ctype)
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                path = self.path.split("?")[0]
                if path == "/state":
                    return self._send(200, json.dumps(srv.state).encode())
                name = "calib.html" if path in ("/", "") else path.lstrip("/")
                f = (HERE / name).resolve()
                if f.parent != HERE or not f.is_file():
                    return self._send(404, b"not found", "text/plain")
                self._send(200, f.read_bytes(), mimetypes.guess_type(f.name)[0] or "application/octet-stream")

            def do_POST(self):
                if self.path == "/key":  # Enter/Space on the page = Enter in the terminal
                    srv.keys.put("page")
                    return self._send(200, b"{}")
                if self.path == "/trace":  # whole contact: points while touching, duration (drag check)
                    d = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
                    d["t_end"] = time.time()
                    srv.traces.put(d)
                    return self._send(200, b"{}")
                if self.path != "/touch":
                    return self._send(404, b"{}")
                d = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
                srv.touches.put({"u": float(d["u"]), "v": float(d["v"]),
                                 "w": d.get("w"), "h": d.get("h"), "t": time.time()})
                srv.touched.set()
                self._send(200, b"{}")

        self.httpd = ThreadingHTTPServer(("0.0.0.0", port), Handler)
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()
        print(f"\nOn the touch screen, open:  http://localhost:{port}/   (full-screen, F11)")
        print("  (separate tablet? use this computer's network address instead of localhost)\n")

    def set(self, **kw):
        self.state.update(kw)

    def drain(self):
        self.touched.clear()
        while not self.touches.empty():
            self.touches.get_nowait()
        while not self.traces.empty():
            self.traces.get_nowait()

    def all_traces(self, wait=0.6):
        """Contacts that ended since drain(), each summarised: drag (how far the stylus slid while touching,
        px), contact time (ms), start/end point. Waits briefly for the page to report the lift-off."""
        end = time.time() + wait
        while self.traces.empty() and time.time() < end:
            time.sleep(0.02)
        out = []
        while not self.traces.empty():
            d = self.traces.get_nowait()
            pts = d.get("pts") or []
            w, h = d.get("w") or 1000, d.get("h") or 1000
            if pts:
                x0, y0 = pts[0][0] * w, pts[0][1] * h
                d["drag_px"] = round(max(((p[0] * w - x0) ** 2 + (p[1] * h - y0) ** 2) ** 0.5 for p in pts), 1)
                d["slide_px"] = round(((pts[-1][0] * w - x0) ** 2 + (pts[-1][1] * h - y0) ** 2) ** 0.5, 1)
            out.append(d)
        return out

    def all_touches(self):
        out = []
        while not self.touches.empty():
            out.append(self.touches.get_nowait())
        return out

    def stop(self):
        self.httpd.shutdown()


def save_log(kind, data):
    """Save a session log to touch/logs/<kind>_<date-time>.json and return the path."""
    LOG_DIR.mkdir(exist_ok=True)
    path = LOG_DIR / f"{kind}_{datetime.now():%Y%m%d_%H%M%S}.json"
    path.write_text(json.dumps(data, indent=1))
    return path
