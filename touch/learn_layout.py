"""Teach the robot the three pages around the image grid:  ./lr captcha --learn-layout

For each page (choice, checkbox, verified) you show it on the touch screen, the camera takes a straightened
photo, and you click on that photo where the robot has to tap (the reCAPTCHA card, the "I'm not a robot"
checkbox, the "Start again" button). The photo's tiny signature is stored too, so the robot can tell the
pages apart later. Result: touch/site_layout.json (machine-specific, git-ignored).

The click happens in a browser: open http://localhost:8765/ (the container publishes the port).
"""

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2

from pagestate import PAGES, TAP_PAGES, SiteLayout

HOWTO = {
    "choice": "the first page: 'Access website' with the three cards (reCAPTCHA / GeeTest / Turnstile)",
    "checkbox": "the 'I'm not a robot' box (click the reCAPTCHA card by hand to get there)",
    "start": "the Training page again (same as 'choice'): this time the robot learns where 'Start session' is, "
             "so ./lr captcha-train can start sessions by itself",
    "verified": "the 'Access granted' page (solve one by hand; faster: Training settings -> Practice mode "
                "'Single tile' -> click the highlighted tile)",
}

PAGE = """<!doctype html><meta charset=utf-8><title>Teach the robot</title>
<style>body{font:16px system-ui;margin:16px;background:#fafafa}#w{position:relative;display:inline-block;cursor:crosshair}
img{max-width:96vw;max-height:78vh;display:block;border:2px solid #333}
#x{position:absolute;width:26px;height:26px;margin:-13px 0 0 -13px;border:3px solid #e00;border-radius:50%;display:none;pointer-events:none}
button{font:inherit;padding:8px 18px;margin-right:8px}</style>
<h2 id=t>...</h2><p id=m></p>
<div id=w><img id=i src=/img.png><div id=x></div></div>
<p><button id=ok disabled>Use this point</button> <span id=s></span></p>
<script>
let u=null,v=null;const i=document.getElementById('i'),x=document.getElementById('x');
fetch('/info').then(r=>r.json()).then(d=>{t.textContent=d.title;m.textContent=d.msg});
i.onclick=e=>{const r=i.getBoundingClientRect();u=(e.clientX-r.left)/r.width;v=(e.clientY-r.top)/r.height;
 x.style.display='block';x.style.left=(e.clientX-r.left+2)+'px';x.style.top=(e.clientY-r.top+2)+'px';ok.disabled=false;
 s.textContent='u='+u.toFixed(3)+'  v='+v.toFixed(3)};
ok.onclick=()=>fetch('/pick',{method:'POST',body:JSON.stringify({u,v})}).then(()=>{s.textContent='Saved. Back to the terminal.';ok.disabled=true});
</script>"""


def pick_point(img, title, msg, port=8765, timeout=600):
    """Show `img` in the browser, return the (u, v) the user clicks and confirms."""
    jpg = cv2.imencode(".png", img)[1].tobytes()
    info = json.dumps({"title": title, "msg": msg}).encode()
    got = {}
    done = threading.Event()

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, body, ctype):
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            p = self.path.split("?")[0]
            if p == "/img.png":
                return self._send(jpg, "image/png")
            if p == "/info":
                return self._send(info, "application/json")
            self._send(PAGE.encode(), "text/html")

        def do_POST(self):
            d = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            got["uv"] = (min(1.0, max(0.0, float(d["u"]))), min(1.0, max(0.0, float(d["v"]))))
            self._send(b"{}", "application/json")
            done.set()

    srv = ThreadingHTTPServer(("0.0.0.0", port), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    print(f"  Open http://localhost:{port}/ in a browser and click the target, then 'Use this point'.", flush=True)
    try:
        if not done.wait(timeout):
            raise SystemExit("Nobody clicked the point in time.")
    finally:
        srv.shutdown()
        srv.server_close()
    return got["uv"]


def _countdown(seconds, what):
    if seconds <= 0:
        return
    print(f"Photo in {seconds:.0f} s: {what}", flush=True)
    end = time.time() + seconds
    while (left := end - time.time()) > 0:
        print(f"  {left:3.0f} ...", end="\r", flush=True)
        time.sleep(min(1, left))
    print("  click!   ", flush=True)


def learn(cam, layout_path, out_dir, delay=5, port=8765, only=None):
    """Teach every page (or just `only`). `cam` is a ScreenCamera. Keeps pages taught earlier."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    layout = SiteLayout.load(layout_path)
    for name in (only or PAGES):
        print(f"\n== {name}: show {HOWTO[name]} on the touch screen, full-screen, 'Camera markers' ticked.")
        input("   Press Enter when it is showing ... ")
        _countdown(delay, "switch to the page on the touch screen")
        img = cam.screen_image(allow_remembered=False)
        if img is None:
            raise SystemExit("Can't see all 4 markers. Tick 'Camera markers' on the site, check the camera "
                             "(./lr cam-test), then run --learn-layout again.")
        cv2.imwrite(str(out_dir / f"layout_{name}.png"), img)
        if name == "start":
            if "choice" not in layout.pages:
                raise SystemExit("Teach the 'choice' page first (./lr captcha --learn-layout --only choice).")
            u, v = pick_point(img, "Click the 'Start session' button",
                              "The photo is what the robot sees. Click the centre of 'Start session'.", port)
            layout.pages["choice"]["start"] = [round(u, 5), round(v, 5)]
            layout.save()
            print(f"   saved start: 'Start session' at u={u:.3f} v={v:.3f}")
            continue
        u, v = pick_point(img, f"Click {TAP_PAGES[name]}",
                          "The photo is what the robot sees. Click the centre of the thing it must tap.", port)
        mark = img.copy()
        cv2.drawMarker(mark, (int(u * img.shape[1]), int(v * img.shape[0])), (0, 0, 255), cv2.MARKER_CROSS, 40, 3)
        cv2.imwrite(str(out_dir / f"layout_{name}_tap.png"), mark)
        layout.teach(name, img, (u, v))
        layout.save()
        print(f"   saved {name}: tap at u={u:.3f} v={v:.3f}  (check {out_dir / f'layout_{name}_tap.png'})")
    missing = layout.missing()
    print("\nAll pages taught." if not missing else f"\nStill missing: {', '.join(missing)}")
    return layout
