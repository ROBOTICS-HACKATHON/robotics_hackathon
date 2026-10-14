"""Photograph the screen, send it to the captcha solver, print where to press.

camera -> straightened screenshot (see.py) -> POST {SOLVER_URL}/api/robot/solve -> presses (tiles, then Verify)

  python solve.py --camera /dev/video4                 # needs SOLVER_URL, CLOUDFLARE_ACCOUNT_ID,
                                                      #       CLOUDFLARE_API_TOKEN in the environment
  python solve.py --image outputs/screen.jpg          # send a saved straightened screenshot instead

The solver answers in normalised screen coordinates (x_norm, y_norm), which are exactly the u, v the
tap system uses (both run between the markers' outer corners). This script only prints them, plus the
`./lr tap ...` command; it never taps (a second tap on a tile toggles it back, so no blind retries).

A refusal the server marks `retry: true` (no dialog found, arm in the way, ...) takes a new photo and
asks again, up to --tries times. Everything is saved in outputs/solve/: the photo sent, the JSON
answer and the photo with the presses drawn on it.
"""

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent))
from see import ScreenCamera  # noqa: E402


def post_image(url, jpg, fields, headers, timeout=120):
    """multipart/form-data POST with one image; returns (http status, parsed JSON or text)."""
    b = uuid.uuid4().hex
    parts = []
    for k, v in fields.items():
        parts.append(f'--{b}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode())
    parts.append(f'--{b}\r\nContent-Disposition: form-data; name="image"; filename="screen.jpg"\r\n'
                 "Content-Type: image/jpeg\r\n\r\n".encode() + jpg + b"\r\n")
    parts.append(f"--{b}--\r\n".encode())
    req = urllib.request.Request(url, data=b"".join(parts), method="POST",
                                 headers={**headers, "Content-Type": f"multipart/form-data; boundary={b}"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            status, body = r.status, r.read()
    except urllib.error.HTTPError as e:  # the robot endpoint reports its errors as JSON with 4xx/5xx
        status, body = e.code, e.read()
    try:
        return status, json.loads(body)
    except ValueError:
        return status, body.decode(errors="replace")[:500]


def draw(img, presses):
    out = img.copy()
    h, w = out.shape[:2]
    for p in presses:
        x, y = int(p["x_norm"] * w), int(p["y_norm"] * h)
        col = (0, 200, 0) if p["role"] == "verify" else (0, 0, 255)
        cv2.circle(out, (x, y), max(8, w // 60), col, 3)
        cv2.putText(out, str(p["seq"]), (x + 12, y - 12), cv2.FONT_HERSHEY_SIMPLEX, 1.0, col, 2)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--camera", default=os.environ.get("CAMERA", "/dev/video4"))
    p.add_argument("--cam-mode", default="auto")
    p.add_argument("--image", help="send this straightened screenshot instead of taking a photo")
    p.add_argument("--size", default="1280x800", help="straightened screenshot size, same aspect as the screen")
    p.add_argument("--url", default=os.environ.get("SOLVER_URL", ""), help="solver base URL (SOLVER_URL)")
    p.add_argument("--target", help="what to select, if you already know it (skips reading the banner)")
    p.add_argument("--model", help="Workers AI model id (default: the server's)")
    p.add_argument("--tries", type=int, default=3, help="new photos to try when the server says retry")
    p.add_argument("--delay", type=float, default=0, help="seconds to switch to the captcha page first")
    p.add_argument("--out", default=str(Path(__file__).resolve().parent.parent / "outputs" / "solve"))
    a = p.parse_args()

    url = a.url.rstrip("/")
    acct, token = os.environ.get("CLOUDFLARE_ACCOUNT_ID", ""), os.environ.get("CLOUDFLARE_API_TOKEN", "")
    missing = [n for n, v in (("SOLVER_URL", url), ("CLOUDFLARE_ACCOUNT_ID", acct),
                              ("CLOUDFLARE_API_TOKEN", token)) if not v]
    if missing:
        raise SystemExit(f"Set {', '.join(missing)} (in lr.conf for ./lr solve, or in the environment).")
    if not url.endswith("/api/robot/solve"):
        url += "/api/robot/solve"
    w, h = map(int, a.size.lower().split("x"))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    cam = None if a.image else ScreenCamera(a.camera, w, h, a.cam_mode)
    if cam is not None and cam.src.kind == "device":
        print(f"camera: {cam.src.describe()}", flush=True)
    if a.delay > 0 and cam is not None:
        print(f"Photo in {a.delay:.0f} s: switch to the captcha page (full-screen, F11)", flush=True)
        time.sleep(a.delay)

    fields = {"screen_width": w, "screen_height": h}
    if a.target:
        fields["target"] = a.target
    if a.model:
        fields["model"] = a.model
    headers = {"X-CF-Account-Id": acct, "X-CF-Api-Token": token, "Accept": "application/json"}

    stamp = time.strftime("%Y%m%d-%H%M%S")
    for attempt in range(1, a.tries + 1):
        if a.image:
            img = cv2.imread(a.image)
            if img is None:
                raise SystemExit(f"Can't read {a.image}")
        else:
            img = cam.screen_image(allow_remembered=False)
            if img is None:
                print(f"try {attempt}: markers 0-3 not all visible, taking another photo", flush=True)
                continue
        jpg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 92])[1].tobytes()
        cv2.imwrite(str(out / f"{stamp}-{attempt}-sent.jpg"), img)
        t0 = time.time()
        print(f"try {attempt}: sending {img.shape[1]}x{img.shape[0]} screenshot ({len(jpg) // 1024} KB) ...",
              flush=True)
        try:
            status, res = post_image(url, jpg, fields, headers)
        except Exception as e:
            print(f"try {attempt}: can't reach {url}: {e}", flush=True)
            continue
        (out / f"{stamp}-{attempt}.json").write_text(json.dumps(res, indent=2) if isinstance(res, dict) else res)
        if not isinstance(res, dict):
            print(f"try {attempt}: HTTP {status}, not JSON: {res[:200]}", flush=True)
            continue
        if res.get("ok"):
            presses = res["presses"]
            cv2.imwrite(str(out / f"{stamp}-{attempt}-presses.jpg"), draw(img, presses))
            print(f"target: {res['target']}   tiles: {res['tiles']}   ({time.time() - t0:.1f} s, {res.get('model')})")
            for pr in presses:
                print(f"  {pr['seq']}. {pr['role']:6s}" + (f" tile {pr['tile']}" if "tile" in pr else "       ")
                      + f"  u={pr['x_norm']:.4f} v={pr['y_norm']:.4f}")
            uv = " ".join(f"{pr['x_norm']:.4f} {pr['y_norm']:.4f}" for pr in presses)
            print(f"tap with:  ./lr tap {uv}")
            print(f"saved: {out}/{stamp}-{attempt}-presses.jpg")
            return
        print(f"try {attempt}: {res.get('error')}: {res.get('detail')}", flush=True)
        if not res.get("retry") or a.image:
            raise SystemExit(1)
    raise SystemExit(f"No answer after {a.tries} tries; photos and answers are in {out}")


if __name__ == "__main__":
    main()
