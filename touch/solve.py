"""Photograph the screen, send it to the captcha solver, print where to press.

camera -> straightened screenshot (see.py) -> POST {SOLVER_URL}/api/robot/solve -> presses (tiles, then Verify)

  python solve.py --camera /dev/video4                 # needs SOLVER_URL, CLOUDFLARE_ACCOUNT_ID,
                                                      #       CLOUDFLARE_API_TOKEN in the environment
  python solve.py --image outputs/screen.jpg          # send a saved straightened screenshot instead

The solver answers in normalised screen coordinates (x_norm, y_norm), which are exactly the u, v the
tap system uses (both run between the markers' outer corners). This script only prints them, plus the
`./lr tap ...` command; it never taps (a second tap on a tile toggles it back, so no blind retries).
`run_captcha.py` is the program that taps.

A refusal the server marks `retry: true` (no dialog found, arm in the way, ...) takes a new photo and
asks again, up to --tries times. Everything is saved in outputs/solve/: the photo sent, the JSON
answer and the photo with the presses drawn on it.

From Python (this is what run_captcha.py does):

    from solve import solver_config, solve_screen, SolverError
    cfg = solver_config()                      # reads SOLVER_URL + the Cloudflare credentials
    try:
        res = solve_screen(img_bgr, cfg)       # {"ok": True, "target", "tiles", "presses": [...], ...}
    except SolverError as e:
        e.error, e.detail, e.retry             # same codes as the server: no_dialog, cropped, ...
"""

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent))

ROBOT_PATH = "/api/robot/solve"


class SolverError(RuntimeError):
    """The solver refused or could not be reached. `error` is the server's code (no_dialog, cropped,
    layout_inconsistent, unreadable_target, no_matches, model_error, bad_credentials, ...) or one of ours
    (bad_config, network, bad_response). `retry` is True when a NEW photo may succeed."""

    def __init__(self, error, detail="", retry=False, status=None):
        super().__init__(f"{error}: {detail}" if detail else str(error))
        self.error, self.detail, self.retry, self.status = error, detail, retry, status


@dataclass
class SolverConfig:
    url: str          # full URL of the robot endpoint
    headers: dict     # Cloudflare credentials (sent on every call, never logged)
    timeout: float = 120


def solver_config(url=None, env=None):
    """Read SOLVER_URL / CLOUDFLARE_ACCOUNT_ID / CLOUDFLARE_API_TOKEN (or the given url). Raises SolverError."""
    env = os.environ if env is None else env
    base = (url or env.get("SOLVER_URL", "")).strip().rstrip("/")
    acct, token = env.get("CLOUDFLARE_ACCOUNT_ID", "").strip(), env.get("CLOUDFLARE_API_TOKEN", "").strip()
    missing = [n for n, v in (("SOLVER_URL", base), ("CLOUDFLARE_ACCOUNT_ID", acct),
                              ("CLOUDFLARE_API_TOKEN", token)) if not v]
    if missing:
        raise SolverError("bad_config", f"Set {', '.join(missing)} (in lr.conf for ./lr, or in the environment).")
    if not base.endswith(ROBOT_PATH):
        base += ROBOT_PATH
    return SolverConfig(base, {"X-CF-Account-Id": acct, "X-CF-Api-Token": token, "Accept": "application/json"})


def post_image(url, jpg, fields, headers, timeout=120):
    """multipart/form-data POST with one image; returns (http status, parsed JSON or text)."""
    b = uuid.uuid4().hex
    parts = []
    for k, v in fields.items():
        parts.append(f'--{b}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode())
    parts.append(f'--{b}\r\nContent-Disposition: form-data; name="image"; filename="screen.jpg"\r\n'
                 "Content-Type: image/jpeg\r\n\r\n".encode() + jpg + b"\r\n")
    parts.append(f"--{b}--\r\n".encode())
    # Cloudflare answers urllib's default "Python-urllib" User-Agent with a 403 (error 1010, "blocked based on your
    # browser's signature") before the request reaches the solver, so identify ourselves.
    req = urllib.request.Request(url, data=b"".join(parts), method="POST",
                                 headers={"User-Agent": "robot-arm-captcha/1.0 (+touch/solve.py)", **headers,
                                          "Content-Type": f"multipart/form-data; boundary={b}"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            status, body = r.status, r.read()
    except urllib.error.HTTPError as e:  # the robot endpoint reports its errors as JSON with 4xx/5xx
        status, body = e.code, e.read()
    try:
        return status, json.loads(body)
    except ValueError:
        return status, body.decode(errors="replace")[:500]


def check_presses(res):
    """The robot never taps from a half-valid answer: tiles then exactly one Verify, all inside the screen."""
    presses = res.get("presses")
    if not isinstance(presses, list) or len(presses) < 2:
        raise SolverError("bad_response", "answer has no presses (need at least one tile and Verify)", retry=True)
    try:
        roles = [p["role"] for p in presses]
        pts = [(float(p["x_norm"]), float(p["y_norm"])) for p in presses]
    except (KeyError, TypeError, ValueError) as e:
        raise SolverError("bad_response", f"malformed press: {e!r}", retry=True)
    if roles[-1] != "verify" or roles[:-1].count("verify") or any(r != "tile" for r in roles[:-1]):
        raise SolverError("bad_response", f"presses must be tiles then one verify, got {roles}", retry=True)
    if any(not (0 <= u <= 1 and 0 <= v <= 1) for u, v in pts):
        raise SolverError("bad_response", "a press lies outside the screen", retry=True)
    tiles = [p.get("tile") for p in presses[:-1]]
    if any(t is None for t in tiles) or len(set(tiles)) != len(tiles):
        raise SolverError("bad_response", f"tile numbers missing or repeated: {tiles}", retry=True)


def solve_screen(img, cfg, target=None, model=None, size=None):
    """Send a straightened screenshot (BGR numpy) to the robot endpoint. Returns the parsed answer
    (ok == True, validated) or raises SolverError."""
    h, w = img.shape[:2]
    if size:
        w, h = size
    fields = {"screen_width": w, "screen_height": h}
    if target:
        fields["target"] = target
    if model:
        fields["model"] = model
    jpg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 92])[1].tobytes()
    try:
        status, res = post_image(cfg.url, jpg, fields, cfg.headers, cfg.timeout)
    except Exception as e:  # DNS, TLS, timeout...
        raise SolverError("network", f"can't reach {cfg.url}: {e}", retry=True)
    if not isinstance(res, dict):
        raise SolverError("bad_response", f"HTTP {status}, not JSON: {str(res)[:200]}", retry=True, status=status)
    if not res.get("ok"):
        raise SolverError(res.get("error") or "unknown", str(res.get("detail") or ""),
                          retry=bool(res.get("retry")), status=status)
    check_presses(res)
    return res


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
    from see import ScreenCamera

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

    try:
        cfg = solver_config(a.url)
    except SolverError as e:
        raise SystemExit(e.detail)
    w, h = map(int, a.size.lower().split("x"))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    cam = None if a.image else ScreenCamera(a.camera, w, h, a.cam_mode)
    if cam is not None and cam.src.kind == "device":
        print(f"camera: {cam.src.describe()}", flush=True)
    if a.delay > 0 and cam is not None:
        print(f"Photo in {a.delay:.0f} s: switch to the captcha page (full-screen, F11)", flush=True)
        time.sleep(a.delay)

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
        cv2.imwrite(str(out / f"{stamp}-{attempt}-sent.jpg"), img)
        t0 = time.time()
        print(f"try {attempt}: sending {img.shape[1]}x{img.shape[0]} screenshot ...", flush=True)
        try:
            res = solve_screen(img, cfg, a.target, a.model, (w, h))
        except SolverError as e:
            print(f"try {attempt}: {e.error}: {e.detail}", flush=True)
            if not e.retry or a.image:
                raise SystemExit(1)
            continue
        (out / f"{stamp}-{attempt}.json").write_text(json.dumps(res, indent=2))
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
    raise SystemExit(f"No answer after {a.tries} tries; photos and answers are in {out}")


if __name__ == "__main__":
    main()
