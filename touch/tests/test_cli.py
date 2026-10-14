"""`run_captcha.py --dry-run` as a real process: a saved screenshot as the "camera" and a local fake solver
server, so the real multipart POST, the credentials headers and the CLI wiring are exercised (no robot)."""

import json
import os
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2

from fakeworld import IMG, SHOTS, World

TOUCH = Path(__file__).resolve().parent.parent


def serve(reply, seen):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_POST(self):
            body = self.rfile.read(int(self.headers["Content-Length"]))
            seen.update(ua=self.headers.get("User-Agent", ""), path=self.path, acct=self.headers.get("X-CF-Account-Id"), token=self.headers.get("X-CF-Api-Token"),
                        ctype=self.headers.get("Content-Type", ""), body=body)
            data = json.dumps(reply).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def photo(tmp_path, name="grid_empty"):
    """A 'camera photo' of the screen: the shot with white all round, so the corner markers can be detected."""
    out = tmp_path / f"{name}_photo.png"
    cv2.imwrite(str(out), cv2.copyMakeBorder(IMG[name], 90, 90, 120, 120, cv2.BORDER_CONSTANT, value=(255, 255, 255)))
    return str(out)


def run(args, env_extra, tmp_path):
    env = {**os.environ, **env_extra}
    return subprocess.run([sys.executable, str(TOUCH / "run_captcha.py"), "--delay", "0", "--out", str(tmp_path / "run"),
                           "--layout", str(tmp_path / "none.json"), *args],
                          capture_output=True, text=True, env=env, timeout=120, cwd=str(tmp_path))


def test_dry_run_sends_the_photo_with_credentials_and_prints_the_plan(tmp_path):
    seen = {}
    srv = serve(World().presses((0, 4, 8)), seen)
    try:
        p = run(["--dry-run", "--camera", photo(tmp_path)],
                {"SOLVER_URL": f"http://127.0.0.1:{srv.server_port}", "CLOUDFLARE_ACCOUNT_ID": "acct123",
                 "CLOUDFLARE_API_TOKEN": "secret-token"}, tmp_path)
    finally:
        srv.shutdown()
    assert p.returncode == 0, p.stdout + p.stderr
    assert "DRY_RUN" in p.stdout and "tiles [0, 4, 8]" in p.stdout
    assert seen["path"] == "/api/robot/solve" and seen["acct"] == "acct123" and seen["token"] == "secret-token"
    assert "Python-urllib" not in seen["ua"] and seen["ua"]     # Cloudflare 403s urllib's default User-Agent
    assert seen["ctype"].startswith("multipart/form-data") and b"\xff\xd8" in seen["body"]
    assert b'name="screen_width"' in seen["body"] and b"1280" in seen["body"]
    assert "secret-token" not in p.stdout + p.stderr           # the token is never printed
    folders = list((tmp_path / "run").iterdir())
    assert any(f.is_dir() and (f / "solver.json").exists() for f in folders)


def test_missing_credentials_stop_early_with_a_clear_message(tmp_path):
    env = {k: v for k, v in os.environ.items() if not k.startswith("CLOUDFLARE")}
    p = subprocess.run([sys.executable, str(TOUCH / "run_captcha.py"), "--dry-run", "--delay", "0", "--camera",
                        photo(tmp_path), "--out", str(tmp_path / "r")],
                       capture_output=True, text=True, env={**env, "SOLVER_URL": "http://127.0.0.1:9"}, cwd=str(tmp_path))
    assert p.returncode != 0 and "CLOUDFLARE_ACCOUNT_ID" in (p.stdout + p.stderr)
