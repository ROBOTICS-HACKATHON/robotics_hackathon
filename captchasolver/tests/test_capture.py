import base64
import io

import pytest
from PIL import Image

from captchasolver import capture


def png_bytes():
    buf = io.BytesIO()
    Image.new("RGB", (4, 4), "red").save(buf, "PNG")
    return buf.getvalue()


def test_wsl_uses_powershell(monkeypatch):
    monkeypatch.setattr(capture, "is_wsl", lambda: True)
    seen = {}

    class R:
        returncode, stderr = 0, ""
        stdout = base64.b64encode(png_bytes()).decode() + "\r\n"

    def fake_run(argv, **kw):
        seen["argv"] = argv
        return R()
    monkeypatch.setattr(capture.subprocess, "run", fake_run)
    assert capture.grab_screen() == png_bytes()
    assert "-EncodedCommand" in seen["argv"]
    # the script we send must be DPI-aware, or Windows scaling gives wrong pixels
    script = base64.b64decode(seen["argv"][-1]).decode("utf-16-le")
    assert "SetProcessDPIAware" in script and "PrimaryScreen" in script


def test_wayland_prefers_grim(monkeypatch):
    monkeypatch.setattr(capture, "is_wsl", lambda: False)
    monkeypatch.setattr(capture.sys, "platform", "linux")
    monkeypatch.setenv("WAYLAND_DISPLAY", "wayland-1")
    monkeypatch.setattr(capture.shutil, "which", lambda n: "/usr/bin/grim" if n == "grim" else None)
    monkeypatch.setattr(capture, "_cmd_to_png", lambda argv, **kw: png_bytes())
    assert capture.grab_screen() == png_bytes()


def test_all_methods_failing_gives_helpful_error(monkeypatch):
    monkeypatch.setattr(capture, "is_wsl", lambda: False)
    monkeypatch.setattr(capture.sys, "platform", "linux")
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    monkeypatch.setattr(capture, "_imagegrab", lambda: (_ for _ in ()).throw(OSError("no display")))
    with pytest.raises(capture.CaptureError, match="no display"):
        capture.grab_screen()
