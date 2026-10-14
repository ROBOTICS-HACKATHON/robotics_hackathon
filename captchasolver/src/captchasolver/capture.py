"""Grab the screen as PNG bytes on Linux (X11/Wayland), macOS, Windows and WSL.

WSL is special: the screen you want is *Windows'*, not WSLg's, so we ask
powershell.exe to capture it (DPI-aware, so the PNG is in physical pixels).
Only the primary display is captured.
"""
from __future__ import annotations

import base64
import io
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


class CaptureError(RuntimeError):
    pass


def is_wsl() -> bool:
    if os.environ.get("WSL_DISTRO_NAME") or os.environ.get("WSL_INTEROP"):
        return True
    try:
        return "microsoft" in Path("/proc/version").read_text().lower()
    except OSError:
        return False


_PS_SCRIPT = r"""
$ProgressPreference = 'SilentlyContinue'
Add-Type -AssemblyName System.Windows.Forms, System.Drawing
Add-Type -TypeDefinition 'using System.Runtime.InteropServices; public class Dpi { [DllImport("user32.dll")] public static extern bool SetProcessDPIAware(); }'
[void][Dpi]::SetProcessDPIAware()
$b = [System.Windows.Forms.Screen]::PrimaryScreen.Bounds
$bmp = New-Object System.Drawing.Bitmap $b.Width, $b.Height
$g = [System.Drawing.Graphics]::FromImage($bmp)
$g.CopyFromScreen($b.Location, [System.Drawing.Point]::Empty, $b.Size)
$ms = New-Object System.IO.MemoryStream
$bmp.Save($ms, [System.Drawing.Imaging.ImageFormat]::Png)
[Console]::Out.Write([Convert]::ToBase64String($ms.ToArray()))
"""


def _wsl() -> bytes:
    exe = shutil.which("powershell.exe") or "/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe"
    encoded = base64.b64encode(_PS_SCRIPT.encode("utf-16-le")).decode()
    r = subprocess.run([exe, "-NoProfile", "-NonInteractive", "-EncodedCommand", encoded],
                       capture_output=True, text=True, timeout=30)
    out = r.stdout.strip()
    if r.returncode != 0 or not out:
        raise CaptureError(f"powershell.exe capture failed ({r.returncode}): {r.stderr[:300]}")
    return base64.b64decode(out)


def _cmd_to_png(argv: list[str], *, file_arg: bool = False) -> bytes:
    if file_arg:
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "shot.png"
            subprocess.run([a.replace("{file}", str(path)) for a in argv],
                           check=True, capture_output=True, timeout=30)
            return path.read_bytes()
    return subprocess.run(argv, check=True, capture_output=True, timeout=30).stdout


def _imagegrab() -> bytes:
    from PIL import ImageGrab
    if sys.platform == "win32":  # physical pixels, not DPI-scaled ones
        import ctypes
        ctypes.windll.user32.SetProcessDPIAware()
    img = ImageGrab.grab()
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


def grab_screen() -> bytes:
    """PNG bytes of the primary screen, using whatever this platform offers."""
    attempts: list[tuple[str, callable]] = []
    if is_wsl():
        attempts.append(("powershell.exe (Windows screen)", _wsl))
    elif sys.platform == "darwin":
        attempts.append(("screencapture", lambda: _cmd_to_png(
            ["screencapture", "-x", "-t", "png", "{file}"], file_arg=True)))
    elif sys.platform == "win32":
        attempts.append(("ImageGrab", _imagegrab))
    else:
        if os.environ.get("WAYLAND_DISPLAY"):
            if shutil.which("grim"):
                attempts.append(("grim", lambda: _cmd_to_png(["grim", "-"])))
            if shutil.which("gnome-screenshot"):
                attempts.append(("gnome-screenshot", lambda: _cmd_to_png(
                    ["gnome-screenshot", "-f", "{file}"], file_arg=True)))
            if shutil.which("spectacle"):
                attempts.append(("spectacle", lambda: _cmd_to_png(
                    ["spectacle", "-b", "-n", "-f", "-o", "{file}"], file_arg=True)))
        attempts.append(("ImageGrab (X11)", _imagegrab))

    errors = []
    for name, fn in attempts:
        try:
            data = fn()
            if data[:8] == b"\x89PNG\r\n\x1a\n":
                return data
            errors.append(f"{name}: output was not a PNG")
        except Exception as e:  # noqa: BLE001 - try the next method
            errors.append(f"{name}: {e}")
    raise CaptureError("could not capture the screen. Tried: " + "; ".join(errors) +
                       ". On Wayland install `grim`; on WSL make sure powershell.exe is reachable.")
