"""Look at the screen through ANY camera and produce a straightened screenshot.

The page on the screen shows ArUco markers 0-3 in its corners (markers.js). Because
the screen is found from those markers in every frame, the camera can be any device,
anywhere, at any angle or resolution: nothing to calibrate, swap cameras freely.

Camera sources (--camera, or CAMERA= in lr.conf):
  /dev/video4  or  0               USB webcam / RealSense (attached with ./lr attach)
  ipwebcam:192.168.1.23            phone with the "IP Webcam" app (Android)
  droidcam:192.168.1.23            phone with the "DroidCam" app (Android / iOS)
  http://192.168.1.23:8080/shot.jpg   any snapshot URL (a fresh JPEG per request)
  http://.../video   rtsp://...    any MJPEG / RTSP stream (/video and .mjpg are read in the background: fast)
  photo.jpg                        a saved image (for testing)

  python see.py --camera ipwebcam:192.168.1.23
  -> outputs/screen.jpg         straightened view; pixel (x, y) here means
                                u = x / width, v = y / height  -> pass to tap.py
  -> outputs/camera_debug.jpg   raw camera frame with the detected markers drawn
  python see.py --camera ... --check     test a camera: resolution, speed, markers

Use from Python:
  from see import ScreenCamera
  cam = ScreenCamera("ipwebcam:192.168.1.23")
  img = cam.screen_image()          # numpy BGR image, or None if the screen isn't found
"""

import argparse
import json
import re
import threading
import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np

DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
_params = cv2.aruco.DetectorParameters()
_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
DETECTOR = cv2.aruco.ArucoDetector(DICT, _params)
# Which corner of each marker is the outer corner of the screen frame
OUTER = {0: 0, 1: 1, 2: 2, 3: 3}  # TL marker -> its TL corner, TR -> TR, BR -> BR, BL -> BL
DETECT_MAX = 1600  # big phone frames are shrunk for finding markers (faster), corners scaled back


def resolve(spec):
    """Turn a short camera spec into something OpenCV / urllib can open."""
    s = str(spec).strip()
    if s.startswith("ipwebcam:"):  # continuous MJPEG stream: much faster than one /shot.jpg per frame
        host = s.split(":", 1)[1]
        return f"http://{host}{'' if ':' in host else ':8080'}/video"
    if s.startswith("droidcam:"):
        host = s.split(":", 1)[1]
        return f"http://{host}{'' if ':' in host else ':4747'}/video"
    return s


ROT_FILE = Path(__file__).resolve().parent.parent / "outputs" / "cam_rotation.json"


def _rotated():
    try:
        return set(json.loads(ROT_FILE.read_text()))
    except Exception:
        return set()


class FrameSource:
    """One interface for every kind of camera: .read() -> BGR image."""

    def __init__(self, spec, want_w=None, want_h=None):
        self.spec = resolve(spec)
        s = self.spec
        self.kind = None
        self.rot180 = self.spec in _rotated()  # camera mounted upside down: turned in software
        self.fps = 0.0  # pictures per second the camera actually delivers (streams)
        # IP Webcam can be remote-controlled (resolution, quality) through its web API
        self.phone_api = self.spec.rsplit("/", 1)[0] if str(spec).strip().startswith("ipwebcam:") else None
        if s.isdigit() or s.startswith("/dev/"):
            self.kind = "device"
            self.cap = cv2.VideoCapture(int(s) if s.isdigit() else s)
            if want_w:
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, want_w)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, want_h)
            time.sleep(1)
        elif s.startswith(("http://", "https://")) and s.lower().split("?")[0].endswith((".jpg", ".jpeg", ".png")):
            self.kind = "snapshot"  # a fresh picture per request: never stale
        elif s.startswith(("http://", "https://")) and s.split("?")[0].rstrip("/").endswith(("/video", ".mjpg", ".mjpeg")):
            self.kind = "mjpeg"  # phone apps: read in the background, always keep the newest picture
            self._jpg, self._t, self._err = None, 0.0, None
            self._cond = threading.Condition()
            threading.Thread(target=self._mjpeg_loop, daemon=True).start()
        elif s.startswith(("http://", "https://", "rtsp://")):
            self.kind = "stream"
            self.cap = cv2.VideoCapture(s)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        elif Path(s).is_file():
            self.kind = "file"
        else:
            raise SystemExit(f"Don't know how to open camera '{spec}'. Examples: /dev/video4, "
                             "ipwebcam:192.168.1.23, droidcam:192.168.1.23, http://.../shot.jpg")

    def _mjpeg_loop(self):
        while True:
            try:
                r = urllib.request.urlopen(self.spec, timeout=5)
                buf = b""
                while True:
                    chunk = r.read1(65536)  # whatever has arrived, no waiting for a full block
                    if not chunk:
                        break
                    buf += chunk
                    jpg = None
                    while True:  # cut out every complete frame, keep the newest
                        start = buf.find(b"\xff\xd8")
                        if start < 0:
                            buf = buf[-1024:]  # may hold the next part's header
                            break
                        m = None
                        for m in re.finditer(rb"Content-Length:\s*(\d+)", buf[max(0, start - 400):start], re.I):
                            pass
                        if m:
                            end = start + int(m.group(1))
                            if len(buf) < end:
                                break
                        else:  # no length given: the frame ends where the next part starts
                            nxt = buf.find(b"\r\n--", start)
                            if nxt < 0:
                                buf = buf[-8_000_000:]
                                break
                            end = buf.rfind(b"\xff\xd9", start, nxt) + 2
                            if end < 2:
                                buf = buf[nxt:]
                                continue
                        jpg, buf = buf[start:end], buf[end:]
                    if jpg:
                        now = time.time()
                        self._stamps = [t for t in getattr(self, "_stamps", []) if now - t < 2] + [now]
                        self.fps = (len(self._stamps) - 1) / max(now - self._stamps[0], 1e-3)
                        with self._cond:
                            self._jpg, self._t, self._err = jpg, now, None
                            self._cond.notify_all()
            except Exception as e:
                with self._cond:
                    self._err = e
                    self._cond.notify_all()
                time.sleep(1)

    def read(self):
        if self.kind == "mjpeg":
            # wait for a picture taken after this call, so a moving arm is never seen late
            t0, deadline = time.time(), time.time() + 5
            with self._cond:
                while self._t < t0 and time.time() < deadline:
                    if self._err is not None and self._jpg is None:  # never got a picture: phone unreachable
                        break
                    self._cond.wait(0.2)
                data, err = (self._jpg if self._t >= t0 else None), self._err
            if data is None:
                raise RuntimeError(f"No picture from the phone at {self.spec}" + (f": {err}" if err else "")
                                   + ". Same Wi-Fi? App running? Address right?")
            f = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        elif self.kind == "snapshot":
            try:
                data = urllib.request.urlopen(self.spec, timeout=5).read()
            except Exception as e:
                raise RuntimeError(f"Can't reach the phone camera at {self.spec}: {e}. "
                                   "Same Wi-Fi? App running? Address right?")
            f = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        elif self.kind == "file":
            f = cv2.imread(self.spec)
        else:
            ok, f = False, None
            for _ in range(5 if self.kind == "device" else 3):  # drop stale buffered frames
                ok, f = self.cap.read()
            if not ok:
                raise RuntimeError(f"No frame from {self.spec}. " + (
                    "Is the camera attached? (./lr attach)" if self.kind == "device"
                    else "Same Wi-Fi? Is the camera app streaming?"))
        if f is None:
            raise RuntimeError(f"Got no image from {self.spec}")
        return cv2.rotate(f, cv2.ROTATE_180) if self.rot180 else f

    def set_rot180(self, on):
        """Turn this camera's picture upside down (remembered for every command)."""
        self.rot180 = bool(on)
        r = _rotated()
        (r.add if on else r.discard)(self.spec)
        ROT_FILE.parent.mkdir(exist_ok=True)
        ROT_FILE.write_text(json.dumps(sorted(r)))

    def phone_set(self, setting, value):
        """Change an IP Webcam setting, e.g. ('video_size', '1280x720') or ('quality', 50).
        Returns None if it worked, else the reason."""
        if not self.phone_api:
            return "not an IP Webcam phone"
        try:
            urllib.request.urlopen(f"{self.phone_api}/settings/{setting}?set={value}", timeout=3).read()
            return None
        except Exception as e:
            return str(e)

    def close(self):
        if getattr(self, "cap", None) is not None:
            self.cap.release()


def find_screen(frame):
    """Return the 4 outer corners (TL, TR, BR, BL) in camera pixels, or None; and the marker ids seen."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    scale = min(1.0, DETECT_MAX / max(gray.shape))
    small = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else gray
    corners, ids, _ = DETECTOR.detectMarkers(small)
    if ids is None:
        return None, []
    found = {int(i): c.reshape(4, 2) / scale for i, c in zip(ids.flatten(), corners)}
    if not all(k in found for k in OUTER):
        return None, sorted(found)
    return np.array([found[k][OUTER[k]] for k in (0, 1, 2, 3)], np.float32), sorted(found)


def straighten(frame, quad, width, height):
    dst = np.array([[0, 0], [width, 0], [width, height], [0, height]], np.float32)
    return cv2.warpPerspective(frame, cv2.getPerspectiveTransform(quad, dst), (width, height),
                               flags=cv2.INTER_AREA)


class ScreenCamera:
    """Finds the screen in any camera. If the arm hides a marker, it reuses the last
    position it saw (as long as the camera hasn't been moved since)."""

    CACHE = Path(__file__).resolve().parent.parent / "outputs" / "last_screen.json"

    def __init__(self, camera, width=1280, height=800, cam_w=None, cam_h=None):
        self.src = FrameSource(camera, cam_w, cam_h)
        self.width, self.height = width, height
        self.last_quad = None
        try:
            c = json.loads(self.CACHE.read_text())
            if c.get("camera") == self.src.spec:
                self.last_quad = (np.array(c["quad"], np.float32), tuple(c["shape"]))
        except Exception:
            pass

    def frame(self):
        return self.src.read()

    def locate(self, f):
        """Screen corners in frame f: freshly detected, or the remembered ones. Returns (quad, fresh)."""
        quad, _ = find_screen(f)
        if quad is not None:
            self.last_quad = (quad, f.shape[:2])
            try:
                self.CACHE.parent.mkdir(exist_ok=True)
                self.CACHE.write_text(json.dumps({"camera": self.src.spec, "quad": quad.tolist(),
                                                  "shape": list(f.shape[:2])}))
            except Exception:
                pass
            return quad, True
        if self.last_quad is not None and tuple(self.last_quad[1]) == f.shape[:2]:
            return self.last_quad[0], False
        return None, False

    def screen_image(self, tries=10, allow_remembered=True):
        for _ in range(tries):
            f = self.frame()
            quad, fresh = self.locate(f)
            if quad is not None and (fresh or allow_remembered):
                return straighten(f, quad, self.width, self.height)
        return None


def check(cam, out):
    """Camera test: resolution, speed, markers."""
    t0 = time.time()
    frames = [cam.frame() for _ in range(5)]
    dt = (time.time() - t0) / 5
    f = frames[-1]
    quad, seen = find_screen(f)
    print(f"camera: {cam.src.spec}  ({cam.src.kind})")
    print(f"resolution: {f.shape[1]}x{f.shape[0]}, {1 / dt:.1f} frames/s as read here")
    print(f"markers seen: {seen or 'none'}  -> " + ("screen FOUND" if quad is not None else "screen NOT found"))
    if quad is not None:
        side = np.mean([np.linalg.norm(quad[i] - quad[(i + 1) % 4]) for i in range(4)])
        print(f"screen is ~{side:.0f}px across in the camera image"
              + ("  (small: move the camera closer or zoom in)" if side < 400 else ""))
    cv2.imwrite(str(out / "camera_debug.jpg"), f)
    print(f"raw frame: {out / 'camera_debug.jpg'}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--camera", default="/dev/video4")
    p.add_argument("--image", help="use a saved photo instead of the camera")
    p.add_argument("--size", default="1280x800", help="output size, same aspect as the screen")
    p.add_argument("--out", default="outputs")
    p.add_argument("--check", action="store_true", help="test the camera: resolution, speed, markers")
    p.add_argument("--delay", type=float, default=0, help="seconds to switch to the page before the picture")
    a = p.parse_args()
    w, h = map(int, a.size.lower().split("x"))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    cam = ScreenCamera(a.image or a.camera, w, h)
    if a.delay > 0 and not a.image:
        print(f"Picture in {a.delay:.0f} s: switch to the page on the screen (full-screen, F11)", flush=True)
        end = time.time() + a.delay
        while (left := end - time.time()) > 0:
            print(f"  {left:3.0f} ...", end="\r", flush=True)
            time.sleep(min(1, left))
        print("  click!   ", flush=True)
    if a.check:
        check(cam, out)
        raise SystemExit
    frame = cam.frame()
    _, seen = find_screen(frame)
    quad, fresh = cam.locate(frame)  # remembers good positions, falls back to them when a marker is hidden
    remembered = quad is not None and not fresh
    dbg = frame.copy()
    if quad is not None:
        cv2.polylines(dbg, [quad.astype(int)], True, (0, 255, 0) if not remembered else (0, 165, 255),
                      max(2, frame.shape[1] // 400))
    cv2.imwrite(str(out / "camera_debug.jpg"), dbg)
    if quad is None:
        raise SystemExit(f"Need markers 0,1,2,3, saw {seen}. Check {out/'camera_debug.jpg'}: "
                         "is the whole screen in view, bright enough, without glare?")
    if remembered:
        print(f"Only saw markers {seen}: using the screen position remembered from the last good frame "
              "(fine if the camera hasn't moved).")
    cv2.imwrite(str(out / "screen.jpg"), straighten(frame, quad, w, h))
    print(f"Saved {out/'screen.jpg'} ({w}x{h}). A point at pixel (x, y) is u=x/{w}, v=y/{h}.")
