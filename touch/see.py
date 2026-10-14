"""Look at the screen through the camera and produce a straightened screenshot.

The page on the screen must show ArUco markers 0-3 in its corners
(top-left, top-right, bottom-right, bottom-left), see touch/markers/.

  python see.py --camera /dev/video4
  -> outputs/screen.jpg         straightened view; pixel (x, y) here means
                                u = x / width, v = y / height  -> pass to tap.py
  -> outputs/camera_debug.jpg   raw camera frame with the detected markers drawn

Use from Python:
  from see import ScreenCamera
  cam = ScreenCamera("/dev/video4")
  img = cam.screen_image()          # numpy BGR image, or None if markers not found
"""

import argparse
import time
from pathlib import Path

import cv2
import numpy as np

DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
# Which corner of each marker is the outer corner of the screen frame
OUTER = {0: 0, 1: 1, 2: 2, 3: 3}  # TL marker -> its TL corner, TR -> TR, BR -> BR, BL -> BL


def find_screen(frame):
    """Return the 4 outer corners (TL, TR, BR, BL) in camera pixels, or None."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = cv2.aruco.ArucoDetector(DICT, cv2.aruco.DetectorParameters()).detectMarkers(gray)
    if ids is None:
        return None, []
    found = {int(i): c.reshape(4, 2) for i, c in zip(ids.flatten(), corners)}
    if not all(k in found for k in OUTER):
        return None, sorted(found)
    return np.array([found[k][OUTER[k]] for k in (0, 1, 2, 3)], np.float32), sorted(found)


def straighten(frame, quad, width, height):
    dst = np.array([[0, 0], [width, 0], [width, height], [0, height]], np.float32)
    return cv2.warpPerspective(frame, cv2.getPerspectiveTransform(quad, dst), (width, height))


class ScreenCamera:
    def __init__(self, camera, width=1280, height=800, cam_w=640, cam_h=480):
        src = int(camera) if str(camera).isdigit() else camera
        self.cap = cv2.VideoCapture(src)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam_w)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_h)
        self.width, self.height = width, height
        time.sleep(1)

    def frame(self):
        for _ in range(5):  # drop stale buffered frames
            ok, f = self.cap.read()
        if not ok:
            raise RuntimeError("No camera frame. Is the camera attached? (./lr attach)")
        return f

    def screen_image(self, tries=10):
        for _ in range(tries):
            f = self.frame()
            quad, _ = find_screen(f)
            if quad is not None:
                return straighten(f, quad, self.width, self.height)
        return None


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--camera", default="/dev/video4")
    p.add_argument("--image", help="use a saved photo instead of the camera")
    p.add_argument("--size", default="1280x800", help="output size, same aspect as the screen")
    p.add_argument("--out", default="outputs")
    a = p.parse_args()
    w, h = map(int, a.size.lower().split("x"))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    frame = cv2.imread(a.image) if a.image else ScreenCamera(a.camera).frame()
    quad, seen = find_screen(frame)
    dbg = frame.copy()
    if quad is not None:
        cv2.polylines(dbg, [quad.astype(int)], True, (0, 255, 0), 2)
    cv2.imwrite(str(out / "camera_debug.jpg"), dbg)
    if quad is None:
        raise SystemExit(f"Need markers 0,1,2,3, saw {seen}. Check {out/'camera_debug.jpg'}: "
                         "is the whole screen in view, bright enough, without glare?")
    cv2.imwrite(str(out / "screen.jpg"), straighten(frame, quad, w, h))
    print(f"Saved {out/'screen.jpg'} ({w}x{h}). A point at pixel (x, y) is u=x/{w}, v=y/{h}.")
