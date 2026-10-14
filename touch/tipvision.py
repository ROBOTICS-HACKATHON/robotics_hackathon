"""Seeing the stylus tip with 0, 1 or 2 cameras.

Put a small bright sticker (pink, green, orange, yellow or blue) near the stylus tip.
Each camera finds the screen (corner markers) and the sticker, and projects the
sticker onto the screen: "where on the screen does this camera see the tip".

  * touching the glass  -> every camera projects it to the same spot (the real one)
  * hovering above it   -> each camera's projection is shifted along its viewing
                           direction (parallax); the higher, the bigger the shift

Before a tap the arm hovers just above the target, looks, and corrects its aim.
The parallax shift of each camera is LEARNED from measured taps (hover view vs.
where the touch really landed), so no camera calibration is needed:
  * 1 camera : landing = seen spot - learned shift
  * 2 cameras: the two shifts also give the HEIGHT (relative: 1.0 = normal hover),
               and the landing estimate no longer assumes a fixed hover height
  * 0 cameras: everything here is simply switched off.
Until enough taps have been learned (MIN_SAMPLES), it only collects data and
never moves the arm, so it can't make taps worse at the start.

  python tipvision.py --cameras /dev/video4,ipwebcam:192.168.1.23 --color pink
      -> camera check: viewing angles, tip found?, pair good for height?
"""

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

from see import ScreenCamera

HERE = Path(__file__).resolve().parent
MODEL_FILE = HERE / "tip_model.json"
MIN_SAMPLES = 5

# HSV ranges (OpenCV: H 0-179). Several ranges per color are OR-ed.
COLORS = {
    "pink": [((140, 70, 90), (179, 255, 255)), ((0, 120, 120), (4, 255, 255))],
    "green": [((40, 80, 70), (85, 255, 255))],
    "orange": [((5, 130, 120), (21, 255, 255))],
    "yellow": [((22, 110, 120), (36, 255, 255))],
    "blue": [((95, 120, 70), (128, 255, 255))],
}
UNIT = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], np.float32)


def parse_cameras(spec):
    """'none' / '' -> [];  'a,b' -> ['a','b'] (max 2)."""
    if not spec or str(spec).strip().lower() in ("none", "off", "0cam"):
        return []
    return [c.strip() for c in str(spec).split(",") if c.strip() and c.strip().lower() != "none"][:2]


def find_tip(frame, color, quad=None, min_area=12):
    """Pixel centre of the biggest blob of `color` (near the screen). Returns ((x, y), area) or None."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = np.zeros(hsv.shape[:2], np.uint8)
    for lo, hi in COLORS[color]:
        mask |= cv2.inRange(hsv, np.array(lo, np.uint8), np.array(hi, np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    if quad is not None:  # only look on/near the screen (the tip can stick out a bit above it)
        c = quad.mean(0)
        grown = (c + (quad - c) * 1.25).astype(np.int32)
        roi = np.zeros_like(mask)
        cv2.fillPoly(roi, [grown], 255)
        mask &= roi
    n, _, stats, cents = cv2.connectedComponentsWithStats(mask)
    if n <= 1:
        return None
    i = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    area = int(stats[i, cv2.CC_STAT_AREA])
    return (tuple(cents[i]), area) if area >= min_area else None


def view_geometry(quad, shape, aspect=1.5):
    """Rough camera position relative to the screen (screen width = 1), from the marker corners.
    Assumes a typical lens (focal length ~ image size). Good enough to judge viewing angles."""
    h, w = shape[:2]
    f = float(max(w, h))
    K = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]], float)
    obj = np.array([[0, 0, 0], [1, 0, 0], [1, 1 / aspect, 0], [0, 1 / aspect, 0]], float)
    ok, rvec, tvec = cv2.solvePnP(obj, quad.astype(float), K, None, flags=cv2.SOLVEPNP_IPPE)
    if not ok:
        return None
    R, _ = cv2.Rodrigues(rvec)
    cam = (-R.T @ tvec).ravel()  # camera centre in screen coordinates (z = height above the glass)
    centre = np.array([0.5, 0.5 / aspect, 0])
    d = cam - centre
    tilt = float(np.degrees(np.arccos(abs(d[2]) / np.linalg.norm(d))))
    return {"pos": cam.tolist(), "dir": (d / np.linalg.norm(d)).tolist(), "tilt_deg": tilt}


class TipVision:
    def __init__(self, cameras, color="pink", cid=None):
        self.color = color
        self.specs = parse_cameras(cameras) if isinstance(cameras, str) else list(cameras or [])
        self.cams = []
        ok_specs = []
        for s in self.specs:
            try:
                cam = ScreenCamera(s)
                cam.frame()  # make sure it really delivers pictures
                self.cams.append(cam)
                ok_specs.append(s)
            except (SystemExit, Exception) as e:
                print(f"(camera '{s}' unavailable, continuing without it: {e})")
        self.specs = ok_specs
        self.cid = cid
        allm = json.loads(MODEL_FILE.read_text()) if MODEL_FILE.exists() else {}
        self.data = allm.get(self.key(), {"samples": []})
        self._fit()

    def key(self):
        return f"{self.cid}|{'|'.join(self.specs)}"

    @property
    def n(self):
        return len(self.cams)

    @property
    def enabled(self):
        return self.n > 0

    @property
    def ready(self):
        """Enough learned taps to trust the estimate for correcting the arm."""
        return self.enabled and len(self.data["samples"]) >= MIN_SAMPLES

    # ---- seeing ----
    def observe(self):
        """One look from every camera. List (per camera) of {'uv','px','area'} or None."""
        out = []
        for cam in self.cams:
            try:
                f = cam.frame()
                quad, _ = cam.locate(f)
                tip = find_tip(f, self.color, quad) if quad is not None else None
                if tip is None:
                    out.append(None)
                    continue
                (x, y), area = tip
                H = cv2.getPerspectiveTransform(quad, UNIT)
                uv = cv2.perspectiveTransform(np.array([[[x, y]]], np.float32), H)[0, 0]
                out.append({"uv": [float(uv[0]), float(uv[1])], "px": [float(x), float(y)], "area": area})
            except Exception as e:
                print(f"(camera read failed: {e})")
                out.append(None)
        return out

    # ---- estimating ----
    def _offset(self, i, uv):
        """Learned parallax shift of camera i at screen point uv (for the normal hover height)."""
        m = self.off.get(i)
        if m is None:
            return None
        f = np.array([1.0, uv[0], uv[1]])
        return f @ m if m.ndim == 2 else m

    def estimate(self, obs):
        """{'landing': (u,v) or None, 'height': relative or None, 'cams': n seen}."""
        seen = [(i, np.array(o["uv"])) for i, o in enumerate(obs) if o]
        res = {"landing": None, "height": None, "cams": len(seen)}
        if not seen:
            return res
        offs = {i: self._offset(i, p) for i, p in seen}
        if len(seen) == 2 and all(o is not None for o in offs.values()):
            (a, pa), (b, pb) = seen
            g = offs[a] - offs[b]
            if np.dot(g, g) > 1e-8:
                h = float(np.dot(pa - pb, g) / np.dot(g, g))  # 1.0 = normal hover, 0 = touching
                land = 0.5 * ((pa - h * offs[a]) + (pb - h * offs[b]))
                res.update(landing=tuple(land), height=h)
                return res
        # one camera (or two without a usable pair model): assume the normal hover height
        ests = [p - offs[i] if offs[i] is not None else p for i, p in seen]
        res["landing"] = tuple(np.mean(ests, 0))
        if len(seen) == 2:
            res["disparity"] = float(np.linalg.norm(seen[0][1] - seen[1][1]))
        return res

    # ---- learning ----
    def learn(self, obs, landed):
        """A hover observation and where that tap really landed (from the page)."""
        if not obs or not any(obs):
            return
        self.data["samples"].append({"obs": obs, "landed": [float(landed[0]), float(landed[1])], "t": time.time()})
        self._fit()

    def _fit(self):
        self.off = {}
        S = self.data["samples"]
        for i in range(max(self.n, 2)):
            rows = [(s["landed"], np.array(s["obs"][i]["uv"]) - s["landed"])
                    for s in S if i < len(s["obs"]) and s["obs"][i]]
            if len(rows) < 2:
                continue
            L = np.array([r[0] for r in rows])
            D = np.array([r[1] for r in rows])
            if len(rows) >= 8:  # shift varies across the screen (perspective): affine in u, v
                F = np.c_[np.ones(len(L)), L]
                self.off[i] = np.linalg.lstsq(F, D, rcond=None)[0]
            else:
                self.off[i] = np.median(D, 0)

    def save(self):
        allm = json.loads(MODEL_FILE.read_text()) if MODEL_FILE.exists() else {}
        allm[self.key()] = self.data
        MODEL_FILE.write_text(json.dumps(allm))

    def describe(self):
        if not self.enabled:
            return "tip vision: off (no camera)"
        s = len(self.data["samples"])
        return (f"tip vision: {self.n} camera{'s' if self.n > 1 else ''} ({', '.join(self.specs)}), "
                f"{s} learned taps" + (", correcting aim" if self.ready else f", learning only until {MIN_SAMPLES}"))


def check_setup(vision, out=Path("outputs")):
    """Camera check: markers, tip, viewing angles, and whether a pair can measure height."""
    out.mkdir(parents=True, exist_ok=True)
    geo = []
    for i, cam in enumerate(vision.cams):
        f = cam.frame()
        quad, fresh = cam.locate(f)
        print(f"\ncamera {i + 1}: {vision.specs[i]}  {f.shape[1]}x{f.shape[0]}")
        if quad is None:
            print("  screen markers NOT found -> fix this camera first (./lr cam-test)")
            geo.append(None)
            continue
        g = view_geometry(quad, f.shape)
        geo.append(g)
        if g:
            print(f"  looks at the screen {g['tilt_deg']:.0f} deg away from straight-on "
                  f"(0 = straight on, 90 = edge-on)")
        tip = find_tip(f, vision.color, quad)
        dbg = f.copy()
        cv2.polylines(dbg, [quad.astype(int)], True, (0, 255, 0), max(2, f.shape[1] // 400))
        if tip:
            (x, y), area = tip
            cv2.circle(dbg, (int(x), int(y)), max(8, f.shape[1] // 80), (255, 0, 255), 3)
            print(f"  {vision.color} tip found ({area} px)")
        else:
            print(f"  {vision.color} tip NOT found: sticker visible? bright enough? try another --color")
        cv2.imwrite(str(out / f"tip_cam{i + 1}.jpg"), dbg)
        print(f"  debug image: {out / f'tip_cam{i + 1}.jpg'}")
    if len(geo) == 2 and all(geo):
        ang = float(np.degrees(np.arccos(np.clip(np.dot(geo[0]["dir"], geo[1]["dir"]), -1, 1))))
        verdict = ("good for measuring height" if ang >= 25 else
                   "OK, height will be rough" if ang >= 12 else
                   "too similar: height can't be measured, move one camera (lower/flatter or more to the side)")
        print(f"\nangle between the two cameras: ~{ang:.0f} deg -> {verdict}")
    obs = vision.observe()
    est = vision.estimate(obs)
    print(f"\nnow: tip seen by {est['cams']} camera(s); landing estimate "
          + (f"u={est['landing'][0]:.3f} v={est['landing'][1]:.3f}" if est["landing"] else "-")
          + (f", height {est['height']:.2f} (1 = normal hover)" if est.get("height") is not None else ""))
    print(vision.describe())


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--cameras", default="", help="'none', one camera, or two separated by a comma")
    p.add_argument("--color", default="pink", choices=sorted(COLORS))
    p.add_argument("--out", default="outputs")
    p.add_argument("--delay", type=float, default=0, help="seconds to switch to the page before the check")
    a = p.parse_args()
    from common import countdown
    countdown(a.delay, "switch to the page with the corner markers (full-screen, F11)")
    v = TipVision(a.cameras, a.color)
    if not v.enabled:
        raise SystemExit("No camera configured (CAMERA / CAMERA2 in lr.conf).")
    check_setup(v, Path(a.out))
