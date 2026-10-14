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
MAX_VALID_ERR = 0.02  # camera must predict held-out taps within ~2% of the screen before it may correct

# HSV ranges (OpenCV: H 0-179). Several ranges per color are OR-ed.
COLORS = {
    "pink": [((140, 70, 90), (179, 255, 255)), ((0, 120, 120), (4, 255, 255))],
    "green": [((40, 80, 70), (85, 255, 255))],
    "orange": [((5, 130, 120), (21, 255, 255))],
    # yellow as cheap webcams really see it: often shifted to orange-yellow and not very saturated
    "yellow": [((10, 80, 70), (36, 255, 255))],
    "blue": [((95, 120, 70), (128, 255, 255))],
}
UNIT = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], np.float32)


def parse_cameras(spec):
    """'none' / '' -> [];  'a,b' -> ['a','b'] (max 2)."""
    if not spec or str(spec).strip().lower() in ("none", "off", "0cam"):
        return []
    return [c.strip() for c in str(spec).split(",") if c.strip() and c.strip().lower() != "none"][:2]


def color_ranges(color):
    """A preset name, or custom HSV ranges 'h1,s1,v1,h2,s2,v2[,...]' (OpenCV: H 0-179)."""
    if color in COLORS:
        return COLORS[color]
    try:
        n = [int(x) for x in str(color).split(",")]
    except ValueError:
        n = []
    if not n or len(n) % 6:
        raise SystemExit(f"Unknown tip colour '{color}': use {', '.join(COLORS)} or 'h1,s1,v1,h2,s2,v2'")
    return [(tuple(n[i:i + 3]), tuple(n[i + 3:i + 6])) for i in range(0, len(n), 6)]


def tip_mask(frame, color, quad=None):
    """Pixels that look like the tip colour (255), near the screen only."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = np.zeros(hsv.shape[:2], np.uint8)
    for lo, hi in color_ranges(color):
        mask |= cv2.inRange(hsv, np.array(lo, np.uint8), np.array(hi, np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    if quad is not None:  # only look on/near the screen (the stylus can stick out over the edge)
        c = quad.mean(0)
        grown = (c + (quad - c) * 1.35).astype(np.int32)
        roi = np.zeros_like(mask)
        cv2.fillPoly(roi, [grown], 255)
        mask &= roi
    return mask


# ---------- tip colour taught by clicking on the photo (live view) ----------
PICK_FILE = HERE / "tip_color.json"


def load_picked():
    try:
        return json.loads(PICK_FILE.read_text())
    except Exception:
        return None


def sample_click(frame, x, y):
    """HSV pixels in a small disc around (x, y) in camera pixels."""
    h, w = frame.shape[:2]
    r = max(4, w // 250)
    x0, x1, y0, y1 = max(0, int(x) - r), min(w, int(x) + r + 1), max(0, int(y) - r), min(h, int(y) + r + 1)
    hsv = cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2HSV).reshape(-1, 3)
    yy, xx = np.mgrid[y0:y1, x0:x1]
    keep = ((xx - x) ** 2 + (yy - y) ** 2).ravel() <= r * r
    return hsv[keep]


def ranges_from_pixels(px):
    """HSV ranges that cover the clicked pixels (with some slack for light changes).
    Returns (color string, warning or None)."""
    px = np.asarray(px, float)
    hue = px[:, 0] * 2 * np.pi / 180
    centre = (np.degrees(np.arctan2(np.sin(hue).mean(), np.cos(hue).mean())) / 2) % 180  # circular mean
    dev = np.abs((px[:, 0] - centre + 90) % 180 - 90)
    half = float(np.clip(np.percentile(dev, 95) + 5, 8, 30))
    # generous on saturation/brightness: shade, glare and a new viewing angle all make the tip duller
    s_lo = int(np.clip(np.percentile(px[:, 1], 5) - 50, 100, 180))
    v_lo = int(np.clip(np.percentile(px[:, 2], 5) - 70, 50, 160))
    lo_h, hi_h = centre - half, centre + half
    spans = ([(lo_h, hi_h)] if 0 <= lo_h and hi_h <= 179 else
             [(0, hi_h), (lo_h + 180, 179)] if lo_h < 0 else [(lo_h, 179), (0, hi_h - 180)])
    out = []
    for a, b in spans:
        out += [int(round(a)), s_lo, v_lo, int(round(b)), 255, 255]
    warn = None
    if np.median(px[:, 1]) < 60:
        warn = ("the clicked spot is almost grey/white/black: colour detection will also find other grey things. "
                "Put bright coloured tape on the tip for reliable detection.")
    return ",".join(map(str, out)), warn


def find_tip(frame, color, quad=None, min_area=12):
    """Where the stylus tip is, in camera pixels. Returns ((x, y), area) or None.

    Finds the biggest blob of `color` near the screen. If the coloured part is long (e.g. the
    whole stylus body is coloured), the tip is the END of that blob that points into the screen,
    not its middle."""
    mask = tip_mask(frame, color, quad)
    n, labels, stats, cents = cv2.connectedComponentsWithStats(mask)
    if n <= 1:
        return None
    # pick the blob that looks most like a coloured marker: strongly coloured, compact and round.
    # (just "the biggest" picked hands/skin and other orange-ish things in real photos)
    sat = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)[:, :, 1].astype(float)
    mean_s = np.bincount(labels.ravel(), weights=sat.ravel(), minlength=n) / np.maximum(stats[:, cv2.CC_STAT_AREA], 1)
    a = stats[:, cv2.CC_STAT_AREA].astype(float)
    w, h = stats[:, cv2.CC_STAT_WIDTH].astype(float), stats[:, cv2.CC_STAT_HEIGHT].astype(float)
    score = (mean_s / 255) ** 6 * a * (a / np.maximum(w * h, 1)) * (np.minimum(w, h) / np.maximum(np.maximum(w, h), 1))
    if quad is not None:  # the tip is over the screen: things beside it (desk, background) count much less
        inside = np.array([cv2.pointPolygonTest(quad.reshape(-1, 1, 2), (float(x), float(y)), False) >= 0
                           for x, y in cents])
        score = np.where(inside, score, score * 0.1)
    score[0] = -1
    score[a < min_area] = -1
    i = int(np.argmax(score))
    if score[i] < 0:
        return None
    area = int(stats[i, cv2.CC_STAT_AREA])
    ys, xs = np.nonzero(labels == i)
    pts = np.c_[xs, ys].astype(float)
    centre = pts.mean(0)
    # elongated blob? -> take its end along the main axis
    w, vecs = np.linalg.eigh(np.cov((pts - centre).T))
    axis = vecs[:, 1]
    if w[1] > 6 * max(w[0], 1e-6):
        proj = (pts - centre) @ axis
        ends = [centre + axis * np.percentile(proj, 2), centre + axis * np.percentile(proj, 98)]
        if quad is not None:  # the tip is the end deeper inside the screen; the other end goes to the gripper
            depth = [cv2.pointPolygonTest(quad.reshape(-1, 1, 2), (float(e[0]), float(e[1])), True) for e in ends]
            tip = ends[int(np.argmax(depth))]
        else:
            tip = ends[1]
        return (float(tip[0]), float(tip[1])), area
    return (float(centre[0]), float(centre[1])), area


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


# ---------- 3D geometry from the screen markers ----------
SCREEN_ASPECT = 1.513   # width / height of the marker frame (your screen: 1961 x 1296 px)
SCREEN_WIDTH_MM = 300.0  # only sets the unit of the height (mm); set SCREEN_WIDTH_MM in lr.conf if known


def estimate_focal(quad, shape, aspect=SCREEN_ASPECT):
    """Focal length (pixels) from how the screen rectangle appears (no camera calibration needed).
    Uses the fact that the screen's two edge directions are perpendicular and correctly scaled."""
    h_img, w_img = shape[:2]
    obj = np.array([[0, 0], [1, 0], [1, 1 / aspect], [0, 1 / aspect]], np.float32)
    H = cv2.getPerspectiveTransform(obj, (quad - [w_img / 2, h_img / 2]).astype(np.float32))
    h1, h2 = H[:, 0], H[:, 1]
    cands = []
    if abs(h1[2] * h2[2]) > 1e-12:
        cands.append(-(h1[0] * h2[0] + h1[1] * h2[1]) / (h1[2] * h2[2]))
    if abs(h1[2] ** 2 - h2[2] ** 2) > 1e-12:
        cands.append(-((h1[0] ** 2 + h1[1] ** 2) - (h2[0] ** 2 + h2[1] ** 2)) / (h1[2] ** 2 - h2[2] ** 2))
    fs = [np.sqrt(c) for c in cands if c > 0 and 0.4 * w_img < np.sqrt(c) < 4 * w_img]
    return float(np.median(fs)) if fs else None


def camera_pose(quad, shape, f, aspect=SCREEN_ASPECT, width_mm=SCREEN_WIDTH_MM):
    """Camera position + orientation in screen coordinates (mm): x right, y down, z into the screen."""
    h_img, w_img = shape[:2]
    K = np.array([[f, 0, w_img / 2], [0, f, h_img / 2], [0, 0, 1]], float)
    W, Hs = width_mm, width_mm / aspect
    obj = np.array([[0, 0, 0], [W, 0, 0], [W, Hs, 0], [0, Hs, 0]], float)
    ok, rvec, tvec = cv2.solvePnP(obj, quad.astype(float), K, None, flags=cv2.SOLVEPNP_IPPE)
    if not ok:
        return None
    R, _ = cv2.Rodrigues(rvec)
    return {"K": K, "R": R, "C": (-R.T @ tvec).ravel()}


def ray(pose, px):
    d = pose["R"].T @ np.linalg.inv(pose["K"]) @ np.array([px[0], px[1], 1.0])
    return pose["C"], d / np.linalg.norm(d)


def triangulate(r1, r2):
    """Closest point between two sight lines, and how far apart they pass (consistency, mm)."""
    (c1, d1), (c2, d2) = r1, r2
    w = c1 - c2
    a, b, c, d, e = d1 @ d1, d1 @ d2, d2 @ d2, d1 @ w, d2 @ w
    den = a * c - b * b
    if abs(den) < 1e-9:
        return None, None
    s, t = (b * e - c * d) / den, (a * e - b * d) / den
    p1, p2 = c1 + s * d1, c2 + t * d2
    return (p1 + p2) / 2, float(np.linalg.norm(p1 - p2))


# ---------- distance teaching: screen shape + each camera's focal length from labelled tips ----------
GEOM_FILE = HERE / "cam_geometry.json"


def load_geometry():
    try:
        return json.loads(GEOM_FILE.read_text())
    except Exception:
        return {}


def _sample_height(views, aspect, frels, width_mm):
    poses = [camera_pose(np.array(v["quad"], np.float32), v["shape"], fr * v["shape"][1], aspect, width_mm)
             for v, fr in zip(views, frels)]
    if any(p is None for p in poses):
        return None, None
    P, gap = triangulate(ray(poses[0], views[0]["px"]), ray(poses[1], views[1]["px"]))
    if P is None:
        return None, None
    side = -1.0 if (poses[0]["C"][2] + poses[1]["C"][2]) < 0 else 1.0
    return float(side * P[2]), gap


def _nelder_mead(fun, x0, step, iters=300):
    n = len(x0)
    pts = [np.array(x0, float)] + [np.array(x0, float) + np.eye(n)[i] * step[i] for i in range(n)]
    vals = [fun(p) for p in pts]
    for _ in range(iters):
        o = np.argsort(vals)
        pts, vals = [pts[i] for i in o], [vals[i] for i in o]
        c = np.mean(pts[:-1], 0)
        xr = c + (c - pts[-1])
        fr = fun(xr)
        if fr < vals[0]:
            xe = c + 2 * (c - pts[-1])
            fe = fun(xe)
            pts[-1], vals[-1] = (xe, fe) if fe < fr else (xr, fr)
        elif fr < vals[-2]:
            pts[-1], vals[-1] = xr, fr
        else:
            xc = c + 0.5 * (pts[-1] - c)
            fc = fun(xc)
            if fc < vals[-1]:
                pts[-1], vals[-1] = xc, fc
            else:
                pts = [pts[0]] + [pts[0] + 0.5 * (q - pts[0]) for q in pts[1:]]
                vals = [vals[0]] + [fun(q) for q in pts[1:]]
    i = int(np.argmin(vals))
    return pts[i], vals[i]


def fit_geometry(samples, width_mm):
    """Find the screen's shape (width/height), both cameras' focal lengths and the marker offset
    (how high the coloured marker sits when the tip touches) so that the labelled tips read their
    true height and both sight lines meet. samples: [{'views': [v1, v2], 'h': mm}]."""
    def cost(x):
        a, f1, f2, b = x
        if not (1.0 < a < 2.6 and 0.3 < f1 < 2.0 and 0.3 < f2 < 2.0 and -10 < b < 40):
            return 1e9
        tot = 0.0
        for smp in samples:
            h, gap = _sample_height(smp["views"], a, (f1, f2), width_mm)
            if h is None:
                return 1e9
            tot += (h - b - smp["h"]) ** 2 + gap ** 2
        return tot / len(samples)
    best = None
    for a0 in (1.6, 1.78):  # starting guesses: 16:10 and 16:9 screens, wide and normal lenses
        for f0 in (0.5, 0.9):
            x, v = _nelder_mead(cost, [a0, f0, f0, 5.0], [0.05, 0.1, 0.1, 3.0])
            if best is None or v < best[1]:
                best = (x, v)
    (a, f1, f2, b), v = best
    res = [(_sample_height(smp["views"], a, (f1, f2), width_mm), smp["h"]) for smp in samples]
    return {"aspect": float(a), "focal_rel": [float(f1), float(f2)], "offset_mm": float(b),
            "rms_mm": float(np.sqrt(v)),
            "check": [{"true_mm": t, "read_mm": round(h - b, 1), "gap_mm": round(g, 1)} for (h, g), t in res]}


class TipVision:
    def __init__(self, cameras, color="pink", cid=None, aspect=SCREEN_ASPECT, width_mm=SCREEN_WIDTH_MM,
                 max_gap_mm=12.0):
        self.color = color
        picked = load_picked()
        if picked and picked.get("color"):  # taught by clicking in the live view: wins over TIP_COLOR
            self.color = picked["color"]
            if picked.get("pixels"):  # re-derive with the current rule (older files were too strict)
                self.color = ranges_from_pixels(picked["pixels"])[0]
            print(f"tip colour: taught by clicking ({len(picked.get('clicks', []))} clicks), not '{color}'. "
                  "Reset it in ./lr tip-live.")
        self.aspect, self.width_mm, self.max_gap = aspect, width_mm, max_gap_mm
        self.geom = load_geometry()  # taught in ./lr tip-live: screen shape + focal lengths
        if self.geom.get("aspect"):
            self.aspect = self.geom["aspect"]
        self.focal = {}  # camera index -> recent focal estimates (median is used)
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
        """Allowed to correct the arm: enough taps AND it has proven it predicts them accurately."""
        return (self.enabled and len(self.data["samples"]) >= MIN_SAMPLES
                and self.valid_err is not None and self.valid_err <= MAX_VALID_ERR)

    # ---- seeing ----
    def observe(self):
        """One look from every camera. List (per camera) of {'uv','px','area'} or None."""
        out = []
        frames = []  # filled first, then swapped in, so the live view never sees a half-empty list
        for cam in self.cams:
            try:
                f = cam.frame()
                quad, _ = cam.locate(f)
                tip = find_tip(f, self.color, quad) if quad is not None else None
                frames.append((f, quad, tip))
                if tip is None:
                    out.append(None)
                    continue
                (x, y), area = tip
                H = cv2.getPerspectiveTransform(quad, UNIT)
                uv = cv2.perspectiveTransform(np.array([[[x, y]]], np.float32), H)[0, 0]
                out.append({"uv": [float(uv[0]), float(uv[1])], "px": [float(x), float(y)], "area": area,
                            "quad": quad.tolist(), "shape": list(f.shape[:2])})
            except Exception as e:
                print(f"(camera read failed: {e})")
                frames.append((None, None, None))
                out.append(None)
        self.last_frames = frames
        return out

    # ---- estimating ----
    def _offset(self, i, uv):
        """Learned parallax shift of camera i at screen point uv (for the normal hover height)."""
        m = self.off.get(i)
        if m is None:
            return None
        f = np.array([1.0, uv[0], uv[1]])
        return f @ m if m.ndim == 2 else m

    def _pose(self, i, o):
        quad = np.array(o["quad"], np.float32)
        frel = self.geom.get("focal_rel", {}).get(self.specs[i]) if i < len(self.specs) else None
        if frel:  # taught: much more reliable than guessing it from the screen outline
            return camera_pose(quad, o["shape"], frel * o["shape"][1], self.aspect, self.width_mm)
        f = estimate_focal(quad, o["shape"], self.aspect)
        hist = self.focal.setdefault(i, [])
        if f:
            hist.append(f)
            del hist[:-30]
        fm = float(np.median(hist)) if hist else max(o["shape"]) / (2 * np.tan(np.radians(32.5)))
        return camera_pose(quad, o["shape"], fm, self.aspect, self.width_mm)

    def teach_distance(self, obs, true_mm):
        """Record 'the tip is true_mm above the glass now' and refit shape + focal lengths."""
        seen = [o for o in obs if o and o.get("quad")]
        if len(seen) < 2:
            return None, "both cameras must see the screen and the tip"
        g = self.geom
        smp = g.setdefault("samples", [])
        smp.append({"specs": self.specs[:2], "h": float(true_mm),
                    "views": [{"px": o["px"], "quad": o["quad"], "shape": o["shape"]} for o in seen[:2]]})
        mine = [x for x in smp if x["specs"] == self.specs[:2]]
        touch = sum(x["h"] < 3 for x in mine)
        if len(mine) < 3 or touch < 1 or touch == len(mine):
            GEOM_FILE.write_text(json.dumps(g))
            return None, (f"{len(mine)} saved ({touch} touching, {len(mine) - touch} raised). Need at least 3, "
                          "with both touching and raised ones, at different places on the screen.")
        fit = fit_geometry(mine, self.width_mm)
        g["aspect"], g["offset_mm"] = fit["aspect"], fit["offset_mm"]
        g.setdefault("focal_rel", {}).update(dict(zip(self.specs[:2], fit["focal_rel"])))
        g["fit"] = fit
        GEOM_FILE.write_text(json.dumps(g))
        self.geom, self.aspect, self.focal = g, fit["aspect"], {}
        return fit, None

    def forget_distance(self):
        GEOM_FILE.unlink(missing_ok=True)
        self.geom, self.aspect, self.focal = {}, SCREEN_ASPECT, {}

    def tip3d(self, obs):
        """3D tip from two cameras: (u, v) under the tip, height above the glass (mm), line gap (mm)."""
        seen = [(i, o) for i, o in enumerate(obs) if o and o.get("quad")]
        if len(seen) < 2:
            return None
        (a, oa), (b, ob) = seen[:2]
        pa, pb = self._pose(a, oa), self._pose(b, ob)
        if pa is None or pb is None:
            return None
        P, gap = triangulate(ray(pa, oa["px"]), ray(pb, ob["px"]))
        if P is None:
            return None
        side = -1.0 if (pa["C"][2] + pb["C"][2]) < 0 else 1.0  # cameras are on the viewer side
        W, H = self.width_mm, self.width_mm / self.aspect
        h = float(side * P[2]) - float(self.geom.get("offset_mm", 0.0))  # taught: marker height at contact
        return {"uv": (float(P[0] / W), float(P[1] / H)), "height_mm": h, "gap_mm": gap,
                "cam_dist_mm": [float(np.linalg.norm(pa["C"])), float(np.linalg.norm(pb["C"]))]}

    def estimate(self, obs):
        """{'landing': (u,v) or None, 'height': relative or None, 'cams': n seen}.
        Two cameras: real 3D (height in mm). One camera: learned parallax."""
        seen = [(i, np.array(o["uv"])) for i, o in enumerate(obs) if o]
        res = {"landing": None, "height": None, "cams": len(seen)}
        if not seen:
            return res
        t3 = self.tip3d(obs) if len(seen) >= 2 else None
        if t3 is not None:
            res.update(height_mm=t3["height_mm"], gap_mm=t3["gap_mm"], tip_uv=t3["uv"], mode="3d")
            if t3["gap_mm"] > self.max_gap:  # the two sight lines don't meet: one camera saw something else
                res["inconsistent"] = True
                return res
            # landing = point under the tip + learned drift while coming down (stylus tilt) * height
            land = np.array(t3["uv"])
            if self.drift is not None:
                land = land + self.drift[0] + self.drift[1] * t3["height_mm"]
            res["landing"] = tuple(land)
            res["height"] = t3["height_mm"]
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
        t3 = self.tip3d(obs)
        if t3 is not None and t3["gap_mm"] <= self.max_gap and -0.2 <= t3["uv"][0] <= 1.2 and -0.2 <= t3["uv"][1] <= 1.2:
            pass  # a consistent 3D reading: keep it
        else:
            obs = [o if (o and -0.15 <= o["uv"][0] <= 1.15 and -0.15 <= o["uv"][1] <= 1.15) else None for o in obs]
        if not any(obs):  # the camera saw something far off the screen: not the tip (or useless angle)
            self.rejected = getattr(self, "rejected", 0) + 1
            return
        self.data["samples"].append({"obs": obs, "landed": [float(landed[0]), float(landed[1])], "t": time.time()})
        self._fit()

    def _fit(self):
        self._fit_offsets()
        self.valid_err = self._validate()

    def _validate(self):
        """Leave-one-out: predict each learned tap from the others; mean error in screen units."""
        S = self.data["samples"]
        if len(S) < MIN_SAMPLES:
            return None
        errs = []
        for i in range(len(S)):
            keep = self.data["samples"]
            self.data["samples"] = keep[:i] + keep[i + 1:]
            self._fit_offsets()
            est = self.estimate(S[i]["obs"])
            self.data["samples"] = keep
            if est["landing"] is not None:
                errs.append(float(np.hypot(*(np.array(est["landing"]) - S[i]["landed"]))))
        self._fit_offsets()
        return float(np.mean(errs)) if errs else None

    def _fit_offsets(self):
        self._fit_drift()
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

    def _fit_drift(self):
        """3D: how the touch point differs from the point under the hovering tip: bias + slope * height."""
        self.drift = None
        rows = []
        for s in self.data["samples"]:
            t3 = self.tip3d(s["obs"]) if len([o for o in s["obs"] if o]) >= 2 else None
            if t3 is not None and t3["gap_mm"] <= self.max_gap:
                rows.append((t3["height_mm"], np.array(s["landed"]) - t3["uv"]))
        if len(rows) >= 2:
            h = np.array([r[0] for r in rows])
            D = np.array([r[1] for r in rows])
            if len(rows) >= 4 and np.ptp(h) > 2:
                A = np.c_[np.ones(len(h)), h]
                c = np.linalg.lstsq(A, D, rcond=None)[0]
                self.drift = (c[0], c[1])
            else:
                self.drift = (np.median(D, 0), np.zeros(2))

    def save(self):
        allm = json.loads(MODEL_FILE.read_text()) if MODEL_FILE.exists() else {}
        allm[self.key()] = self.data
        MODEL_FILE.write_text(json.dumps(allm))

    def describe(self):
        if not self.enabled:
            return "tip vision: off (no camera)"
        s = len(self.data["samples"])
        ve = self.valid_err
        state = ("correcting aim" if self.ready else
                 f"learning only (needs {MIN_SAMPLES} taps)" if s < MIN_SAMPLES else
                 f"NOT trusted: predicts taps only to ~{100 * ve:.1f}% of the screen (needs <= {100 * MAX_VALID_ERR:.0f}%)"
                 " - move the camera above the screen / put the marker right at the tip")
        rj = getattr(self, "rejected", 0)
        return (f"tip vision: {self.n} camera{'s' if self.n > 1 else ''} ({', '.join(self.specs)}), {s} learned taps"
                + (f", accuracy ~{100 * ve:.1f}% of screen" if ve is not None else "") + f", {state}"
                + (f", {rj} impossible readings ignored" if rj else ""))


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
    print(f"\nnow: tip seen by {est['cams']} camera(s)")
    if est.get("mode") == "3d":
        print(f"  3D: tip is over u={est['tip_uv'][0]:.3f} v={est['tip_uv'][1]:.3f}, "
              f"{est['height_mm']:.1f} mm above the glass  (sight lines meet within {est['gap_mm']:.1f} mm"
              + (" -> INCONSISTENT: one camera sees something else)" if est.get("inconsistent") else ")"))
        if vision.focal:
            print("  estimated lens focal length per camera (px): "
                  + ", ".join(f"{np.median(h):.0f}" for h in vision.focal.values()))
        print("  (height is in mm if SCREEN_WIDTH_MM in lr.conf matches the marker-to-marker width of your screen)")
    elif est["landing"]:
        print(f"  landing estimate u={est['landing'][0]:.3f} v={est['landing'][1]:.3f} (1 camera: no height)")
    print(vision.describe())


LIVE_PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>Tip live view</title>
<style>
 body{margin:0;font-family:system-ui,sans-serif;background:#111;color:#eee}
 header{display:flex;gap:28px;align-items:baseline;padding:14px 20px;background:#1c1c1c;flex-wrap:wrap}
 .big{font-size:44px;font-weight:700;font-variant-numeric:tabular-nums}
 .lbl{font-size:13px;color:#aaa;display:block}
 .ok{color:#5bd38a}.bad{color:#ff6b6b}.warn{color:#ffcc55}
 main{display:flex;flex-wrap:wrap;gap:10px;padding:10px}
 main div{flex:1 1 420px}
 #c0,#c1{cursor:crosshair}
 img{width:100%;border-radius:6px;background:#222}
 nav{display:flex;gap:26px;align-items:center;padding:10px 20px;background:#181818;flex-wrap:wrap;font-size:15px}
 nav input[type=range]{width:140px;vertical-align:middle} nav b{display:inline-block;min-width:22px}
 select,button{background:#2a2a2a;color:#eee;border:1px solid #444;border-radius:4px;padding:3px 8px}
 #map{position:relative;aspect-ratio:1.78;background:#fff;border-radius:6px;max-width:420px}
 #dot{position:absolute;width:18px;height:18px;margin:-9px 0 0 -9px;border-radius:50%;background:#e2b300;display:none}
</style></head><body>
<header>
 <div><span class="lbl">height above the glass</span><span id="h" class="big">-</span></div>
 <div><span class="lbl">tip over the screen (u, v)</span><span id="uv" class="big">-</span></div>
 <div><span class="lbl">cameras agree (sight lines meet within)</span><span id="gap" class="big">-</span></div>
 <div><span class="lbl">status</span><span id="st" class="big">-</span></div>
 <div><span class="lbl">camera placement (angle between cameras)</span><span id="pl" class="big">-</span>
  <span id="tl" class="lbl"></span></div>
</header>
<nav>
 <label>max frame rate <input id="fps" type="range" min="1" max="30" value="15"> <b id="fpsv">15</b> fps</label>
 <label>phone resolution <select id="res"><option value="">(unchanged)</option><option>640x480</option>
  <option>800x480</option><option>1280x720</option><option>1920x1080</option></select></label>
 <label>phone quality <input id="q" type="range" min="20" max="90" step="5" value="60"> <b id="qv">60</b>
  <button id="qset">apply</button></label>
 <span id="rate" class="lbl"></span><span id="ctlmsg" class="lbl"></span>
</nav>
<nav>
 <button id="snap">&#128247; save photo</button>
 <label><input id="pause" type="checkbox"> pause (freeze the picture to click precisely)</label>
 <label><input id="mask" type="checkbox"> show what it thinks is the pen</label>
 <button id="reset">reset taught colour</button>
 <button id="flip0">flip cam 1</button> <button id="flip1">flip cam 2</button>
 <span class="lbl" style="color:#9cf">Click the pen TIP in a camera picture to teach what it looks like and where it is.</span>
</nav>
<nav>
 <span>teach distance:</span>
 <button id="t0">tip is TOUCHING the glass now</button>
 <label><button id="tr">tip is this far now:</button> <input id="trmm" type="number" value="80" style="width:60px"> mm</label>
 <button id="tforget">forget distance teaching</button>
 <span class="lbl">hold the pen still; do 2+ touching and 2+ raised, at different places on the screen</span>
</nav>
<div id="pick" style="padding:6px 20px;color:#ffcc55;min-height:20px"></div>
<main><div><img id="c0"></div><div><img id="c1"></div>
<div><span class="lbl">where the tip is over the screen</span><div id="map"><div id="dot"></div></div></div></main>
<script>
const $=id=>document.getElementById(id);
let fps=15; try{fps=+localStorage.getItem('fps')||15}catch(e){}
$('fps').value=fps; $('fpsv').textContent=fps;
async function ctl(q){ try{ const r=await (await fetch('/set?'+q,{cache:'no-store'})).json();
  $('ctlmsg').textContent=r.msg||''; }catch(e){ $('ctlmsg').textContent='not applied'; } }
$('fps').oninput=e=>{ fps=+e.target.value; $('fpsv').textContent=fps; };
$('fps').onchange=()=>{ try{localStorage.setItem('fps',fps)}catch(e){} ctl('fps='+fps); };
$('res').onchange=e=>{ if(e.target.value) ctl('res='+e.target.value); };
$('q').oninput=e=>$('qv').textContent=e.target.value;
$('qset').onclick=()=>ctl('quality='+$('q').value);
ctl('fps='+fps);
async function say(url){ try{ const r=await (await fetch(url,{cache:'no-store'})).json(); $('pick').textContent=r.msg||''; }
  catch(e){ $('pick').textContent='no answer from the robot laptop'; } }
for(const i of [0,1]) $('c'+i).onclick=e=>{ const el=e.target;
  say('/click?cam='+i+'&x='+(e.offsetX/el.clientWidth).toFixed(4)+'&y='+(e.offsetY/el.clientHeight).toFixed(4)); };
$('snap').onclick=()=>say('/snap');
$('reset').onclick=()=>{ if(confirm('Forget the taught pen colour and go back to TIP_COLOR?')) say('/reset'); };
$('t0').onclick=()=>say('/teach?mm=0');
$('tr').onclick=()=>say('/teach?mm='+(+$('trmm').value||80));
$('tforget').onclick=()=>{ if(confirm('Forget the taught distances?')) say('/teach?forget=1'); };
for(const i of [0,1]) $('flip'+i).onclick=()=>say('/set?flip='+i);
$('pause').onchange=e=>ctl('pause='+(e.target.checked?1:0));
$('mask').onchange=e=>ctl('mask='+(e.target.checked?1:0));
async function tick(){
  try{
    const s = await (await fetch('/state',{cache:'no-store'})).json();
    const H=document.getElementById('h'), U=document.getElementById('uv'), G=document.getElementById('gap'), S=document.getElementById('st');
    if(s.height_mm!=null){ H.textContent = s.height_mm<2 ? 'touching' : s.height_mm.toFixed(1)+' mm';
      H.className='big '+(s.height_mm<2?'ok':''); } else H.textContent='-';
    if(s.uv){ U.textContent=s.uv[0].toFixed(3)+', '+s.uv[1].toFixed(3);
      const d=document.getElementById('dot'); d.style.display='block'; d.style.left=(s.uv[0]*100)+'%'; d.style.top=(s.uv[1]*100)+'%'; }
    else { U.textContent='-'; document.getElementById('dot').style.display='none'; }
    G.textContent = s.gap_mm!=null ? s.gap_mm.toFixed(1)+' mm' : '-';
    G.className='big '+(s.gap_mm==null?'':(s.gap_mm<=12?'ok':'bad'));
    S.textContent=s.status; S.className='big '+(s.status==='3D OK'?'ok':'warn');
    const P=document.getElementById('pl');
    if(s.pair_deg!=null){ P.textContent=s.pair_deg+'° '+s.placement; P.className='big '+(s.placement==='good'?'ok':'warn'); }
    else P.textContent='-';
    document.getElementById('tl').textContent = (s.cam_tilt||[]).map((t,i)=>'cam '+(i+1)+': '+(t==null?'no screen':t+'° from straight-on')).join('   ');
    for(const i of [0,1]){ document.getElementById('c'+i).src='/cam'+i+'.jpg?t='+Date.now(); }
    if(s.aspect) $('map').style.aspectRatio=s.aspect;
    $('rate').textContent='processing '+(s.loop_fps??'-')+' fps   '+(s.cam_fps||[]).map((f,i)=>'cam '+(i+1)+' sends '+(f==null?'?':f)+' fps').join('   ');
  }catch(e){ document.getElementById('st').textContent='waiting...'; }
  setTimeout(tick, 1000/fps);
}
tick();
</script></body></html>"""


def live(vision, port=8766, out_dir="outputs"):
    """Live test without the arm: both camera feeds with the screen + tip drawn, and the 3D reading.
    Click the pen tip in a picture to teach its colour; 'save photo' stores everything for analysis."""
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    from urllib.parse import parse_qs

    shared = {"jpg": {}, "state": {"status": "starting"}, "fps": 15.0, "pause": False, "mask": False,
              "marks": {}}  # marks: camera -> last clicked point (camera px)
    srcs = [c.src for c in vision.cams]
    start_color = vision.color if not load_picked() else None
    picked = load_picked() or {}
    pixels = [list(map(int, p)) for p in picked.get("pixels", [])]
    clicks = list(picked.get("clicks", []))

    def frame_of(i):
        fr = vision.last_frames[i] if i < len(vision.last_frames) else None
        return fr if fr and fr[0] is not None else None

    def on_click(q):
        i = int(q.get("cam", 0))
        fr = frame_of(i)
        if fr is None:
            return {"msg": f"camera {i + 1} has no picture yet"}
        f, quad, _ = fr
        x, y = float(q["x"]) * f.shape[1], float(q["y"]) * f.shape[0]
        shared["marks"][i] = (x, y)
        px = sample_click(f, x, y)
        pixels.extend(px.astype(int).tolist())
        color, warn = ranges_from_pixels(pixels)
        vision.color = color
        tip = find_tip(f, color, quad)
        dist = None if tip is None else float(np.hypot(tip[0][0] - x, tip[0][1] - y))
        med = np.median(px, 0).astype(int).tolist()
        clicks.append({"t": time.strftime("%Y-%m-%d %H:%M:%S"), "cam": i, "x": round(x, 1), "y": round(y, 1),
                       "frame": list(f.shape[:2]), "hsv_median": med, "detected_dist_px": dist})
        try:
            PICK_FILE.write_text(json.dumps({"color": color, "clicks": clicks, "pixels": pixels[-4000:]}))
        except Exception as e:
            warn = f"couldn't save the colour ({e})"
        n = len(clicks)
        msg = (f"cam {i + 1}: clicked colour H{med[0]} S{med[1]} V{med[2]}; taught from {n} click{'s' * (n > 1)} "
               f"-> HSV ranges {color}. "
               + ("Detected tip is now ON your click." if dist is not None and dist < 15 else
                  f"Detected tip is {dist:.0f} px from your click (something else has a similar colour; "
                  "click the tip a few more times, also in the other camera)." if dist is not None else
                  "Tip still not found with this colour: click again on the tip."))
        print("live view:", msg)
        return {"msg": msg + (" WARNING: " + warn if warn else "")}

    def on_reset():
        nonlocal pixels, clicks
        pixels, clicks = [], []
        shared["marks"] = {}
        PICK_FILE.unlink(missing_ok=True)
        vision.color = start_color or "pink"
        return {"msg": f"taught colour forgotten, using '{vision.color}'"
                       + ("" if start_color else " (restart tip-live to use TIP_COLOR from your settings)")}

    def on_snap():
        d = Path(out_dir) / "snapshots" / time.strftime("%Y%m%d-%H%M%S")
        d.mkdir(parents=True, exist_ok=True)
        info = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "cameras": vision.specs, "color": vision.color,
                "state": shared["state"], "clicks": clicks, "views": []}
        for i, fr in enumerate(vision.last_frames):
            if fr is None or fr[0] is None:
                info["views"].append(None)
                continue
            f, quad, tip = fr
            cv2.imwrite(str(d / f"cam{i + 1}_raw.jpg"), f, [cv2.IMWRITE_JPEG_QUALITY, 95])
            cv2.imwrite(str(d / f"cam{i + 1}_mask.png"), tip_mask(f, vision.color, quad))
            if i in shared["jpg"]:
                (d / f"cam{i + 1}_view.jpg").write_bytes(shared["jpg"][i])
            info["views"].append({"size": list(f.shape[:2]), "quad": None if quad is None else quad.tolist(),
                                  "tip": None if tip is None else list(tip[0]), "tip_area": None if tip is None else tip[1],
                                  "clicked": shared["marks"].get(i)})
        (d / "info.json").write_text(json.dumps(info, indent=1, default=float))
        host = str(d).replace("/work/outputs", "outputs")
        print(f"live view: photo saved to {host}/")
        return {"msg": f"saved to {host}/  (./lr share copies it to your Downloads)"}

    def control(q):
        q = {k: v[0] for k, v in parse_qs(q).items()}
        out = []
        if "flip" in q:
            i = int(q["flip"])
            if i >= len(vision.cams):
                return {"msg": f"there is no camera {i + 1}"}
            cam = vision.cams[i]
            cam.src.set_rot180(not cam.src.rot180)
            cam.last_quad = None  # the remembered screen position is upside down now
            shared["marks"].pop(i, None)
            return {"msg": f"camera {i + 1} is now " + ("turned upside down" if cam.src.rot180 else "shown as it comes")
                    + " (remembered for tapping and practice too)"}
        for key in ("pause", "mask"):
            if key in q:
                shared[key] = q[key] == "1"
                out.append(f"{key} {'on' if shared[key] else 'off'}")
        if "fps" in q:
            shared["fps"] = min(30.0, max(1.0, float(q["fps"])))
            out.append(f"max {shared['fps']:.0f} fps")
        for key, setting in (("res", "video_size"), ("quality", "quality")):
            if key in q:
                errs = [e for e in (s.phone_set(setting, q[key]) for s in srcs) if e]
                phones = sum(s.phone_api is not None for s in srcs)
                out.append(f"{setting} {q[key]}: " + ("set on " + str(phones) + " phone(s)" if phones and not errs
                                                       else "failed (" + "; ".join(errs)[:120] + ")"))
                print("live view control:", out[-1])
        return {"msg": ", ".join(out)}

    class Handler(BaseHTTPRequestHandler):
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
            if p == "/state":
                return self._send(json.dumps(shared["state"]).encode(), "application/json")
            q = {k: v[0] for k, v in parse_qs(self.path.split("?", 1)[1] if "?" in self.path else "").items()}
            try:
                if p == "/click":
                    return self._send(json.dumps(on_click(q)).encode(), "application/json")
                if p == "/snap":
                    return self._send(json.dumps(on_snap()).encode(), "application/json")
                if p == "/teach":
                    if q.get("forget"):
                        vision.forget_distance()
                        return self._send(b'{"msg": "distance teaching forgotten"}', "application/json")
                    fit, why = vision.teach_distance(shared.get("obs") or [], float(q.get("mm", 0)))
                    if fit is None:
                        msg = why
                    else:
                        a = fit["aspect"]
                        msg = (f"taught from {len(fit['check'])} tips: screen shape {a:.2f}:1"
                               + (" (16:9)" if abs(a - 16 / 9) < 0.04 else " (16:10)" if abs(a - 1.6) < 0.04 else
                                  " (3:2)" if abs(a - 1.5) < 0.04 else "")
                               + f", readings now within ~{fit['rms_mm']:.1f} mm: "
                               + ", ".join(f"{c['true_mm']:.0f}->{c['read_mm']:.0f}" for c in fit["check"]))
                    print("live view:", msg)
                    return self._send(json.dumps({"msg": msg}).encode(), "application/json")
                if p == "/reset":
                    return self._send(json.dumps(on_reset()).encode(), "application/json")
            except Exception as e:
                return self._send(json.dumps({"msg": f"error: {e}"}).encode(), "application/json")
            if p == "/set":
                q = self.path.split("?", 1)[1] if "?" in self.path else ""
                return self._send(json.dumps(control(q)).encode(), "application/json")
            if p.startswith("/cam") and p.endswith(".jpg"):
                i = int(p[4:-4]) if p[4:-4].isdigit() else 0
                blank = cv2.imencode(".jpg", np.full((240, 320, 3), 34, np.uint8))[1].tobytes()
                return self._send(shared["jpg"].get(i, blank), "image/jpeg")
            self._send(LIVE_PAGE.encode(), "text/html; charset=utf-8")

    srv = ThreadingHTTPServer(("0.0.0.0", port), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    print(f"Live view: open  http://localhost:{port}/  in your browser.  Ctrl+C to stop.")
    loop_fps, last = 0.0, time.time()
    est = {"cams": 0, "landing": None}
    try:
        while True:
            if shared["pause"]:  # frozen picture: just redraw (marks, mask) so clicks show up
                for i, fr in enumerate(vision.last_frames):
                    if fr[0] is not None and fr[1] is not None:
                        vision.last_frames[i] = (fr[0], fr[1], find_tip(fr[0], vision.color, fr[1]))
            else:
                obs = vision.observe()
                est = vision.estimate(obs)
                shared["obs"] = obs
            for i, (f, quad, tip) in enumerate(vision.last_frames):
                if f is None:
                    continue
                k = min(1.0, 800 / f.shape[1])  # the browser only needs a small picture: faster
                d = cv2.resize(f, None, fx=k, fy=k, interpolation=cv2.INTER_AREA) if k < 1 else f.copy()
                th = max(2, d.shape[1] // 400)
                if shared["mask"]:  # tint everything it considers "pen colour"
                    m = cv2.resize(tip_mask(f, vision.color, quad), (d.shape[1], d.shape[0])) > 0
                    d[m] = (0.4 * d[m] + 0.6 * np.array([255, 255, 0])).astype(np.uint8)
                if i in shared["marks"]:  # your last click: cyan cross
                    mx, my = shared["marks"][i]
                    r = max(8, d.shape[1] // 70)
                    cv2.drawMarker(d, (int(mx * k), int(my * k)), (255, 255, 0), cv2.MARKER_CROSS, 2 * r, th)
                if quad is not None:
                    cv2.polylines(d, [(quad * k).astype(int)], True, (0, 255, 0), th)
                if tip is not None:
                    (x, y), _ = tip
                    cv2.circle(d, (int(x * k), int(y * k)), max(10, d.shape[1] // 60), (255, 0, 255), th + 1)
                msg = (f"cam {i + 1}{' (paused)' if shared['pause'] else ''}: " + ("screen OK" if quad is not None else "NO SCREEN")
                       + (", tip OK" if tip is not None else ", NO TIP"))
                cv2.putText(d, msg, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4, cv2.LINE_AA)
                cv2.putText(d, msg, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
                shared["jpg"][i] = cv2.imencode(".jpg", d, [cv2.IMWRITE_JPEG_QUALITY, 80])[1].tobytes()
            st = {"cams_seen": est["cams"], "uv": None, "height_mm": None, "gap_mm": None}
            if est.get("mode") == "3d":
                st.update(uv=list(est["tip_uv"]), height_mm=est["height_mm"], gap_mm=est["gap_mm"],
                          status="cameras disagree" if est.get("inconsistent") else "3D OK")
            elif est["landing"] is not None:
                st.update(uv=list(est["landing"]), status="1 camera: no height")
            else:
                st["status"] = ("no camera sees the tip" if all(fr[1] is not None for fr in vision.last_frames)
                                else "screen markers not visible")
            # camera placement: each camera's angle to the screen + the angle between the two
            geo = [view_geometry(q, f.shape, vision.aspect) if (f is not None and q is not None) else None
                   for f, q, _ in vision.last_frames]
            st["cam_tilt"] = [round(g["tilt_deg"]) if g else None for g in geo]
            if len(geo) == 2 and all(geo):
                ang = float(np.degrees(np.arccos(np.clip(np.dot(geo[0]["dir"], geo[1]["dir"]), -1, 1))))
                st["pair_deg"] = round(ang)
                st["placement"] = ("good" if ang >= 25 else "OK, height rough" if ang >= 12
                                   else "too similar: move one camera")
            st["cam_fps"] = [round(s.fps) if s.kind == "mjpeg" else None for s in srcs]
            # frame-rate cap: wait out the rest of this frame's time slot
            time.sleep(max(0.0, 1 / shared["fps"] - (time.time() - last)))
            now = time.time()
            loop_fps = 0.8 * loop_fps + 0.2 / max(now - last, 1e-3)
            last = now
            st["loop_fps"] = round(loop_fps, 1)
            st["aspect"] = round(vision.aspect, 3)
            shared["state"] = st
    except KeyboardInterrupt:
        pass
    finally:
        srv.shutdown()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--cameras", default="", help="'none', one camera, or two separated by a comma")
    p.add_argument("--color", default="pink", help="pink, green, orange, yellow, blue, or 'h1,s1,v1,h2,s2,v2'")
    p.add_argument("--out", default="outputs")
    p.add_argument("--delay", type=float, default=0, help="seconds to switch to the page before the check")
    p.add_argument("--screen-mm", type=float, default=SCREEN_WIDTH_MM, help="marker-to-marker width of the screen")
    p.add_argument("--live", action="store_true", help="live view in the browser (no arm needed)")
    p.add_argument("--port", type=int, default=8766)
    a = p.parse_args()
    from common import countdown
    countdown(a.delay, "switch to the page with the corner markers (full-screen, F11)")
    v = TipVision(a.cameras, a.color, width_mm=a.screen_mm)
    if not v.enabled:
        raise SystemExit("No camera configured (CAMERA / CAMERA2 in lr.conf).")
    if a.live:
        live(v, a.port, a.out)
    else:
        check_setup(v, Path(a.out))
