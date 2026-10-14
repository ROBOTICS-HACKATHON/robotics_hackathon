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
    """A preset name, or a custom HSV range 'h1,s1,v1,h2,s2,v2' (OpenCV: H 0-179)."""
    if color in COLORS:
        return COLORS[color]
    n = [int(x) for x in str(color).split(",")]
    if len(n) != 6:
        raise SystemExit(f"Unknown tip colour '{color}': use {', '.join(COLORS)} or 'h1,s1,v1,h2,s2,v2'")
    return [(tuple(n[:3]), tuple(n[3:]))]


def find_tip(frame, color, quad=None, min_area=12):
    """Where the stylus tip is, in camera pixels. Returns ((x, y), area) or None.

    Finds the biggest blob of `color` near the screen. If the coloured part is long (e.g. the
    whole stylus body is coloured), the tip is the END of that blob that points into the screen,
    not its middle."""
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
    n, labels, stats, cents = cv2.connectedComponentsWithStats(mask)
    if n <= 1:
        return None
    i = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    area = int(stats[i, cv2.CC_STAT_AREA])
    if area < min_area:
        return None
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


class TipVision:
    def __init__(self, cameras, color="pink", cid=None, aspect=SCREEN_ASPECT, width_mm=SCREEN_WIDTH_MM,
                 max_gap_mm=12.0):
        self.color = color
        self.aspect, self.width_mm, self.max_gap = aspect, width_mm, max_gap_mm
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
        self.last_frames = []
        for cam in self.cams:
            try:
                f = cam.frame()
                quad, _ = cam.locate(f)
                tip = find_tip(f, self.color, quad) if quad is not None else None
                self.last_frames.append((f, quad, tip))
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
                self.last_frames.append((None, None, None))
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

    def _pose(self, i, o):
        quad = np.array(o["quad"], np.float32)
        f = estimate_focal(quad, o["shape"], self.aspect)
        hist = self.focal.setdefault(i, [])
        if f:
            hist.append(f)
            del hist[:-30]
        fm = float(np.median(hist)) if hist else max(o["shape"]) / (2 * np.tan(np.radians(32.5)))
        return camera_pose(quad, o["shape"], fm, self.aspect, self.width_mm)

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
        return {"uv": (float(P[0] / W), float(P[1] / H)), "height_mm": float(side * P[2]), "gap_mm": gap,
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
 img{width:100%;border-radius:6px;background:#222}
 #map{position:relative;aspect-ratio:1.513;background:#fff;border-radius:6px;max-width:420px}
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
<main><div><img id="c0"></div><div><img id="c1"></div>
<div><span class="lbl">where the tip is over the screen</span><div id="map"><div id="dot"></div></div></div></main>
<script>
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
  }catch(e){ document.getElementById('st').textContent='waiting...'; }
  setTimeout(tick, 200);
}
tick();
</script></body></html>"""


def live(vision, port=8766):
    """Live test without the arm: both camera feeds with the screen + tip drawn, and the 3D reading."""
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    shared = {"jpg": {}, "state": {"status": "starting"}}

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
            if p.startswith("/cam") and p.endswith(".jpg"):
                i = int(p[4:-4]) if p[4:-4].isdigit() else 0
                blank = cv2.imencode(".jpg", np.full((240, 320, 3), 34, np.uint8))[1].tobytes()
                return self._send(shared["jpg"].get(i, blank), "image/jpeg")
            self._send(LIVE_PAGE.encode(), "text/html; charset=utf-8")

    srv = ThreadingHTTPServer(("0.0.0.0", port), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    print(f"Live view: open  http://localhost:{port}/  in your browser.  Ctrl+C to stop.")
    try:
        while True:
            obs = vision.observe()
            est = vision.estimate(obs)
            for i, (f, quad, tip) in enumerate(vision.last_frames):
                if f is None:
                    continue
                d = f.copy()
                th = max(2, f.shape[1] // 400)
                if quad is not None:
                    cv2.polylines(d, [quad.astype(int)], True, (0, 255, 0), th)
                if tip is not None:
                    (x, y), _ = tip
                    cv2.circle(d, (int(x), int(y)), max(10, f.shape[1] // 60), (255, 0, 255), th + 1)
                msg = (f"cam {i + 1}: " + ("screen OK" if quad is not None else "NO SCREEN")
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
            shared["state"] = st
            time.sleep(0.05)
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
        live(v, a.port)
    else:
        check_setup(v, Path(a.out))
