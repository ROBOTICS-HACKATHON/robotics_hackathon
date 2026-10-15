"""What is on the screen right now? Pixels only, no model call, no access to the page's code.

Input: the straightened screenshot from see.py (BGR numpy image, any size, normally 1280x800).

  grid      the image dialog: blue banner, 3x3 tiles, Verify. `state.grid` has its geometry;
            `state.error` is True when it shows the red "Incorrect selection" text.
  choice    the page with the three challenge cards         }  recognised by comparing a tiny blurred
  checkbox  the "I'm not a robot" widget                    }  copy of the screen with the copies taught
  verified  "Access granted" after a pass                   }  by `./lr captcha --learn-layout`
  unknown   anything else (loading, page changing, arm in the way)

The grid is found by colour and proportions (adapted from captchasolver/screen.py, kept dependency-free:
numpy + OpenCV only). It REFUSES rather than guesses: a robot must never press from a half-found layout.

The taught pages live in touch/site_layout.json (machine-specific, git-ignored): for each page its tiny
signature and, where the robot has to tap something (card, checkbox, "Start again"), the (u, v) to tap.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

LAYOUT_PATH = Path(__file__).resolve().parent / "site_layout.json"
SIG_W, SIG_H = 64, 40
PAGES = ("choice", "checkbox", "verified")   # pages taught with --learn-layout
TAP_PAGES = {"choice": "the reCAPTCHA card", "checkbox": "the 'I'm not a robot' checkbox",
             "verified": "the 'Start again' button"}


class GridNotFound(ValueError):
    """No usable image dialog. `code` is no_dialog | layout_inconsistent | cropped (same words as the solver)."""

    def __init__(self, message, code="no_dialog"):
        super().__init__(message)
        self.code = code


@dataclass
class Grid:
    header: tuple          # x, y, w, h of the blue banner
    box: tuple             # x, y, side, side of the 3x3 tile area
    verify: tuple          # x, y of the Verify button centre (px)
    ref_blue: tuple        # banner colour as seen in THIS photo (B, G, R)
    size: tuple            # (W, H) of the image the numbers refer to
    rows: int = 3
    cols: int = 3

    def tile_box(self, i):
        gx, gy, side, _ = self.box
        r, c = divmod(i, self.cols)
        tw, th = side / self.cols, side / self.rows
        return gx + c * tw, gy + r * th, tw, th

    def tile_center(self, i):
        x, y, tw, th = self.tile_box(i)
        return x + tw / 2, y + th / 2

    def tile_center_uv(self, i):
        x, y = self.tile_center(i)
        return x / self.size[0], y / self.size[1]


@dataclass
class PageState:
    kind: str = "unknown"
    grid: Grid = None
    error: bool = False
    scores: dict = field(default_factory=dict)
    note: str = ""

    def __str__(self):
        extra = " + red error text" if self.error else ""
        sc = ", ".join(f"{k} {v:.2f}" for k, v in self.scores.items())
        return f"{self.kind}{extra}" + (f" ({sc})" if sc else "") + (f" [{self.note}]" if self.note else "")


# ---------------------------------------------------------------------------------- the image dialog

def _blueness(img):
    a = img.astype(np.int16)  # BGR
    return a[..., 0] - np.maximum(a[..., 1], a[..., 2])


def _blue_mask(img):
    """Pixels that are 'the banner blue' IN THIS PHOTO: threshold = half the typical blueness of its
    clearly-blue pixels, so a dim, washed-out or tinted photo still works."""
    b = _blueness(img)
    strong = b[b > 25]
    if strong.size < 50:
        raise GridNotFound("No blue instruction banner found in the image.")
    return b > 0.5 * float(np.percentile(strong, 90))


def _first_block(idx, max_gap):
    breaks = np.where(np.diff(idx) > max_gap + 1)[0]
    return int(idx[0]), int(idx[breaks[0]] if breaks.size else idx[-1])


def _find_banner(mask):
    H, W = mask.shape
    per_row = mask.sum(1)
    peak = int(per_row.max())
    if peak < max(12, 0.02 * W):
        raise GridNotFound("No blue instruction banner found in the image.")
    wide = np.where(per_row >= 0.4 * peak)[0]
    y0, y1 = _first_block(wide, max(2, int(0.01 * H)))
    cols = np.where(mask[y0:y1 + 1].sum(0) > (y1 - y0) // 2)[0]
    if cols.size == 0:
        raise GridNotFound("No blue instruction banner found in the image.")
    return int(cols.min()), y0, int(cols.max()), y1


def find_grid(img):
    """Locate the dialog in an upright image: banner -> square grid under it -> Verify under the grid."""
    H, W = img.shape[:2]
    mask = _blue_mask(img)
    x0, y0, x1, y1 = _find_banner(mask)
    bw, bh = x1 - x0 + 1, y1 - y0 + 1
    side = bw
    if not 2.0 <= bw / bh <= 5.5 or side < max(30, 0.04 * W):
        raise GridNotFound("The blue block found does not look like the instruction banner.", "layout_inconsistent")

    # grid top = strongest horizontal edge just under the banner (white gap -> tile border)
    gray = img[:, x0:x1 + 1].astype(np.float32).mean(2)
    edge = np.abs(np.diff(gray, axis=0)).mean(1)
    lo, hi = y1 + 1 + int(0.008 * side), min(H - 2, y1 + 1 + int(0.045 * side))
    if hi <= lo:
        raise GridNotFound("The grid is not visible under the banner.", "cropped")
    gy = lo + int(np.argmax(edge[lo:hi])) + 1
    if gy + side > H:
        raise GridNotFound("The grid runs off the bottom of the image.", "cropped")

    # Verify = the narrower blue block under the grid, on the right. The error line pushes it down, so it is
    # searched in a band (not at one exact distance); it must still be narrower than the banner.
    r0, r1 = gy + int(0.985 * side), min(H, gy + int(1.45 * side))
    c0, c1 = x0 + int(0.45 * side), min(W, x1 + int(0.04 * side))
    sub = mask[r0:r1, c0:c1]
    if not sub.size or sub.sum(1).max() < 0.08 * side or sub.sum(1).max() > 0.6 * side:
        raise GridNotFound("The Verify button was not found under the grid.", "layout_inconsistent")
    rows = np.where(sub.sum(1) >= 0.5 * sub.sum(1).max())[0]
    vy0, vy1 = _first_block(rows, max(2, int(0.01 * H)))
    vcols = np.where(sub[vy0:vy1 + 1].sum(0) >= 0.5 * (vy1 - vy0 + 1))[0]
    if not vcols.size:
        raise GridNotFound("The Verify button was not found under the grid.", "layout_inconsistent")
    vx = c0 + (int(vcols.min()) + int(vcols.max())) // 2
    vy = r0 + (vy0 + vy1) // 2
    if vx < x0 + 0.5 * side:
        raise GridNotFound("The Verify button is on the wrong side of the dialog.", "layout_inconsistent")

    # banner colour in this photo = median of the banner's own pixels
    ref = np.median(img[y0 + 2:y1 - 1, x0 + 2:x1 - 1].reshape(-1, 3)[::7], axis=0)
    return Grid((x0, y0, bw, bh), (x0, gy, side, side), (vx, vy), tuple(float(c) for c in ref), (W, H))


def has_error_text(img, grid):
    """Red 'Incorrect selection. Try again.' line between the tiles and the footer."""
    gx, gy, side, _ = grid.box
    # start a little under the grid: the last row of tiles (a red bicycle...) must not count as text
    band = img[gy + side + int(0.025 * side): min(img.shape[0], gy + int(1.12 * side)), gx: gx + side].astype(np.int16)
    if band.size == 0:
        return False
    redness = band[..., 2] - np.maximum(band[..., 0], band[..., 1])
    return int((redness > 45).sum()) >= max(40, int(0.0008 * side * side))


# ---------------------------------------------------------------------------------- reading tile selection

@dataclass
class TileRead:
    index: int
    selected: bool        # blue check disc in the tile's top-left corner
    disc: float           # 0..1 how disc-like the best blue blob is
    shrunk: float         # 0..1 how white the tile's outer rim is (a selected image shrinks to 80%)

    @property
    def agree(self):
        return self.selected == (self.shrunk >= 0.5)


def _disc(img, grid, i):
    """Is there a blue circle with a white tick at tile i's top-left corner? Returns a 0..1 score.
    The window reaches a little outside the nominal tile corner: a photo that is a degree or two off, or
    slightly stretched, moves the corner by a few pixels."""
    tx, ty, tw, th = grid.tile_box(i)
    pad = int(0.09 * tw)
    x0, y0 = max(0, int(round(tx)) - pad), max(0, int(round(ty)) - pad)
    win = img[y0: int(round(ty)) + int(0.46 * th) + 2, x0: int(round(tx)) + int(0.46 * tw) + 2]
    if win.size == 0:
        return 0.0
    ref = np.array(grid.ref_blue, np.float32)
    dist = np.linalg.norm(win.astype(np.float32) - ref, axis=2)
    mask = (dist < 70).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    best = 0.0
    for k in range(1, n):
        x, y, w, h, area = stats[k]
        if not (0.09 * tw <= w <= 0.40 * tw and 0.09 * th <= h <= 0.40 * th):
            continue
        if not 0.7 <= w / h <= 1.4:
            continue
        fill = area / float(w * h)             # circle ~ 0.785, minus the white tick ~ 0.6-0.75
        if not 0.45 <= fill <= 0.88:
            continue
        # the disc sits in the very corner of the tile (allowing for the padding and a few px of drift)
        if x + x0 > tx + 0.10 * tw + 3 or y + y0 > ty + 0.10 * th + 3:
            continue
        if x + x0 < tx - pad - 1 or y + y0 < ty - pad - 1:
            continue
        shape = 1.0 - min(1.0, abs(fill - 0.70) / 0.25)
        best = max(best, 0.5 + 0.5 * shape)
    return best


def _rim(img, grid, i):
    """Selected tiles show the photo at 80%: a white rim all round. 0..1 = fraction of rim sides that are white."""
    tx, ty, tw, th = grid.tile_box(i)
    x0, y0, x1, y1 = int(tx), int(ty), int(tx + tw), int(ty + th)
    t = max(2, int(0.035 * tw))
    inset = int(0.012 * tw) + 1
    # skip the top-left corner (the disc) on the top and left strips
    skip = int(0.40 * tw)
    sides = [img[y0 + inset: y0 + inset + t, x0 + skip: x1 - inset],           # top
             img[y1 - inset - t: y1 - inset, x0 + inset: x1 - inset],         # bottom
             img[y0 + skip: y1 - inset, x0 + inset: x0 + inset + t],           # left
             img[y0 + inset: y1 - inset, x1 - inset - t: x1 - inset]]          # right
    white = 0
    for s in sides:
        if s.size and s.mean() > 232 and s.std() < 14:
            white += 1
    return white / 4.0


def read_selection(img, grid):
    """Per tile: is it selected? (blue disc is the evidence; the white rim is a second opinion.)"""
    out = []
    for i in range(grid.rows * grid.cols):
        d, r = _disc(img, grid, i), _rim(img, grid, i)
        out.append(TileRead(i, d >= 0.5, d, r))
    return out


def selected_set(reads):
    return {r.index for r in reads if r.selected}


def check_presses_against_grid(presses, grid, max_tile_frac=0.30):
    """The solver's tile centres must agree with where THIS image's grid has those tiles. Returns the worst
    miss as a fraction of a tile (the caller refuses to tap above max_tile_frac)."""
    worst = 0.0
    tw = grid.box[2] / grid.cols
    W, H = grid.size
    for p in presses:
        if p.get("role") != "tile":
            continue
        cx, cy = grid.tile_center(p["tile"])
        worst = max(worst, float(np.hypot(p["x_norm"] * W - cx, p["y_norm"] * H - cy)) / tw)
    return worst, worst <= max_tile_frac


# ---------------------------------------------------------------------------------- taught pages

def signature(img):
    """Tiny 'edge map' of the screen, normalised. Edges (not brightness) because the checkbox page is the choice
    page blurred: only the structure tells them apart, and it survives a camera's brightness and noise.
    The four marker squares are blanked (every page has them)."""
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    H, W = g.shape
    g = g.copy()
    m = int(0.2 * H)
    for ys, xs in ((slice(0, m), slice(0, m)), (slice(0, m), slice(W - m, W)),
                   (slice(H - m, H), slice(0, m)), (slice(H - m, H), slice(W - m, W))):
        g[ys, xs] = 255
    g = cv2.GaussianBlur(cv2.resize(g, (256, 160), interpolation=cv2.INTER_AREA).astype(np.float32), (0, 0), 1.0)
    e = np.sqrt(cv2.Sobel(g, cv2.CV_32F, 1, 0) ** 2 + cv2.Sobel(g, cv2.CV_32F, 0, 1) ** 2)
    e = cv2.GaussianBlur(np.sqrt(cv2.resize(e, (SIG_W, SIG_H), interpolation=cv2.INTER_AREA)), (0, 0), 1.0)
    e -= e.mean()
    n = np.linalg.norm(e)
    return (e / n).ravel() if n > 1e-6 else e.ravel()


def local_signature(img, tap, cw=360, ch=180):
    """Contrast-free look at the neighbourhood of a page's tap target (a 360x180 window of the 1280x800 screen).
    The checkbox page is a small dialog on a washed-out backdrop: in a dim, blurry photo its whole-screen edge map
    is mostly noise, but the dialog itself (an NCC of a band-passed crop) stays unmistakable."""
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    H, W = g.shape
    x0 = int(np.clip(tap[0] * W - cw * W / 1280 / 2, 0, W - cw * W / 1280))
    y0 = int(np.clip(tap[1] * H - ch * H / 800 / 2, 0, H - ch * H / 800))
    c = g[y0:y0 + int(ch * H / 800), x0:x0 + int(cw * W / 1280)]
    c = cv2.resize(c, (72, 36), interpolation=cv2.INTER_AREA).astype(np.float32)
    c = cv2.GaussianBlur(c, (0, 0), 0.8) - cv2.GaussianBlur(c, (0, 0), 4)
    c -= c.mean()
    n = np.linalg.norm(c)
    return (c / n).ravel() if n > 1e-6 else c.ravel()


LOCAL_MIN, LOCAL_GAP = 0.60, 0.30


class SiteLayout:
    """The taught pages: {name: {"sig": [...], "tap": [u, v]}} in touch/site_layout.json."""

    def __init__(self, pages=None, path=LAYOUT_PATH):
        self.pages, self.path = pages or {}, Path(path)

    @classmethod
    def load(cls, path=LAYOUT_PATH):
        try:
            d = json.loads(Path(path).read_text())
            return cls({k: v for k, v in d.get("pages", {}).items() if "sig" in v}, path)
        except (OSError, ValueError):
            return cls({}, path)

    def save(self):
        self.path.write_text(json.dumps({"version": 1, "pages": self.pages}))

    def teach(self, name, img, tap=None):
        page = {"sig": [round(float(x), 5) for x in signature(img)]}
        if tap is not None:
            page["tap"] = [round(float(tap[0]), 5), round(float(tap[1]), 5)]
            page["win"] = [round(float(x), 5) for x in local_signature(img, tap)]
        self.pages[name] = page

    def tap(self, name):
        t = self.pages.get(name, {}).get("tap")
        return (t[0], t[1]) if t else None

    def start_tap(self):
        """Where 'Start session' is on the choice (Training) page, if taught (--learn-layout --only start)."""
        t = self.pages.get("choice", {}).get("start")
        return (t[0], t[1]) if t else None

    def missing(self):
        return [n for n in PAGES if n not in self.pages or (n in TAP_PAGES and "tap" not in self.pages[n])]

    def match(self, img, min_score=0.50, min_margin=0.20):
        """(best page name or None, {name: score}). Signatures are compared after subtracting what all taught
        pages share (the layout every page has), which is what makes choice vs checkbox clear-cut. Needs a
        clear winner. With fewer than 3 taught pages the plain correlation is used, with a stricter bar."""
        if not self.pages:
            return None, {}
        names = list(self.pages)
        s = signature(img)
        P = np.array([self.pages[n]["sig"] for n in names], np.float32)
        if len(names) >= 3:
            mu, bar = P.mean(0), (min_score, min_margin)
        else:
            mu, bar = 0.0, (0.90, 0.10)
        a = s - mu
        na = np.linalg.norm(a)
        if na < 1e-6:
            return None, {}
        sc = {}
        for n, p in zip(names, P):
            b = p - mu
            nb = np.linalg.norm(b)
            sc[n] = float(np.dot(a, b) / (na * nb)) if nb > 1e-6 else 0.0
        ranked = sorted(sc.items(), key=lambda kv: -kv[1])
        best, top = ranked[0]
        second = ranked[1][1] if len(ranked) > 1 else -1.0
        glob = best if top >= bar[0] and top - second >= bar[1] else None
        loc = self._local_winner(img)
        if loc and glob in (None, loc):
            return loc, sc
        if glob and loc is None:
            return glob, sc
        return None, sc

    def _local_winner(self, img):
        """The page whose tap-target neighbourhood matches clearly (and the others clearly don't), else None.
        Only pages taught with a tap point (and in this version) take part."""
        loc = {}
        for n, pg in self.pages.items():
            if "win" in pg and "tap" in pg:
                loc[n] = float(np.dot(local_signature(img, pg["tap"]), np.array(pg["win"], np.float32)))
        if len(loc) < 2:
            return None
        ranked = sorted(loc.items(), key=lambda kv: -kv[1])
        (b, top), (_, second) = ranked[0], ranked[1]
        return b if top >= LOCAL_MIN and top - second >= LOCAL_GAP else None


def classify(img, layout=None):
    """Grid first (it must satisfy every geometry check), then the taught pages, else unknown."""
    try:
        grid = find_grid(img)
    except GridNotFound as e:
        grid, why = None, str(e)
    else:
        return PageState("grid", grid, has_error_text(img, grid))
    if layout is not None:
        name, scores = layout.match(img)
        if name:
            return PageState(name, scores=scores)
        return PageState("unknown", scores=scores, note=f"no taught page matches; {why}")
    return PageState("unknown", note=why)
