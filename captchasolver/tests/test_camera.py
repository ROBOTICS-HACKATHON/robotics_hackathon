"""The dialog finder must cope with camera-like images (photo of a screen, already straightened to
screen size), and must REFUSE anything it is unsure of: a robot must never press a half-found layout.

Degradations are simulated on the real reference screenshot, and judged by what the robot presses
(Verify and the nine tile centres), against where those points truly are in the degraded image."""
import io
import math
import random

import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter

from captchasolver import screen
from captchasolver.grid import tile_centers

EXAMPLE = __import__("pathlib").Path(__file__).parent.parent / "examples" / "1.png"
pytestmark = pytest.mark.skipif(not EXAMPLE.exists(), reason="examples/1.png missing")

VERIFY_TOL, TILE_TOL = 18, 40      # px: Verify is ~57 px tall, a tile ~172 px


@pytest.fixture(scope="module")
def base():
    return Image.open(EXAMPLE).convert("RGB")


@pytest.fixture(scope="module")
def truth(base):
    ref = screen.find_layout(base)
    gx, gy, gw, _ = ref.grid
    return {"verify": ref.verify, **{f"t{i}": (gx + x, gy + y) for i, (x, y) in enumerate(tile_centers((gw, gw), 3, 3))}}


def tint(img, r, g, b):
    return Image.fromarray((np.asarray(img).astype(float) * [r, g, b]).clip(0, 255).astype("uint8"))


def noise(img, s, seed=0):
    rng = np.random.default_rng(seed)
    a = np.asarray(img).astype(float) + rng.normal(0, s, img.size[::-1] + (1,)) + rng.normal(0, s / 2, img.size[::-1] + (3,))
    return Image.fromarray(a.clip(0, 255).astype("uint8"))


def jpeg(img, q):
    b = io.BytesIO(); img.save(b, "JPEG", quality=q)
    return Image.open(io.BytesIO(b.getvalue())).convert("RGB")


def vignette(img, k):
    w, h = img.size
    y, x = np.mgrid[0:h, 0:w]
    d = ((x - w / 2) / (w / 2)) ** 2 + ((y - h / 2) / (h / 2)) ** 2
    return Image.fromarray((np.asarray(img).astype(float) * (1 - k * d[..., None])).clip(0, 255).astype("uint8"))


def warp(img, deg=0.0, key=0.0):
    """Rotate CCW, then perspective-skew. Returns the image and a fn: base point -> its true position."""
    w, h = img.size
    out = img.rotate(deg, resample=Image.BICUBIC, fillcolor=(255, 255, 255)) if deg else img
    g = key / w
    if key:
        out = out.transform((w, h), Image.PERSPECTIVE, (1, 0, 0, 0, 1, 0, g, 0), Image.BICUBIC, fillcolor=(255, 255, 255))

    def fwd(x, y):
        if deg:
            x, y = screen._rotate_point(x, y, deg, (w, h), back=False)
        if key:
            d = 1 - g * x
            x, y = x / d, y / d
        return x, y
    return out, fwd


def presses(img):
    """Where the robot would press (photo coordinates), through the same mapping solve_screenshot uses."""
    up, layout, tilt = screen.locate(img)
    back = (lambda x, y: screen._rotate_point(x, y, tilt, img.size, back=True)) if tilt else (lambda x, y: (x, y))
    gx, gy, gw, _ = layout.grid
    return {"verify": back(*layout.verify),
            **{f"t{i}": back(gx + x, gy + y) for i, (x, y) in enumerate(tile_centers((gw, gw), 3, 3))}}


def assert_accurate(img, truth, fwd=lambda x, y: (x, y)):
    got = presses(img)
    assert math.dist(got["verify"], fwd(*truth["verify"])) <= VERIFY_TOL
    for k in got:
        if k != "verify":
            assert math.dist(got[k], fwd(*truth[k])) <= TILE_TOL, k


# ---------------------------------------------------------------- conditions a camera adds
@pytest.mark.parametrize("name,make", [
    ("warm tint", lambda i: tint(i, 1.2, 1.0, 0.8)),
    ("cool tint", lambda i: tint(i, 0.85, 1.0, 1.15)),
    ("dim", lambda i: ImageEnhance.Brightness(i).enhance(0.4)),
    ("washed out", lambda i: ImageEnhance.Contrast(i).enhance(0.45)),
    ("soft focus", lambda i: i.filter(ImageFilter.GaussianBlur(7))),
    ("sensor noise", lambda i: noise(i, 25)),
    ("jpeg q35", lambda i: jpeg(i, 35)),
    ("vignette", lambda i: vignette(i, 0.6)),
    ("side lighting", lambda i: Image.fromarray((np.asarray(i).astype(float) * np.linspace(0.45, 1.15, i.size[0])[None, :, None]).clip(0, 255).astype("uint8"))),
    ("typical phone photo", lambda i: jpeg(noise(vignette(tint(i.filter(ImageFilter.GaussianBlur(1.5)), 1.06, 1, 0.94), 0.3), 6), 70)),
    ("rough photo", lambda i: jpeg(noise(vignette(tint(ImageEnhance.Contrast(i.filter(ImageFilter.GaussianBlur(3))).enhance(0.7), 1.15, 1, 0.85), 0.5), 12), 45)),
])
def test_photometric_conditions(base, truth, name, make):
    assert_accurate(make(base), truth)


@pytest.mark.parametrize("deg", [-6, -3, 1, 3, 6, 9])
def test_residual_tilt_is_estimated_and_removed(base, truth, deg):
    img, fwd = warp(base, deg)
    assert screen.estimate_tilt(img) == pytest.approx(-deg, abs=0.3)
    assert_accurate(img, truth, fwd)


@pytest.mark.parametrize("key", [-0.06, -0.03, 0.03, 0.06])
def test_residual_perspective_skew(base, truth, key):
    img, fwd = warp(base, 0, key)
    assert_accurate(img, truth, fwd)


def test_tilt_and_skew_together(base, truth):
    img, fwd = warp(base, 3, 0.03)
    assert_accurate(img, truth, fwd)


def test_random_camera_like_mixtures_are_never_wrong(base, truth):
    """Any outcome is acceptable EXCEPT confidently pressing the wrong place."""
    rng, wrong, ok = random.Random(7), [], 0
    for n in range(40):
        i = base.filter(ImageFilter.GaussianBlur(rng.uniform(0.5, 4)))
        i = tint(i, rng.uniform(.85, 1.2), 1.0, rng.uniform(.8, 1.15))
        i = ImageEnhance.Brightness(i).enhance(rng.uniform(.5, 1.25))
        i = ImageEnhance.Contrast(i).enhance(rng.uniform(.5, 1.2))
        i = vignette(i, rng.uniform(.1, .6))
        i, fwd = warp(i, rng.uniform(-3, 3), rng.uniform(-.03, .03))
        i = jpeg(noise(i, rng.uniform(2, 15), seed=n), rng.randint(35, 85))
        try:
            got = presses(i)
        except screen.LayoutNotFound:
            continue                                   # refusing is safe
        errs = [math.dist(got[k], fwd(*truth[k])) for k in got]
        if errs[0] > VERIFY_TOL or max(errs[1:]) > TILE_TOL:
            wrong.append(n)
        else:
            ok += 1
    assert not wrong, f"confidently wrong on mixtures {wrong}"
    assert ok >= 36                                    # and it should mostly work


# ---------------------------------------------------------------- when it must refuse
def box(img, xy, color):
    i = img.copy(); ImageDraw.Draw(i).rectangle(xy, fill=color); return i


@pytest.mark.parametrize("name,make,code", [
    ("arm over Verify", lambda i: box(i, (1050, 850, 1215, 925), (30, 30, 30)), "layout_inconsistent"),
    ("arm over banner", lambda i: box(i, (686, 160, 950, 312), (30, 30, 30)), "layout_inconsistent"),
    ("blue object above", lambda i: box(i, (300, 20, 1650, 120), (20, 90, 200)), "layout_inconsistent"),
    ("blue object below", lambda i: box(i, (300, 950, 1650, 1070), (20, 90, 200)), "layout_inconsistent"),
    ("blue block beside", lambda i: box(i, (50, 200, 600, 700), (20, 90, 200)), "layout_inconsistent"),
    ("no dialog at all", lambda i: Image.new("RGB", i.size, (200, 200, 200)), "no_dialog"),
])
def test_refuses_instead_of_guessing(base, name, make, code):
    with pytest.raises(screen.LayoutNotFound) as e:
        screen.locate(make(base))
    assert e.value.code == code


def test_refuses_excessive_tilt(base):
    img, _ = warp(base, 14)
    with pytest.raises(screen.LayoutNotFound, match="tilted"):
        screen.locate(img)


def test_cropped_dialog_is_refused_with_its_own_code(base):
    with pytest.raises(screen.LayoutNotFound) as e:
        screen.locate(base.crop((0, 0, base.width, 700)))      # grid runs off the bottom
    assert e.value.code in ("cropped", "layout_inconsistent")


# ---------------------------------------------------------------- geometry helper
def test_rotate_point_round_trips_and_matches_pil():
    size = (400, 300)
    img = Image.new("RGB", size, "white")
    ImageDraw.Draw(img).rectangle((300, 40, 310, 50), fill="black")      # a marker
    rot = img.rotate(20, resample=Image.BICUBIC, fillcolor=(255, 255, 255))
    ys, xs = np.where(np.asarray(rot).sum(2) < 200)
    seen = (xs.mean(), ys.mean())
    expect = screen._rotate_point(305, 45, 20, size, back=False)
    assert math.dist(seen, expect) < 2                                    # same convention as PIL
    back = screen._rotate_point(*expect, 20, size, back=True)
    assert math.dist(back, (305, 45)) < 1e-6
