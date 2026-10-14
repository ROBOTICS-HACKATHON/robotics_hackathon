import cv2
import numpy as np
import pytest

import pagestate as ps
from fakeworld import IMG, POS, SHOTS
from verify_selection import check_selection, observe_stable


@pytest.fixture(scope="module")
def layout(tmp_path_factory):
    lay = ps.SiteLayout({}, tmp_path_factory.mktemp("l") / "site_layout.json")
    for name in ps.PAGES:
        lay.teach(name, IMG[name], POS[name])
    return lay


def degrade(img, blur=1.5, bright=0.85, noise=6, rot=0.7, seed=0):
    rng = np.random.default_rng(seed)
    d = cv2.GaussianBlur(img, (0, 0), blur).astype(np.float32) * bright + 255 * (1 - bright) * 0.3
    d = np.clip(d + rng.normal(0, noise, img.shape), 0, 255).astype(np.uint8)
    m = cv2.getRotationMatrix2D((640, 400), rot, 1.0)
    return cv2.warpAffine(d, m, (img.shape[1], img.shape[0]), borderValue=(255, 255, 255))


@pytest.mark.parametrize("name,kind", [("choice", "choice"), ("checkbox", "checkbox"), ("verified", "verified"),
                                       ("grid_empty", "grid"), ("grid_three_selected", "grid"),
                                       ("grid_all_selected", "grid"), ("grid_error", "grid")])
def test_classify_real_screenshots(layout, name, kind):
    assert ps.classify(IMG[name], layout).kind == kind


def test_error_text_only_when_the_site_shows_it(layout):
    errors = {n: ps.classify(IMG[n], layout).error for n in IMG if n.startswith("grid_")}
    assert errors == {"grid_empty": False, "grid_three_selected": False, "grid_all_selected": False,
                      "grid_tiles_0_4": False, "grid_error": True}


def test_choice_and_checkbox_are_not_confused_even_when_the_photo_is_poor(layout):
    for name in ps.PAGES:
        assert ps.classify(degrade(IMG[name]), layout).kind == name


def test_grid_found_in_a_degraded_photo():
    for name in ("grid_empty", "grid_three_selected"):
        st = ps.classify(degrade(IMG[name], blur=2.0, rot=1.0))
        assert st.kind == "grid"


def test_unrelated_screens_are_unknown(layout):
    assert ps.classify(np.full((800, 1280, 3), 255, np.uint8), layout).kind == "unknown"
    assert ps.classify(np.zeros((800, 1280, 3), np.uint8), layout).kind == "unknown"
    noise = np.random.default_rng(1).integers(0, 255, (800, 1280, 3), dtype=np.uint8)
    assert ps.classify(noise, layout).kind == "unknown"


def test_without_taught_pages_only_the_grid_is_recognised():
    assert ps.classify(IMG["grid_empty"]).kind == "grid"
    assert ps.classify(IMG["choice"]).kind == "unknown"


def test_grid_geometry_matches_the_real_page():
    """Tile centres and Verify found from pixels agree with where the DOM says they are (within 10 px: the photo's crop at the markers stretches the page ~2%)."""
    g = ps.find_grid(IMG["grid_empty"])
    for i, (u, v) in enumerate(POS["tiles"]):
        cx, cy = g.tile_center(i)
        assert abs(cx - u * 1280) < 10 and abs(cy - v * 800) < 10, i
    assert abs(g.verify[0] - POS["verify"][0] * 1280) < 12 and abs(g.verify[1] - POS["verify"][1] * 800) < 12


@pytest.mark.parametrize("name,expected", [("grid_empty", set()), ("grid_tiles_0_4", {0, 4}),
                                           ("grid_three_selected", {0, 4, 8}), ("grid_all_selected", set(range(9))),
                                           ("grid_error", {0, 4})])
def test_selection_is_read_correctly(name, expected):
    chk = check_selection(IMG[name], expected)
    assert chk.observed == expected and chk.ok and not chk.to_tap


def test_selection_survives_a_poor_photo():
    for name, expected in (("grid_three_selected", {0, 4, 8}), ("grid_empty", set()), ("grid_all_selected", set(range(9)))):
        for seed in range(3):
            assert check_selection(degrade(IMG[name], blur=2.0, rot=0.8, seed=seed), expected).ok, (name, seed)


def test_difference_lists_only_what_has_to_be_tapped():
    chk = check_selection(IMG["grid_three_selected"], {0, 4})     # 8 is selected but should not be
    assert not chk.ok and chk.extra == {8} and chk.missing == set() and chk.to_tap == [8]
    chk = check_selection(IMG["grid_tiles_0_4"], {0, 4, 8})
    assert chk.missing == {8} and chk.to_tap == [8]


def test_a_blue_sky_or_blue_banner_is_not_a_check_disc():
    img = IMG["grid_empty"].copy()
    g = ps.find_grid(img)
    x, y, tw, th = g.tile_box(2)
    img[int(y):int(y + 0.3 * th), int(x):int(x + 0.5 * tw)] = np.array(g.ref_blue, np.uint8)   # solid blue block
    assert 2 not in ps.selected_set(ps.read_selection(img, g))


def test_presses_must_agree_with_the_grid():
    g = ps.find_grid(IMG["grid_empty"])
    ok = [{"role": "tile", "tile": t, "x_norm": POS["tiles"][t][0], "y_norm": POS["tiles"][t][1]} for t in (0, 4)]
    assert ps.check_presses_against_grid(ok, g)[1]
    bad = [{"role": "tile", "tile": 0, "x_norm": POS["tiles"][1][0], "y_norm": POS["tiles"][1][1]}]   # tile 1's spot
    assert not ps.check_presses_against_grid(bad, g)[1]


def test_observe_stable_needs_two_agreeing_looks():
    seq = [IMG["grid_tiles_0_4"], IMG["grid_three_selected"], IMG["grid_three_selected"]]
    it = iter(seq)
    chk = observe_stable(lambda: next(it), {0, 4, 8}, sleep=lambda s: None)
    assert chk.ok
    it = iter([None, None, None, None, None])
    assert observe_stable(lambda: next(it), {0}, sleep=lambda s: None) is None


def test_layout_roundtrip_and_missing(tmp_path, layout):
    p = tmp_path / "l.json"
    lay = ps.SiteLayout({}, p)
    assert lay.missing() == list(ps.PAGES)
    lay.teach("choice", IMG["choice"], (0.3, 0.2))
    lay.save()
    again = ps.SiteLayout.load(p)
    assert again.tap("choice") == (0.3, 0.2) and again.missing() == ["checkbox", "verified"]
    assert ps.SiteLayout.load(tmp_path / "nope.json").pages == {}


def test_dim_washed_out_checkbox_is_still_recognised_by_its_dialog(layout):
    """Regression: a dim, blurry, noisy photo of the small checkbox dialog gave 'unknown' with the edge map alone."""
    img = degrade(IMG["checkbox"], blur=2.5, bright=0.6, noise=10, rot=0.5)
    assert ps.classify(img, layout).kind == "checkbox"
    assert ps.classify(degrade(IMG["choice"], blur=2.5, bright=0.6, noise=10, rot=0.5), layout).kind == "choice"
