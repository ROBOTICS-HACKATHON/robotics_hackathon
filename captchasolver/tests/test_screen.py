from pathlib import Path

import pytest
from PIL import Image

from captchasolver import grid, screen
from captchasolver.batch import solve_dir

EXAMPLE = Path(__file__).parent.parent / "examples" / "1.png"
needs_example = pytest.mark.skipif(not EXAMPLE.exists(), reason="examples/1.png missing")


@needs_example
def test_layout_on_real_screenshot():
    lay = screen.find_layout(Image.open(EXAMPLE))
    assert lay.header == (686, 160, 517, 153)
    gx, gy, gw, gh = lay.grid
    assert (gx, gw, gh) == (686, 517, 517) and 320 <= gy <= 326
    assert lay.verify is not None and abs(lay.verify[0] - 1135) < 5 and abs(lay.verify[1] - 890) < 5


def test_no_dialog_raises():
    with pytest.raises(screen.LayoutNotFound):
        screen.find_layout(Image.new("RGB", (800, 600), "white"))


@needs_example
async def test_solve_screenshot_end_to_end_with_fake_model(monkeypatch):
    # Pretend the model reads "flowers" and says yes only for tiles 2, 4, 6.
    async def fake_vlm(ai, image, question, **kw):
        if "banner" in question:
            return "Flowers."
        fake_vlm.calls += 1
        return "yes" if (fake_vlm.calls - 1) in (2, 4, 6) else "no"
    fake_vlm.calls = 0

    async def ordered_gather(*coros):  # keep call order == tile order for the fake
        return [await c for c in coros]
    monkeypatch.setattr(grid, "vlm_query", fake_vlm)
    monkeypatch.setattr(screen, "vlm_query", fake_vlm)
    monkeypatch.setattr(grid.asyncio, "gather", ordered_gather)

    res, _, lay = await screen.solve_screenshot(None, EXAMPLE.read_bytes())
    assert res["target"] == "flowers"
    assert res["tiles"] == [2, 4, 6]
    # tile 2 = row 0 / col 2 -> centre must be inside the screenshot's grid box, right column
    (x, y) = res["click"][0]
    assert 686 + 2 * 517 // 3 < x < 686 + 517 and lay.grid[1] < y < lay.grid[1] + 517 // 3
    assert res["verify"]


@needs_example
async def test_dir_reports_bad_screenshot_without_crashing(tmp_path):
    Image.new("RGB", (400, 300), "white").save(tmp_path / "blank.png")
    res = await solve_dir(None, tmp_path)
    assert "LayoutNotFound" in res["results"][0]["error"]


@needs_example
@pytest.mark.parametrize("width", [640, 960, 1280, 1914, 2560, 3840])
def test_layout_is_resolution_independent(width):
    base = Image.open(EXAMPLE)
    img = base.resize((width, round(base.height * width / base.width)), Image.LANCZOS)
    lay = screen.find_layout(img)
    ref = screen.find_layout(base)
    W, H = img.size
    # grid box and verify point agree with the full-size layout, in normalized units
    for got, want, dim in zip(lay.grid, ref.grid, (base.width, base.height) * 2):
        assert abs(got / (W if dim == base.width else H) - want / dim) < 0.006
    assert abs(lay.verify[0] / W - ref.verify[0] / base.width) < 0.006
    assert abs(lay.verify[1] / H - ref.verify[1] / base.height) < 0.006


@needs_example
@pytest.mark.parametrize("canvas,offset", [((2560, 1440), (300, 120)), ((1366, 1200), (-200, 60)),
                                           ((3440, 1440), (900, 0))])
def test_layout_other_screen_sizes_and_positions(canvas, offset):
    base = Image.open(EXAMPLE)
    img = Image.new("RGB", canvas, "white")
    img.paste(base, offset)
    lay = screen.find_layout(img)
    gx, gy, gw, gh = lay.grid
    ref = screen.find_layout(base).grid
    assert (gx, gy) == (ref[0] + offset[0], ref[1] + offset[1]) and (gw, gh) == (ref[2], ref[3])


@needs_example
async def test_result_has_normalized_coords_and_screen_size(monkeypatch):
    async def always_yes(ai, image, question, **kw):
        return "Flowers" if "banner" in question else "yes"
    monkeypatch.setattr(grid, "vlm_query", always_yes)
    monkeypatch.setattr(screen, "vlm_query", always_yes)
    res, _, _ = await screen.solve_screenshot(None, EXAMPLE.read_bytes())
    W, H = res["image_size"]
    assert (W, H) == (1914, 1080)
    for (x, y), (nx, ny) in zip(res["click"], res["click_norm"]):
        assert abs(nx * W - x) < 1 and abs(ny * H - y) < 1
    assert len(res["tiles"]) == 9
