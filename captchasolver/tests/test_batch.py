import io

import pytest
from PIL import Image

from captchasolver import grid
from captchasolver.batch import list_pngs, solve_dir


def png(path, color="white", size=(60, 60)):
    Image.new("RGB", size, color).save(path, "PNG")


async def fake_vlm(ai, image, question, **kw):
    # "click" red images, ignore everything else
    px = Image.open(io.BytesIO(image)).convert("RGB").getpixel((1, 1))
    return "yes" if px == (255, 0, 0) else "no"


async def test_tile_folder(tmp_path, monkeypatch):
    monkeypatch.setattr(grid, "vlm_query", fake_vlm)
    png(tmp_path / "tile10.png", "red"); png(tmp_path / "tile2.png", "blue")
    png(tmp_path / "tile1.png", "red")
    (tmp_path / "notes.txt").write_text("ignored")
    res = await solve_dir(None, tmp_path, "cats", layout="tiles")
    assert [r["file"] for r in res["results"]] == ["tile1.png", "tile2.png", "tile10.png"]
    assert res["click"] == ["tile1.png", "tile10.png"]


async def test_grid_folder(tmp_path, monkeypatch):
    monkeypatch.setattr(grid, "vlm_query", fake_vlm)
    img = Image.new("RGB", (60, 60), "blue")
    img.paste(Image.new("RGB", (30, 30), "red"), (30, 30))  # bottom-right of 2x2
    img.save(tmp_path / "g.png")
    res = await solve_dir(None, tmp_path, "cats", layout="grid", grid=(2, 2))
    assert res["results"] == [{"file": "g.png", "target": "cats", "tiles": [3], "click": [[45, 45]]}]


def test_empty_dir(tmp_path):
    with pytest.raises(FileNotFoundError):
        list_pngs(tmp_path)
