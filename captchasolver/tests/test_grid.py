import pytest

from captchasolver.grid import Box, tile_centers, tiles_from_boxes, to_pixels

SIZE = (300, 300)  # 3x3 grid -> 100px tiles


def test_small_object_in_one_tile():
    assert tiles_from_boxes([Box(110, 10, 130, 50)], SIZE, 3, 3) == [1]


def test_big_object_spans_tiles_but_ignores_sliver():
    # covers cols 0-1 / rows 0-1 fully, plus a 3px sliver into col 2
    assert tiles_from_boxes([Box(0, 0, 203, 200)], SIZE, 3, 3) == [0, 1, 3, 4]


def test_normalized_boxes_are_scaled():
    (b,) = to_pixels([Box(0.5, 0.5, 1.0, 1.0)], SIZE)
    assert (b.x_min, b.y_max) == (150, 300)


def test_pixel_boxes_pass_through():
    boxes = [Box(10, 10, 120, 80)]
    assert to_pixels(boxes, SIZE) == boxes


def test_garbage_coordinates_raise():
    with pytest.raises(ValueError):
        to_pixels([Box(0, 0, 1000, 1000)], SIZE)


def test_no_boxes():
    assert tiles_from_boxes(to_pixels([], SIZE), SIZE, 3, 3) == []


def test_centers():
    assert tile_centers(SIZE, 3, 3)[4] == (150, 150)


def test_singular_and_classify_word_boundary():
    import re
    from captchasolver.grid import singular
    assert [singular(w) for w in ("cats", "bicycles", "buses", "flowers", "class")] == \
        ["cat", "bicycle", "bus", "flower", "class"]
    pat = re.compile(rf"\b{singular('cats')}s?\b")
    assert pat.search("tiger cat") and not pat.search("catamaran, trimaran")


def test_find_answer_shapes():
    import pytest
    from captchasolver.client import find_answer
    assert find_answer({"answer": "Yes"}) == "Yes"
    assert find_answer({"query": {"answer": "no"}}) == "no"
    assert find_answer("yes") == "yes"
    assert find_answer({"choices": [{"message": {"content": "yes"}}]}) == "yes"
    with pytest.raises(ValueError, match="no text answer"):
        find_answer({"objects": []})
