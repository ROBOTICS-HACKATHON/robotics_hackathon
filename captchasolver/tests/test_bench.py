from captchasolver.bench import Record, select_challenges, split_of, summarize

OPTIONS = [(i, f"train-grid-{i}") for i in range(80)] + [(i, f"test-grid-{i}") for i in range(80, 100)]


def test_split_by_label():
    assert split_of("test-cats-1") == "test" and split_of("cats-1") == "train"
    assert split_of("train-grid-62") == "train"


def test_default_count_covers_all_100():
    assert len(select_challenges(OPTIONS, 100, "all", seed=0)) == 100


def test_split_filter_and_clamp():
    test = select_challenges(OPTIONS, 100, "test", seed=0)
    assert len(test) == 20 and all(split_of(lbl) == "test" for _, lbl in test)
    assert len(select_challenges(OPTIONS, 5, "train", seed=0)) == 5


def test_seed_is_reproducible_and_changes_order():
    a = select_challenges(OPTIONS, 10, "all", seed=1)
    assert a == select_challenges(OPTIONS, 10, "all", seed=1)
    assert a != select_challenges(OPTIONS, 10, "all", seed=2)
    assert len({i for i, _ in select_challenges(OPTIONS, 100, "all", seed=3)}) == 100  # no repeats


def test_summarize():
    recs = [
        Record(0, "a", "train", subject="cats", target="cats", passed=True, clicked_ok=True, seconds=2),
        Record(1, "b", "train", subject="cats", target="dogs", passed=False, clicked_ok=True, seconds=4),
        Record(2, "c", "test", subject="cars", target="cars", passed=True, clicked_ok=False, seconds=3),
        Record(3, "d", "test", error="TimeoutError: x", seconds=1),
    ]
    s = summarize(recs)
    assert (s["total"], s["passed"], s["errors"]) == (4, 2, 1)
    assert s["accuracy"] == 0.5
    assert s["banner_read_accuracy"] == round(2 / 3, 4)
    assert s["click_accuracy"] == round(2 / 3, 4)
    assert s["by_subject"] == {"cars": "1/1", "cats": "1/2"}


def test_empty_summary():
    assert summarize([])["accuracy"] == 0.0


def _records():
    return [
        Record(0, "cats-1", "train", subject="cats", target="cats", tiles=[0, 4], clicked_ok=True, passed=True),
        Record(1, "dogs-1", "train", subject="dogs", target="cats", tiles=[2], clicked_ok=True),
        Record(2, "cars-1", "test", subject="cars", target="cars", tiles=[1, 5], clicked_ok=False),
        Record(3, "birds-1", "test", subject="birds", target="birds", tiles=[3], clicked_ok=True),
        Record(4, "bad-1", "test", error="LayoutNotFound: no banner"),
    ]


def test_failure_causes():
    from captchasolver.bench import failure_cause
    r = _records()
    assert failure_cause(r[0]) == ""
    assert "banner misread" in failure_cause(r[1]) and "'cats'" in failure_cause(r[1])
    assert "clicks missed" in failure_cause(r[2])
    assert failure_cause(r[3]) == "wrong tile selection: picked [3]"
    assert failure_cause(r[4]).startswith("error:")


def test_summary_lists_failures_and_never_hides_them():
    from captchasolver.bench import format_summary
    recs = _records()
    text = format_summary(recs, summarize(recs), color=False)
    assert "1 passed" in text and "3 failed" in text and "1 error" in text
    assert "Failures (4)" in text and "No failures" not in text
    for label in ("dogs-1", "cars-1", "birds-1", "bad-1"):
        assert label in text
    assert "1 misread" in text and "1 off-target" in text
    assert "❌" in text and "💥" in text and "✅" in text
    assert "\033[" not in text          # plain when colour is off


def test_all_pass_says_no_failures_and_colour_toggles():
    from captchasolver.bench import format_summary, format_line
    ok = [Record(0, "cats-1", "train", subject="cats", target="cats", clicked_ok=True, passed=True)]
    assert "No failures" in format_summary(ok, summarize(ok), color=False)
    assert "\033[32m" in format_line(ok[0], 1, 1, color=True)       # green PASS
    assert "\033[31m" in format_line(_records()[1], 2, 5, color=True)  # red FAIL
    assert "\033[" not in format_line(_records()[1], 2, 5, color=False)


def test_fail_lines_carry_their_reason_on_a_second_line():
    from captchasolver.bench import format_line
    recs = _records()
    misread = format_line(recs[1], 2, 5, color=False).splitlines()
    assert len(misread) == 2 and "FAIL" in misread[0]
    assert misread[1].strip().startswith("↳ banner misread: read 'cats', page says 'dogs'")
    assert "clicks missed" in format_line(recs[2], 3, 5, color=False).splitlines()[1]
    assert "wrong tile selection: picked [3]" in format_line(recs[3], 4, 5, color=False).splitlines()[1]
    # passes and errors stay on one line (errors already show the reason inline)
    assert "\n" not in format_line(recs[0], 1, 5, color=False)
    err = format_line(recs[4], 5, 5, color=False)
    assert "\n" not in err and "LayoutNotFound" in err
    assert "\033[31m" in format_line(recs[1], 2, 5, color=True).splitlines()[1]  # reason is red too
