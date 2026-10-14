"""How the harness decides the site's answer after Verify. A slow site must never become a FAIL."""
import asyncio
import time

import pytest
from playwright.async_api import async_playwright

from captchasolver import bench

PAGE = """<!doctype html><section id="image-challenge" data-state="idle"><button class="verify-button">Verify</button></section>
<script>
 const mode = location.hash.slice(1), sec = document.getElementById('image-challenge');
 document.querySelector('.verify-button').onclick = () => {
   if (mode === 'error') sec.dataset.state = 'error';
   if (mode === 'success') { sec.dataset.state = 'success'; setTimeout(() => location.assign('/verified/'), 100); }
   if (mode === 'nav') setTimeout(() => location.assign('/verified/'), 100);   // like the real site: redirect only
 };
</script>"""


async def run(mode: str, nav_delay: float, timeout_ms: int):
    """Open the mock challenge, press Verify, ask _outcome. The /verified/ page is served after nav_delay."""
    async with async_playwright() as pw:
        browser = await pw.chromium.launch()
        page = await browser.new_page()

        async def handler(route):
            if route.request.url.rstrip("/").endswith("/verified"):
                await asyncio.sleep(nav_delay)                      # a slow redirect: the bug's trigger
                await route.fulfill(body="<h1>Verified</h1>", content_type="text/html")
            else:
                await route.fulfill(body=PAGE, content_type="text/html")
        await page.route("**/*", handler)
        await page.goto(f"http://mock.test/#{mode}")
        await page.click(".verify-button")
        t0 = time.perf_counter()
        try:
            return await bench._outcome(page, timeout_ms=timeout_ms), time.perf_counter() - t0
        finally:
            await browser.close()


async def test_rejection_is_reported_at_once_not_after_the_timeout():
    verdict, took = await run("error", nav_delay=0, timeout_ms=20000)
    assert verdict is False and took < 3


async def test_acceptance_does_not_wait_for_a_slow_redirect():
    verdict, took = await run("success", nav_delay=6, timeout_ms=20000)       # the redirect takes 6 s
    assert verdict is True and took < 3                                          # but the success state is seen at once


async def test_a_slow_redirect_alone_is_still_a_pass_not_a_fail():
    # This is the real bug: the site only navigates (after 'accepting'); under load it took longer than 5 s.
    verdict, took = await run("nav", nav_delay=7, timeout_ms=20000)       # slower than the old 5 s limit
    assert verdict is True


async def test_silence_is_an_error_never_a_fail():
    with pytest.raises(bench.SiteDidNotAnswer):
        await run("silent", nav_delay=0, timeout_ms=1500)


async def test_a_redirect_slower_than_the_timeout_is_an_error_not_a_fail():
    with pytest.raises(bench.SiteDidNotAnswer):
        await run("nav", nav_delay=6, timeout_ms=1500)


async def test_the_after_screenshot_waits_for_a_slow_redirect_so_it_is_not_blank(tmp_path):
    """A passing run: Verify starts a slow redirect. The 'after' image must be the page it lands on."""
    async with async_playwright() as pw:
        browser = await pw.chromium.launch()
        page = await browser.new_page()

        async def handler(route):
            if route.request.url.rstrip("/").endswith("/verified"):
                await asyncio.sleep(2.5)
                await route.fulfill(body="<body style='background:#1e8e3e;margin:0'><h1>Verified</h1></body>", content_type="text/html")
            else:
                await route.fulfill(body=PAGE, content_type="text/html")
        await page.route("**/*", handler)
        await page.goto("http://mock.test/#nav")
        await page.click(".verify-button")
        assert await bench._outcome(page, timeout_ms=15000) is True          # the redirect has only just committed
        out = tmp_path / "after.png"
        url = await bench._save_after(page, out)
        await browser.close()
    assert url.rstrip("/").endswith("/verified")
    from PIL import Image
    import numpy as np
    px = np.asarray(Image.open(out).convert("RGB")).reshape(-1, 3)
    assert (np.abs(px - [30, 142, 62]).max(1) < 12).mean() > 0.9            # the green success page, not a blank one


def test_the_zip_has_one_folder_per_challenge_and_only_this_runs_files(tmp_path):
    import json
    import zipfile
    out = tmp_path / "run"
    out.mkdir()
    recs = [bench.Record(5, "cats-1", "train", subject="cats", target="cats", tiles=[0], clicked_ok=True, passed=True),
            bench.Record(9, "dogs-2", "train", subject="dogs", target="dogs", tiles=[1], clicked_ok=True)]
    for r in recs:
        for suffix in (".png", ".json", ".plan.png", ".after.png"):
            (out / f"{r.index:03d}-{r.label}{suffix}").write_bytes(b"x")
    (out / "042-other-1.png").write_bytes(b"x")                              # a stranger from an earlier run
    z = bench.bundle_zip(recs, out, tmp_path / "sub" / "example.zip", {"count": 2}, bench.summarize(recs))
    names = sorted(zipfile.ZipFile(z).namelist())
    assert names == ["005-cats-1/after.png", "005-cats-1/before.png", "005-cats-1/plan.png", "005-cats-1/solver.json",
                     "009-dogs-2/after.png", "009-dogs-2/before.png", "009-dogs-2/plan.png", "009-dogs-2/solver.json",
                     "README.txt", "report.json"]
    rep = json.loads(zipfile.ZipFile(z).read("report.json"))
    assert rep["summary"]["total"] == 2 and rep["config"] == {"count": 2}
    assert [r["label"] for r in rep["records"]] == ["cats-1", "dogs-2"]


def test_a_missing_file_does_not_break_the_zip(tmp_path):
    import zipfile
    out = tmp_path
    r = bench.Record(1, "cats-1", "train", error="TimeoutError: x")
    (out / "001-cats-1.png").write_bytes(b"x")                                 # the run died before JSON/after existed
    z = bench.bundle_zip([r], out, tmp_path / "e.zip", {}, bench.summarize([r]))
    assert sorted(zipfile.ZipFile(z).namelist()) == ["001-cats-1/before.png", "README.txt", "report.json"]
