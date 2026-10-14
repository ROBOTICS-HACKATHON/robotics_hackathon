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
