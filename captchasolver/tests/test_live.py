"""Live checks against the real demo site (network). Opt in with CAPTCHASOLVER_LIVE=1.

Kills Chromium in the middle of a run and checks the run recovers instead of breaking or hanging."""
import os
import signal
import sys

import pytest

from captchasolver import bench, grid, screen

pytestmark = pytest.mark.skipif(os.environ.get("CAPTCHASOLVER_LIVE") != "1" or not sys.platform.startswith("linux"),
                                reason="set CAPTCHASOLVER_LIVE=1 (needs network, Linux)")


def my_chromium_pids():
    """Chromium processes that descend from THIS process (never anyone else's)."""
    me, parent = os.getpid(), {}
    for entry in os.listdir("/proc"):
        if entry.isdigit():
            try:
                parent[int(entry)] = int(open(f"/proc/{entry}/stat").read().rsplit(")", 1)[1].split()[1])
            except OSError:
                pass

    def descends(pid):
        while pid in parent:
            pid = parent[pid]
            if pid == me:
                return True
        return False
    out = []
    for pid in parent:
        try:
            if descends(pid) and "chrom" in open(f"/proc/{pid}/cmdline").read():
                out.append(pid)
        except OSError:
            pass
    return out


async def test_run_recovers_when_chromium_is_killed_mid_run(monkeypatch):
    os.environ.setdefault("CLOUDFLARE_ACCOUNT_ID", "x")
    os.environ.setdefault("CLOUDFLARE_API_TOKEN", "x")
    killed = []

    async def fake(ai, image, question, **kw):
        if not killed:                                   # the first model call: pull the rug
            for pid in my_chromium_pids():
                os.kill(pid, signal.SIGKILL)
            killed.append(True)
        return "cats" if "banner" in question else "no"
    monkeypatch.setattr(grid, "vlm_query", fake)
    monkeypatch.setattr(screen, "vlm_query", fake)
    notes = []
    records, summary = await bench.run_tests(bench.DEFAULT_URL if hasattr(bench, "DEFAULT_URL") else "https://skibidisigma67rizz.pages.dev/",
                                             count=6, concurrency=2, fail_images=False, note=notes.append)
    assert killed, "the test never killed anything"
    assert len(records) == 6, "every challenge must produce a result"
    assert not [r for r in records if "closed" in r.error or "TargetClosed" in r.error], [r.error for r in records]
    assert any("Chromium" in n for n in notes), notes      # and it said what happened


async def test_run_stops_early_and_explains_when_chromium_keeps_dying(monkeypatch):
    """If Chromium cannot stay alive, a run must end quickly with the reason, not crawl for an hour."""
    import time
    os.environ.setdefault("CLOUDFLARE_ACCOUNT_ID", "x")
    os.environ.setdefault("CLOUDFLARE_API_TOKEN", "x")

    async def always_kill(ai, image, question, **kw):
        for pid in my_chromium_pids():
            os.kill(pid, signal.SIGKILL)
        return "cats" if "banner" in question else "no"
    monkeypatch.setattr(grid, "vlm_query", always_kill)
    monkeypatch.setattr(screen, "vlm_query", always_kill)
    notes, t0 = [], time.perf_counter()
    records, summary = await bench.run_tests("https://skibidisigma67rizz.pages.dev/", count=12, concurrency=2,
                                             fail_images=False, note=notes.append)
    assert time.perf_counter() - t0 < 120, "the run crawled instead of stopping"
    assert len(records) == 12                                          # every challenge is accounted for
    assert any("keeps crashing" in n for n in notes), notes
    assert any("BrowserKeepsCrashing" in r.error for r in records)     # and the reason names the next step
    assert all("/api/diagnostics" in r.error for r in records if "BrowserKeepsCrashing" in r.error)
