from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from captchasolver import diag
from captchasolver.server import api as appmod
from captchasolver.server.api import create_app

MB = 1024 * 1024


def write(root: Path, files: dict):
    for name, text in files.items():
        p = root / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)


def test_cgroup_v2_numbers_and_oom_kills(tmp_path):
    write(tmp_path, {"memory.current": str(1500 * MB), "memory.max": str(6144 * MB), "memory.peak": str(2300 * MB),
                     "memory.events": "low 0\nhigh 0\nmax 3\noom 2\noom_kill 2\n"})
    m = diag.memory_info(tmp_path)
    assert (m["current_mb"], m["limit_mb"], m["peak_mb"], m["oom_kill"]) == (1500, 6144, 2300, 2)


def test_cgroup_v2_without_a_limit_reads_as_none(tmp_path):
    write(tmp_path, {"memory.current": str(100 * MB), "memory.max": "max", "memory.events": "oom_kill 0\n"})
    m = diag.memory_info(tmp_path)
    assert m["limit_mb"] is None and m["oom_kill"] == 0


def test_cgroup_v1_is_understood_and_unlimited_is_not_a_number(tmp_path):
    write(tmp_path, {"memory/memory.usage_in_bytes": str(800 * MB), "memory/memory.limit_in_bytes": "9223372036854771712",
                     "memory/memory.max_usage_in_bytes": str(900 * MB), "memory/memory.oom_control": "oom_kill_disable 0\nunder_oom 0\noom_kill 1\n"})
    m = diag.memory_info(tmp_path)
    assert (m["current_mb"], m["limit_mb"], m["peak_mb"], m["oom_kill"]) == (800, None, 900, 1)


def test_missing_files_never_raise(tmp_path, monkeypatch):
    monkeypatch.setenv("CAPTCHASOLVER_PROC", str(tmp_path / "no-proc"))      # nothing to fall back on either
    m = diag.memory_info(tmp_path / "nowhere")
    assert m["current_mb"] is None and m["oom_kill"] is None
    assert isinstance(diag.hints(), str) and diag.hints()
    assert diag.disk_mb("/definitely/not/here") is None


def test_hints_name_the_numbers_that_explain_a_crash(tmp_path, monkeypatch):
    write(tmp_path, {"memory.current": str(5900 * MB), "memory.max": str(6144 * MB), "memory.peak": str(6100 * MB),
                     "memory.events": "oom_kill 3\n"})
    monkeypatch.setenv("CAPTCHASOLVER_CGROUP", str(tmp_path))
    text = diag.hints()
    assert "5900 MB used of 6144 MB" in text and "peak 6100" in text and "killed 3 process" in text


@pytest.fixture
def client():
    with TestClient(create_app()) as c:
        yield c


def test_diagnostics_endpoint_reports_resources_and_runs_a_real_chromium_selftest(client):
    r = client.get("/api/diagnostics")
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["chromium"]["ok"] is True, d["chromium"]
    assert any("launched" in s for s in d["chromium"]["steps"]) and any("screenshot ok" in s for s in d["chromium"]["steps"])
    assert {"before", "after", "oom_kills_during_test", "uptime_s", "hints"} <= set(d)
    assert "memory" in d["before"] and "dev_shm" in d["before"] and "tmp" in d["before"]


def test_diagnostics_reports_a_broken_chromium_instead_of_failing(client, monkeypatch):
    async def broken(stress):
        return {"ok": False, "seconds": 0.1, "steps": [], "error": "launch: browser died"}
    monkeypatch.setattr(appmod, "_chromium_selftest", broken)
    d = client.get("/api/diagnostics").json()
    assert d["chromium"]["ok"] is False and "died" in d["chromium"]["error"]


def test_only_one_diagnostics_run_at_a_time(client, monkeypatch):
    import asyncio
    import threading

    started, release = threading.Event(), threading.Event()

    async def slow(stress):
        started.set()
        while not release.is_set():
            await asyncio.sleep(0.05)
        return {"ok": True, "seconds": 0, "steps": []}
    monkeypatch.setattr(appmod, "_chromium_selftest", slow)
    results = []
    t = threading.Thread(target=lambda: results.append(client.get("/api/diagnostics").status_code))
    t.start()
    assert started.wait(5)
    assert client.get("/api/diagnostics").status_code == 429
    release.set()
    t.join(10)
    assert results == [200]


def test_without_cgroup_files_it_falls_back_to_the_vms_own_numbers(tmp_path, monkeypatch):
    write(tmp_path, {"meminfo": "MemTotal:        6291456 kB\nMemFree:  100 kB\nMemAvailable:    4194304 kB\n",
                     "vmstat": "nr_free_pages 1\noom_kill 3\npgfault 9\n"})
    monkeypatch.setenv("CAPTCHASOLVER_PROC", str(tmp_path))
    m = diag.memory_info(tmp_path / "no-cgroup-here")
    assert (m["current_mb"], m["limit_mb"], m["oom_kill"], m["source"]) == (2048, 6144, 3, "proc")


def test_chromium_processes_are_counted_with_their_memory_and_zombies(tmp_path, monkeypatch):
    def proc(pid, cmd, state, rss_pages):
        write(tmp_path, {f"{pid}/cmdline": cmd, f"{pid}/comm": cmd.split("/")[-1][:15],
                         f"{pid}/stat": f"{pid} ({cmd.split('/')[-1][:15]}) {state} 1 1 1 0",
                         f"{pid}/statm": f"1000 {rss_pages} 10 0 0 0 0"})
    proc(101, "/ms-playwright/chrome-headless-shell", "S", 25600)       # 100 MB
    proc(102, "/ms-playwright/chrome-headless-shell", "S", 12800)       # 50 MB
    proc(103, "/ms-playwright/chrome-headless-shell", "Z", 0)           # a leftover zombie
    proc(200, "/usr/bin/python3", "S", 99999)                           # not Chromium: ignored
    monkeypatch.setenv("CAPTCHASOLVER_PROC", str(tmp_path))
    assert diag.chromium_processes() == {"processes": 3, "rss_mb": 150, "zombies": 1}
    assert "Chromium: 3 process(es) holding 150 MB, 1 zombie(s)" in diag.hints()


def test_hints_say_when_there_is_no_dev_shm(monkeypatch):
    monkeypatch.setattr(diag, "disk_mb", lambda p: None if p == "/dev/shm" else {"total_mb": 100, "free_mb": 50})
    assert "no /dev/shm" in diag.hints()
