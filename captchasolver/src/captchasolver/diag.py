"""What the machine we run on can actually give us: memory limit/use, OOM kills, /dev/shm, /tmp.

Chromium dying inside a container is usually a resource limit, and the symptom ('Target page, context or
browser has been closed') says nothing about which. These readings are what turns a mystery into a cause.
Everything degrades to None when a file does not exist (not Linux, not a container, cgroup v1...).
"""
from __future__ import annotations

import os
from pathlib import Path

MB = 1024 * 1024


def _root() -> Path:
    return Path(os.environ.get("CAPTCHASOLVER_CGROUP", "/sys/fs/cgroup"))


def _proc() -> Path:
    return Path(os.environ.get("CAPTCHASOLVER_PROC", "/proc"))


def _read(path: Path) -> str | None:
    try:
        return path.read_text().strip()
    except OSError:
        return None


def _num(text: str | None) -> int | None:
    try:
        return int(text)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None               # "max" (no limit) or missing


def _mb(n: int | None) -> int | None:
    return None if n is None else round(n / MB)


def memory_info(root: Path | None = None) -> dict:
    """cgroup v2 first, then v1. Sizes in MB; `oom_kill` is the kernel's count of processes it killed."""
    r = root or _root()
    cur, lim, peak = _num(_read(r / "memory.current")), _num(_read(r / "memory.max")), _num(_read(r / "memory.peak"))
    if cur is None:                                                    # cgroup v1
        cur = _num(_read(r / "memory" / "memory.usage_in_bytes"))
        lim = _num(_read(r / "memory" / "memory.limit_in_bytes"))
        peak = _num(_read(r / "memory" / "memory.max_usage_in_bytes"))
        events_text = _read(r / "memory" / "memory.oom_control") or ""
    else:
        events_text = _read(r / "memory.events") or ""
    events = {}
    for line in events_text.splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[1].isdigit():
            events[parts[0]] = int(parts[1])
    if lim is not None and lim > 1 << 50:                              # v1 reports "unlimited" as a huge number
        lim = None
    source = "cgroup" if cur is not None else None
    if cur is None:                      # e.g. a micro-VM container with no cgroup files: use the VM's own numbers
        total, avail = _meminfo("MemTotal"), _meminfo("MemAvailable")
        if total is not None and avail is not None:
            cur, lim, source = (total - avail) * 1024, total * 1024, "proc"
        vm = _read(_proc() / "vmstat") or ""
        for line in vm.splitlines():
            if line.startswith("oom_kill "):
                events["oom_kill"] = int(line.split()[1])
    return {"current_mb": _mb(cur), "limit_mb": _mb(lim), "peak_mb": _mb(peak),
            "oom_kill": events.get("oom_kill"), "cpus": os.cpu_count(), "source": source}


def _meminfo(key: str) -> int | None:
    """kB value of a /proc/meminfo line."""
    for line in (_read(_proc() / "meminfo") or "").splitlines():
        if line.startswith(key + ":"):
            return _num(line.split()[1])
    return None


def chromium_processes() -> dict:
    """How many Chromium processes exist, how much memory they hold, and how many are leftover zombies."""
    n = rss = zombies = 0
    root = _proc()
    try:
        entries = [e for e in os.listdir(root) if e.isdigit()]
    except OSError:
        return {"processes": None, "rss_mb": None, "zombies": None}
    for pid in entries:
        try:
            if "chrom" not in (root / pid / "cmdline").read_text(errors="ignore") and \
               "chrom" not in (root / pid / "comm").read_text(errors="ignore"):
                continue
            stat = (root / pid / "stat").read_text()
            state = stat.rsplit(")", 1)[1].split()[0]
            n += 1
            zombies += state == "Z"
            rss += int((root / pid / "statm").read_text().split()[1]) * 4096
        except (OSError, IndexError, ValueError):
            continue
    return {"processes": n, "rss_mb": _mb(rss), "zombies": zombies}


def disk_mb(path: str) -> dict | None:
    try:
        st = os.statvfs(path)
    except OSError:
        return None
    return {"total_mb": round(st.f_blocks * st.f_frsize / MB), "free_mb": round(st.f_bavail * st.f_frsize / MB)}


def snapshot() -> dict:
    return {"memory": memory_info(), "dev_shm": disk_mb("/dev/shm"), "tmp": disk_mb("/tmp"),
            "chromium": chromium_processes()}


def hints() -> str:
    """One line for a crash note: the numbers that explain most Chromium deaths."""
    s = snapshot()
    m, shm, tmp, ch = s["memory"], s["dev_shm"], s["tmp"], s["chromium"]
    parts = []
    if m["current_mb"] is not None:
        used = f"memory {m['current_mb']} MB used" + (f" of {m['limit_mb']} MB" if m["limit_mb"] else "")
        if m["peak_mb"] is not None:
            used += f" (peak {m['peak_mb']} MB)"
        parts.append(used)
    if m["oom_kill"] is not None:
        parts.append(f"the kernel has killed {m['oom_kill']} process(es) for memory")
    if ch["processes"] is not None:
        parts.append(f"Chromium: {ch['processes']} process(es) holding {ch['rss_mb']} MB"
                     + (f", {ch['zombies']} zombie(s)" if ch["zombies"] else ""))
    parts.append(f"/dev/shm {shm['free_mb']}/{shm['total_mb']} MB free" if shm else "no /dev/shm")
    if tmp:
        parts.append(f"/tmp {tmp['free_mb']}/{tmp['total_mb']} MB free")
    return "; ".join(parts) or "no resource information available"
