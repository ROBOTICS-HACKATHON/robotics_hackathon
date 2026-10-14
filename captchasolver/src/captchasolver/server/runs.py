"""In-memory test runs: start, stream events, cancel, serve failure images. Nothing is persisted.

A run keeps the caller's Cloudflare client only while it is running; the token is dropped (and
the client closed) the moment the run ends or is cancelled. Failure images live in a temp
directory that is deleted when the run is evicted.
"""
from __future__ import annotations

import asyncio
import io
import json
import os
import shutil
import tempfile
import time
import uuid
import zipfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .. import bench
from ..client import WorkersAI, describe_exc

MAX_KEPT_RUNS = 4
MAX_RUN_SECONDS = int(os.environ.get("CAPTCHASOLVER_MAX_RUN_SECONDS", 3600))
FINISHED_TTL_S = 30 * 60


@dataclass
class Run:
    id: str
    params: dict
    dir: Path
    created: float = field(default_factory=time.time)
    status: str = "running"            # running | done | cancelled | error
    events: list[dict] = field(default_factory=list)
    records: list[bench.Record] = field(default_factory=list)
    summary: dict | None = None
    error: str = ""
    total: int = 0
    task: asyncio.Task | None = None
    cond: asyncio.Condition = field(default_factory=asyncio.Condition)
    cancel_requested: bool = False

    @property
    def finished(self) -> bool:
        return self.status != "running"

    def images_for(self, rec: bench.Record) -> dict[str, str]:
        base = f"{rec.index:03d}-{rec.label}"
        out = {}
        for key, name in (("state", f"{base}.png"), ("plan", f"{base}-plan.png"),
                          ("error", f"{base}-error.png")):
            if (self.dir / "failed_runs" / name).exists():
                out[key] = f"/api/tests/{self.id}/images/{name}"
        return out

    def record_dict(self, rec: bench.Record) -> dict:
        d = asdict(rec)
        d["screenshot"] = ""          # server paths are not meaningful to clients
        d["fail_image"] = ""
        d["cause"] = bench.failure_cause(rec)
        d["images"] = self.images_for(rec)
        return d

    def report(self) -> dict:
        return {"config": self.params, "status": self.status,
                "summary": self.summary or bench.summarize(self.records),
                "records": [self.record_dict(r) for r in sorted(self.records, key=lambda r: r.index)]}

    def state(self) -> dict:
        return {"id": self.id, "status": self.status, "params": self.params, "total": self.total,
                "done": len(self.records), "created": self.created, "error": self.error,
                "summary": self.summary,
                "records": [self.record_dict(r) for r in self.records]}

    def failures_zip(self) -> bytes:
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
            fdir = self.dir / "failed_runs"
            for p in sorted(fdir.glob("*.png")) if fdir.exists() else []:
                z.write(p, f"failed_runs/{p.name}")
            z.writestr("report.json", json.dumps(self.report(), indent=2))
        return buf.getvalue()


class RunManager:
    def __init__(self, max_concurrent: int = 1):
        self.runs: dict[str, Run] = {}
        self.max_concurrent = max_concurrent

    # ---- lifecycle -----------------------------------------------------------------
    def running(self) -> list[Run]:
        return [r for r in self.runs.values() if not r.finished]

    def get(self, run_id: str) -> Run | None:
        return self.runs.get(run_id)

    def _evict(self) -> None:
        now = time.time()
        done = sorted((r for r in self.runs.values() if r.finished), key=lambda r: r.created)
        drop = [r for r in done if now - r.created > FINISHED_TTL_S]
        keep = [r for r in done if r not in drop]
        drop += keep[:max(0, len(keep) - (MAX_KEPT_RUNS - 1))]
        for r in drop:
            shutil.rmtree(r.dir, ignore_errors=True)
            self.runs.pop(r.id, None)

    def start(self, params: dict, ai: WorkersAI) -> Run:
        if len(self.running()) >= self.max_concurrent:
            raise RuntimeError("a test run is already in progress")
        self._evict()
        run = Run(id=uuid.uuid4().hex, params=params,
                  dir=Path(tempfile.mkdtemp(prefix="captchasolver-run-")))
        (run.dir / "failed_runs").mkdir()
        self.runs[run.id] = run
        run.task = asyncio.create_task(self._drive(run, ai))
        return run

    def cancel(self, run: Run) -> None:
        if not run.finished and run.task:
            run.cancel_requested = True
            run.task.cancel()

    # ---- internals -----------------------------------------------------------------
    async def _emit(self, run: Run, type_: str, data: dict[str, Any]) -> None:
        async with run.cond:
            run.events.append({"type": type_, "data": data})
            run.cond.notify_all()

    async def _drive(self, run: Run, ai: WorkersAI) -> None:
        loop = asyncio.get_running_loop()

        def queue(type_: str, data: dict) -> None:   # callbacks are sync; hand over to the loop
            loop.create_task(self._emit(run, type_, data))

        def on_record(rec: bench.Record, k: int, n: int) -> None:
            run.records.append(rec)
            run.total = n
            queue("record", {"k": k, "n": n, "record": run.record_dict(rec)})

        p = run.params
        try:
            await self._emit(run, "start", {"id": run.id, "params": p})
            records, summary = await asyncio.wait_for(bench.run_tests(
                p["url"], count=p["count"], split=p["split"], seed=p["seed"],
                concurrency=p["concurrency"], pipeline=p["pipeline"], viewport=tuple(p["viewport"]),
                dpr=p["dpr"], mode="ask", model=p["model"], fail_dir=str(run.dir / "failed_runs"),
                fail_images=True, progress=on_record,
                note=lambda m: queue("note", {"message": m}), ai=ai), MAX_RUN_SECONDS)
            run.summary, run.status = summary, "done"
        except asyncio.TimeoutError:
            run.status, run.summary = "error", bench.summarize(run.records)
            run.error = f"The run took longer than {MAX_RUN_SECONDS // 60} minutes and was stopped."
            await self._emit(run, "error", {"message": run.error})
        except asyncio.CancelledError:
            if not run.cancel_requested:
                raise
            run.status = "cancelled"
            run.summary = bench.summarize(run.records)
        except Exception as e:  # noqa: BLE001 - reported to the client, never crashes the server
            run.status, run.error = "error", describe_exc(e, 300)
            run.summary = bench.summarize(run.records)
            await self._emit(run, "error", {"message": run.error})
        finally:
            await ai.aclose()                         # the token goes away with the client
            await asyncio.sleep(0)                    # let queued record events land first
            await self._emit(run, "summary", {"status": run.status, "summary": run.summary})
            await self._emit(run, "end", {"status": run.status})

    async def stream(self, run: Run, after: int = 0):
        """Yield SSE frames from event index `after` (resumable via Last-Event-ID)."""
        i = max(0, after)
        while True:
            async with run.cond:
                try:
                    await asyncio.wait_for(
                        run.cond.wait_for(lambda: len(run.events) > i), timeout=15)
                except asyncio.TimeoutError:
                    pass
            if i >= len(run.events):
                yield ": keep-alive\n\n"
                continue
            while i < len(run.events):
                ev = run.events[i]
                i += 1
                yield f"id: {i}\nevent: {ev['type']}\ndata: {json.dumps(ev['data'])}\n\n"
                if ev["type"] == "end":
                    return
