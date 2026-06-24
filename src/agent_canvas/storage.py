"""In-memory run registry.

v0.1 stores traces in RAM only. v0.2 will back this with SQLite.

The registry's job is small but critical:
  - Track every run (so the UI can list them)
  - Buffer trace events per run (so the UI can replay / scrub)
  - Hand events to live WebSocket subscribers

Three read paths:
  - `list_runs()` → for the sidebar
  - `get_events(run_id)` → for replay / scrub
  - `subscribe(run_id)` → for live WS streaming
"""

from __future__ import annotations

import queue as _queue
import threading
from collections import defaultdict
from typing import Iterator

from agent_canvas.trace import TraceEvent


class RunRecord:
    """One run's worth of trace events + metadata."""

    __slots__ = ("run_id", "events", "name", "started_at", "ended_at", "status")

    def __init__(self, run_id: str, name: str = "") -> None:
        self.run_id = run_id
        self.name = name
        self.events: list[TraceEvent] = []
        self.started_at: float | None = None
        self.ended_at: float | None = None
        self.status: str = "idle"  # idle | running | completed | error

    def to_summary(self) -> dict:
        """Compact summary for the run list sidebar."""
        return {
            "run_id": self.run_id,
            "name": self.name,
            "status": self.status,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "event_count": len(self.events),
        }


class InMemoryStore:
    """Thread-safe in-memory store. Single process, multiple WS clients."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._runs: dict[str, RunRecord] = {}
        # subscriber queues: run_id -> list of (threading.Queue, subscriber_id)
        self._subscribers: dict[str, list[tuple[_queue.Queue, int]]] = defaultdict(list)
        self._sub_counter = 0

    # ---------- runs ----------

    def create_run(self, name: str = "", run_id: str | None = None) -> RunRecord:
        import uuid

        rid = run_id or f"run_{uuid.uuid4().hex[:8]}"
        with self._lock:
            rec = RunRecord(run_id=rid, name=name)
            self._runs[rid] = rec
            return rec

    def get_run(self, run_id: str) -> RunRecord | None:
        with self._lock:
            return self._runs.get(run_id)

    def list_runs(self) -> list[dict]:
        with self._lock:
            return [r.to_summary() for r in self._runs.values()]

    # ---------- events ----------

    def append_event(self, run_id: str, event: TraceEvent) -> None:
        """Append an event to a run and fan it out to subscribers."""
        with self._lock:
            rec = self._runs.get(run_id)
            if rec is None:
                rec = RunRecord(run_id=run_id)
                self._runs[run_id] = rec
            rec.events.append(event)
            if event.type.value == "run_start" and rec.started_at is None:
                rec.started_at = event.ts
                rec.status = "running"
            elif event.type.value == "run_end":
                rec.ended_at = event.ts
                rec.status = "completed"
            elif event.type.value == "error" and rec.status == "running":
                rec.status = "error"
            subs = list(self._subscribers.get(run_id, []))

        for q, _sid in subs:
            try:
                q.put_nowait(event)
            except _queue.Full:
                pass

    def get_events(self, run_id: str) -> list[TraceEvent]:
        with self._lock:
            rec = self._runs.get(run_id)
            if rec is None:
                return []
            return list(rec.events)

    # ---------- subscriptions (live WS streaming) ----------

    def subscribe(self, run_id: str) -> tuple[int, _queue.Queue]:
        """Register a new subscriber for a run.

        Returns (subscriber_id, threading.Queue) — caller must call unsubscribe().
        """
        self._sub_counter += 1
        sid = self._sub_counter
        q: _queue.Queue = _queue.Queue(maxsize=1000)
        with self._lock:
            self._subscribers[run_id].append((q, sid))
        return sid, q

    def unsubscribe(self, run_id: str, sid: int) -> None:
        with self._lock:
            subs = self._subscribers.get(run_id, [])
            self._subscribers[run_id] = [(q, s) for q, s in subs if s != sid]

    def _subscribe_refs(self, run_id: str, sid: int) -> tuple[_queue.Queue, int]:
        with self._lock:
            for q, s in self._subscribers.get(run_id, []):
                if s == sid:
                    return q, sid
        raise KeyError(f"subscriber {sid} not found for run {run_id}")


# Module-level singleton for v0.1. v0.2 swaps this for a factory.
_default_store: InMemoryStore | None = None


def get_default_store() -> InMemoryStore:
    global _default_store
    if _default_store is None:
        _default_store = InMemoryStore()
    return _default_store
