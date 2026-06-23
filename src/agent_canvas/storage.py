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
        # subscriber queues: run_id -> list of (Queue, subscriber_id)
        self._subscribers: dict[str, list[tuple]] = defaultdict(list)
        self._sub_counter = 0

    # ---------- runs ----------

    def create_run(self, name: str = "") -> RunRecord:
        import uuid

        run_id = f"run_{uuid.uuid4().hex[:8]}"
        with self._lock:
            rec = RunRecord(run_id=run_id, name=name)
            self._runs[run_id] = rec
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
        import queue as queue_mod

        with self._lock:
            rec = self._runs.get(run_id)
            if rec is None:
                # auto-create so adapter doesn't have to call create_run first
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

        # fan out outside the lock — never call user code while holding it
        for q, _sid in subs:
            try:
                q.put_nowait(event)
            except queue_mod.Full:
                # drop on slow consumer; UI will catch up on reconnect
                pass

    def get_events(self, run_id: str) -> list[TraceEvent]:
        with self._lock:
            rec = self._runs.get(run_id)
            if rec is None:
                return []
            return list(rec.events)

    # ---------- subscriptions (live WS streaming) ----------

    def subscribe(self, run_id: str) -> tuple[int, "queue.Queue[TraceEvent]"]:
        """Register a new subscriber for a run.

        Returns (subscriber_id, queue) — caller must call unsubscribe().
        """
        import queue as queue_mod

        with self._lock:
            self._sub_counter += 1
            sid = self._sub_counter
            q: queue_mod.Queue[TraceEvent] = queue_mod.Queue(maxsize=1000)
            self._subscribers[run_id].append((q, sid))
        return sid, q

    def unsubscribe(self, run_id: str, sid: int) -> None:
        with self._lock:
            subs = self._subscribers.get(run_id, [])
            self._subscribers[run_id] = [(q, s) for q, s in subs if s != sid]

    def iter_live(self, run_id: str, sid: int) -> Iterator[TraceEvent]:
        """Generator that yields events as they arrive for this subscriber.

        Yields sentinel-like behaviour on unsubscribe by checking the queue
        with a small timeout — caller should call .close() to stop.
        """
        import queue as queue_mod

        _, q = self._subscribe_refs(run_id, sid)
        while True:
            try:
                yield q.get(timeout=0.5)
            except queue_mod.Empty:
                # check whether we were unsubscribed
                with self._lock:
                    still = any(s == sid for _, s in self._subscribers.get(run_id, []))
                if not still:
                    return

    def _subscribe_refs(self, run_id: str, sid: int) -> tuple[int, "queue.Queue[TraceEvent]"]:
        with self._lock:
            for q, s in self._subscribers.get(run_id, []):
                if s == sid:
                    return sid, q
        raise KeyError(f"subscriber {sid} not found for run {run_id}")


# Module-level singleton for v0.1. v0.2 swaps this for a factory.
_default_store: InMemoryStore | None = None


def get_default_store() -> InMemoryStore:
    global _default_store
    if _default_store is None:
        _default_store = InMemoryStore()
    return _default_store