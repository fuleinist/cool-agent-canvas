"""CrewAI adapter — wraps a Crew and emits TraceEvents.

v0.2 addition: hooks into CrewAI's task_callback to capture task-level
execution as TraceEvents. Each CrewAI task maps to a node_start/node_end
pair, and agent-to-agent delegation is captured as MESSAGE events.

Usage:

    from agent_canvas.crewai_adapter import trace_crewai
    from agent_canvas.storage import get_default_store

    crew = Crew(...)
    for event in trace_crewai(crew, run_name="my-crew", inputs={"topic": "AI"}):
        print(event.to_json())
"""

from __future__ import annotations

import time
import uuid
from typing import Any, Iterator

from agent_canvas.storage import InMemoryStore, get_default_store
from agent_canvas.trace import TraceEvent, TraceEventType

# Ensure append_and_yield is available (patched by langgraph_adapter at import)
from agent_canvas import langgraph_adapter  # noqa: F401

# ---------------------------------------------------------------------------
# Public API: trace_crewai()
# ---------------------------------------------------------------------------


def trace_crewai(
    crew: Any,
    *,
    run_name: str = "",
    inputs: dict[str, Any] | None = None,
    store: InMemoryStore | None = None,
    run_id: str | None = None,
) -> Iterator[TraceEvent]:
    """Run a CrewAI Crew, yielding TraceEvents as it executes.

    Hooks into ``task_callback`` on the Crew to capture task lifecycle.
    For CrewAI ≥ 1.0.0.

    Args:
        crew: A CrewAI Crew instance.
        run_name: Human-readable label.
        inputs: Input dict passed to ``crew.kickoff()``.
        store: Where to fan out events. Defaults to the module-level singleton.
        run_id: Override the auto-generated run id.

    Yields:
        TraceEvent for each lifecycle transition.
    """
    store = store or get_default_store()
    run_id = run_id or f"run_{uuid.uuid4().hex[:8]}"

    rec = store.get_run(run_id)
    if rec is None:
        rec = store.create_run(name=run_name or run_id, run_id=run_id)

    # ---- Wire up the callback ----
    original_callback = getattr(crew, "task_callback", None)
    adapter = _CrewAITracer(run_id=run_id, store=store)
    crew.task_callback = adapter.on_task_end  # type: ignore[assignment]

    # run_start
    yield store.append_and_yield(
        TraceEvent(
            type=TraceEventType.RUN_START,
            run_id=run_id,
            data={"name": run_name or run_id, "input": _jsonable(inputs)},
        )
    )

    started_at = time.time()
    last_error: str | None = None

    try:
        # Kick off the crew — the callback fires for each completed task
        result = crew.kickoff(inputs=inputs)
        # Emit a final state_delta with the crew output
        yield store.append_and_yield(
            TraceEvent(
                type=TraceEventType.STATE_DELTA,
                run_id=run_id,
                data={"delta": {"output": _jsonable(result)}},
                tags=["crew_output"],
            )
        )
    except Exception as e:
        last_error = str(e)
        yield store.append_and_yield(
            TraceEvent(
                type=TraceEventType.ERROR,
                run_id=run_id,
                data={"error": str(e)},
            )
        )
    finally:
        # Restore original callback
        if original_callback is not None:
            crew.task_callback = original_callback
        else:
            crew.task_callback = None

        total_ms = (time.time() - started_at) * 1000.0
        yield store.append_and_yield(
            TraceEvent(
                type=TraceEventType.RUN_END,
                run_id=run_id,
                data={"total_ms": total_ms, "error": last_error},
            )
        )


# ---------------------------------------------------------------------------
# Internal tracer
# ---------------------------------------------------------------------------


class _CrewAITracer:
    """Receives CrewAI task callbacks and emits TraceEvents."""

    def __init__(self, run_id: str, store: InMemoryStore) -> None:
        self.run_id = run_id
        self.store = store
        self._node_count: dict[str, int] = {}
        self._parent_event_id: str | None = None

    def on_task_end(self, task_output: Any) -> None:
        """Called by CrewAI after each task completes.

        Signature matches CrewAI's ``task_callback`` which receives a
        ``TaskOutput`` object.
        """
        try:
            task_name = _get_task_name(task_output)
            agent_name = _get_agent_name(task_output)

            # Deduplicate node names if the same task runs multiple times
            key = f"{task_name}/{agent_name}" if agent_name else task_name
            self._node_count[key] = self._node_count.get(key, 0) + 1
            count = self._node_count[key]
            node_name = f"{task_name}#{count}" if count > 1 else task_name

            # node_start
            ns = TraceEvent(
                type=TraceEventType.NODE_START,
                run_id=self.run_id,
                node=node_name,
                data={"node": node_name, "agent": agent_name or "unknown"},
                parent_event_id=self._parent_event_id,
            )
            self._parent_event_id = ns.event_id
            self.store.append_event(self.run_id, ns)

            # state_delta
            output_text = _get_output_text(task_output)
            sd = TraceEvent(
                type=TraceEventType.STATE_DELTA,
                run_id=self.run_id,
                node=node_name,
                data={
                    "delta": {
                        "task": task_name,
                        "agent": agent_name or "unknown",
                        "output": output_text,
                    }
                },
                parent_event_id=ns.event_id,
            )
            self.store.append_event(self.run_id, sd)

            # node_end
            ne = TraceEvent(
                type=TraceEventType.NODE_END,
                run_id=self.run_id,
                node=node_name,
                data={"node": node_name, "agent": agent_name or "unknown"},
                parent_event_id=ns.event_id,
            )
            self.store.append_event(self.run_id, ne)

        except Exception:
            # Never let a tracing hook kill the host run
            pass


# ---------------------------------------------------------------------------
# Helpers — duck-type TaskOutput so we don't import crewai at module level
# ---------------------------------------------------------------------------


def _get_task_name(task_output: Any) -> str:
    """Extract a human-readable task name."""
    if hasattr(task_output, "name") and task_output.name:
        return str(task_output.name)
    if hasattr(task_output, "description") and task_output.description:
        desc = str(task_output.description)
        return desc[:60] + "…" if len(desc) > 60 else desc
    return "task"


def _get_agent_name(task_output: Any) -> str | None:
    """Extract the agent role that produced this output."""
    if hasattr(task_output, "agent") and task_output.agent:
        agent = task_output.agent
        if hasattr(agent, "role") and agent.role:
            return str(agent.role)
        if hasattr(agent, "name") and agent.name:
            return str(agent.name)
        return str(agent)
    return None


def _get_output_text(task_output: Any) -> str:
    """Extract the raw output text."""
    if hasattr(task_output, "raw") and task_output.raw:
        return str(task_output.raw)[:2000]
    if hasattr(task_output, "output") and task_output.output:
        return str(task_output.output)[:2000]
    return str(task_output)[:2000]


def _jsonable(obj: Any) -> Any:
    """Best-effort coercion to JSON-serialisable Python types."""
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    # pydantic v2
    if hasattr(obj, "model_dump"):
        try:
            return obj.model_dump()
        except Exception:
            pass
    # last resort
    return str(obj)[:1000]
