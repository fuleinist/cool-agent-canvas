"""CrewAI adapter — wraps a Crew and emits TraceEvents via the CrewAI event bus.

v0.2 hooks into CrewAI's built-in event system (crewai_event_bus) to capture:

  - Crew lifecycle: kickoff start/end/fail → RUN_START / RUN_END / ERROR
  - Agent execution: agent start/end/error → NODE_START / NODE_END / ERROR
  - Task lifecycle: task start/complete/fail → NODE_START / NODE_END / ERROR
  - LLM calls: LLM call start/complete → MESSAGE events
  - Tool usage: tool start/finish/error → TOOL_CALL events

Usage:

    from agent_canvas.crewai_adapter import trace_crew

    crew = Crew(agents=[...], tasks=[...], process=Process.sequential)
    for event in trace_crew(crew, inputs={"topic": "AI"}):
        print(event.to_json())
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from typing import Any, Callable, Iterator

from agent_canvas.storage import InMemoryStore, get_default_store
from agent_canvas.trace import TraceEvent, TraceEventType

logger = logging.getLogger("agent_canvas.crewai_adapter")

# ---------------------------------------------------------------------------
# Lazy imports — crewai is an optional dependency
# ---------------------------------------------------------------------------

_HAS_CREWAI = False
try:
    from crewai import Crew
    from crewai.events import crewai_event_bus
    from crewai.events.types.crew_events import (
        CrewKickoffCompletedEvent,
        CrewKickoffFailedEvent,
        CrewKickoffStartedEvent,
    )
    from crewai.events.types.agent_events import (
        AgentExecutionCompletedEvent,
        AgentExecutionErrorEvent,
        AgentExecutionStartedEvent,
    )
    from crewai.events.types.task_events import (
        TaskCompletedEvent,
        TaskFailedEvent,
        TaskStartedEvent,
    )
    from crewai.events.types.llm_events import (
        LLMCallCompletedEvent,
        LLMCallStartedEvent,
    )
    from crewai.events.types.tool_usage_events import (
        ToolUsageFinishedEvent,
        ToolUsageStartedEvent,
    )

    _HAS_CREWAI = True
except ImportError:  # pragma: no cover
    Crew = Any  # type: ignore[misc,assignment]


# ---------------------------------------------------------------------------
# Event bus listener — bridges CrewAI events → TraceEvents
# ---------------------------------------------------------------------------


class _CrewEventBridge:
    """Listens on the CrewAI event bus and forwards to an InMemoryStore.

    Thread-safe: handlers may fire from any thread; we use a lock around
    store writes and a buffer for the iterator.
    """

    def __init__(self, run_id: str, store: InMemoryStore) -> None:
        self.run_id = run_id
        self.store = store
        self._buffer: list[TraceEvent] = []
        self._lock = threading.Lock()
        self._handlers: list[tuple[type, Callable]] = []

    # ---- register / unregister ----

    def install(self) -> None:
        """Register event bus handlers."""
        bus = crewai_event_bus
        bus.on(CrewKickoffStartedEvent)(self._on_crew_start)
        self._handlers.append((CrewKickoffStartedEvent, self._on_crew_start))
        bus.on(CrewKickoffCompletedEvent)(self._on_crew_complete)
        self._handlers.append((CrewKickoffCompletedEvent, self._on_crew_complete))
        bus.on(CrewKickoffFailedEvent)(self._on_crew_failed)
        self._handlers.append((CrewKickoffFailedEvent, self._on_crew_failed))
        bus.on(AgentExecutionStartedEvent)(self._on_agent_start)
        self._handlers.append((AgentExecutionStartedEvent, self._on_agent_start))
        bus.on(AgentExecutionCompletedEvent)(self._on_agent_complete)
        self._handlers.append((AgentExecutionCompletedEvent, self._on_agent_complete))
        bus.on(AgentExecutionErrorEvent)(self._on_agent_error)
        self._handlers.append((AgentExecutionErrorEvent, self._on_agent_error))
        bus.on(TaskStartedEvent)(self._on_task_start)
        self._handlers.append((TaskStartedEvent, self._on_task_start))
        bus.on(TaskCompletedEvent)(self._on_task_complete)
        self._handlers.append((TaskCompletedEvent, self._on_task_complete))
        bus.on(TaskFailedEvent)(self._on_task_failed)
        self._handlers.append((TaskFailedEvent, self._on_task_failed))
        bus.on(LLMCallStartedEvent)(self._on_llm_start)
        self._handlers.append((LLMCallStartedEvent, self._on_llm_start))
        bus.on(LLMCallCompletedEvent)(self._on_llm_complete)
        self._handlers.append((LLMCallCompletedEvent, self._on_llm_complete))
        bus.on(ToolUsageStartedEvent)(self._on_tool_start)
        self._handlers.append((ToolUsageStartedEvent, self._on_tool_start))
        bus.on(ToolUsageFinishedEvent)(self._on_tool_finish)
        self._handlers.append((ToolUsageFinishedEvent, self._on_tool_finish))

    def _on_crew_start(self, source, event):
        self._emit(TraceEvent(
            type=TraceEventType.RUN_START,
            run_id=self.run_id,
            data={
                "name": event.crew_name or "crew",
                "input": event.inputs or {},
                "crew_name": event.crew_name,
            },
        ))

    def _on_crew_complete(self, source, event):
        output = event.output
        if hasattr(output, "raw"):
            output = output.raw
        self._emit(TraceEvent(
            type=TraceEventType.RUN_END,
            run_id=self.run_id,
            data={
                "output": str(output)[:4000] if output else "",
                "total_tokens": event.total_tokens,
            },
        ))

    def _on_crew_failed(self, source, event):
        self._emit(TraceEvent(
            type=TraceEventType.ERROR,
            run_id=self.run_id,
            data={"error": event.error},
        ))

    def _on_agent_start(self, source, event):
        role = getattr(event.agent, "role", None) or "unknown"
        goal = getattr(event.agent, "goal", None) or ""
        self._emit(TraceEvent(
            type=TraceEventType.NODE_START,
            run_id=self.run_id,
            node=role,
            data={
                "agent_role": role,
                "agent_goal": str(goal)[:500],
                "task_prompt": str(event.task_prompt)[:2000],
                "type": "agent",
            },
        ))

    def _on_agent_complete(self, source, event):
        role = getattr(event.agent, "role", None) or "unknown"
        self._emit(TraceEvent(
            type=TraceEventType.NODE_END,
            run_id=self.run_id,
            node=role,
            data={
                "agent_role": role,
                "output": str(event.output)[:4000],
                "type": "agent",
            },
        ))

    def _on_agent_error(self, source, event):
        role = getattr(event.agent, "role", None) or "unknown"
        self._emit(TraceEvent(
            type=TraceEventType.ERROR,
            run_id=self.run_id,
            node=role,
            data={
                "agent_role": role,
                "error": str(event.error)[:2000],
                "type": "agent",
            },
        ))

    def _on_task_start(self, source, event):
        task_name = event.task_name or "task"
        context = event.context or ""
        self._emit(TraceEvent(
            type=TraceEventType.NODE_START,
            run_id=self.run_id,
            node=task_name,
            data={
                "task_name": task_name,
                "context": str(context)[:1000],
                "type": "task",
            },
        ))

    def _on_task_complete(self, source, event):
        task_name = event.task_name or "task"
        output = event.output
        raw = getattr(output, "raw", None) or str(output)
        self._emit(TraceEvent(
            type=TraceEventType.NODE_END,
            run_id=self.run_id,
            node=task_name,
            data={
                "task_name": task_name,
                "output": str(raw)[:4000],
                "type": "task",
            },
        ))

    def _on_task_failed(self, source, event):
        task_name = event.task_name or "task"
        self._emit(TraceEvent(
            type=TraceEventType.ERROR,
            run_id=self.run_id,
            node=task_name,
            data={
                "task_name": task_name,
                "error": str(event.error)[:2000],
                "type": "task",
            },
        ))

    def _on_llm_start(self, source, event):
        messages = event.messages
        if isinstance(messages, list):
            content = messages[-1].get("content", "") if messages else ""
        else:
            content = str(messages or "")
        self._emit(TraceEvent(
            type=TraceEventType.MESSAGE,
            run_id=self.run_id,
            node=event.agent_role or event.task_name,
            data={
                "role": "user",
                "content": str(content)[:4000],
                "model": event.model or "",
                "call_id": event.call_id,
            },
            tags=["llm"],
        ))

    def _on_llm_complete(self, source, event):
        response = event.response
        content = ""
        if hasattr(response, "content"):
            content = response.content
        elif isinstance(response, str):
            content = response
        elif isinstance(response, dict):
            content = response.get("content", str(response))
        else:
            content = str(response)
        self._emit(TraceEvent(
            type=TraceEventType.MESSAGE,
            run_id=self.run_id,
            node=event.agent_role or event.task_name,
            data={
                "role": "assistant",
                "content": str(content)[:4000],
                "model": event.model or "",
                "call_id": event.call_id,
                "finish_reason": event.finish_reason or "",
            },
            tags=["llm"],
        ))

    def _on_tool_start(self, source, event):
        self._emit(TraceEvent(
            type=TraceEventType.TOOL_CALL,
            run_id=self.run_id,
            data={
                "tool": event.tool_name,
                "input": str(event.tool_args)[:4000],
                "phase": "start",
            },
            tags=["tool"],
        ))

    def _on_tool_finish(self, source, event):
        self._emit(TraceEvent(
            type=TraceEventType.TOOL_CALL,
            run_id=self.run_id,
            data={
                "tool": event.tool_name,
                "input": str(event.tool_args)[:4000],
                "phase": "end",
            },
            tags=["tool"],
        ))

    def uninstall(self) -> None:
        """Unregister all event bus handlers."""
        bus = crewai_event_bus
        for event_type, handler in self._handlers:
            try:
                bus.off(event_type, handler)
            except Exception:
                pass
        self._handlers.clear()

    # ---- internal ----

    def _emit(self, event: TraceEvent) -> None:
        """Thread-safe emit: append to store AND buffer."""
        with self._lock:
            self.store.append_event(self.run_id, event)
            self._buffer.append(event)

    def drain(self) -> list[TraceEvent]:
        """Drain buffered events (for the iterator)."""
        with self._lock:
            events = list(self._buffer)
            self._buffer.clear()
            return events


# ---------------------------------------------------------------------------
# Public API: trace_crew()
# ---------------------------------------------------------------------------


def trace_crew(
    crew: Crew,
    *,
    inputs: dict[str, Any] | None = None,
    store: InMemoryStore | None = None,
    run_id: str | None = None,
) -> Iterator[TraceEvent]:
    """Run a CrewAI crew, yielding TraceEvents as it executes.

    Hooks into the CrewAI event bus to capture crew, agent, task, LLM,
    and tool events. Works with sequential and hierarchical processes.

    Args:
        crew: A CrewAI Crew instance.
        inputs: Input variables for the crew (passed to crew.kickoff()).
        store: Where to fan out events. Defaults to the module-level singleton.
        run_id: Override the auto-generated run id.

    Yields:
        TraceEvent for each lifecycle transition.
    """
    if not _HAS_CREWAI:
        raise ImportError(
            "crewai is not installed. Install with: pip install agent-canvas[crewai]"
        )

    store = store or get_default_store()
    run_id = run_id or f"run_{uuid.uuid4().hex[:8]}"
    rec = store.get_run(run_id)
    if rec is None:
        rec = store.create_run(name=crew.name or "crew-run", run_id=run_id)

    bridge = _CrewEventBridge(run_id=run_id, store=store)
    bridge.install()

    started_at = time.time()

    try:
        # Drain any events that fired during setup
        for event in bridge.drain():
            yield event

        output = crew.kickoff(inputs=inputs)

        # Drain events emitted during kickoff
        for event in bridge.drain():
            yield event

    except Exception as e:
        # If kickoff itself raised (before event bus caught it)
        import traceback as _tb

        tb = _tb.format_exc(limit=4)
        err_evt = TraceEvent(
            type=TraceEventType.ERROR,
            run_id=run_id,
            data={"error": str(e), "traceback": tb},
        )
        store.append_event(run_id, err_evt)
        yield err_evt

        # Still emit RUN_END so the trace isn't orphaned
        end_evt = TraceEvent(
            type=TraceEventType.RUN_END,
            run_id=run_id,
            data={"total_ms": (time.time() - started_at) * 1000.0, "error": str(e)},
        )
        store.append_event(run_id, end_evt)
        yield end_evt
    else:
        # Ensure RUN_END was emitted (the event bus should have done this,
        # but be defensive)
        events = store.get_events(run_id)
        if not any(e.type is TraceEventType.RUN_END for e in events):
            end_evt = TraceEvent(
                type=TraceEventType.RUN_END,
                run_id=run_id,
                data={"total_ms": (time.time() - started_at) * 1000.0},
            )
            store.append_event(run_id, end_evt)
            yield end_evt
    finally:
        bridge.uninstall()
