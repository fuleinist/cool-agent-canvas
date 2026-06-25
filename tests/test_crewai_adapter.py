"""Tests for the CrewAI adapter.

We build a real 2-task CrewAI crew and verify the adapter emits the
right trace events.
"""

from __future__ import annotations

from typing import Any

import pytest

from agent_canvas.crewai_adapter import trace_crewai
from agent_canvas.storage import InMemoryStore
from agent_canvas.trace import TraceEventType


# ---------------------------------------------------------------------------
# Mock TaskOutput (avoids importing crewai in tests that don't need it)
# ---------------------------------------------------------------------------


class FakeTaskOutput:
    """Duck-typed TaskOutput for testing the tracer in isolation."""

    def __init__(self, name: str, agent_role: str, raw: str) -> None:
        self.name = name
        self.description = name
        self.raw = raw
        self.agent = FakeAgent(role=agent_role)


class FakeAgent:
    def __init__(self, role: str) -> None:
        self.role = role
        self.name = role


def test_tracer_emits_start_end_for_single_task():
    store = InMemoryStore()
    run_id = "run_test_single"

    # Manually drive the tracer with one fake task output
    from agent_canvas.crewai_adapter import _CrewAITracer

    tracer = _CrewAITracer(run_id=run_id, store=store)

    # Simulate run_start
    from agent_canvas.trace import TraceEvent, TraceEventType

    store.append_event(
        run_id,
        TraceEvent(type=TraceEventType.RUN_START, run_id=run_id, data={"name": "test"}),
    )

    tracer.on_task_end(FakeTaskOutput("research", "Researcher", "found data"))

    store.append_event(
        run_id,
        TraceEvent(type=TraceEventType.RUN_END, run_id=run_id, data={"total_ms": 100}),
    )

    events = store.get_events(run_id)
    types = [e.type for e in events]

    assert TraceEventType.RUN_START in types
    assert TraceEventType.RUN_END in types
    assert TraceEventType.NODE_START in types
    assert TraceEventType.NODE_END in types
    assert TraceEventType.STATE_DELTA in types

    # Check node name
    starts = [e for e in events if e.type is TraceEventType.NODE_START]
    assert len(starts) == 1
    assert starts[0].node == "research"


def test_tracer_emits_events_for_multiple_tasks():
    store = InMemoryStore()
    run_id = "run_test_multi"

    from agent_canvas.crewai_adapter import _CrewAITracer
    from agent_canvas.trace import TraceEvent, TraceEventType

    tracer = _CrewAITracer(run_id=run_id, store=store)

    store.append_event(
        run_id,
        TraceEvent(type=TraceEventType.RUN_START, run_id=run_id, data={"name": "test"}),
    )

    tracer.on_task_end(FakeTaskOutput("research", "Researcher", "data"))
    tracer.on_task_end(FakeTaskOutput("write", "Writer", "article"))

    store.append_event(
        run_id,
        TraceEvent(type=TraceEventType.RUN_END, run_id=run_id, data={"total_ms": 200}),
    )

    events = store.get_events(run_id)
    starts = [e for e in events if e.type is TraceEventType.NODE_START]
    assert len(starts) == 2
    assert starts[0].node == "research"
    assert starts[1].node == "write"


def test_tracer_deduplicates_repeated_task_names():
    """When the same task name runs twice, the second gets a #2 suffix."""
    store = InMemoryStore()
    run_id = "run_test_dedup"

    from agent_canvas.crewai_adapter import _CrewAITracer
    from agent_canvas.trace import TraceEvent, TraceEventType

    tracer = _CrewAITracer(run_id=run_id, store=store)

    store.append_event(
        run_id,
        TraceEvent(type=TraceEventType.RUN_START, run_id=run_id, data={"name": "test"}),
    )

    tracer.on_task_end(FakeTaskOutput("retry", "Agent", "first"))
    tracer.on_task_end(FakeTaskOutput("retry", "Agent", "second"))

    store.append_event(
        run_id,
        TraceEvent(type=TraceEventType.RUN_END, run_id=run_id, data={"total_ms": 100}),
    )

    events = store.get_events(run_id)
    starts = [e for e in events if e.type is TraceEventType.NODE_START]
    assert len(starts) == 2
    assert starts[0].node == "retry"
    assert starts[1].node == "retry#2"


def test_store_receives_all_events():
    store = InMemoryStore()
    run_id = "run_test_store"

    from agent_canvas.crewai_adapter import _CrewAITracer
    from agent_canvas.trace import TraceEvent, TraceEventType

    tracer = _CrewAITracer(run_id=run_id, store=store)

    store.append_event(
        run_id,
        TraceEvent(type=TraceEventType.RUN_START, run_id=run_id, data={"name": "test"}),
    )

    tracer.on_task_end(FakeTaskOutput("task1", "Agent", "output"))

    store.append_event(
        run_id,
        TraceEvent(type=TraceEventType.RUN_END, run_id=run_id, data={"total_ms": 50}),
    )

    stored = store.get_events(run_id)
    # RUN_START, NODE_START, STATE_DELTA, NODE_END, RUN_END = 5
    assert len(stored) == 5


# ---------------------------------------------------------------------------
# Integration tests — require crewai installed
# ---------------------------------------------------------------------------


def _crewai_available() -> bool:
    try:
        import crewai  # noqa: F401

        return True
    except ImportError:
        return False


pytestmark = pytest.mark.skipif(
    not _crewai_available(),
    reason="crewai not installed",
)


def test_integration_with_real_crew():
    """Run a real 2-agent CrewAI crew and verify trace events."""
    import crewai

    researcher = crewai.Agent(
        role="Researcher",
        goal="Find information",
        backstory="You are a researcher.",
        allow_delegation=False,
        verbose=False,
    )
    writer = crewai.Agent(
        role="Writer",
        goal="Write output",
        backstory="You are a writer.",
        allow_delegation=False,
        verbose=False,
    )

    research_task = crewai.Task(
        name="research",
        description="Research the topic of AI agents",
        expected_output="A summary of findings",
        agent=researcher,
    )
    write_task = crewai.Task(
        name="write",
        description="Write a short report based on research",
        expected_output="A short report",
        agent=writer,
    )

    crew = crewai.Crew(
        agents=[researcher, writer],
        tasks=[research_task, write_task],
        verbose=False,
    )

    store = InMemoryStore()
    events = list(trace_crewai(crew, run_name="integration-test", store=store))

    types = [e.type for e in events]
    assert TraceEventType.RUN_START in types
    assert TraceEventType.RUN_END in types
    assert events[0].type is TraceEventType.RUN_START
    assert events[-1].type is TraceEventType.RUN_END
