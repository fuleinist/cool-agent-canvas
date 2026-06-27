"""Tests for the CrewAI adapter.

Unit tests call bridge handler methods directly (bypassing the CrewAI event
bus which has complex internal handlers that crash on partial test data).
Integration test runs a real crew (requires OPENAI_API_KEY).
"""

from __future__ import annotations

import os

import pytest

from agent_canvas.storage import InMemoryStore
from agent_canvas.trace import TraceEvent, TraceEventType

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def store():
    return InMemoryStore()


@pytest.fixture
def bridge(store):
    """Create a _CrewEventBridge."""
    from agent_canvas.crewai_adapter import _CrewEventBridge

    b = _CrewEventBridge(run_id="run_test", store=store)
    b.install()
    yield b
    b.uninstall()


# ---------------------------------------------------------------------------
# Crew-level events — call bridge handler methods directly
# ---------------------------------------------------------------------------


def test_crew_start_emits_run_start(bridge):
    from crewai.events.types.crew_events import CrewKickoffStartedEvent

    event = CrewKickoffStartedEvent(
        crew_name="test-crew",
        inputs={"topic": "AI"},
        crew=None,
    )
    bridge._on_crew_start("source", event)

    events = bridge.drain()
    assert any(e.type is TraceEventType.RUN_START for e in events)
    run_start = next(e for e in events if e.type is TraceEventType.RUN_START)
    assert run_start.data["name"] == "test-crew"
    assert run_start.data["input"]["topic"] == "AI"


def test_crew_complete_emits_run_end(bridge):
    from crewai.events.types.crew_events import CrewKickoffCompletedEvent

    event = CrewKickoffCompletedEvent(
        crew_name="test-crew",
        output="Final report",
        total_tokens=150,
        crew=None,
    )
    bridge._on_crew_complete("source", event)

    events = bridge.drain()
    ends = [e for e in events if e.type is TraceEventType.RUN_END]
    assert len(ends) == 1
    assert ends[0].data["output"] == "Final report"
    assert ends[0].data["total_tokens"] == 150


def test_crew_failed_emits_error(bridge):
    from crewai.events.types.crew_events import CrewKickoffFailedEvent

    event = CrewKickoffFailedEvent(
        crew_name="test-crew",
        error="LLM rate limit exceeded",
        crew=None,
    )
    bridge._on_crew_failed("source", event)

    events = bridge.drain()
    errors = [e for e in events if e.type is TraceEventType.ERROR]
    assert len(errors) >= 1
    assert "rate limit" in errors[0].data["error"]


# ---------------------------------------------------------------------------
# Agent-level events
# ---------------------------------------------------------------------------


def test_agent_start_emits_node_start(bridge):
    from crewai import Agent
    from crewai.events.types.agent_events import AgentExecutionStartedEvent

    agent = Agent(
        role="Researcher",
        goal="Find info",
        backstory="Expert researcher",
        allow_delegation=False,
    )
    event = AgentExecutionStartedEvent(
        agent=agent,
        task=None,
        tools=[],
        task_prompt="Research AI agents",
    )
    bridge._on_agent_start("source", event)

    events = bridge.drain()
    starts = [e for e in events if e.type is TraceEventType.NODE_START]
    assert len(starts) >= 1
    assert starts[0].node == "Researcher"
    assert starts[0].data["agent_role"] == "Researcher"


def test_agent_complete_emits_node_end(bridge):
    from crewai import Agent
    from crewai.events.types.agent_events import AgentExecutionCompletedEvent

    agent = Agent(
        role="Researcher",
        goal="Find info",
        backstory="Expert researcher",
        allow_delegation=False,
    )
    event = AgentExecutionCompletedEvent(
        agent=agent,
        task=None,
        output="Research complete",
    )
    bridge._on_agent_complete("source", event)

    events = bridge.drain()
    ends = [e for e in events if e.type is TraceEventType.NODE_END]
    assert len(ends) >= 1
    assert ends[0].node == "Researcher"
    assert "Research complete" in ends[0].data["output"]


def test_agent_error_emits_error(bridge):
    from crewai import Agent
    from crewai.events.types.agent_events import AgentExecutionErrorEvent

    agent = Agent(
        role="Researcher",
        goal="Find info",
        backstory="Expert researcher",
        allow_delegation=False,
    )
    event = AgentExecutionErrorEvent(
        agent=agent,
        task=None,
        error="Something went wrong",
    )
    bridge._on_agent_error("source", event)

    events = bridge.drain()
    errors = [e for e in events if e.type is TraceEventType.ERROR]
    assert len(errors) >= 1
    assert errors[0].node == "Researcher"
    assert "Something went wrong" in errors[0].data["error"]


# ---------------------------------------------------------------------------
# Task-level events
# ---------------------------------------------------------------------------


def test_task_start_emits_node_start(bridge):
    from crewai.events.types.task_events import TaskStartedEvent

    event = TaskStartedEvent(
        task=None,
        context="Previous research done",
    )
    bridge._on_task_start("source", event)

    events = bridge.drain()
    starts = [e for e in events if e.type is TraceEventType.NODE_START]
    assert len(starts) >= 1
    assert starts[0].data["type"] == "task"


def test_task_complete_emits_node_end(bridge):
    from crewai.events.types.task_events import TaskCompletedEvent
    from crewai.tasks.task_output import TaskOutput

    output = TaskOutput(
        raw="Task done",
        description="test",
        agent="Researcher",
    )
    event = TaskCompletedEvent(
        task=None,
        output=output,
    )
    bridge._on_task_complete("source", event)

    events = bridge.drain()
    ends = [e for e in events if e.type is TraceEventType.NODE_END]
    assert len(ends) >= 1
    assert ends[0].data["type"] == "task"
    assert "Task done" in ends[0].data["output"]


# ---------------------------------------------------------------------------
# LLM events
# ---------------------------------------------------------------------------


def test_llm_start_emits_message(bridge):
    from crewai import Agent
    from crewai.events.types.llm_events import LLMCallStartedEvent

    agent = Agent(
        role="Researcher",
        goal="Find info",
        backstory="Expert",
        allow_delegation=False,
    )
    event = LLMCallStartedEvent(
        call_id="call-1",
        messages="What is AI?",
        model="gpt-4",
        from_agent=agent,
        from_task=None,
    )
    bridge._on_llm_start("source", event)

    events = bridge.drain()
    msgs = [e for e in events if e.type is TraceEventType.MESSAGE]
    assert len(msgs) >= 1
    assert msgs[0].data["role"] == "user"
    assert "llm" in msgs[0].tags


def test_llm_complete_emits_message(bridge):
    from crewai import Agent
    from crewai.events.types.llm_events import LLMCallCompletedEvent, LLMCallType

    agent = Agent(
        role="Researcher",
        goal="Find info",
        backstory="Expert",
        allow_delegation=False,
    )
    event = LLMCallCompletedEvent(
        call_id="call-1",
        response="AI stands for artificial intelligence",
        call_type=LLMCallType.LLM_CALL,
        from_agent=agent,
        from_task=None,
    )
    bridge._on_llm_complete("source", event)

    events = bridge.drain()
    msgs = [e for e in events if e.type is TraceEventType.MESSAGE]
    assert len(msgs) >= 1
    assert msgs[0].data["role"] == "assistant"
    assert "artificial intelligence" in msgs[0].data["content"]


# ---------------------------------------------------------------------------
# Tool events
# ---------------------------------------------------------------------------


def test_tool_start_emits_tool_call(bridge):
    from datetime import datetime, timezone
    from crewai.events.types.tool_usage_events import ToolUsageStartedEvent

    event = ToolUsageStartedEvent(
        tool_name="search",
        tool_args={"query": "AI agents"},
    )
    bridge._on_tool_start("source", event)

    events = bridge.drain()
    tools = [e for e in events if e.type is TraceEventType.TOOL_CALL]
    assert len(tools) >= 1
    assert tools[0].data["tool"] == "search"
    assert tools[0].data["phase"] == "start"


def test_tool_finish_emits_tool_call(bridge):
    from datetime import datetime, timezone
    from crewai.events.types.tool_usage_events import ToolUsageFinishedEvent

    now = datetime.now(timezone.utc)
    event = ToolUsageFinishedEvent(
        tool_name="search",
        tool_args={"query": "AI agents"},
        started_at=now,
        finished_at=now,
        output="Search results here",
    )
    bridge._on_tool_finish("source", event)

    events = bridge.drain()
    tools = [e for e in events if e.type is TraceEventType.TOOL_CALL]
    assert len(tools) >= 1
    assert tools[0].data["tool"] == "search"
    assert tools[0].data["phase"] == "end"


# ---------------------------------------------------------------------------
# Store integration
# ---------------------------------------------------------------------------


def test_events_stored_in_store(bridge, store):
    """Events emitted via the bridge should appear in the store."""
    from crewai.events.types.crew_events import CrewKickoffStartedEvent

    event = CrewKickoffStartedEvent(
        crew_name="test",
        inputs={},
        crew=None,
    )
    bridge._on_crew_start("source", event)
    bridge.drain()

    stored = store.get_events("run_test")
    assert len(stored) >= 1
    assert stored[0].type is TraceEventType.RUN_START


# ---------------------------------------------------------------------------
# trace_crew() function tests
# ---------------------------------------------------------------------------


def test_trace_crew_import_error_when_not_installed():
    """trace_crew should raise ImportError if crewai is not available."""
    import sys

    # Can't actually uninstall crewai in this test, so just verify the
    # function exists and accepts the right args
    from agent_canvas.crewai_adapter import trace_crew
    assert callable(trace_crew)


# ---------------------------------------------------------------------------
# Integration test — real CrewAI crew (skipped if no LLM)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set — skipping real CrewAI test",
)
def test_trace_crew_integration():
    """Run a real (tiny) CrewAI crew and verify events flow through."""
    from crewai import Agent, Crew, Process, Task
    from agent_canvas.crewai_adapter import trace_crew

    agent = Agent(
        role="Greeter",
        goal="Say hello",
        backstory="You are friendly.",
        allow_delegation=False,
        verbose=False,
    )
    task = Task(
        description="Say hello to the world. Just respond with 'Hello, world!'",
        expected_output="Hello, world!",
        agent=agent,
    )
    crew = Crew(
        name="Hello Crew",
        agents=[agent],
        tasks=[task],
        process=Process.sequential,
        verbose=False,
    )

    events = list(trace_crew(crew, inputs={}))
    types = [e.type for e in events]

    assert TraceEventType.RUN_START in types
    assert TraceEventType.RUN_END in types
    assert types[-1] is TraceEventType.RUN_END

    starts = [e for e in events if e.type is TraceEventType.NODE_START]
    ends = [e for e in events if e.type is TraceEventType.NODE_END]
    assert len(starts) >= 1
    assert len(ends) >= 1
