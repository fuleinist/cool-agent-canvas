"""Tests for the LangGraph adapter — the v0.1 core.

We don't mock langgraph here. We build a real 2-node graph and verify the
adapter emits the right trace events. This catches integration regressions
when langgraph upgrades.
"""

from __future__ import annotations

from typing import Any, TypedDict

import pytest

from agent_canvas.langgraph_adapter import trace_langgraph
from agent_canvas.storage import InMemoryStore
from agent_canvas.trace import TraceEventType


class S(TypedDict, total=False):
    """Minimal graph state."""

    value: int
    log: list[str]


def _node_double(state: S) -> S:
    return {"value": state["value"] * 2, "log": [*state.get("log", []), "double"]}


def _node_plus_one(state: S) -> S:
    return {"value": state["value"] + 1, "log": [*state.get("log", []), "plus_one"]}


def _build_two_node_graph():
    """Build a tiny linear graph: START -> double -> plus_one -> END."""
    from langgraph.graph import END, START, StateGraph

    g = StateGraph(S)
    g.add_node("double", _node_double)
    g.add_node("plus_one", _node_plus_one)
    g.add_edge(START, "double")
    g.add_edge("double", "plus_one")
    g.add_edge("plus_one", END)
    return g.compile()


def test_emits_run_start_and_end():
    store = InMemoryStore()
    graph = _build_two_node_graph()
    events = list(trace_langgraph(graph, run_name="t", input={"value": 1}, store=store))
    types = [e.type for e in events]
    assert TraceEventType.RUN_START in types
    assert TraceEventType.RUN_END in types
    assert types[0] is TraceEventType.RUN_START
    assert types[-1] is TraceEventType.RUN_END


def test_emits_node_start_and_end_for_each_node():
    store = InMemoryStore()
    graph = _build_two_node_graph()
    events = list(trace_langgraph(graph, run_name="t", input={"value": 1}, store=store))
    starts = [e for e in events if e.type is TraceEventType.NODE_START]
    ends = [e for e in events if e.type is TraceEventType.NODE_END]
    nodes_started = sorted({e.node for e in starts})
    nodes_ended = sorted({e.node for e in ends})
    assert nodes_started == ["double", "plus_one"]
    assert nodes_ended == ["double", "plus_one"]


def test_emits_state_delta_with_node_output():
    store = InMemoryStore()
    graph = _build_two_node_graph()
    events = list(trace_langgraph(graph, run_name="t", input={"value": 3}, store=store))
    deltas = [e for e in events if e.type is TraceEventType.STATE_DELTA]
    assert len(deltas) == 2
    # first delta is the output of `double` (value=6)
    assert deltas[0].node == "double"
    assert deltas[0].data["delta"]["value"] == 6
    # second delta is the output of `plus_one` (value=7)
    assert deltas[1].node == "plus_one"
    assert deltas[1].data["delta"]["value"] == 7


def test_node_end_includes_latency_ms():
    store = InMemoryStore()
    graph = _build_two_node_graph()
    events = list(trace_langgraph(graph, run_name="t", input={"value": 1}, store=store))
    ends = [e for e in events if e.type is TraceEventType.NODE_END]
    assert all(e.data.get("latency_ms") is not None for e in ends)
    assert all(e.data["latency_ms"] >= 0 for e in ends)


def test_store_receives_all_events():
    store = InMemoryStore()
    graph = _build_two_node_graph()
    events = list(trace_langgraph(graph, run_name="t", input={"value": 1}, store=store))
    rec = store.list_runs()
    assert len(rec) == 1
    stored = store.get_events(rec[0]["run_id"])
    # streamed events should match stored events 1:1
    assert len(stored) == len(events)
    assert [e.event_id for e in stored] == [e.event_id for e in events]


def test_error_emitted_when_node_raises():
    store = InMemoryStore()

    def boom(state: S) -> S:
        raise RuntimeError("kaboom")

    from langgraph.graph import END, START, StateGraph

    g = StateGraph(S)
    g.add_node("boom", boom)
    g.add_edge(START, "boom")
    g.add_edge("boom", END)
    graph = g.compile()

    events = list(trace_langgraph(graph, run_name="t", input={"value": 1}, store=store))
    errors = [e for e in events if e.type is TraceEventType.ERROR]
    assert len(errors) == 1
    assert "kaboom" in errors[0].data["error"]
    # even on error, RUN_END fires
    assert events[-1].type is TraceEventType.RUN_END


def test_run_id_override_used_in_events():
    store = InMemoryStore()
    graph = _build_two_node_graph()
    events = list(trace_langgraph(graph, run_name="t", input={"value": 1}, store=store, run_id="run_fixed"))
    assert all(e.run_id == "run_fixed" for e in events)
