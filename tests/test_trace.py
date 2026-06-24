"""Tests for the trace event primitives (no external deps)."""

from __future__ import annotations

import json
import time

from agent_canvas.trace import TraceEvent, TraceEventType


def test_event_id_is_unique():
    a = TraceEvent(type=TraceEventType.NODE_START, run_id="r")
    b = TraceEvent(type=TraceEventType.NODE_START, run_id="r")
    assert a.event_id != b.event_id


def test_to_dict_serialises_type_as_value():
    e = TraceEvent(type=TraceEventType.NODE_START, run_id="r", node="x")
    d = e.to_dict()
    assert d["type"] == "node_start"
    assert d["run_id"] == "r"
    assert d["node"] == "x"


def test_to_json_is_valid_json():
    e = TraceEvent(type=TraceEventType.NODE_END, run_id="r", data={"latency_ms": 12.3})
    raw = e.to_json()
    parsed = json.loads(raw)
    assert parsed["type"] == "node_end"
    assert parsed["data"]["latency_ms"] == 12.3


def test_from_dict_round_trip():
    e = TraceEvent(type=TraceEventType.TOOL_CALL, run_id="r", data={"tool": "search", "phase": "start"}, tags=["tool"])
    e2 = TraceEvent.from_dict(e.to_dict())
    assert e2.event_id == e.event_id
    assert e2.type is TraceEventType.TOOL_CALL
    assert e2.data == e.data
    assert e2.tags == ["tool"]


def test_from_json_round_trip():
    e = TraceEvent(type=TraceEventType.ERROR, run_id="r", data={"error": "boom"})
    e2 = TraceEvent.from_json(e.to_json())
    assert e2.event_id == e.event_id
    assert e2.type is TraceEventType.ERROR
    assert e2.data == e.data


def test_event_types_are_strings():
    """TraceEventType values are wire-stable strings (UI depends on this)."""
    for t in TraceEventType:
        assert isinstance(t.value, str)
        assert t.value == t.value.lower()
