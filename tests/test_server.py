"""Tests for the FastAPI server + WebSocket endpoint."""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from agent_canvas.server import create_app
from agent_canvas.storage import InMemoryStore
from agent_canvas.trace import TraceEvent, TraceEventType


@pytest.fixture
def store():
    return InMemoryStore()


@pytest.fixture
def client(store):
    app = create_app(store=store)
    return TestClient(app)


def test_list_runs_empty(client):
    resp = client.get("/runs")
    assert resp.status_code == 200
    assert resp.json() == []


def test_list_runs_with_data(client, store):
    store.create_run(name="test-run")
    resp = client.get("/runs")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) == 1
    assert data[0]["name"] == "test-run"


def test_get_run_not_found(client):
    resp = client.get("/runs/nonexistent")
    assert resp.status_code == 404
    assert "error" in resp.json()


def test_get_run_found(client, store):
    rec = store.create_run(name="my-run")
    resp = client.get(f"/runs/{rec.run_id}")
    assert resp.status_code == 200
    assert resp.json()["name"] == "my-run"


def test_get_run_events_empty(client, store):
    rec = store.create_run(name="empty")
    resp = client.get(f"/runs/{rec.run_id}/events")
    assert resp.status_code == 200
    assert resp.json() == []


def test_get_run_events_with_data(client, store):
    rec = store.create_run(name="data")
    event = TraceEvent(type=TraceEventType.NODE_START, run_id=rec.run_id, node="test")
    store.append_event(rec.run_id, event)
    resp = client.get(f"/runs/{rec.run_id}/events")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) == 1
    assert data[0]["type"] == "node_start"
    assert data[0]["node"] == "test"


def test_websocket_receives_events(client, store):
    """Test that WS client receives events appended after connection."""
    from fastapi.testclient import TestClient as TC

    rec = store.create_run(name="ws-test")

    with client.websocket_connect(f"/ws/trace/{rec.run_id}") as ws:
        # Append an event
        event = TraceEvent(type=TraceEventType.NODE_START, run_id=rec.run_id, node="ws-node")
        store.append_event(rec.run_id, event)

        data = ws.receive_text()
        parsed = json.loads(data)
        assert parsed["type"] == "node_start"
        assert parsed["node"] == "ws-node"


def test_websocket_sends_existing_events(client, store):
    """Test that WS client receives events that existed before connection."""
    rec = store.create_run(name="ws-replay")
    event = TraceEvent(type=TraceEventType.NODE_END, run_id=rec.run_id, node="pre-node")
    store.append_event(rec.run_id, event)

    with client.websocket_connect(f"/ws/trace/{rec.run_id}") as ws:
        data = ws.receive_text()
        parsed = json.loads(data)
        assert parsed["type"] == "node_end"
        assert parsed["node"] == "pre-node"


def test_static_index_served(client):
    """Static files should be served at /."""
    resp = client.get("/")
    assert resp.status_code == 200
    assert "Agent Canvas" in resp.text
