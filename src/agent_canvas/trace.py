"""Trace event primitives — the language the adapter speaks and the UI renders.

Every interaction with a multi-agent workflow is captured as a TraceEvent.
The adapter emits them, the server stores them, and the WebSocket streams
them to the browser. The frontend renders them as nodes lighting up, edges
animating, and tool calls expanding inline.

Keep this module small, dependency-free, and trivially serializable. It's
the contract between every other component.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class TraceEventType(str, Enum):
    """The set of trace events the UI understands.

    Extending the enum is non-breaking — the frontend just ignores unknown
    types. Removing a type is breaking and needs a UI bump.
    """

    RUN_START = "run_start"          # emitted once at workflow start
    RUN_END = "run_end"              # emitted once at workflow end
    NODE_START = "node_start"        # a node began executing
    NODE_END = "node_end"            # a node finished executing
    TOOL_CALL = "tool_call"          # a tool was invoked
    MESSAGE = "message"              # agent-to-agent message
    STATE_DELTA = "state_delta"      # a change to the workflow state
    ERROR = "error"                  # something went wrong
    CHECKPOINT = "checkpoint"        # state was persisted (v0.2+)


@dataclass
class TraceEvent:
    """A single trace event.

    Wire format (JSON):
        {
          "event_id": "uuid-v4",
          "run_id": "run_xxx",
          "ts": 1718707320.123,
          "type": "node_start",
          "node": "researcher",
          "data": {...},
          "parent_event_id": "..." | null,
          "tags": ["..."]
        }

    The `data` payload is intentionally untyped — each event type carries
    its own shape. Keep it JSON-serializable (str, int, float, bool, None,
    list, dict). Frontend knows how to render each type's data.
    """

    type: TraceEventType
    run_id: str
    data: dict[str, Any] = field(default_factory=dict)
    node: str | None = None
    parent_event_id: str | None = None
    tags: list[str] = field(default_factory=list)
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    ts: float = field(default_factory=time.time)

    def to_json(self) -> str:
        """Serialize to a JSON string. Suitable for WebSocket frames."""
        return json.dumps(self.to_dict(), separators=(",", ":"))

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-safe dict."""
        d = asdict(self)
        d["type"] = self.type.value
        return d

    @classmethod
    def from_json(cls, raw: str) -> "TraceEvent":
        """Deserialize from a JSON string."""
        d = json.loads(raw)
        return cls.from_dict(d)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "TraceEvent":
        """Build a TraceEvent from a dict, e.g. read back from disk."""
        return cls(
            event_id=d.get("event_id", str(uuid.uuid4())),
            run_id=d["run_id"],
            ts=d.get("ts", time.time()),
            type=TraceEventType(d["type"]),
            node=d.get("node"),
            data=d.get("data", {}),
            parent_event_id=d.get("parent_event_id"),
            tags=d.get("tags", []),
        )