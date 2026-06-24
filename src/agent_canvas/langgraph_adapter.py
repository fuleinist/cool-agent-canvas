"""LangGraph adapter — wraps a compiled StateGraph and emits TraceEvents.

v0.1 captures two streams from a LangGraph execution:

  1. Node transitions (via `graph.stream(stream_mode="updates")`)
     → emits NODE_START / NODE_END / STATE_DELTA / RUN_START / RUN_END / ERROR

  2. LLM + tool calls inside nodes (via a custom BaseCallbackHandler)
     → emits MESSAGE / TOOL_CALL events with the actual prompt/response text

The adapter is intentionally framework-agnostic at the seams: any agent
framework that exposes (a) ordered node transitions and (b) tool/llm call
hooks could feed the same TraceEvent stream. CrewAI / AutoGen land in v0.2.

Usage:

    from agent_canvas.langgraph_adapter import trace_langgraph

    graph = build_my_graph().compile()
    for event in trace_langgraph(graph, run_name="my-run", input={"x": 1}):
        print(event.to_json())
"""

from __future__ import annotations

import time
import traceback as _traceback
import uuid
from typing import Any, Iterator

from agent_canvas.storage import InMemoryStore, get_default_store
from agent_canvas.trace import TraceEvent, TraceEventType

try:
    # LangGraph ≥ 0.2 exposes CompiledStateGraph as the type
    from langgraph.graph import CompiledStateGraph  # type: ignore
except Exception:  # pragma: no cover - optional dep
    CompiledStateGraph = Any  # type: ignore[misc,assignment]


# ---------------------------------------------------------------------------
# Optional callback handler — captures LLM / tool events inside nodes
# ---------------------------------------------------------------------------


class _CanvasTracer:
    """Forwards langchain_core callback events into TraceEvents.

    Attached to `graph.stream(..., config={"callbacks": [tracer]})`. We don't
    subclass BaseCallbackHandler because that requires importing langchain-core
    at module load time (and we want the adapter module importable without
    optional deps installed). Instead we expose `install(graph_run_kwargs)`
    that returns kwargs to pass to graph.stream().

    Attributes `ignore_chain` and `raise_error` are required by langchain's
    callback manager to avoid AttributeError when checking handler capabilities.
    """

    ignore_chain: bool = True
    raise_error: bool = False

    def __init__(self, run_id: str, store: InMemoryStore) -> None:
        self.run_id = run_id
        self.store = store

    # ----- internal helpers -----

    def _emit(self, event: TraceEvent) -> None:
        self.store.append_event(self.run_id, event)

    # ----- langchain callback surface -----
    # Methods are called by langchain-core if the handler is passed via
    # config={"callbacks": [self]}. We define them dynamically; if
    # langchain-core isn't installed, the handler simply won't be wired up
    # but the adapter still works for node-level tracing.

    def on_llm_start(self, serialized, prompts, *, run_id, **kwargs):  # type: ignore[no-untyped-def]
        try:
            text = prompts[0] if prompts else ""
            if isinstance(text, list):
                text = " ".join(str(m) for m in text)
            self._emit(TraceEvent(
                type=TraceEventType.MESSAGE,
                run_id=self.run_id,
                data={"role": "user", "content": str(text)[:4000]},
                tags=["llm"],
                parent_event_id=str(run_id) if run_id else None,
            ))
        except Exception:
            # Never let a tracing hook kill the host run
            pass

    def on_llm_end(self, response, *, run_id, **kwargs):  # type: ignore[no-untyped-def]
        try:
            text = ""
            generations = getattr(response, "generations", None) or []
            if generations and generations[0]:
                msg = generations[0][0]
                text = getattr(msg, "text", None) or str(msg)
            self._emit(TraceEvent(
                type=TraceEventType.MESSAGE,
                run_id=self.run_id,
                data={"role": "assistant", "content": str(text)[:4000]},
                tags=["llm"],
                parent_event_id=str(run_id) if run_id else None,
            ))
        except Exception:
            pass

    def on_tool_start(self, serialized, input_str, *, run_id, **kwargs):  # type: ignore[no-untyped-def]
        try:
            name = (serialized or {}).get("name") if isinstance(serialized, dict) else None
            self._emit(TraceEvent(
                type=TraceEventType.TOOL_CALL,
                run_id=self.run_id,
                data={"tool": name or "tool", "input": str(input_str)[:4000], "phase": "start"},
                tags=["tool"],
                parent_event_id=str(run_id) if run_id else None,
            ))
        except Exception:
            pass

    def on_tool_end(self, output, *, run_id, **kwargs):  # type: ignore[no-untyped-def]
        try:
            self._emit(TraceEvent(
                type=TraceEventType.TOOL_CALL,
                run_id=self.run_id,
                data={"phase": "end", "output": str(output)[:4000]},
                tags=["tool"],
                parent_event_id=str(run_id) if run_id else None,
            ))
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Public API: trace_langgraph()
# ---------------------------------------------------------------------------


def trace_langgraph(
    graph: "CompiledStateGraph | Any",
    *,
    run_name: str = "",
    input: dict[str, Any] | None = None,
    config: dict[str, Any] | None = None,
    store: InMemoryStore | None = None,
    run_id: str | None = None,
) -> Iterator[TraceEvent]:
    """Run a compiled LangGraph graph, yielding TraceEvents as it executes.

    This is a *synchronous, generator* interface. For long-running workflows,
    run it in a background thread or wrap in `asyncio.to_thread()`.

    Args:
        graph: A compiled LangGraph graph (anything with a `.stream(...)` method).
        run_name: Human-readable label for the sidebar.
        input: Initial state passed to `graph.stream`.
        config: Optional LangGraph config (recursion_limit, callbacks, etc.).
                The adapter will *add* its own callback handler; user-supplied
                callbacks are preserved.
        store: Where to fan out events. Defaults to the module-level singleton.
        run_id: Override the auto-generated run id (useful for replay).

    Yields:
        TraceEvent for each lifecycle transition, plus any LLM/tool events
        fired inside nodes. The caller may consume them, persist them, or
        ignore the iterator — events are *also* appended to `store` so live
        WS subscribers see them.
    """
    store = store or get_default_store()
    run_id = run_id or f"run_{uuid.uuid4().hex[:8]}"
    rec = store.get_run(run_id)
    if rec is None:
        rec = store.create_run(name=run_name or run_id, run_id=run_id)

    cfg = dict(config or {})
    tracer = _CanvasTracer(run_id=run_id, store=store)
    existing_cbs = list(cfg.get("callbacks", []) or [])
    cfg["callbacks"] = [*existing_cbs, tracer]

    # run_start
    yield store.append_and_yield(_make_run_start(run_id=run_id, run_name=run_name or run_id, input_payload=input))

    # walk the graph
    started_at = time.time()
    last_node: str | None = None
    last_node_started_at: float | None = None
    parent_event_id: str | None = None

    try:
        for chunk in graph.stream(input or {}, config=cfg, stream_mode="updates"):
            # chunk is dict: {node_name: state_delta}
            if not isinstance(chunk, dict):
                continue
            for node_name, delta in chunk.items():
                # node_start (if first time we see this node, OR if it re-entered)
                ns = TraceEvent(
                    type=TraceEventType.NODE_START,
                    run_id=run_id,
                    node=node_name,
                    data={"node": node_name},
                    parent_event_id=parent_event_id,
                )
                parent_event_id = ns.event_id
                last_node = node_name
                last_node_started_at = ns.ts
                yield store.append_and_yield(ns)

                # state_delta (what changed in this step)
                sd = TraceEvent(
                    type=TraceEventType.STATE_DELTA,
                    run_id=run_id,
                    node=node_name,
                    data={"delta": _jsonable(delta)},
                    parent_event_id=ns.event_id,
                )
                yield store.append_and_yield(sd)

                # node_end
                latency_ms = ((sd.ts - ns.ts) * 1000.0) if ns.ts else None
                ne = TraceEvent(
                    type=TraceEventType.NODE_END,
                    run_id=run_id,
                    node=node_name,
                    data={"node": node_name, "latency_ms": latency_ms},
                    parent_event_id=ns.event_id,
                )
                yield store.append_and_yield(ne)

    except Exception as e:  # pragma: no cover - error path is exercised in tests
        tb = _traceback.format_exc(limit=4)
        yield store.append_and_yield(TraceEvent(
            type=TraceEventType.ERROR,
            run_id=run_id,
            node=last_node,
            data={"error": str(e), "traceback": tb},
            parent_event_id=parent_event_id,
        ))
    finally:
        total_ms = (time.time() - started_at) * 1000.0
        yield store.append_and_yield(TraceEvent(
            type=TraceEventType.RUN_END,
            run_id=run_id,
            node=last_node,
            data={"total_ms": total_ms},
        ))


def _make_run_start(*, run_id: str, run_name: str, input_payload: Any) -> TraceEvent:
    return TraceEvent(
        type=TraceEventType.RUN_START,
        run_id=run_id,
        data={"name": run_name, "input": _jsonable(input_payload)},
    )


def _jsonable(obj: Any) -> Any:
    """Best-effort coercion to JSON-serialisable Python types.

    LangGraph state can include BaseMessage objects, dataclasses, pydantic
    models. We don't deeply serialise — we just ensure the top-level is
    safe to put in a JSON frame.
    """
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    # known duck-types: pydantic v2
    if hasattr(obj, "model_dump"):
        try:
            return obj.model_dump()
        except Exception:
            pass
    # LangChain BaseMessage
    if hasattr(obj, "content") and hasattr(obj, "type"):
        try:
            return {"role": getattr(obj, "type", "message"), "content": obj.content}
        except Exception:
            pass
    # last resort: string repr, truncated
    return str(obj)[:1000]


# ---------------------------------------------------------------------------
# Patch the store to expose append_and_yield (used above).
# We add it as a method so callers who already hold a store reference work.
# ---------------------------------------------------------------------------


def _append_and_yield(self: InMemoryStore, event: TraceEvent) -> TraceEvent:  # type: ignore[no-redef]
    """Append an event to this run and return it (for chaining in adapters)."""
    self.append_event(event.run_id, event)
    return event


InMemoryStore.append_and_yield = _append_and_yield  # type: ignore[attr-defined]
