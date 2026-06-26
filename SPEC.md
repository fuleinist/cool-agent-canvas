# Agent Canvas — SPEC.md

## 1. Concept & Vision

**What it does:** A visual canvas for designing, debugging, and tracing multi-agent workflows. Drop in a LangGraph / CrewAI / AutoGen workflow, run it, and see every node execution, message exchange, tool call, and state transition streamed in real time on a directed graph.

**What it feels like:** Chrome DevTools Network tab meets TensorBoard, for agents. Dark IDE-native. The graph itself is the UI — you watch the trace paint itself as the workflow runs. Each node lights up when it executes; edges show message flow; tool calls expand inline. Time advances left-to-right.

**v0.1 scope (this build):** LangGraph Python backend + FastAPI + WebSocket server + a single-file HTML/JS trace viewer. React UI deferred to v0.3.

**v0.2 (complete):** CrewAI adapter — `trace_crewai()` wraps a CrewAI `Crew` and emits `TraceEvent`s via `task_callback`. See `src/agent_canvas/crewai_adapter.py` and `examples/crewai_basic/`.

## 2. Design Language

**Aesthetic:** Dark IDE-native. Monospace for data, clean sans-serif for UI chrome. The graph is the centerpiece — high-contrast nodes with role-based color, edges weighted by message count.

**Color palette:**
- Background: `#0d1117`
- Surface: `#161b22`
- Border: `#30363d`
- Primary accent: `#58a6ff` (graph edges, active state)
- Agent / LLM node: `#a371f7` (purple)
- Tool node: `#3fb950` (green)
- Conditional / router node: `#d29922` (amber)
- Error node: `#f85149` (red)
- Text primary: `#e6edf3`
- Text muted: `#8b949e`

**Typography:**
- UI: `Inter` or system sans-serif
- Code/data: `JetBrains Mono` or `Fira Code`

**Motion:** Minimal but meaningful. When a node executes, it pulses for 400ms. Edges animate message packets (small circles) flowing along the path. No decorative animation.

## 3. Layout & Structure

```
┌─────────────────────────────────────────────────────────────────┐
│ HEADER: Agent Canvas · Run ID · Status · Frame ◀ ▶ [Live]      │
├──────────────┬──────────────────────────────────┬───────────────┤
│ RUN LIST     │  GRAPH CANVAS                     │ INSPECTOR     │
│              │  - Nodes laid out auto/hand       │  - Selected   │
│ • run_001    │  - Edges with message counts      │    node:      │
│ • run_002 ●  │  - Tool calls expand inline       │    - State    │
│ • run_003    │  - Live tick animation            │    - Messages │
│              │  - Pan/zoom/drag                  │    - Latency  │
│              │                                   │               │
├──────────────┴──────────────────────────────────┴───────────────┤
│ TIMELINE: frames ▶▶▶ scrubber ──────────●────────── 12.4s       │
└─────────────────────────────────────────────────────────────────┘
```

**Responsive:** v0.1 desktop-only. Mobile/responsive deferred to v0.3.

## 4. Features & Interactions

### v0.1 — Core (this cycle)

#### Backend (Python)
- `agent_canvas.server` — FastAPI app + WebSocket endpoint at `/ws/trace/{run_id}`
- `agent_canvas.langgraph_adapter` — wraps a LangGraph `CompiledStateGraph` and emits `TraceEvent` objects for:
  - `node_start` — node name, input state snapshot
  - `node_end` — node name, output state diff, latency ms
  - `tool_call` — tool name, args, result
  - `message` — agent-to-agent message
  - `error` — exception + traceback
- `agent_canvas.trace` — `TraceEvent` dataclass + JSON serialization
- `agent_canvas.storage` — in-memory run registry (defer SQLite to v0.2)
- `agent_canvas.cli` — `agent-canvas run path/to/graph.py` boots server + opens browser

#### Frontend (single HTML file)
- `static/index.html` — vanilla JS, no build step
- Connects to `/ws/trace/{run_id}`, renders nodes + edges on a `<canvas>` or SVG
- Live mode: append events as they arrive
- Replay mode: scrub timeline, re-render from stored trace
- Click a node → inspector pane shows state, messages, latency

#### Example
- `examples/langgraph_basic/` — minimal LangGraph workflow (`planner → researcher → writer`) that the adapter wraps and emits traces for

### v0.2 — Multi-framework (in progress)
- ✅ CrewAI adapter (crew + agent + task abstraction → TraceEvent)
- AutoGen adapter (GroupChat / ConversableAgent → TraceEvent)
- SQLite-backed run persistence
- Search/filter runs by name/tag/timestamp

### v0.3 — Polish
- React frontend (Vite + React Flow or custom SVG renderer)
- Drag-to-arrange node positions
- Save/load layout
- Diff two runs side-by-side
- Export trace as JSON / replay script

## 5. Acceptance Criteria (v0.1)

A working v0.1 satisfies:

1. `pip install -e .` from `G:\dev\projects\cool-agent-canvas` installs without error on Python 3.11+.
2. `agent-canvas run examples/langgraph_basic/graph.py` boots a server and opens a browser.
3. Running the example workflow emits at least 3 distinct `TraceEvent` types (`node_start`, `node_end`, `message`) for each node transition.
4. WebSocket at `/ws/trace/{run_id}` streams JSON events in <100ms from emission.
5. `static/index.html` connects, renders the graph, animates node execution as events arrive.
6. Clicking a node in the UI shows its state, messages, and latency.
7. After the run completes, the trace can be replayed via timeline scrubber.
8. `pytest tests/` passes — at minimum: trace serialization, LangGraph adapter wrapping a 2-node graph, WebSocket round-trip.

## 6. Non-goals (v0.1)

- ❌ CrewAI / AutoGen adapters (v0.2)
- ❌ React frontend (v0.3)
- ❌ SQLite persistence (v0.2)
- ❌ Mobile responsive (v0.3)
- ❌ Multi-user / auth (out of scope)
- ❌ Cloud-hosted runs (local-only for v0.1)

## 7. Tech Stack

- **Backend:** Python 3.11+, FastAPI, uvicorn, websockets, pydantic
- **Adapter:** LangGraph ≥ 0.2 (`langgraph` PyPI package)
- **Frontend:** Vanilla JS + SVG (no build step in v0.1)
- **Tests:** pytest, pytest-asyncio, httpx
- **CLI:** click or typer

## 8. Iteration Plan

Per the cron-builder budget (max 10 cycles/day, commit after each):

| Cycle | Goal | Verify |
|-------|------|--------|
| 1 (today) | Repo scaffold + SPEC.md + Python package skeleton + LangGraph adapter interface | `pip install -e .` works, `pytest -k test_trace_event` passes |
| 2 | Implement LangGraph adapter: emit node_start/node_end from a compiled graph | Run example graph, dump trace JSON, verify ≥3 event types |
| 3 | FastAPI server + WebSocket endpoint + in-memory run registry | `curl /runs` returns list, WS round-trip works via pytest |
| 4 | `examples/langgraph_basic/` — planner/researcher/writer graph | `python examples/langgraph_basic/graph.py` emits trace end-to-end |
| 5 | `static/index.html` — vanilla JS trace viewer, SVG node/edge rendering | Open in browser, see graph appear, click node → inspector shows data |
| 6 | Timeline scrubber + replay mode | Click scrubber, trace re-renders correctly |
| 7 | Live tick animation, polish inspector pane | Visual review |
| 8 | CLI (`agent-canvas run ...`) + auto-open browser | End-to-end smoke test |
| 9 | README, contributing guide, example screenshots | Doc review |
| 10 | Final acceptance test against §5 criteria, package for PyPI publish | All 8 acceptance criteria pass |

If v0.1 lands in fewer cycles, advance to v0.2 (CrewAI adapter).

## 9. Open Questions

- LangGraph's callback API is the natural integration point — `langchain_core.tracers.BaseTracer` or `langgraph.pregel.StreamProtocol`? Will pick whichever is most stable as of v0.2.x.
- WebSocket vs SSE — v0.1 uses WS for bidirectional (so frontend can send commands like "pause" later). SSE is simpler if we never need client→server.
- SVG vs `<canvas>` for rendering — SVG is easier for v0.1 interactivity (click on element, hit-test). Canvas is faster for huge traces (>500 nodes). Will start with SVG.