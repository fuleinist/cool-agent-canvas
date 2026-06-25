# Agent Canvas

A visual canvas for designing, debugging, and tracing multi-agent workflows.

```
pip install agent-canvas
agent-canvas run examples/langgraph_basic/graph.py
```

Watch your LangGraph / CrewAI / AutoGen workflow paint itself as it runs —
every node, edge, message, and tool call streamed live to a browser UI.

## Status: v0.1 (complete)

| Framework | Status |
|-----------|--------|
| LangGraph | ✅ v0.1 (complete) |
| CrewAI    | 📋 v0.2 |
| AutoGen   | 📋 v0.2 |

See [SPEC.md](SPEC.md) for the full v0.1 design and acceptance criteria.

All 8 v0.1 acceptance criteria pass — 24 unit tests + end-to-end smoke test.

## Quickstart

```bash
git clone https://github.com/fuleinist/cool-agent-canvas
cd cool-agent-canvas
pip install -e .[langgraph,dev]
agent-canvas run examples/langgraph_basic/graph.py
# Browser opens to http://localhost:8765 with the live trace viewer
```

## Why

Multi-agent frameworks are everywhere. Debugging them is black-box chaos —
you stare at a console log and try to reconstruct what the agent did, in what
order, with what state. Agent Canvas turns the workflow itself into the UI:
you watch the graph execute, click any node to see its state, scrub the
timeline to replay, and diff runs side-by-side.

## License

MIT