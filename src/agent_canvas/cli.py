"""CLI entry point for Agent Canvas.

Usage:
    agent-canvas run examples/langgraph_basic/graph.py
    agent-canvas serve
"""

from __future__ import annotations

import importlib.util
import logging
import sys
from pathlib import Path

import typer

from agent_canvas.server import serve

logger = logging.getLogger("agent_canvas.cli")
app = typer.Typer()


@app.command()
def run(
    graph_path: str = typer.Argument(..., help="Path to a Python file that defines a compiled LangGraph graph as `graph`."),
    host: str = typer.Option("127.0.0.1", "--host", help="Bind address."),
    port: int = typer.Option(8765, "--port", "-p", help="HTTP port."),
    no_browser: bool = typer.Option(False, "--no-browser", help="Don't open browser automatically."),
) -> None:
    """Run a LangGraph workflow and open the trace viewer."""
    # Import the graph module
    graph_path = Path(graph_path).resolve()
    if not graph_path.exists():
        logger.error("Graph file not found: %s", graph_path)
        raise typer.Exit(code=1)

    spec = importlib.util.spec_from_file_location("user_graph", graph_path)
    if spec is None or spec.loader is None:
        logger.error("Could not load graph file: %s", graph_path)
        raise typer.Exit(code=1)

    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    if not hasattr(mod, "graph"):
        logger.error("Graph file must define a `graph` variable (a compiled LangGraph graph).")
        raise typer.Exit(code=1)

    graph = mod.graph
    run_input = getattr(mod, "input", {})

    from agent_canvas.langgraph_adapter import trace_langgraph
    from agent_canvas.storage import get_default_store

    store = get_default_store()
    run_name = graph_path.stem

    # Start server in background
    import threading

    server_thread = threading.Thread(
        target=serve,
        kwargs={"host": host, "port": port, "store": store, "open_browser": not no_browser},
        daemon=True,
    )
    server_thread.start()

    # Run the graph and stream events
    logger.info("Running graph: %s with input=%s", graph_path.name, run_input)
    for event in trace_langgraph(graph, run_name=run_name, input=run_input, store=store):
        logger.debug("Trace: %s %s", event.type.value, event.node or "")

    logger.info("Graph run complete. Viewer at http://%s:%s", host, port)

    # Keep alive
    try:
        server_thread.join()
    except KeyboardInterrupt:
        logger.info("Shutting down.")


@app.command()
def serve_only(
    host: str = typer.Option("127.0.0.1", "--host", help="Bind address."),
    port: int = typer.Option(8765, "--port", "-p", help="HTTP port."),
) -> None:
    """Start the server without running a graph."""
    serve(host=host, port=port, open_browser=True)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    app()


if __name__ == "__main__":
    main()
