"""Minimal LangGraph workflow: planner -> researcher -> writer.

Demonstrates the Agent Canvas trace adapter with a 3-node linear workflow
that plans a topic, researches it, and writes a summary.
"""

from __future__ import annotations

from typing import TypedDict


class State(TypedDict, total=False):
    """Workflow state passed between nodes."""

    topic: str
    plan: str
    research: str
    output: str
    log: list[str]


def planner(state: State) -> State:
    """Plan the approach for the given topic."""
    topic = state.get("topic", "unknown")
    plan = f"Plan for '{topic}': outline key points, gather facts, synthesize."
    return {
        "plan": plan,
        "log": [*state.get("log", []), f"planner: created plan for '{topic}'"],
    }


def researcher(state: State) -> State:
    """Research the planned topic."""
    topic = state.get("topic", "unknown")
    research = f"Research notes on '{topic}': found 3 relevant sources."
    return {
        "research": research,
        "log": [*state.get("log", []), f"researcher: gathered data on '{topic}'"],
    }


def writer(state: State) -> State:
    """Write the final output based on plan and research."""
    topic = state.get("topic", "unknown")
    plan = state.get("plan", "No plan available.")
    research = state.get("research", "No research available.")
    output = f"# {topic}\n\n{plan}\n\n{research}\n\n## Summary\n\nA concise summary of '{topic}'."
    return {
        "output": output,
        "log": [*state.get("log", []), f"writer: produced output for '{topic}'"],
    }


# Build the graph
from langgraph.graph import END, START, StateGraph

builder = StateGraph(State)
builder.add_node("planner", planner)
builder.add_node("researcher", researcher)
builder.add_node("writer", writer)
builder.add_edge(START, "planner")
builder.add_edge("planner", "researcher")
builder.add_edge("researcher", "writer")
builder.add_edge("writer", END)

graph = builder.compile()

# Default input — overridable by CLI
input = {"topic": "AI Agent Observability"}
