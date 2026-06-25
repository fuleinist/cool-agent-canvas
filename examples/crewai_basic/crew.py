"""Minimal CrewAI workflow: Researcher -> Writer.

Demonstrates the Agent Canvas CrewAI trace adapter with a 2-agent
sequential crew that researches a topic and writes a summary.
"""

from __future__ import annotations

from agent_canvas.crewai_adapter import trace_crewai
from agent_canvas.storage import InMemoryStore


def main() -> None:
    import crewai

    researcher = crewai.Agent(
        role="Researcher",
        goal="Find information about the given topic",
        backstory="You are a thorough researcher who finds relevant facts.",
        allow_delegation=False,
        verbose=False,
    )
    writer = crewai.Agent(
        role="Writer",
        goal="Write a clear summary based on research",
        backstory="You are a concise writer who produces well-structured summaries.",
        allow_delegation=False,
        verbose=False,
    )

    research_task = crewai.Task(
        name="research",
        description="Research the topic: AI Agent Observability",
        expected_output="A list of key facts and findings",
        agent=researcher,
    )
    write_task = crewai.Task(
        name="write",
        description="Write a short report based on the research findings",
        expected_output="A short report in markdown",
        agent=writer,
    )

    crew = crewai.Crew(
        agents=[researcher, writer],
        tasks=[research_task, write_task],
        verbose=False,
    )

    store = InMemoryStore()
    events = list(trace_crewai(crew, run_name="crewai-demo", store=store))

    print(f"Captured {len(events)} trace events:")
    for ev in events:
        print(f"  [{ev.type.value:12s}] node={ev.node or '-'}  {_summary(ev)}")


def _summary(ev) -> str:
    if ev.type.value == "run_start":
        return ev.data.get("name", "")
    if ev.type.value == "run_end":
        return f"{ev.data.get('total_ms', 0):.0f}ms"
    if ev.type.value == "state_delta":
        d = ev.data.get("delta", {})
        return f"agent={d.get('agent', '?')}  output={d.get('output', '')[:60]}"
    return ""


if __name__ == "__main__":
    main()
