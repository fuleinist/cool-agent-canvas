"""Minimal CrewAI workflow for testing the CrewAI adapter.

A research crew with two agents: a Researcher who gathers info and a Writer
who produces the final report. Uses sequential process.
"""

from crewai import Agent, Crew, Process, Task

researcher = Agent(
    role="Researcher",
    goal="Find and summarize information about {topic}",
    backstory="You are an expert researcher who finds accurate information quickly.",
    allow_delegation=False,
    verbose=False,
)

writer = Agent(
    role="Writer",
    goal="Write a clear, concise report about {topic} based on research",
    backstory="You are a technical writer who produces clear, well-structured reports.",
    allow_delegation=False,
    verbose=False,
)

research_task = Task(
    description="Research {topic} and provide key findings, trends, and important facts.",
    expected_output="A bullet-point summary of key findings about {topic}",
    agent=researcher,
)

write_task = Task(
    description="Write a concise report about {topic} based on the research findings.",
    expected_output="A well-structured report about {topic} with introduction, key points, and conclusion",
    agent=writer,
)

crew = Crew(
    name="Research Crew",
    agents=[researcher, writer],
    tasks=[research_task, write_task],
    process=Process.sequential,
    verbose=False,
)

# Default input for standalone execution
input = {"topic": "AI agents for software development"}
