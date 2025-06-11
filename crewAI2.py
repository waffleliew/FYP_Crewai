import streamlit as st
import os
import sys
import time
import json
from datetime import datetime
from pathlib import Path

from crewai import Agent, Task, Crew, LLM
from langchain_groq import ChatGroq
from tools.research_tools import ResearchTools
from crewai_tools import FirecrawlCrawlWebsiteTool


# Function to run the analysis
def run_analysis(ticker, model, temp=0.7, verbose_mode=True):
    try:
        print(f"Running analysis using model {model}")

        # Initialize LLM
        llm = LLM(model=f"ollama/{model}", base_url="http://localhost:11434")

        # Initialize tools
        research_tool = ResearchTools()

        # Define agents
        researcher = Agent(
            role='Senior Research Analyst',
            goal=f'Uncover insights into {ticker} and their financials',
            backstory="You work as a research analyst at Goldman Sachs, focusing on fundamental research.",
            verbose=verbose_mode,
            allow_delegation=False,
            tools=[
                ResearchTools.search_financial_data,
                ResearchTools.sec_filing,
            ],
            llm=llm
        )

        researcher2 = Agent(
            role='Competitive Intelligence Analyst',
            goal=f'Analyze market sentiment and public perception of {ticker}',
            backstory="You monitor social media, analyst opinions, and financial news to assess sentiment.",
            verbose=verbose_mode,
            allow_delegation=False,
            tools=[
                ResearchTools.analyze_market_sentiment,
                ResearchTools.web_search,  # Added Google search capability
            ],
            llm=llm
        )

        visionary = Agent(
            role='Visionary',
            goal='Deep thinking on the implications of the analysis',
            backstory="You are a visionary technologist with a keen eye for future industry shifts.",
            verbose=verbose_mode,
            allow_delegation=False,
            llm=llm
        )

        writer = Agent(
            role='Senior Editor',
            goal='Write a professional report that is easy to understand',
            backstory="You are a senior editor at the Wall Street Journal, known for insightful reporting.",
            verbose=verbose_mode,
            llm=llm,
            allow_delegation=True
        )

        # Create tasks with explicit names
        task1 = Task(
            name="sec_analysis",
            description=f"""Analyze {ticker}'s latest SEC Form 10-K covering business model, risks, MD&A, competition, and future outlook.""",
            expected_output="Full analysis report in bullet points",
            agent=researcher
        )

        task2 = Task(
            name="competitive_analysis",
            description=f"""Analyze {ticker}'s market share, competitor advantages, threats, and strategic position.""",
            expected_output="Comprehensive competitive analysis report with market positioning insights",
            agent=researcher2
        )

        # Create crew to run tasks 1 and 2 first
        initial_crew = Crew(
            agents=[researcher, researcher2],
            tasks=[task1, task2],
            verbose=verbose_mode,
        )

        initial_results = initial_crew.kickoff()
        research_report = initial_results["sec_analysis"]
        competitive_report = initial_results["competitive_analysis"]

        # Task 3 now depends on task1's result
        task3 = Task(
            name="visionary_insights",
            description=f"""Using the insights from the research report on {ticker}, think deeply about future industry implications.""",
            context=[research_report],
            expected_output="Analysis report with deeper insights in implications",
            agent=visionary
        )

        # Run visionary independently
        vision_crew = Crew(agents=[visionary], tasks=[task3], verbose=verbose_mode)
        vision_result = vision_crew.kickoff()
        vision_report = vision_result["visionary_insights"]

        # Final report generation with all inputs
        task4 = Task(
            name="final_report",
            description=f"""Using the insights from previous tasks on {ticker}, write a publishable WSJ-style markdown report with sections like executive summary, company overview, financial analysis, competition, future outlook, and investment implications.""",
            context=[research_report, competitive_report, vision_report],
            expected_output="A detailed comprehensive markdown report",
            agent=writer
        )

        # Final crew just for writing
        final_crew = Crew(agents=[writer], tasks=[task4], verbose=verbose_mode)
        final_result = final_crew.kickoff()
        final_report = final_result["final_report"]

        return final_report

    except Exception as e:
        return f"Error during analysis: {str(e)}"


# Function to save the report to a file
def save_report(ticker, content):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{ticker}_{timestamp}.md"
    reports_dir = Path('saved_reports')
    reports_dir.mkdir(exist_ok=True)
    filepath = reports_dir / filename

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)

    return str(filepath)
