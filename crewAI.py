import streamlit as st
import os
import sys
import time
import json
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
# Import your existing modules
from crewai import Agent, Task, Crew, LLM
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from tools.research_tools import ResearchTools

def get_llm(model="llama-3.3-70b-versatile"):
    """
    Get the appropriate LLM based on the model selection.
    Args:
        model (str): The model name/identifier
    Returns:
        LLM: The configured LLM instance
    """
    if model.startswith("gpt-"):
        # OpenAI models
        return ChatOpenAI(
            model=model,
            temperature=0.7,
            api_key=os.getenv("OPENAI_API_KEY")
        )
    else:
        # Groq models (default)
        return ChatGroq(model=f"groq/llama-3.3-70b-versatile")

# Function to run the analysis
def run_analysis(transcript, ticker=None, model="llama-3.3-70b-versatile", temp=0.7, verbose_mode=True):
    """
    Run analysis on an earnings call transcript, optionally enriching with external data.
    Args:
        transcript (str): The earnings call transcript text.
        ticker (str, optional): The company ticker symbol for enrichment.
        model (str): LLM model name.
        temp (float): LLM temperature.
        verbose_mode (bool): Verbosity flag.
    Returns:
        str: The generated analyst report.
    """
    try:
        print(f"Running analysis using model {model}")

        # Initialize LLM 
        llm = get_llm(model)

        # Initialize tools
        research_tool = ResearchTools()
        
        # Define agents with memory
        financial_analyst = Agent(
            role='Financial Analyst',
            goal=f"Analyze the company's financial health and performance based solely on the provided earnings call transcript.",
            backstory="""You are a senior financial analyst specializing in extracting quantitative insights from earnings call transcripts. Your analyses are known for their accuracy, attention to detail, and ability to identify key financial drivers using only the transcript content.""",
            verbose=verbose_mode,
            allow_delegation=False,
            memory=False,
            tools=[
                ResearchTools.search_financial_data,
                ResearchTools.analyze_financial_ratios,
                ResearchTools.analyze_historical_trends,
                ResearchTools.sec_mda
            ],
            llm=llm
        )

        market_analyst = Agent(
            role='Market Analyst',
            goal=f"Analyze the company's market position and sentiment based solely on the provided earnings call transcript.",
            backstory="""You are a market analysis expert specializing in extracting competitive and sentiment insights from earnings call transcripts. Your analyses are known for their strategic insights and ability to identify key market drivers using only the transcript content.""",
            verbose=verbose_mode,
            allow_delegation=False,
            memory=False,
            tools=[
                ResearchTools.analyze_competitors,
                ResearchTools.compare_industry_metrics,
                ResearchTools.sec_business_overview,
                ResearchTools.sec_mda,
                ResearchTools.analyze_market_sentiment_yahoo
            ],
            llm=llm
        )

        risk_analyst = Agent(
            role='Risk Analyst',
            goal=f"Identify and assess key risks based solely on the provided earnings call transcript.",
            backstory="""You are a risk analysis specialist focusing on extracting and evaluating risk factors from earnings call transcripts. Your analyses are known for their thorough risk assessment and balanced view of opportunities and threats using only the transcript content.""",
            verbose=verbose_mode,
            allow_delegation=False,
            memory=False,
            tools=[
                ResearchTools.analyze_financial_ratios,
                ResearchTools.sec_risk_factors,
                ResearchTools.sec_market_risk,
                ResearchTools.assess_management_quality
            ],
            llm=llm
        )

        report_writer = Agent(
            role='Investment Report Writer',
            goal=f"Create a persuasive, well-structured investment report based solely on transcript analysis.",
            backstory="""You are a senior investment report writer specializing in synthesizing transcript-based analyses into clear, actionable investment reports. Your reports are known for their readability, actionable insights, and professional presentation, using only the transcript content.""",
            verbose=verbose_mode,
            allow_delegation=False,
            tools=[],  # No tools needed as this agent synthesizes others' work
            llm=llm
        )

        # Define tasks
        task1 = Task(
            description=f"""
Analyze the following earnings call transcript for {ticker}:

{transcript}

Extract and analyze financial health and performance. Focus on:
- Revenue and profit margins discussed in the call
- Key financial ratios (profitability, liquidity, leverage)
- Growth rates and trends mentioned
- Any material changes or red flags highlighted by management
Use only the information present in the transcript.
""",
            expected_output="Key financial metrics and ratios analysis, with transcript quotes.",
            agent=financial_analyst
        )

        task2 = Task(
            description=f"""
Analyze the following earnings call transcript for {ticker}:

{transcript}

Analyze the company's market position and sentiment as discussed in the transcript. Focus on:
- Management's comments on market share, competition, and industry trends
- Analyst or participant questions about market position
- Any sentiment signals (positive/negative) from the transcript
Use only the information present in the transcript.
""",
            expected_output="Comprehensive market and sentiment analysis, with transcript evidence.",
            agent=market_analyst
        )

        task3 = Task(
            description=f"""
Analyze the following earnings call transcript for {ticker}:

{transcript}

Identify and assess key risks mentioned in the transcript. Focus on:
- Business, financial, and operational risks discussed by management or analysts
- Any new/emerging risks highlighted in the call
Use only the information present in the transcript.
""",
            expected_output="Key risk factors and risk assessment, with transcript evidence.",
            agent=risk_analyst
        )

        task4 = Task(
            description=f"""
Synthesize the findings from transcript analysis into a persuasive Markdown investment report for {ticker}. The report should include:
1. Executive Summary
   - Key investment thesis
   - Major findings
   - Investment recommendation
   - Critical risks
2. Financial Analysis (from Task 1)
   - Key metrics and transcript quotes
3. Market Position & Sentiment (from Task 2)
   - Transcript-based findings
4. Risk Assessment (from Task 3)
   - Prioritized risks with transcript evidence
5. Investment Recommendation
   - For each time frame (next day, next week, next month):
     - Provide a clear Long or Short recommendation
     - Give a concise rationale for each time frame based only on the transcript
Format:
- Use Markdown headers for sections
- Create tables for numerical data
- Use bullet points for key findings
- Include transcript quotes as blockquotes
- Maintain professional tone

Earnings call transcript for reference:

{transcript}
Use only the information present in the transcript.
""",
            expected_output="Well-structured Markdown investment report with clear recommendations and supporting data. The investment recommendation section must include explicit Long/Short calls for the next day, week, and month, each with a rationale based only on the transcript.",
            agent=report_writer
        )

        # Create the crew
        crew = Crew(
            agents=[
                financial_analyst,
                market_analyst,
                risk_analyst,
                report_writer
            ],
            tasks=[task1, task2, task3, task4],
            sequential=True,
            verbose=verbose_mode,
            memory=False
        )

        # Run the analysis
        result = crew.kickoff()
        
        # Return the result
        return result
    
    except Exception as e:
        return f"Error during analysis: {str(e)}"

# Function to save the report to a file
def save_report(ticker, content):
    """
    Save the analysis report to a file.
    
    Args:
        ticker (str): The company ticker symbol
        content (str): The report content
        
    Returns:
        str: The path to the saved file
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{ticker}_{timestamp}.md"
    filepath = Path('saved_reports') / filename
    
    with open(filepath, "w") as f:
        f.write(str(content))
    
    return str(filepath)
