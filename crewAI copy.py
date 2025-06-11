import streamlit as st
import os
import sys
import time
import json
from datetime import datetime
from pathlib import Path

# Import your existing modules
from crewai import Agent, Task, Crew, LLM
from langchain_groq import ChatGroq
from tools.research_tools import ResearchTools
from crewai_tools import FirecrawlCrawlWebsiteTool


# Function to run the analysis
def run_analysis(ticker, model, temp=0.7, verbose_mode=True):
    try:
        print(f"Running analysis using model {model}")

        # Initialize LLM 
        ##Groq
        # llm = ChatGroq(model=f"groq/{model}", temperature=temp)
        ##Ollama
        # llm = Ollama(
        #     model=model,
        #     temperature=temp,
        #     timeout=600,
        #     verbose=verbose_mode)
        llm=LLM(model=f"ollama/{model}", base_url="http://localhost:11434")
        # Initialize tools
        research_tool = ResearchTools()
        
        # Define agents
        researcher = Agent(
            role='Senior Research Analyst',
            goal=f'Uncover insights into {ticker} and their financials',
            backstory=
            """You work as a research analyst at Goldman Sachs, focusing on fundamental research for potential companies to invest in.""",
            verbose=verbose_mode,
            allow_delegation=False,
            tools=[
                ResearchTools.search_financial_data,
                ResearchTools.sec_filing,
                
            ],
            llm=llm)

        researcher2 = Agent(
            role='Competitive Intelligence Analyst',
            goal=f'Analyze market sentiment and public perception of {ticker}',
            backstory=
            """You monitor social media, analyst opinions, and financial news to assess sentiment.""",
            verbose=verbose_mode,
            allow_delegation=False,
            tools=[
                ResearchTools.analyze_market_sentiment,
                ResearchTools.web_search,  # Added Google search capability

            ],
            llm=llm)

        visionary = Agent(
            role='Visionary',
            goal='Deep thinking on the implications of the analysis',
            backstory=
            """You are a visionary technologist with a keen eye for identifying emerging trends and predicting their potential impact on various industries.""",
            verbose=verbose_mode,
            allow_delegation=False,
            llm=llm)

        writer = Agent(
            role='Senior Editor',
            goal='Write a professional quality report that is easy to understand',
            backstory=
            """You are a details-oriented senior editor at the Wall Street Journal known for your insightful and engaging articles.""",
            verbose=verbose_mode,
            llm=llm,
            allow_delegation=True)

        # Create tasks
        task1 = Task(
            description=
            f"""Please conduct a comprehensive analysis of the latest SEC Form 10-K annual report for {ticker}. The analysis should cover the following key aspects in detail:

            1. Business Overview:
            Provide an in-depth overview of the company's business model, product portfolio, services, and primary target markets.
            
            2. Risk Factors:
            Identify and critically analyze the significant risk factors disclosed in the Risk Factors section of the 10-K filing.
            
            3. Management's Discussion and Analysis (MD&A):
            Summarize the key points from the MD&A section, including changes in revenue, costs, profitability, and cash flows.
            
            4. Competitive Landscape:
            Provide an in-depth analysis of the competitive position of the company within its market.
            
            5. Future Outlook:
            Based on the information in the 10-K filing, industry trends, and your analysis, provide an outlook on the future performance of the company.

            Please ensure that all information and analysis are sourced directly from the latest SEC Form 10-K filings. The analysis should be unbiased, factual, and supported by evidence from the filings.""",
            expected_output="Full analysis report in bullet points",
            agent=researcher)

        task2 = Task(
            description=
            f"""Analyze the competitive landscape for {ticker}. Your analysis should include:

            1. Market Share Analysis:
            - Current market share in key segments
            - Historical market share trends over the past 3-5 years
            - Factors influencing market share changes

            2. Competitor Strengths and Weaknesses:-
            - Core technological advantages of the company
            - Product portfolio assessment
            - Pricing strategies and their effectiveness
            - R&D capabilities and innovation track record

            3. Market Positioning:
            - Brand perception and reputation
            - Customer segments and loyalty
            - Strategic partnerships and ecosystem advantages

            4. Competitive Threats:
            - Emerging competitors in key markets
            - Potential disruptive technologies
            - Regulatory challenges affecting competitive dynamics

            Use all available tools to gather comprehensive competitive intelligence. Your analysis should be data-driven, objective, and provide actionable insights.
            """,
            expected_output="Comprehensive competitive analysis report with market positioning insights",
            agent=researcher2)

        task3 = Task(
            description=
            f"""Using the insights provided by the Senior Research Analyst about {ticker}, think through deeply the future implications of the points that are made. Consider the following questions as you craft your response:

            What are the current limitations or pain points that the technologies mentioned in the Senior Research Analyst report could address?

            How might these technologies disrupt traditional business models and create new opportunities for innovation?

            What are the potential risks and challenges associated with the adoption of these technologies, and how might they be mitigated?

            How could these technologies impact consumers, employees, and society as a whole?

            What are the long-term implications of these technologies, and how might they shape the future of the industry?

            Provide a detailed analysis of the potential impact, backed by relevant examples, data, and insights.""",
            expected_output="Analysis report with deeper insights in implications",
            agent=visionary)

        task4 = Task(
            description=
            f"""Using the insights provided by the Senior Research Analyst, Competitive Intelligence Analyst, and Visionary about {ticker}, please craft an expertly styled report that is targeted towards the investor community. Make sure to also include the long-term implications insights that your co-worker, Visionary, has shared.  
            
            Please ensure that the report is written in a professional tone and style, and that all information is sourced from the latest SEC 10-K filings for {ticker}, as well as the competitive analysis provided. Write in a format and style worthy to be published in the Wall Street Journal.
            
            The report should be structured with clear sections, including an executive summary, company overview, financial analysis, competitive positioning, future outlook, and investment considerations.""",
            expected_output=
            f"A detailed comprehensive report about {ticker} that expertly presents the research done by your co-workers. Please provide report in markdown language",
            agent=writer)

        # Create and run the crew
        crew = Crew(
            agents=[researcher, researcher2, visionary, writer],
            tasks=[task1, task2, task3, task4],
            verbose=verbose_mode,
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
        f.write(content)
    
    return str(filepath)
