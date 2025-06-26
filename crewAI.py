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
from tools.research_tools import ResearchTools


# Function to run the analysis
def run_analysis(ticker, model, temp=0.7, verbose_mode=True):
    try:
        print(f"Running analysis using model {model}")

        # Initialize LLM 
        ##Groq
        llm = ChatGroq(model=f"groq/llama-3.3-70b-versatile")
        #Ollama
        # llm = Ollama(
        #     model=model,
        #     temperature=temp,
        #     timeout=600,
        #     verbose=verbose_mode)
        # llm=LLM(model=f"ollama/{model}", base_url="http://localhost:11434", temperature=temp)
        # llm=LLM(model=f"ollama/{model}", base_url="http://localhost:11434")

        # Initialize tools
        research_tool = ResearchTools()
        
        # Define agents with memory
        financial_analyst = Agent(
            role='Financial Analyst',
            goal=f'Analyze {ticker}\'s financial health and historical performance',
            backstory="""You are a senior financial analyst specializing in quantitative analysis and financial metrics. 
            Your expertise lies in analyzing financial statements, calculating key ratios, and identifying financial trends. 
            You excel at using financial data tools to assess company health, profitability, and growth potential.
            Your analyses are known for their accuracy, attention to detail, and ability to identify key financial drivers.""",
            verbose=verbose_mode,
            allow_delegation=False,
            memory=False,
            tools=[
                ResearchTools.search_financial_data,
                ResearchTools.analyze_financial_ratios,
                ResearchTools.analyze_historical_trends,
                # ResearchTools.sec_financial_statements,
                ResearchTools.sec_mda
            ],
            llm=llm
        )

        market_analyst = Agent(
            role='Market Analyst',
            goal=f'Analyze {ticker}\'s market position and competitive landscape',
            backstory="""You are a market analysis expert specializing in competitive positioning and industry dynamics. 
            Your expertise lies in analyzing business models, market share, and competitive advantages. 
            You excel at using business overview and competitor analysis tools to assess market position and growth potential.
            Your analyses are known for their strategic insights and ability to identify key market drivers.
            You are particularly skilled at analyzing market sentiment and news to gauge market perception.""",
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
            goal=f'Analyze {ticker}\'s key business and financial risks',
            backstory="""You are a risk analysis specialist focusing on business, financial, and competitive risks. 
            Your expertise lies in identifying and assessing key risk factors that could impact investment decisions. 
            You excel at using risk factor and market risk tools to evaluate financial stability and risk exposure.
            Your analyses are known for their thorough risk assessment and balanced view of opportunities and threats.""",
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
            goal=f'Create a clear, well-structured investment report for {ticker} in Markdown format',
            backstory="""You are a senior investment report writer specializing in creating clear, actionable investment reports. 
            Your expertise lies in transforming complex analyses into well-structured, executive-friendly reports using Markdown formatting.
            You excel at:
            - Creating concise executive summaries
            - Organizing data into clear tables
            - Using bullet points for key findings
            - Presenting recommendations with clear rationale
            - Maintaining professional tone and clarity
            Your reports are known for their readability, actionable insights, and professional presentation.""",
            verbose=verbose_mode,
            allow_delegation=False,
            tools=[],  # No tools needed as this agent synthesizes others' work
            llm=llm
        )

        # Define tasks
        task1a = Task(
            description=f"""Analyze current financial health for {ticker}:

            Use search_financial_data and analyze_financial_ratios to evaluate:
            - Revenue and profit margins
            - Key financial ratios (profitability, liquidity, leverage)
            - Current growth rates

            Focus on metrics that indicate financial health and performance.
            """,
            expected_output="Key financial metrics and ratios analysis.",
            agent=financial_analyst
        )

        task1b = Task(
            description=f"""Analyze historical financial performance for {ticker}:

            Use analyze_historical_trends to evaluate:
            - Revenue and profit growth trends
            - Financial stability over time
            - Historical volatility

            Focus on performance consistency and trends.
        """,
            expected_output="Historical financial performance analysis.",
            agent=financial_analyst
        )

        task2 = Task(
            description=f"""Analyze market position and sentiment for {ticker}:

            Use analyze_competitors and compare_industry_metrics to evaluate:
            - Market share and competitive position
            - Financial performance vs industry peers
            - Growth potential and analyst estimates

            Use analyze_market_sentiment_yahoo to assess:

            1. Two-Level Sentiment Analysis
            - For each article: Title, Date, Source, Headline/Content Sentiment
            - Key quotes with sentiment labels
            - Overall sentiment ratios (Positive:Neutral:Negative)

            2. Sentiment Trends & Patterns
            - 30-60 day tone changes
            - Recurring themes and narratives
            - Event-linked sentiment spikes/drops

            3. Sentiment Impact Outlook
            - Short-term (1-3 months) momentum impact
            - Medium-term (3-6 months) confidence outlook
            - Long-term perception shifts

            4. Actionable Signals
            - Key sentiment triggers to monitor
            - Early warning indicators
            - Investment exposure recommendations

            Focus on competitive position, industry context, and market sentiment.
            """,
            expected_output="""Comprehensive market analysis including:
            - Market position and competitive analysis
            - Market sentiment and news analysis
            - Industry context and trends
            - Strategic implications and recommendations""",
            agent=market_analyst
        )

        task3 = Task(
            description=f"""Analyze key risks for {ticker}:

            Use sec_risk_factors and sec_market_risk to identify:
            - Major business and financial risks
            - Market and competitive risks
            - Regulatory and operational risks

            Use analyze_financial_ratios to assess:
            - Financial stability
            - Debt levels
            - Liquidity risks

            Focus on risks that could significantly impact investment returns.
            """,
            expected_output="Key risk factors and financial risk assessment.",
            agent=risk_analyst
        )

        task4 = Task(
            description=f"""Create investment analysis report for {ticker}:

            Produce a coherent Markdown investment report with an executive summary, bullet-point key findings, data tables, and clear recommendations.

            Synthesize the following analyses into a clear investment report:

            1. Executive Summary
            - Key investment thesis
            - Major findings
            - Investment recommendation
            - Critical risks

            2. Financial Analysis
            - Current financial health (from Task 1a)
            - Historical performance trends (from Task 1b)
            - Key financial ratios and metrics
            [Present key metrics in Markdown tables]

            3. Market Position & Sentiment
            - Competitive position and market share
            - Industry comparison
            - Growth potential
            [Use bullet points for key findings]

            Market Sentiment Analysis:
            a) Two-Level Sentiment Analysis
            - Article-by-article breakdown:
                • Title, Date, Source
                • Headline Sentiment (Positive/Neutral/Negative)
                • Content Sentiment (Positive/Neutral/Negative/Mixed)
                • Key aspects in article content with sentiment labels
            - Overall Sentiment Ratios:
                • Headline: Positive:Neutral:Negative
                • Content: Positive:Neutral:Negative

            b) Sentiment Trends & Patterns
            - 30-60 day tone changes
            - Recurring themes and narratives

            c) Sentiment Impact Outlook
            - Short-term (1-3 months) momentum impact
            - Medium-term (3-6 months) confidence outlook
            - Long-term perception shifts

            d) Actionable Sentiment Signals
            - Key sentiment triggers to monitor
            - Early warning indicators
            - Investment exposure recommendations

            4. Risk Assessment
            - Key business and financial risks
            - Market and competitive risks
            - Financial stability and liquidity
            [Prioritize risks by impact]

            5. Investment Recommendation
            - Buy/Sell/Hold recommendation
            - Key investment drivers
            - Major risks to consider
            [Clear, actionable recommendation]

            Format Requirements:
            - Use Markdown headers for sections
            - Create tables for numerical data
            - Use bullet points for key findings
            - Include clear section breaks
            - Maintain professional tone
            - Use tables for sentiment ratios
            - Use blockquotes for key article quotes
            """,
            expected_output="Well-structured Markdown investment report with clear recommendations and supporting data.",
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
            tasks=[task1a, task1b, task2, task3, task4],
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
