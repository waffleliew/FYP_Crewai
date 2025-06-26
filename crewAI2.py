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
from crewai_tools import SerperDevTool


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
        
        # Define agents
        researcher = Agent(
            role='Senior Research Analyst',
            goal=f'Provide investment-relevant insights into {ticker} by analyzing its financial statements, performance trends, and stability',
            backstory="""You are a senior financial analyst at Goldman Sachs specializing in dissecting corporate financials, SEC filings, and live market data to inform institutional investment decisions. You have developed a systematic approach to handling large documents:

            1. Document Processing Strategy:
            - Break down large SEC filings into manageable sections
            - Process one section at a time (e.g., Item 1, then Item 1A)
            - Extract key information before moving to the next section
            - Maintain focus on the specific task requirements

            2. Analysis Framework:
            - Start with financial metrics from Yahoo Finance
            - Then analyze business model from SEC Item 1
            - Finally review risk factors from SEC Item 1A
            - Structure findings according to the task outline

            3. Quality Control:
            - Verify all data points are properly sourced
            - Cross-reference information between sections
            - Ensure complete coverage of required topics
            - Format output according to specified structure

            Your expertise lies in transforming complex financial documents into clear, actionable insights while maintaining accuracy and completeness.""",
            verbose=verbose_mode,
            allow_delegation=False,
            tools=[
                research_tool.search_financial_data,  # calls yahoo finance
                research_tool.sec_filing,  # crawls sec.gov for 10-K filings
            ],
            llm=llm)

        researcher2 = Agent(
            role='Investment Sentiment Analyst',
            goal=f'Provide actionable investment insights through comprehensive sentiment analysis of {ticker}',
            backstory="""You are a seasoned investment sentiment analyst with expertise in both quantitative and qualitative market analysis. Your specialty is breaking down complex market sentiment into actionable investment insights. You excel at identifying patterns in news coverage, distinguishing between noise and significant market signals, and providing evidence-based analysis that helps investors make informed decisions. You have a track record of identifying early market sentiment shifts that proved valuable for investment timing.""",
            verbose=verbose_mode,
            allow_delegation=False,
            tools=[
                research_tool.analyze_market_sentiment_yahoo,
                # research_tool.analyze_market_sentiment_serper,  # uses Serper API for google search results (headlines)
            ],
            llm=llm)

        visionary = Agent(
            role='Visionary',
            goal='Forecast the future strategic and financial trajectory of the company based on trends, innovation potential, and industry shifts',
            backstory="""You are a technology futurist with a strong financial acumen. You specialize in analyzing how emerging technologies, shifting consumer behavior, and global trends impact a company's long-term viability and competitive edge.""",
            verbose=verbose_mode,
            allow_delegation=False,
            llm=llm)

        writer = Agent(
            role='Senior Editor',
            goal='Translate analysis into a comprehensive, evidence-based investment report with detailed supporting data',
            backstory="""You are a senior editor at The Wall Street Journal, known for crafting detailed and insightful financial reports for institutional and retail investors alike. Your strength lies in providing comprehensive analysis with concrete evidence and supporting data. You always:
            1. Include specific excerpts and quotes to support your analysis
            2. Provide detailed breakdowns of all metrics and ratios
            3. Show the evolution of trends over time
            4. Include concrete numbers and facts
            5. Structure information in clear, hierarchical sections
            6. Never summarize without providing supporting evidence""",
            verbose=verbose_mode,
            llm=llm,
            allow_delegation=False)

        task1 = Task(
            description=f"""Conduct a current-state financial and operational analysis of {ticker} using the most recent financial data and SEC filings. Your focus is to provide a clear, objective view of the company's health and business model based on available disclosures.

            Use the following structure:

            1. **Key Financial Metrics & Trends**  
            - Source: Yahoo Finance.  
            - Analyze revenue growth, profit margins, debt/equity levels, and key financial ratios (ROE, current ratio, debt-to-equity).  
            - Highlight material changes or red flags over time (YoY/QoQ comparisons where relevant).

            2. **Business Model Snapshot**  
            - Source: SEC 10-K Filing, Item 1 (Business).  
            - Describe how the company generates revenue and its current market positioning.  
            - Summarize its customer base, product lines, and go-to-market strategy.  
            - Avoid forward-looking assessments (leave this for strategic analysis later).

            3. **Disclosed Risk Factors**  
            - Source: SEC 10-K Filing, Item 1A (Risk Factors).  
            - Summarize the company's own stated risks—legal, operational, market, or geopolitical—and their potential investor implications.

            4. **Bottom-Line Financial Insight**  
            - Provide a 3–4 sentence summary on the company's current financial standing, operational stability, and any red flags or positives that are particularly relevant to investors.

            Use all assigned tools. Focus purely on **present-state fundamentals** and **disclosed risks**.
            """,
            expected_output="Structured investor-grade snapshot of financials, business operations, and risk disclosure.",
            agent=researcher
        )


        task2 = Task(
            description=f"""**CRITICAL INSTRUCTIONS** :As an investment analyst, conduct a structured media sentiment analysis for {ticker} to support near-term decision-making. Your report should follow this format:

            1. **Two-Level Sentiment Analysis**
            For each article analyzed, include:
            - Title: [Full Title]
            - Date: [Publication Date]
            - Source: [News Source]
            - Headline Sentiment: [Positive / Neutral / Negative]
            - Content Sentiment: [Positive / Neutral / Negative / Mixed]

            - Excerpts & Sentiment Labels:
                • "[Key quote from the article]" — **Sentiment**
                • "[Second relevant quote]" — **Sentiment**

            After reviewing all articles:
            - Headline Sentiment Ratio: Positive : Neutral : Negative
            - Content Sentiment Ratio: Positive : Neutral : Negative
            - Highlight any major discrepancies between headlines and article bodies.

            2. **Sentiment Trends & Patterns**
            - Changes in tone over the last 30–60 days
            - Recurring media themes or investor narratives
            - Spikes or drops in sentiment linked to company events or market news

            3. **Sentiment-Driven Impact Outlook**
            - Short-Term (1–3 months): Is media tone likely to affect stock momentum?
            - Medium-Term (3–6 months): Will sentiment support or challenge investor confidence?
            - Long-Term (>6 months): Any early sentiment signals of lasting perception shifts?

            4. **Actionable Sentiment Signals**
            - Recommended triggers to monitor (e.g., shift to overwhelmingly negative press)
            - Early warning signs or opportunity indicators from news patterns
            - Tactical considerations for adjusting investment exposure

            Use available sentiment tools to extract and synthesize relevant news data. Avoid duplicating financial or risk analysis from fundamental reports — focus specifically on how **market sentiment** may shape investor behavior in the near to medium term.
            """,
            expected_output=(
                "A sentiment-driven investment report including:\n"
                "1. Two-Level Sentiment Analysis (headline vs. content)\n"
                "2. Sentiment Trends & Patterns\n"
                "3. Sentiment-Driven Impact Outlook (1–6+ month windows)\n"
                "4. Actionable Sentiment Signals (investor triggers and alerts)"
            ),
            agent=researcher2
        )


        task3 = Task(
            description=f"""Analyze the **forward-looking strategic and innovation outlook** for {ticker}, building on previous financial and sentiment analyses. Your goal is to assess the company's positioning over the next 3–5 years in the context of industry trends, disruption risks, and its competitive edge.

            Focus on the following:

            1. **Innovation & Growth Potential**  
            - How well is the company positioned to leverage emerging trends such as AI, automation, sustainability, and digital transformation?  
            - Mention relevant R&D, partnerships, or market expansions if available.

            2. **Disruption Risk Profile**  
            - Identify macro and industry-specific risks (e.g., regulatory shifts, tech disruption, consumer shifts, competitive threats) that could impact the company's trajectory.  
            - Incorporate relevant signals from recent sentiment or news analysis if useful.

            3. **Moat Sustainability Outlook**  
            - Assess whether the company's current competitive advantages (brand, tech, scale, distribution, etc.) are likely to hold or erode.  
            - Discuss risks to market leadership and how adaptable the company seems to be.

            Deliver a forward-looking strategic analysis focused on **long-term value creation**, **resilience**, and **industry dynamics**. Avoid rehashing financials unless they're directly tied to strategic capacity.""",
            expected_output="Strategic 3–5 year outlook covering innovation positioning, disruption risk, and competitive durability.",
            agent=visionary,
            context=[task1, task2]
        )


        task4 = Task(
            description=f"""Create a comprehensive investment recommendation report for {ticker} that:

            1. **Financial Analysis**
               - Include specific metrics and ratios with supporting data
               - Show year-over-year comparisons
               - Break down revenue streams and growth rates
               - Provide detailed margin analysis

            2. **Sentiment Analysis**
               - Two-Level Sentiment Analysis:
                 • Headline vs. Content sentiment comparison
                 • Sentiment ratios (Positive:Neutral:Negative)
                 • Key quote excerpts with sentiment labels
               - Sentiment Trends & Patterns:
                 • 30-60 day sentiment evolution
                 • Recurring media themes and narratives
                 • Event-linked sentiment spikes/drops
               - Sentiment-Driven Impact:
                 • Short-term (1-3 month) momentum indicators
                 • Medium-term (3-6 month) confidence signals
                 • Long-term perception shift indicators
               - Actionable Signals:
                 • Key sentiment triggers to monitor
                 • Early warning indicators
                 • Investment exposure recommendations

            3. **Risk Assessment**
               - List specific risks with supporting evidence
               - Include regulatory concerns with concrete examples
               - Break down market risks with data points
               - Provide competitive analysis with specific metrics

            4. **Investment Recommendation**
               - Provide detailed pros/cons with supporting data
               - Include short-term and long-term perspectives
               - Give specific price targets with rationale
               - State clear investment timeframes
               - Specify target investor profiles

            The report must be markdown formatted, evidence-based, and comprehensive. Every claim must be supported by specific data or excerpts.""",
            expected_output=f"A detailed investment recommendation report for {ticker} in markdown format with supporting evidence",
            agent=writer,
            context=[task1, task2, task3]
            )

        # Create and run the crew
        crew = Crew(
            agents=[researcher, researcher2, visionary, writer],
            tasks=[task1 ,task2, task3, task4],
            sequential=True,
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
        f.write(str(content))
    
    return str(filepath)
