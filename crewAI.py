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
        llm2 = ChatGroq(model=f"groq/gemma2-9b-it", temperature=temp)
        #Ollama
        # llm = Ollama(
        #     model=model,
        #     temperature=temp,
        #     timeout=600,
        #     verbose=verbose_mode)
        # llm=LLM(model=f"ollama/{model}", base_url="http://localhost:11434", temperature=temp)
        llm=LLM(model=f"ollama/{model}", base_url="http://localhost:11434")

        # Initialize tools
        research_tool = ResearchTools()
        # Define agents
        researcher = Agent(
            role='Senior Research Analyst',
            goal=f'Provide investment-relevant insights into {ticker} by analyzing its financial statements, performance trends, and stability',
            backstory="""You are a senior financial analyst at Goldman Sachs specializing in dissecting corporate financials, SEC filings, and live market data to inform institutional investment decisions. You are deeply familiar with interpreting 10-Ks, financial ratios, and identifying red flags in business models.""",
            verbose=verbose_mode,
            allow_delegation=False,
            tools=[
                ResearchTools.search_financial_data, #calls yahoo finance
                ResearchTools.sec_filing, #crawls sec.gov for 10-K filings
            ],
            llm=llm)

        researcher2 = Agent(
            role='Market Sentiment Analyst',
            goal=f'Deliver actionable insights on {ticker} by assessing public sentiment and evaluating its competitive positioning in the market',
            backstory="""You specialize in real-time market sentiment tracking. Your expertise includes parsing financial news, social media, and analyst reports to assess public perception and company positioning relative to peers.""",
            verbose=verbose_mode,
            allow_delegation=False,
            tools=[
                # ResearchTools.analyze_market_sentiment_yahoo,
                ResearchTools.analyze_market_sentiment_serper, # uses Serper API for google search results (headlines)
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
            goal='Translate analysis into a polished, investor-friendly report with actionable takeaways',
            backstory="""You are a senior editor at The Wall Street Journal, known for crafting compelling and insightful financial reports for institutional and retail investors alike. Your strength lies in distilling complexity into clarity.""",
            verbose=verbose_mode,
            llm=llm,
            allow_delegation=False)

        # Create tasks
        # task1 = Task(
        #     description=
        #     f"""Analyze {ticker}'s financial health and business model based on the latest SEC 10-K filing. Focus on:
            
        #     1. Key financial metrics and trends (revenue, profit margins, debt levels)
        #     2. Business model viability and competitive advantages
        #     3. Major risk factors that could impact investment
            
        #     Provide a concise assessment of the company's financial stability and growth potential.""",
        #     expected_output="Concise financial analysis with investment-relevant insights",
        #     agent=researcher)
        task1 = Task(
            description=f"""Perform a comprehensive financial health and business model analysis of {ticker} using financial data from Yahoo Finance the most recent SEC 10-K filing. For each section below, extract information from your tools:

        1. **Key Financial Metrics & Trends**
        - Source: Yahoo Finance.
        - Analyze revenue growth, profit margins, debt levels, and other core indicators. Highlight any material changes or red flags. Include relevant financial ratios such as ROE, current ratio, and debt-to-equity. Compare year-over-year or quarter-over-quarter performance where applicable.

        2. **Business Model Viability & Moat**  
        - Source: SEC 10-K Filing: Item 1 (Business).
        - Assess how the company makes money, its market position, and any sustainable competitive advantages.

        3. **Risk Factors**  
        - Source: SEC 10-K Filing: Item 1A (Risk Factors).
        - Summarize significant risks disclosed in the 10-K (e.g., legal, operational, market-based) that could impact investors.

        Use all available tools to gather up-to-date and comprehensive data. Your goal is to support fundamental investment decision-making with a data-driven summary.

        ### Output Format:

        **{ticker} Financial Health & Business Model Assessment**

        **1. Key Financial Metrics & Trends**:  
        ...summary here...

        **2. Business Model Viability & Competitive Advantage**:  
        ...summary here...

        **3. Major Risk Factors**:  
        ...summary here...

        **Investment-Relevant Insight**:  
        Provide a bottom-line assessment (e.g., "financially stable with strong margins, but watch for regulatory risks in international operations").
        """,
            expected_output="Structured investor-grade analysis covering metrics, model, risks, and a bottom-line insight.",
            agent=researcher
        )

        task2 = Task(
            description=f"""Conduct a comprehensive sentiment and market position analysis for {ticker}. Your output should support investors in understanding both qualitative sentiment and competitive standing. Your analysis must include:

            1. **Overall Sentiment Assessment** — Summarize the general tone of recent news, analyst reports, and public discussions (e.g., Positive, Neutral, Negative, or Mixed). Focus on developments from the past 30–60 days to ensure timely relevance.
            2. **Article Sentiment Count** — Tally the number of recent articles classified as Positive, Neutral, or Negative based on sentiment cues.
            3. **Key Themes Driving Sentiment** — Identify 2–3 recurring narratives or issues shaping sentiment (e.g., earnings beats, regulatory concerns, market competition).
            4. **Competitive Position Overview** — Briefly describe the company's market share, major competitors, and any notable shifts in positioning.
            5. **Investment-Relevant Insights** — Highlight any implications for current or prospective investors based on your findings (e.g., caution flags, growth signals).

            Use web search and sentiment analysis tools to form a clear, structured, investor-friendly report based on current and authoritative sources.

            ### Example Output Format:

            **{ticker} Sentiment Analysis Report**

            **Overall Sentiment Assessment**: Mixed/Neutral

            **Article Sentiment Count**:  
            - Positive: 1  
            - Neutral: 3  
            - Negative: 3  

            aspect base sentiment analysis - go through articles and assess sentiment of different aspects 


            **Key Themes Driving Sentiment**:  
            1. **Market Performance Concerns** – Negative sentiment driven by challenging market environment and stock underperformance.  
            2. **Investor Caution** – Mixed outlook from analysts; some recommend holding rather than buying.  
            3. **Potential for Growth** – Positive tone in some reports anticipating long-term recovery or strength.

            **Competitive Position Overview**:  
            The company maintains a strong presence in its sector but faces increasing pressure from competitors. Recent developments may shift this positioning.

            **Investment-Relevant Insights**:  
            Given the mixed sentiment and competitive pressures, investors may adopt a cautious stance. Long-term growth potential exists, but short-term risks remain.
            """,
            expected_output=(
                "An investor-focused market sentiment and positioning report structured as follows:\n"
                "- Overall Sentiment Assessment\n"
                "- Article Sentiment Count\n"
                "- Key Themes Driving Sentiment\n"
                "- Competitive Position Overview\n"
                "- Investment-Relevant Insights"
            ),
            agent=researcher2
        )


        task3 = Task(
            description=f"""Based on the financial and market analysis of {ticker}, evaluate its strategic outlook and innovation potential. Include:

            1. **Future Growth & Innovation** — Evaluate trends like AI, AR/VR, sustainability, and how the company is positioned to capitalize.
            2. **Potential Disruptions** — Identify external or internal risks (technological, geopolitical, regulatory) that could threaten its model.
            3. **Sustainability of Competitive Moat** — Assess if the company can defend or extend its advantages in 3–5 years.

            Focus on long-term value creation, scalability, and leadership in innovation. Consider broader macro and industry-level shifts.""",
            expected_output="Forward-looking analysis of investment potential",
            agent=visionary,
            context=[task1, task2])

        task4 = Task(
            description=f"""Create a final investment recommendation report for {ticker} that:

            1. Summarizes financials, sentiment, and outlook in bullet-point format
            2. Highlights investment pros/cons from both short- and long-term perspectives
            3. Provides a clear recommendation (Buy/Hold/Sell) with rationale
            4. States who the recommendation is most appropriate for (e.g., risk-averse long-term investors vs. short-term traders)

            The report must be markdown formatted, evidence-based, and easy to scan. Include key metrics, themes, and a summary paragraph.""",
            expected_output=
            f"A concise investment recommendation report for {ticker} in markdown format",
            agent=writer,
            context=[task1, task2, task3])

        # Create and run the crew
        crew = Crew(
            agents=[researcher, researcher2, visionary, writer],
            tasks=[task1, task2, task3, task4],
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
