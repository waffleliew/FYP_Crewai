import streamlit as st
import os
import sys
import time
import json
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
import autogen
from langchain_groq import ChatGroq
from tools.research_tools import ResearchTools
# Set AutoGen Docker configuration
os.environ["AUTOGEN_USE_DOCKER"] = "0"

def get_config_list(model="llama-3.3-70b-versatile"):
    """
    Get the appropriate config list based on the model selection.
    Args:
        model (str): The model name/identifier
    Returns:
        list: Configuration list for the selected model
    """
    if model.startswith("gpt-"):
        # OpenAI models
        return [
            {
                "model": model,
                "api_key": os.getenv("OPENAI_API_KEY"),
                "api_type": "openai"
            }
        ]
    else:
        # Groq models (default)
        return [
            {
                "model": "groq/llama-3.3-70b-versatile",
                "api_key": os.getenv("GROQ_API_KEY"), 
                "api_type": "groq"
            }
        ]

def run_analysis(transcript, ticker=None, model="llama-3.3-70b-versatile", temp=0.7, verbose_mode=True, max_rounds=2, year=None, quarter=None):
    """
    Run analysis on an earnings call transcript using an AutoGen multi-agent feedback loop.
    Returns:
        str: The generated analyst report.
    """
    try:
        print(f"Running analysis using model {model}, for transcript year: {year}, quarter: {quarter}")

        config_list = get_config_list(model)
        # llm_config = {"config_list": config_list, "temperature": temp}
        llm_config = {"config_list": config_list}

        # Agents
        client_agent = autogen.UserProxyAgent(
            name="Client",
            human_input_mode="NEVER",
            llm_config=llm_config,
            max_consecutive_auto_reply=3,
            system_message=(
                "You are the Client (Investor). Review the latest investment report drafted by the Writer Agent. "
                "Your goal is to validate whether the report is complete, correct, and integrates all feedback from the Analyst Agent. "
                "Use the following checklist for evaluation. Do not skip any check. Your final reply should be based solely on this checklist."

                "\n\n### Checklist Evaluation (Answer YES or NO to each):"

                "\n1. Does the report include all **four required sections** with Markdown headers?"

                "\n2. Are the financial metrics properly populated and formatted?"
                "\n   a) Is the financial table complete with all nine required metrics:"
                "\n      - Revenue"
                "\n      - EPS"
                "\n      - Gross Margin"
                "\n      - Operating Margin"
                "\n      - Net Margin"
                "\n      - Operating Cash Flow"
                "\n      - Capex"
                "\n      - Total Debt"
                "\n      - Cash & Cash Equivalents"
                "\n   b) Does each metric have:"
                "\n      - Current quarter value"
                "\n      - Previous quarter value"
                "\n      - QoQ change percentage (with + or - sign)"
                "\n   c) Are missing values marked as 'N/A' instead of being left blank?"
                "\n   d) Are all percentages formatted consistently?"

                "\n3. Does the Risk Assessment section include properly populated risks with supporting evidence from the transcript, including Likelihood and Impact values?"

                "\n4. Refer to the feedback from the Analyst Agent and ensure **all Analyst feedback points clearly addressed** in the final report"

                "\n5. Are any placeholder phrases like **'No information available'** omitted from the report?"

                "\n6. Are **tables used consistently** for structured data, and bullet points used for key findings?"


                "\n\n### Final Instruction:"
                "\n- If **ALL** answers are YES, reply only with: TERMINATE"
                "\n- If **ANY** answer is NO, return a numbered list of the failed checks and specify what exactly needs to be revised or added."
            )
        )


        writer_prompt = (
            """You are the Writer Agent. Your task is to draft and revise a structured investment report using Markdown formatting, based on the earnings call transcript (prepared remarks + Q&A) and feedback from other agents.

        The report must include the following four sections. Use **Markdown tables** for every subsection where structured data or comparisons are applicable. If data is missing, omit the subsection.

        ---

        ## 1. Financial Analysis

        Summarize financial performance using the following table format:
        First, analyze the transcript and populate this table with any metrics explicitly mentioned or that can be calculated from the transcript data:

        ### Key Financial Metrics
        
        | Metric                    | Current Quarter | Previous Quarter | QoQ Change |
        |---------------------------|-----------------|------------------|------------|
        | Revenue                   |                |                  |            | 
        | Earnings per Share (EPS)  |                |                  |            | 
        | Gross Profit             |                |                  |            | 
        | Operating Income         |                |                  |            |
        | Net Income               |                |                  |            | 
        | Operating Cash Flow      |                |                  |            | 
        | Capex                    |                |                  |            | 
        | Short-term Debt          |                |                  |            |
        | Long-term Debt           |                |                  |            |
        | Cash & Cash Equivalents  |                |                  |            | 

        **Important:**
        - Only populate metrics that are explicitly mentioned in the transcript or can be calculated from transcript data
        - Leave cells blank if the data is not available in the transcript
        - Calculate QoQ changes where both quarters' data is available
        - The Analyst will fact-check these values against financial statements and provide corrections for you to revise.

        
        ### Key Financial Ratios and Investment Insights

        | Metric                | Current Quarter | Previous Quarter   | Formula   | Interpretation |
        | --------------------- | -------- | ------------ | --------- | -------------- |
        |                       |          |              |           |                |

        *The Analyst Agent will calculate and populate these ratios based on financial statement data.*

        **Revision Instructions:**
        - When the Analyst Agent provides the completed “Key Financial Ratios and Investment Insights” table, replace the placeholder table above with the full table from the Analyst.
        - Do not change, recalculate, or omit any values or interpretations from the Analyst’s table.
        

        
        ## 2. Market Analysis

        ### Opening Remarks Summary:  
        Use a short blockquote to include one key quote:  
        > "…"

        Then summarize key management themes using this table:

        | Theme              | Key Message Summary                          |
        |--------------------|----------------------------------------------|
        | Strategy / Vision  |                                              |
        | Market Outlook     |                                              |
        | AI / Innovation    |                                              |

        ### Market Position & Trends:

        #### Competitive Landscape

        | Competitor   | Mentioned? | Strategic Position | Commentary            |
        |--------------|------------|--------------------|-----------------------|
        |              |            |                    |                       |

        #### Industry & Regulatory Trends

        | Trend               | Impact              | Impact Summary         |
        |---------------------|---------------------|------------------------|
        |                     |                     |                       |

        **Impact Legend:**
        - Positive: Beneficial effect on company performance
        - Negative: Adverse effect on company performance
        - Mixed: Both positive and negative effects
        - Neutral: No significant impact

        #### Growth Opportunities & M&A

        | Opportunity   | Description                       | Timing / Likelihood     |
        |---------------|-----------------------------------|-------------------------|
        |               |                                   |                         |

        #### Customer Segments

        | Segment Name  | Performance Summary               |
        |---------------|-----------------------------------|
        |               |                                   |

        ### Sentiment Analysis:

        **Leave this section blank.**  
        The Analyst Agent will fetch relevant news sentiment data and provide a Markdown-formatted summary and table. Once available, insert that content here during revision.

        ---

        ## 3. Risk Assessment

        Review the transcript and identify only those risks that are explicitly mentioned or can be reasonably inferred from management or analyst statements. For each risk you include:

        - Specify the risk category (e.g., Business, Financial, Market, Regulatory/Legal, Management, Forward-looking).
        - Provide a concise description of the risk.
        - Assign a Likelihood (Low, Medium, High) and Impact (1–5) based on transcript evidence.
        - For each risk, include supporting evidence from the transcript in the 'Supporting Evidence' column. This can be one or more quotes, phrases, sentences, or even a full paragraph, as needed to fully support the risk. Use blockquotes (`> ...`) for each distinct piece of evidence.

        **Important:**  
        - Only include risk categories and rows for which you find clear support in the transcript.  
        - Do NOT fabricate or force risks for categories that are not mentioned or implied in the transcript.  
        - Omit any risk category/row that is not supported by transcript evidence.

        Example table (populate only with supported risks):

        | Risk Category         | Description                        | Likelihood | Impact (1–5) | Supporting Evidence                |
        |----------------------|------------------------------------|------------|--------------|------------------------------------|
        | Business Risks       | Slowdown in refinance market       | Medium     | 3            | > "Potential slowdown in refinance markets in H2 2020."
        |                      |                                    |            |              | > "Management noted a decrease in application volume in late Q4."
        | Financial Risks      | Decline in investment income       | High       | 3            | > "Anticipated reduction in investment income affecting overall margins."
        | ...                  | ...                                | ...        | ...          | ...                                |

        ---

        **Impact Level Scale:**
        - **1** = Very Low (Minimal effect)
        - **2** = Low (Minor impact on performance)
        - **3** = Moderate (May affect segment or short-term earnings)
        - **4** = High (Significant impact on operations or financials)
        - **5** = Critical (Severe or existential threat)

        **Likelihood:**
        - **Low** = Unlikely under current conditions
        - **Medium** = Reasonably possible
        - **High** = Likely or already emerging

        **Include Impact Level Scale and Likelihood Legend in this section**

        ---

        ## 4. Investment Recommendation

        - **Key Investment Drivers**: Based on financial analysis and market position
        - **Major Risks**: From risk assessment above
        - **Recommendation**: 
          - **Next Day**: Buy/Sell/Hold
          - **Next Week**: Buy/Sell/Hold  
          - **Next Month**: Buy/Sell/Hold
        - **Price Target**: If mentioned in transcript
        - **Catalysts**: Upcoming events or milestones

        **Use tables, blockquotes for transcript quotes, and bullet points as appropriate. Focus on actionable insights and clear recommendations based on the earnings call analysis.**
        """
        )

        writer_agent = autogen.AssistantAgent(
            name="Writer",
            system_message=writer_prompt,
            llm_config=llm_config,
            max_consecutive_auto_reply=4
        )

        analyst_prompt = f"""
        You are the Analyst Agent. After the Writer drafts an initial investment report based on the transcript, you are responsible for fact-checking and correcting the financial metrics using actual financial data.

        Your responsibilities:

        1. Identify fiscal Period:
            Extract the fiscal year and quarter from the Prepared Remarks section of the transcript content itself. Look for phrases that indicate the fiscal year and quarter like:
                - "Fourth Quarter and Full Year 2022" → year="2022", quarter="Q4"
                - "Q4 2022 Earnings Call" → year="2022", quarter="Q4"
                - "First Quarter of 2022" → year="2022", quarter="Q1"
                
                Only if you cannot find the year or quarter in the transcript content, then use the provided year: {year} and quarter: {quarter}.
                IMPORTANT: Quarter format MUST be "Q1", "Q2", "Q3", or "Q4" (not just the number).

        2. Fetch financial articles for the same call date using: @historicalfinancialdata(ticker, year, quarter). Use the same year and quarter extracted from step 1.   

            After receiving the tool response:
                a) Compare the Writer's transcript-based values with the actual financial metrics:
            
                    **NOTE: All financial metrics are available in the tool response under response['financial_metrics'] except QoQ change (quarter to quarter change), where you need to calculate it.**
                    
                    If you find any discrepancies, provide this feedback:
                    ```
                    FINANCIAL METRICS CORRECTION REQUIRED:

                    Replace the following metrics with the correct values in the Financial Analysis table:

                    | Metric | Current Quarter | Previous Quarter | QoQ Change |
                    |--------|----------------|------------------|------------|
                    | [Metric Name] | [Correct Value] | [Correct Value] | [QoQ Change] |

                    Notes:
                    - Correct values are calculated from financial statements
                    ```

                    If any metrics were left blank by the Writer but are available in the financial data, provide:
                    ```
                    ADDITIONAL METRICS AVAILABLE:

                    Add these values to metrics with blank cells in the Financial Analysis table:

                    | Metric | Current Quarter | Previous Quarter | QoQ Change |
                    |--------|----------------|------------------|------------|
                    | [Metric] | [Value] | [Value] | [QoQ Change] |
                    ```

                b) Populate the “Key Financial Ratios and Investment Insights” table using the tool data. 
                    All required ratios are already provided by the historicalfinancialdata() tool under response['key_financial_ratios']—do not recalculate them.
                    
                    The table must use these headers:
                    | Metric                | Current Quarter | Previous Quarter | Formula   | Interpretation |
                    |-----------------------|----------------|------------------|-----------|----------------|
                    
                    For each metric, provide:
                    - The value for both quarters (rounded appropriately)
                    - The formula used (as a string)
                    - A brief interpretation of what the ratio means
                    
                    Insert the completed table in place of the placeholder in the Writer’s draft.

        2. Fetch market sentiment articles for the same call date using:
        @analyzemarketsentiment(ticker, date). Use the same year and quarter extracted from step 1.

        For the Sentiment Analysis section, ALWAYS provide a complete structure:

        a) If no news data is found:
           Instruct the Writer to include this structure:

           ### Sentiment Analysis

           **Management Tone & Guidance**: [Extract and analyze management's tone from the earnings call transcript]
           - Look for phrases indicating confidence, caution, or concern
           - Analyze forward-looking statements and guidance
           - Assess management's overall outlook (positive/neutral/negative)

           **Market Sentiment Indicators**: No relevant news articles were found for this earnings period.

           | Date | Source | Sentiment | Score | Headline |
           |------|--------|-----------|-------|----------|
           | N/A  | N/A    | N/A       | N/A   | N/A      |

           **Sentiment Score Legend:**
           - Score Range: -1.0 to 1.0
           - Interpretation:
             - -1.0 to -0.6: Very Bearish
             - -0.6 to -0.2: Bearish
             - -0.2 to 0.2: Neutral
             - 0.2 to 0.6: Bullish
             - 0.6 to 1.0: Very Bullish

        b) If news data IS found:
           Provide a complete sentiment analysis section with:
           - Management tone analysis from transcript
           - News sentiment table with date, source, sentiment, score, headline
           - Sentiment score legend
           - Overall sentiment summary

        **IMPORTANT:** Once feedback and insertion instructions are complete, hand off the conversation to the Editor Agent."""

        analyst_agent = autogen.AssistantAgent(
            name="Analyst",
            system_message=analyst_prompt,
            llm_config={"config_list": config_list},
        )

        # analyst_agent = autogen.AssistantAgent(
        #     name="Analyst",
        #     system_message=(
        #         "You are the Analyst Agent. After the Writer drafts or revises the report, you are responsible for providing detailed feedback on the draft.\n\n"
        #         "You have access to special tools for past quarter financial data and market sentiment analysis (news) for the past 7 days. Use these tools as needed to supplement your analysis. For example, to get financials: @get_previous_quarter_financials(ticker='AAPL', date='2023-12-31').\n"
        #         "Use these tools to fact-check, supplement, and critique the Writer's draft, and cite data or news directly in your feedback, based on the date of the earnings call. Use date format: YYYY-MM-DD when using the tools.\n"
        #         "\n**IMPORTANT: After you finish your feedback, hand off to the Editor agent for further review.**\n\n"
        #     ),
        #     llm_config={"config_list": config_list},
        # )
        #     # Register the tool signature with the assistant agent
        
        analyst_agent.register_for_llm(name="historicalfinancialdata", description="Get the previous quarter's financials for a given ticker and date.")(ResearchTools.get_previous_quarter_financials)
            # Register the tool function with the user proxy agent
        analyst_agent.register_for_execution(name="historicalfinancialdata")(ResearchTools.get_previous_quarter_financials)
        analyst_agent.register_for_llm(name="analyzemarketsentiment", description="Analyze market sentiment from alpha vantage api for a given ticker and date.")(ResearchTools.analyze_market_sentiment_alphavantage)
        analyst_agent.register_for_execution(name="analyzemarketsentiment")(ResearchTools.analyze_market_sentiment_alphavantage)
        
        
        editor_agent = autogen.AssistantAgent(
            name="Editor",
            system_message=(
                "You are the Editor Agent whose role is to review and provide detailed feedback on the Markdown-formatted investment report.\n\n"

                "Review the report and provide specific feedback in these areas:\n"
                "1. **Structure & Completeness**:\n"
                "- Are all four main sections present (Financial Analysis, Market Analysis, Risk Assessment, Investment Recommendation)?\n"
                "- Are Markdown headers properly formatted (e.g. `## Financial Analysis`)?\n"
                "- Are missing info rows properly omitted (not shown as blanks)?\n"
                "- Is the Analyst's sentiment feedback properly integrated in the Market/Sentiment Analysis section?\n"
                "- Are Impact level scale and Likelihood legend included in the Risk Assessment section?\n\n"

                "2. **Formatting**:\n"
                "- Are tables in valid Markdown format (header row, separator row, aligned columns)?\n"
                "- Are bullet lists consistently using `-` or `*`?\n"
                "- Are transcript quotes properly formatted with blockquotes (`> `)?\n\n"

                "3. **Recommendation Clarity**:\n"
                "- Is there a clear Buy/Sell/Hold call for each timeframe (next day, week, month)?\n"
                "- Are recommendations properly labeled (e.g. `- **Next Day:** Buy`)?\n\n"

                "4. **Tone & Readability**:\n"
                "- Is the language concise, professional, and persuasive?\n"
                "- Are there any grammar, punctuation, or style inconsistencies?\n\n"

                "Provide your feedback in a structured format:\n"
                "1. List specific issues that need to be addressed\n"
                "2. Give clear examples of how to fix each issue\n"
                "3. Highlight any sections that are particularly well done\n\n"

                "Do not rewrite the report yourself. Instead, provide clear, actionable feedback for the Writer agent to implement the necessary changes."
            ),
            llm_config=llm_config
        )


        groupchat = autogen.GroupChat(
            agents=[client_agent, writer_agent, analyst_agent, editor_agent],
            messages=[],
            max_round=15,
            select_speaker_auto_verbose=True,
            select_speaker_prompt_template=(
                "Read the above conversation. Follow these sequence rules STRICTLY:\n\n"
                "1. EXACT Required Sequence: Writer → Analyst → Writer → Editor → Writer → Client\n"
                "2. Current Rules:\n"
                "   - After Writer's initial draft: Select Analyst\n"
                "   - After Analyst's feedback: Select Writer for revision\n"
                "   - After Writer's revision: Select Editor\n"
                "   - After Editor's feedback: Select Writer for final revision\n"
                "   - After Writer's final revision: Select Client\n"
                "3. IMPORTANT: After Editor gives feedback, you MUST select Writer, not Client.\n\n"
                "Based on these rules and the conversation history, select the next role from {agentlist}. Only return the role name."
            )
        )


        manager = autogen.GroupChatManager(groupchat=groupchat, llm_config=llm_config)

        # Add this to check agent names
        print("Available agent names in group chat:")
        for agent in groupchat.agents:
            print(f"- {agent.name}")

        initial_request = (
            "Write a full investment report for the following earnings call transcript:\n\n"
            f"Fiscal Year: {year if year else 'Unknown'}\n\n"
            f"{transcript}\n\n"
            "Make sure the report includes all required sections as described in your instructions. "
            "Use tables, blockquotes, and bullet points as appropriate."
        )

        current_report = None
        round_num = 0

        while round_num < max_rounds:
            round_num += 1
            print(f"\n--- Round {round_num} ---")
            if current_report is None:
                client_agent.initiate_chat(manager, message=initial_request)
            else:
                client_agent.send(
                    message=f"Please revise the report based on my previous feedback:\n\n{current_report}",
                    recipient=manager
                )

            chat_history = groupchat.messages
            # print("chat_history: ",chat_history)
            latest_message = next(
                (m.get("content") for m in reversed(chat_history) if m.get("content")), None
            )

            # Find the latest non-empty Client message
            latest_client_message = next(
                (m.get("content") for m in reversed(chat_history)
                 if m.get("name") == "Client" and m.get("content") and m.get("content").strip()),
                None
            )

            if latest_client_message and "TERMINATE" in latest_client_message.upper():
                print("Client accepted the report. Ending session.")
                break
            elif latest_message:
                # Use the latest non-empty Client message as current_report if it exists, else fallback to latest_message
                current_report = latest_client_message if latest_client_message else latest_message
                print("Continuing to next round...")
            else:
                print("No valid message. Skipping to next round...")

        # After the loop, always return the last Writer message if available, else last message with content
        chat_history = groupchat.messages
        writer_messages = [m["content"] for m in chat_history if m.get("name") == "Writer" and m.get("content")]
        if writer_messages:
            final_report = writer_messages[-1]
        else:
            content_messages = [m["content"] for m in chat_history if m.get("content")]
            final_report = content_messages[-1] if content_messages else "No final report was generated."
        return final_report

    except Exception as e:
        return f"Error during analysis: {str(e)}"

# function to save the report to a file
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

if __name__ == "__main__":
    # Example usage (uncomment to test)
    # transcript = "Sample earnings call transcript..."
    # report = run_analysis(transcript, ticker="AAPL")
    # print(save_report("AAPL", report))
    pass




# client_agent = autogen.UserProxyAgent(
        #     name="Client",
        #     human_input_mode="NEVER",
        #     llm_config=llm_config,       
        #     max_consecutive_auto_reply=3,
        #     system_message=(
        #         "You are the Client (Investor). Review the report sent by the Writer. "
        #         "If the report meets ALL of the following criteria, reply with 'TERMINATE'. Otherwise, provide specific feedback for revision. "
        #         "Criteria: "
        #         "- The report includes all required sections: Financial Analysis, Market Analysis, Risk Assessment, Investment Recommendation. "
        #         "- Each section is clearly structured with Markdown headers. "
        #         "- Key financial metrics and trends are presented in tables where appropriate. "
        #         "- Bullet points are used for key findings. "
        #         "- The report is clear, actionable, and free of major omissions or errors. "
        #         "- Any missing information is must be ommitted and not be indicated as 'No information available'. "
        #         "- **The report addresses all feedback from Analyst. **"
        #         "If any of these criteria are not met, specify exactly what needs to be improved."
        #     )
        # )

        # writer_prompt = (
        #     """You are the Writer Agent. Your job is to draft and revise an investment report based on the Client's request and feedback from the Analyst and Editor agents.\n\n"
        #     "Your report MUST follow this structure and include all the following sections, using Markdown headers and tables where appropriate.\n"
        #     "Create a well-structured Markdown investment report that includes:\n"
        #     "\n"
        #     "1. Financial Analysis\n"
        #     "- Revenue and earnings (quarterly, growth rates, EPS)\n"
        #     "- Margins (gross, operating, net; changes and drivers)\n"
        #     "- Expenses (notable changes in COGS, SG&A, R&D, etc.)\n"
        #     "- Cash flow (operating, free cash flow, capex)\n"
        #     "- Balance sheet (debt, cash, liquidity ratios)\n"
        #     "- Guidance (forward-looking statements on revenue, profit, margins)\n"
        #     "- Key performance indicators (KPIs; e.g., user growth, ARPU, churn)\n"
        #     "- Historical comparison (YoY, QoQ trends)\n"
        #     "- Red flags (missed targets, losses, restatements, accounting changes)\n"
        #     "[Present key metrics in Markdown tables]\n\n"
        #     "2. Market Analysis\n"
        #     "- Market share and management's comments on competitive standing\n"
        #     "- Industry trends, regulatory changes, macroeconomic factors\n"
        #     "- Competitor mentions and disruptive technologies\n"
        #     "- Growth opportunities (new markets, products, partnerships, M&A)\n"
        #     "- Customer/segment performance (growing/lagging segments)\n"
        #     "- Sentiment (tone of management and analysts: optimism, caution, etc.)\n"
        #     "- Analyst Q&A highlights (concerns, excitement)\n"
        #     "- Media/news impact (recent news, PR, controversies)\n"
        #     "[Use bullet points for key findings]\n\n"
        #     "3. Risk Assessment\n"
        #     "- Business risks (supply chain, customer concentration, operations)\n"
        #     "- Financial risks (leverage, liquidity, credit, covenants)\n"
        #     "- Market risks (commodity, FX, interest rates, economic cycles)\n"
        #     "- Regulatory/legal risks (litigation, investigations, compliance)\n"
        #     "- Management risks (turnover, succession, credibility)\n"
        #     "- Forward-looking/emerging risks (geopolitical, inflation, ESG, tech disruption)\n"
        #     "- Mitigation strategies (management's plans to address/monitor risks)\n"
        #     "[Prioritize risks by impact]\n\n"
        #     "4. Investment Recommendation\n"
        #     "- Key investment drivers based on the financial analysis and market sentiment\n"
        #     "- Major risks to consider based on the risk assessment\n"
        #     "- Buy/Sell/Hold recommendation for the next day, week and month\n"
        #     "[Clear, actionable recommendation]\n\n"
        #     "Use tables, blockquotes for transcript quotes, and bullet points as appropriate. Focus on actionable insights and clear recommendations.\n\n"
        #     "If you lack information for any section, it is okay to state 'No information available' or leave it blank.\n"
        #     """
        # )