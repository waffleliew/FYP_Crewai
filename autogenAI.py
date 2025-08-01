import streamlit as st
import os
import sys
import time
import json
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
import autogen
from research_tools import ResearchTools
# Set AutoGen Docker configuration
os.environ["AUTOGEN_USE_DOCKER"] = "0"
load_dotenv()

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

def run_analysis(transcript, ticker=None, model="llama-3.3-70b-versatile", temp=0.3, verbose_mode=True, max_rounds=2, year=None, quarter=None):
    """
    Run analysis on an earnings call transcript using an AutoGen multi-agent feedback loop.
    Returns:
        str: The generated analyst report.
    """
    try:
        print(f"Running analysis using model {model}, for ticker: {ticker}, transcript year: {year}, quarter: {quarter}")

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
                "You are the Client (Investor). Review the latest investment report generated by the Writer Agent. "
                "Your goal is to validate whether the LATEST report generated by the Writer Agent is complete, correct, and integrates all feedback from the Analyst Agent. "
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
                "\n      - Quarter-over-Quarter (QoQ) change percentage (with + or - sign)"
                "\n      - Year-over-year (YoY) change percentage (with + or - sign)"
                "\n   c) Are missing values marked as 'N/A' instead of being left blank?"
                "\n   d) Are all percentages formatted consistently?"

                "\n3. Are the financial ratios properly populated and formatted?"
                "\n   a) Is the ratios table complete with all ten required metrics:"
                "\n      - Gross Margin (%)"
                "\n      - Operating Margin (%)"
                "\n      - Net Margin (%)"
                "\n      - EPS Surprise (%)"
                "\n      - Free Cash Flow"
                "\n      - Capex / OCF (%)"
                "\n      - Cash Conversion Ratio"
                "\n      - Net Debt"
                "\n      - Current Ratio"
                "\n      - Debt-to-Equity"
                "\n   b) Does each ratio have:"
                "\n      - Current Quarter value"
                "\n      - Previous Quarter value"
                "\n      - Previous Year value"
                "\n      - Formula"
                "\n      - Interpretation"
                "\n   c) Are missing values marked as 'N/A' instead of being left blank?"
                "\n   d) Are all percentages and ratios formatted consistently?"

                "\n4. Is the Key Highlights subsection in financial analysis section populated with actual bullet points from the analysis (not empty or placeholder bullets)?"

                "\n5. Is the Concluding Summary subsection in financial analysis section populated with a meaningful summary (not a placeholder message)?"

                "\n6. Does the Risk Assessment section include properly populated risks with supporting evidence from the transcript, including Likelihood and Impact values?"

                "\n7. Refer to the feedback from the Analyst Agent and ensure **all Analyst feedback points clearly addressed** in the final report, especially the INVESTMENT RECOMMENDATION FEEDBACK"

                "\n8. Are any placeholder phrases like **'No information available'** omitted from the report?"

                "\n9. Are **tables used consistently** for structured data, and bullet points used for key findings?"


                "\n\n### Final Instruction:"
                "\n- If **ALL** answers are YES, reply only with: TERMINATE"
                "\n- If **ANY** answer is NO, return a numbered list of the failed checks and specify what exactly needs to be revised or added."
            )
        )


        writer_prompt = f"""
        ### INSTRUCTIONS START ###

        You are the Writer Agent. Your task is to draft and revise a structured investment report using Markdown formatting, based on the earnings call transcript and feedback from other agents.

        **Key Rules:**
        - Your output MUST ONLY be the report content. Do NOT include any of these instructions in the report.
        - Determine the full company name from the transcript and replace [Company Name] in the title with it.
        - **For initial draft: Leave placeholders intact only for the following subsections: Key Financial Ratios and Investment Insights, News Sentiment and Investment Recommendations. For all other sections, populate based on your analysis of the TRANSCRIPT.**
        - Use Markdown tables for structured data. Omit subsections if data missing.

        **Revision Guidelines:**
        - Strictly follow the revision guidelines: only update based on specific feedback or UPDATE messages from Analyst or Editor.
        - Carefully adhere to the specific instructions for each section below without deviation.
        - Always output only the full revised report without any extra text or commentary.
        - If prompted again without new instructions, respond only with the report, no conversational replies.
        - Always output only the full revised report without extra text.
        - If you are prompted again, without any new instructions, only reply with the report. Do not generate any conversational reply.


        ### INSTRUCTIONS END ###

        ### REPORT TEMPLATE START ###

        ---
        
        # [Company Name] ({ticker}) Investment Report - Fiscal {year} {quarter}
        **Instructions:** (Do not include this in output)
        - Replace [Company Name] with the full name of the company from the transcript.
        
        ---

        ## 1. Financial Analysis

        ### Key Highlights
        - 
        - 
        -

        *The Analyst Agent will generate and provide key highlights based on financial metrics and ratios from financial statement.*
        **Revision Instructions:** (Do not include this in output)
        - Replace with content from "KEY HIGHLIGHTS UPDATE" without changes.

        ### Key Financial Metrics

        | Metric                    | Current Quarter | Previous Quarter | QoQ Change | Previous Year | YoY Change |
        |---------------------------|-----------------|------------------|------------|---------------|------------|
        | Revenue                   |                 |                  |            |               |            |
        | Earnings per Share (EPS)  |                 |                  |            |               |            |
        | Gross Profit              |                 |                  |            |               |            |
        | Operating Income          |                 |                  |            |               |            |
        | Net Income                |                 |                  |            |               |            |
        | Operating Cash Flow       |                 |                  |            |               |            |
        | Capex                     |                 |                  |            |               |            |
        | Short-term Debt           |                 |                  |            |               |            |
        | Long-term Debt            |                 |                  |            |               |            |
        | Cash & Cash Equivalents   |                 |                  |            |               |            |

        **Important:** (Do not include this in output)
        - Populate only from transcript data.
        - Leave blank if unavailable.
        - Analyst will provide corrections.

        ### Key Financial Ratios and Investment Insights

        | Metric                | Current Quarter | Previous Quarter | Previous Year | Formula   | Interpretation |
        | --------------------- | --------------- | ---------------- | ------------- | --------- | -------------- |
        | Gross Margin (%)      |                 |                  |               |           |                |
        | Operating Margin (%)  |                 |                  |               |           |                |
        | Net Margin (%)        |                 |                  |               |           |                |
        | EPS Surprise (%)      |                 |                  |               |           |                |
        | Free Cash Flow        |                 |                  |               |           |                |
        | Capex / OCF (%)       |                 |                  |               |           |                |
        | Cash Conversion Ratio |                 |                  |               |           |                |
        | Net Debt              |                 |                  |               |           |                |
        | Current Ratio         |                 |                  |               |           |                |
        | Debt-to-Equity        |                 |                  |               |           |                |

        *The Analyst Agent will calculate and populate these ratios based on financial statement data.*
        **Revision Instructions:** (Do not include this in output)
        - When the Analyst provides the completed table, replace this placeholder with their full table.
        - Do not alter any values or interpretations.
        - If analyst did not provide formula, include formula, for these ratios.


        ### Concluding Summary

        *The Analyst Agent will provide a concluding summary based on financial metrics and ratios.*
        **Revision Instructions:** (Do not include this in output)
        - Replace with content from "FINANCIAL ANALYSIS SUMMARY UPDATE" without changes.

        ---

        ## 2. Market Analysis

        ### Opening Remarks Summary:
        > "[opening remarks summary here]"
        **Instructions:** (Do not include in output)
        - Summarize the key opening remarks from the transcript in a concise blockquote.
        - Do not leave as ellipsis; provide a meaningful summary based on the transcript content.


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

        **Instructions:** (Do not include in output)
        - Include the Impact Level Scale in your report.

        #### Growth Opportunities & M&A

        | Opportunity   | Description                       | Timing / Likelihood     |
        |---------------|-----------------------------------|-------------------------|
        |               |                                   |                         |

        #### Customer Segments

        | Segment Name  | Performance Summary               |
        |---------------|-----------------------------------|
        |               |                                   |

        ### News Sentiment:

        **Leave this section blank.**
        The Analyst will provide content. (Do not include this in output)

        ---

        ## 3. Risk Assessment

        **Instructions for this section:** (Do not include in output)
        - Identify risks only from transcript.
        - Use table format.
        - Include the legend for the Impact Level Scale and Likelihood in your report.

        | Risk Category         | Description                        | Likelihood | Impact (1–5) | Supporting Evidence                |
        |----------------------|------------------------------------|------------|--------------|------------------------------------|

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

        ---

        ## 4. Investment Recommendation

        *The Editor Agent will analyze sections 1-3 and provide a comprehensive investment recommendation with specific drivers, risks, and timeframe-based recommendations.*

        **Placeholder Structure:**
        - **Key Investment Drivers**: [To be populated based on Editor feedback]
        - **Major Risks**: [To be populated based on Editor feedback] 
        - **Recommendation**:
          - **Next Day**: [To be populated based on Editor feedback]
          - **Next Week**: [To be populated based on Editor feedback]
          - **Next Month**: [To be populated based on Editor feedback]
        - **Catalysts**: [To be populated based on Editor feedback]

        **Revision Instructions:** (Do not include this in output)
        - When you receive an "INVESTMENT RECOMMENDATION FEEDBACK" from the Editor Agent, you MUST carefully review and incorporate the specific feedback into your Investment Recommendation section (replace [placeholder] with the specific feedback).
        - This is MANDATORY for every revision after receiving Editor feedback.
        - Maintain the overall structure of your original investment recommendation while making the specific changes suggested by the Editor.
        - Pay special attention to incorporating financial health indicators with QoQ/YoY references as suggested by the Editor.


        **General:** Use tables, blockquotes, bullets appropriately. (Do not include in output)

        ### REPORT TEMPLATE END ###"""
    

        writer_agent = autogen.AssistantAgent(
            name="Writer",
            system_message=(
                writer_prompt +
                "\n\nIMPORTANT: Your role is strictly limited to drafting and revising the investment report based on the transcript and feedback from other agents. " +
                "Do not engage in general conversation or provide responses outside of report drafting and revision tasks. " +
                "When receiving feedback, your only action should be to incorporate the changes into the report structure provided above."
            ),
            llm_config=llm_config,
            max_consecutive_auto_reply=4
        )

        analyst_prompt = f"""
        You are the Analyst Agent. After the Writer drafts an initial investment report based on the transcript, you are responsible for fact-checking and correcting the financial metrics using actual financial data.

        Your responsibilities:

        1. Fetch financial data using @historicalfinancialdata({ticker}, {year}, {quarter}) and perform the following actions:

            a) Fact-Check and Correct Financial Metrics:
                - Compare the financial metrics in the Writer's draft against the data from the tool.
                - Calculate the Quarter-on-Quarter (QoQ) and Year-on-Year (YoY) changes for each metric where possible.
                - Consolidate all corrections and additions into a single, comprehensive table.

                Provide your feedback to the Writer in the following format:
                ```
                FINANCIAL ANALYSIS UPDATE:

                Please update the "Key Financial Metrics" table with the following verified data. This includes corrections to existing values and the addition of metrics that were previously missing.

                | Metric | Current Quarter | Previous Quarter | QoQ Change | Previous Year | YoY Change |
                |--------|----------------|------------------|------------|---------------|------------|
                | [Metric Name] | [Correct Value] | [Correct Value] | [Calculated QoQ] | [Previous Year Value] | [Calculated YoY] |
                | ...    | ...            | ...              | ...        | ...           | ...        |

                **Summary of Key Changes:**
                - [Provide a brief, bulleted list of the most significant corrections or additions. For example: "- Revenue was overstated by 5% in the initial draft." or "- Added Operating Cash Flow, which was missing."]

                **Note:** All values are sourced directly from the financial statements provided by the tool.
                ```

            b) Verify and Enrich the "Key Financial Ratios and Investment Insights" Table:
                - Use the `key_financial_ratios` from the tool's response to verify the data in the Writer's draft.
                - Calculate the Year-on-Year (YoY) change for each ratio.
                - Provide a concise interpretation for each ratio, explaining its significance in the context of the company's performance.
                - Include the formula for each ratio.

                Provide your feedback to the Writer in the following format:
                ```
                FINANCIAL RATIOS UPDATE:

                Please update the "Key Financial Ratios and Investment Insights" table with the following verified data and insights.

                | Metric | Current Quarter | Previous Quarter | Previous Year | Formula | Interpretation |
                |--------|-----------------|------------------|---------------|---------|----------------|
                | [Ratio Name] | [Correct Value] | [Correct Value] | [Previous Year Value] | [Formula] | [Concise Interpretation] |
                | ... | ... | ... | ... | ... | ... |

                **Summary of Key Insights:**
                - [Provide a brief, bulleted list of the most important insights derived from the ratios. For example: "- Profitability has improved significantly YoY, driven by higher margins." or "- The company's liquidity position has weakened, requiring closer monitoring."]
                ```

            c) Generate a key highlights section based on the key financial metrics, financial ratios and forecast tables.

                Provide your key highlights to the Writer in the following format:
                ```
                KEY HIGHLIGHTS UPDATE:

                Please replace the Key Highlights placeholder with the following content:

                ### Key Highlights

                - [Metric Highlight: e.g., Revenue grew +X% YoY and +Y% QoQ, reflecting Z.]
                - [Ratio Highlight: e.g., Gross Margin rose to X%, indicating Y.]
                - [Forecast Highlight: e.g., Projected EPS of $X for next quarter, driven by Z.]
                - ...
                ```
                Ensure each bullet provides a brief description explaining key aspects from the metrics, ratios, and forecast tables.

            d) Generate Concluding Summary for Financial Analysis:
                - Based on the key financial metrics and ratios, provide a brief concluding summary that highlights the company's financial health, trends, and implications.
            

                Provide your feedback to the Writer in the following format:
                ```
                FINANCIAL ANALYSIS SUMMARY UPDATE:

                Please add the following as the concluding summary at the end of the Financial Analysis section:

                ### Concluding Summary

                [Your summary text here, 3-5 sentences.]
                ```

        2. Fetch market sentiment articles for the same call date using:
        @analyzemarketsentiment(ticker, year, quarter). Use the same year and quarter extracted from step 1.

        For the Sentiment Analysis section, ALWAYS provide a complete structure:

        a) If no news data is found:
           **Instruct the Writer to omit the entire News Sentiment section.**

        b) If news data IS found:
           Provide a complete News Sentiment section with:
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
        
        analyst_agent.register_for_llm(name="historicalfinancialdata", description="Get the previous quarter's financials for a given ticker and date.")(ResearchTools.historicalfinancialdata)
            # Register the tool function with the user proxy agent
        analyst_agent.register_for_execution(name="historicalfinancialdata")(ResearchTools.historicalfinancialdata)
        analyst_agent.register_for_llm(name="analyzemarketsentiment", description="Analyze market sentiment from alpha vantage api for a given ticker and date.")(ResearchTools.analyzemarketsentiment)
        analyst_agent.register_for_execution(name="analyzemarketsentiment")(ResearchTools.analyzemarketsentiment)
        
        
        editor_agent = autogen.AssistantAgent(
            name="Editor",
            system_message=(
                "You are the Editor Agent whose role is to review and provide detailed feedback on the Markdown-formatted investment report.\n\n"

                "Review the report and provide specific feedback in these areas:\n"
                "1. Generate Investment Recommendation:\n"
                    "- Based on the information provided in sections 1-3 (Financial Analysis, Market Analysis, Risk Assessment), generate a comprehensive investment recommendation.\n"
                    "- You MUST provide an INVESTMENT RECOMMENDATION message for the Writer to populate section 4, using the following format:\n"
                    "```\n"
                    "INVESTMENT RECOMMENDATION FEEDBACK:\n\n"
                    "Based on the analysis in sections 1-3, please update section 4 with the following recommendation:\n\n"
                    "- **Key Investment Drivers**: [List 3-4 specific drivers from Financial Analysis section, including QoQ/YoY trends, strategic initiatives from Market Analysis]\n"
                    "- **Major Risks**: [List key risks identified in Risk Assessment section with their impact levels]\n"
                    "- **Recommendation**:\n"
                        "- **Next Day**: [Provide Buy/Hold/Sell recommendation based primarily on financial metrics and immediate risks]\n"
                        "- **Next Week**: [Provide Buy/Hold/Sell recommendation based on market analysis and emerging trends]\n"
                        "- **Next Month**: [Provide Buy/Hold/Sell recommendation based on long-term strategy and fundamental trends]\n"
                    "- **Catalysts**: [List specific upcoming events or milestones that could impact stock performance]\n"
                    
                    "Each recommendation must reference specific data points from the analysis sections to justify the investment call.\n"
                    "```"
                    "**- The Writer Agent MUST use this recommendation to populate the placeholder in section 4 of the report.**"
                    
                "2. **Structure & Completeness**:\n"
                "- Are all four main sections present (Financial Analysis, Market Analysis, Risk Assessment, Investment Recommendation)?\n"
                "- Are Markdown headers properly formatted (e.g. `## Financial Analysis`)?\n"
                "- Are missing info rows properly omitted (not shown as blanks)?\n"
                "- Is the Analyst's sentiment feedback properly integrated in the Market/Sentiment Analysis section?\n"
                "- Are Impact level scale and Likelihood legend included in the Risk Assessment section?\n"
                "- Ensure that all subsections like 'Concluding Summary', 'Key Highlights' are populated with meaningful content and not left empty or as placeholders.\n\n"
                "- Ensure all cells in the 'Key Financial Metrics' and 'Key Financial Ratios and Investment Insights' tables are populated and not left empty. For example, formula.\n"

                "3. **Formatting**:\n"
                "- Are the tables in 'Key Financial Metrics' subsections properly formatted in terms of columns and rows. there should be 6 columns (Metric Name, Current Quarter, Previous Quarter, QoQ Change, Previous Year, YoY Change) in total.\n"
                "- Are the tables in 'Key Financial Ratios and Investment Insights' subsections properly formatted in terms of columns and rows. there should be 6 columns (Ratio Name, Current Quarter, Previous Quarter, Previous Year, Formula, Interpretation) in total.\n"
                "- Are tables in valid Markdown format (header row, separator row, aligned columns)?\n"
                "- Are bullet lists consistently using `-` or `*`?\n"
                "- Are transcript quotes properly formatted with blockquotes (`> `)?\n\n"


                "4. **Tone & Readability**:\n"
                "- Is the language concise, professional, and persuasive?\n"
                "- Are there any grammar, punctuation, or style inconsistencies?\n\n"

                "5. **Content Purity**:\n"
                "- Ensure no internal prompt instructions, placeholders, or notes (e.g., '*The Analyst Agent will generate...*') are included in the report.\n"
                "- If found, instruct the Writer to remove them to keep the report clean and professional.\n\n"

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
                "3. IMPORTANT: 1. DO NOT REPEAT AGENT SELECTION CONSECUTIVELY. 2. After Editor gives feedback, you MUST select Writer, not Client and not Analyst.\n\n"
            
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
def save_report(ticker, year, quarter, content):
    """
    Save the analysis report to a file.
    Args:
        ticker (str): The company ticker symbol
        content (str): The report content
    Returns:
        str: The path to the saved file
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{ticker}_{year}_{quarter}.md"
    filepath = Path('Earnings2Insights_Result/Generated_Reports') / filename
    
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