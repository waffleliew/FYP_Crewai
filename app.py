import os 
import streamlit as st 
from dotenv import load_dotenv 
import tempfile 
import time
from datetime import datetime
from typing import Annotated
import re
# Set AutoGen Docker configuration
os.environ["AUTOGEN_USE_DOCKER"] = "0"

from autogenAI import run_analysis, save_report
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from PyPDF2 import PdfReader
 
# Load environment variables 
load_dotenv() 
 
 
# Set page configuration 
st.set_page_config( 
    page_title="Financial Analysis Multi-Agent System", 
    layout="wide" 
) 
 
# Initialize session state variables 
if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = None
if "analysis_history" not in st.session_state:
    st.session_state.analysis_history = []
if "is_analyzing" not in st.session_state:
    st.session_state.is_analyzing = False


def extract_year_from_filename(filename):
    match = re.search(r'(20\d{2})', filename)
    if match:
        return match.group(1)
    return None

def extract_quarter_from_filename(filename):
    match = re.search(r'q([1-4])', filename.lower())
    if match:
        return f"Q{match.group(1)}"
    return None
 
# Sidebar for configuration
with st.sidebar: 
    st.title("Financial Analysis Configuration") 
    
    # Company selection
    st.subheader("Company Selection")
    company_ticker = st.text_input("Enter company ticker symbol (e.g., AAPL, MSFT)", "FAF")
    
    # Model selection
    st.subheader("Model Selection")
    model_option = st.selectbox(
        "Select LLM Model",
        [
            "gpt-4o",                   # OpenAI GPT-4 Omni (Default)
            "gpt-4o-mini",              # OpenAI GPT-4 Omni Mini
            "gpt-4-turbo",              # OpenAI GPT-4 Turbo
            "gpt-3.5-turbo",             # OpenAI GPT-3.5 Turbo
            "llama-3.3-70b-versatile",  # Groq Llama-3.3 70B Versatile
        ]
    )

    # Transcript upload
    st.subheader("Upload Earnings Call Transcript")
    uploaded_transcript = st.file_uploader("Upload transcript (.txt or .md)", type=["txt", "md"]) 

    # Advanced options
    with st.expander("Advanced Options"):
        temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.3, step=0.1)
        verbose = st.checkbox("Verbose Mode", value=True)
    
    # Run analysis button
    if st.button("Run Analysis"):
        transcript_text = None
        year = None
        if uploaded_transcript is not None:
            transcript_text = uploaded_transcript.read().decode("utf-8")
            year = extract_year_from_filename(uploaded_transcript.name)
            quarter = extract_quarter_from_filename(uploaded_transcript.name)
        
        if transcript_text:
            st.session_state.is_analyzing = True
            with st.spinner("Analyzing uploaded transcript... This may take several minutes."):
                try:
                    result = run_analysis(transcript_text, ticker=company_ticker, model=model_option, temp=temperature, verbose_mode=verbose, year=year, quarter=quarter)
                    st.session_state.analysis_result = result
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
                    st.session_state.analysis_history.append({
                        "ticker": company_ticker,
                        "timestamp": timestamp,
                        "result": result
                    })
                    st.success("Analysis of uploaded transcript completed!")
                except Exception as e:
                    st.error(f"Error during analysis: {str(e)}")
                finally:
                    st.session_state.is_analyzing = False
        elif not company_ticker:
            st.error("Please enter a company ticker symbol or upload a transcript.")
        else:
            st.session_state.is_analyzing = True
            with st.spinner(f"Analyzing {company_ticker}... This may take several minutes."):
                try:
                    result = run_analysis(company_ticker, model=model_option, temp=temperature, verbose_mode=verbose)
                    st.session_state.analysis_result = result
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
                    st.session_state.analysis_history.append({
                        "ticker": company_ticker,
                        "timestamp": timestamp,
                        "result": result
                    })
                    st.success(f"Analysis of {company_ticker} completed!")
                except Exception as e:
                    st.error(f"Error during analysis: {str(e)}")
                finally:
                    st.session_state.is_analyzing = False

# Main content area
st.title("Financial Analysis Multi-Agent System")

st.markdown("""
## About This App

**Earnings2Insights: Analyst Report Generation for Investment Guidance**

This app is built for the Earnings2Insights shared task at FinNLP @ EMNLP-2025. The competition challenges participants to automatically generate actionable investment reports from earnings call transcripts, evaluated by human annotators on their ability to guide Long/Short investment decisions for the next day, week, and month.

**Competition Requirements:**
- Generate a report for each of 64 earnings call transcripts (ECTSum and Professional subsets).
- Reports must be persuasive, actionable, and follow a detailed structure.
- Submission is a single JSON file: each entry has the transcript ID and the generated report.
- Evaluation is based on human annotators' investment decisions using your reports.

**Our Solution:**
- Uses a multi-agent LLM system (AutoGen) with specialized agents (Investor, Writer, Analyst, Editor) collaborating in a feedback loop.
- Produces structured, transparent reports with explicit Long/Short recommendations for multiple time frames.

For more details, see the [Task Description](https://sigfintech.github.io/fineval.html).
""")

# Display analysis result
if st.session_state.is_analyzing:
    st.info("Analysis in progress... Please wait.")
    

elif st.session_state.analysis_result:
    st.subheader("Analysis Result")
    
    # Display the markdown content
    st.markdown(st.session_state.analysis_result)
    
    # Save report button
    if st.button("Save Report"):
        try:
            year = extract_year_from_filename(uploaded_transcript.name)
            quarter = extract_quarter_from_filename(uploaded_transcript.name)
            filepath = save_report(company_ticker, year, quarter, st.session_state.analysis_result)
            st.success(f"Report saved to {filepath}")
        except Exception as e:
            st.error(f"Error saving report: {str(e)}")

# History section
if st.session_state.analysis_history:
    st.subheader("Analysis History")
    
    for i, analysis in enumerate(reversed(st.session_state.analysis_history)):
        with st.expander(f"{analysis['ticker']} - {analysis['timestamp']}"):
            st.markdown(analysis['result'])
else:
    st.info("No analysis history yet. Run an analysis to get started!")


