import os 
import streamlit as st 
from dotenv import load_dotenv 
import tempfile 
import time
from datetime import datetime
from crewAI import run_analysis, save_report
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from PyPDF2 import PdfReader
 
# Load environment variables 
load_dotenv() 
 
# Set page configuration 
st.set_page_config( 
    page_title="Financial Analysis Multi-Agent System", 
    page_icon="📊", 
    layout="wide" 
) 
 
# Initialize session state variables 
if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = None
if "analysis_history" not in st.session_state:
    st.session_state.analysis_history = []
if "is_analyzing" not in st.session_state:
    st.session_state.is_analyzing = False
 
# Sidebar for configuration
with st.sidebar: 
    st.title("Financial Analysis Configuration") 
    
    # Company selection
    st.subheader("Company Selection")
    company_ticker = st.text_input("Enter company ticker symbol (e.g., AAPL, MSFT)", "AAPL")
    
    # Model selection
    st.subheader("Model Selection")
    model_option = st.selectbox(
        "Select LLM Model",
        ["qwen3:8b","gemma3:12b","qwen2.5:7b","llama3.1"]
        # ["llama3.1", "meta-llama/llama-4-scout-17b-16e-instruct","llama-3.3-70b-versatile", "llama-3.1-70b-versatile", "llama-3.1-8b-versatile", "mixtral-8x7b-32768"]
    )

    
    # Advanced options
    with st.expander("Advanced Options"):
        temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.7, step=0.1)
        verbose = st.checkbox("Verbose Mode", value=True)
    
    # # SEC Filing Upload Section
    # st.subheader("SEC Filing Upload")
    # uploaded_file = st.file_uploader("Upload SEC Filing PDF", type="pdf")
    
    # if uploaded_file is not None:
    #     # Display file details
    #     file_details = {"Filename": uploaded_file.name, "FileType": uploaded_file.type, "FileSize": f"{uploaded_file.size / 1024:.2f} KB"}
    #     st.write(file_details)
        
    #     # Process and upload button
    #     if st.button("Process and Upload SEC Filing"):
    #         with st.spinner("Processing and uploading SEC filing..."):
    #             try:
    #                 # Import the PineconeStore class
    #                 from upload_sec_filing import PineconeStore
                    
    #                 # Create a temporary file to save the uploaded PDF
    #                 with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
    #                     tmp_file.write(uploaded_file.getvalue())
    #                     tmp_path = tmp_file.name
                    
    #                 # Extract text from PDF
    #                 pdf_reader = PdfReader(tmp_path)
    #                 text = ""
    #                 for page in pdf_reader.pages:
    #                     text += page.extract_text()
                    
    #                 # Split text into chunks
    #                 text_splitter = RecursiveCharacterTextSplitter(
    #                     chunk_size=1000,
    #                     chunk_overlap=200,
    #                     length_function=len
    #                 )
    #                 chunks = text_splitter.split_text(text)
                    
    #                 # Create documents
    #                 documents = [
    #                     Document(
    #                         page_content=chunk,
    #                         metadata={
    #                             "source": uploaded_file.name,
    #                             "company": company_ticker,
    #                             "file_type": "SEC Filing"
    #                         }
    #                     ) for chunk in chunks
    #                 ]
                    
    #                 # Initialize Pinecone store and upload documents
    #                 pinecone_store = PineconeStore(
    #                     index_name="company-sec-index",
    #                     namespace=company_ticker.lower()
    #                 )
                    
    #                 # Add documents to vector store
    #                 pinecone_store.add_documents(documents)
                    
    #                 # Clean up temporary file
    #                 os.unlink(tmp_path)
                    
    #                 st.success(f"Successfully processed and uploaded {len(documents)} chunks from {uploaded_file.name}")
                    
    #             except Exception as e:
    #                 st.error(f"Error processing SEC filing: {str(e)}")
    
    # Run analysis button
    if st.button("Run Analysis"):
        if not company_ticker:
            st.error("Please enter a company ticker symbol")
        else:
            st.session_state.is_analyzing = True
            with st.spinner(f"Analyzing {company_ticker}... This may take several minutes."):
                try:
                    result = run_analysis(company_ticker, model_option, temperature, verbose)
                    st.session_state.analysis_result = result
                    
                    # Add to history
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
This application uses a multi-agent system to perform comprehensive financial analysis on publicly traded companies.
The system employs multiple AI agents working together:

- **Research Analyst**: Analyzes SEC filings and financial data
- **Market Sentiment Analyst**: Examines market sentiment to gauge public opinion
- **Visionary**: Explores future implications and strategic considerations
- **Senior Editor**: Compiles findings into a professional report
""")

# Display analysis result
if st.session_state.is_analyzing:
    st.info("Analysis in progress... Please wait.")
    
    # Add a placeholder for progress updates
    progress_placeholder = st.empty()
    progress_bar = st.progress(0)
    
    # Simulate progress updates
    for i in range(100):
        # Update progress
        progress_bar.progress(i + 1)
        if i == 20:
            progress_placeholder.info("Research Analyst gathering financial data...")
        elif i == 40:
            progress_placeholder.info("Market Sentiment Analyst analyzing market sentiment...")
        elif i == 60:
            progress_placeholder.info("Visionary exploring future implications...")
        elif i == 80:
            progress_placeholder.info("Senior Editor compiling final report...")
        time.sleep(0.1)

elif st.session_state.analysis_result:
    st.subheader("Analysis Result")
    
    # Display the markdown content
    st.markdown(st.session_state.analysis_result)
    
    # Save report button
    if st.button("Save Report"):
        try:
            filepath = save_report(company_ticker, st.session_state.analysis_result)
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