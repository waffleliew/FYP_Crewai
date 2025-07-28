# AI Automation Finance

Welcome! This project features a multi-agent system for automated Financial Analysis built for the Earnings2Insights shared task at FinNLP @ EMNLP-2025. The competition challenges participants to automatically generate actionable investment reports from earnings call transcripts, evaluated by human annotators on their ability to guide Long/Short investment decisions for the next day, week, and month.

## Project Overview

This project leverages large language models (LLMs) and multi-agent collaboration framework from AutoGen to automate the analysis of earnings call transcripts and financial data from external tools. The system generates structured investment reports with actionable recommendations, integrating feedback from specialized agents (Writer, Analyst, Editor, Client). 

### Key Features
- **Multi-Agent Investment Report Generation:** Automated drafting, analysis, and revision of investment reports using LLM-powered agents.
- **Financial Data Integration:** Fact-checks and updates financial metrics and ratios using real historical data.
- **Market Sentiment Analysis:** Integrates news sentiment and management tone analysis.
- **Actionable Recommendations:** Provides clear Buy/Hold/Sell signals with rationale based on financial, market, and risk analysis.
- **Markdown Report Output:** Generates well-structured, human-readable Markdown reports.

## Directory Structures
```
/ (root)
├── autogenAI.py                # Main multi-agent orchestration script
├── app.py                      # Streamlit app for running the platform
├── requirements.txt            
├── README.md                   
├── Documentation/              # Project documentation and diagram
├── Earnings2Insights/          # Main dataset and processing directory
│   ├── Dataset/              
│   │   ├── ECTsum/             # ECTsum dataset
│   │   └── Professional/        # Professional dataset
│   ├── Convert_MD_Report_To_Json_Tools/  
│   │   ├── Convert_all_MD_Reports.py    # Batch conversion script for converting markdown reports to JSON format
│   │   └── print_report_based_on_ECC.py # Report viewer by ECC transcript ID
│   ├── Generated_Reports/      
│   └── Earnings2Insights_Result_final.json  # Processed results for submission
│            
├── research_tools.py       # Financial data and analysis tools
└── test_research_tools.py  # Test script for testing tools

```

## File Descriptions

### Core Files
- **autogenAI.py:** Orchestrates the multi-agent system for report generation and analysis
- **app.py:** Streamlit web interface for running the platform
- **requirements.txt:** Lists all Python package dependencies
- **research_tools.py:** Contains utilities for financial data processing and sentiment analysis

### Earnings2Insights Directory
- **Dataset/:** Contains two subdirectories for different types of financial data:
  - **ECTsum/:** Contains 40 earnings call transcript summaries
  - **Professional/:** Contains 24 professional financial reports
- **Convert_MD_Report_To_Json_Tools/:**
  - **Convert_all_MD_Reports.py:** Converts markdown reports to JSON format
  - **print_report_based_on_ECC.py:** Utility to view specific reports by their ECC transcript ID
- **Generated_Reports/:** Stores the generated markdown reports
- **Earnings2Insights_Result_final.json:** Contains the processed and structured data from all reports

### Documentation
- **Documentation/:** Contains project documentation, diagrams, and sample feedback templates

## Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/FYP.git
   cd FYP
   ```

2. Set up your environment variables in a `.env` file:
   ```
   OPENAI_API_KEY=your_openai_api_key_here
   ALPHA_VANTAGE_API_KEY=your_alpha_vantage_api_key_here  (get your alphavantage api key [here](https://www.alphavantage.co/support/#api-key))
   ```

3. Set up python virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # To activate virtual env on windows, use `venv\Scripts\activate`.
   ```

4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

5. Run the Streamlit app and follow the instructions in the APP:
   ```bash
   streamlit run app.py
   ```



## Earnings2Insight Submission

### Viewing Reports
To view a specific report by its ECC transcrip ID:
1. Run: 
```bash
python Earnings2Insights/Convert_MD_Report_To_Json_Tools/print_report_based_on_ECC.py
```
2. Enter the ECC transcript ID when prompted (e.g., "ABM_q3_2021"). 
3. The report will be displayed in the console. Copy and Paste the MD report into https://www.markdowntopdf.com to convert it to PDF for better viewing (Optional).

### Converting Reports
To convert all markdown reports to JSON format for Earnings2Insight Submission:
1. Run:
```bash
python Earnings2Insights/Convert_MD_Report_To_Json_Tools/Convert_all_MD_Reports.py
```


