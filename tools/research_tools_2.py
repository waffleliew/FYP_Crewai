from langchain.tools import tool
import requests
import os
import json
from datetime import datetime
import yfinance as yf
import pandas as pd
from bs4 import BeautifulSoup
import re
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore, PineconeEmbeddings
from crewAI import LLM
from crewai_tools import FirecrawlCrawlWebsiteTool, SerperDevTool
from dotenv import load_dotenv
from langchain.tools import tool
from firecrawl import FirecrawlApp
from gnews import GNews
from nltk.sentiment import SentimentIntensityAnalyzer
load_dotenv()

class ResearchTools:
    
    @tool("Search for company financial data")
    def search_financial_data(company_ticker):
        """
        Fetches financial data about a company using its ticker symbol via Yahoo Finance.
        The input to this tool should be a company ticker symbol (e.g., NVDA, AMD).
        """
        try:
            # Get data from Yahoo Finance
            ticker = yf.Ticker(company_ticker)
            info = ticker.info
            
            # Extract key financial metrics
            financial_data = {
                "name": info.get("shortName", "N/A"),
                "ticker": company_ticker.upper(),
                "sector": info.get("sector", "N/A"),
                "industry": info.get("industry", "N/A"),
                "market_cap": f"${info.get('marketCap', 0) / 1000000000:.2f}B",
                "pe_ratio": f"{info.get('trailingPE', 0):.2f}",
                "revenue_ttm": f"${info.get('totalRevenue', 0) / 1000000000:.2f}B",
                "gross_margin": f"{info.get('grossMargins', 0) * 100:.1f}%",
                "profit_margin": f"{info.get('profitMargins', 0) * 100:.1f}%",
                "return_on_equity": f"{info.get('returnOnEquity', 0) * 100:.1f}%",
                "beta": f"{info.get('beta', 0):.2f}",
                "52_week_high": f"${info.get('fiftyTwoWeekHigh', 0):.2f}",
                "52_week_low": f"${info.get('fiftyTwoWeekLow', 0):.2f}",
                "last_updated": datetime.now().strftime("%Y-%m-%d")
            }
            
            # Get quarterly financials
            try:
                quarterly_financials = ticker.quarterly_financials
                if not quarterly_financials.empty:
                    # Convert to dictionary format
                    financial_data["quarterly_revenue"] = quarterly_financials.loc["Total Revenue"].to_dict()
                    financial_data["quarterly_net_income"] = quarterly_financials.loc["Net Income"].to_dict()
            except:
                financial_data["quarterly_financials"] = "Data not available"
            print("Financial data fetched successfully for:", company_ticker)
            return financial_data
            
        except Exception as e:
            return f"Error fetching financial data for {company_ticker}: {str(e)}"
    


    @tool("Fetch latest 10-K filing text")
    def sec_filing(ticker: str) -> str:
        """
        Fetch and return the text of the latest SEC 10-K filing for a given ticker.
        """
        USER_AGENT = "JohnWilter john19@gmail.com"  # Required by SEC
        Firecrawl_API_KEY = os.environ.get("FIRECRAWL_API_KEY")
        url = "https://www.sec.gov/files/company_tickers.json"
        def get_cik(ticker):
            try:
                response = requests.get(url, headers={"User-Agent": USER_AGENT})
                
                # Check if request was successful
                if response.status_code == 200:
                    data = response.json()
                    print("Response status code:", response.status_code)
                    
                    for entry in data.values():
                        if entry['ticker'].lower() == ticker.lower():
                            cik = str(entry['cik_str']).zfill(10)
                            print("Found CIK:", cik)
                            return cik
                    
                    print("AAPL ticker not found in data")
                    return None                            

            except Exception:
                print("Error fetching CIK")
                return None

        def get_filing_url(cik, ticker):
            try:
                # form_type="10-K"
                url = f"https://data.sec.gov/submissions/CIK{cik}.json"
                res = requests.get(url, headers={"User-Agent": USER_AGENT}).json()
                recent = res.get("filings", {}).get("recent", {})
                for i, form in enumerate(recent.get("form", [])):
                    if str(form) == "10-K":
                        print("Found form index:", i)
                        acc_no = recent["accessionNumber"][i].replace("-", "")
                        print("Found accession number:", acc_no)
                        filing_index = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc_no}/index.json"
                        index_json = requests.get(filing_index, headers={"User-Agent": USER_AGENT}).json()
                        for item in index_json["directory"]["item"]:
                            if item['name'].endswith('.htm') and ticker.lower() in item['name'].lower():
                                print("Found filing URL:", item['name'])
                                return f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc_no}/{item['name']}"
            except Exception:
                print("Error fetching filing URL")
                return None

        try:
            cik = get_cik(ticker)
            if not cik:
                return f"CIK not found for ticker '{ticker}'"

            filing_url = get_filing_url(cik,ticker)
            if not filing_url:
                return f"Could not retrieve recent 10-K filing for '{ticker}'"

            firecrawl = FirecrawlApp(api_key=os.environ.get("FIRECRAWL_API_KEY"))
            result = firecrawl.scrape_url(url=filing_url)
            # print("Filing markdown:", result.get("data")[0].get("markdown"))
            filing_text = result['markdown']

            # Extract specific items from the filing text
            # Skip first 3 pages which contain table of contents
            filing_text = filing_text[filing_text.find("Item 1.", filing_text.find("Item 1.") + 1):]
            # items_to_extract = ["Item 1.", "Item 7.", "Item 8."]
            items_to_extract = ["Item 1."]

            idx = [1]  # Corresponding item numbers to extract
            extracted_items = {}
            count = 0
            for item in items_to_extract:
                start_index = filing_text.find(item)
                if start_index != -1:
                    end_index = filing_text.find(f"Item {idx[count]+1}.", start_index + 1)
                    extracted_items[item] = filing_text[start_index:end_index].strip()
                    count+=1

            
            return extracted_items

        except Exception as e:
            return f"Error fetching SEC 10-K filing: {str(e)}"


    @tool("Analyze market sentiment from Yahoo")
    def analyze_market_sentiment_yahoo(company_name):
        """
        Fetches recent news articles about a company from Yahoo Finance.
        The input to this tool should be a company name (e.g., NVIDIA, AMD).
        Returns the 2 most recent news articles with their headlines and content.
        """
        try:
            # Use Yahoo Finance to get news
            if len(company_name.split()) > 1:
                # If company name has multiple words, try to find ticker
                search_term = company_name.replace(" ", "+")
                url = f"https://finance.yahoo.com/lookup?s={search_term}"
                response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Try to find the ticker from search results
                ticker = None
                table = soup.find('table', {'class': 'W(100%)'})
                if table:
                    rows = table.find_all('tr')
                    for row in rows:
                        cells = row.find_all('td')
                        if len(cells) > 1:
                            name_cell = cells[1].text
                            if company_name.lower() in name_cell.lower():
                                ticker = cells[0].text
                                break
            else:
                # Assume input is already a ticker
                ticker = company_name
            
            if not ticker:
                return f"Could not find ticker for {company_name}"
            
            # Get news for the ticker
            ticker_obj = yf.Ticker(ticker)
            news = ticker_obj.news
            
            if not news:
                return f"No recent news found for {company_name}"
            
            # Format the 2 most recent news articles
            articles = []
            for article in news[:2]:  # Get the 2 most recent news
                # Get the article URL from canonical URL
                article_url = article.get('content', {}).get('canonicalUrl', {}).get('url', '')
                print("article_url: ", article_url)
                article_content = ""
                
                if article_url:
                    try:
                        # Initialize Firecrawl
                        firecrawl = FirecrawlApp(api_key=os.environ.get("FIRECRAWL_API_KEY"))
                        
                        # Scrape the article content with specific options
                        result = firecrawl.scrape_url(
                            url=article_url,
                        )
                        if result:
                            article_content = result["markdown"]
                            print("article content: ", article_content)
                        else:
                            # Fallback to summary if full content not found
                            article_content = article.get('content', {}).get('summary', '')
                    except Exception as e:
                        print(f"Error fetching article content: {str(e)}")
                        article_content = ""

                formatted_article = {
                    "title": article.get('content', {}).get('title', ''),
                    "date": datetime.fromtimestamp(article.get('providerPublishTime', 0)).strftime("%Y-%m-%d"),
                    "source": article.get('content', {}).get('provider', {}).get('displayName', ''),
                    "content": article_content,
                    "link": article_url
                }
                articles.append(formatted_article)
            
            return {
                "company": company_name,
                "ticker": ticker,
                "articles": articles,
                "last_updated": datetime.now().strftime("%Y-%m-%d")
            }
            
        except Exception as e:
            return f"Error fetching news for {company_name}: {str(e)}"
    
    @tool("Analyze market sentiment from Serper")
    def analyze_market_sentiment_serper(company_name):
        """
        Fetches recent news articles about a company using Serper API.
        The input should be a company name (e.g., NVIDIA, AMD).
        
        Returns a list of articles with their titles, dates, sources, and content snippets.
        """
        try:
            url = "https://google.serper.dev/search"
            payload = json.dumps({ "q": company_name })
            headers = {
                'X-API-KEY': os.environ.get("SERPER_API_KEY"),
                'Content-Type': 'application/json'
            }

            response = requests.post(url, headers=headers, data=payload)
            results = response.json()

            # Preprocess topStories
            top_stories = results.get("topStories", [])
            top_items = [
                {
                    "source": story.get("source"),
                    "title": story.get("title"),
                    "snippet": story.get("snippet"),
                    "date": story.get("date"),
                    "link": story.get("link", "")
                }
                for story in top_stories
                if story.get("title") and story.get("snippet")
            ]

            # Preprocess organic results
            organic = results.get("organic", [])
            organic_items = [
                {
                    "source": item.get("source"),
                    "title": item.get("title"),
                    "snippet": item.get("snippet"),
                    "date": item.get("date"),
                    "link": item.get("link", "")
                }
                for item in organic
                if item.get("title") and item.get("snippet")
            ]

            # Merge articles and remove duplicates based on title
            seen_titles = set()
            combined_articles = []
            
            for article in top_items + organic_items:
                if article["title"] not in seen_titles:
                    seen_titles.add(article["title"])
                    combined_articles.append(article)
            
            return {
                "company": company_name,
                "articles": combined_articles
            }

        except Exception as e:
            return {
                "error": f"Error fetching news for {company_name}: {str(e)}"
            }

    # @tool("Analyze competition from Serper")
    # def get_competitor_analysis(company_name):
    #     """
    #     Provides a competitive analysis of a company and its main competitors.
    #     The input to this tool should include a company ticker symbol (e.g., NVDA, AMD).
    #     """
    #     try:



    # @tool("Get competitor analysis")
    # def get_competitor_analysis(company_ticker):
        # """
        # Provides a competitive analysis of a company and its main competitors.
        # The input to this tool should be a company ticker symbol (e.g., NVDA, AMD).
        # """
        # try:
        #     # Define competitor mappings
        #     competitor_mappings = {
        #         "NVDA": ["AMD", "INTC", "QCOM"],
        #         "AMD": ["NVDA", "INTC", "QCOM"],
        #         "INTC": ["AMD", "NVDA", "QCOM", "TSM"],
        #         "QCOM": ["NVDA", "AMD", "INTC", "AAPL"],
        #         "AAPL": ["MSFT", "GOOG", "AMZN", "SSNLF"],
        #         "MSFT": ["AAPL", "GOOG", "AMZN", "ORCL"],
        #         "GOOG": ["MSFT", "AAPL", "AMZN", "FB"],
        #         "AMZN": ["MSFT", "GOOG", "AAPL", "WMT"]
        #     }
            
        #     ticker = company_ticker.upper()
            
        #     # Get competitors for the given company
        #     if ticker not in competitor_mappings:
        #         return f"No competitor data available for {ticker}"
            
        #     competitors = competitor_mappings[ticker]
            
        #     # Get financial data for the company and its competitors
        #     main_company = yf.Ticker(ticker)
        #     main_info = main_company.info
            
        #     competitor_data = []
        #     for comp_ticker in competitors:
        #         try:
        #             comp = yf.Ticker(comp_ticker)
        #             comp_info = comp.info
                    
        #             competitor_data.append({
        #                 "ticker": comp_ticker,
        #                 "name": comp_info.get("shortName", "N/A"),
        #                 "market_cap": f"${comp_info.get('marketCap', 0) / 1000000000:.2f}B",
        #                 "pe_ratio": f"{comp_info.get('trailingPE', 0):.2f}",
        #                 "revenue_ttm": f"${comp_info.get('totalRevenue', 0) / 1000000000:.2f}B",
        #                 "profit_margin": f"{comp_info.get('profitMargins', 0) * 100:.1f}%",
        #                 "return_on_equity": f"{comp_info.get('returnOnEquity', 0) * 100:.1f}%"
        #             })
        #         except:
        #             competitor_data.append({
        #                 "ticker": comp_ticker,
        #                 "name": "Data not available",
        #                 "error": "Could not fetch data for this competitor"
        #             })
            
        #     # Get historical price performance for comparison
        #     end_date = datetime.now()
        #     start_date = end_date - pd.DateOffset(years=1)
            
        #     # Get historical data for main company
        #     main_hist = main_company.history(start=start_date, end=end_date)
        #     main_return = ((main_hist['Close'].iloc[-1] / main_hist['Close'].iloc[0]) - 1) * 100
            
        #     # Get historical data for competitors
        #     competitor_returns = {}
        #     for comp_ticker in competitors:
        #         try:
        #             comp = yf.Ticker(comp_ticker)
        #             comp_hist = comp.history(start=start_date, end=end_date)
        #             comp_return = ((comp_hist['Close'].iloc[-1] / comp_hist['Close'].iloc[0]) - 1) * 100
        #             competitor_returns[comp_ticker] = f"{comp_return:.2f}%"
        #         except:
        #             competitor_returns[comp_ticker] = "Data not available"
            
        #     # Compile the analysis
        #     analysis = {
        #         "company": main_info.get("shortName", ticker),
        #         "ticker": ticker,
        #         "main_competitors": competitors,
        #         "company_1yr_return": f"{main_return:.2f}%",
        #         "competitor_1yr_returns": competitor_returns,
        #         "competitor_details": competitor_data,
        #         "last_updated": datetime.now().strftime("%Y-%m-%d")
        #     }
            
        #     return analysis
            
        # except Exception as e:
        #     return f"Error getting competitor analysis for {company_ticker}: {str(e)}"