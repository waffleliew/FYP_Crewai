from langchain.tools import tool
import requests
import os
import json
from datetime import datetime, timedelta
import yfinance as yf
import pandas as pd
from bs4 import BeautifulSoup
import re
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore, PineconeEmbeddings
from crewai_tools import FirecrawlCrawlWebsiteTool, SerperDevTool
from dotenv import load_dotenv
from langchain.tools import tool
from firecrawl import FirecrawlApp
from gnews import GNews
from nltk.sentiment import SentimentIntensityAnalyzer
from dateutil.relativedelta import relativedelta
from typing import Union
load_dotenv()

class ResearchTools:
    
#####AUTOGEN TOOLS#####

    ALPHAVANTAGE_API_KEY = os.environ.get("ALPHAVANTAGE_API_KEY")

    @staticmethod
    def _fetch_earnings_data(ticker: str):
        """
        Helper function to fetch earnings data from Alpha Vantage API.
        Returns the earnings data response and any error message.
        """
        if not ResearchTools.ALPHAVANTAGE_API_KEY:
            return None, "Alpha Vantage API key not set in environment variable ALPHAVANTAGE_API_KEY."
            
        earnings_url = (
            f"https://www.alphavantage.co/query?function=EARNINGS&symbol={ticker}&apikey={ResearchTools.ALPHAVANTAGE_API_KEY}"
        )
        try:
            earnings_response = requests.get(earnings_url)
            earnings_data = earnings_response.json()
            
            if "Error Message" in earnings_data:
                return None, f"Alpha Vantage API error: {earnings_data['Error Message']}"
                
            return earnings_data, None
        except Exception as e:
            return None, f"Error fetching earnings data: {str(e)}"

    @staticmethod
    def historicalfinancialdata(ticker: str, year: str = None, quarter: str = None):
        """
        Returns the current quarter, previous quarter's and previous year's financial data (EPS, cash flow, income statement, balance sheet) using Alpha Vantage.
        This enables QoQ comparisons for investment analysis.
        Args:
            ticker (str): The stock ticker symbol
            year (str): The fiscal year as a string (e.g., "2022")
            quarter (str): The fiscal quarter (e.g., "Q1", "Q2", "Q3", "Q4")
        """
        try:
            print("Getting current and previous quarter's financials for ", ticker, " and reference year: ", year, " quarter: ", quarter)
            
            #to test without making api call
#             return {'financial_metrics': {'current_quarter': {'ActualEPS': '1.07', 
#                                            'Capex': '17537000',
#                                            'CashAndEquivalents': '1123660000',
#                                            'CurrentAssets': '2573493000',
#                                            'CurrentLiabilities': '1138463000',
#                                            'EstimatedEPS': '1.01',
#                                            'GrossProfit': '426350000',
#                                            'LongTermDebt': '2256910000',
#                                            'NetIncome': '219233000',
#                                            'OperatingCashFlow': '284407000',
#                                            'OperatingIncome': '293345000',
#                                            'Revenue': '1215742000',
#                                            'ShareholdersEquity': '6120748000',
#                                            'ShortTermDebt': '139786000',
#                                            'TotalDebt': 2396696000.0},
#                        'previous_quarter': {'ActualEPS': '1.08',
#                                             'Capex': '37035000',
#                                             'CashAndEquivalents': '1212822000',
#                                             'CurrentAssets': '2522470000',
#                                             'CurrentLiabilities': '1075915000',
#                                             'EstimatedEPS': '1.03',
#                                             'GrossProfit': '428979000',
#                                             'LongTermDebt': '2281441000',
#                                             'NetIncome': '221025000',
#                                             'OperatingCashFlow': '385882000',
#                                             'OperatingIncome': '298113000',
#                                             'Revenue': '1198947000',
#                                             'ShareholdersEquity': '5949346000',
#                                             'ShortTermDebt': '177232000',
#                                             'TotalDebt': 2458673000.0},
#                        'previous_year': {'ActualEPS': '1.02',
#                                          'Capex': '16931000',
#                                          'CashAndEquivalents': '1253382000',
#                                          'CurrentAssets': '2774491000',
#                                          'CurrentLiabilities': '1477671000',
#                                          'EstimatedEPS': '0.96',
#                                          'GrossProfit': '377571000',
#                                          'LongTermDebt': '2741798000',
#                                          'NetIncome': '280616000',
#                                          'OperatingCashFlow': '270752000',
#                                          'OperatingIncome': '232040000',
#                                          'Revenue': '1202218000',
#                                          'ShareholdersEquity': '5326003000',
#                                          'ShortTermDebt': '552793000',
#                                          'TotalDebt': 3294591000.0}},
#  'key_financial_ratios': {'current_quarter': {'Capex / OCF (%)': 6.17,
#                                               'Cash Conversion Ratio': 1.2973,
#                                               'Current Ratio': 2.2605,
#                                               'Debt-to-Equity': 0.3916,
#                                               'EPS Surprise (%)': 5.94,
#                                               'Free Cash Flow': 266870000.0,
#                                               'Gross Margin (%)': 35.07,
#                                               'Net Debt': 1273036000.0,
#                                               'Net Margin (%)': 18.03,
#                                               'Operating Margin (%)': 24.13},
#                           'previous_quarter': {'Capex / OCF (%)': 9.6,
#                                                'Cash Conversion Ratio': 1.7459,
#                                                'Current Ratio': 2.3445,
#                                                'Debt-to-Equity': 0.4133,
#                                                'EPS Surprise (%)': 4.85,
#                                                'Free Cash Flow': 348847000.0,
#                                                'Gross Margin (%)': 35.78,
#                                                'Net Debt': 1245851000.0,
#                                                'Net Margin (%)': 18.43,
#                                                'Operating Margin (%)': 24.86},
#                           'previous_year': {'Capex / OCF (%)': 6.25,
#                                             'Cash Conversion Ratio': 0.9648,
#                                             'Current Ratio': 1.8776,
#                                             'Debt-to-Equity': 0.6186,
#                                             'EPS Surprise (%)': 6.25,
#                                             'Free Cash Flow': 253821000.0,
#                                             'Gross Margin (%)': 31.41,
#                                             'Net Debt': 2041209000.0,
#                                             'Net Margin (%)': 23.34,
#                                             'Operating Margin (%)': 19.3}}}

            # Validate year parameter
            if year is not None and not isinstance(year, str):
                return {"error": f"Year must be a string, got {type(year).__name__} instead"}

            # Get earnings data
            earnings_data, error = ResearchTools._fetch_earnings_data(ticker)
            if error:
                return {"error": error}

            # Find fiscal year end for the given year
            fiscal_year_end = None
            try:
                if "annualEarnings" in earnings_data:
                    # First try to find exact match for the given year
                    for entry in earnings_data["annualEarnings"]:
                        if entry["fiscalDateEnding"].startswith(str(year)):
                            fiscal_year_end = entry["fiscalDateEnding"]
                            break
                    
                    # If not found, use the most recent fiscal year end
                    if not fiscal_year_end and len(earnings_data["annualEarnings"]) > 0:
                        fiscal_year_end = earnings_data["annualEarnings"][0]["fiscalDateEnding"]
                        print(f"Using most recent fiscal year end: {fiscal_year_end}")
                
                if not fiscal_year_end:
                    return {"error": f"Could not determine fiscal year end for {ticker} {year}"}
                
                # Validate fiscal year end date format
                if not re.match(r'\d{4}-\d{2}-\d{2}', fiscal_year_end):
                    return {"error": f"Invalid fiscal year end date format: {fiscal_year_end}"}
            except Exception as e:
                return {"error": f"Error processing fiscal year end date: {str(e)}"}

            # Calculate quarter end dates
            try:
                print(f"\nCalculating quarter end dates:")
                print(f"Fiscal year end: {fiscal_year_end}")
                print(f"Target quarter: {quarter}")
                
                fy_end_date = datetime.strptime(fiscal_year_end, "%Y-%m-%d")
                q_map = {
                    "Q4": fy_end_date,
                    "Q3": fy_end_date - relativedelta(months=3),
                    "Q2": fy_end_date - relativedelta(months=6),
                    "Q1": fy_end_date - relativedelta(months=9),
                }
                
                if quarter not in q_map:
                    return {"error": f"Invalid quarter: {quarter}"}
                
                quarter_end_date = q_map[quarter].strftime("%Y-%m-%d")
                print(f"Calculated quarter end date: {quarter_end_date}") #company's corresponding quarter end date for this earnings call
            except Exception as e:
                return {"error": f"Error calculating quarter end dates: {str(e)}"}
            # Previous quarter
            prev_quarter_order = {"Q1": "Q4", "Q2": "Q1", "Q3": "Q2", "Q4": "Q3"}
            prev_quarter = prev_quarter_order[quarter]
            prev_year = int(year) if quarter != "Q1" else int(year) - 1
            # Get prev fiscal year end if needed
            if prev_quarter == "Q4":
                prev_fy_end = None
                for entry in earnings_data["annualEarnings"]:
                    if entry["fiscalDateEnding"].startswith(str(prev_year)):
                        prev_fy_end = entry["fiscalDateEnding"]
                        break
                if not prev_fy_end and len(earnings_data["annualEarnings"]) > 0:
                    prev_fy_end = earnings_data["annualEarnings"][0]["fiscalDateEnding"]
                prev_fy_end_date = datetime.strptime(prev_fy_end, "%Y-%m-%d")
                prev_q_map = {
                    "Q4": prev_fy_end_date,
                    "Q3": prev_fy_end_date - relativedelta(months=3),
                    "Q2": prev_fy_end_date - relativedelta(months=6),
                    "Q1": prev_fy_end_date - relativedelta(months=9),
                }
                prev_quarter_end_date = prev_q_map[prev_quarter].strftime("%Y-%m-%d")
            else:
                prev_quarter_end_date = q_map[prev_quarter].strftime("%Y-%m-%d")
                prev_year = year

            # --- Helper to get quarter from fiscalDateEnding ---
            def match_quarter(date_str, target_date):
                try:
                    # Ensure both dates are in YYYY-MM-DD format
                    if not (isinstance(date_str, str) and isinstance(target_date, str)):
                        return False
                    if not (re.match(r'\d{4}-\d{2}-\d{2}', date_str) and re.match(r'\d{4}-\d{2}-\d{2}', target_date)):
                        return False
                    date1 = datetime.strptime(date_str, '%Y-%m-%d')
                    date2 = datetime.strptime(target_date, '%Y-%m-%d')
                    return abs((date1 - date2).days) <= 7
                except Exception:
                    return False

            # --- Get EPS Data ---
            current_eps = None
            previous_eps = None
            if "quarterlyEarnings" in earnings_data:
                for entry in earnings_data["quarterlyEarnings"]:
                    fiscal_date = entry["fiscalDateEnding"]
                    if match_quarter(fiscal_date, quarter_end_date) and not current_eps:
                        current_eps = {
                            "fiscalDateEnding": entry.get("fiscalDateEnding"),
                            "reportedDate": entry.get("reportedDate"),
                            "reportedEPS": entry.get("reportedEPS"),
                            "estimatedEPS": entry.get("estimatedEPS"),
                            "surprise": entry.get("surprise"),
                            "surprisePercentage": entry.get("surprisePercentage"),
                        }
                    if match_quarter(fiscal_date, prev_quarter_end_date) and not previous_eps:
                        previous_eps = {
                            "fiscalDateEnding": entry.get("fiscalDateEnding"),
                            "reportedDate": entry.get("reportedDate"),
                            "reportedEPS": entry.get("reportedEPS"),
                            "estimatedEPS": entry.get("estimatedEPS"),
                            "surprise": entry.get("surprise"),
                            "surprisePercentage": entry.get("surprisePercentage"),
                        }

            # --- Get Cash Flow Data ---
            cashflow_url = (
                f"https://www.alphavantage.co/query?function=CASH_FLOW&symbol={ticker}&apikey={ResearchTools.ALPHAVANTAGE_API_KEY}"
            )
            cashflow_response = requests.get(cashflow_url)
            cashflow_data = cashflow_response.json()
            current_cashflow = None
            previous_cashflow = None
            if "quarterlyReports" in cashflow_data:
                for entry in cashflow_data["quarterlyReports"]:
                    fiscal_date = entry["fiscalDateEnding"]
                    if match_quarter(fiscal_date, quarter_end_date) and not current_cashflow:
                        current_cashflow = {k: (None if v == "None" else v) for k, v in entry.items()}
                    if match_quarter(fiscal_date, prev_quarter_end_date) and not previous_cashflow:
                        previous_cashflow = {k: (None if v == "None" else v) for k, v in entry.items()}

            # --- Get Income Statement Data ---
            income_url = (
                f"https://www.alphavantage.co/query?function=INCOME_STATEMENT&symbol={ticker}&apikey={ResearchTools.ALPHAVANTAGE_API_KEY}"
            )
            income_response = requests.get(income_url)
            income_data = income_response.json()
            current_income = None
            previous_income = None
            if "quarterlyReports" in income_data:
                for entry in income_data["quarterlyReports"]:
                    fiscal_date = entry["fiscalDateEnding"]
                    if match_quarter(fiscal_date, quarter_end_date) and not current_income:
                        current_income = {k: (None if v == "None" else v) for k, v in entry.items()}
                    if match_quarter(fiscal_date, prev_quarter_end_date) and not previous_income:
                        previous_income = {k: (None if v == "None" else v) for k, v in entry.items()}

            # --- Get Balance Sheet Data ---
            balance_url = (
                f"https://www.alphavantage.co/query?function=BALANCE_SHEET&symbol={ticker}&apikey={ResearchTools.ALPHAVANTAGE_API_KEY}"
            )
            balance_response = requests.get(balance_url)
            balance_data = balance_response.json()
            current_balance = None
            previous_balance = None
            if "quarterlyReports" in balance_data:
                for entry in balance_data["quarterlyReports"]:
                    fiscal_date = entry["fiscalDateEnding"]
                    if match_quarter(fiscal_date, quarter_end_date) and not current_balance:
                        current_balance = {k: (None if v == "None" else v) for k, v in entry.items()}
                    if match_quarter(fiscal_date, prev_quarter_end_date) and not previous_balance:
                        previous_balance = {k: (None if v == "None" else v) for k, v in entry.items()}

            # --- Merge Results ---
            # --- Get Previous Year Data ---
            try:
                # Calculate previous year's fiscal year end date
                print(f"Current fiscal year end: {fiscal_year_end}")
                fy_end_datetime = datetime.strptime(fiscal_year_end, "%Y-%m-%d")
                prev_fy_end_datetime = fy_end_datetime - relativedelta(years=1)
                
                # Use the same quarter mapping logic for previous year
                prev_year_q_map = {
                    "Q4": prev_fy_end_datetime,
                    "Q3": prev_fy_end_datetime - relativedelta(months=3),
                    "Q2": prev_fy_end_datetime - relativedelta(months=6),
                    "Q1": prev_fy_end_datetime - relativedelta(months=9),
                }
                prev_year_quarter_end_date = prev_year_q_map[quarter].strftime("%Y-%m-%d")
                print(f"Previous year quarter end date: {prev_year_quarter_end_date}")
            except Exception as e:
                print(f"Error calculating previous year date: {str(e)}")
                return {"error": f"Failed to calculate previous year date: {str(e)}"}

            # Previous year EPS
            previous_year_eps = None
            if "quarterlyEarnings" in earnings_data:
                for entry in earnings_data["quarterlyEarnings"]:
                    fiscal_date = entry["fiscalDateEnding"]
                    if match_quarter(fiscal_date, prev_year_quarter_end_date) and not previous_year_eps:
                        previous_year_eps = {k: (None if v == "None" else v) for k, v in entry.items()}
                        break

            # Previous year cash flow
            previous_year_cashflow = None
            if "quarterlyReports" in cashflow_data:
                for entry in cashflow_data["quarterlyReports"]:
                    fiscal_date = entry["fiscalDateEnding"]
                    if match_quarter(fiscal_date, prev_year_quarter_end_date) and not previous_year_cashflow:
                        previous_year_cashflow = {k: (None if v == "None" else v) for k, v in entry.items()}

            # Previous year income statement
            previous_year_income = None
            if "quarterlyReports" in income_data:
                for entry in income_data["quarterlyReports"]:
                    fiscal_date = entry["fiscalDateEnding"]
                    if match_quarter(fiscal_date, prev_year_quarter_end_date) and not previous_year_income:
                        previous_year_income = {k: (None if v == "None" else v) for k, v in entry.items()}

            # Previous year balance sheet
            previous_year_balance = None
            if "quarterlyReports" in balance_data:
                for entry in balance_data["quarterlyReports"]:
                    fiscal_date = entry["fiscalDateEnding"]
                    if match_quarter(fiscal_date, prev_year_quarter_end_date) and not previous_year_balance:
                        previous_year_balance = {k: (None if v == "None" else v) for k, v in entry.items()}


            # --- Merge Results ---
            if not current_eps and not current_cashflow and not current_income and not current_balance:
                return {"error": "No current quarter found for given year and quarter."}
            
            merged = {}
            
            # Current quarter data
            if current_eps:
                merged["current_quarter_eps"] = current_eps
            if current_cashflow:
                merged["current_quarter_cashflow"] = current_cashflow
            if current_income:
                merged["current_quarter_income_statement"] = current_income
            if current_balance:
                merged["current_quarter_balance_sheet"] = current_balance
            
            # Previous quarter data (for QoQ comparison)
            if previous_eps:
                merged["previous_quarter_eps"] = previous_eps
            if previous_cashflow:
                merged["previous_quarter_cashflow"] = previous_cashflow
            if previous_income:
                merged["previous_quarter_income_statement"] = previous_income
            if previous_balance:
                merged["previous_quarter_balance_sheet"] = previous_balance

            # Previous year data
            if previous_year_eps:
                merged["previous_year_eps"] = previous_year_eps
            if previous_year_cashflow:
                merged["previous_year_cashflow"] = previous_year_cashflow
            if previous_year_income:
                merged["previous_year_income_statement"] = previous_year_income
            if previous_year_balance:
                merged["previous_year_balance_sheet"] = previous_year_balance
            
            print(merged)
            # --- Compute Key Financial Ratios ---
            def safe_float(val):
                try:
                    return float(val)
                except (TypeError, ValueError):
                    return None

            def safe_div(n, d):
                n = safe_float(n)
                d = safe_float(d)
                if n is None or d is None or d == 0:
                    return 'N/A'
                return round(n / d, 4)

            def safe_pct(n, d):
                val = safe_div(n, d)
                if val == 'N/A':
                    return val
                return round(val * 100, 2)

            def safe_sub(a, b):
                a = safe_float(a)
                b = safe_float(b)
                if a is None or b is None:
                    return 'N/A'
                return round(a - b, 2)

            def safe_add(*args):
                vals = [safe_float(x) for x in args]
                if any(v is None for v in vals):
                    return 'N/A'
                return round(sum(vals), 2)

            def get_ratio_data(merged, prefix):
                def g(key):
                    # Try all possible locations for the key
                    for section in [
                        f'{prefix}_income_statement',
                        f'{prefix}_cashflow',
                        f'{prefix}_cash_flow',
                        f'{prefix}_balance_sheet',
                        f'{prefix}_eps',
                    ]:
                        d = merged.get(section, {})
                        if key in d:
                            return d.get(key)
                    return None
                return {
                    'Revenue': g('totalRevenue'),
                    'GrossProfit': g('grossProfit'),
                    'OperatingIncome': g('operatingIncome'),
                    'NetIncome': g('netIncome'),
                    'OperatingCashFlow': g('operatingCashflow'),
                    'Capex': g('capitalExpenditures'),
                    'ShortTermDebt': g('shortTermDebt'),
                    'LongTermDebt': g('longTermDebt'),
                    'CashAndEquivalents': g('cashAndCashEquivalentsAtCarryingValue'),
                    'TotalDebt': safe_add(g('shortTermDebt'), g('longTermDebt')),
                    'ShareholdersEquity': g('totalShareholderEquity'),
                    'CurrentAssets': g('totalCurrentAssets'),
                    'CurrentLiabilities': g('totalCurrentLiabilities'),
                    'ActualEPS': g('reportedEPS'),
                    'EstimatedEPS': g('estimatedEPS'),
                }

            current = get_ratio_data(merged, 'current_quarter')
            previous = get_ratio_data(merged, 'previous_quarter')
            previous_year = get_ratio_data(merged, 'previous_year')

            def compute_ratios(data):
                ratios = {}
                ratios['Gross Margin (%)'] = safe_pct(data['GrossProfit'], data['Revenue'])
                ratios['Operating Margin (%)'] = safe_pct(data['OperatingIncome'], data['Revenue'])
                ratios['Net Margin (%)'] = safe_pct(data['NetIncome'], data['Revenue'])
                ratios['EPS Surprise (%)'] = (
                    safe_pct(safe_sub(data['ActualEPS'], data['EstimatedEPS']), data['EstimatedEPS'])
                    if data['ActualEPS'] not in [None, 'N/A'] and data['EstimatedEPS'] not in [None, 'N/A'] else 'N/A'
                )
                ratios['Free Cash Flow'] = (
                    safe_sub(data['OperatingCashFlow'], data['Capex'])
                    if data['OperatingCashFlow'] not in [None, 'N/A'] and data['Capex'] not in [None, 'N/A'] else 'N/A'
                )
                ratios['Cash Conversion Ratio'] = (
                    safe_div(data['OperatingCashFlow'], data['NetIncome'])
                    if data['OperatingCashFlow'] not in [None, 'N/A'] and data['NetIncome'] not in [None, 'N/A'] else 'N/A'
                )
                ratios['Net Debt'] = (
                    safe_sub(safe_add(data['ShortTermDebt'], data['LongTermDebt']), data['CashAndEquivalents'])
                    if data['ShortTermDebt'] not in [None, 'N/A'] and data['LongTermDebt'] not in [None, 'N/A'] and data['CashAndEquivalents'] not in [None, 'N/A'] else 'N/A'
                )
                ratios['Debt-to-Equity'] = (
                    safe_div(data['TotalDebt'], data['ShareholdersEquity'])
                    if data['TotalDebt'] not in [None, 'N/A'] and data['ShareholdersEquity'] not in [None, 'N/A'] else 'N/A'
                )
                ratios['Capex / OCF (%)'] = (
                    safe_pct(data['Capex'], data['OperatingCashFlow'])
                    if data['Capex'] not in [None, 'N/A'] and data['OperatingCashFlow'] not in [None, 'N/A'] else 'N/A'
                )
                ratios['Current Ratio'] = (
                    safe_div(data['CurrentAssets'], data['CurrentLiabilities'])
                    if data['CurrentAssets'] not in [None, 'N/A'] and data['CurrentLiabilities'] not in [None, 'N/A'] else 'N/A'
                )
                return ratios

            result = {}
            result['key_financial_ratios'] = {
                'current_quarter': compute_ratios(current),
                'previous_quarter': compute_ratios(previous),
                'previous_year': compute_ratios(previous_year)
            }
            result['financial_metrics'] = {
                'current_quarter': current,
                'previous_quarter': previous,
                'previous_year': previous_year
            }

            return result

        except Exception as e:
            return {"error": f"Error retrieving financial data for {ticker}: {str(e)}"}

    @staticmethod
    def analyzemarketsentiment(ticker: str, year: str = None, quarter: str = None):
        """
        Fetches news articles about a company from Alpha Vantage within the 30 days prior to the earnings report date.
        Args:
            ticker (str): The stock ticker symbol
            year (str): The fiscal year as a string (e.g., "2022")
            quarter (str): The fiscal quarter (e.g., "Q1", "Q2", "Q3", "Q4")
        """
        try:
            print("Analyzing market sentiment for: ", ticker, " for year: ", year, " quarter: ", quarter)

            # return f"No relevant news articles found for {ticker} in the 7 days prior to earnings reported date"
            if not ResearchTools.ALPHAVANTAGE_API_KEY:
                return {"error": "Alpha Vantage API key not set in environment variable ALPHAVANTAGE_API_KEY."}

            # Get earnings data
            earnings_data, error = ResearchTools._fetch_earnings_data(ticker)
            if error:
                return {"error": error}
            # print("earnings_data: ", earnings_data)
            # Find fiscal year end for the given year
            fiscal_year_end = None
            if "annualEarnings" in earnings_data or "annualReports" in earnings_data:
                for entry in earnings_data["annualEarnings"]:
                    if entry["fiscalDateEnding"].startswith(str(year)):
                        fiscal_year_end = entry["fiscalDateEnding"]
                        break
                if not fiscal_year_end and len(earnings_data["annualEarnings"]) > 0:
                    fiscal_year_end = earnings_data["annualEarnings"][0]["fiscalDateEnding"]
            if not fiscal_year_end:
                return {"error": f"Could not determine fiscal year end for {ticker} {year}"}

            # Calculate quarter end date to match with quarterlyEarnings
            fy_end_date = datetime.strptime(fiscal_year_end, "%Y-%m-%d")
            q_map = {
                "Q4": fy_end_date,
                "Q3": fy_end_date - relativedelta(months=3),
                "Q2": fy_end_date - relativedelta(months=6),
                "Q1": fy_end_date - relativedelta(months=9),
            }
            if quarter not in q_map:
                return {"error": f"Invalid quarter: {quarter}"}
            
            calculated_quarter_end = q_map[quarter]
            print("calculated_quarter_end: ", calculated_quarter_end)
            # Find the reportedDate for this quarter from quarterlyEarnings
            reported_date = None
            actual_quarter_end = None
            if "quarterlyEarnings" in earnings_data:
                for entry in earnings_data["quarterlyEarnings"]:
                    entry_date = datetime.strptime(entry["fiscalDateEnding"], "%Y-%m-%d")
                    # Allow for a 3-day window on either side of our calculated date
                    date_diff = abs((entry_date - calculated_quarter_end).days)
                    if date_diff <= 3:  # Within 3 days of our calculated date
                        reported_date = entry.get("reportedDate")
                        actual_quarter_end = entry["fiscalDateEnding"]
                        break
            
            if not reported_date:
                return {"error": f"Could not find reported date for {ticker} {year} {quarter} (calculated quarter end: {calculated_quarter_end.strftime('%Y-%m-%d')})"}

            # Use 7 days before reportedDate for news search
            report_date = datetime.strptime(reported_date, "%Y-%m-%d")
            start_date = report_date - timedelta(days=30)  # Get news from 7 days before earnings report
            
            # Alpha Vantage expects YYYYMMDDTHHMM format
            time_from = start_date.strftime("%Y%m%dT%H%M")
            time_to = report_date.strftime("%Y%m%dT%H%M")
            
            url = (
                f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT"
                f"&tickers={ticker}"
                f"&time_from={time_from}"
                f"&time_to={time_to}"
                f"&limit=10"
                f"&apikey={ResearchTools.ALPHAVANTAGE_API_KEY}"
            )
            print("URL:", url)
            response = requests.get(url)
            data = response.json()
            
            if "feed" not in data:
                return f"No relevant news articles found for {ticker} in the 30 days prior to earnings report on {reported_date}"
            
            # Filter and format up to 2 articles
            articles = []
            for article in data["feed"]:
                articles.append({
                    "title": article.get("title", ""),
                    "date": article.get("time_published", ""),
                    "source": article.get("source", ""),
                    "summary": article.get("summary", ""),
                    "url": article.get("url", ""),
                    "overall_sentiment_score": article.get("overall_sentiment_score", ""),
                    "overall_sentiment_label": article.get("overall_sentiment_label", ""),
                })
                if len(articles) >= 2:
                    break
                    
            return {
                "ticker": ticker,
                "calculated_quarter_end": calculated_quarter_end.strftime("%Y-%m-%d"),
                "actual_quarter_end": actual_quarter_end,
                "earnings_report_date": reported_date,
                "articles": articles,
                "last_updated": datetime.now().strftime("%Y-%m-%d")
            }
            
        except Exception as e:
            return f"Error fetching news for {ticker}: {str(e)}"

    # @staticmethod
    # def analyze_market_sentiment_yahoo_autogen(ticker: str, date: str = None):
    #     """
    #     Fetches news articles about a company from Yahoo Finance within anathe 7 days prior to the specified date (YYYY-MM-DD).
    #     The input to this tool should be a company name (e.g., NVIDIA, AMD) or ticker (e.g., NVDA, AAPL).
    #     Returns up to 2 relevant news articles within the window.
    #     """
    #     try:
    #         print("Analyzing market sentiment for: ", ticker, " on date: ", date)
    #         # First, determine if input is a ticker or company name
    #         is_ticker = len(ticker.split()) == 1 and len(ticker) <= 5
            
    #         if is_ticker:
    #             # If it's a ticker, use it directly
    #             ticker = ticker.upper()
    #             # Get company name from Yahoo Finance
    #             ticker_obj = yf.Ticker(ticker)
    #             company_info = ticker_obj.info
    #             company_name = company_info.get('shortName', ticker)
    #         else:
    #             # If it's a company name, try to find ticker
    #             search_term = ticker.replace(" ", "+")
    #             url = f"https://finance.yahoo.com/lookup?s={search_term}"
    #             response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
    #             soup = BeautifulSoup(response.text, 'html.parser')
                
    #             # Try to find the ticker from search results
    #             ticker = None
    #             table = soup.find('table', {'class': 'W(100%)'})
    #             if table:
    #                 rows = table.find_all('tr')
    #                 for row in rows:
    #                     cells = row.find_all('td')
    #                     if len(cells) > 1:
    #                         name_cell = cells[1].text
    #                         if ticker.lower() in name_cell.lower():
    #                             ticker = cells[0].text
    #                             break
            
    #         if not ticker:
    #             return f"Could not find ticker for {ticker}"
            
    #         # Get news for the ticker
    #         ticker_obj = yf.Ticker(ticker)
    #         news = ticker_obj.news
            
    #         if not news:
    #             return f"No recent news found for {ticker}"
            
    #         # Parse the date parameter
    #         if date:
    #             try:
    #                 end_date = datetime.strptime(date, "%Y-%m-%d")
    #                 start_date = end_date - timedelta(days=7)
    #             except Exception as e:
    #                 return f"Invalid date format: {date}. Please use YYYY-MM-DD."
    #         else:
    #             end_date = datetime.now()
    #             start_date = end_date - timedelta(days=7)

    #         articles = []
    #         articles_checked = 0
    #         max_articles_to_check = 20

    #         company_words = set(word.lower() for word in company_name.split())
    #         company_words.add(ticker.lower())

    #         for article in news:
    #             if len(articles) >= 2 or articles_checked >= max_articles_to_check:
    #                 break
    #             articles_checked += 1

    #             publish_time = article.get('providerPublishTime', 0)
    #             article_date = datetime.fromtimestamp(publish_time)
    #             # Only include articles within the 7-day window
    #             if not (start_date <= article_date <= end_date):
    #                 continue

    #             article_title = article.get('content', {}).get('title', '').lower()
    #             if not any(word in article_title for word in company_words):
    #                 continue
                
    #             # Get the article URL from canonical URL
    #             article_url = article.get('content', {}).get('canonicalUrl', {}).get('url', '')
    #             print("article_url: ", article_url)
    #             article_content = ""
                
    #             if article_url:
    #                 try:
    #                     # Initialize Firecrawl
    #                     firecrawl = FirecrawlApp(api_key=os.environ.get("FIRECRAWL_API_KEY"))
                        
    #                     # Scrape the article content with specific options
    #                     result = firecrawl.scrape_url(
    #                         url=article_url,
    #                     )
    #                     if result:
    #                         article_content = result["markdown"]
    #                         # print("article content: ", article_content)
    #                     else:
    #                         # Fallback to summary if full content not found
    #                         print("fail to fetch full article content, retrieving summary now")
    #                         article_content = article.get('content', {}).get('summary', '')
    #                 except Exception as e:
    #                     print(f"Error fetching article content: {str(e)}")
    #                     article_content = ""

    #             formatted_article = {
    #                 "title": article.get('content', {}).get('title', ''),
    #                 "date": article_date.strftime("%Y-%m-%d"),
    #                 "source": article.get('content', {}).get('provider', {}).get('displayName', ''),
    #                 "content": article_content,
    #                 "link": article_url
    #             }
    #             articles.append(formatted_article)
            
    #         if not articles:
    #             return f"No relevant news articles found for {ticker} in the 7 days prior to {date or end_date.strftime('%Y-%m-%d')}"
            
    #         return {
    #             "company": company_name,
    #             "ticker": ticker,
    #             "articles": articles,
    #             "last_updated": datetime.now().strftime("%Y-%m-%d")
    #         }
            
    #     except Exception as e:
    #         return f"Error fetching news for {ticker}: {str(e)}"
    
