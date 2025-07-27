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
    def get_previous_quarter_financials(ticker: str, year: str = None, quarter: str = None):
        """
        Returns the current quarter and previous quarter's financial data (EPS, cash flow, income statement, balance sheet) using Alpha Vantage.
        This enables QoQ comparisons for investment analysis.
        Args:
            ticker (str): The stock ticker symbol
            year (str): The fiscal year as a string (e.g., "2022")
            quarter (str): The fiscal quarter (e.g., "Q1", "Q2", "Q3", "Q4")
        """
        try:
            print("Getting current and previous quarter's financials for ", ticker, " and reference year: ", year, " quarter: ", quarter)
            
    #         return {'current_quarter_eps': {'fiscalDateEnding': '2019-12-31', 'reportedDate': '2020-02-13', 'reportedEPS': '1.8', 'estimatedEPS': '1.48', 'surprise': '0.32', 'surprisePercentage': '21.6216'}, 'current_quarter_cashflow': {'fiscalDateEnding': '2019-12-31', 'reportedCurrency': 'USD', 'operatingCashflow': '301408000', 'paymentsForOperatingActivities': None, 'proceedsFromOperatingActivities': None, 'changeInOperatingLiabilities': None, 'changeInOperatingAssets': None, 'depreciationDepletionAndAmortization': '31484000', 'capitalExpenditures': '28821000', 'changeInReceivables': '25517000', 'changeInInventory': '42071000', 'profitLoss': None, 'cashflowFromInvestment': '-27177000', 'cashflowFromFinancing': '-503223000', 'proceedsFromRepaymentsOfShortTermDebt': None, 'paymentsForRepurchaseOfCommonStock': None, 'paymentsForRepurchaseOfEquity': None, 'paymentsForRepurchaseOfPreferredStock': None, 'dividendPayout': '47196000', 'dividendPayoutCommonStock': '47196000', 'dividendPayoutPreferredStock': None, 'proceedsFromIssuanceOfCommonStock': None, 'proceedsFromIssuanceOfLongTermDebtAndCapitalSecuritiesNet': None, 'proceedsFromIssuanceOfPreferredStock': None, 'proceedsFromRepurchaseOfEquity': '-42000', 'proceedsFromSaleOfTreasuryStock': None, 'changeInCashAndCashEquivalents': '-228992000', 'changeInExchangeRate': None, 'netIncome': '224606000'}, 'current_quarter_income_statement': {'fiscalDateEnding': '2019-12-31', 'reportedCurrency': 'USD', 'grossProfit': '805366000', 'totalRevenue': '1728664000', 'costOfRevenue': '923298000', 'costofGoodsAndServicesSold': '923298000', 'operatingIncome': '1193855000', 'sellingGeneralAndAdministrative': '476683000', 'researchAndDevelopment': None, 'operatingExpenses': '-1329322000', 'investmentIncomeNet': None, 'netInterestIncome': '-11744000', 'interestIncome': '11744000', 'interestExpense': '11744000', 'nonInterestIncome': None, 'otherNonOperatingIncome': None, 'depreciation': None, 'depreciationAndAmortization': '31484000', 'incomeBeforeTax': '288513000', 'incomeTaxExpense': '63907000', 'interestAndDebtExpense': None, 'netIncomeFromContinuingOperations': '224606000', 'comprehensiveIncomeNetOfTax': None, 'ebit': '300257000', 'ebitda': '331741000', 'netIncome': '223998000'}, 'current_quarter_balance_sheet': {'fiscalDateEnding': '2019-12-31', 'reportedCurrency': 'USD', 'totalAssets': '15944169000', 'totalCurrentAssets': '1810344000', 'cashAndCashEquivalentsAtCarryingValue': '1530381000', 'cashAndShortTermInvestments': '1530381000', 'inventory': None, 'currentNetReceivables': '622811000', 'totalNonCurrentAssets': '1976140000', 'propertyPlantEquipment': '442014000', 'accumulatedDepreciationAmortizationPPE': None, 'intangibleAssets': '91833000', 'intangibleAssetsExcludingGoodwill': '91833000', 'goodwill': '1150908000', 'investments': None, 'longTermInvestments': '6589443000', 'shortTermInvestments': '283489000', 'otherCurrentAssets': '-1828806000', 'otherNonCurrentAssets': None, 'totalLiabilities': '11519167000', 'totalCurrentLiabilities': '102334000', 'currentAccountsPayable': '58576000', 'deferredRevenue': None, 'currentDebt': None, 'shortTermDebt': '438412000', 'totalNonCurrentLiabilities': '322776000', 'capitalLeaseObligations': '322776000', 'longTermDebt': '568232000', 'currentLongTermDebt': '160000000', 'longTermDebtNoncurrent': None, 'shortLongTermDebtTotal': '1445056000', 'otherCurrentLiabilities': '-412937000', 'otherNonCurrentLiabilities': '1916563000', 'totalShareholderEquity': '4420484000', 'treasuryStock': None, 'retainedEarnings': '2161049000', 'commonStock': '1000', 'commonStockSharesOutstanding': '113984000'}, 'previous_quarter_eps': {'fiscalDateEnding': '2019-09-30', 'reportedDate': '2019-10-24', 'reportedEPS': '1.64', 'estimatedEPS': '1.45', 'surprise': '0.19', 'surprisePercentage': '13.1034'}, 'previous_quarter_cashflow': {'fiscalDateEnding': '2019-09-30', 'reportedCurrency': 'USD', 'operatingCashflow': '310572000', 'paymentsForOperatingActivities': None, 'proceedsFromOperatingActivities': None, 'changeInOperatingLiabilities': None, 'changeInOperatingAssets': None, 'depreciationDepletionAndAmortization': '31719000', 'capitalExpenditures': '24207000', 'changeInReceivables': '-17615000', 'changeInInventory': None, 'profitLoss': None, 'cashflowFromInvestment': '-57732000', 'cashflowFromFinancing': '49588000', 'proceedsFromRepaymentsOfShortTermDebt': None, 'paymentsForRepurchaseOfCommonStock': None, 'paymentsForRepurchaseOfEquity': None, 'paymentsForRepurchaseOfPreferredStock': None, 'dividendPayout': '47139000', 'dividendPayoutCommonStock': '47139000', 'dividendPayoutPreferredStock': None, 'proceedsFromIssuanceOfCommonStock': None, 'proceedsFromIssuanceOfLongTermDebtAndCapitalSecuritiesNet': None, 'proceedsFromIssuanceOfPreferredStock': None, 'proceedsFromRepurchaseOfEquity': '-67000', 'proceedsFromSaleOfTreasuryStock': None, 'changeInCashAndCashEquivalents': '302428000', 'changeInExchangeRate': None, 'netIncome': '188167000'}, 'previous_quarter_income_statement': {'fiscalDateEnding': '2019-09-30', 'reportedCurrency': 'USD', 'grossProfit': '1422944000', 'totalRevenue': '1671196000', 'costOfRevenue': '248252000', 'costofGoodsAndServicesSold': '248252000', 'operatingIncome': '256566000', 'sellingGeneralAndAdministrative': '470683000', 'researchAndDevelopment': None, 'operatingExpenses': '-1166378000', 'investmentIncomeNet': None, 'netInterestIncome': '-12213000', 'interestIncome': '12213000', 'interestExpense': '12213000', 'nonInterestIncome': None, 'otherNonOperatingIncome': None, 'depreciation': None, 'depreciationAndAmortization': '31719000', 'incomeBeforeTax': '245338000', 'incomeTaxExpense': '57171000', 'interestAndDebtExpense': None, 'netIncomeFromContinuingOperations': '188167000', 'comprehensiveIncomeNetOfTax': None, 'ebit': '257551000', 'ebitda': '289270000', 'netIncome': '187182000'}, 'previous_quarter_balance_sheet': {'fiscalDateEnding': '2019-09-30', 'reportedCurrency': 'USD', 'totalAssets': '11804092000', 'totalCurrentAssets': '2062815000', 'cashAndCashEquivalentsAtCarryingValue': '1749458000', 'cashAndShortTermInvestments': '1749458000', 'inventory': None, 'currentNetReceivables': '351816000', 'totalNonCurrentAssets': '2315087000', 'propertyPlantEquipment': '447980000', 'accumulatedDepreciationAmortizationPPE': None, 'intangibleAssets': '92855000', 'intangibleAssetsExcludingGoodwill': '92855000', 'goodwill': '1149765000', 'investments': None, 'longTermInvestments': '6553134000', 'shortTermInvestments': '215766000', 'otherCurrentAssets': '-411175000', 'otherNonCurrentAssets': None, 'totalLiabilities': '7557870000', 'totalCurrentLiabilities': '351562000', 'currentAccountsPayable': '812499000', 'deferredRevenue': None, 'currentDebt': None, 'shortTermDebt': '233483000', 'totalNonCurrentLiabilities': '1526337000', 'capitalLeaseObligations': '319445000', 'longTermDebt': '728332000', 'currentLongTermDebt': None, 'longTermDebtNoncurrent': None, 'shortLongTermDebtTotal': '1281260000', 'otherCurrentLiabilities': '498508000', 'otherNonCurrentLiabilities': '1869115000', 'totalShareholderEquity': '4242141000', 'treasuryStock': None, 'retainedEarnings': '1985073000', 'commonStock': '1000', 'commonStockSharesOutstanding': '113741000'},    # Computed financial ratios for both quarters
    # "key_financial_ratios": {
    #     "current_quarter": {
    #         "Gross Margin (%)": 46.6,
    #         "Operating Margin (%)": 69.1,
    #         "Net Margin (%)": 13.0,
    #         "EPS Surprise (%)": 21.62,
    #         "Free Cash Flow": 272586000.0,
    #         "Cash Conversion Ratio": 1.34,
    #         "Net Debt": -523977000.0,
    #         "Debt-to-Equity": 0.23,
    #         "Capex / OCF (%)": 9.56,
    #         "Current Ratio": 17.69
    #     },
    #     "previous_quarter": {
    #         "Gross Margin (%)": 85.2,
    #         "Operating Margin (%)": 15.3,
    #         "Net Margin (%)": 11.2,
    #         "EPS Surprise (%)": 13.1,
    #         "Free Cash Flow": 286365000.0,
    #         "Cash Conversion Ratio": 1.66,
    #         "Net Debt": -786369000.0,
    #         "Debt-to-Equity": 0.23,
    #         "Capex / OCF (%)": 7.8,
    #         "Current Ratio": 5.87
    #     }
    # },}

            # Validate year parameter
            if year is not None and not isinstance(year, str):
                return {"error": f"Year must be a string, got {type(year).__name__} instead"}

            # Get earnings data
            earnings_data, error = ResearchTools._fetch_earnings_data(ticker)
            if error:
                return {"error": error}

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

            # Calculate quarter end dates
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
            print(quarter_end_date) #company's corresponding quarter end date for this earnings call
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
                return date_str == target_date

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
                'previous_quarter': compute_ratios(previous)
            }
            result['financial_metrics'] = {
                'current_quarter': current,
                'previous_quarter': previous
            }

            return result

        except Exception as e:
            return {"error": f"Error retrieving financial data for {ticker}: {str(e)}"}

    @staticmethod
    def analyze_market_sentiment_alphavantage(ticker: str, year: str = None, quarter: str = None):
        """
        Fetches news articles about a company from Alpha Vantage within the 7 days prior to the earnings report date.
        Args:
            ticker (str): The stock ticker symbol
            year (str): The fiscal year as a string (e.g., "2022")
            quarter (str): The fiscal quarter (e.g., "Q1", "Q2", "Q3", "Q4")
        """
        try:
            print("Analyzing market sentiment for: ", ticker, " for year: ", year, " quarter: ", quarter)

            # return f"No relevant news articles found for {ticker} in the 7 days prior to earnings report on d"
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
            start_date = report_date - timedelta(days=7)  # Get news from 7 days before earnings report
            
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
                return f"No relevant news articles found for {ticker} in the 7 days prior to earnings report on {reported_date}"
            
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
    


#####CREWAI TOOLS#####
    # Class variable to store filing text
    _filing_text_cache = {}
    
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
    

    @tool("Analyze market sentiment from Yahoo")
    def analyze_market_sentiment_yahoo_crewai(company_ticker):
        """
        Fetches recent news articles about a company from Yahoo Finance.
        The input to this tool should be a company name (e.g., NVIDIA, AMD) or ticker (e.g., NVDA, AAPL).
        Returns the 2 most recent news articles with their headlines and content.
        """
        try:
            # First, determine if input is a ticker or company name
            is_ticker = len(company_ticker.split()) == 1 and len(company_ticker) <= 5
            
            if is_ticker:
                # If it's a ticker, use it directly
                ticker = company_ticker.upper()
                # Get company name from Yahoo Finance
                ticker_obj = yf.Ticker(ticker)
                company_info = ticker_obj.info
                company_name = company_info.get('shortName', ticker)
            else:
                # If it's a company name, try to find ticker
                search_term = company_ticker.replace(" ", "+")
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
                            if company_ticker.lower() in name_cell.lower():
                                ticker = cells[0].text
                                break
            
            if not ticker:
                return f"Could not find ticker for {company_ticker}"
            
            # Get news for the ticker
            ticker_obj = yf.Ticker(ticker)
            news = ticker_obj.news
            
            if not news:
                return f"No recent news found for {company_ticker}"
            
            # Format the 2 most recent relevant news articles
            articles = []
            articles_checked = 0
            max_articles_to_check = 10  # Check up to 10 articles to find 2 relevant ones
            
            # Split company name into words for partial matching
            company_words = set(word.lower() for word in company_name.split())
            # Add ticker to the set of words to check
            company_words.add(ticker.lower())
            
            for article in news:
                if len(articles) >= 2 or articles_checked >= max_articles_to_check:
                    break
                    
                articles_checked += 1
                
                # Check if article title contains any of the company words
                article_title = article.get('content', {}).get('title', '').lower()
                if not any(word in article_title for word in company_words):
                    continue
                
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
                            # print("article content: ", article_content)
                        else:
                            # Fallback to summary if full content not found
                            print("fail to fetch full article content, retrieving summary now")
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
            
            if not articles:
                return f"No relevant news articles found for {company_ticker} in the last {max_articles_to_check} articles"
            
            return {
                "company": company_name,
                "ticker": ticker,
                "articles": articles,
                "last_updated": datetime.now().strftime("%Y-%m-%d")
            }
            
        except Exception as e:
            return f"Error fetching news for {company_ticker}: {str(e)}"
    
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

    @tool("Compare with industry peers")
    def compare_industry_metrics(ticker):
        """
        Compares company metrics with industry averages and peers.
        The input should be a company ticker symbol (e.g., NVDA, AMD).
        Returns a comparison of key financial metrics with industry averages and top competitors.
        """
        try:
            # Get company data
            company = yf.Ticker(ticker)
            company_info = company.info
            
            # Get industry and sector
            industry = company_info.get('industry', '')
            sector = company_info.get('sector', '')
            
            # Get competitors using Firecrawl
            competitors = []
            try:
                # Search for companies in the same industry
                search_url = f"https://finance.yahoo.com/screener/predefined/ms_technology?count=100"
                print("Searching URL:", search_url)
                
                # Initialize Firecrawl
                firecrawl = FirecrawlApp(api_key=os.environ.get("FIRECRAWL_API_KEY"))
                
                # Scrape the page
                result = firecrawl.scrape_url(url=search_url)
                print("Firecrawl result received:", bool(result))
                
                if result and 'markdown' in result:
                    print("Markdown content found")
                    # Split the markdown content into lines
                    lines = result['markdown'].split('\n')
                    print("Number of lines in markdown:", len(lines))
                    
                    # Find the table with company data
                    table_start = False
                    for line in lines:
                        # Look for the table header that contains "Name | Last Price"
                        if "| Name | Last Price |" in line:
                            table_start = True
                            continue
                        
                        # Process table rows
                        if table_start and line.startswith('|'):
                            # Split the line by | and get the first column
                            columns = line.split('|')
                            if len(columns) >= 2:
                                # Extract ticker from the first column
                                # Format is usually [TICKER Company Name]
                                ticker_text = columns[1].strip()
                                if '[' in ticker_text and ']' in ticker_text:
                                    comp_ticker = ticker_text.split('[')[1].split(' ')[0]
                                    if comp_ticker != ticker:  # Exclude the original company
                                        competitors.append(comp_ticker)
                                        if len(competitors) >= 5:  # Limit to top 5 competitors
                                            break
                        
                        # Stop if we hit the next section
                        if table_start and line.startswith('Start Investing'):
                            break
            except Exception as e:
                print(f"Error finding competitors through web search: {str(e)}")
            print("competitors from ci metrics: ", competitors)
            # If no competitors found, use sector-based fallback
            if not competitors:
                sector_competitors = {
                    'Technology': ['MSFT', 'GOOGL', 'AAPL', 'INTC', 'AMD'],
                    'Healthcare': ['JNJ', 'PFE', 'MRK', 'ABBV', 'LLY'],
                    'Financial Services': ['JPM', 'BAC', 'WFC', 'GS', 'MS'],
                    'Consumer Cyclical': ['AMZN', 'HD', 'MCD', 'SBUX', 'NKE'],
                    'Energy': ['XOM', 'CVX', 'COP', 'SLB', 'EOG']
                }
                competitors = sector_competitors.get(sector, ['MSFT', 'GOOGL', 'AAPL'])
            
            # Get financial metrics for comparison
            metrics = {
                'market_cap': company_info.get('marketCap', 0),
                'pe_ratio': company_info.get('trailingPE', 0),
                'profit_margin': company_info.get('profitMargins', 0),
                'revenue_growth': company_info.get('revenueGrowth', 0),
                'operating_margin': company_info.get('operatingMargins', 0),
                'return_on_equity': company_info.get('returnOnEquity', 0),
                'debt_to_equity': company_info.get('debtToEquity', 0)
            }
            
            # Get peer data
            peer_data = []
            for peer in competitors[:5]:  # Limit to top 5 peers
                try:
                    peer_ticker = yf.Ticker(peer)
                    peer_info = peer_ticker.info
                    
                    peer_metrics = {
                        'ticker': peer,
                        'name': peer_info.get('shortName', peer),
                        'market_cap': peer_info.get('marketCap', 0),
                        'pe_ratio': peer_info.get('trailingPE', 0),
                        'profit_margin': peer_info.get('profitMargins', 0),
                        'revenue_growth': peer_info.get('revenueGrowth', 0),
                        'operating_margin': peer_info.get('operatingMargins', 0),
                        'return_on_equity': peer_info.get('returnOnEquity', 0),
                        'debt_to_equity': peer_info.get('debtToEquity', 0)
                    }
                    peer_data.append(peer_metrics)
                except:
                    continue
            
            # Calculate industry averages
            industry_averages = {}
            for metric in metrics.keys():
                values = [p[metric] for p in peer_data if p[metric] is not None]
                if values:
                    industry_averages[metric] = sum(values) / len(values)
            
            # Format the comparison
            comparison = {
                'company': {
                    'ticker': ticker,
                    'name': company_info.get('shortName', ticker),
                    'industry': industry,
                    'sector': sector,
                    'metrics': metrics
                },
                'industry_averages': industry_averages,
                'peers': peer_data,
                'last_updated': datetime.now().strftime("%Y-%m-%d")
            }
            
            return comparison
            
        except Exception as e:
            return f"Error comparing industry metrics for {ticker}: {str(e)}"

    @tool("Analyze historical trends")
    def analyze_historical_trends(ticker: str):
        """
        Analyzes historical financial trends over multiple periods.
        The input should be a company ticker symbol (e.g., NVDA, AMD).
        Returns historical trends for key financial metrics over different time periods.
        """
        try:
            company = yf.Ticker(ticker)
            
            # Get historical price data
            periods = {
                '1y': company.history(period='1y'),
                '2y': company.history(period='2y'),
                '5y': company.history(period='5y')
            }
            
            # Get financials
            quarterly = company.quarterly_financials
            annual = company.financials
            
            trends = {
                'price_trends': {},
                'volume_trends': {},
                'financial_trends': {}
            }

            # Price trends and CAGR
            for period, data in periods.items():
                if not data.empty:
                    start_price = data['Close'].iloc[0]
                    end_price = data['Close'].iloc[-1]
                    years = len(data) / 252  # ~252 trading days/year
                    cagr = ((end_price / start_price) ** (1 / years) - 1) * 100

                    trends['price_trends'][period] = {
                        'start_price': float(start_price),
                        'end_price': float(end_price),
                        'high': float(data['High'].max()),
                        'low': float(data['Low'].min()),
                        'avg_volume': float(data['Volume'].mean()),
                        'price_change_pct': ((end_price / start_price) - 1) * 100,
                        'cagr_pct': cagr
                    }

            # Format quarterly financials with readable quarter labels
            if not quarterly.empty:
                for metric in ['Total Revenue', 'Net Income', 'Operating Income']:
                    if metric in quarterly.index:
                        series = quarterly.loc[metric]
                        formatted = {
                            f"{d.year}Q{((d.month - 1) // 3 + 1)}": float(v)
                            for d, v in series.items()
                        }
                        trends['financial_trends'][f'quarterly_{metric.lower().replace(" ", "_")}'] = formatted

            # Include annual financials if available
            if not annual.empty:
                for metric in ['Total Revenue', 'Net Income', 'Operating Income']:
                    if metric in annual.index:
                        series = annual.loc[metric]
                        formatted = {
                            str(d.year): float(v)
                            for d, v in series.items()
                        }
                        trends['financial_trends'][f'annual_{metric.lower().replace(" ", "_")}'] = formatted

            # Compute YoY growth using most recent and 4th previous quarter
            growth_rates = {}
            for key, values in trends['financial_trends'].items():
                if 'quarterly' in key and len(values) >= 5:
                    sorted_keys = sorted(values.keys(), reverse=True)
                    current = values[sorted_keys[0]]
                    previous = values[sorted_keys[4]]
                    if previous != 0:
                        growth_rates[key.replace("quarterly_", "")] = ((current - previous) / abs(previous)) * 100

            trends['growth_rates_yoy'] = growth_rates

            return {
                'ticker': ticker,
                'company_name': company.info.get('shortName', ticker),
                'trends': trends,
                'last_updated': datetime.now().strftime("%Y-%m-%d")
            }

        except Exception as e:
            return {"error": f"Error analyzing historical trends for {ticker}: {str(e)}"}

       # @tool("Analyze historical trends")
 

    @tool("Analyze competitors")
    def analyze_competitors(ticker):
        """
        Provides detailed competitor analysis and market positioning.
        The input should be a company ticker symbol (e.g., NVDA, AMD).
        Returns a comprehensive analysis of competitors, market share, and competitive advantages.
        """
        try:
            company = yf.Ticker(ticker)
            company_info = company.info
            
            # Get industry and sector
            industry = company_info.get('industry', '')
            sector = company_info.get('sector', '')
            
            # Get competitors from Yahoo Finance
            competitors = []

            # Method 2: If no competitors found, search for companies in the same industry
            try:
                # Search for companies in the same industry
                search_url = f"https://finance.yahoo.com/screener/predefined/ms_technology?count=100"
                print("Searching URL:", search_url)
                
                # Initialize Firecrawl
                firecrawl = FirecrawlApp(api_key=os.environ.get("FIRECRAWL_API_KEY"))
                
                # Scrape the page
                result = firecrawl.scrape_url(url=search_url)
                print("Firecrawl result received:", bool(result))
                
                if result and 'markdown' in result:
                    print("Markdown content found")
                    # Split the markdown content into lines
                    lines = result['markdown'].split('\n')
                    print("Number of lines in markdown:", len(lines))
                    
                    # Find the table with company data
                    table_start = False
                    for line in lines:
                        # Look for the table header that contains "Name | Last Price"
                        if "| Name | Last Price |" in line:
                            table_start = True
                            continue
                        
                        # Process table rows
                        if table_start and line.startswith('|'):
                            # print("Processing table row:", line)
                            # Split the line by | and get the first column
                            columns = line.split('|')
                            if len(columns) >= 2:
                                # Extract ticker from the first column
                                # Format is usually [TICKER Company Name]
                                ticker_text = columns[1].strip()
                                # print("Ticker text found:", ticker_text)
                                if '[' in ticker_text and ']' in ticker_text:
                                    ticker = ticker_text.split('[')[1].split(' ')[0]
                                    print("Extracted ticker:", ticker)
                                    competitors.append(ticker)
                                    print("Added competitor:", ticker)
                                    if len(competitors) >= 10:  # Stop after getting 10 companies
                                        break
                        
                        # Stop if we hit the next section
                        if table_start and line.startswith('Start Investing'):
                            print("Reached end of company table")
                            break
                else:
                    print("No markdown content found in result")
                    
            except Exception as e:
                print(f"Error finding competitors through web search: {str(e)}")
            
            print("Final competitors list:", competitors)
            # Method 3: If still no competitors, use sector-based fallback
            if not competitors:
                sector_competitors = {
                    'Technology': ['MSFT', 'GOOGL', 'AAPL', 'INTC', 'AMD'],
                    'Healthcare': ['JNJ', 'PFE', 'MRK', 'ABBV', 'LLY'],
                    'Financial Services': ['JPM', 'BAC', 'WFC', 'GS', 'MS'],
                    'Consumer Cyclical': ['AMZN', 'HD', 'MCD', 'SBUX', 'NKE'],
                    'Energy': ['XOM', 'CVX', 'COP', 'SLB', 'EOG']
                }
                competitors = sector_competitors.get(sector, ['MSFT', 'GOOGL', 'AAPL'])
            
            # Get financial data for company and competitors
            company_data = {
                'ticker': ticker,
                'name': company_info.get('shortName', ticker),
                'market_cap': company_info.get('marketCap', 0),
                'revenue': company_info.get('totalRevenue', 0),
                'profit_margin': company_info.get('profitMargins', 0),
                'operating_margin': company_info.get('operatingMargins', 0),
                'pe_ratio': company_info.get('trailingPE', 0),
                'beta': company_info.get('beta', 0)
            }
            
            competitor_data = []
            for comp_ticker in competitors[:5]:  # Limit to top 5 competitors
                try:
                    comp = yf.Ticker(comp_ticker)
                    comp_info = comp.info
                    
                    comp_data = {
                        'ticker': comp_ticker,
                        'name': comp_info.get('shortName', comp_ticker),
                        'market_cap': comp_info.get('marketCap', 0),
                        'revenue': comp_info.get('totalRevenue', 0),
                        'profit_margin': comp_info.get('profitMargins', 0),
                        'operating_margin': comp_info.get('operatingMargins', 0),
                        'pe_ratio': comp_info.get('trailingPE', 0),
                        'beta': comp_info.get('beta', 0)
                    }
                    competitor_data.append(comp_data)
                except:
                    continue
            
            # Calculate market share
            total_revenue = company_data['revenue'] + sum(comp['revenue'] for comp in competitor_data)
            if total_revenue > 0:
                company_data['market_share'] = (company_data['revenue'] / total_revenue) * 100
                for comp in competitor_data:
                    comp['market_share'] = (comp['revenue'] / total_revenue) * 100
            
            # Get recent news for competitive analysis
            try:
                news = company.news
                competitor_news = []
                for comp in competitors[:3]:  # Limit to top 3 competitors
                    try:
                        comp_news = yf.Ticker(comp).news
                        if comp_news:
                            competitor_news.extend(comp_news[:2])  # Get 2 most recent news items
                    except:
                        continue
                
                news_analysis = {
                    'company_news': news[:5],  # Top 5 news items
                    'competitor_news': competitor_news
                }
            except:
                news_analysis = None
            
            return {
                'company': company_data,
                'competitors': competitor_data,
                'industry': industry,
                'sector': sector,
                'news_analysis': news_analysis,
                'last_updated': datetime.now().strftime("%Y-%m-%d")
            }
            
        except Exception as e:
            return f"Error analyzing competitors for {ticker}: {str(e)}"

    @tool("Analyze financial ratios")
    def analyze_financial_ratios(ticker):
        """
        Analyzes key financial ratios and their trends for a given ticker.
        The input should be a company ticker symbol (e.g., NVDA, AMD).
        Returns a comprehensive analysis of key financial ratios and their trends.
        """
        try:
            company = yf.Ticker(ticker)
            info = company.info
            
            # Calculate key financial ratios
            ratios = {
                'profitability_ratios': {
                    'gross_margin': info.get('grossMargins', 0) * 100,
                    'operating_margin': info.get('operatingMargins', 0) * 100,
                    'net_margin': info.get('profitMargins', 0) * 100,
                    'return_on_equity': info.get('returnOnEquity', 0) * 100,
                    'return_on_assets': info.get('returnOnAssets', 0) * 100
                },
                'liquidity_ratios': {
                    'current_ratio': info.get('currentRatio', 0),
                    'quick_ratio': info.get('quickRatio', 0),
                    'cash_ratio': info.get('cashRatio', 0)
                },
                'leverage_ratios': {
                    'debt_to_equity': info.get('debtToEquity', 0),
                    'debt_to_assets': info.get('debtToAssets', 0),
                    'interest_coverage': info.get('interestCoverage', 0)
                },
                'efficiency_ratios': {
                    'asset_turnover': info.get('assetTurnover', 0),
                    'inventory_turnover': info.get('inventoryTurnover', 0),
                    'receivables_turnover': info.get('receivablesTurnover', 0)
                }
            }
            
            # Get historical data for trend analysis
            historical = company.history(period='5y')
            quarterly = company.quarterly_financials
            
            # Calculate trends
            trends = {
                'revenue_growth': info.get('revenueGrowth', 0) * 100,
                'earnings_growth': info.get('earningsGrowth', 0) * 100,
                'dividend_growth': info.get('dividendRate', 0) / info.get('trailingAnnualDividendRate', 1) - 1 if info.get('trailingAnnualDividendRate', 0) > 0 else 0
            }
            
            return {
                'ticker': ticker,
                'company_name': info.get('shortName', ticker),
                'ratios': ratios,
                'trends': trends,
                'last_updated': datetime.now().strftime("%Y-%m-%d")
            }
            
        except Exception as e:
            return f"Error analyzing financial ratios for {ticker}: {str(e)}"

    @tool("Evaluate growth potential")
    def evaluate_growth_potential(ticker):
        """
        Evaluates company's growth potential and opportunities.
        The input should be a company ticker symbol (e.g., NVDA, AMD).
        Returns a comprehensive analysis of growth potential and opportunities.
        """
        try:
            company = yf.Ticker(ticker)
            info = company.info
            
            # Get growth metrics
            growth_metrics = {
                'revenue_growth': info.get('revenueGrowth', 0) * 100,
                'earnings_growth': info.get('earningsGrowth', 0) * 100,
                'earnings_quarterly_growth': info.get('earningsQuarterlyGrowth', 0) * 100,
                'earnings_annual_growth': info.get('earningsAnnualGrowth', 0) * 100
            }
            
            # Get market opportunity metrics
            market_metrics = {
                'market_cap': info.get('marketCap', 0),
                'enterprise_value': info.get('enterpriseValue', 0),
                'forward_pe': info.get('forwardPE', 0),
                'peg_ratio': info.get('pegRatio', 0)
            }
            
            # Get growth initiatives
            growth_initiatives = {
                'research_development': info.get('researchAndDevelopment', 0),
                'capital_expenditure': info.get('capitalExpenditure', 0),
                'acquisitions': info.get('acquisitions', 0)
            }
            
            # Get analyst estimates
            analyst_estimates = {
                'revenue_estimate': info.get('revenueEstimate', 0),
                'earnings_estimate': info.get('earningsEstimate', 0),
                'growth_estimate': info.get('growthEstimate', 0)
            }
            
            return {
                'ticker': ticker,
                'company_name': info.get('shortName', ticker),
                'growth_metrics': growth_metrics,
                'market_metrics': market_metrics,
                'growth_initiatives': growth_initiatives,
                'analyst_estimates': analyst_estimates,
                'last_updated': datetime.now().strftime("%Y-%m-%d")
            }
            
        except Exception as e:
            return f"Error evaluating growth potential for {ticker}: {str(e)}"

    @tool("Assess management quality")
    def assess_management_quality(ticker):
        """
        Assesses management team quality and track record.
        The input should be a company ticker symbol (e.g., NVDA, AMD).
        Returns a comprehensive analysis of management quality and track record.
        """
        try:
            company = yf.Ticker(ticker)
            info = company.info
            
            # Get management information
            management = {
                'officers': info.get('companyOfficers', []),
                'board_members': info.get('boardMembers', []),
                'insider_holders': info.get('insiderHolders', []),
                'institutional_holders': info.get('institutionalHolders', [])
            }
            
            # Get management performance metrics
            performance_metrics = {
                'return_on_equity': info.get('returnOnEquity', 0) * 100,
                'return_on_assets': info.get('returnOnAssets', 0) * 100,
                'return_on_capital': info.get('returnOnCapital', 0) * 100,
                'profit_margin': info.get('profitMargins', 0) * 100
            }
            
            # Get corporate governance metrics
            governance_metrics = {
                'insider_ownership': info.get('insiderOwnership', 0) * 100,
                'institutional_ownership': info.get('institutionalOwnership', 0) * 100,
                'short_interest': info.get('shortInterest', 0),
                'short_percent_of_float': info.get('shortPercentOfFloat', 0) * 100
            }
            
            # Get management compensation
            compensation = {
                'executive_compensation': info.get('executiveCompensation', {}),
                'compensation_as_of_epoch_date': info.get('compensationAsOfEpochDate', 0)
            }
            
            return {
                'ticker': ticker,
                'company_name': info.get('shortName', ticker),
                'management': management,
                'performance_metrics': performance_metrics,
                'governance_metrics': governance_metrics,
                'compensation': compensation,
                'last_updated': datetime.now().strftime("%Y-%m-%d")
            }
            
        except Exception as e:
            return f"Error assessing management quality for {ticker}: {str(e)}"


    @staticmethod
    def sec_filing(ticker: str) -> str:
        """
        Fetch and return the text of the latest SEC 10-K filing for a given ticker.
        Uses caching to avoid repeated scraping of the same filing.
        """
        # Check if we already have the filing text cached
        if ticker in ResearchTools._filing_text_cache:
            return ResearchTools._filing_text_cache[ticker]

        USER_AGENT = "Johnnndilter john10@gmail.com"  # Required by SEC
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
            filing_text = result['markdown']

            # Keep only content from Item 16 onwards
            item16_index = filing_text.find("Item 16.")
            if item16_index != -1:
                filing_text = filing_text[item16_index:].strip()
            
            # Cache the filing text
            ResearchTools._filing_text_cache[ticker] = filing_text
            
            return filing_text

        except Exception as e:
            return f"Error fetching SEC 10-K filing: {str(e)}"


    @tool("Fetch business overview from 10-K")
    def sec_business_overview(ticker: str) -> str:
        """
        Fetch and return Item 1 (Business Overview) from the latest SEC 10-K filing.
        Relevant for company overview and business profile analysis.
        """
        
        try:
            filing_text = ResearchTools.sec_filing(ticker)

            # Extract Item 1
            start_index = filing_text.find("Item 1.")
            if start_index == -1:
                return "Item 1 not found in filing"
            
            end_index = filing_text.find("Item 1A.", start_index + 1)
            if end_index == -1:
                end_index = len(filing_text)
            
            item_1 = filing_text[start_index:end_index].strip()
            
            # Save to file for reference
            with open('business_overview.json', 'w') as f:
                json.dump({
                    "ticker": ticker,
                    "item": "Item 1 - Business Overview",
                    "content": item_1
                }, f, indent=4)

            return {
                "ticker": ticker,
                "item": "Item 1 - Business Overview",
                "content": item_1
            }
        except Exception as e:
            return f"Error extracting business overview: {str(e)}"

    @tool("Fetch financial statements from 10-K")
    def sec_financial_statements(ticker: str) -> str:
        """
        Fetch and return Item 8 (Financial Statements) from the latest SEC 10-K filing.
        Relevant for financial statements analysis and financial ratios analysis.
        """
        try:
            filing_text = ResearchTools.sec_filing(ticker)

            # Extract Item 8
            start_index = filing_text.find("Item 8.")
            if start_index == -1:
                return "Item 8 not found in filing"
            
            end_index = filing_text.find("Item 9.", start_index + 1)
            if end_index == -1:
                end_index = len(filing_text)
            
            item_8 = filing_text[start_index:end_index].strip()
            
            # Save to file for reference
            with open('financial_statements.json', 'w') as f:
                json.dump({
                    "ticker": ticker,
                    "item": "Item 8 - Financial Statements and Supplementary Data",
                    "content": item_8
                }, f, indent=4)

            return {
                "ticker": ticker,
                "item": "Item 8 - Financial Statements and Supplementary Data",
                "content": item_8
            }
        except Exception as e:
            return f"Error extracting financial statements: {str(e)}"

    @tool("Fetch MD&A from 10-K")
    def sec_mda(ticker: str) -> str:
        """
        Fetch and return Item 7 (Management's Discussion and Analysis) from the latest SEC 10-K filing.
        Relevant for strategic analysis and future outlook.
        """
        try:
            filing_text = ResearchTools.sec_filing(ticker)

            # Extract Item 7
            start_index = filing_text.find("Item 7.")
            if start_index == -1:
                return "Item 7 not found in filing"
            
            end_index = filing_text.find("Item 7A.", start_index + 1)
            if end_index == -1:
                end_index = len(filing_text)
            
            item_7 = filing_text[start_index:end_index].strip()
            
            # Save to file for reference
            with open('mda_data.json', 'w') as f:
                json.dump({
                    "ticker": ticker,
                    "item": "Item 7 - Management's Discussion and Analysis",
                    "content": item_7
                }, f, indent=4)
            
            return {
                "ticker": ticker,
                "item": "Item 7 - Management's Discussion and Analysis",
                "content": item_7
            }
        except Exception as e:
            return f"Error extracting MD&A: {str(e)}"

    @tool("Fetch risk factors from 10-K")
    def sec_risk_factors(ticker: str) -> str:
        """
        Fetch and return Item 1A (Risk Factors) from the latest SEC 10-K filing.
        Relevant for risk analysis and assessment.
        """
        try:
            filing_text = ResearchTools.sec_filing(ticker)

            # Extract Item 1A
            start_index = filing_text.find("Item 1A.")
            if start_index == -1:
                return "Item 1A not found in filing"
            
            end_index = filing_text.find("Item 1B.", start_index + 1)
            if end_index == -1:
                end_index = len(filing_text)
            
            item_1a = filing_text[start_index:end_index].strip()
            
            # Save to file for reference
            with open('risk_factors.json', 'w') as f:
                json.dump({
                    "ticker": ticker,
                    "item": "Item 1A - Risk Factors",
                    "content": item_1a
                }, f, indent=4)

            return {
                "ticker": ticker,
                "item": "Item 1A - Risk Factors",
                "content": item_1a
            }
        except Exception as e:
            return f"Error extracting risk factors: {str(e)}"

    @tool("Fetch market risk disclosures from 10-K")
    def sec_market_risk(ticker: str) -> str:
        """
        Fetch and return Item 7A (Market Risk Disclosures) from the latest SEC 10-K filing.
        Relevant for risk analysis and market risk assessment.
        """
        try:
            filing_text = ResearchTools.sec_filing(ticker)
            # Extract Item 7A
            start_index = filing_text.find("Item 7A.")
            if start_index == -1:
                return "Item 7A not found in filing"
            
            end_index = filing_text.find("Item 8.", start_index + 1)
            if end_index == -1:
                end_index = len(filing_text)
            
            item_7a = filing_text[start_index:end_index].strip()
            
            # Save to file for reference
            with open('market_risk.json', 'w') as f:
                json.dump({
                    "ticker": ticker,
                    "item": "Item 7A - Quantitative and Qualitative Disclosures About Market Risk",
                    "content": item_7a
                }, f, indent=4)

            return {
                "ticker": ticker,
                "item": "Item 7A - Quantitative and Qualitative Disclosures About Market Risk",
                "content": item_7a
            }
        except Exception as e:
            return f"Error extracting market risk disclosures: {str(e)}"

    @tool("Fetch corporate governance from 10-K")
    def sec_corporate_governance(ticker: str) -> str:
        """
        Fetch and return corporate governance related items (Items 9-14) from the latest SEC 10-K filing.
        Relevant for governance analysis and risk assessment.
        """
        try:
            filing_text = ResearchTools.sec_filing(ticker)

            governance_items = {
                "Item 9.": "Changes in and Disagreements with Accountants",
                "Item 9A.": "Controls and Procedures",
                "Item 9B.": "Other Information",
                "Item 10.": "Directors, Executive Officers and Corporate Governance",
                "Item 11.": "Executive Compensation",
                "Item 12.": "Security Ownership of Certain Beneficial Owners and Management",
                "Item 13.": "Certain Relationships and Related Transactions",
                "Item 14.": "Principal Accountant Fees and Services"
            }

            extracted_items = {}
            for item, description in governance_items.items():
                start_index = filing_text.find(item)
                if start_index != -1:
                    # Find the next item
                    next_item_index = float('inf')
                    for next_item in governance_items.keys():
                        if next_item > item:
                            next_index = filing_text.find(next_item, start_index + 1)
                            if next_index != -1:
                                next_item_index = min(next_item_index, next_index)
                    
                    end_index = next_item_index if next_item_index != float('inf') else len(filing_text)
                    content = filing_text[start_index:end_index].strip()
                    
                    extracted_items[item] = {
                        "description": description,
                        "content": content
                    }
            
            # Save to file for reference
            with open('corporate_governance.json', 'w') as f:
                json.dump({
                    "ticker": ticker,
                    "items": extracted_items
                }, f, indent=4)

            return {
                "ticker": ticker,
                "items": extracted_items
            }
        except Exception as e:
            return f"Error extracting corporate governance items: {str(e)}"
        


