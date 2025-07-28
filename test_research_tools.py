from research_tools import ResearchTools
import json
from pprint import pprint
import os
def test_research_tools():
    # Initialize research tools
    tools = ResearchTools()
    
    # Test ticker
    # ticker = "PCAR"  # NVIDIA as an example
    
    print("\n=== Testing Research Tools ===\n")
    
    # # Test financial data
    # print("1. Testing Financial Data...")
    # financial_data = tools.search_financial_data(ticker)
    # print("Financial Data Response:")
    # pprint(financial_data)
    # print("\n" + "="*50 + "\n")
    
    # Test market sentiment
    # print("2. Testing Market Sentiment...")
    # sentiment = tools.analyze_market_sentiment_yahoo(ticker)
    # print("Market Sentiment Response:")
    # # pprint(sentiment)
    # # print("\n" + "="*50 + "\n")
    # os.makedirs('test_results', exist_ok=True)
    # with open('test_results/yahoo_market_sentimentanalysis.json', 'w') as f:
    #     json.dump(sentiment, f, indent=4)
    # Test industry comparison
    # print("3. Testing Industry Comparison...")
    # comparison = tools.compare_industry_metrics(ticker)
    # print("Industry Comparison Response:")
    # # pprint(comparison)
    # # print("\n" + "="*50 + "\n")
    # os.makedirs('test_results', exist_ok=True)
    # with open('test_results/compare_industry_metrics.json', 'w') as f:
    #     json.dump(comparison, f, indent=4)

    
    # # Test historical trends
    # print("4. Testing Historical Trends...")
    # trends = tools.analyze_historical_trends(ticker)
    # print("Historical Trends Response:")
    # pprint(trends)
    # print("\n" + "="*50 + "\n")
    
    # # Test competitors analysis
    # print("5. Testing Competitors Analysis...")
    # competitors = tools.analyze_competitors(ticker)
    # print("Competitors Analysis Response:")
    # # pprint(competitors)
    # # print("\n" + "="*50 + "\n")
    # os.makedirs('test_results', exist_ok=True)
    # with open('test_results/analyze_competitors.json', 'w') as f:
    #     json.dump(competitors, f, indent=4)

    
    # # Test financial ratios
    # print("6. Testing Financial Ratios...")
    # ratios = tools.analyze_financial_ratios(ticker)
    # print("Financial Ratios Response:")
    # pprint(ratios)
    # print("\n" + "="*50 + "\n")
    
    # # Test growth potential
    # print("7. Testing Growth Potential...")
    # growth = tools.evaluate_growth_potential(ticker)
    # print("Growth Potential Response:")
    # pprint(growth)
    # print("\n" + "="*50 + "\n")
    
    # # Test management quality
    # print("8. Testing Management Quality...")
    # management = tools.assess_management_quality(ticker)
    # print("Management Quality Response:")
    # pprint(management)
    # print("\n" + "="*50 + "\n")
    
    # # Test SEC filings
    # print("9. Testing SEC Business Overview...")
    # business = tools.sec_business_overview(ticker)
    # print("SEC Business Overview Response:")
    # # pprint(business)
    # # print("\n" + "="*50 + "\n")
    # # Save the business overview response to a JSON file
    # # Create test_results directory if it doesn't exist
    # os.makedirs('test_results', exist_ok=True)
    
    # with open('test_results/business_overview.json', 'w') as f:
    #     json.dump(business, f, indent=4)

    # print("10. Testing SEC Financial Statements...")
    # financials = tools.sec_financial_statements(ticker)
    # print("SEC Financial Statements Response:")
    # # pprint(financials)
    # # print("\n" + "="*50 + "\n")
    # os.makedirs('test_results', exist_ok=True)
    
    # with open('test_results/financial_statement.json', 'w') as f:
    #     json.dump(financials, f, indent=4)

    
    # print("11. Testing SEC MD&A...")
    # mda = tools.sec_mda(ticker)
    # print("SEC MD&A Response:")
    # # pprint(mda)
    # # print("\n" + "="*50 + "\n")
    # os.makedirs('test_results', exist_ok=True)
    # with open('test_results/MDA_overview.json', 'w') as f:
    #     json.dump(mda, f, indent=4)
    
    # print("12. Testing SEC Risk Factors...")
    # risks = tools.sec_risk_factors(ticker)
    # print("SEC Risk Factors Response:")
    # # pprint(risks)
    # # print("\n" + "="*50 + "\n")
    # os.makedirs('test_results', exist_ok=True)
    # with open('test_results/Risks_overview.json', 'w') as f:
    #     json.dump(risks, f, indent=4)
    
    # print("13. Testing SEC Market Risk...")
    # market_risk = tools.sec_market_risk(ticker)
    # print("SEC Market Risk Response:")
    # # pprint(market_risk)
    # # print("\n" + "="*50 + "\n")
    # os.makedirs('test_results', exist_ok=True)
    # with open('test_results/Risk_overview.json', 'w') as f:
    #     json.dump(market_risk, f, indent=4)
    
    # print("14. Testing SEC Corporate Governance...")
    # governance = tools.sec_corporate_governance(ticker)
    # print("SEC Corporate Governance Response:")
    # # pprint(governance)
    # # print("\n" + "="*50 + "\n")
    # os.makedirs('test_results', exist_ok=True)
    # with open('test_results/SEC_Gov_overview.json', 'w') as f:
    #     json.dump(governance, f, indent=4)


    date = "2023-12-31"  # Example date, adjust as needed
    print("\n=== Testing historicalfinancialdata ===\n")
    result = tools.historicalfinancialdata(ticker="FN", year="2021", quarter="Q2")
    print("historicalfinancialdata Response:")
    pprint(result)
    os.makedirs('test_results', exist_ok=True)
    with open('test_results/historicalfinancialdata.json', 'w') as f:
        json.dump(result, f, indent=4)


    # date = "2024-04-30"  # Example date, adjust as needed
    # print("\n=== Testing analyzemarketsentiment ===\n")
    # result = tools.analyzemarketsentiment(ticker="VSH", year="2021", quarter="Q2")
    # print("analyzemarketsentiment Response:")
    # pprint(result)
    # os.makedirs('test_results', exist_ok=True)
    # with open('test_results/analyzemarketsentiment.json', 'w') as f:
    #     json.dump(result, f, indent=4)

if __name__ == "__main__":
    test_research_tools()
