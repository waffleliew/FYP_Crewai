def analyze_market_sentiment_serper(company_name):
    """
    Analyzes market sentiment about a company based on recent news headlines.
    The input to this tool should be a company name (e.g., NVIDIA, AMD).
    """
    try:
        url = "https://google.serper.dev/search"
        payload = json.dumps({
        "q": company_name
        })
        headers = {
        'X-API-KEY': os.environ.get("SERPER_API_KEY"),
        'Content-Type': 'application/json'
        }

        response = requests.request("POST", url, headers=headers, data=payload)
        results = response.json()
        print(result)

                    return results
    except Exception as e:
        re