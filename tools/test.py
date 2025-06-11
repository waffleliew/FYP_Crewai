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
from langchain.tools import tool
from firecrawl import FirecrawlApp
from gnews import GNews
from nltk.sentiment import SentimentIntensityAnalyzer
from dotenv import load_dotenv

load_dotenv()

def analyze_market_sentiment_yahoo(company_name):
    """
    Analyzes market sentiment about a company based on recent news headlines.
    The input to this tool should be a company name (e.g., NVIDIA, AMD).
    """
    try:
        # Use Yahoo Finance to get news
        if len(company_name.split()) > 1:
            # If company name has multiple words, try to find ticker
            search_term = company_name.replace(" ", "+")
            url = f"https://finance.yahoo.com/lookup?s={search_term}"
            response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
            soup = BeautifulSoup(response.text, 'html.parser')
            print("soup: ", soup)
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
        print("news: ", news)
        # Simple sentiment analysis based on keywords
        positive_keywords = ['surge', 'jump', 'rise', 'gain', 'positive', 'growth', 'profit', 'up', 'beat', 'strong', 'bullish']
        negative_keywords = ['drop', 'fall', 'decline', 'loss', 'negative', 'down', 'miss', 'weak', 'bearish', 'concern', 'risk']
        
        sentiment_scores = []
        recent_headlines = []
        
        for article in news[:10]:  # Analyze the 10 most recent news
            # Extract title from content if it exists, otherwise try direct title access
            headline = article.get('content', {}).get('title', '')
            recent_headlines.append(headline)

            # Calculate sentiment score
            score = 0
            for word in positive_keywords:
                if re.search(r'\b' + word + r'\b', headline.lower()):
                    score += 1
            for word in negative_keywords:
                if re.search(r'\b' + word + r'\b', headline.lower()):
                    score -= 1
            
            sentiment_scores.append(score)
        
        # Calculate overall sentiment
        avg_sentiment = sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0
        
        if avg_sentiment > 0.5:
            overall_sentiment = "Positive"
        elif avg_sentiment < -0.5:
            overall_sentiment = "Negative"
        else:
            overall_sentiment = "Neutral"
        
        # Normalize to a 0-1 scale
        normalized_score = (avg_sentiment + 3) / 6  # Assuming scores range from -3 to +3
        normalized_score = max(0, min(1, normalized_score))  # Clamp between 0 and 1
        
        return {
            "company": company_name,
            "ticker": ticker,
            "overall_sentiment": overall_sentiment,
            "sentiment_score": f"{normalized_score:.2f}",
            "recent_headlines": recent_headlines[:5],  # Return the 5 most recent headlines
            "last_updated": datetime.now().strftime("%Y-%m-%d")
        }
        
    except Exception as e:
        return f"Error analyzing market sentiment for {company_name}: {str(e)}"


    
@tool("Analyze market sentiment from Serper")
def analyze_market_sentiment_serper(company_name):
    """
    Analyzes market sentiment about a company based on recent news headlines.
    The input to this tool should be a company name (e.g., NVIDIA, AMD).
    
    Returns a structured list of news items (title, snippet, source, date) from topStories and organic results.
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
                "date": story.get("date")
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
                "date": item.get("date")
            }
            for item in organic
            if item.get("title") and item.get("snippet")
        ]

        # Merge and return the processed articles
        combined_articles = top_items + organic_items
        return {
            "company": company_name,
            "article_count": len(combined_articles),
            "articles": combined_articles
        }

    except Exception as e:
        return {
            "error": f"Error analyzing market sentiment for {company_name}: {str(e)}"
        }


if __name__ == "__main__":
    # Test the function with a sample company
    test_company = "AAPL"
    result = analyze_market_sentiment_serper(test_company)
    print(result)
    # result = analyze_market_sentiment_yahoo(test_company)

    # print(f"\nMarket Sentiment Analysis for {test_company}:")
    # print(f"Ticker: {result['ticker']}")
    # print(f"Overall Sentiment: {result['overall_sentiment']}")
    # print(f"Sentiment Score: {result['sentiment_score']}")
    # print("\nRecent Headlines:")
    # for headline in result['recent_headlines']:
    #     print(f"- {headline}")

    # print(f"\nLast Updated: {result['last_updated']}")
