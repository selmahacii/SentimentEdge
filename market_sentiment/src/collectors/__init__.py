"""
Data Collectors Package

This package contains data collection modules for:
- PriceCollector: Fetches OHLCV data from Yahoo Finance
- RedditCollector: Scrapes Reddit posts and comments
- TrendsCollector: Retrieves Google Trends search interest

All collectors:
- Cache data locally to minimize API calls
- Handle rate limiting gracefully
- Return standardized pandas DataFrames
- Use loguru for logging

Usage:
    from src.collectors import PriceCollector, RedditCollector, TrendsCollector
    
    # Price data
    price_collector = PriceCollector(['AAPL', 'TSLA'], './data/raw')
    prices = price_collector.fetch_all()
    
    # Reddit data
    reddit_collector = RedditCollector(client_id, secret, agent, './data/raw')
    posts = reddit_collector.search_ticker_mentions('AAPL', ['wallstreetbets'])
    
    # Trends data
    trends_collector = TrendsCollector('./data/raw')
    trends = trends_collector.fetch_interest('AAPL')
"""

from src.collectors.price_collector import PriceCollector
from src.collectors.reddit_collector import RedditCollector
from src.collectors.trends_collector import TrendsCollector

__all__ = [
    "PriceCollector",
    "RedditCollector", 
    "TrendsCollector",
]
