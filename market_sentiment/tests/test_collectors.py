#!/usr/bin/env python3
"""
Test script for data collectors.

Tests all three collectors:
1. PriceCollector - Yahoo Finance (free, no API key)
2. RedditCollector - Reddit API (requires credentials)
3. TrendsCollector - Google Trends (may simulate)

Run with:
    python test_collectors.py

Set environment variables for Reddit:
    REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, REDDIT_USER_AGENT
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from loguru import logger


def test_price_collector():
    """Test PriceCollector with Yahoo Finance data."""
    print("\n" + "=" * 60)
    print("TEST 1: PriceCollector (Yahoo Finance)")
    print("=" * 60)
    
    from src.collectors.price_collector import PriceCollector
    
    collector = PriceCollector(
        tickers=["AAPL", "TSLA"],
        cache_dir="./data/raw"
    )
    
    # Test single ticker
    print("\nFetching AAPL price data...")
    df = collector.fetch_prices("AAPL", period="1mo", use_cache=False)
    
    if df.empty:
        print("❌ Failed to fetch AAPL data")
        return False
    
    print(f"✓ Fetched {len(df)} days of AAPL data")
    print(f"  Columns: {df.columns.tolist()}")
    print(f"  Date range: {df['date'].min().date()} to {df['date'].max().date()}")
    
    # Test data quality
    assert "close" in df.columns, "Missing 'close' column"
    assert "daily_return" in df.columns, "Missing 'daily_return' column"
    assert df["daily_return"].notna().sum() > 0, "No valid daily returns"
    
    print("\nSample data:")
    print(df[["date", "close", "volume", "daily_return"]].head())
    
    # Test fetch all
    print("\nFetching all tickers...")
    all_prices = collector.fetch_all(period="1mo")
    print(f"✓ Fetched {len(all_prices)} tickers: {list(all_prices.keys())}")
    
    # Test market context
    print("\nFetching SPY market benchmark...")
    spy = collector.get_market_context(period="1mo")
    if not spy.empty:
        print(f"✓ Fetched {len(spy)} days of SPY data")
    
    # Test beta calculation
    print("\nCalculating AAPL beta...")
    beta = collector.calculate_beta("AAPL", market_data=spy)
    print(f"✓ AAPL beta: {beta:.2f}")
    
    print("\n✅ PriceCollector tests PASSED")
    return True


def test_reddit_collector():
    """Test RedditCollector (requires credentials)."""
    print("\n" + "=" * 60)
    print("TEST 2: RedditCollector (Reddit API)")
    print("=" * 60)
    
    import os
    
    # Check for credentials
    client_id = os.getenv("REDDIT_CLIENT_ID")
    client_secret = os.getenv("REDDIT_CLIENT_SECRET")
    user_agent = os.getenv("REDDIT_USER_AGENT", "test/1.0")
    
    if not client_id or not client_secret:
        print("⚠️ Reddit credentials not set")
        print("  Set REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, REDDIT_USER_AGENT")
        print("  Get credentials at: https://www.reddit.com/prefs/apps")
        print("\n⚠️ RedditCollector tests SKIPPED")
        return None
    
    from src.collectors.reddit_collector import RedditCollector
    
    collector = RedditCollector(
        client_id=client_id,
        client_secret=client_secret,
        user_agent=user_agent,
        cache_dir="./data/raw",
        rate_limit_delay=0.5
    )
    
    # Verify read-only mode
    if not collector.reddit.read_only:
        print("❌ Reddit not in read-only mode")
        return False
    
    print(f"✓ Reddit read-only mode: {collector.reddit.read_only}")
    
    # Test ticker search
    print("\nSearching for AAPL mentions...")
    posts = collector.search_ticker_mentions(
        "AAPL",
        ["wallstreetbets"],
        days_back=7,
        max_posts=5,
        use_cache=False
    )
    
    print(f"✓ Found {len(posts)} posts")
    
    if posts:
        print("\nSample post:")
        post = posts[0]
        print(f"  Title: {post['title'][:60]}...")
        print(f"  Score: {post['score']}")
        print(f"  Subreddit: {post['subreddit']}")
        print(f"  Comments: {len(post.get('comments', []))}")
    
    # Test daily aggregation
    print("\nAggregating to daily counts...")
    daily = collector.get_daily_mention_count(posts)
    if not daily.empty:
        print(f"✓ Generated {len(daily)} daily records")
        print(daily.head())
    
    print("\n✅ RedditCollector tests PASSED")
    return True


def test_trends_collector():
    """Test TrendsCollector (Google Trends or simulation)."""
    print("\n" + "=" * 60)
    print("TEST 3: TrendsCollector (Google Trends)")
    print("=" * 60)
    
    from src.collectors.trends_collector import TrendsCollector
    
    collector = TrendsCollector(
        cache_dir="./data/raw",
        rate_limit_delay=2.0
    )
    
    # Test single ticker
    print("\nFetching AAPL trends data...")
    df = collector.fetch_interest("AAPL", period_months=3, use_cache=False)
    
    if df.empty:
        print("❌ Failed to fetch AAPL trends")
        return False
    
    is_simulated = df["is_simulated"].iloc[0] if "is_simulated" in df.columns else False
    
    if is_simulated:
        print("⚠️ Using SIMULATED trends data (pytrends unavailable or rate limited)")
    else:
        print("✓ Using real Google Trends data")
    
    print(f"✓ Fetched {len(df)} days of trends data")
    print(f"  Date range: {df['date'].min().date()} to {df['date'].max().date()}")
    
    # Test statistics
    print("\nInterest statistics:")
    print(f"  Mean: {df['search_interest'].mean():.1f}")
    print(f"  Max: {df['search_interest'].max():.1f}")
    print(f"  Min: {df['search_interest'].min():.1f}")
    
    # Test interest change
    print("\nInterest change analysis:")
    change = collector.get_interest_change(df)
    print(f"  Recent avg: {change.get('recent_avg', 'N/A')}")
    print(f"  Change: {change.get('change_pct', 0):+.1f}%")
    print(f"  Trend: {change.get('trend', 'N/A')}")
    
    print("\n✅ TrendsCollector tests PASSED")
    return True


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("Market Sentiment Analyzer - Collector Tests")
    print("=" * 60)
    
    results = {}
    
    # Test PriceCollector
    try:
        results["price"] = test_price_collector()
    except Exception as e:
        print(f"\n❌ PriceCollector test FAILED: {e}")
        results["price"] = False
    
    # Test RedditCollector
    try:
        results["reddit"] = test_reddit_collector()
    except Exception as e:
        print(f"\n❌ RedditCollector test FAILED: {e}")
        results["reddit"] = False
    
    # Test TrendsCollector
    try:
        results["trends"] = test_trends_collector()
    except Exception as e:
        print(f"\n❌ TrendsCollector test FAILED: {e}")
        results["trends"] = False
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for v in results.values() if v is True)
    skipped = sum(1 for v in results.values() if v is None)
    failed = sum(1 for v in results.values() if v is False)
    
    print(f"\nPriceCollector:  {'✅ PASS' if results.get('price') else '❌ FAIL' if results.get('price') is False else '⚠️ SKIP'}")
    print(f"RedditCollector: {'✅ PASS' if results.get('reddit') else '❌ FAIL' if results.get('reddit') is False else '⚠️ SKIP'}")
    print(f"TrendsCollector: {'✅ PASS' if results.get('trends') else '❌ FAIL' if results.get('trends') is False else '⚠️ SKIP'}")
    
    print(f"\nTotal: {passed} passed, {skipped} skipped, {failed} failed")
    
    if failed > 0:
        print("\n❌ Some tests failed")
        return 1
    elif passed >= 2:  # At least price and trends should pass
        print("\n✅ Core tests passed")
        return 0
    else:
        print("\n⚠️ Limited tests run (check credentials)")
        return 0


if __name__ == "__main__":
    sys.exit(main())
