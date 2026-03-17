#!/usr/bin/env python3
"""
Test script for Storage Module.

Tests all database operations:
1. Database initialization
2. Price data save/load
3. Reddit posts save/load
4. Sentiment data save/load
5. Trends data save/load
6. Feature matrix save/load
7. Query operations

Run with:
    python test_storage.py
"""

import sys
import os
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))


def test_database_initialization():
    """Test database and table creation."""
    print("\n" + "=" * 60)
    print("TEST 1: Database Initialization")
    print("=" * 60)
    
    from src.storage import init_db, get_data_summary
    
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        
        # Initialize database
        db = init_db(db_path)
        
        # Verify file created
        assert os.path.exists(db_path), "Database file not created"
        print(f"✓ Database created: {db_path}")
        
        # Verify tables exist
        summary = get_data_summary(db)
        assert isinstance(summary, dict), "Summary should be a dict"
        print(f"✓ Tables created: {list(summary.keys())}")
        
        # Check initial counts are zero
        for table, info in summary.items():
            assert info["count"] == 0, f"{table} should be empty initially"
        print("✓ All tables empty initially")
        
    print("\n✅ Database initialization test PASSED")
    return True


def test_price_data_operations():
    """Test price data save and load operations."""
    print("\n" + "=" * 60)
    print("TEST 2: Price Data Operations")
    print("=" * 60)
    
    import pandas as pd
    from src.storage import init_db, save_prices, load_price_history, get_latest_price
    
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        db = init_db(db_path)
        
        # Create test data
        test_data = pd.DataFrame({
            "date": [datetime.now() - timedelta(days=i) for i in range(5, 0, -1)],
            "ticker": ["AAPL"] * 5,
            "open": [175.0, 176.0, 177.0, 178.0, 179.0],
            "high": [176.5, 177.5, 178.5, 179.5, 180.5],
            "low": [174.5, 175.5, 176.5, 177.5, 178.5],
            "close": [176.0, 177.0, 178.0, 179.0, 180.0],
            "volume": [50000000, 48000000, 52000000, 55000000, 49000000],
            "daily_return": [None, 0.0057, 0.0056, 0.0056, 0.0056],
            "log_return": [None, 0.0057, 0.0056, 0.0056, 0.0056],
            "volatility_10d": [None, None, None, None, 0.0056],
        })
        
        # Save data
        count = save_prices(test_data, db)
        assert count == 5, f"Expected 5 records saved, got {count}"
        print(f"✓ Saved {count} price records")
        
        # Load data back
        loaded = load_price_history(db, "AAPL")
        assert len(loaded) == 5, f"Expected 5 records loaded, got {len(loaded)}"
        print(f"✓ Loaded {len(loaded)} price records")
        
        # Verify data integrity
        assert "close" in loaded.columns, "Missing 'close' column"
        assert "daily_return" in loaded.columns, "Missing 'daily_return' column"
        assert loaded["close"].iloc[-1] == 180.0, "Data mismatch"
        print("✓ Data integrity verified")
        
        # Test get_latest_price
        latest = get_latest_price(db, "AAPL")
        assert latest is not None, "Latest price should not be None"
        assert latest["close"] == 180.0, "Latest close mismatch"
        print(f"✓ Latest price: ${latest['close']}")
        
        # Test upsert (update existing records)
        updated_data = test_data.copy()
        updated_data["close"] = updated_data["close"] + 1.0  # Increment all closes
        
        count = save_prices(updated_data, db)
        # Should update, not insert new
        loaded = load_price_history(db, "AAPL")
        assert len(loaded) == 5, "Should still have 5 records after upsert"
        assert loaded["close"].iloc[-1] == 181.0, "Update not applied"
        print("✓ Upsert (update) working correctly")
        
    print("\n✅ Price data operations test PASSED")
    return True


def test_reddit_posts_operations():
    """Test Reddit posts save operations."""
    print("\n" + "=" * 60)
    print("TEST 3: Reddit Posts Operations")
    print("=" * 60)
    
    from src.storage import init_db, save_reddit_posts, DatabaseManager
    from src.storage import RedditPost
    
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        db = init_db(db_path)
        
        # Create test posts
        test_posts = [
            {
                "post_id": "test123",
                "ticker": "AAPL",
                "title": "Apple is going to the moon! 🚀",
                "selftext": "Buy AAPL now, it's going to skyrocket!",
                "combined_text": "Apple is going to the moon! 🚀 Buy AAPL now, it's going to skyrocket!",
                "score": 150,
                "upvote_ratio": 0.95,
                "num_comments": 45,
                "subreddit": "wallstreetbets",
                "url": "https://reddit.com/r/wallstreetbets/comments/test123",
                "created_at": datetime.now().isoformat(),
                "comments": [
                    {
                        "comment_id": "comment1",
                        "body": "To the moon!",
                        "score": 50,
                        "created_at": datetime.now().isoformat(),
                    }
                ]
            },
            {
                "post_id": "test456",
                "ticker": "TSLA",
                "title": "TSLA bearish outlook",
                "selftext": "I think TSLA is overvalued",
                "combined_text": "TSLA bearish outlook I think TSLA is overvalued",
                "score": 80,
                "upvote_ratio": 0.75,
                "num_comments": 25,
                "subreddit": "investing",
                "url": "https://reddit.com/r/investing/comments/test456",
                "created_at": datetime.now().isoformat(),
                "comments": []
            }
        ]
        
        # Save posts
        count = save_reddit_posts(test_posts, db)
        assert count == 2, f"Expected 2 posts saved, got {count}"
        print(f"✓ Saved {count} Reddit posts")
        
        # Verify posts saved
        with db.session() as session:
            posts = session.query(RedditPost).all()
            assert len(posts) == 2, f"Expected 2 posts in DB, got {len(posts)}"
            print(f"✓ Verified {len(posts)} posts in database")
            
            # Check combined_text
            aapl_post = [p for p in posts if p.ticker == "AAPL"][0]
            assert "moon" in aapl_post.combined_text.lower(), "Combined text missing"
            print("✓ Combined text stored correctly")
        
        # Test duplicate handling (upsert)
        test_posts[0]["score"] = 200  # Update score
        count = save_reddit_posts(test_posts, db)
        
        with db.session() as session:
            posts = session.query(RedditPost).all()
            assert len(posts) == 2, "Should still have 2 posts (no duplicates)"
            updated_post = [p for p in posts if p.post_id == "test123"][0]
            assert updated_post.score == 200, "Score not updated"
        print("✓ Duplicate handling (upsert) working")
        
    print("\n✅ Reddit posts operations test PASSED")
    return True


def test_sentiment_data_operations():
    """Test sentiment data save and load operations."""
    print("\n" + "=" * 60)
    print("TEST 4: Sentiment Data Operations")
    print("=" * 60)
    
    import pandas as pd
    from src.storage import init_db, save_daily_sentiment, load_sentiment_history
    
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        db = init_db(db_path)
        
        # Create test sentiment data
        test_sentiment = pd.DataFrame({
            "date": [datetime.now() - timedelta(days=i) for i in range(3, 0, -1)],
            "ticker": ["TSLA"] * 3,
            "post_count": [15, 12, 8],
            "avg_score": [150.5, 120.3, 85.2],
            "total_comments": [45, 36, 24],
            "vader_compound": [0.35, 0.28, 0.15],
            "vader_positive": [0.45, 0.38, 0.30],
            "vader_negative": [0.10, 0.12, 0.15],
            "vader_neutral": [0.45, 0.50, 0.55],
            "finbert_positive": [0.55, 0.48, 0.40],
            "finbert_negative": [0.15, 0.22, 0.30],
            "finbert_neutral": [0.30, 0.30, 0.30],
            "weighted_compound": [0.42, 0.32, 0.18],
            "dominant_sentiment": ["positive", "positive", "neutral"],
        })
        
        # Save sentiment
        count = save_daily_sentiment(test_sentiment, db)
        assert count == 3, f"Expected 3 records saved, got {count}"
        print(f"✓ Saved {count} sentiment records")
        
        # Load back
        loaded = load_sentiment_history(db, "TSLA")
        assert len(loaded) == 3, f"Expected 3 records loaded, got {len(loaded)}"
        print(f"✓ Loaded {len(loaded)} sentiment records")
        
        # Verify VADER compound
        assert "vader_compound" in loaded.columns, "Missing vader_compound"
        assert loaded["vader_compound"].iloc[-1] == 0.35, "VADER compound mismatch"
        print("✓ VADER sentiment stored correctly")
        
        # Verify dominant sentiment
        sentiments = loaded["dominant_sentiment"].tolist()
        assert "positive" in sentiments, "Missing 'positive' sentiment"
        print(f"✓ Sentiment labels: {sentiments}")
        
    print("\n✅ Sentiment data operations test PASSED")
    return True


def test_trends_data_operations():
    """Test Google Trends data save and load."""
    print("\n" + "=" * 60)
    print("TEST 5: Trends Data Operations")
    print("=" * 60)
    
    import pandas as pd
    from src.storage import init_db, save_trends, load_trends
    
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        db = init_db(db_path)
        
        # Create test trends data
        test_trends = pd.DataFrame({
            "date": [datetime.now() - timedelta(days=i) for i in range(7, 0, -1)],
            "ticker": ["NVDA"] * 7,
            "search_interest": [55.0, 48.0, 42.0, 38.0, 45.0, 52.0, 60.0],
            "is_simulated": [False] * 7,
        })
        
        # Save trends
        count = save_trends(test_trends, db)
        assert count == 7, f"Expected 7 records saved, got {count}"
        print(f"✓ Saved {count} trend records")
        
        # Load back
        loaded = load_trends(db, "NVDA")
        assert len(loaded) == 7, f"Expected 7 records loaded, got {len(loaded)}"
        print(f"✓ Loaded {len(loaded)} trend records")
        
        # Verify search interest values
        assert "search_interest" in loaded.columns, "Missing search_interest"
        assert loaded["search_interest"].max() <= 100, "Interest should be <= 100"
        print(f"✓ Search interest range: {loaded['search_interest'].min():.0f} - {loaded['search_interest'].max():.0f}")
        
    print("\n✅ Trends data operations test PASSED")
    return True


def test_feature_matrix_operations():
    """Test feature matrix save and load."""
    print("\n" + "=" * 60)
    print("TEST 6: Feature Matrix Operations")
    print("=" * 60)
    
    import pandas as pd
    from src.storage import init_db, save_features, load_feature_matrix
    
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        db = init_db(db_path)
        
        # Create test feature matrix
        test_features = pd.DataFrame({
            "date": [datetime.now() - timedelta(days=i) for i in range(5, 0, -1)],
            "ticker": ["MSFT"] * 5,
            "close": [380.0, 382.0, 385.0, 383.0, 388.0],
            "volume": [20000000, 21000000, 19500000, 22000000, 20500000],
            "daily_return": [None, 0.0053, 0.0079, -0.0052, 0.0131],
            "volatility_10d": [None, None, None, 0.006, 0.008],
            "rsi_14": [55.0, 58.0, 62.0, 60.0, 65.0],
            "macd": [1.5, 1.8, 2.1, 1.9, 2.3],
            "vader_compound": [0.25, 0.30, 0.28, 0.35, 0.40],
            "sentiment_lag1": [None, 0.25, 0.30, 0.28, 0.35],
            "search_interest": [45.0, 48.0, 52.0, 50.0, 55.0],
            "target_label": [1, 1, 0, 1, None],  # Last one is prediction target
            "target_return": [0.0053, 0.0079, -0.0052, 0.0131, None],
            "post_count": [10, 12, 8, 15, 11],
            "total_comments": [30, 36, 24, 45, 33],
        })
        
        # Save features
        count = save_features(test_features, db)
        assert count == 5, f"Expected 5 records saved, got {count}"
        print(f"✓ Saved {count} feature records")
        
        # Load back
        loaded = load_feature_matrix(db, ticker="MSFT")
        assert len(loaded) == 5, f"Expected 5 records loaded, got {len(loaded)}"
        print(f"✓ Loaded {len(loaded)} feature records")
        
        # Verify columns
        assert "target_label" in loaded.columns, "Missing target_label"
        assert "vader_compound" in loaded.columns, "Missing vader_compound"
        assert "sentiment_lag1" in loaded.columns, "Missing sentiment_lag1"
        print("✓ All key feature columns present")
        
        # Load all tickers
        all_features = load_feature_matrix(db)
        assert len(all_features) == 5, "Should load all records"
        print("✓ Can load features for all tickers")
        
    print("\n✅ Feature matrix operations test PASSED")
    return True


def test_database_summary():
    """Test database summary function."""
    print("\n" + "=" * 60)
    print("TEST 7: Database Summary")
    print("=" * 60)
    
    import pandas as pd
    from src.storage import (
        init_db, get_data_summary,
        save_prices, save_daily_sentiment, save_trends
    )
    
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        db = init_db(db_path)
        
        # Add some data
        save_prices(pd.DataFrame({
            "date": [datetime.now()], "ticker": ["AAPL"],
            "open": [180], "high": [181], "low": [179],
            "close": [180.5], "volume": [50000000]
        }), db)
        
        save_daily_sentiment(pd.DataFrame({
            "date": [datetime.now()], "ticker": ["AAPL"],
            "post_count": [10], "avg_score": [100], "total_comments": [30],
            "vader_compound": [0.3], "vader_positive": [0.4],
            "vader_negative": [0.1], "dominant_sentiment": ["positive"]
        }), db)
        
        # Get summary
        summary = get_data_summary(db)
        
        assert "price_data" in summary, "Missing price_data in summary"
        assert "sentiment_data" in summary, "Missing sentiment_data in summary"
        assert summary["price_data"]["count"] >= 1, "Price count should be >= 1"
        
        print("Database Summary:")
        for table, info in summary.items():
            print(f"  {table}: {info}")
        
        print("\n✅ Database summary test PASSED")
        return True


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("Market Sentiment Analyzer - Storage Tests")
    print("=" * 60)
    
    tests = [
        ("Database Initialization", test_database_initialization),
        ("Price Data Operations", test_price_data_operations),
        ("Reddit Posts Operations", test_reddit_posts_operations),
        ("Sentiment Data Operations", test_sentiment_data_operations),
        ("Trends Data Operations", test_trends_data_operations),
        ("Feature Matrix Operations", test_feature_matrix_operations),
        ("Database Summary", test_database_summary),
    ]
    
    results = {}
    
    for name, test_func in tests:
        try:
            results[name] = test_func()
        except Exception as e:
            print(f"\n❌ {name} FAILED: {e}")
            results[name] = False
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for v in results.values() if v)
    failed = sum(1 for v in results.values() if not v)
    
    for name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {name}: {status}")
    
    print(f"\nTotal: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("\n🎉 All storage tests PASSED!")
        return 0
    else:
        print("\n❌ Some tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
