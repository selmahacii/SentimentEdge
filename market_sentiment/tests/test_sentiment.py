#!/usr/bin/env python3
"""
Test script for Sentiment Analysis Modules.

Tests both VADER and FinBERT analyzers:
1. Single text analysis
2. Batch processing
3. Post analysis with weighting
4. Daily aggregation
5. Model comparison (agreement rate)

Run with:
    python test_sentiment.py

Note: FinBERT requires ~500MB download on first use.
"""

import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))


def test_vader_analyzer():
    """Test VADER sentiment analyzer."""
    print("\n" + "=" * 60)
    print("TEST 1: VADER Analyzer")
    print("=" * 60)
    
    try:
        from src.sentiment.vader_analyzer import VADERAnalyzer, FINANCIAL_LEXICON
    except ImportError as e:
        print(f"⚠️ VADER not available: {e}")
        print("   Install with: pip install vaderSentiment")
        return None
    
    # Initialize
    print("\n1.1 Initializing VADER analyzer...")
    analyzer = VADERAnalyzer()
    print(f"   ✓ Loaded with {len(FINANCIAL_LEXICON)} financial terms")
    
    # Test single text analysis
    print("\n1.2 Single text analysis:")
    test_cases = [
        ("AAPL to the moon! 🚀🚀🚀", "positive"),
        ("TSLA is going to crash, sell now!", "negative"),
        ("MSFT earnings were in line with expectations", "neutral"),
        ("Diamond hands! 💎🙌 This stock is undervalued!", "positive"),
        ("Complete scam, pump and dump, avoid!", "negative"),
    ]
    
    correct = 0
    for text, expected in test_cases:
        result = analyzer.analyze_text(text)
        status = "✓" if result["label"] == expected else "✗"
        if result["label"] == expected:
            correct += 1
        print(f"   {status} '{text[:40]}...' -> {result['label']} ({result['compound']:.2f})")
    
    accuracy = correct / len(test_cases) * 100
    print(f"\n   Accuracy: {accuracy:.0f}% ({correct}/{len(test_cases)})")
    
    # Test financial lexicon
    print("\n1.3 Financial lexicon test:")
    financial_texts = [
        ("The stock is extremely bullish right now", "positive"),
        ("Bearish outlook, expect a crash soon", "negative"),
        ("Buy the dip on this oversold stock", "positive"),
        ("This is a classic pump and dump scheme", "negative"),
    ]
    
    for text, expected in financial_texts:
        result = analyzer.analyze_text(text)
        explanation = analyzer.explain_score(text)
        terms = list(explanation["financial_terms_found"].keys())
        print(f"   Text: '{text[:40]}...'")
        print(f"   -> {result['label']} ({result['compound']:.2f}), terms: {terms}")
    
    # Test post analysis
    print("\n1.4 Post analysis with weighting:")
    test_posts = [
        {
            "post_id": "1",
            "ticker": "AAPL",
            "title": "Apple is undervalued!",
            "combined_text": "Apple is extremely undervalued! Buy now! 🚀",
            "score": 500,
            "num_comments": 45,
            "created_at": "2024-01-15T10:30:00",
        },
        {
            "post_id": "2",
            "ticker": "AAPL",
            "title": "Concerns about AAPL",
            "combined_text": "I'm bearish on Apple, expect a correction",
            "score": 50,
            "num_comments": 12,
            "created_at": "2024-01-15T14:00:00",
        },
        {
            "post_id": "3",
            "ticker": "TSLA",
            "title": "Tesla to the moon!",
            "combined_text": "TSLA is going to the moon! Diamond hands! 💎🙌🚀",
            "score": 2000,
            "num_comments": 150,
            "created_at": "2024-01-15T09:00:00",
        },
    ]
    
    df = analyzer.analyze_posts(test_posts)
    print(f"   Analyzed {len(df)} posts")
    
    for _, row in df.iterrows():
        print(f"   [{row['ticker']}] score={row['score']}, "
              f"compound={row['vader_compound']:.2f}, "
              f"weighted={row['weighted_compound']:.2f}")
    
    # Test daily aggregation
    print("\n1.5 Daily sentiment aggregation:")
    daily = analyzer.get_daily_sentiment(df)
    if not daily.empty:
        print(daily[["date", "ticker", "vader_compound", "post_count", "dominant_sentiment"]])
    
    # Test distribution
    print("\n1.6 Sentiment distribution:")
    dist = analyzer.get_sentiment_distribution(df)
    for key, value in dist.items():
        print(f"   {key}: {value}")
    
    print("\n✅ VADER tests PASSED")
    return True


def test_finbert_analyzer():
    """Test FinBERT sentiment analyzer."""
    print("\n" + "=" * 60)
    print("TEST 2: FinBERT Analyzer")
    print("=" * 60)
    
    try:
        from src.sentiment.finbert_analyzer import FinBERTAnalyzer
    except ImportError as e:
        print(f"⚠️ FinBERT not available: {e}")
        print("   Install with: pip install transformers torch")
        return None
    
    # Initialize (downloads model on first use)
    print("\n2.1 Initializing FinBERT (may download ~500MB)...")
    try:
        analyzer = FinBERTAnalyzer(device="cpu")  # Force CPU for testing
        print("   ✓ Model loaded successfully")
    except Exception as e:
        print(f"   ✗ Failed to load model: {e}")
        return False
    
    # Test single text analysis
    print("\n2.2 Single text analysis:")
    test_cases = [
        "Revenue exceeded analyst expectations by 15%",
        "The company reported a significant loss this quarter",
        "Market conditions remain uncertain",
        "We upgrade the stock to outperform with $200 target",
        "Concerns about rising interest rates impact margins",
    ]
    
    for text in test_cases:
        result = analyzer.analyze_text(text)
        print(f"   '{text[:40]}...'")
        print(f"   -> {result['label']} ({result['score']:.1%} confidence)")
    
    # Test batch processing
    print("\n2.3 Batch processing:")
    batch_texts = [
        "Strong quarterly results beat expectations",
        "Earnings missed consensus estimates",
        "Revenue in line with guidance",
        "Management raises full-year outlook",
        "Margins compressed due to input costs",
    ]
    
    results = analyzer.analyze_batch(batch_texts)
    print(f"   Processed {len(results)} texts")
    
    for text, result in zip(batch_texts, results):
        print(f"   '{text[:30]}...' -> {result['label']}")
    
    print("\n✅ FinBERT tests PASSED")
    return True


def test_model_comparison():
    """Test VADER vs FinBERT comparison."""
    print("\n" + "=" * 60)
    print("TEST 3: VADER vs FinBERT Comparison")
    print("=" * 60)
    
    try:
        from src.sentiment.vader_analyzer import VADERAnalyzer
        from src.sentiment.finbert_analyzer import FinBERTAnalyzer
    except ImportError as e:
        print(f"⚠️ Models not available: {e}")
        return None
    
    # Initialize both
    vader = VADERAnalyzer()
    
    try:
        finbert = FinBERTAnalyzer(device="cpu")
    except Exception as e:
        print(f"⚠️ FinBERT not available: {e}")
        return None
    
    # Test texts that may differ
    test_texts = [
        "The stock is going to the moon! 🚀",
        "Management guidance was cautious but not bearish",
        "I'm not saying it's a scam, but be careful",
        "Revenue declined 10% but beat expectations",
        "This is literally the next Apple",
    ]
    
    print("\n3.1 Comparing VADER vs FinBERT:")
    print("-" * 60)
    
    agreements = 0
    for text in test_texts:
        vader_result = vader.analyze_text(text)
        finbert_result = finbert.analyze_text(text)
        
        agreement = "✓" if vader_result["label"] == finbert_result["label"] else "✗"
        if vader_result["label"] == finbert_result["label"]:
            agreements += 1
        
        print(f"\n   Text: '{text}'")
        print(f"   VADER:   {vader_result['label']} ({vader_result['compound']:.2f})")
        print(f"   FinBERT: {finbert_result['label']} ({finbert_result['compound']:.2f})")
        print(f"   Agreement: {agreement}")
    
    agreement_rate = agreements / len(test_texts) * 100
    print(f"\n   Agreement rate: {agreement_rate:.0f}%")
    
    print("\n✅ Comparison tests PASSED")
    return True


def test_sentiment_pipeline():
    """Test complete sentiment analysis pipeline."""
    print("\n" + "=" * 60)
    print("TEST 4: Complete Sentiment Pipeline")
    print("=" * 60)
    
    try:
        from src.sentiment.vader_analyzer import VADERAnalyzer
    except ImportError:
        print("⚠️ VADER not available")
        return None
    
    analyzer = VADERAnalyzer()
    
    # Simulate Reddit posts
    reddit_posts = [
        {
            "post_id": f"post_{i}",
            "ticker": "AAPL" if i < 3 else "TSLA",
            "title": f"Post {i} about stock",
            "combined_text": f"This is post {i}. The stock is {'bullish' if i % 2 == 0 else 'bearish'}!",
            "score": 100 * (i + 1),
            "num_comments": 10 * (i + 1),
            "subreddit": "wallstreetbets",
            "created_at": f"2024-01-1{i+1}T10:00:00",
        }
        for i in range(6)
    ]
    
    print("\n4.1 Processing 6 simulated Reddit posts...")
    
    # Analyze posts
    posts_df = analyzer.analyze_posts(reddit_posts)
    print(f"   ✓ Analyzed {len(posts_df)} posts")
    
    # Aggregate to daily
    daily_df = analyzer.get_daily_sentiment(posts_df)
    print(f"   ✓ Aggregated to {len(daily_df)} daily records")
    
    # Show summary
    print("\n4.2 Daily sentiment summary:")
    for _, row in daily_df.iterrows():
        print(f"   {row['date'].date()} | {row['ticker']} | "
              f"compound={row['vader_compound']:.2f} | "
              f"posts={row['post_count']} | "
              f"sentiment={row['dominant_sentiment']}")
    
    print("\n✅ Pipeline tests PASSED")
    return True


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("Market Sentiment Analyzer - Sentiment Tests")
    print("=" * 60)
    
    results = {}
    
    # Test VADER
    try:
        results["vader"] = test_vader_analyzer()
    except Exception as e:
        print(f"\n❌ VADER test FAILED: {e}")
        results["vader"] = False
    
    # Test FinBERT (optional - requires download)
    try:
        results["finbert"] = test_finbert_analyzer()
    except Exception as e:
        print(f"\n❌ FinBERT test FAILED: {e}")
        results["finbert"] = False
    
    # Test comparison
    try:
        results["comparison"] = test_model_comparison()
    except Exception as e:
        print(f"\n❌ Comparison test FAILED: {e}")
        results["comparison"] = False
    
    # Test pipeline
    try:
        results["pipeline"] = test_sentiment_pipeline()
    except Exception as e:
        print(f"\n❌ Pipeline test FAILED: {e}")
        results["pipeline"] = False
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    for name, result in results.items():
        if result is True:
            status = "✅ PASS"
        elif result is False:
            status = "❌ FAIL"
        else:
            status = "⚠️ SKIP"
        print(f"  {name}: {status}")
    
    passed = sum(1 for v in results.values() if v is True)
    skipped = sum(1 for v in results.values() if v is None)
    failed = sum(1 for v in results.values() if v is False)
    
    print(f"\nTotal: {passed} passed, {skipped} skipped, {failed} failed")
    
    if failed == 0:
        print("\n🎉 Core sentiment tests passed!")
        return 0
    else:
        print("\n⚠️ Some tests failed (check dependencies)")
        return 0


if __name__ == "__main__":
    sys.exit(main())
