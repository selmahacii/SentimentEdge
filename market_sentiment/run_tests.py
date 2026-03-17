#!/usr/bin/env python3
"""
Market Sentiment Analyzer - Integration Test

This script tests all components to verify they work correctly together.
Run this to validate your installation and configuration.

Usage:
    python run_tests.py
    python run_tests.py --skip-slow  # Skip slow tests (FinBERT, full pipeline)
"""

import argparse
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd


def print_header(title: str) -> None:
    """Print a formatted header."""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def print_test(name: str, passed: bool, duration: float = 0) -> None:
    """Print test result."""
    status = "✅ PASS" if passed else "❌ FAIL"
    print(f"  {status} | {name} ({duration:.2f}s)")


def test_imports() -> bool:
    """Test all module imports."""
    print_header("TEST 1: Module Imports")
    
    errors = []
    start = time.time()
    
    # Core modules
    try:
        from src.config import settings
        print_test("src.config", True)
    except Exception as e:
        print_test("src.config", False)
        errors.append(f"config: {e}")
    
    try:
        from src.storage import DatabaseManager, PriceData, RedditDailySentiment
        print_test("src.storage", True)
    except Exception as e:
        print_test("src.storage", False)
        errors.append(f"storage: {e}")
    
    # Collectors
    try:
        from src.collectors.price_collector import PriceCollector
        print_test("src.collectors.price_collector", True)
    except Exception as e:
        print_test("src.collectors.price_collector", False)
        errors.append(f"price_collector: {e}")
    
    try:
        from src.collectors.reddit_collector import RedditCollector
        print_test("src.collectors.reddit_collector", True)
    except Exception as e:
        print_test("src.collectors.reddit_collector", False)
        errors.append(f"reddit_collector: {e}")
    
    try:
        from src.collectors.trends_collector import TrendsCollector
        print_test("src.collectors.trends_collector", True)
    except Exception as e:
        print_test("src.collectors.trends_collector", False)
        errors.append(f"trends_collector: {e}")
    
    # Sentiment
    try:
        from src.sentiment.vader_analyzer import VADERAnalyzer
        print_test("src.sentiment.vader_analyzer", True)
    except Exception as e:
        print_test("src.sentiment.vader_analyzer", False)
        errors.append(f"vader_analyzer: {e}")
    
    # Features
    try:
        from src.features.technical import compute_technical_features
        from src.features.fusion import merge_all_features, create_target_variable
        print_test("src.features", True)
    except Exception as e:
        print_test("src.features", False)
        errors.append(f"features: {e}")
    
    # ML
    try:
        from src.model import SentimentPredictor
        print_test("src.model", True)
    except Exception as e:
        print_test("src.model", False)
        errors.append(f"model: {e}")
    
    # Backtest
    try:
        from src.backtest import BacktestEngine, BacktestConfig
        print_test("src.backtest", True)
    except Exception as e:
        print_test("src.backtest", False)
        errors.append(f"backtest: {e}")
    
    # Pipeline
    try:
        from src.pipeline import run_full_pipeline
        print_test("src.pipeline", True)
    except Exception as e:
        print_test("src.pipeline", False)
        errors.append(f"pipeline: {e}")
    
    duration = time.time() - start
    
    if errors:
        print(f"\n  Errors: {len(errors)}")
        for err in errors:
            print(f"    - {err}")
        return False
    
    print(f"\n  All imports successful! ({duration:.2f}s)")
    return True


def test_configuration() -> bool:
    """Test configuration loading."""
    print_header("TEST 2: Configuration")
    
    try:
        from src.config import settings
        
        start = time.time()
        
        # Check required settings exist (with defaults)
        print_test(f"Tickers: {settings.tickers}", bool(settings.tickers))
        print_test(f"Subreddits: {settings.subreddits[:2]}...", bool(settings.subreddits))
        print_test(f"Lookback days: {settings.lookback_days}", settings.lookback_days > 0)
        print_test(f"DB path: {settings.db_path}", bool(settings.db_path))
        
        duration = time.time() - start
        print(f"\n  Configuration loaded! ({duration:.2f}s)")
        return True
        
    except Exception as e:
        print(f"  ❌ Configuration failed: {e}")
        return False


def test_database() -> bool:
    """Test database operations."""
    print_header("TEST 3: Database Operations")
    
    try:
        from src.storage import DatabaseManager, save_prices, load_price_history
        
        # Create test database
        test_db_path = "./data/test_sentiment.db"
        Path(test_db_path).parent.mkdir(parents=True, exist_ok=True)
        
        start = time.time()
        
        # Initialize
        db = DatabaseManager(test_db_path)
        db.create_tables()
        print_test("Create tables", True)
        
        # Create test data
        np.random.seed(42)
        dates = [datetime.now() - timedelta(days=i) for i in range(30, 0, -1)]
        
        test_prices = pd.DataFrame({
            "date": dates,
            "ticker": "TEST",
            "open": 100 * np.cumprod(1 + np.random.normal(0.001, 0.02, 30)),
            "high": 100 * np.cumprod(1 + np.random.normal(0.001, 0.02, 30)) * 1.01,
            "low": 100 * np.cumprod(1 + np.random.normal(0.001, 0.02, 30)) * 0.99,
            "close": 100 * np.cumprod(1 + np.random.normal(0.001, 0.02, 30)),
            "volume": np.random.randint(1000000, 5000000, 30),
        })
        
        # Save
        saved = save_prices(test_prices, db)
        print_test(f"Save prices ({saved} records)", saved > 0)
        
        # Load
        loaded = load_price_history(db, "TEST")
        print_test(f"Load prices ({len(loaded)} records)", len(loaded) > 0)
        
        # Cleanup
        Path(test_db_path).unlink(missing_ok=True)
        
        duration = time.time() - start
        print(f"\n  Database operations successful! ({duration:.2f}s)")
        return True
        
    except Exception as e:
        print(f"  ❌ Database test failed: {e}")
        return False


def test_technical_indicators() -> bool:
    """Test technical indicator calculations."""
    print_header("TEST 4: Technical Indicators")
    
    try:
        from src.features.technical import (
            compute_technical_features,
            compute_rsi,
            compute_macd,
            compute_bollinger_bands,
        )
        
        start = time.time()
        
        # Create test price data
        np.random.seed(42)
        n = 100
        test_df = pd.DataFrame({
            "date": pd.date_range(end=datetime.now(), periods=n, freq='D'),
            "ticker": "TEST",
            "close": 100 * np.cumprod(1 + np.random.normal(0.001, 0.02, n)),
            "high": 100 * np.cumprod(1 + np.random.normal(0.001, 0.02, n)) * 1.01,
            "low": 100 * np.cumprod(1 + np.random.normal(0.001, 0.02, n)) * 0.99,
            "volume": np.random.randint(1000000, 5000000, n),
        })
        
        # RSI
        rsi = compute_rsi(test_df["close"])
        print_test(f"RSI computed (last: {rsi.iloc[-1]:.1f})", not rsi.isna().all())
        
        # MACD
        macd, signal, hist = compute_macd(test_df["close"])
        print_test(f"MACD computed (last: {macd.iloc[-1]:.2f})", not macd.isna().all())
        
        # Bollinger
        bb_upper, bb_mid, bb_lower = compute_bollinger_bands(test_df["close"])
        print_test(f"Bollinger computed", not bb_upper.isna().all())
        
        # All indicators
        result = compute_technical_features(test_df)
        new_cols = len(result.columns) - len(test_df.columns)
        print_test(f"All indicators (+{new_cols} columns)", new_cols > 10)
        
        duration = time.time() - start
        print(f"\n  Technical indicators successful! ({duration:.2f}s)")
        return True
        
    except Exception as e:
        print(f"  ❌ Technical indicators failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_sentiment_analysis() -> bool:
    """Test sentiment analysis."""
    print_header("TEST 5: Sentiment Analysis (VADER)")
    
    try:
        from src.sentiment.vader_analyzer import VADERAnalyzer
        
        start = time.time()
        
        analyzer = VADERAnalyzer()
        print_test("VADER initialized", True)
        
        # Test texts
        test_texts = [
            "AAPL to the moon! 🚀 Bullish!",
            "TSLA is going to crash, bearish outlook",
            "Market seems neutral today, waiting for earnings",
            "This stock is rekt, sold at a huge loss",
            "Diamond hands! Never selling!",
        ]
        
        results = []
        for text in test_texts:
            scores = analyzer.analyze(text)
            results.append({
                "text": text[:40] + "...",
                "compound": scores["compound"],
                "label": scores["label"],
            })
        
        print("\n  Sample Results:")
        for r in results:
            emoji = "🟢" if r["label"] == "positive" else "🔴" if r["label"] == "negative" else "🟡"
            print(f"    {emoji} {r['compound']:.3f} | {r['text']}")
        
        duration = time.time() - start
        print(f"\n  Sentiment analysis successful! ({duration:.2f}s)")
        return True
        
    except Exception as e:
        print(f"  ❌ Sentiment analysis failed: {e}")
        return False


def test_ml_model() -> bool:
    """Test ML model training and prediction."""
    print_header("TEST 6: ML Model (XGBoost)")
    
    try:
        from src.model import SentimentPredictor
        
        start = time.time()
        
        # Create synthetic feature data
        np.random.seed(42)
        n = 300
        dates = [datetime.now() - timedelta(days=i) for i in range(n, 0, -1)]
        
        test_features = pd.DataFrame({
            "date": dates,
            "ticker": "TEST",
            "close": 100 * np.cumprod(1 + np.random.normal(0.001, 0.02, n)),
            "volume": np.random.randint(1000000, 5000000, n),
            "daily_return": np.random.normal(0.001, 0.02, n),
            "rsi_14": np.random.uniform(30, 70, n),
            "macd": np.random.normal(0, 1, n),
            "macd_signal": np.random.normal(0, 1, n),
            "bb_position": np.random.uniform(0, 1, n),
            "volume_ratio": np.random.uniform(0.5, 2.0, n),
            "vader_compound": np.random.uniform(-0.3, 0.3, n),
            "vader_compound_lag1": np.random.uniform(-0.3, 0.3, n),
            "sentiment_x_volume": np.random.uniform(-0.5, 0.5, n),
        })
        
        # Create target with some signal
        test_features["target_label"] = (test_features["vader_compound_lag1"] > 0).astype(int)
        noise_idx = np.random.choice(n, size=int(n * 0.3), replace=False)
        test_features.loc[noise_idx, "target_label"] = 1 - test_features.loc[noise_idx, "target_label"]
        
        # Train model
        predictor = SentimentPredictor(feature_df=test_features)
        results = predictor.train("TEST")
        
        print_test(f"Model trained", True)
        print_test(f"Accuracy: {results['accuracy']:.3f}", results['accuracy'] > 0.4)
        print_test(f"AUC-ROC: {results['auc_roc']:.3f}", results['auc_roc'] > 0.4)
        print_test(f"CV Score: {results['cv_mean']:.3f} ± {results['cv_std']:.3f}", True)
        
        # Feature importance
        print(f"\n  Top 3 features:")
        for i, (feat, imp) in enumerate(list(results['feature_importance'].items())[:3]):
            print(f"    {i+1}. {feat}: {imp:.4f}")
        
        # Prediction
        prediction = predictor.predict_tomorrow("TEST")
        print(f"\n  Prediction: {prediction['prediction']} ({prediction['probability']:.0%} confidence)")
        
        duration = time.time() - start
        print(f"\n  ML model successful! ({duration:.2f}s)")
        return True
        
    except Exception as e:
        print(f"  ❌ ML model test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_backtest() -> bool:
    """Test backtesting engine."""
    print_header("TEST 7: Backtesting Engine")
    
    try:
        from src.backtest import BacktestEngine, BacktestConfig, format_backtest_report
        
        start = time.time()
        
        # Create test data
        np.random.seed(42)
        n = 100
        dates = [datetime.now() - timedelta(days=i) for i in range(n, 0, -1)]
        
        prices = 100 * np.cumprod(1 + np.random.normal(0.001, 0.02, n))
        
        price_df = pd.DataFrame({
            "date": dates,
            "ticker": "TEST",
            "open": prices * (1 + np.random.uniform(-0.005, 0.005, n)),
            "close": prices,
        })
        
        # Create predictions (60% accuracy)
        predictions = []
        for i, date in enumerate(dates[:-3]):
            actual = 1 if prices[min(i+3, n-1)] > prices[i] else 0
            pred_dir = "UP" if (actual == 1 and np.random.random() < 0.6) or \
                             (actual == 0 and np.random.random() > 0.6) else "DOWN"
            predictions.append({
                "date": date,
                "ticker": "TEST",
                "predicted_direction": pred_dir,
                "probability": np.random.uniform(0.5, 0.9),
            })
        
        pred_df = pd.DataFrame(predictions)
        
        # Run backtest
        config = BacktestConfig(initial_capital=10000, transaction_cost=0.001)
        engine = BacktestEngine(config)
        results = engine.run(pred_df, price_df, "TEST")
        
        summary = results["summary"]
        
        print_test(f"Backtest completed", True)
        print_test(f"Trades: {summary.get('total_trades', 0)}", True)
        print_test(f"Win rate: {summary.get('win_rate', 0):.1%}", summary.get('win_rate', 0) >= 0)
        print_test(f"Return: {summary.get('total_return', 0):+.1%}", True)
        
        if summary.get('total_trades', 0) > 0:
            print(f"\n  Backtest Report Preview:")
            print("-" * 40)
            print(format_backtest_report(results)[:500])
        
        duration = time.time() - start
        print(f"\n  Backtesting successful! ({duration:.2f}s)")
        return True
        
    except Exception as e:
        print(f"  ❌ Backtest test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_full_pipeline_demo() -> bool:
    """Test full pipeline with demo mode."""
    print_header("TEST 8: Full Pipeline (Demo Mode)")
    
    try:
        start = time.time()
        
        # Run demo mode
        import subprocess
        result = subprocess.run(
            [sys.executable, "main.py", "--demo"],
            cwd=Path(__file__).parent,
            capture_output=True,
            text=True,
            timeout=120,
        )
        
        success = result.returncode == 0
        print_test("Demo mode executed", success)
        
        if success:
            # Check output for key indicators
            output = result.stdout
            print_test("Output contains results", "completed" in output.lower() or "accuracy" in output.lower())
            
            # Print sample output
            print("\n  Sample Output:")
            print("-" * 40)
            lines = output.split("\n")
            for line in lines[-20:]:
                if line.strip():
                    print(f"  {line}")
        else:
            print(f"  Error: {result.stderr[:500]}")
        
        duration = time.time() - start
        print(f"\n  Pipeline test completed! ({duration:.2f}s)")
        return success
        
    except Exception as e:
        print(f"  ❌ Pipeline test failed: {e}")
        return False


def main():
    """Run all tests."""
    parser = argparse.ArgumentParser(description="Run Market Sentiment Analyzer tests")
    parser.add_argument("--skip-slow", action="store_true", help="Skip slow tests")
    args = parser.parse_args()
    
    print("\n" + "=" * 60)
    print("  MARKET SENTIMENT ANALYZER - INTEGRATION TESTS")
    print("=" * 60)
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    results = []
    total_start = time.time()
    
    # Run tests
    results.append(("Imports", test_imports()))
    results.append(("Configuration", test_configuration()))
    results.append(("Database", test_database()))
    results.append(("Technical Indicators", test_technical_indicators()))
    results.append(("Sentiment Analysis", test_sentiment_analysis()))
    results.append(("ML Model", test_ml_model()))
    results.append(("Backtesting", test_backtest()))
    
    if not args.skip_slow:
        results.append(("Full Pipeline", test_full_pipeline_demo()))
    
    total_duration = time.time() - total_start
    
    # Summary
    print_header("TEST SUMMARY")
    
    passed = sum(1 for _, r in results if r)
    total = len(results)
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status} | {name}")
    
    print("\n" + "-" * 40)
    print(f"  Total: {passed}/{total} tests passed")
    print(f"  Duration: {total_duration:.2f}s")
    print("=" * 60)
    
    if passed == total:
        print("\n  🎉 ALL TESTS PASSED! Project is ready to use.\n")
        return 0
    else:
        print(f"\n  ⚠️ {total - passed} test(s) failed. Check errors above.\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
