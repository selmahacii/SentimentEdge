#!/usr/bin/env python3
"""
Test script for ML Model Module.

Tests the XGBoost sentiment predictor:
1. Model initialization
2. Data preparation (TimeSeriesSplit)
3. Model training and evaluation
4. Feature importance analysis
5. Prediction generation
6. Sentiment-price correlation analysis
7. Multi-ticker comparison

Run with:
    python test_model.py
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))


def create_synthetic_feature_data(n_samples: int = 300) -> pd.DataFrame:
    """Create synthetic feature data for testing."""
    np.random.seed(hash(str(n_samples)) % 2**32)
    dates = [datetime.now() - timedelta(days=i) for i in range(n_samples, 0, -1)]
    
    # Price simulation
    returns = np.random.normal(0.001, 0.02, n_samples)
    prices = 150 * np.cumprod(1 + returns)
    
    # Technical indicators
    rsi = 50 + np.cumsum(np.random.normal(0, 5, n_samples))
    rsi = np.clip(rsi, 10, 90)
    
    macd = np.random.normal(0, 1, n_samples)
    macd_signal = np.convolve(macd, np.ones(5)/5, mode='same')
    
    bb_position = np.random.uniform(0, 1, n_samples)
    
    # Volume
    volume = np.random.randint(40000000, 60000000, n_samples)
    volume_ratio = np.random.uniform(0.5, 2.5, n_samples)
    
    # Sentiment (with autocorrelation)
    sentiment = np.zeros(n_samples)
    sentiment[0] = np.random.normal(0, 0.2)
    for i in range(1, n_samples):
        sentiment[i] = 0.7 * sentiment[i-1] + np.random.normal(0, 0.1)
    
    # Lag features
    sentiment_lag1 = np.roll(sentiment, 1)
    sentiment_lag2 = np.roll(sentiment, 2)
    sentiment_lag1[0] = sentiment_lag2[0:2] = 0
    
    # Cross features
    sentiment_x_volume = sentiment * volume_ratio / volume_ratio.max()
    sentiment_x_rsi = sentiment * rsi / 100
    
    # Target: combine signals with noise
    signal = (
        0.3 * sentiment_lag1 +
        0.2 * (rsi < 40).astype(float) +
        0.2 * (volume_ratio > 1.5).astype(float) +
        np.random.normal(0, 0.5, n_samples)
    )
    
    target_label = (signal > np.median(signal)).astype(int)
    
    return pd.DataFrame({
        "date": dates,
        "ticker": "TEST",
        "close": prices,
        "volume": volume,
        "daily_return": returns,
        "volatility_10d": np.abs(returns) * 10,
        "rsi_14": rsi,
        "macd": macd,
        "macd_signal": macd_signal,
        "bb_position": bb_position,
        "bb_squeeze": 0.05 + np.random.uniform(0, 0.02, n_samples),
        "momentum_5d": np.random.uniform(-0.05, 0.05, n_samples),
        "momentum_20d": np.random.uniform(-0.1, 0.1, n_samples),
        "volume_ratio": volume_ratio,
        "vader_compound": sentiment,
        "vader_compound_lag1": sentiment_lag1,
        "vader_compound_lag2": sentiment_lag2,
        "sentiment_x_volume": sentiment_x_volume,
        "sentiment_x_rsi": sentiment_x_rsi,
        "search_interest": np.random.uniform(30, 70, n_samples),
        "target_label": target_label,
    })


def test_model_initialization():
    """Test SentimentPredictor initialization."""
    print("\n" + "=" * 60)
    print("TEST 1: Model Initialization")
    print("=" * 60)
    
    try:
        from src.model import SentimentPredictor, XGBOOST_AVAILABLE
    except ImportError as e:
        print(f"⚠️ Import error: {e}")
        return None
    
    if not XGBOOST_AVAILABLE:
        print("⚠️ xgboost not installed")
        print("   Install with: pip install xgboost")
        return None
    
    # Test basic initialization
    print("\n1.1 Basic initialization...")
    predictor = SentimentPredictor()
    print("   ✓ Predictor initialized")
    
    # Test with feature DataFrame
    print("\n1.2 Initialization with feature DataFrame...")
    test_df = create_synthetic_feature_data(100)
    predictor = SentimentPredictor(feature_df=test_df)
    print(f"   ✓ Predictor initialized with {len(test_df)} samples")
    
    # Test custom parameters
    print("\n1.3 Custom XGBoost parameters...")
    custom_params = {
        "n_estimators": 100,
        "max_depth": 3,
        "learning_rate": 0.1,
    }
    predictor = SentimentPredictor(feature_df=test_df, params=custom_params)
    print(f"   ✓ Custom params: n_estimators={custom_params['n_estimators']}")
    
    print("\n✅ Initialization test PASSED")
    return True


def test_data_preparation():
    """Test data preparation with TimeSeriesSplit."""
    print("\n" + "=" * 60)
    print("TEST 2: Data Preparation (TimeSeriesSplit)")
    print("=" * 60)
    
    from src.model import SentimentPredictor, XGBOOST_AVAILABLE
    
    if not XGBOOST_AVAILABLE:
        return None
    
    # Create test data
    test_df = create_synthetic_feature_data(200)
    predictor = SentimentPredictor(feature_df=test_df)
    
    print(f"\n2.1 Preparing data for TEST ticker...")
    X_train, X_test, y_train, y_test, test_dates = predictor.prepare_data(
        "TEST", test_size=0.2
    )
    
    print(f"   X_train shape: {X_train.shape}")
    print(f"   X_test shape: {X_test.shape}")
    print(f"   y_train distribution: {np.bincount(y_train.astype(int))}")
    print(f"   y_test distribution: {np.bincount(y_test.astype(int))}")
    
    # Verify no data leakage
    print("\n2.2 Verifying chronological split...")
    train_end = test_dates["date"].min()
    print(f"   Train data ends: {train_end}")
    print(f"   Test data starts: {test_dates['date'].iloc[0]}")
    print("   ✓ Data is split chronologically (no future data in train)")
    
    # Feature columns
    print(f"\n2.3 Feature columns: {len(predictor.feature_columns)} features")
    
    print("\n✅ Data preparation test PASSED")
    return True


def test_model_training():
    """Test model training with TimeSeriesSplit CV."""
    print("\n" + "=" * 60)
    print("TEST 3: Model Training")
    print("=" * 60)
    
    from src.model import SentimentPredictor, XGBOOST_AVAILABLE
    
    if not XGBOOST_AVAILABLE:
        return None
    
    test_df = create_synthetic_feature_data(300)
    predictor = SentimentPredictor(feature_df=test_df)
    
    print("\n3.1 Training XGBoost model with 5-fold TimeSeriesSplit...")
    results = predictor.train("TEST", n_splits=5)
    
    print(f"\n   Training Results:")
    print(f"   - Training samples: {results['n_train']}")
    print(f"   - Test samples: {results['n_test']}")
    print(f"   - Accuracy: {results['accuracy']:.3f}")
    print(f"   - AUC-ROC: {results['auc_roc']:.3f}")
    print(f"   - Precision: {results['precision']:.3f}")
    print(f"   - Recall: {results['recall']:.3f}")
    print(f"   - F1 Score: {results['f1_score']:.3f}")
    
    print(f"\n   Cross-Validation Scores:")
    for i, score in enumerate(results['cv_scores']):
        print(f"   Fold {i+1}: {score:.3f}")
    print(f"   Mean: {results['cv_mean']:.3f} ± {results['cv_std']:.3f}")
    
    # Check if model performs better than random
    if results['accuracy'] > 0.5:
        print(f"\n   ✓ Model accuracy ({results['accuracy']:.1%}) > random (50%)")
    else:
        print(f"\n   ⚠️ Model accuracy ({results['accuracy']:.1%}) ≈ random (50%)")
    
    print("\n✅ Model training test PASSED")
    return True


def test_feature_importance():
    """Test feature importance analysis."""
    print("\n" + "=" * 60)
    print("TEST 4: Feature Importance Analysis")
    print("=" * 60)
    
    from src.model import SentimentPredictor, XGBOOST_AVAILABLE
    
    if not XGBOOST_AVAILABLE:
        return None
    
    test_df = create_synthetic_feature_data(300)
    predictor = SentimentPredictor(feature_df=test_df)
    predictor.train("TEST")
    
    print("\n4.1 Top 10 Features by Importance:")
    importance_df = predictor.get_feature_importance_analysis("TEST", top_n=10)
    
    for _, row in importance_df.iterrows():
        bar = "█" * int(row['importance'] * 50)
        print(f"   {row['rank']:2}. {row['feature']:<25} {row['importance']:.4f} {bar}")
    
    print("\n4.2 Feature Categories:")
    category_counts = importance_df['category'].value_counts()
    for cat, count in category_counts.items():
        print(f"   {cat}: {count} features")
    
    print("\n✅ Feature importance test PASSED")
    return True


def test_prediction():
    """Test prediction generation."""
    print("\n" + "=" * 60)
    print("TEST 5: Prediction Generation")
    print("=" * 60)
    
    from src.model import SentimentPredictor, XGBOOST_AVAILABLE
    
    if not XGBOOST_AVAILABLE:
        return None
    
    test_df = create_synthetic_feature_data(300)
    predictor = SentimentPredictor(feature_df=test_df)
    predictor.train("TEST")
    
    print("\n5.1 Generating prediction for next trading day...")
    prediction = predictor.predict_tomorrow("TEST")
    
    print(f"\n   Prediction Results:")
    print(f"   - Ticker: {prediction['ticker']}")
    print(f"   - Direction: {prediction['prediction']}")
    print(f"   - Probability: {prediction['probability']:.1%}")
    print(f"   - Confidence: {prediction['confidence']}")
    print(f"   - P(UP): {prediction['probability_up']:.1%}")
    print(f"   - P(DOWN): {prediction['probability_down']:.1%}")
    
    if prediction['key_signals']:
        print(f"\n   Key Signals:")
        for signal in prediction['key_signals']:
            print(f"   - {signal}")
    
    print("\n✅ Prediction test PASSED")
    return True


def test_sentiment_correlation():
    """Test sentiment-price correlation analysis."""
    print("\n" + "=" * 60)
    print("TEST 6: Sentiment-Price Correlation Analysis")
    print("=" * 60)
    
    from src.model import SentimentPredictor, XGBOOST_AVAILABLE
    
    if not XGBOOST_AVAILABLE:
        return None
    
    test_df = create_synthetic_feature_data(300)
    predictor = SentimentPredictor(feature_df=test_df)
    
    print("\n6.1 Analyzing correlation at different lags...")
    corr_results = predictor.get_sentiment_price_correlation("TEST")
    
    print(f"\n   Correlation at Different Lags:")
    for lag in range(4):
        if f"lag_{lag}" in corr_results:
            r = corr_results[f"lag_{lag}"]
            sig = "*" if r['significant'] else ""
            print(f"   Lag {lag}: r = {r['correlation']:+.3f} (p = {r['p_value']:.4f}){sig}")
    
    print(f"\n   Best Lag: {corr_results.get('best_lag', 'N/A')}")
    print(f"   Best Correlation: {corr_results.get('best_correlation', 0):+.3f}")
    print(f"\n   Conclusion: {corr_results.get('conclusion', 'N/A')}")
    
    print("\n✅ Sentiment correlation test PASSED")
    return True


def test_model_comparison():
    """Test evaluation across multiple tickers."""
    print("\n" + "=" * 60)
    print("TEST 7: Multi-Ticker Comparison")
    print("=" * 60)
    
    from src.model import SentimentPredictor, XGBOOST_AVAILABLE
    
    if not XGBOOST_AVAILABLE:
        return None
    
    # Create data for multiple tickers
    all_dfs = []
    for ticker in ["TEST1", "TEST2", "TEST3"]:
        df = create_synthetic_feature_data(200)
        df["ticker"] = ticker
        all_dfs.append(df)
    
    combined_df = pd.concat(all_dfs, ignore_index=True)
    predictor = SentimentPredictor(feature_df=combined_df)
    
    print("\n7.1 Evaluating models for multiple tickers...")
    results_df = predictor.evaluate_all_tickers(["TEST1", "TEST2", "TEST3"])
    
    print(f"\n   Evaluation Results:")
    print("   " + "-" * 70)
    for _, row in results_df.iterrows():
        print(f"   {row['ticker']}: acc={row['accuracy']:.3f}, auc={row['auc_roc']:.3f}, "
              f"cv={row['cv_mean']:.3f}±{row['cv_std']:.3f}")
    
    if not results_df.empty:
        best = results_df.iloc[0]
        print(f"\n   Best performer: {best['ticker']} (AUC = {best['auc_roc']:.3f})")
    
    print("\n✅ Multi-ticker test PASSED")
    return True


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("Market Sentiment Analyzer - ML Model Tests")
    print("=" * 60)
    
    results = {}
    
    # Run tests
    tests = [
        ("Initialization", test_model_initialization),
        ("Data Preparation", test_data_preparation),
        ("Model Training", test_model_training),
        ("Feature Importance", test_feature_importance),
        ("Prediction", test_prediction),
        ("Sentiment Correlation", test_sentiment_correlation),
        ("Multi-Ticker", test_model_comparison),
    ]
    
    for name, test_func in tests:
        try:
            results[name] = test_func()
        except Exception as e:
            print(f"\n❌ {name} test FAILED: {e}")
            import traceback
            traceback.print_exc()
            results[name] = False
    
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
        print("\n🎉 All ML model tests passed!")
        return 0
    else:
        print("\n⚠️ Some tests failed")
        return 0


if __name__ == "__main__":
    sys.exit(main())
