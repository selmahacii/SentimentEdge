#!/usr/bin/env python3
"""
Test script for Feature Engineering Modules.

Tests both technical indicators and feature fusion:
1. Technical indicator calculations
2. Feature fusion pipeline
3. Lag feature creation
4. Target variable generation
5. ML preparation

Run with:
    python test_features.py
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))


def create_test_price_data(days: int = 100) -> pd.DataFrame:
    """Create realistic test price data."""
    np.random.seed(42)
    dates = [datetime.now() - timedelta(days=i) for i in range(days, 0, -1)]
    
    # Simulate price with trend and volatility
    base_price = 150
    returns = np.random.normal(0.001, 0.02, days)  # Slight upward drift
    prices = base_price * np.cumprod(1 + returns)
    
    # Create OHLCV data
    df = pd.DataFrame({
        "date": dates,
        "ticker": "AAPL",
        "open": prices * (1 + np.random.uniform(-0.01, 0.01, days)),
        "high": prices * (1 + np.random.uniform(0, 0.02, days)),
        "low": prices * (1 - np.random.uniform(0, 0.02, days)),
        "close": prices,
        "volume": np.random.randint(40000000, 60000000, days),
    })
    
    # Add basic returns
    df["daily_return"] = df["close"].pct_change()
    
    return df


def create_test_sentiment_data(dates: list) -> pd.DataFrame:
    """Create test sentiment data with gaps."""
    np.random.seed(123)
    
    # Simulate gaps (weekends, etc.)
    sentiment_dates = dates[::2]  # Every other day
    
    df = pd.DataFrame({
        "date": sentiment_dates,
        "ticker": "AAPL",
        "vader_compound": np.random.uniform(-0.5, 0.5, len(sentiment_dates)),
        "vader_positive": np.random.uniform(0.2, 0.5, len(sentiment_dates)),
        "vader_negative": np.random.uniform(0.1, 0.3, len(sentiment_dates)),
        "vader_neutral": np.random.uniform(0.3, 0.6, len(sentiment_dates)),
        "weighted_compound": np.random.uniform(-0.3, 0.3, len(sentiment_dates)),
        "sentiment_momentum": np.random.uniform(-0.2, 0.2, len(sentiment_dates)),
        "post_count": np.random.randint(5, 30, len(sentiment_dates)),
        "avg_score": np.random.uniform(50, 200, len(sentiment_dates)),
        "total_comments": np.random.randint(20, 100, len(sentiment_dates)),
    })
    
    return df


def create_test_trends_data(dates: list) -> pd.DataFrame:
    """Create test Google Trends data."""
    np.random.seed(456)
    
    df = pd.DataFrame({
        "date": dates,
        "ticker": "AAPL",
        "search_interest": np.random.uniform(30, 70, len(dates)),
    })
    
    return df


def test_technical_indicators():
    """Test technical indicator calculations."""
    print("\n" + "=" * 60)
    print("TEST 1: Technical Indicators")
    print("=" * 60)
    
    from src.features.technical import (
        compute_technical_features,
        compute_rsi,
        compute_macd,
        compute_bollinger_bands,
        get_feature_summary,
    )
    
    # Create test data
    print("\n1.1 Creating test price data...")
    prices_df = create_test_price_data(100)
    print(f"   Created {len(prices_df)} rows of price data")
    
    # Compute all technical features
    print("\n1.2 Computing technical features...")
    df = compute_technical_features(prices_df)
    
    # Check features were added
    new_cols = [c for c in df.columns if c not in prices_df.columns]
    print(f"   Added {len(new_cols)} new feature columns:")
    for col in sorted(new_cols):
        print(f"     - {col}")
    
    # Test RSI
    print("\n1.3 Testing RSI calculation...")
    rsi = df["rsi_14"].dropna()
    print(f"   RSI range: {rsi.min():.1f} - {rsi.max():.1f}")
    print(f"   RSI mean: {rsi.mean():.1f}")
    
    # Check RSI signals
    overbought = (df["rsi_14"] > 70).sum()
    oversold = (df["rsi_14"] < 30).sum()
    print(f"   Overbought (>70): {overbought}")
    print(f"   Oversold (<30): {oversold}")
    
    # Test MACD
    print("\n1.4 Testing MACD calculation...")
    macd_valid = df[["macd", "macd_signal", "macd_hist"]].dropna()
    print(f"   Valid MACD rows: {len(macd_valid)}")
    
    # Check crossovers
    bullish_cross = (df["macd_crossover"] == 1).sum()
    bearish_cross = (df["macd_crossover"] == -1).sum()
    print(f"   Bullish crossovers: {bullish_cross}")
    print(f"   Bearish crossovers: {bearish_cross}")
    
    # Test Bollinger Bands
    print("\n1.5 Testing Bollinger Bands...")
    bb_valid = df[["bb_upper", "bb_mid", "bb_lower", "bb_position"]].dropna()
    print(f"   Valid BB rows: {len(bb_valid)}")
    print(f"   BB position range: {df['bb_position'].min():.2f} - {df['bb_position'].max():.2f}")
    
    # Test volume features
    print("\n1.6 Testing volume features...")
    vol_valid = df["volume_ratio"].dropna()
    unusual_vol = (df["volume_ratio"] > 2.0).sum()
    print(f"   Volume ratio range: {vol_valid.min():.2f} - {vol_valid.max():.2f}")
    print(f"   Unusual volume (>2x): {unusual_vol}")
    
    # Get feature summary
    print("\n1.7 Feature summary:")
    summary = get_feature_summary(df)
    for col, stats in list(summary.items())[:3]:
        print(f"   {col}: mean={stats['mean']:.4f}, std={stats['std']:.4f}")
    
    print("\n✅ Technical indicators test PASSED")
    return True, df


def test_feature_fusion():
    """Test feature fusion pipeline."""
    print("\n" + "=" * 60)
    print("TEST 2: Feature Fusion")
    print("=" * 60)
    
    from src.features.fusion import (
        merge_all_features,
        prepare_for_ml,
        create_lag_features,
        create_target_variable,
        get_feature_importance_order,
    )
    
    # Create test data
    print("\n2.1 Creating test data sources...")
    prices_df = create_test_price_data(50)
    
    # Add some technical features first
    from src.features.technical import compute_technical_features
    prices_df = compute_technical_features(prices_df)
    
    dates = prices_df["date"].tolist()
    sentiment_df = create_test_sentiment_data(dates)
    trends_df = create_test_trends_data(dates)
    
    print(f"   Prices: {len(prices_df)} rows")
    print(f"   Sentiment: {len(sentiment_df)} rows (with gaps)")
    print(f"   Trends: {len(trends_df)} rows")
    
    # Test data alignment
    print("\n2.2 Testing data alignment...")
    feature_matrix = merge_all_features(
        prices_df,
        sentiment_df,
        trends_df,
        prediction_horizon=3
    )
    
    print(f"   Merged shape: {feature_matrix.shape}")
    print(f"   Total columns: {len(feature_matrix.columns)}")
    
    # Check for lag features
    print("\n2.3 Checking lag features...")
    lag_cols = [c for c in feature_matrix.columns if "lag" in c.lower()]
    print(f"   Lag features: {lag_cols}")
    
    # Check for cross features
    print("\n2.4 Checking cross features...")
    cross_cols = [c for c in feature_matrix.columns if "_x_" in c.lower()]
    print(f"   Cross features: {cross_cols}")
    
    # Check target variable
    print("\n2.5 Checking target variable...")
    if "target_label" in feature_matrix.columns:
        valid_targets = feature_matrix["target_label"].notna().sum()
        positive_rate = feature_matrix["target_label"].mean()
        print(f"   Valid targets: {valid_targets}")
        print(f"   Positive rate: {positive_rate:.1%}")
    
    # Test ML preparation
    print("\n2.6 Testing ML preparation...")
    X, y, feature_cols = prepare_for_ml(feature_matrix, drop_na=True)
    
    print(f"   X shape: {X.shape}")
    print(f"   y shape: {y.shape}")
    print(f"   Number of features: {len(feature_cols)}")
    
    # Feature groups
    print("\n2.7 Feature groups:")
    feature_groups = get_feature_importance_order(feature_matrix, feature_cols)
    if not feature_groups.empty:
        print(feature_groups.groupby("group").size().to_string())
    
    print("\n✅ Feature fusion test PASSED")
    return True, feature_matrix


def test_lag_features():
    """Test lag feature creation specifically."""
    print("\n" + "=" * 60)
    print("TEST 3: Lag Features (Key Hypothesis)")
    print("=" * 60)
    
    from src.features.fusion import create_lag_features
    
    # Create simple test data
    dates = [datetime.now() - timedelta(days=i) for i in range(10, 0, -1)]
    df = pd.DataFrame({
        "date": dates,
        "ticker": "TEST",
        "vader_compound": [0.1, 0.2, 0.3, 0.4, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0],
    })
    
    print("\n3.1 Original sentiment data:")
    print(df[["date", "vader_compound"]].to_string())
    
    # Create lag features
    print("\n3.2 Creating lag features...")
    df_lagged = create_lag_features(df, column="vader_compound", lags=[1, 2, 3])
    
    print("\n3.3 Data with lag features:")
    lag_cols = ["vader_compound", "vader_compound_lag1", "vader_compound_lag2", "vader_compound_lag3"]
    print(df_lagged[lag_cols].to_string())
    
    print("\n3.4 Interpretation:")
    print("   - lag1 = yesterday's sentiment")
    print("   - lag2 = 2 days ago sentiment")
    print("   - lag3 = 3 days ago sentiment")
    print("   - If lag features predict price, sentiment precedes movement!")
    
    print("\n✅ Lag features test PASSED")
    return True


def test_target_creation():
    """Test target variable creation."""
    print("\n" + "=" * 60)
    print("TEST 4: Target Variable Creation")
    print("=" * 60)
    
    from src.features.fusion import create_target_variable
    
    # Create test price data
    dates = [datetime.now() - timedelta(days=i) for i in range(10, 0, -1)]
    prices = [100, 101, 102, 101, 100, 99, 100, 101, 102, 103]
    
    df = pd.DataFrame({
        "date": dates,
        "ticker": "TEST",
        "close": prices,
    })
    
    print("\n4.1 Original price data:")
    print(df[["date", "close"]].to_string())
    
    # Create target (horizon=3)
    print("\n4.2 Creating target (horizon=3 days)...")
    df = create_target_variable(df, horizon=3)
    
    print("\n4.3 Data with target:")
    print(df[["date", "close", "future_close", "target_return", "target_label"]].to_string())
    
    print("\n4.4 Interpretation:")
    print("   - target_label: 1 = price UP in 3 days, 0 = price DOWN/FLAT")
    print("   - target_return: actual % change in 3 days")
    print("   - Last 3 rows have NaN (no future data available)")
    
    print("\n✅ Target creation test PASSED")
    return True


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("Market Sentiment Analyzer - Feature Engineering Tests")
    print("=" * 60)
    
    results = {}
    
    # Test technical indicators
    try:
        results["technical"], _ = test_technical_indicators()
    except Exception as e:
        print(f"\n❌ Technical indicators test FAILED: {e}")
        results["technical"] = False
    
    # Test feature fusion
    try:
        results["fusion"], _ = test_feature_fusion()
    except Exception as e:
        print(f"\n❌ Feature fusion test FAILED: {e}")
        results["fusion"] = False
    
    # Test lag features
    try:
        results["lag"] = test_lag_features()
    except Exception as e:
        print(f"\n❌ Lag features test FAILED: {e}")
        results["lag"] = False
    
    # Test target creation
    try:
        results["target"] = test_target_creation()
    except Exception as e:
        print(f"\n❌ Target creation test FAILED: {e}")
        results["target"] = False
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    for name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {name}: {status}")
    
    passed = sum(1 for v in results.values() if v)
    failed = sum(1 for v in results.values() if not v)
    
    print(f"\nTotal: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("\n🎉 All feature engineering tests PASSED!")
        return 0
    else:
        print("\n❌ Some tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
