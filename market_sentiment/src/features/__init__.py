"""
Feature Engineering Package

This package provides feature engineering capabilities:

technical.py:
- RSI (Relative Strength Index) - momentum oscillator
- MACD (Moving Average Convergence Divergence) - trend indicator
- Bollinger Bands - volatility indicator
- Volume indicators - unusual volume detection
- Price momentum - rate of change
- Moving Averages - SMA, EMA
- ATR (Average True Range) - volatility measure

fusion.py:
- Merges sentiment, price, and trends data on (ticker, date)
- Creates lag features (sentiment → price hypothesis)
- Creates cross-features (sentiment × volume, sentiment × RSI)
- Generates target labels for ML
- Handles missing data appropriately

Key Hypothesis:
Lag features test whether Reddit sentiment precedes price
movements by 1-3 days. If lag features have high importance
in the ML model, it validates this hypothesis.

Usage:
    from src.features import compute_technical_features, merge_all_features
    
    # Add technical indicators to price data
    prices_with_features = compute_technical_features(prices_df)
    
    # Merge all data sources into feature matrix
    feature_matrix = merge_all_features(prices_df, sentiment_df, trends_df)
"""

from src.features.technical import (
    compute_rsi,
    compute_rsi_signal,
    compute_macd,
    compute_macd_crossover,
    compute_bollinger_bands,
    compute_bb_position,
    compute_bb_squeeze,
    compute_volume_sma,
    compute_volume_ratio,
    compute_momentum,
    compute_price_vs_high,
    compute_sma,
    compute_ema,
    compute_price_vs_sma,
    compute_atr,
    compute_volatility,
    compute_technical_features,
    get_feature_summary,
)

from src.features.fusion import (
    merge_all_features,
    prepare_for_ml,
    create_lag_features,
    create_rolling_features,
    create_cross_features,
    create_target_variable,
    get_feature_importance_order,
)

__all__ = [
    # Technical indicators - individual
    "compute_rsi",
    "compute_rsi_signal",
    "compute_macd",
    "compute_macd_crossover",
    "compute_bollinger_bands",
    "compute_bb_position",
    "compute_bb_squeeze",
    "compute_volume_sma",
    "compute_volume_ratio",
    "compute_momentum",
    "compute_price_vs_high",
    "compute_sma",
    "compute_ema",
    "compute_price_vs_sma",
    "compute_atr",
    "compute_volatility",
    # Technical indicators - main function
    "compute_technical_features",
    "get_feature_summary",
    # Feature fusion
    "merge_all_features",
    "prepare_for_ml",
    "create_lag_features",
    "create_rolling_features",
    "create_cross_features",
    "create_target_variable",
    "get_feature_importance_order",
]
