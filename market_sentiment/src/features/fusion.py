"""
Feature Fusion Module

Merges heterogeneous data sources (prices, sentiment, trends) into a
unified feature matrix ready for machine learning.

This is the critical module that:
1. Aligns data on (ticker, date) temporal key
2. Handles missing data (weekend gaps in sentiment)
3. Creates lag features (testing hypothesis: sentiment → price)
4. Creates cross-features (sentiment × volume, sentiment × RSI)
5. Generates target variables for ML

Key Hypothesis:
==============
We test whether Reddit sentiment PRECEDES price movements by 1-3 days.
Lag features (sentiment_lag1, sentiment_lag2, sentiment_lag3) capture this.

If lag features have high importance in the ML model, it validates the
hypothesis that retail sentiment on Reddit predicts short-term price moves.

Target Variable:
===============
target_label: 1 if price UP at J+3, 0 if price DOWN/FLAT
target_return: Actual return at J+3 (for regression analysis)

CRITICAL: These target columns use shift(-3), meaning they look 3 days
into the future. They must NEVER be used as features. They are flagged
in EXCLUDE_COLUMNS in config.py.

DISCLAIMER: Feature engineering for educational purposes only.
Past patterns do not guarantee future results.
"""

import numpy as np
import pandas as pd
from loguru import logger
from typing import Optional


# ============================================================
# Data Alignment Functions
# ============================================================

def align_on_date(
    prices_df: pd.DataFrame,
    sentiment_df: pd.DataFrame,
    trends_df: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """
    Align all DataFrames on (ticker, date) key.
    
    This function merges price data with sentiment and trends data.
    Different data sources may have different date coverage:
    - Prices: Market days only (no weekends)
    - Sentiment: Any day with Reddit activity (can be weekends)
    - Trends: Daily (interpolated from weekly)
    
    Args:
        prices_df: DataFrame with price data (must have date, ticker columns)
        sentiment_df: DataFrame with sentiment data (must have date, ticker columns)
        trends_df: Optional DataFrame with trends data
    
    Returns:
        Merged DataFrame aligned on (ticker, date)
    """
    logger.info("Aligning data sources on (ticker, date)...")
    
    # Ensure date columns are datetime
    prices_df = prices_df.copy()
    sentiment_df = sentiment_df.copy()
    
    prices_df["date"] = pd.to_datetime(prices_df["date"])
    sentiment_df["date"] = pd.to_datetime(sentiment_df["date"])
    
    # Merge prices with sentiment (left join - keep all price dates)
    # Using left join ensures we keep all trading days
    merged = pd.merge(
        prices_df,
        sentiment_df,
        on=["date", "ticker"],
        how="left",
        suffixes=("", "_sentiment")
    )
    
    logger.info(f"Merged prices + sentiment: {len(merged)} rows")
    
    # Merge with trends if available
    if trends_df is not None and not trends_df.empty:
        trends_df = trends_df.copy()
        trends_df["date"] = pd.to_datetime(trends_df["date"])
        
        merged = pd.merge(
            merged,
            trends_df[["date", "ticker", "search_interest"]],
            on=["date", "ticker"],
            how="left",
            suffixes=("", "_trends")
        )
        
        logger.info(f"Merged with trends: {len(merged)} rows")
    
    return merged


def handle_missing_sentiment(
    df: pd.DataFrame,
    forward_fill_days: int = 2
) -> pd.DataFrame:
    """
    Handle missing sentiment data appropriately.
    
    Sentiment data has gaps on days with no Reddit activity.
    We use a forward-fill strategy with a limit:
    - Forward fill up to N days (sentiment persists over weekends)
    - After N days of no data, fill with 0 (neutral)
    - Add a flag column indicating if sentiment was imputed
    
    Args:
        df: DataFrame with sentiment columns
        forward_fill_days: Maximum days to forward fill
    
    Returns:
        DataFrame with missing sentiment handled
    """
    df = df.copy()
    
    # Identify sentiment columns
    sentiment_cols = [
        "vader_compound", "vader_positive", "vader_negative", "vader_neutral",
        "finbert_positive", "finbert_negative", "finbert_neutral",
        "weighted_compound", "sentiment_momentum", "post_count",
        "avg_score", "total_comments"
    ]
    
    # Track which rows have sentiment data
    if "vader_compound" in df.columns:
        df["sentiment_data_available"] = df["vader_compound"].notna()
    
    # Sort for proper forward fill
    if "ticker" in df.columns:
        df = df.sort_values(["ticker", "date"])
    else:
        df = df.sort_values("date")
    
    # Forward fill sentiment columns (per ticker)
    for col in sentiment_cols:
        if col in df.columns:
            if "ticker" in df.columns:
                df[col] = df.groupby("ticker")[col].transform(
                    lambda x: x.ffill(limit=forward_fill_days)
                )
            else:
                df[col] = df[col].ffill(limit=forward_fill_days)
    
    # Fill remaining NaN with neutral values
    fill_values = {
        "vader_compound": 0.0,
        "vader_positive": 0.0,
        "vader_negative": 0.0,
        "vader_neutral": 1.0,  # Default to neutral
        "weighted_compound": 0.0,
        "sentiment_momentum": 0.0,
        "post_count": 0,
        "avg_score": 0.0,
        "total_comments": 0,
        "search_interest": df["search_interest"].median() if "search_interest" in df.columns else 50.0,
    }
    
    for col, value in fill_values.items():
        if col in df.columns:
            df[col] = df[col].fillna(value)
    
    # Count filled values
    filled_count = (~df["sentiment_data_available"]).sum()
    logger.info(f"Handled missing sentiment: {filled_count} rows filled ({filled_count/len(df)*100:.1f}%)")
    
    return df


# ============================================================
# Lag Feature Creation
# ============================================================

def create_lag_features(
    df: pd.DataFrame,
    column: str = "vader_compound",
    lags: list[int] = [1, 2, 3]
) -> pd.DataFrame:
    """
    Create lag features for a column.
    
    Lag features test the hypothesis that sentiment PRECEDES price.
    If sentiment_lag1 is highly predictive, it means yesterday's
    sentiment predicts today's price movement.
    
    Formula:
        sentiment_lag1 = sentiment.shift(1)  # Yesterday's sentiment
        sentiment_lag2 = sentiment.shift(2)  # 2 days ago
        sentiment_lag3 = sentiment.shift(3)  # 3 days ago
    
    Args:
        df: DataFrame with the column to lag
        column: Column name to create lags for
        lags: List of lag periods (default: [1, 2, 3])
    
    Returns:
        DataFrame with lag columns added
    
    Note:
        Lag features use shift(positive), meaning they look BACKWARD.
        This is valid for ML - we're using PAST data to predict FUTURE.
    """
    df = df.copy()
    
    if column not in df.columns:
        logger.warning(f"Column {column} not found, skipping lag features")
        return df
    
    # Sort for proper lag calculation
    if "ticker" in df.columns:
        df = df.sort_values(["ticker", "date"])
        
        # Calculate lags per ticker
        for lag in lags:
            lag_col = f"{column}_lag{lag}"
            df[lag_col] = df.groupby("ticker")[column].shift(lag)
    else:
        df = df.sort_values("date")
        
        for lag in lags:
            lag_col = f"{column}_lag{lag}"
            df[lag_col] = df[column].shift(lag)
    
    logger.debug(f"Created {len(lags)} lag features for {column}")
    
    return df


def create_rolling_features(
    df: pd.DataFrame,
    column: str = "vader_compound",
    windows: list[int] = [3, 7]
) -> pd.DataFrame:
    """
    Create rolling average features.
    
    Rolling features capture sentiment trends over time.
    3-day average shows short-term trend.
    7-day average shows week-long trend.
    
    Args:
        df: DataFrame with the column
        column: Column name
        windows: List of window sizes
    
    Returns:
        DataFrame with rolling features added
    """
    df = df.copy()
    
    if column not in df.columns:
        return df
    
    if "ticker" in df.columns:
        df = df.sort_values(["ticker", "date"])
        
        for window in windows:
            roll_col = f"{column}_rolling{window}"
            df[roll_col] = df.groupby("ticker")[column].transform(
                lambda x: x.rolling(window=window).mean()
            )
    else:
        df = df.sort_values("date")
        
        for window in windows:
            roll_col = f"{column}_rolling{window}"
            df[roll_col] = df[column].rolling(window=window).mean()
    
    return df


# ============================================================
# Cross-Feature Creation
# ============================================================

def create_cross_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create cross-features (interaction terms).
    
    Cross-features capture interactions between sentiment and technical:
    - sentiment_x_volume: Strong sentiment + high volume = stronger signal
    - sentiment_x_rsi: Bullish sentiment + oversold RSI = reversal signal
    
    Args:
        df: DataFrame with sentiment and technical features
    
    Returns:
        DataFrame with cross-features added
    """
    df = df.copy()
    
    # Sentiment × Volume Ratio
    # Strong sentiment with high volume = more conviction
    if "vader_compound" in df.columns and "volume_ratio" in df.columns:
        df["sentiment_x_volume"] = df["vader_compound"] * df["volume_ratio"]
        logger.debug("Created sentiment_x_volume cross-feature")
    
    # Sentiment × RSI
    # Normalized RSI to 0-1 scale
    if "vader_compound" in df.columns and "rsi_14" in df.columns:
        rsi_normalized = df["rsi_14"] / 100
        df["sentiment_x_rsi"] = df["vader_compound"] * rsi_normalized
        logger.debug("Created sentiment_x_rsi cross-feature")
    
    # Sentiment × Price Momentum
    # Sentiment aligned with momentum = confirmation
    if "vader_compound" in df.columns and "momentum_5d" in df.columns:
        df["sentiment_x_momentum"] = df["vader_compound"] * df["momentum_5d"]
        logger.debug("Created sentiment_x_momentum cross-feature")
    
    # Sentiment × Volatility
    # High volatility changes sentiment impact
    if "vader_compound" in df.columns and "volatility_20d" in df.columns:
        # Normalize volatility (divide by typical value)
        vol_normalized = df["volatility_20d"] / df["volatility_20d"].mean()
        df["sentiment_x_volatility"] = df["vader_compound"] * vol_normalized
        logger.debug("Created sentiment_x_volatility cross-feature")
    
    return df


# ============================================================
# Target Variable Creation
# ============================================================

def create_target_variable(
    df: pd.DataFrame,
    horizon: int = 3
) -> pd.DataFrame:
    """
    Create target variable for ML prediction.
    
    CRITICAL: This function uses shift(-horizon), which means
    it looks FORWARD in time. The target columns must NEVER
    be used as features - they are what we're trying to predict.
    
    Target types:
    - target_label: Binary classification (1=up, 0=down/flat)
    - target_return: Regression target (actual return at J+N)
    
    Args:
        df: DataFrame with close prices
        horizon: Prediction horizon in days (default: 3)
    
    Returns:
        DataFrame with target columns added
    
    WARNING: After creating targets, these columns must be
    excluded from feature matrix before ML training.
    See EXCLUDE_COLUMNS in config.py.
    """
    df = df.copy()
    
    if "close" not in df.columns:
        logger.error("Missing 'close' column, cannot create target")
        return df
    
    # Sort for proper shift
    if "ticker" in df.columns:
        df = df.sort_values(["ticker", "date"])
        
        # Future close price
        df["future_close"] = df.groupby("ticker")["close"].shift(-horizon)
        
        # Target return (percentage change)
        df["target_return"] = (df["future_close"] - df["close"]) / df["close"]
        
        # Target label (binary: 1 if up, 0 if down/flat)
        df["target_label"] = (df["target_return"] > 0).astype(int)
        
    else:
        df = df.sort_values("date")
        
        df["future_close"] = df["close"].shift(-horizon)
        df["target_return"] = (df["future_close"] - df["close"]) / df["close"]
        df["target_label"] = (df["target_return"] > 0).astype(int)
    
    # Log target distribution
    if "target_label" in df.columns:
        positive = df["target_label"].sum()
        total = df["target_label"].notna().sum()
        if total > 0:
            logger.info(
                f"Target distribution: {positive}/{total} ({positive/total*100:.1f}%) positive"
            )
    
    # Count rows without target (last N days)
    no_target = df["target_label"].isna().sum()
    logger.info(f"Created target variable (horizon={horizon}). {no_target} rows without target (most recent)")
    
    return df


# ============================================================
# Main Fusion Function
# ============================================================

def merge_all_features(
    prices_df: pd.DataFrame,
    sentiment_df: pd.DataFrame,
    trends_df: Optional[pd.DataFrame] = None,
    prediction_horizon: int = 3
) -> pd.DataFrame:
    """
    Merge all data sources into unified feature matrix.
    
    This is the main orchestration function that:
    1. Aligns data on (ticker, date)
    2. Handles missing sentiment data
    3. Creates lag features (sentiment → price hypothesis)
    4. Creates cross-features
    5. Creates target variables
    
    Args:
        prices_df: DataFrame with price data and technical indicators
        sentiment_df: DataFrame with daily sentiment scores
        trends_df: Optional DataFrame with Google Trends data
        prediction_horizon: Days ahead to predict (default: 3)
    
    Returns:
        Complete feature matrix with all features and target
    
    Note:
        Final DataFrame will have NaN rows at:
        - Beginning: due to rolling windows (technical indicators)
        - End: due to target variable shift (no future data)
        These rows should be dropped before ML training.
    """
    logger.info("=" * 50)
    logger.info("Starting feature fusion...")
    logger.info(f"Prices: {len(prices_df)} rows")
    logger.info(f"Sentiment: {len(sentiment_df)} rows")
    if trends_df is not None:
        logger.info(f"Trends: {len(trends_df)} rows")
    
    # Step 1: Align data on date
    df = align_on_date(prices_df, sentiment_df, trends_df)
    
    # Step 2: Handle missing sentiment
    df = handle_missing_sentiment(df, forward_fill_days=2)
    
    # Step 3: Create lag features for sentiment
    # KEY: These test the hypothesis that sentiment precedes price
    df = create_lag_features(df, column="vader_compound", lags=[1, 2, 3])
    df = create_rolling_features(df, column="vader_compound", windows=[3])
    
    # Rename for clarity
    if "vader_compound_rolling3" in df.columns:
        df = df.rename(columns={"vader_compound_rolling3": "sentiment_momentum_3d"})
    
    # Step 4: Create cross-features
    df = create_cross_features(df)
    
    # Step 5: Create target variable
    # CRITICAL: This uses future data, must be excluded from features
    df = create_target_variable(df, horizon=prediction_horizon)
    
    # Step 6: Final cleanup
    # Drop rows with all NaN in feature columns
    feature_cols = [c for c in df.columns if c not in 
                   ["date", "ticker", "target_label", "target_return", "future_close"]]
    
    # Log final shape
    total_rows = len(df)
    complete_rows = df.dropna(subset=["target_label"]).shape[0]
    
    logger.info("=" * 50)
    logger.info(f"Feature matrix complete: {total_rows} total rows")
    logger.info(f"Complete rows (with target): {complete_rows}")
    logger.info(f"Features: {len(feature_cols)} columns")
    logger.info("=" * 50)
    
    return df


def prepare_for_ml(
    df: pd.DataFrame,
    drop_na: bool = True
) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """
    Prepare feature matrix for ML training.
    
    Separates features from target and handles missing values.
    
    Args:
        df: Complete feature matrix
        drop_na: Whether to drop rows with NaN values
    
    Returns:
        Tuple of (X, y, feature_columns)
    """
    # Columns to exclude from features
    exclude_cols = {
        "id", "ticker", "date", "target_label", "target_return",
        "future_close", "combined_text", "title", "selftext",
        "created_at", "fetched_at", "sentiment_data_available"
    }
    
    # Get feature columns
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    
    # Separate features and target
    X = df[feature_cols].copy()
    y = df["target_label"].copy() if "target_label" in df.columns else None
    
    # Drop rows with NaN
    if drop_na and y is not None:
        valid_idx = X.notna().all(axis=1) & y.notna()
        X = X[valid_idx]
        y = y[valid_idx]
        
        logger.info(f"Prepared {len(X)} samples for ML (dropped {len(df) - len(X)} with NaN)")
    
    return X, y, feature_cols


def get_feature_importance_order(
    df: pd.DataFrame,
    feature_cols: list[str]
) -> pd.DataFrame:
    """
    Get feature columns ordered by type for analysis.
    
    Args:
        df: Feature matrix
        feature_cols: List of feature column names
    
    Returns:
        DataFrame with feature groupings
    """
    groups = {
        "price": ["close", "volume", "daily_return", "log_return", "volatility_10d"],
        "technical": ["rsi_14", "macd", "macd_signal", "macd_hist", "bb_position", 
                      "bb_squeeze", "volume_ratio", "momentum_5d", "momentum_20d"],
        "sentiment": ["vader_compound", "vader_positive", "vader_negative",
                      "weighted_compound", "sentiment_momentum", "post_count"],
        "lag": ["vader_compound_lag1", "vader_compound_lag2", "vader_compound_lag3",
                "sentiment_momentum_3d"],
        "trends": ["search_interest"],
        "cross": ["sentiment_x_volume", "sentiment_x_rsi"],
    }
    
    result = []
    for group, cols in groups.items():
        for col in cols:
            if col in feature_cols:
                result.append({"feature": col, "group": group})
    
    return pd.DataFrame(result)


# ============================================================
# Module Test
# ============================================================

if __name__ == "__main__":
    from datetime import datetime, timedelta
    import numpy as np
    
    print("Feature Fusion Test")
    print("=" * 60)
    
    # Create test price data
    np.random.seed(42)
    dates = [datetime.now() - timedelta(days=i) for i in range(50, 0, -1)]
    
    prices = 150 * np.cumprod(1 + np.random.normal(0.001, 0.02, 50))
    
    prices_df = pd.DataFrame({
        "date": dates,
        "ticker": "AAPL",
        "close": prices,
        "volume": np.random.randint(40000000, 60000000, 50),
        "daily_return": np.insert(np.diff(prices) / prices[:-1], 0, np.nan),
        "volatility_10d": np.random.uniform(0.01, 0.03, 50),
        "rsi_14": np.random.uniform(30, 70, 50),
        "macd": np.random.uniform(-2, 2, 50),
        "macd_signal": np.random.uniform(-2, 2, 50),
        "bb_position": np.random.uniform(0, 1, 50),
        "volume_ratio": np.random.uniform(0.5, 2.0, 50),
        "momentum_5d": np.random.uniform(-0.05, 0.05, 50),
        "momentum_20d": np.random.uniform(-0.1, 0.1, 50),
    })
    
    # Create test sentiment data (with gaps)
    sentiment_dates = dates[::2]  # Every other day
    sentiment_df = pd.DataFrame({
        "date": sentiment_dates,
        "ticker": "AAPL",
        "vader_compound": np.random.uniform(-0.5, 0.5, len(sentiment_dates)),
        "vader_positive": np.random.uniform(0.1, 0.5, len(sentiment_dates)),
        "vader_negative": np.random.uniform(0.1, 0.3, len(sentiment_dates)),
        "vader_neutral": np.random.uniform(0.3, 0.6, len(sentiment_dates)),
        "weighted_compound": np.random.uniform(-0.3, 0.3, len(sentiment_dates)),
        "post_count": np.random.randint(5, 20, len(sentiment_dates)),
        "avg_score": np.random.uniform(50, 200, len(sentiment_dates)),
        "total_comments": np.random.randint(20, 100, len(sentiment_dates)),
    })
    
    # Create test trends data
    trends_df = pd.DataFrame({
        "date": dates,
        "ticker": "AAPL",
        "search_interest": np.random.uniform(30, 70, 50),
    })
    
    print(f"\nTest data created:")
    print(f"  Prices: {len(prices_df)} rows")
    print(f"  Sentiment: {len(sentiment_df)} rows (gappy)")
    print(f"  Trends: {len(trends_df)} rows")
    
    # Test feature fusion
    print("\n" + "=" * 60)
    print("Testing Feature Fusion...")
    print("=" * 60)
    
    feature_matrix = merge_all_features(
        prices_df, 
        sentiment_df, 
        trends_df,
        prediction_horizon=3
    )
    
    print(f"\nFeature matrix shape: {feature_matrix.shape}")
    print(f"\nColumns: {list(feature_matrix.columns)}")
    
    # Show sample
    print("\nSample data (first 5 rows):")
    print(feature_matrix[["date", "close", "vader_compound", "vader_compound_lag1", 
                          "sentiment_x_volume", "target_label"]].head())
    
    # Test prepare_for_ml
    print("\n" + "=" * 60)
    print("Testing ML Preparation...")
    print("=" * 60)
    
    X, y, feature_cols = prepare_for_ml(feature_matrix, drop_na=True)
    
    print(f"\nX shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print(f"Features: {len(feature_cols)}")
    print(f"Target distribution: {y.sum()}/{len(y)} positive ({y.mean()*100:.1f}%)")
    
    # Feature groups
    print("\n" + "=" * 60)
    print("Feature Groups:")
    print("=" * 60)
    
    feature_groups = get_feature_importance_order(feature_matrix, feature_cols)
    print(feature_groups.groupby("group").count())
    
    print("\n" + "=" * 60)
    print("✅ Feature fusion test completed!")
