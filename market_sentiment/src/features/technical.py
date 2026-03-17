"""
Technical Indicators Module

Computes technical analysis indicators from OHLCV price data.
These indicators capture price momentum, volatility, and trend signals
that are commonly used in quantitative trading.

Indicators Included:
====================

1. RSI (Relative Strength Index)
   - Measures momentum and overbought/oversold conditions
   - RSI > 70 = overbought (potential reversal down)
   - RSI < 30 = oversold (potential reversal up)
   - Period: 14 days (standard)

2. MACD (Moving Average Convergence Divergence)
   - Trend-following momentum indicator
   - MACD line = EMA(12) - EMA(26)
   - Signal line = EMA(9) of MACD
   - Crossover signals: MACD > Signal = bullish

3. Bollinger Bands
   - Volatility indicator
   - Middle = SMA(20)
   - Upper/Lower = Middle ± 2 * StdDev(20)
   - Squeeze = low volatility, often precedes big moves

4. Volume Indicators
   - Volume SMA(20)
   - Volume ratio = current / SMA
   - Unusual volume (>2x average) = potential breakout

5. Price Momentum
   - 5-day momentum
   - 20-day momentum
   - Distance from 52-week high

6. Moving Averages
   - SMA(20), SMA(50)
   - EMA(12), EMA(26)
   - Price vs SMA signals

All calculations use pandas for efficiency.
First ~30 rows may have NaN due to rolling windows.

DISCLAIMER: Technical indicators are for educational purposes only.
They do not guarantee profitable trading signals.
"""

import numpy as np
import pandas as pd
from loguru import logger
from typing import Optional


# ============================================================
# RSI (Relative Strength Index)
# ============================================================

def compute_rsi(
    series: pd.Series,
    period: int = 14
) -> pd.Series:
    """
    Compute Relative Strength Index (RSI).
    
    RSI measures the speed and magnitude of recent price changes
    to evaluate overbought or oversold conditions.
    
    Formula:
        RSI = 100 - (100 / (1 + RS))
        RS = Average Gain / Average Loss
        Average Gain = EMA of gains over period
        Average Loss = EMA of losses over period
    
    Args:
        series: Price series (typically close prices)
        period: RSI period (default: 14)
    
    Returns:
        RSI values (0-100 scale)
    
    Interpretation:
        RSI > 70: Overbought (potentially overvalued)
        RSI < 30: Oversold (potentially undervalued)
    """
    # Calculate price changes
    delta = series.diff()
    
    # Separate gains and losses
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    
    # Calculate average gain and loss using EMA
    # Using Wilder's smoothing method (similar to EMA)
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
    
    # Calculate RS
    rs = avg_gain / avg_loss
    
    # Calculate RSI
    rsi = 100 - (100 / (1 + rs))
    
    return rsi


def compute_rsi_signal(rsi: pd.Series) -> pd.Series:
    """
    Convert RSI values to categorical signals.
    
    Args:
        rsi: RSI values (0-100)
    
    Returns:
        Categorical series: "overbought", "oversold", "neutral"
    """
    conditions = [
        rsi >= 70,
        rsi <= 30,
    ]
    choices = ["overbought", "oversold"]
    
    return pd.Series(
        np.select(conditions, choices, default="neutral"),
        index=rsi.index
    )


# ============================================================
# MACD (Moving Average Convergence Divergence)
# ============================================================

def compute_macd(
    series: pd.Series,
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Compute MACD (Moving Average Convergence Divergence).
    
    MACD is a trend-following momentum indicator that shows the
    relationship between two exponential moving averages.
    
    Formula:
        MACD Line = EMA(fast) - EMA(slow)
        Signal Line = EMA(signal) of MACD Line
        Histogram = MACD Line - Signal Line
    
    Args:
        series: Price series (typically close prices)
        fast_period: Fast EMA period (default: 12)
        slow_period: Slow EMA period (default: 26)
        signal_period: Signal line period (default: 9)
    
    Returns:
        Tuple of (macd_line, signal_line, histogram)
    
    Interpretation:
        MACD > Signal: Bullish momentum
        MACD < Signal: Bearish momentum
        Histogram positive: Bullish
        Histogram negative: Bearish
    """
    # Calculate EMAs
    ema_fast = series.ewm(span=fast_period, adjust=False).mean()
    ema_slow = series.ewm(span=slow_period, adjust=False).mean()
    
    # MACD line
    macd_line = ema_fast - ema_slow
    
    # Signal line
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
    
    # Histogram
    histogram = macd_line - signal_line
    
    return macd_line, signal_line, histogram


def compute_macd_crossover(
    macd: pd.Series,
    signal: pd.Series
) -> pd.Series:
    """
    Detect MACD crossover signals.
    
    Crossover = when MACD line crosses the signal line.
    Bullish crossover: MACD crosses above signal
    Bearish crossover: MACD crosses below signal
    
    Args:
        macd: MACD line values
        signal: Signal line values
    
    Returns:
        Series with values: 1 (bullish), -1 (bearish), 0 (no crossover)
    """
    # Calculate the difference
    diff = macd - signal
    prev_diff = diff.shift(1)
    
    # Detect crossovers
    bullish_crossover = (prev_diff <= 0) & (diff > 0)
    bearish_crossover = (prev_diff >= 0) & (diff < 0)
    
    crossover = pd.Series(0, index=macd.index)
    crossover[bullish_crossover] = 1
    crossover[bearish_crossover] = -1
    
    return crossover


# ============================================================
# Bollinger Bands
# ============================================================

def compute_bollinger_bands(
    series: pd.Series,
    period: int = 20,
    num_std: float = 2.0
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Compute Bollinger Bands.
    
    Bollinger Bands are volatility bands placed above and below
    a moving average. They widen during high volatility and
    narrow during low volatility.
    
    Formula:
        Middle Band = SMA(period)
        Upper Band = Middle + (num_std * StdDev)
        Lower Band = Middle - (num_std * StdDev)
    
    Args:
        series: Price series (typically close prices)
        period: SMA period (default: 20)
        num_std: Number of standard deviations (default: 2)
    
    Returns:
        Tuple of (upper_band, middle_band, lower_band)
    
    Interpretation:
        Price near upper band: Potentially overbought
        Price near lower band: Potentially oversold
        Narrow bands (squeeze): Low volatility, potential breakout
    """
    # Calculate middle band (SMA)
    middle_band = series.rolling(window=period).mean()
    
    # Calculate standard deviation
    std = series.rolling(window=period).std()
    
    # Calculate bands
    upper_band = middle_band + (num_std * std)
    lower_band = middle_band - (num_std * std)
    
    return upper_band, middle_band, lower_band


def compute_bb_position(
    price: pd.Series,
    upper: pd.Series,
    lower: pd.Series
) -> pd.Series:
    """
    Compute Bollinger Band position.
    
    Measures where the current price is relative to the bands.
    0 = at lower band, 1 = at upper band, 0.5 = at middle
    
    Args:
        price: Price series
        upper: Upper Bollinger Band
        lower: Lower Bollinger Band
    
    Returns:
        Position values (0-1 scale)
    """
    band_width = upper - lower
    position = (price - lower) / band_width
    
    # Clip to reasonable range (can go outside bands)
    return position


def compute_bb_squeeze(
    upper: pd.Series,
    lower: pd.Series,
    middle: pd.Series
) -> pd.Series:
    """
    Compute Bollinger Band squeeze indicator.
    
    Squeeze measures the width of the bands relative to the middle.
    Low squeeze = narrow bands = low volatility = potential breakout.
    
    Args:
        upper: Upper Bollinger Band
        lower: Lower Bollinger Band
        middle: Middle Bollinger Band
    
    Returns:
        Squeeze values (band width as % of middle)
    """
    squeeze = (upper - lower) / middle
    return squeeze


# ============================================================
# Volume Indicators
# ============================================================

def compute_volume_sma(
    volume: pd.Series,
    period: int = 20
) -> pd.Series:
    """
    Compute Volume Simple Moving Average.
    
    Args:
        volume: Volume series
        period: SMA period (default: 20)
    
    Returns:
        Volume SMA values
    """
    return volume.rolling(window=period).mean()


def compute_volume_ratio(
    volume: pd.Series,
    volume_sma: pd.Series
) -> pd.Series:
    """
    Compute Volume Ratio.
    
    Ratio of current volume to average volume.
    High ratio (>2) indicates unusual activity.
    
    Args:
        volume: Volume series
        volume_sma: Volume SMA series
    
    Returns:
        Volume ratio values
    
    Interpretation:
        Ratio > 2.0: Unusually high volume (potential breakout)
        Ratio < 0.5: Low volume (weak conviction)
    """
    return volume / volume_sma


# ============================================================
# Price Momentum
# ============================================================

def compute_momentum(
    series: pd.Series,
    period: int
) -> pd.Series:
    """
    Compute Price Momentum (Rate of Change).
    
    Momentum = (Current Price / Price N days ago) - 1
    
    Args:
        series: Price series
        period: Lookback period
    
    Returns:
        Momentum values (as decimal, e.g., 0.05 = 5%)
    """
    return series / series.shift(period) - 1


def compute_price_vs_high(
    series: pd.Series,
    lookback: int = 252
) -> pd.Series:
    """
    Compute Price vs 52-Week High.
    
    Ratio of current price to highest price in lookback period.
    1.0 = at high, <1.0 = below high
    
    Args:
        series: Price series
        lookback: Lookback period (default: 252 trading days = 1 year)
    
    Returns:
        Price vs high ratio
    """
    rolling_high = series.rolling(window=lookback).max()
    return series / rolling_high


# ============================================================
# Moving Averages
# ============================================================

def compute_sma(
    series: pd.Series,
    period: int
) -> pd.Series:
    """
    Compute Simple Moving Average.
    
    Args:
        series: Price series
        period: SMA period
    
    Returns:
        SMA values
    """
    return series.rolling(window=period).mean()


def compute_ema(
    series: pd.Series,
    period: int
) -> pd.Series:
    """
    Compute Exponential Moving Average.
    
    EMA gives more weight to recent prices.
    
    Args:
        series: Price series
        period: EMA period
    
    Returns:
        EMA values
    """
    return series.ewm(span=period, adjust=False).mean()


def compute_price_vs_sma(
    price: pd.Series,
    sma: pd.Series
) -> pd.Series:
    """
    Compute Price relative to SMA.
    
    Positive = price above SMA (bullish)
    Negative = price below SMA (bearish)
    
    Args:
        price: Price series
        sma: SMA series
    
    Returns:
        Price vs SMA (as decimal)
    """
    return (price - sma) / sma


# ============================================================
# Volatility Indicators
# ============================================================

def compute_atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14
) -> pd.Series:
    """
    Compute Average True Range (ATR).
    
    ATR measures market volatility by decomposing the entire
    range of an asset price for that period.
    
    True Range = max(high-low, abs(high-prev_close), abs(low-prev_close))
    ATR = EMA of True Range
    
    Args:
        high: High prices
        low: Low prices
        close: Close prices
        period: ATR period (default: 14)
    
    Returns:
        ATR values
    """
    prev_close = close.shift(1)
    
    tr1 = high - low
    tr2 = abs(high - prev_close)
    tr3 = abs(low - prev_close)
    
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = true_range.ewm(span=period, adjust=False).mean()
    
    return atr


def compute_volatility(
    returns: pd.Series,
    period: int = 20
) -> pd.Series:
    """
    Compute Rolling Volatility (Standard Deviation of Returns).
    
    Args:
        returns: Returns series
        period: Rolling window
    
    Returns:
        Volatility values
    """
    return returns.rolling(window=period).std()


# ============================================================
# Main Feature Engineering Function
# ============================================================

def compute_technical_features(
    df: pd.DataFrame,
    include_all: bool = True
) -> pd.DataFrame:
    """
    Compute all technical indicators from OHLCV data.
    
    This is the main function that adds all technical indicators
    to the price DataFrame.
    
    Args:
        df: DataFrame with columns: date, open, high, low, close, volume
        include_all: Whether to include all indicators (default: True)
    
    Returns:
        DataFrame with additional technical indicator columns:
        - RSI: rsi_14, rsi_signal
        - MACD: macd, macd_signal, macd_hist, macd_crossover
        - Bollinger: bb_upper, bb_mid, bb_lower, bb_position, bb_squeeze
        - Volume: volume_sma_20, volume_ratio
        - Momentum: momentum_5d, momentum_20d, price_vs_52w_high
        - Moving Averages: sma_20, sma_50, ema_12, ema_26
        - Volatility: volatility_20d, atr_14
    
    Note:
        First ~30 rows will have NaN values due to rolling windows.
        These should be dropped before ML training.
    """
    if df.empty:
        logger.warning("Empty DataFrame, skipping technical feature computation")
        return df
    
    df = df.copy()
    
    # Ensure we have required columns
    required = ["close"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    logger.info(f"Computing technical features for {len(df)} rows")
    
    # ============================================================
    # RSI (Relative Strength Index)
    # ============================================================
    logger.debug("Computing RSI...")
    df["rsi_14"] = compute_rsi(df["close"], period=14)
    df["rsi_signal"] = compute_rsi_signal(df["rsi_14"])
    
    # ============================================================
    # MACD (Moving Average Convergence Divergence)
    # ============================================================
    logger.debug("Computing MACD...")
    df["macd"], df["macd_signal"], df["macd_hist"] = compute_macd(df["close"])
    df["macd_crossover"] = compute_macd_crossover(df["macd"], df["macd_signal"])
    
    # ============================================================
    # Bollinger Bands
    # ============================================================
    logger.debug("Computing Bollinger Bands...")
    df["bb_upper"], df["bb_mid"], df["bb_lower"] = compute_bollinger_bands(df["close"])
    df["bb_position"] = compute_bb_position(df["close"], df["bb_upper"], df["bb_lower"])
    df["bb_squeeze"] = compute_bb_squeeze(df["bb_upper"], df["bb_lower"], df["bb_mid"])
    
    # ============================================================
    # Volume Indicators
    # ============================================================
    if "volume" in df.columns:
        logger.debug("Computing Volume indicators...")
        df["volume_sma_20"] = compute_volume_sma(df["volume"], period=20)
        df["volume_ratio"] = compute_volume_ratio(df["volume"], df["volume_sma_20"])
    
    # ============================================================
    # Price Momentum
    # ============================================================
    logger.debug("Computing Price Momentum...")
    df["momentum_5d"] = compute_momentum(df["close"], period=5)
    df["momentum_20d"] = compute_momentum(df["close"], period=20)
    df["price_vs_52w_high"] = compute_price_vs_high(df["close"], lookback=252)
    
    # ============================================================
    # Moving Averages
    # ============================================================
    logger.debug("Computing Moving Averages...")
    df["sma_20"] = compute_sma(df["close"], period=20)
    df["sma_50"] = compute_sma(df["close"], period=50)
    df["ema_12"] = compute_ema(df["close"], period=12)
    df["ema_26"] = compute_ema(df["close"], period=26)
    
    # Price relative to moving averages
    df["price_vs_sma_20"] = compute_price_vs_sma(df["close"], df["sma_20"])
    df["price_vs_sma_50"] = compute_price_vs_sma(df["close"], df["sma_50"])
    
    # ============================================================
    # Volatility
    # ============================================================
    logger.debug("Computing Volatility...")
    if "daily_return" in df.columns:
        df["volatility_20d"] = compute_volatility(df["daily_return"], period=20)
    else:
        # Calculate returns first
        returns = df["close"].pct_change()
        df["volatility_20d"] = compute_volatility(returns, period=20)
    
    # ATR (requires OHLC)
    if all(c in df.columns for c in ["high", "low", "close"]):
        df["atr_14"] = compute_atr(df["high"], df["low"], df["close"], period=14)
    
    # ============================================================
    # Post-Processing
    # ============================================================
    
    # Count NaN rows (due to rolling windows)
    nan_rows = df.isnull().any(axis=1).sum()
    logger.info(f"Technical features computed. {nan_rows} rows have NaN values (expected for first ~30 rows)")
    
    return df


def get_feature_summary(df: pd.DataFrame) -> dict:
    """
    Get summary statistics for computed features.
    
    Args:
        df: DataFrame with technical features
    
    Returns:
        Dictionary with feature statistics
    """
    feature_cols = [
        "rsi_14", "macd", "macd_hist", "bb_position", "bb_squeeze",
        "volume_ratio", "momentum_5d", "momentum_20d", "price_vs_52w_high",
        "price_vs_sma_20", "volatility_20d"
    ]
    
    summary = {}
    for col in feature_cols:
        if col in df.columns:
            summary[col] = {
                "mean": df[col].mean(),
                "std": df[col].std(),
                "min": df[col].min(),
                "max": df[col].max(),
                "nan_count": df[col].isna().sum(),
            }
    
    return summary


# ============================================================
# Module Test
# ============================================================

if __name__ == "__main__":
    from datetime import datetime, timedelta
    
    print("Technical Indicators Test")
    print("=" * 60)
    
    # Create test data
    np.random.seed(42)
    dates = [datetime.now() - timedelta(days=i) for i in range(100, 0, -1)]
    
    # Simulate price data with trend
    base_price = 150
    returns = np.random.normal(0.001, 0.02, 100)  # Slight upward drift
    prices = base_price * np.cumprod(1 + returns)
    
    test_df = pd.DataFrame({
        "date": dates,
        "ticker": "AAPL",
        "open": prices * (1 + np.random.uniform(-0.01, 0.01, 100)),
        "high": prices * (1 + np.random.uniform(0, 0.02, 100)),
        "low": prices * (1 - np.random.uniform(0, 0.02, 100)),
        "close": prices,
        "volume": np.random.randint(40000000, 60000000, 100),
    })
    
    # Calculate returns
    test_df["daily_return"] = test_df["close"].pct_change()
    
    print(f"\nTest data: {len(test_df)} rows")
    print(test_df[["date", "close", "volume"]].head())
    
    # Compute technical features
    print("\n" + "=" * 60)
    print("Computing Technical Features...")
    print("=" * 60)
    
    df_features = compute_technical_features(test_df)
    
    # Show feature columns
    feature_cols = [c for c in df_features.columns if c not in ["date", "ticker", "open", "high", "low", "close", "volume", "daily_return"]]
    print(f"\nAdded {len(feature_cols)} feature columns:")
    print(feature_cols)
    
    # Show sample
    print("\nSample data (last 5 rows):")
    print(df_features[["date", "close", "rsi_14", "macd", "bb_position", "volume_ratio"]].tail())
    
    # RSI distribution
    print("\n" + "=" * 60)
    print("RSI Distribution:")
    print("=" * 60)
    rsi_valid = df_features["rsi_14"].dropna()
    print(f"  Overbought (>70): {(rsi_valid > 70).sum()}")
    print(f"  Oversold (<30): {(rsi_valid < 30).sum()}")
    print(f"  Neutral: {((rsi_valid >= 30) & (rsi_valid <= 70)).sum()}")
    
    # Feature summary
    print("\n" + "=" * 60)
    print("Feature Summary:")
    print("=" * 60)
    summary = get_feature_summary(df_features)
    for col, stats in summary.items():
        print(f"\n{col}:")
        print(f"  Mean: {stats['mean']:.4f}")
        print(f"  Std: {stats['std']:.4f}")
        print(f"  Range: [{stats['min']:.4f}, {stats['max']:.4f}]")
        print(f"  NaN: {stats['nan_count']}")
    
    print("\n" + "=" * 60)
    print("✅ Technical indicators test completed!")
