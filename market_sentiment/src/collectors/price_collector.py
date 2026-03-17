"""
Price Collector Module

Fetches OHLCV (Open, High, Low, Close, Volume) data from Yahoo Finance.
Uses yfinance library which provides free, no-API-key access to historical data.

Features:
- Caching to minimize API calls (24-hour cache validity)
- Automatic calculation of returns and volatility
- Market context (SPY benchmark) retrieval
- Handles missing data and splits automatically

DISCLAIMER: Data from Yahoo Finance is for educational purposes only.
May have 15-minute delay for real-time data.
"""

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf
from loguru import logger


class PriceCollector:
    """
    Collects stock price data from Yahoo Finance.
    
    This class handles:
    - Fetching historical OHLCV data
    - Computing derived metrics (returns, volatility)
    - Caching data locally to minimize API calls
    - Market benchmark data (SPY for S&P 500)
    
    Attributes:
        tickers: List of stock ticker symbols to track
        cache_dir: Directory for caching raw data
        cache_expiry_hours: Hours before cache is considered stale
    
    Example:
        >>> collector = PriceCollector(['AAPL', 'TSLA'], './data/raw')
        >>> prices = collector.fetch_all()
        >>> print(prices['AAPL'].shape)
        (252, 10)
    """
    
    def __init__(
        self,
        tickers: list[str],
        cache_dir: str,
        cache_expiry_hours: int = 24
    ) -> None:
        """
        Initialize the price collector.
        
        Args:
            tickers: List of ticker symbols (e.g., ['AAPL', 'TSLA'])
            cache_dir: Directory path for caching CSV files
            cache_expiry_hours: Hours before cached data is considered stale
        """
        self.tickers = [t.upper() for t in tickers]
        self.cache_dir = Path(cache_dir) / "prices"
        self.cache_expiry_hours = cache_expiry_hours
        
        # Ensure cache directory exists
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(
            f"PriceCollector initialized for {len(self.tickers)} tickers: "
            f"{', '.join(self.tickers)}"
        )
    
    def _get_cache_path(self, ticker: str) -> Path:
        """Get the cache file path for a ticker."""
        return self.cache_dir / f"{ticker.upper()}.csv"
    
    def _is_cache_valid(self, cache_path: Path) -> bool:
        """
        Check if cached file exists and is recent enough.
        
        Args:
            cache_path: Path to the cached file
            
        Returns:
            True if cache exists and is less than expiry_hours old
        """
        if not cache_path.exists():
            return False
        
        # Check file modification time
        file_mtime = datetime.fromtimestamp(cache_path.stat().st_mtime)
        age = datetime.now() - file_mtime
        
        return age < timedelta(hours=self.cache_expiry_hours)
    
    def _load_from_cache(self, ticker: str) -> Optional[pd.DataFrame]:
        """
        Load price data from cache if available and valid.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            DataFrame with price data, or None if cache invalid
        """
        cache_path = self._get_cache_path(ticker)
        
        if self._is_cache_valid(cache_path):
            try:
                df = pd.read_csv(cache_path, parse_dates=["date"])
                logger.debug(f"Loaded {ticker} from cache ({len(df)} rows)")
                return df
            except Exception as e:
                logger.warning(f"Failed to load cache for {ticker}: {e}")
        
        return None
    
    def _save_to_cache(self, ticker: str, df: pd.DataFrame) -> None:
        """
        Save price data to cache.
        
        Args:
            ticker: Stock ticker symbol
            df: DataFrame with price data
        """
        cache_path = self._get_cache_path(ticker)
        try:
            df.to_csv(cache_path, index=False)
            logger.debug(f"Saved {ticker} to cache: {cache_path}")
        except Exception as e:
            logger.warning(f"Failed to save cache for {ticker}: {e}")
    
    def fetch_prices(
        self,
        ticker: str,
        period: str = "1y",
        use_cache: bool = True
    ) -> pd.DataFrame:
        """
        Fetch historical price data for a single ticker.
        
        Uses yfinance.download() which returns data with:
        - Open, High, Low, Close, Volume
        - auto_adjust=True adjusts for splits and dividends
        
        Args:
            ticker: Stock ticker symbol (e.g., "AAPL")
            period: Time period to fetch ("1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max")
            use_cache: Whether to use cached data if available
            
        Returns:
            DataFrame with columns:
            - date: Trading date
            - ticker: Stock symbol
            - open, high, low, close, volume: OHLCV data
            - daily_return: Percentage return from previous close
            - log_return: Logarithmic return (better for statistics)
            - volatility_10d: 10-day rolling standard deviation of returns
        
        Raises:
            ValueError: If ticker is empty or invalid
        """
        ticker = ticker.upper()
        
        if not ticker:
            raise ValueError("Ticker cannot be empty")
        
        # Try to load from cache first
        if use_cache:
            cached_df = self._load_from_cache(ticker)
            if cached_df is not None:
                logger.info(
                    f"Loaded {ticker} from cache ({len(cached_df)} trading days)"
                )
                return cached_df
        
        logger.info(f"Fetching {ticker} price data for period: {period}")
        
        try:
            # Download data from Yahoo Finance
            # auto_adjust=True adjusts prices for splits and dividends
            df = yf.download(
                ticker,
                period=period,
                progress=False,
                auto_adjust=True
            )
            
            if df.empty:
                logger.warning(f"No data returned for {ticker}")
                return pd.DataFrame()
            
            # Handle MultiIndex columns (yfinance 0.2.x behavior)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            
            # Flatten column names to lowercase
            df.columns = [col.lower() for col in df.columns]
            
            # Reset index to get date as column
            df = df.reset_index()
            df = df.rename(columns={"Date": "date", "index": "date"})
            
            # Ensure date column is datetime
            if "date" not in df.columns:
                # Find the date column (might be named differently)
                date_cols = [c for c in df.columns if "date" in c.lower()]
                if date_cols:
                    df = df.rename(columns={date_cols[0]: "date"})
                else:
                    # Use index
                    df["date"] = df.index
            
            df["date"] = pd.to_datetime(df["date"])
            
            # Add ticker column
            df["ticker"] = ticker
            
            # Calculate derived metrics
            # Daily return: percentage change from previous close
            # This measures day-over-day price movement
            df["daily_return"] = df["close"].pct_change()
            
            # Log return: log(current_price / previous_price)
            # Log returns are more suitable for statistical analysis
            # because they are additive over time
            df["log_return"] = np.log(df["close"] / df["close"].shift(1))
            
            # 10-day volatility: rolling standard deviation of daily returns
            # This measures recent price volatility
            # Higher volatility = riskier asset
            df["volatility_10d"] = df["daily_return"].rolling(window=10).std()
            
            # Select and order columns
            columns = [
                "date", "ticker", "open", "high", "low", "close", "volume",
                "daily_return", "log_return", "volatility_10d"
            ]
            df = df[[c for c in columns if c in df.columns]]
            
            # Drop rows with all NaN values (can happen at the start)
            df = df.dropna(how="all")
            
            # Save to cache
            self._save_to_cache(ticker, df)
            
            logger.info(
                f"Fetched {len(df)} days of {ticker} price data "
                f"({df['date'].min().date()} to {df['date'].max().date()})"
            )
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to fetch {ticker} price data: {e}")
            raise
    
    def fetch_all(
        self,
        period: str = "1y",
        use_cache: bool = True
    ) -> dict[str, pd.DataFrame]:
        """
        Fetch price data for all configured tickers.
        
        Iterates through all tickers and fetches data for each.
        Uses caching to minimize API calls.
        
        Args:
            period: Time period to fetch for all tickers
            use_cache: Whether to use cached data
            
        Returns:
            Dictionary mapping ticker to DataFrame:
            {"AAPL": df_aapl, "TSLA": df_tsla, ...}
        """
        results = {}
        
        for ticker in self.tickers:
            try:
                df = self.fetch_prices(ticker, period=period, use_cache=use_cache)
                if not df.empty:
                    results[ticker] = df
            except Exception as e:
                logger.error(f"Failed to fetch {ticker}: {e}")
                continue
        
        logger.info(
            f"Fetched price data for {len(results)}/{len(self.tickers)} tickers"
        )
        
        return results
    
    def get_market_context(
        self,
        period: str = "1y",
        use_cache: bool = True
    ) -> pd.DataFrame:
        """
        Fetch SPY (S&P 500 ETF) as market benchmark.
        
        SPY is used as the market proxy for:
        - Calculating beta (stock volatility relative to market)
        - Benchmark comparisons in backtesting
        - Market-relative performance metrics
        
        Args:
            period: Time period to fetch
            use_cache: Whether to use cached data
            
        Returns:
            DataFrame with SPY price data
        """
        logger.info("Fetching SPY market benchmark data")
        
        spy_data = self.fetch_prices("SPY", period=period, use_cache=use_cache)
        
        if spy_data.empty:
            logger.warning("Failed to fetch SPY data, market context unavailable")
        
        return spy_data
    
    def get_latest_prices(self) -> dict[str, dict]:
        """
        Get the latest price for each ticker.
        
        Useful for dashboard displays and quick lookups.
        
        Returns:
            Dictionary mapping ticker to latest price info:
            {
                "AAPL": {
                    "close": 178.50,
                    "change": 2.30,
                    "change_pct": 1.31,
                    "volume": 50000000,
                    "date": "2024-01-15"
                },
                ...
            }
        """
        results = {}
        
        for ticker in self.tickers:
            try:
                df = self.fetch_prices(ticker, period="5d", use_cache=True)
                
                if df.empty or len(df) < 2:
                    continue
                
                latest = df.iloc[-1]
                previous = df.iloc[-2]
                
                results[ticker] = {
                    "close": float(latest["close"]),
                    "change": float(latest["close"] - previous["close"]),
                    "change_pct": float(
                        (latest["close"] - previous["close"]) / previous["close"] * 100
                    ),
                    "volume": int(latest["volume"]),
                    "date": latest["date"].strftime("%Y-%m-%d"),
                }
                
            except Exception as e:
                logger.warning(f"Failed to get latest price for {ticker}: {e}")
                continue
        
        return results
    
    def calculate_beta(
        self,
        ticker: str,
        market_data: Optional[pd.DataFrame] = None,
        period: str = "1y"
    ) -> float:
        """
        Calculate beta (systematic risk) relative to market.
        
        Beta measures a stock's volatility relative to the overall market:
        - Beta > 1: More volatile than market (aggressive)
        - Beta < 1: Less volatile than market (defensive)
        - Beta = 1: Same volatility as market
        
        Formula: Beta = Cov(stock, market) / Var(market)
        
        Args:
            ticker: Stock ticker symbol
            market_data: Market benchmark data (SPY). If None, fetches automatically.
            period: Time period for calculation
            
        Returns:
            Beta value (float)
        """
        stock_data = self.fetch_prices(ticker, period=period)
        
        if stock_data.empty:
            logger.warning(f"Cannot calculate beta: no data for {ticker}")
            return 1.0
        
        if market_data is None:
            market_data = self.get_market_context(period=period)
        
        if market_data.empty:
            logger.warning("Cannot calculate beta: no market data")
            return 1.0
        
        # Merge on date to align data
        merged = pd.merge(
            stock_data[["date", "daily_return"]],
            market_data[["date", "daily_return"]],
            on="date",
            suffixes=("_stock", "_market")
        )
        
        # Drop NaN values
        merged = merged.dropna()
        
        if len(merged) < 30:  # Need sufficient data points
            logger.warning("Insufficient data for beta calculation")
            return 1.0
        
        # Calculate covariance matrix
        cov_matrix = merged[["daily_return_stock", "daily_return_market"]].cov()
        
        # Beta = Cov(stock, market) / Var(market)
        beta = cov_matrix.iloc[0, 1] / cov_matrix.iloc[1, 1]
        
        logger.info(f"{ticker} beta: {beta:.2f}")
        
        return beta


# Convenience function for direct use
def fetch_stock_prices(
    tickers: list[str],
    period: str = "1y",
    cache_dir: str = "./data/raw"
) -> dict[str, pd.DataFrame]:
    """
    Convenience function to fetch prices for multiple tickers.
    
    Args:
        tickers: List of ticker symbols
        period: Time period to fetch
        cache_dir: Directory for caching
        
    Returns:
        Dictionary mapping ticker to DataFrame
    """
    collector = PriceCollector(tickers, cache_dir)
    return collector.fetch_all(period=period)


if __name__ == "__main__":
    # Test the collector
    from src.config import settings
    
    collector = PriceCollector(settings.tickers[:3], settings.cache_dir)
    
    # Test single ticker
    print("\nTesting single ticker fetch (AAPL):")
    df = collector.fetch_prices("AAPL", period="1mo")
    print(df.head())
    print(f"\nShape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    # Test latest prices
    print("\nLatest prices:")
    latest = collector.get_latest_prices()
    for ticker, info in latest.items():
        print(f"{ticker}: ${info['close']:.2f} ({info['change_pct']:+.2f}%)")
