"""
Google Trends Collector Module

Collects search interest data from Google Trends using the pytrends library.
Provides insight into public interest in specific stocks over time.

Features:
- Search interest over time (0-100 normalized)
- Multiple keyword queries per ticker
- Rate limit handling (Google Trends is aggressive!)
- Weekly to daily interpolation
- Caching with expiry

DISCLAIMER: Google Trends data is normalized (0-100) and for educational use.
Weekly granularity is interpolated to daily - this introduces approximation.
"""

import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger

# Note: pytrends has issues with Python 3.11+, using try/except
try:
    from pytrends.request import TrendReq
    PYTRENDS_AVAILABLE = True
except ImportError:
    PYTRENDS_AVAILABLE = False
    logger.warning(
        "pytrends not available. Google Trends collection will be simulated. "
        "Install with: pip install pytrends"
    )


class TrendsCollector:
    """
    Collects Google Trends search interest data.
    
    Google Trends provides search interest normalized to 0-100 scale:
    - 100 = Peak interest in the time period
    - 50 = Half the interest of the peak
    - 0 = Insufficient data
    
    Note: Google Trends is aggressive with rate limiting.
    We add significant delays between requests.
    
    Attributes:
        cache_dir: Directory for caching raw data
        rate_limit_delay: Seconds to wait between API calls
        pytrends: TrendReq instance (if available)
    
    Example:
        >>> collector = TrendsCollector('./data/raw')
        >>> df = collector.fetch_interest("AAPL", period_months=12)
        >>> print(df.head())
    """
    
    # Google Trends rate limits aggressively - use long delays
    RATE_LIMIT_DELAY = 5.0  # 5 seconds between requests
    
    # Maximum retries on failure
    MAX_RETRIES = 3
    
    def __init__(
        self,
        cache_dir: str,
        rate_limit_delay: float = 5.0
    ) -> None:
        """
        Initialize the trends collector.
        
        Args:
            cache_dir: Directory path for caching CSV files
            rate_limit_delay: Seconds to wait between API calls
        """
        self.cache_dir = Path(cache_dir) / "trends"
        self.rate_limit_delay = rate_limit_delay
        
        # Ensure cache directory exists
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize pytrends if available
        self.pytrends = None
        if PYTRENDS_AVAILABLE:
            try:
                # hl = language, tz = timezone (360 = CST/UTC-6)
                self.pytrends = TrendReq(
                    hl="en-US",
                    tz=360,  # Central time
                    timeout=(10, 25),  # Connect, read timeout
                    retries=2,
                    backoff_factor=0.5
                )
                logger.info("TrendsCollector initialized with pytrends")
            except Exception as e:
                logger.warning(f"Failed to initialize pytrends: {e}")
        else:
            logger.warning(
                "TrendsCollector initialized in simulation mode "
                "(pytrends not available)"
            )
    
    def _get_cache_path(self, ticker: str) -> Path:
        """Get the cache file path for a ticker."""
        return self.cache_dir / f"{ticker.upper()}_trends.csv"
    
    def _is_cache_valid(self, cache_path: Path, max_age_hours: int = 168) -> bool:
        """
        Check if cached file exists and is recent enough.
        
        Google Trends data changes slowly, so we use longer cache times.
        Default: 1 week (168 hours)
        
        Args:
            cache_path: Path to the cached file
            max_age_hours: Maximum age in hours
            
        Returns:
            True if cache exists and is recent
        """
        if not cache_path.exists():
            return False
        
        file_mtime = datetime.fromtimestamp(cache_path.stat().st_mtime)
        age = datetime.now() - file_mtime
        
        return age < timedelta(hours=max_age_hours)
    
    def _load_from_cache(self, ticker: str) -> Optional[pd.DataFrame]:
        """
        Load trends data from cache if available.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            DataFrame with trends data, or None if cache invalid
        """
        cache_path = self._get_cache_path(ticker)
        
        if self._is_cache_valid(cache_path):
            try:
                df = pd.read_csv(cache_path, parse_dates=["date"])
                logger.debug(f"Loaded {ticker} trends from cache ({len(df)} rows)")
                return df
            except Exception as e:
                logger.warning(f"Failed to load cache for {ticker}: {e}")
        
        return None
    
    def _save_to_cache(self, ticker: str, df: pd.DataFrame) -> None:
        """
        Save trends data to cache.
        
        Args:
            ticker: Stock ticker symbol
            df: DataFrame with trends data
        """
        cache_path = self._get_cache_path(ticker)
        try:
            df.to_csv(cache_path, index=False)
            logger.debug(f"Saved {ticker} trends to cache: {cache_path}")
        except Exception as e:
            logger.warning(f"Failed to save cache for {ticker}: {e}")
    
    def _build_timeframe(self, period_months: int) -> str:
        """
        Build timeframe string for pytrends.
        
        Format: "YYYY-MM-DD YYYY-MM-DD"
        
        Args:
            period_months: Number of months to look back
            
        Returns:
            Timeframe string
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=period_months * 30)
        
        return f"{start_date.strftime('%Y-%m-%d')} {end_date.strftime('%Y-%m-%d')}"
    
    def _interpolate_weekly_to_daily(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Interpolate weekly trends data to daily.
        
        Google Trends often returns weekly data.
        We interpolate to daily for alignment with price data.
        
        Note: This introduces approximation. We use linear interpolation.
        
        Args:
            df: DataFrame with weekly data
            
        Returns:
            DataFrame with daily data
        """
        if df.empty:
            return df
        
        # Ensure date is datetime
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")
        
        # Resample to daily and interpolate
        # Linear interpolation is reasonable for search interest
        daily_df = df.resample("D").interpolate(method="linear")
        
        # Reset index
        daily_df = daily_df.reset_index()
        
        logger.debug(f"Interpolated {len(df)} weekly points to {len(daily_df)} daily")
        
        return daily_df
    
    def _generate_simulated_data(
        self,
        ticker: str,
        period_months: int
    ) -> pd.DataFrame:
        """
        Generate simulated trends data for testing.
        
        Used when pytrends is unavailable or for testing.
        Creates realistic-looking search interest data.
        
        Args:
            ticker: Stock ticker symbol
            period_months: Number of months of data
            
        Returns:
            DataFrame with simulated trends data
        """
        logger.warning(f"Generating SIMULATED trends data for {ticker}")
        
        # Create date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=period_months * 30)
        dates = pd.date_range(start=start_date, end=end_date, freq="D")
        
        # Generate realistic-looking search interest
        # Base level around 40-60 with some variation
        np.random.seed(hash(ticker) % 2**32)  # Reproducible by ticker
        
        n = len(dates)
        base = np.random.uniform(40, 60)  # Base interest level
        
        # Add trend (slight increase/decrease over time)
        trend = np.linspace(0, np.random.uniform(-10, 15), n)
        
        # Add weekly seasonality (higher on weekdays)
        weekday_factor = np.array([
            1.0 if d.weekday() < 5 else 0.7 for d in dates
        ])
        
        # Add random noise
        noise = np.random.normal(0, 5, n)
        
        # Add occasional spikes (news events)
        spike_prob = 0.02  # 2% chance of spike
        spikes = np.random.choice([0, 30], size=n, p=[1-spike_prob, spike_prob])
        
        # Combine components
        interest = base + trend + noise + spikes
        interest = interest * weekday_factor
        
        # Normalize to 0-100 scale
        interest = np.clip(interest, 0, 100)
        interest = (interest / interest.max() * 100).round(1)
        
        df = pd.DataFrame({
            "date": dates,
            "ticker": ticker.upper(),
            "search_interest": interest,
            "is_simulated": True  # Flag for transparency
        })
        
        return df
    
    def fetch_interest(
        self,
        ticker: str,
        period_months: int = 12,
        keywords: Optional[list[str]] = None,
        use_cache: bool = True
    ) -> pd.DataFrame:
        """
        Fetch search interest data for a ticker.
        
        Searches for multiple keywords:
        - Ticker symbol (e.g., "AAPL")
        - Ticker + "stock" (e.g., "AAPL stock")
        - Ticker + "buy" (e.g., "AAPL buy")
        
        Returns the highest interest among all keywords.
        
        Args:
            ticker: Stock ticker symbol
            period_months: Number of months to fetch (default: 12)
            keywords: Custom keyword list (default: auto-generated)
            use_cache: Whether to use cached data
            
        Returns:
            DataFrame with columns:
            - date: Date
            - ticker: Stock symbol
            - search_interest: Normalized interest (0-100)
            - keyword: The keyword that generated the interest
        """
        ticker = ticker.upper()
        
        # Try to load from cache first
        if use_cache:
            cached_df = self._load_from_cache(ticker)
            if cached_df is not None:
                logger.info(
                    f"Loaded {ticker} trends from cache ({len(cached_df)} days)"
                )
                return cached_df
        
        # Check if pytrends is available
        if self.pytrends is None:
            return self._generate_simulated_data(ticker, period_months)
        
        # Default keywords to search
        if keywords is None:
            keywords = [
                ticker,
                f"{ticker} stock",
                f"{ticker} buy",
            ]
        
        logger.info(
            f"Fetching Google Trends for {ticker} "
            f"(keywords: {keywords}, period: {period_months} months)"
        )
        
        timeframe = self._build_timeframe(period_months)
        
        all_interest = []
        
        for i, keyword in enumerate(keywords):
            try:
                logger.debug(f"Fetching keyword: {keyword}")
                
                # Build payload
                self.pytrends.build_payload(
                    kw_list=[keyword],
                    timeframe=timeframe,
                    cat=0,  # All categories
                    gprop=""  # Web search (default)
                )
                
                # Get interest over time
                interest_df = self.pytrends.build_payload(
                    kw_list=[keyword],
                    timeframe=timeframe
                )
                interest_df = self.pytrends.interest_over_time()
                
                if interest_df.empty:
                    logger.debug(f"No data for keyword: {keyword}")
                    continue
                
                # Process the data
                # interest_df has columns: [keyword, isPartial]
                interest_df = interest_df.reset_index()
                interest_df = interest_df.rename(columns={"date": "date", keyword: "interest"})
                
                # Filter out partial data
                if "isPartial" in interest_df.columns:
                    interest_df = interest_df[interest_df["isPartial"] == False]
                
                interest_df["keyword"] = keyword
                interest_df = interest_df[["date", "interest", "keyword"]]
                interest_df = interest_df.rename(columns={"interest": "search_interest"})
                
                all_interest.append(interest_df)
                
                logger.debug(f"Got {len(interest_df)} data points for {keyword}")
                
                # Rate limit delay between keywords
                if i < len(keywords) - 1:
                    time.sleep(self.rate_limit_delay)
                    
            except Exception as e:
                logger.warning(f"Failed to fetch trends for '{keyword}': {e}")
                
                # Wait longer on error (might be rate limited)
                time.sleep(self.rate_limit_delay * 2)
                continue
        
        if not all_interest:
            logger.warning(f"No trends data available for {ticker}, using simulation")
            return self._generate_simulated_data(ticker, period_months)
        
        # Combine all keywords
        combined = pd.concat(all_interest, ignore_index=True)
        
        # Group by date and take max interest
        # This captures the highest interest across all keywords
        daily_df = combined.groupby("date").agg(
            search_interest=("search_interest", "max"),
            keyword=("keyword", "first")  # Keep one keyword as reference
        ).reset_index()
        
        # Interpolate if weekly data
        # Check if data has gaps (weekly vs daily)
        date_diffs = daily_df["date"].diff().dropna()
        if date_diffs.dt.days.median() > 1:
            daily_df = self._interpolate_weekly_to_daily(daily_df)
        
        # Add ticker column
        daily_df["ticker"] = ticker
        daily_df["is_simulated"] = False
        
        # Ensure column order
        daily_df = daily_df[["date", "ticker", "search_interest", "is_simulated"]]
        
        # Save to cache
        self._save_to_cache(ticker, daily_df)
        
        logger.info(
            f"Fetched {len(daily_df)} days of trends data for {ticker}"
        )
        
        return daily_df
    
    def fetch_all(
        self,
        tickers: list[str],
        period_months: int = 12,
        use_cache: bool = True
    ) -> dict[str, pd.DataFrame]:
        """
        Fetch trends data for all tickers.
        
        Adds significant delays between requests to avoid rate limiting.
        Google Trends is very aggressive - we need to be careful.
        
        Args:
            tickers: List of ticker symbols
            period_months: Number of months to fetch
            use_cache: Whether to use cached data
            
        Returns:
            Dictionary mapping ticker to DataFrame
        """
        results = {}
        
        logger.info(
            f"Starting Google Trends collection for {len(tickers)} tickers"
        )
        
        for i, ticker in enumerate(tickers, 1):
            logger.info(f"Processing {ticker} ({i}/{len(tickers)})")
            
            try:
                df = self.fetch_interest(
                    ticker,
                    period_months=period_months,
                    use_cache=use_cache
                )
                
                if not df.empty:
                    results[ticker] = df
                    
                logger.info(
                    f"[{i}/{len(tickers)}] {ticker}: {len(df)} days collected"
                )
                
                # Long delay between tickers to avoid rate limiting
                if i < len(tickers):
                    logger.debug(
                        f"Waiting {self.rate_limit_delay}s to avoid rate limit..."
                    )
                    time.sleep(self.rate_limit_delay)
                    
            except Exception as e:
                logger.error(f"Failed to fetch trends for {ticker}: {e}")
                # Generate simulated data as fallback
                results[ticker] = self._generate_simulated_data(
                    ticker, period_months
                )
        
        logger.info(
            f"Google Trends collection complete: "
            f"{len(results)}/{len(tickers)} tickers"
        )
        
        return results
    
    def get_interest_change(
        self,
        df: pd.DataFrame,
        days: int = 7
    ) -> dict:
        """
        Calculate change in search interest over recent period.
        
        Args:
            df: DataFrame with trends data
            days: Number of days to compare
            
        Returns:
            Dictionary with interest change metrics
        """
        if df.empty:
            return {}
        
        df = df.sort_values("date")
        
        # Get recent and previous periods
        recent = df.tail(days)["search_interest"].mean()
        previous = df.tail(days * 2).head(days)["search_interest"].mean()
        
        if previous == 0:
            change_pct = 0.0
        else:
            change_pct = (recent - previous) / previous * 100
        
        return {
            "recent_avg": round(recent, 1),
            "previous_avg": round(previous, 1),
            "change_pct": round(change_pct, 1),
            "trend": "increasing" if change_pct > 10 else "decreasing" if change_pct < -10 else "stable"
        }
    
    def compare_tickers(
        self,
        tickers: list[str],
        period_months: int = 3
    ) -> pd.DataFrame:
        """
        Compare search interest across multiple tickers.
        
        Useful for identifying which stocks are getting more attention.
        
        Args:
            tickers: List of ticker symbols to compare
            period_months: Number of months to compare
            
        Returns:
            DataFrame with comparative interest metrics
        """
        comparison = []
        
        for ticker in tickers:
            try:
                df = self.fetch_interest(ticker, period_months=period_months)
                
                if df.empty:
                    continue
                
                metrics = {
                    "ticker": ticker,
                    "avg_interest": df["search_interest"].mean(),
                    "max_interest": df["search_interest"].max(),
                    "recent_interest": df.tail(7)["search_interest"].mean(),
                    **self.get_interest_change(df)
                }
                
                comparison.append(metrics)
                
                # Delay between requests
                time.sleep(self.rate_limit_delay)
                
            except Exception as e:
                logger.warning(f"Failed to compare {ticker}: {e}")
        
        if not comparison:
            return pd.DataFrame()
        
        df = pd.DataFrame(comparison)
        df = df.sort_values("recent_interest", ascending=False)
        
        return df


if __name__ == "__main__":
    # Test the collector
    print("Google Trends Collector Test")
    print("=" * 50)
    
    collector = TrendsCollector("./data/raw")
    
    # Test single ticker
    print("\nTesting trends fetch for AAPL (may use simulation)...")
    df = collector.fetch_interest("AAPL", period_months=3)
    
    print(f"\nData shape: {df.shape}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"Simulated: {df['is_simulated'].iloc[0] if len(df) > 0 else 'N/A'}")
    
    print("\nSample data:")
    print(df.head(10))
    
    print("\nInterest statistics:")
    print(df["search_interest"].describe())
    
    # Test interest change
    print("\nInterest change analysis:")
    change = collector.get_interest_change(df)
    print(f"Recent avg: {change.get('recent_avg')}")
    print(f"Change: {change.get('change_pct', 0):+.1f}%")
    print(f"Trend: {change.get('trend')}")
