"""
Configuration module using Pydantic Settings.

This module centralizes all configuration for the Market Sentiment Analyzer.
Settings are loaded from environment variables and .env file.

Key Features:
- Type-safe configuration with validation
- Environment variable loading
- Default values for development
- Secrets management (never hardcoded)

Usage:
    from src.config import settings
    
    print(settings.tickers)  # ['AAPL', 'TSLA', 'NVDA', 'MSFT', 'AMZN']
"""

from functools import lru_cache
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.
    
    All settings can be overridden via:
    1. Environment variables (highest priority)
    2. .env file
    3. Default values (lowest priority)
    
    Pydantic automatically:
    - Validates types
    - Converts comma-separated strings to lists
    - Handles missing optional fields
    """
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",  # Ignore extra env vars
    )
    
    # ============================================================
    # Reddit API Configuration
    # ============================================================
    reddit_client_id: str = Field(
        ...,
        description="Reddit API client ID from https://www.reddit.com/prefs/apps",
    )
    reddit_client_secret: str = Field(
        ...,
        description="Reddit API client secret",
    )
    reddit_user_agent: str = Field(
        default="market_sentiment/1.0",
        description="User agent string for Reddit API (format: app_name/version by username)",
    )
    
    # ============================================================
    # GLM API Configuration
    # ============================================================
    zhipuai_api_key: str = Field(
        ...,
        description="ZhipuAI API key for GLM model access",
    )
    
    # ============================================================
    # Database Configuration
    # ============================================================
    db_path: str = Field(
        default="./data/market_sentiment.db",
        description="Path to SQLite database file",
    )
    
    # ============================================================
    # Data Collection Settings
    # ============================================================
    tickers: list[str] = Field(
        default=["AAPL", "TSLA", "NVDA", "MSFT", "AMZN"],
        description="List of stock tickers to track (heavily discussed on Reddit)",
    )
    
    subreddits: list[str] = Field(
        default=["wallstreetbets", "investing", "stocks", "SecurityAnalysis"],
        description="List of subreddits to monitor for sentiment",
    )
    
    lookback_days: int = Field(
        default=365,
        ge=30,
        le=730,
        description="Historical data lookback period in days (30-730)",
    )
    
    prediction_horizon: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Days ahead to predict price direction (J+N)",
    )
    
    sentiment_model: Literal["vader", "finbert"] = Field(
        default="vader",
        description="Sentiment analysis model: 'vader' (fast) or 'finbert' (accurate)",
    )
    
    cache_dir: str = Field(
        default="./data/raw",
        description="Directory for caching raw data",
    )
    
    # ============================================================
    # Backtesting Settings
    # ============================================================
    initial_capital: float = Field(
        default=10000.0,
        ge=1000.0,
        description="Initial capital for backtesting simulation",
    )
    
    transaction_cost: float = Field(
        default=0.001,
        ge=0.0,
        le=0.01,
        description="Transaction cost per trade (default: 0.1%)",
    )
    
    # ============================================================
    # XGBoost Model Settings
    # ============================================================
    xgboost_n_estimators: int = Field(
        default=200,
        ge=50,
        le=1000,
        description="Number of boosting rounds",
    )
    
    xgboost_max_depth: int = Field(
        default=4,
        ge=2,
        le=10,
        description="Maximum tree depth",
    )
    
    xgboost_learning_rate: float = Field(
        default=0.05,
        ge=0.01,
        le=0.3,
        description="Learning rate / step size",
    )
    
    # ============================================================
    # API Rate Limiting
    # ============================================================
    reddit_rate_limit_delay: float = Field(
        default=1.0,
        ge=0.5,
        le=5.0,
        description="Seconds to wait between Reddit API calls",
    )
    
    trends_rate_limit_delay: float = Field(
        default=2.0,
        ge=1.0,
        le=10.0,
        description="Seconds to wait between Google Trends calls",
    )
    
    # ============================================================
    # Logging Configuration
    # ============================================================
    log_level: str = Field(
        default="INFO",
        description="Logging level: DEBUG, INFO, WARNING, ERROR, CRITICAL",
    )
    
    log_file: str = Field(
        default="./logs/market_sentiment.log",
        description="Path to log file",
    )
    
    # ============================================================
    # Validators
    # ============================================================
    @field_validator("tickers", mode="before")
    @classmethod
    def parse_tickers(cls, v: str | list[str]) -> list[str]:
        """Parse comma-separated ticker string into list."""
        if isinstance(v, str):
            return [t.strip().upper() for t in v.split(",")]
        return [t.upper() for t in v]
    
    @field_validator("subreddits", mode="before")
    @classmethod
    def parse_subreddits(cls, v: str | list[str]) -> list[str]:
        """Parse comma-separated subreddits string into list."""
        if isinstance(v, str):
            return [s.strip().lower() for s in v.split(",")]
        return [s.lower() for s in v]
    
    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """Validate log level is a valid Python logging level."""
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        v_upper = v.upper()
        if v_upper not in valid_levels:
            raise ValueError(f"Invalid log level: {v}. Must be one of {valid_levels}")
        return v_upper


@lru_cache
def get_settings() -> Settings:
    """
    Get cached settings instance.
    
    Uses lru_cache to ensure settings are only loaded once.
    This is important because:
    1. Avoids re-reading .env file on every access
    2. Ensures consistent settings throughout application
    3. Improves performance
    
    Returns:
        Settings: Application settings instance
    """
    return Settings()


# Global settings instance for easy import
# Usage: from src.config import settings
settings = get_settings()


# ============================================================
# Feature Column Definitions
# ============================================================

# Columns that should NEVER be used as features (identifiers, targets, raw text)
EXCLUDE_COLUMNS = {
    "id", "ticker", "date", "target_label", "target_return",
    "combined_text", "title", "selftext", "created_at", "fetched_at",
    "future_close_3d",  # This is the target calculation column
}

# Price-based features
PRICE_FEATURES = [
    "close", "volume", "daily_return", "log_return", "volatility_10d",
    "volume_ratio", "momentum_5d", "momentum_20d", "price_vs_52w_high",
]

# Technical indicator features
TECHNICAL_FEATURES = [
    "rsi_14", "rsi_signal", "macd", "macd_signal", "macd_hist",
    "macd_crossover", "bb_upper", "bb_mid", "bb_lower", "bb_position",
    "bb_squeeze", "volume_sma_20",
]

# Sentiment features
SENTIMENT_FEATURES = [
    "vader_compound", "vader_positive", "vader_negative", "vader_neutral",
    "finbert_positive", "finbert_negative", "finbert_neutral",
    "weighted_compound", "sentiment_momentum", "post_count",
    "avg_score", "total_comments", "search_interest",
]

# Lag features (KEY: test hypothesis that sentiment precedes price)
LAG_FEATURES = [
    "sentiment_lag1", "sentiment_lag2", "sentiment_lag3",
    "sentiment_momentum_3d",
]

# Cross-features (interaction between sentiment and technical)
CROSS_FEATURES = [
    "sentiment_x_volume", "sentiment_x_rsi",
]

# All feature columns for ML
ALL_FEATURE_COLUMNS = (
    PRICE_FEATURES + TECHNICAL_FEATURES + 
    SENTIMENT_FEATURES + LAG_FEATURES + CROSS_FEATURES
)


if __name__ == "__main__":
    # Test configuration loading
    print("Configuration Test")
    print("=" * 50)
    print(f"Tickers: {settings.tickers}")
    print(f"Subreddits: {settings.subreddits}")
    print(f"Lookback days: {settings.lookback_days}")
    print(f"Prediction horizon: {settings.prediction_horizon}")
    print(f"Sentiment model: {settings.sentiment_model}")
    print(f"Database path: {settings.db_path}")
    print(f"Log level: {settings.log_level}")
    print("=" * 50)
    print("Configuration loaded successfully!")
