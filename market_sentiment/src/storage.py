"""
Data Storage Module

SQLAlchemy ORM models and database operations for SQLite storage.
Provides persistent storage for all collected and processed data.

Tables:
- price_data: OHLCV data with computed returns and volatility
- reddit_posts: Raw posts and comments from Reddit
- reddit_daily_sentiment: Aggregated daily sentiment scores (VADER + FinBERT)
- google_trends: Search interest per ticker per date
- feature_matrix: Final ML-ready features with target labels
- model_predictions: Store model predictions for analysis

Features:
- Automatic table creation
- Upsert operations (insert or update)
- Batch insert for performance
- Query helpers for common operations
- Context manager for session handling

DISCLAIMER: All data stored is for educational purposes only.
"""

from contextlib import contextmanager
from datetime import datetime
from typing import Any, Iterator, Optional

import pandas as pd
from loguru import logger
from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    Index,
    Integer,
    String,
    Text,
    create_engine,
    delete,
    func,
    select,
    text,
)
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    Session,
    mapped_column,
    sessionmaker,
)
from sqlalchemy.types import TypeDecorator


# ============================================================
# Custom Types
# ============================================================

class TimestampType(TypeDecorator):
    """Custom type to handle datetime serialization."""
    impl = DateTime
    cache_ok = True

    def process_bind_param(self, value: Optional[datetime]):
        if isinstance(value, str):
            try:
                return datetime.fromisoformat(value.replace('Z', '+00:00'))
            except ValueError:
                return None
        return value


# ============================================================
# Base Model
# ============================================================

class Base(DeclarativeBase):
    """SQLAlchemy declarative base for all models."""
    pass


# ============================================================
# Table: price_data
# Stores OHLCV data with computed returns and volatility
# ============================================================

class PriceData(Base):
    """
    Stock price data table.
    
    Stores OHLCV data with computed returns and volatility.
    Unique constraint on (ticker, date) prevents duplicates.
    
    Columns:
    - ticker: Stock symbol (e.g., 'AAPL')
    - date: Trading date
    - open, high, low, close, volume: OHLCV data
    - daily_return: Percentage change from previous close
    - log_return: Logarithmic return
    - volatility_10d: 10-day rolling standard deviation
    """
    __tablename__ = "price_data"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ticker: Mapped[str] = mapped_column(String(10), nullable=False, index=True)
    date: Mapped[datetime] = mapped_column(DateTime, nullable=False, index=True)
    open: Mapped[float] = mapped_column(Float, nullable=False)
    high: Mapped[float] = mapped_column(Float, nullable=False)
    low: Mapped[float] = mapped_column(Float, nullable=False)
    close: Mapped[float] = mapped_column(Float, nullable=False)
    volume: Mapped[int] = mapped_column(Integer, nullable=False)
    daily_return: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    log_return: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    volatility_10d: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, nullable=False
    )
    
    # Unique constraint on ticker + date
    __table_args__ = (
        Index("ix_price_data_ticker_date", "ticker", "date", unique=True),
    )
    
    def __repr__(self) -> str:
        return f"<PriceData(ticker={self.ticker}, date={self.date}, close={self.close})>"


# ============================================================
# Table: reddit_posts
# Stores raw Reddit posts and comments
# ============================================================

class RedditPost(Base):
    """
    Raw Reddit posts and comments table.
    
    Stores post content for sentiment analysis.
    combined_text is used as input to NLP models.
    
    Columns:
    - post_id: Unique Reddit post ID (primary identifier)
    - ticker: Stock symbol mentioned
    - title: Post title
    - selftext: Post body content
    - combined_text: Title + body (for sentiment analysis)
    - score: Net upvotes
    - num_comments: Number of comments
    - subreddit: Source subreddit
    - created_at: Post timestamp
    """
    __tablename__ = "reddit_posts"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    post_id: Mapped[str] = mapped_column(String(20), unique=True, nullable=False)
    ticker: Mapped[str] = mapped_column(String(10), nullable=False, index=True)
    title: Mapped[str] = mapped_column(Text, nullable=False)
    selftext: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    combined_text: Mapped[str] = mapped_column(Text, nullable=False)
    score: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    upvote_ratio: Mapped[float] = mapped_column(Float, default=1.0, nullable=True)
    num_comments: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    subreddit: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    url: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, index=True)
    fetched_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, nullable=False
    )
    
    def __repr__(self) -> str:
        return f"<RedditPost(post_id={self.post_id}, ticker={self.ticker}, score={self.score})>"


# ============================================================
# Table: reddit_comments
# Stores top comments from Reddit posts
# ============================================================

class RedditComment(Base):
    """
    Reddit comments table.
    
    Stores top comments for additional sentiment context.
    Linked to reddit_posts via post_id.
    """
    __tablename__ = "reddit_comments"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    comment_id: Mapped[str] = mapped_column(String(20), unique=True, nullable=False)
    post_id: Mapped[str] = mapped_column(String(20), nullable=False, index=True)
    body: Mapped[str] = mapped_column(Text, nullable=False)
    score: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    
    def __repr__(self) -> str:
        return f"<RedditComment(comment_id={self.comment_id}, score={self.score})>"


# ============================================================
# Table: reddit_daily_sentiment
# Aggregated daily sentiment per ticker
# ============================================================

class RedditDailySentiment(Base):
    """
    Aggregated daily sentiment per ticker.
    
    Pre-computed daily averages for both VADER and FinBERT scores.
    Used for feature matrix construction.
    
    Key columns:
    - vader_compound: Average VADER compound score (-1 to +1)
    - finbert_positive/negative/neutral: FinBERT probability scores
    - dominant_sentiment: "positive", "negative", or "neutral"
    - weighted_compound: Upvote-weighted sentiment
    """
    __tablename__ = "reddit_daily_sentiment"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ticker: Mapped[str] = mapped_column(String(10), nullable=False, index=True)
    date: Mapped[datetime] = mapped_column(DateTime, nullable=False, index=True)
    
    # Post metrics
    post_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    avg_score: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    total_comments: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    
    # VADER sentiment (fast, rule-based)
    vader_compound: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    vader_positive: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    vader_negative: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    vader_neutral: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    
    # FinBERT sentiment (transformer-based)
    finbert_positive: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    finbert_negative: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    finbert_neutral: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    
    # Weighted sentiment (viral posts have more impact)
    weighted_compound: Mapped[float] = mapped_column(Float, default=0.0, nullable=True)
    
    # Sentiment momentum (change from previous day)
    sentiment_momentum: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    
    # Dominant sentiment label
    dominant_sentiment: Mapped[str] = mapped_column(
        String(20), default="neutral", nullable=False
    )
    
    # Unique constraint on ticker + date
    __table_args__ = (
        Index("ix_reddit_sentiment_ticker_date", "ticker", "date", unique=True),
    )
    
    def __repr__(self) -> str:
        return f"<RedditDailySentiment(ticker={self.ticker}, date={self.date}, vader={self.vader_compound:.3f})>"


# ============================================================
# Table: google_trends
# Search interest data from Google Trends
# ============================================================

class GoogleTrend(Base):
    """
    Google Trends search interest table.
    
    Stores normalized search interest (0-100) per ticker per date.
    
    Columns:
    - ticker: Stock symbol
    - date: Date
    - search_interest: Normalized interest (0-100)
    - is_simulated: Whether data is simulated (pytrends unavailable)
    """
    __tablename__ = "google_trends"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ticker: Mapped[str] = mapped_column(String(10), nullable=False, index=True)
    date: Mapped[datetime] = mapped_column(DateTime, nullable=False, index=True)
    search_interest: Mapped[float] = mapped_column(Float, nullable=False)
    is_simulated: Mapped[bool] = mapped_column(Boolean, default=False, nullable=True)
    
    # Unique constraint on ticker + date
    __table_args__ = (
        Index("ix_google_trends_ticker_date", "ticker", "date", unique=True),
    )
    
    def __repr__(self) -> str:
        return f"<GoogleTrend(ticker={self.ticker}, date={self.date}, interest={self.search_interest})>"


# ============================================================
# Table: feature_matrix
# Final ML-ready feature matrix with targets
# ============================================================

class FeatureMatrix(Base):
    """
    Final ML-ready feature matrix.
    
    Contains all computed features and target variables.
    target_label: 1 = price up at J+3, 0 = price down/flat
    target_return: actual return at J+3 (for regression analysis)
    
    IMPORTANT: target_* columns must NEVER be used as features!
    They are the prediction targets.
    """
    __tablename__ = "feature_matrix"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ticker: Mapped[str] = mapped_column(String(10), nullable=False, index=True)
    date: Mapped[datetime] = mapped_column(DateTime, nullable=False, index=True)
    
    # Price features
    close: Mapped[float] = mapped_column(Float, nullable=False)
    volume: Mapped[int] = mapped_column(Integer, nullable=False)
    daily_return: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    log_return: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    volatility_10d: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    volume_ratio: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    momentum_5d: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    momentum_20d: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    price_vs_52w_high: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    
    # Technical indicators
    rsi_14: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    rsi_signal: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)
    macd: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    macd_signal: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    macd_hist: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    macd_crossover: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    bb_upper: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    bb_mid: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    bb_lower: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    bb_position: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    bb_squeeze: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    
    # Sentiment features
    vader_compound: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    vader_positive: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    vader_negative: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    weighted_compound: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    sentiment_momentum: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    post_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    total_comments: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    
    # Lag features (KEY: sentiment preceding price hypothesis)
    sentiment_lag1: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    sentiment_lag2: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    sentiment_lag3: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    sentiment_momentum_3d: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    
    # Trend features
    search_interest: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    
    # Cross features
    sentiment_x_volume: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    sentiment_x_rsi: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    
    # Target variables (NEVER use as features!)
    # These are what we're trying to predict
    target_label: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    target_return: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    
    # Metadata
    sentiment_data_available: Mapped[bool] = mapped_column(Boolean, default=False, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, nullable=False
    )
    
    # Unique constraint on ticker + date
    __table_args__ = (
        Index("ix_feature_matrix_ticker_date", "ticker", "date", unique=True),
    )
    
    def __repr__(self) -> str:
        return f"<FeatureMatrix(ticker={self.ticker}, date={self.date}, target={self.target_label})>"


# ============================================================
# Table: model_predictions
# Store model predictions for analysis and backtesting
# ============================================================

class ModelPrediction(Base):
    """
    Model predictions table.
    
    Stores predictions made by the ML model for later analysis.
    Used for backtesting and model performance tracking.
    """
    __tablename__ = "model_predictions"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ticker: Mapped[str] = mapped_column(String(10), nullable=False, index=True)
    date: Mapped[datetime] = mapped_column(DateTime, nullable=False, index=True)
    prediction_date: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    
    # Prediction
    predicted_direction: Mapped[int] = mapped_column(Integer, nullable=False)  # 1=up, 0=down
    probability: Mapped[float] = mapped_column(Float, nullable=False)
    confidence: Mapped[str] = mapped_column(String(20), nullable=False)  # high/medium/low
    
    # Key signals (top features driving prediction)
    key_signals: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    
    # Actual outcome (filled later)
    actual_return: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    actual_direction: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    
    # Model info
    model_version: Mapped[str] = mapped_column(String(50), default="v1.0", nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, nullable=False
    )
    
    def __repr__(self) -> str:
        return f"<ModelPrediction(ticker={self.ticker}, date={self.date}, pred={self.predicted_direction})>"


# ============================================================
# Database Engine and Session Management
# ============================================================

class DatabaseManager:
    """
    Database manager for SQLAlchemy operations.
    
    Provides:
    - Connection management
    - Session handling with context manager
    - Table creation
    - Bulk operations
    - Query helpers
    
    Example:
        >>> db = DatabaseManager("./data/market_sentiment.db")
        >>> with db.session() as session:
        ...     prices = session.query(PriceData).filter_by(ticker="AAPL").all()
    """
    
    def __init__(self, db_path: str) -> None:
        """
        Initialize database manager.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self.engine = create_engine(
            f"sqlite:///{db_path}",
            echo=False,  # Set True to see SQL queries
            pool_pre_ping=True,  # Check connection health
            pool_recycle=3600,  # Recycle connections after 1 hour
        )
        self.SessionLocal = sessionmaker(
            bind=self.engine,
            autocommit=False,
            autoflush=False,
            expire_on_commit=False,
        )
        
        logger.info(f"DatabaseManager initialized: {db_path}")
    
    def create_tables(self) -> None:
        """Create all tables if they don't exist."""
        Base.metadata.create_all(self.engine)
        logger.info("Database tables created/verified")
    
    def drop_tables(self) -> None:
        """Drop all tables (use with caution!)."""
        Base.metadata.drop_all(self.engine)
        logger.warning("All database tables dropped")
    
    @contextmanager
    def session(self) -> Iterator[Session]:
        """
        Context manager for database sessions.
        
        Automatically handles commit/rollback and closing.
        
        Yields:
            SQLAlchemy Session object
            
        Example:
            >>> with db.session() as session:
            ...     session.add(price_record)
            ...     # Auto-commits on exit
        """
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Database session error: {e}")
            raise
        finally:
            session.close()
    
    def get_session(self) -> Session:
        """Get a raw session (caller responsible for closing)."""
        return self.SessionLocal()
    
    def execute_raw(self, sql: str) -> Any:
        """
        Execute raw SQL query.
        
        Args:
            sql: SQL query string
            
        Returns:
            Query result
        """
        with self.session() as session:
            return session.execute(text(sql))


# ============================================================
# Save Functions (Upsert Operations)
# ============================================================

def save_prices(
    df: pd.DataFrame,
    db: DatabaseManager,
    batch_size: int = 1000
) -> int:
    """
    Save price data to database with upsert logic.
    
    Inserts new records, updates existing ones based on (ticker, date).
    
    Args:
        df: DataFrame with price data
        db: DatabaseManager instance
        batch_size: Number of records to insert per batch
        
    Returns:
        Number of records saved
    """
    if df.empty:
        logger.warning("Empty DataFrame, skipping price save")
        return 0
    
    # Ensure date is datetime
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    
    saved_count = 0
    
    with db.session() as session:
        for _, row in df.iterrows():
            try:
                # Check if record exists
                existing = session.execute(
                    select(PriceData).where(
                        PriceData.ticker == row["ticker"],
                        PriceData.date == row["date"]
                    )
                ).scalar_one_or_none()
                
                if existing:
                    # Update existing record
                    existing.open = row.get("open", existing.open)
                    existing.high = row.get("high", existing.high)
                    existing.low = row.get("low", existing.low)
                    existing.close = row.get("close", existing.close)
                    existing.volume = row.get("volume", existing.volume)
                    existing.daily_return = row.get("daily_return")
                    existing.log_return = row.get("log_return")
                    existing.volatility_10d = row.get("volatility_10d")
                else:
                    # Insert new record
                    price = PriceData(
                        ticker=row["ticker"],
                        date=row["date"],
                        open=row["open"],
                        high=row["high"],
                        low=row["low"],
                        close=row["close"],
                        volume=row["volume"],
                        daily_return=row.get("daily_return"),
                        log_return=row.get("log_return"),
                        volatility_10d=row.get("volatility_10d"),
                    )
                    session.add(price)
                    saved_count += 1
                    
            except Exception as e:
                logger.warning(f"Error saving price {row.get('ticker')} {row.get('date')}: {e}")
                continue
        
        logger.info(f"Saved {saved_count} new price records")
    
    return saved_count


def save_reddit_posts(
    posts: list[dict],
    db: DatabaseManager
) -> int:
    """
    Save Reddit posts to database with upsert logic.
    
    Args:
        posts: List of post dictionaries from RedditCollector
        db: DatabaseManager instance
        
    Returns:
        Number of new posts saved
    """
    if not posts:
        logger.warning("Empty posts list, skipping save")
        return 0
    
    saved_count = 0
    
    with db.session() as session:
        for post in posts:
            try:
                # Check if post already exists
                existing = session.execute(
                    select(RedditPost).where(RedditPost.post_id == post["post_id"])
                ).scalar_one_or_none()
                
                if existing:
                    # Update score and comments (they change over time)
                    existing.score = post.get("score", existing.score)
                    existing.num_comments = post.get("num_comments", existing.num_comments)
                    continue
                
                # Parse timestamp
                created_at = post.get("created_at")
                if isinstance(created_at, str):
                    created_at = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
                
                # Insert new post
                reddit_post = RedditPost(
                    post_id=post["post_id"],
                    ticker=post["ticker"],
                    title=post["title"],
                    selftext=post.get("selftext", ""),
                    combined_text=post["combined_text"],
                    score=post.get("score", 0),
                    upvote_ratio=post.get("upvote_ratio", 1.0),
                    num_comments=post.get("num_comments", 0),
                    subreddit=post["subreddit"],
                    url=post.get("url"),
                    created_at=created_at,
                )
                session.add(reddit_post)
                saved_count += 1
                
                # Save comments if present
                for comment in post.get("comments", []):
                    try:
                        comment_created = comment.get("created_at")
                        if isinstance(comment_created, str):
                            comment_created = datetime.fromisoformat(comment_created)
                        
                        comment_record = RedditComment(
                            comment_id=comment["comment_id"],
                            post_id=post["post_id"],
                            body=comment["body"],
                            score=comment.get("score", 0),
                            created_at=comment_created,
                        )
                        session.add(comment_record)
                    except IntegrityError:
                        # Comment already exists
                        session.rollback()
                        continue
                        
            except Exception as e:
                logger.warning(f"Error saving post {post.get('post_id')}: {e}")
                continue
        
        logger.info(f"Saved {saved_count} new Reddit posts")
    
    return saved_count


def save_daily_sentiment(
    df: pd.DataFrame,
    db: DatabaseManager
) -> int:
    """
    Save aggregated daily sentiment to database.
    
    Args:
        df: DataFrame with daily sentiment data
        db: DatabaseManager instance
        
    Returns:
        Number of records saved
    """
    if df.empty:
        logger.warning("Empty DataFrame, skipping sentiment save")
        return 0
    
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    
    saved_count = 0
    
    with db.session() as session:
        for _, row in df.iterrows():
            try:
                existing = session.execute(
                    select(RedditDailySentiment).where(
                        RedditDailySentiment.ticker == row["ticker"],
                        RedditDailySentiment.date == row["date"]
                    )
                ).scalar_one_or_none()
                
                if existing:
                    # Update existing
                    for col in ["vader_compound", "vader_positive", "vader_negative",
                               "weighted_compound", "post_count", "dominant_sentiment"]:
                        if col in row:
                            setattr(existing, col, row[col])
                else:
                    # Insert new
                    sentiment = RedditDailySentiment(
                        ticker=row["ticker"],
                        date=row["date"],
                        post_count=row.get("post_count", 0),
                        avg_score=row.get("avg_score", 0),
                        total_comments=row.get("total_comments", 0),
                        vader_compound=row.get("vader_compound", 0),
                        vader_positive=row.get("vader_positive", 0),
                        vader_negative=row.get("vader_negative", 0),
                        vader_neutral=row.get("vader_neutral", 0),
                        finbert_positive=row.get("finbert_positive"),
                        finbert_negative=row.get("finbert_negative"),
                        finbert_neutral=row.get("finbert_neutral"),
                        weighted_compound=row.get("weighted_compound", 0),
                        dominant_sentiment=row.get("dominant_sentiment", "neutral"),
                    )
                    session.add(sentiment)
                    saved_count += 1
                    
            except Exception as e:
                logger.warning(f"Error saving sentiment {row.get('ticker')} {row.get('date')}: {e}")
                continue
        
        logger.info(f"Saved {saved_count} new sentiment records")
    
    return saved_count


def save_trends(
    df: pd.DataFrame,
    db: DatabaseManager
) -> int:
    """
    Save Google Trends data to database.
    
    Args:
        df: DataFrame with trends data
        db: DatabaseManager instance
        
    Returns:
        Number of records saved
    """
    if df.empty:
        logger.warning("Empty DataFrame, skipping trends save")
        return 0
    
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    
    saved_count = 0
    
    with db.session() as session:
        for _, row in df.iterrows():
            try:
                existing = session.execute(
                    select(GoogleTrend).where(
                        GoogleTrend.ticker == row["ticker"],
                        GoogleTrend.date == row["date"]
                    )
                ).scalar_one_or_none()
                
                if existing:
                    existing.search_interest = row["search_interest"]
                    existing.is_simulated = row.get("is_simulated", False)
                else:
                    trend = GoogleTrend(
                        ticker=row["ticker"],
                        date=row["date"],
                        search_interest=row["search_interest"],
                        is_simulated=row.get("is_simulated", False),
                    )
                    session.add(trend)
                    saved_count += 1
                    
            except Exception as e:
                logger.warning(f"Error saving trend {row.get('ticker')} {row.get('date')}: {e}")
                continue
        
        logger.info(f"Saved {saved_count} new trend records")
    
    return saved_count


def save_features(
    df: pd.DataFrame,
    db: DatabaseManager
) -> int:
    """
    Save feature matrix to database.
    
    Args:
        df: DataFrame with feature matrix
        db: DatabaseManager instance
        
    Returns:
        Number of records saved
    """
    if df.empty:
        logger.warning("Empty DataFrame, skipping features save")
        return 0
    
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    
    saved_count = 0
    
    with db.session() as session:
        for _, row in df.iterrows():
            try:
                existing = session.execute(
                    select(FeatureMatrix).where(
                        FeatureMatrix.ticker == row["ticker"],
                        FeatureMatrix.date == row["date"]
                    )
                ).scalar_one_or_none()
                
                if existing:
                    # Update all columns
                    for col in FeatureMatrix.__table__.columns.keys():
                        if col in ["id", "created_at"]:
                            continue
                        if col in row:
                            setattr(existing, col, row[col])
                else:
                    # Create new record with all available columns
                    feature_dict = {}
                    for col in FeatureMatrix.__table__.columns.keys():
                        if col in ["id", "created_at"]:
                            continue
                        if col in row:
                            feature_dict[col] = row[col]
                    
                    feature = FeatureMatrix(**feature_dict)
                    session.add(feature)
                    saved_count += 1
                    
            except Exception as e:
                logger.warning(f"Error saving feature {row.get('ticker')} {row.get('date')}: {e}")
                continue
        
        logger.info(f"Saved {saved_count} new feature records")
    
    return saved_count


# ============================================================
# Load Functions (Query Operations)
# ============================================================

def load_price_history(
    db: DatabaseManager,
    ticker: str,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None
) -> pd.DataFrame:
    """
    Load price history for a ticker.
    
    Args:
        db: DatabaseManager instance
        ticker: Stock ticker symbol
        start_date: Optional start date filter
        end_date: Optional end date filter
        
    Returns:
        DataFrame with price data
    """
    with db.session() as session:
        query = select(PriceData).where(PriceData.ticker == ticker.upper())
        
        if start_date:
            query = query.where(PriceData.date >= start_date)
        if end_date:
            query = query.where(PriceData.date <= end_date)
        
        query = query.order_by(PriceData.date)
        
        results = session.execute(query).scalars().all()
        
        if not results:
            return pd.DataFrame()
        
        data = [
            {
                "date": r.date,
                "ticker": r.ticker,
                "open": r.open,
                "high": r.high,
                "low": r.low,
                "close": r.close,
                "volume": r.volume,
                "daily_return": r.daily_return,
                "log_return": r.log_return,
                "volatility_10d": r.volatility_10d,
            }
            for r in results
        ]
        
        df = pd.DataFrame(data)
        logger.info(f"Loaded {len(df)} price records for {ticker}")
        
        return df


def load_sentiment_history(
    db: DatabaseManager,
    ticker: str,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None
) -> pd.DataFrame:
    """
    Load sentiment history for a ticker.
    
    Args:
        db: DatabaseManager instance
        ticker: Stock ticker symbol
        start_date: Optional start date filter
        end_date: Optional end date filter
        
    Returns:
        DataFrame with sentiment data
    """
    with db.session() as session:
        query = select(RedditDailySentiment).where(
            RedditDailySentiment.ticker == ticker.upper()
        )
        
        if start_date:
            query = query.where(RedditDailySentiment.date >= start_date)
        if end_date:
            query = query.where(RedditDailySentiment.date <= end_date)
        
        query = query.order_by(RedditDailySentiment.date)
        
        results = session.execute(query).scalars().all()
        
        if not results:
            return pd.DataFrame()
        
        data = [
            {
                "date": r.date,
                "ticker": r.ticker,
                "post_count": r.post_count,
                "avg_score": r.avg_score,
                "total_comments": r.total_comments,
                "vader_compound": r.vader_compound,
                "vader_positive": r.vader_positive,
                "vader_negative": r.vader_negative,
                "finbert_positive": r.finbert_positive,
                "finbert_negative": r.finbert_negative,
                "finbert_neutral": r.finbert_neutral,
                "weighted_compound": r.weighted_compound,
                "dominant_sentiment": r.dominant_sentiment,
            }
            for r in results
        ]
        
        df = pd.DataFrame(data)
        logger.info(f"Loaded {len(df)} sentiment records for {ticker}")
        
        return df


def load_feature_matrix(
    db: DatabaseManager,
    ticker: Optional[str] = None,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None
) -> pd.DataFrame:
    """
    Load feature matrix for ML training.
    
    Args:
        db: DatabaseManager instance
        ticker: Optional ticker filter (None = all tickers)
        start_date: Optional start date filter
        end_date: Optional end date filter
        
    Returns:
        DataFrame with feature matrix
    """
    with db.session() as session:
        query = select(FeatureMatrix)
        
        if ticker:
            query = query.where(FeatureMatrix.ticker == ticker.upper())
        if start_date:
            query = query.where(FeatureMatrix.date >= start_date)
        if end_date:
            query = query.where(FeatureMatrix.date <= end_date)
        
        query = query.order_by(FeatureMatrix.ticker, FeatureMatrix.date)
        
        results = session.execute(query).scalars().all()
        
        if not results:
            return pd.DataFrame()
        
        # Get all column names from the model
        columns = [c.name for c in FeatureMatrix.__table__.columns]
        
        data = []
        for r in results:
            row = {col: getattr(r, col) for col in columns}
            data.append(row)
        
        df = pd.DataFrame(data)
        logger.info(f"Loaded {len(df)} feature records")
        
        return df


def load_trends(
    db: DatabaseManager,
    ticker: str,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None
) -> pd.DataFrame:
    """
    Load Google Trends data for a ticker.
    
    Args:
        db: DatabaseManager instance
        ticker: Stock ticker symbol
        start_date: Optional start date filter
        end_date: Optional end date filter
        
    Returns:
        DataFrame with trends data
    """
    with db.session() as session:
        query = select(GoogleTrend).where(GoogleTrend.ticker == ticker.upper())
        
        if start_date:
            query = query.where(GoogleTrend.date >= start_date)
        if end_date:
            query = query.where(GoogleTrend.date <= end_date)
        
        query = query.order_by(GoogleTrend.date)
        
        results = session.execute(query).scalars().all()
        
        if not results:
            return pd.DataFrame()
        
        data = [
            {
                "date": r.date,
                "ticker": r.ticker,
                "search_interest": r.search_interest,
                "is_simulated": r.is_simulated,
            }
            for r in results
        ]
        
        return pd.DataFrame(data)


def get_latest_price(db: DatabaseManager, ticker: str) -> Optional[dict]:
    """
    Get the latest price for a ticker.
    
    Args:
        db: DatabaseManager instance
        ticker: Stock ticker symbol
        
    Returns:
        Dictionary with latest price info, or None
    """
    with db.session() as session:
        result = session.execute(
            select(PriceData)
            .where(PriceData.ticker == ticker.upper())
            .order_by(PriceData.date.desc())
            .limit(1)
        ).scalar_one_or_none()
        
        if result:
            return {
                "date": result.date,
                "close": result.close,
                "volume": result.volume,
                "daily_return": result.daily_return,
            }
        
        return None


def get_data_summary(db: DatabaseManager) -> dict:
    """
    Get summary of data in database.
    
    Args:
        db: DatabaseManager instance
        
    Returns:
        Dictionary with record counts and date ranges
    """
    with db.session() as session:
        summary = {}
        
        # Price data summary
        price_count = session.execute(
            select(func.count(PriceData.id))
        ).scalar()
        price_tickers = session.execute(
            select(PriceData.ticker).distinct()
        ).scalars().all()
        summary["price_data"] = {
            "count": price_count,
            "tickers": list(price_tickers),
        }
        
        # Sentiment data summary
        sentiment_count = session.execute(
            select(func.count(RedditDailySentiment.id))
        ).scalar()
        summary["sentiment_data"] = {"count": sentiment_count}
        
        # Feature matrix summary
        feature_count = session.execute(
            select(func.count(FeatureMatrix.id))
        ).scalar()
        summary["feature_matrix"] = {"count": feature_count}
        
        # Trends data summary
        trends_count = session.execute(
            select(func.count(GoogleTrend.id))
        ).scalar()
        summary["trends_data"] = {"count": trends_count}
        
        return summary


# ============================================================
# Initialization Function
# ============================================================

def init_db(db_path: str) -> DatabaseManager:
    """
    Initialize database with all tables.
    
    Args:
        db_path: Path to SQLite database file
        
    Returns:
        DatabaseManager instance
    """
    db = DatabaseManager(db_path)
    db.create_tables()
    
    # Verify tables were created
    summary = get_data_summary(db)
    logger.info(f"Database initialized: {summary}")
    
    return db


# ============================================================
# Module Initialization
# ============================================================

logger.info("Storage module loaded successfully")


if __name__ == "__main__":
    # Test the storage module
    import tempfile
    import os
    
    print("Storage Module Test")
    print("=" * 60)
    
    # Create temporary database
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        
        # Initialize database
        print("\n1. Initializing database...")
        db = init_db(db_path)
        print(f"   ✓ Database created: {db_path}")
        
        # Test price data
        print("\n2. Testing price data...")
        test_prices = pd.DataFrame({
            "date": [datetime.now() - timedelta(days=i) for i in range(5)],
            "ticker": ["AAPL"] * 5,
            "open": [178.0, 179.0, 177.5, 180.0, 181.0],
            "high": [179.5, 180.5, 179.0, 182.0, 183.0],
            "low": [177.0, 178.0, 176.5, 179.5, 180.0],
            "close": [179.0, 178.5, 178.0, 181.5, 182.0],
            "volume": [50000000, 48000000, 52000000, 55000000, 49000000],
            "daily_return": [None, -0.0028, -0.0028, 0.0197, 0.0028],
            "log_return": [None, -0.0028, -0.0028, 0.0195, 0.0028],
            "volatility_10d": [None, None, None, None, 0.012],
        })
        
        count = save_prices(test_prices, db)
        print(f"   ✓ Saved {count} price records")
        
        # Load back
        loaded = load_price_history(db, "AAPL")
        print(f"   ✓ Loaded {len(loaded)} records")
        
        # Test sentiment data
        print("\n3. Testing sentiment data...")
        test_sentiment = pd.DataFrame({
            "date": [datetime.now() - timedelta(days=i) for i in range(3)],
            "ticker": ["AAPL"] * 3,
            "post_count": [15, 12, 8],
            "avg_score": [150.5, 120.3, 85.2],
            "total_comments": [45, 36, 24],
            "vader_compound": [0.35, 0.28, 0.15],
            "vader_positive": [0.45, 0.38, 0.30],
            "vader_negative": [0.10, 0.12, 0.15],
            "vader_neutral": [0.45, 0.50, 0.55],
            "dominant_sentiment": ["positive", "positive", "neutral"],
        })
        
        count = save_daily_sentiment(test_sentiment, db)
        print(f"   ✓ Saved {count} sentiment records")
        
        # Test trends data
        print("\n4. Testing trends data...")
        test_trends = pd.DataFrame({
            "date": [datetime.now() - timedelta(days=i) for i in range(3)],
            "ticker": ["AAPL"] * 3,
            "search_interest": [55.0, 48.0, 42.0],
            "is_simulated": [False, False, False],
        })
        
        count = save_trends(test_trends, db)
        print(f"   ✓ Saved {count} trend records")
        
        # Get summary
        print("\n5. Database summary:")
        summary = get_data_summary(db)
        for table, info in summary.items():
            print(f"   {table}: {info}")
        
        print("\n✅ All storage tests passed!")
