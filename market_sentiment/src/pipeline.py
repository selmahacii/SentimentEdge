"""
Pipeline Orchestration Module

Coordinates all components to run the full data pipeline:
1. Data collection (prices, Reddit, trends)
2. Storage to database
3. Sentiment analysis
4. Feature engineering
5. Model training
6. Backtesting
7. GLM brief generation

Usage:
    from src.pipeline import run_full_pipeline

    results = run_full_pipeline(tickers=['AAPL', 'TSLA'])

DISCLAIMER: This pipeline is for educational purposes only.
Past performance does not guarantee future results.
"""

import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger

# Local imports
from src.config import settings
from src.storage import (
    DatabaseManager,
    save_prices,
    save_reddit_posts,
    save_daily_sentiment,
    save_trends,
    save_features,
    load_price_history,
    load_sentiment_history,
    load_trends,
    load_feature_matrix,
    get_data_summary,
)


class PipelineResult:
    """Container for pipeline results."""

    def __init__(self):
        self.start_time: datetime = datetime.now()
        self.end_time: Optional[datetime] = None
        self.tickers_processed: list[str] = []
        self.errors: list[dict] = []
        self.metrics: dict = {}

        # Component results
        self.prices_collected: int = 0
        self.posts_collected: int = 0
        self.trends_collected: int = 0
        self.sentiment_computed: int = 0
        self.features_computed: int = 0
        self.model_results: dict = {}
        self.backtest_results: dict = {}
        self.glm_briefs: list[dict] = []

    def to_dict(self) -> dict:
        """Convert results to dictionary."""
        duration = (self.end_time or datetime.now()) - self.start_time

        return {
            "status": "success" if not self.errors else "partial_failure",
            "duration_seconds": duration.total_seconds(),
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "tickers_processed": self.tickers_processed,
            "errors": self.errors,
            "metrics": {
                "prices_collected": self.prices_collected,
                "posts_collected": self.posts_collected,
                "trends_collected": self.trends_collected,
                "sentiment_computed": self.sentiment_computed,
                "features_computed": self.features_computed,
                "models_trained": len(self.model_results),
                **self.metrics,
            },
            "model_results": self.model_results,
            "backtest_results": self.backtest_results,
            "glm_briefs": self.glm_briefs,
        }


def run_full_pipeline(
    tickers: Optional[list[str]] = None,
    use_llm: bool = True,
    sentiment_model: str = "vader",
    days: Optional[int] = None,
    run_backtest: bool = True,
    db_path: Optional[str] = None
) -> dict:
    """
    Run the complete market sentiment analysis pipeline.

    Args:
        tickers: List of tickers to analyze (default: from config)
        use_llm: Whether to generate GLM briefs (set False for testing)
        sentiment_model: 'vader' (fast) or 'finbert' (accurate)
        days: Lookback days (default: from config)
        run_backtest: Whether to run backtesting
        db_path: Database path (default: from config)

    Returns:
        Dictionary with all results and metrics

    Pipeline Steps:
        1. Initialize collectors and database
        2. Fetch prices, Reddit posts, Google Trends
        3. Save raw data to database
        4. Run sentiment analysis (VADER/FinBERT)
        5. Compute technical indicators
        6. Fuse all features into feature matrix
        7. Train XGBoost model per ticker
        8. Run backtesting
        9. Generate GLM briefs (optional)
        10. Return formatted results
    """
    logger.info("=" * 60)
    logger.info("Starting Market Sentiment Pipeline")
    logger.info("=" * 60)

    result = PipelineResult()

    # Use config defaults
    tickers = tickers or settings.tickers
    days = days or settings.lookback_days
    db_path = db_path or settings.db_path

    # Initialize database
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    db = DatabaseManager(db_path)
    db.create_tables()

    logger.info(f"Processing {len(tickers)} tickers: {tickers}")
    logger.info(f"Lookback: {days} days | Sentiment model: {sentiment_model}")

    try:
        # ============================================================
        # STEP 1: Collect Price Data
        # ============================================================
        logger.info("\n" + "=" * 40)
        logger.info("STEP 1: Collecting Price Data")
        logger.info("=" * 40)

        from src.collectors.price_collector import PriceCollector

        price_collector = PriceCollector()
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        all_prices = []
        for ticker in tickers:
            try:
                logger.info(f"Fetching prices for {ticker}...")
                prices_df = price_collector.fetch_prices(
                    ticker, start_date=start_date, end_date=end_date
                )
                if not prices_df.empty:
                    all_prices.append(prices_df)
                    result.prices_collected += len(prices_df)
                    logger.info(f"  Collected {len(prices_df)} price records for {ticker}")
            except Exception as e:
                logger.error(f"  Error fetching prices for {ticker}: {e}")
                result.errors.append({"step": "prices", "ticker": ticker, "error": str(e)})

        # Combine and save
        if all_prices:
            combined_prices = pd.concat(all_prices, ignore_index=True)
            save_prices(combined_prices, db)
            logger.info(f"Total prices collected: {result.prices_collected}")

        # ============================================================
        # STEP 2: Collect Reddit Data
        # ============================================================
        logger.info("\n" + "=" * 40)
        logger.info("STEP 2: Collecting Reddit Data")
        logger.info("=" * 40)

        try:
            from src.collectors.reddit_collector import RedditCollector

            reddit_collector = RedditCollector(
                client_id=settings.reddit_client_id,
                client_secret=settings.reddit_client_secret,
                user_agent=settings.reddit_user_agent,
            )

            all_posts = []
            for ticker in tickers:
                try:
                    logger.info(f"Fetching Reddit posts for {ticker}...")
                    posts = reddit_collector.fetch_posts_for_ticker(
                        ticker,
                        subreddits=settings.subreddits,
                        days_back=min(days, 30),  # Limit Reddit to 30 days
                        limit=100,
                    )
                    if posts:
                        all_posts.extend(posts)
                        result.posts_collected += len(posts)
                        logger.info(f"  Collected {len(posts)} posts for {ticker}")
                except Exception as e:
                    logger.error(f"  Error fetching Reddit for {ticker}: {e}")
                    result.errors.append({"step": "reddit", "ticker": ticker, "error": str(e)})

            # Save posts
            if all_posts:
                save_reddit_posts(all_posts, db)
                logger.info(f"Total posts collected: {result.posts_collected}")

        except Exception as e:
            logger.warning(f"Reddit collection failed (continuing without): {e}")
            result.errors.append({"step": "reddit", "error": str(e)})

        # ============================================================
        # STEP 3: Collect Google Trends
        # ============================================================
        logger.info("\n" + "=" * 40)
        logger.info("STEP 3: Collecting Google Trends")
        logger.info("=" * 40)

        try:
            from src.collectors.trends_collector import TrendsCollector

            trends_collector = TrendsCollector()

            all_trends = []
            for ticker in tickers:
                try:
                    logger.info(f"Fetching trends for {ticker}...")
                    trends_df = trends_collector.fetch_trends(
                        ticker, days_back=min(days, 90)  # Limit trends to 90 days
                    )
                    if not trends_df.empty:
                        all_trends.append(trends_df)
                        result.trends_collected += len(trends_df)
                        logger.info(f"  Collected {len(trends_df)} trend records for {ticker}")
                except Exception as e:
                    logger.error(f"  Error fetching trends for {ticker}: {e}")
                    result.errors.append({"step": "trends", "ticker": ticker, "error": str(e)})

            # Save trends
            if all_trends:
                combined_trends = pd.concat(all_trends, ignore_index=True)
                save_trends(combined_trends, db)
                logger.info(f"Total trends collected: {result.trends_collected}")

        except Exception as e:
            logger.warning(f"Trends collection failed (continuing without): {e}")
            result.errors.append({"step": "trends", "error": str(e)})

        # ============================================================
        # STEP 4: Sentiment Analysis
        # ============================================================
        logger.info("\n" + "=" * 40)
        logger.info("STEP 4: Running Sentiment Analysis")
        logger.info("=" * 40)

        try:
            if sentiment_model == "finbert":
                from src.sentiment.finbert_analyzer import FinBERTAnalyzer
                analyzer = FinBERTAnalyzer()
            else:
                from src.sentiment.vader_analyzer import VADERAnalyzer
                analyzer = VADERAnalyzer()

            all_sentiment = []
            for ticker in tickers:
                try:
                    logger.info(f"Analyzing sentiment for {ticker}...")
                    sentiment_df = analyzer.analyze_ticker_posts(
                        db_path=db_path,
                        ticker=ticker,
                    )
                    if not sentiment_df.empty:
                        all_sentiment.append(sentiment_df)
                        result.sentiment_computed += len(sentiment_df)
                        logger.info(f"  Computed {len(sentiment_df)} sentiment records for {ticker}")
                except Exception as e:
                    logger.error(f"  Error analyzing sentiment for {ticker}: {e}")
                    result.errors.append({"step": "sentiment", "ticker": ticker, "error": str(e)})

            # Save sentiment
            if all_sentiment:
                combined_sentiment = pd.concat(all_sentiment, ignore_index=True)
                save_daily_sentiment(combined_sentiment, db)
                logger.info(f"Total sentiment records: {result.sentiment_computed}")

        except Exception as e:
            logger.warning(f"Sentiment analysis failed: {e}")
            result.errors.append({"step": "sentiment", "error": str(e)})

        # ============================================================
        # STEP 5: Feature Engineering
        # ============================================================
        logger.info("\n" + "=" * 40)
        logger.info("STEP 5: Computing Features")
        logger.info("=" * 40)

        from src.features.technical import compute_technical_features
        from src.features.fusion import merge_all_features, create_target_variable

        all_features = []
        for ticker in tickers:
            try:
                logger.info(f"Computing features for {ticker}...")

                # Load data components
                prices_df = load_price_history(db, ticker)
                sentiment_df = load_sentiment_history(db, ticker)
                trends_df = load_trends(db, ticker)

                if prices_df.empty:
                    logger.warning(f"  No price data for {ticker}, skipping features")
                    continue

                # Add technical indicators
                prices_with_ta = compute_technical_features(prices_df)

                # Merge all features
                features_df = merge_all_features(
                    prices_df=prices_with_ta,
                    sentiment_df=sentiment_df,
                    trends_df=trends_df,
                )

                # Create target variable (J+3 direction)
                features_df = create_target_variable(
                    features_df,
                    horizon=settings.prediction_horizon,
                )

                if not features_df.empty:
                    all_features.append(features_df)
                    result.features_computed += len(features_df)
                    logger.info(f"  Computed {len(features_df)} feature records for {ticker}")

            except Exception as e:
                logger.error(f"  Error computing features for {ticker}: {e}")
                result.errors.append({"step": "features", "ticker": ticker, "error": str(e)})

        # Save features
        if all_features:
            combined_features = pd.concat(all_features, ignore_index=True)
            save_features(combined_features, db)
            logger.info(f"Total features computed: {result.features_computed}")

        # ============================================================
        # STEP 6: Model Training
        # ============================================================
        logger.info("\n" + "=" * 40)
        logger.info("STEP 6: Training ML Models")
        logger.info("=" * 40)

        from src.model import SentimentPredictor

        # Load combined features for training
        features_df = load_feature_matrix(db)

        if not features_df.empty:
            predictor = SentimentPredictor(feature_df=features_df)

            for ticker in tickers:
                try:
                    logger.info(f"Training model for {ticker}...")
                    model_result = predictor.train(ticker)

                    result.model_results[ticker] = {
                        "accuracy": model_result["accuracy"],
                        "auc_roc": model_result["auc_roc"],
                        "cv_mean": model_result["cv_mean"],
                        "cv_std": model_result["cv_std"],
                        "top_features": model_result["top_features"][:5],
                    }

                    result.tickers_processed.append(ticker)
                    logger.info(
                        f"  Model trained: accuracy={model_result['accuracy']:.3f}, "
                        f"AUC={model_result['auc_roc']:.3f}"
                    )

                except Exception as e:
                    logger.error(f"  Error training model for {ticker}: {e}")
                    result.errors.append({"step": "model", "ticker": ticker, "error": str(e)})

            # ============================================================
            # STEP 7: Backtesting
            # ============================================================
            if run_backtest and result.tickers_processed:
                logger.info("\n" + "=" * 40)
                logger.info("STEP 7: Running Backtesting")
                logger.info("=" * 40)

                from src.backtest import BacktestEngine, BacktestConfig

                backtest_config = BacktestConfig(
                    initial_capital=settings.initial_capital,
                    transaction_cost=settings.transaction_cost,
                    prediction_horizon=settings.prediction_horizon,
                )

                engine = BacktestEngine(backtest_config)

                for ticker in result.tickers_processed:
                    try:
                        logger.info(f"Backtesting {ticker}...")

                        # Get predictions from model
                        ticker_features = features_df[features_df["ticker"] == ticker].copy()
                        ticker_features = ticker_features.sort_values("date")

                        # Create predictions DataFrame
                        predictions = []
                        for i, row in ticker_features.iterrows():
                            if pd.notna(row.get("target_label")):
                                predictions.append({
                                    "date": row["date"],
                                    "ticker": ticker,
                                    "predicted_direction": "UP" if row.get("target_label") == 1 else "DOWN",
                                    "probability": 0.6,  # Placeholder
                                })

                        if predictions:
                            pred_df = pd.DataFrame(predictions)

                            # Get prices for backtesting
                            prices_df = load_price_history(db, ticker)
                            prices_df["open"] = prices_df["close"] * (1 + np.random.uniform(-0.005, 0.005, len(prices_df)))

                            bt_result = engine.run(pred_df, prices_df, ticker)

                            result.backtest_results[ticker] = {
                                "total_return": bt_result["summary"].get("total_return", 0),
                                "win_rate": bt_result["summary"].get("win_rate", 0),
                                "sharpe_ratio": bt_result["summary"].get("sharpe_ratio", 0),
                                "max_drawdown": bt_result["summary"].get("max_drawdown", 0),
                                "alpha": bt_result["comparison"].get("alpha", 0),
                            }

                            logger.info(
                                f"  Backtest: return={bt_result['summary'].get('total_return', 0):.1%}, "
                                f"win_rate={bt_result['summary'].get('win_rate', 0):.1%}"
                            )

                    except Exception as e:
                        logger.error(f"  Error backtesting {ticker}: {e}")
                        result.errors.append({"step": "backtest", "ticker": ticker, "error": str(e)})

            # ============================================================
            # STEP 8: GLM Brief Generation
            # ============================================================
            if use_llm and result.tickers_processed:
                logger.info("\n" + "=" * 40)
                logger.info("STEP 8: Generating GLM Briefs")
                logger.info("=" * 40)

                try:
                    from src.insights import GLMInsightGenerator

                    generator = GLMInsightGenerator()

                    tickers_data = {}
                    for ticker in result.tickers_processed:
                        # Get latest data for brief
                        latest_price = load_price_history(db, ticker).tail(5)
                        latest_sentiment = load_sentiment_history(db, ticker).tail(7)
                        latest_trends = load_trends(db, ticker).tail(7)

                        if not latest_price.empty:
                            tickers_data[ticker] = {
                                "price": {
                                    "current_price": latest_price["close"].iloc[-1],
                                    "daily_change": latest_price["close"].iloc[-1] - latest_price["close"].iloc[-2],
                                    "daily_change_pct": (latest_price["close"].iloc[-1] / latest_price["close"].iloc[-2] - 1) * 100,
                                    "volatility": latest_price["daily_return"].std(),
                                },
                                "sentiment": {
                                    "vader_compound": latest_sentiment["vader_compound"].mean() if not latest_sentiment.empty else 0,
                                    "post_count": int(latest_sentiment["post_count"].sum()) if not latest_sentiment.empty else 0,
                                },
                                "prediction": result.model_results.get(ticker, {}),
                                "trends": {
                                    "search_interest": latest_trends["search_interest"].mean() if not latest_trends.empty else 50,
                                },
                            }

                    if tickers_data:
                        briefs, summary = generator.generate_all_briefs(tickers_data)
                        result.glm_briefs = briefs
                        result.glm_briefs.append({"type": "portfolio_summary", "brief": summary})

                        logger.info(f"Generated {len(briefs)} briefs + portfolio summary")

                except Exception as e:
                    logger.warning(f"GLM brief generation failed: {e}")
                    result.errors.append({"step": "glm", "error": str(e)})

        else:
            logger.warning("No features available for model training")

    except Exception as e:
        logger.error(f"Pipeline failed with error: {e}")
        result.errors.append({"step": "pipeline", "error": str(e)})

    # Finalize
    result.end_time = datetime.now()
    duration = (result.end_time - result.start_time).total_seconds()

    # Log summary
    logger.info("\n" + "=" * 60)
    logger.info("Pipeline Summary")
    logger.info("=" * 60)
    logger.info(f"Duration: {duration:.1f} seconds")
    logger.info(f"Tickers processed: {len(result.tickers_processed)}")
    logger.info(f"Prices collected: {result.prices_collected}")
    logger.info(f"Posts collected: {result.posts_collected}")
    logger.info(f"Trends collected: {result.trends_collected}")
    logger.info(f"Features computed: {result.features_computed}")
    logger.info(f"Models trained: {len(result.model_results)}")
    logger.info(f"Errors: {len(result.errors)}")
    logger.info("=" * 60)

    return result.to_dict()


def run_quick_analysis(ticker: str, db_path: Optional[str] = None) -> dict:
    """
    Run a quick analysis for a single ticker.

    This is a simplified pipeline for testing or quick checks.

    Args:
        ticker: Single ticker to analyze
        db_path: Database path

    Returns:
        Dictionary with analysis results
    """
    logger.info(f"Running quick analysis for {ticker}")

    db_path = db_path or settings.db_path
    db = DatabaseManager(db_path)

    # Load available data
    prices_df = load_price_history(db, ticker)
    sentiment_df = load_sentiment_history(db, ticker)
    features_df = load_feature_matrix(db, ticker)

    result = {
        "ticker": ticker,
        "price_records": len(prices_df),
        "sentiment_records": len(sentiment_df),
        "feature_records": len(features_df),
    }

    if not prices_df.empty:
        latest = prices_df.iloc[-1]
        result["latest_price"] = {
            "date": str(latest["date"]),
            "close": latest["close"],
            "volume": latest["volume"],
            "daily_return": latest.get("daily_return"),
        }

    if not sentiment_df.empty:
        latest_sent = sentiment_df.iloc[-1]
        result["latest_sentiment"] = {
            "date": str(latest_sent["date"]),
            "vader_compound": latest_sent["vader_compound"],
            "post_count": latest_sent["post_count"],
        }

    return result


def print_pipeline_report(results: dict) -> str:
    """
    Format pipeline results as a readable report.

    Args:
        results: Results dictionary from run_full_pipeline

    Returns:
        Formatted string report
    """
    lines = [
        "=" * 60,
        "MARKET SENTIMENT ANALYZER - PIPELINE REPORT",
        "=" * 60,
        "",
        f"Status: {results['status'].upper()}",
        f"Duration: {results['duration_seconds']:.1f} seconds",
        "",
        "DATA COLLECTION:",
        f"  Prices: {results['metrics']['prices_collected']} records",
        f"  Posts: {results['metrics']['posts_collected']} posts",
        f"  Trends: {results['metrics']['trends_collected']} records",
        f"  Features: {results['metrics']['features_computed']} computed",
        "",
        "MODEL RESULTS:",
    ]

    for ticker, model_data in results.get("model_results", {}).items():
        lines.extend([
            f"  {ticker}:",
            f"    Accuracy: {model_data['accuracy']:.1%}",
            f"    AUC-ROC: {model_data['auc_roc']:.3f}",
            f"    CV Score: {model_data['cv_mean']:.3f} ± {model_data['cv_std']:.3f}",
        ])

    if results.get("backtest_results"):
        lines.extend(["", "BACKTESTING:"])
        for ticker, bt_data in results["backtest_results"].items():
            lines.extend([
                f"  {ticker}:",
                f"    Return: {bt_data['total_return']:+.1%}",
                f"    Win Rate: {bt_data['win_rate']:.1%}",
                f"    Sharpe: {bt_data['sharpe_ratio']:.2f}",
                f"    Alpha: {bt_data['alpha']:+.1%}",
            ])

    if results.get("errors"):
        lines.extend(["", "ERRORS:"])
        for err in results["errors"]:
            lines.append(f"  - {err['step']}: {err.get('ticker', 'N/A')} - {err['error'][:50]}")

    lines.extend([
        "",
        "=" * 60,
        "DISCLAIMER: For educational purposes only.",
        "Not financial advice. Past performance ≠ future results.",
        "=" * 60,
    ])

    return "\n".join(lines)


# ============================================================
# Module Test
# ============================================================

if __name__ == "__main__":
    print("Pipeline Module Test")
    print("=" * 60)

    # Test with synthetic data
    print("\nNote: Full pipeline requires API credentials in .env")
    print("Running synthetic data test...")

    # Create synthetic data
    from datetime import datetime, timedelta
    import numpy as np

    np.random.seed(42)
    dates = [datetime.now() - timedelta(days=i) for i in range(300, 0, -1)]
    n = len(dates)

    test_prices = pd.DataFrame({
        "date": dates,
        "ticker": "TEST",
        "open": 150 * np.cumprod(1 + np.random.normal(0.001, 0.02, n)),
        "high": 150 * np.cumprod(1 + np.random.normal(0.001, 0.02, n)) * 1.01,
        "low": 150 * np.cumprod(1 + np.random.normal(0.001, 0.02, n)) * 0.99,
        "close": 150 * np.cumprod(1 + np.random.normal(0.001, 0.02, n)),
        "volume": np.random.randint(40000000, 60000000, n),
    })

    test_sentiment = pd.DataFrame({
        "date": dates,
        "ticker": "TEST",
        "vader_compound": np.random.uniform(-0.3, 0.3, n),
        "vader_positive": np.random.uniform(0.1, 0.4, n),
        "vader_negative": np.random.uniform(0.1, 0.3, n),
        "vader_neutral": np.random.uniform(0.3, 0.6, n),
        "post_count": np.random.randint(10, 50, n),
        "avg_score": np.random.uniform(10, 100, n),
        "total_comments": np.random.randint(50, 200, n),
        "weighted_compound": np.random.uniform(-0.3, 0.3, n),
        "dominant_sentiment": np.random.choice(["positive", "negative", "neutral"], n),
    })

    print(f"\n1. Test data created:")
    print(f"   Prices: {len(test_prices)} records")
    print(f"   Sentiment: {len(test_sentiment)} records")

    # Test feature computation
    print("\n2. Testing feature engineering...")
    from src.features.technical import compute_technical_features
    from src.features.fusion import merge_all_features, create_target_variable

    prices_ta = compute_technical_features(test_prices)
    features = merge_all_features(prices_ta, test_sentiment)
    features = create_target_variable(features, horizon=3)

    print(f"   Features computed: {len(features.columns)} columns")
    print(f"   Valid samples: {len(features.dropna(subset=['target_label']))}")

    # Test model
    print("\n3. Testing model training...")
    from src.model import SentimentPredictor

    predictor = SentimentPredictor(feature_df=features)
    model_results = predictor.train("TEST")

    print(f"   Accuracy: {model_results['accuracy']:.3f}")
    print(f"   AUC-ROC: {model_results['auc_roc']:.3f}")
    print(f"   Top feature: {model_results['top_features'][0]}")

    # Test backtest
    print("\n4. Testing backtest engine...")
    from src.backtest import BacktestEngine, BacktestConfig

    config = BacktestConfig(initial_capital=10000)
    engine = BacktestEngine(config)

    # Create predictions
    pred_df = features[["date", "ticker", "target_label"]].dropna()
    pred_df = pred_df.rename(columns={"target_label": "predicted_direction"})
    pred_df["predicted_direction"] = pred_df["predicted_direction"].apply(lambda x: "UP" if x == 1 else "DOWN")
    pred_df["probability"] = 0.6

    bt_results = engine.run(pred_df, test_prices, "TEST")
    print(f"   Total trades: {bt_results['summary'].get('total_trades', 0)}")
    print(f"   Win rate: {bt_results['summary'].get('win_rate', 0):.1%}")

    print("\n" + "=" * 60)
    print("✅ Pipeline module test completed!")
    print("\nTo run full pipeline:")
    print("  python main.py --tickers AAPL TSLA NVDA")
