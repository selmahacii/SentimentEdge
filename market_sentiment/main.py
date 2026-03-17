#!/usr/bin/env python3
"""
Market Sentiment Analyzer - Entry Point

This is the main entry point for the Market Sentiment Analyzer.
Provides CLI interface for running the full pipeline or individual components.

Usage:
    python main.py --tickers AAPL TSLA NVDA --days 90
    python main.py --model finbert --no-llm
    python main.py --help

DISCLAIMER: This project is for educational purposes only.
It does not constitute financial advice. Never use it for real investment decisions.
"""

import argparse
from pathlib import Path

from loguru import logger

# Configure logging
logger.add(
    "logs/market_sentiment.log",
    rotation="10 MB",
    retention="7 days",
    level="INFO",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {module}:{function}:{line} | {message}"
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Market Sentiment Analyzer - Predict stock movements from social sentiment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python main.py --tickers AAPL TSLA NVDA --days 90
    python main.py --model finbert --no-llm
    python main.py --backtest-only
    python main.py --init-db

DISCLAIMER: This is for educational purposes only.
Not financial advice. Never use for real investment decisions.

        """
    )

    parser.add_argument(
        "--tickers",
        nargs="+",
        default=None,
        help="Space-separated list of tickers (default: from config)"
    )

    parser.add_argument(
        "--days",
        type=int,
        default=None,
        help="Number of days to look back (default: from config)"
    )

    parser.add_argument(
        "--model",
        choices=["vader", "finbert"],
        default="vader",
        help="Sentiment model: vader (fast) or finbert (accurate)"
    )

    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Skip GLM brief generation (useful for testing)"
    )

    parser.add_argument(
        "--no-backtest",
        action="store_true",
        help="Skip backtesting step"
    )

    parser.add_argument(
        "--init-db",
        action="store_true",
        help="Initialize database tables and exit"
    )

    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run quick analysis for single ticker (requires --tickers with one ticker)"
    )

    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run with synthetic demo data (no API calls required)"
    )

    return parser.parse_args()


def run_demo_mode() -> None:
    """Run pipeline with synthetic demo data."""
    import numpy as np
    import pandas as pd
    from datetime import datetime, timedelta

    print("\n" + "=" * 60)
    print("DEMO MODE - Using Synthetic Data")
    print("=" * 60)

    # Create synthetic data
    np.random.seed(42)
    tickers = ["AAPL", "TSLA", "NVDA"]
    days = 300

    print(f"\n1. Creating synthetic data for {len(tickers)} tickers...")

    all_features = []

    for ticker in tickers:
        dates = [datetime.now() - timedelta(days=i) for i in range(days, 0, -1)]
        n = len(dates)

        # Generate price data
        closes = 100 * np.cumprod(1 + np.random.normal(0.001, 0.02, n))

        features = pd.DataFrame({
            "date": dates,
            "ticker": ticker,
            "close": closes,
            "volume": np.random.randint(40000000, 60000000, n),
            "daily_return": np.diff(closes, prepend=closes[0]) / np.roll(closes, 1),
            "log_return": np.log(closes / np.roll(closes, 1)),
            "volatility_10d": pd.Series(closes).pct_change().rolling(10).std().fillna(0).values,
            "volume_ratio": np.random.uniform(0.5, 2.0, n),
            "momentum_5d": pd.Series(closes).pct_change(5).fillna(0).values,
            "momentum_20d": pd.Series(closes).pct_change(20).fillna(0).values,
            "rsi_14": np.random.uniform(30, 70, n),
            "macd": np.random.normal(0, 1, n),
            "macd_signal": np.random.normal(0, 1, n),
            "macd_hist": np.random.normal(0, 0.5, n),
            "bb_position": np.random.uniform(0, 1, n),
            "vader_compound": np.random.uniform(-0.3, 0.3, n),
            "vader_compound_lag1": np.random.uniform(-0.3, 0.3, n),
            "vader_compound_lag2": np.random.uniform(-0.3, 0.3, n),
            "vader_compound_lag3": np.random.uniform(-0.3, 0.3, n),
            "sentiment_momentum": np.random.uniform(-0.2, 0.2, n),
            "post_count": np.random.randint(10, 50, n),
            "search_interest": np.random.uniform(30, 70, n),
            "sentiment_x_volume": np.random.uniform(-0.5, 0.5, n),
        })

        # Create target (J+3 direction)
        features["future_close"] = features["close"].shift(-3)
        features["target_return"] = (features["future_close"] - features["close"]) / features["close"]
        features["target_label"] = (features["target_return"] > 0).astype(int)
        features = features.dropna(subset=["target_label"])

        all_features.append(features)

    combined_features = pd.concat(all_features, ignore_index=True)
    print(f"   Created {len(combined_features)} feature records")

    # Train models
    print("\n2. Training ML models...")
    from src.model import SentimentPredictor

    predictor = SentimentPredictor(feature_df=combined_features)

    model_results = {}
    for ticker in tickers:
        try:
            result = predictor.train(ticker)
            model_results[ticker] = result
            print(f"   {ticker}: accuracy={result['accuracy']:.3f}, AUC={result['auc_roc']:.3f}")
        except Exception as e:
            print(f"   {ticker}: ERROR - {e}")

    # Generate predictions
    print("\n3. Generating predictions...")
    predictions = {}
    for ticker in tickers:
        try:
            pred = predictor.predict_tomorrow(ticker)
            predictions[ticker] = pred
            print(f"   {ticker}: {pred['prediction']} ({pred['probability']:.0%} confidence)")
        except Exception as e:
            print(f"   {ticker}: ERROR - {e}")

    # Run backtest
    print("\n4. Running backtesting...")
    from src.backtest import BacktestEngine, BacktestConfig

    for ticker in tickers:
        ticker_features = combined_features[combined_features["ticker"] == ticker].copy()
        prices_df = ticker_features[["date", "ticker", "close"]].copy()
        prices_df["open"] = prices_df["close"] * (1 + np.random.uniform(-0.005, 0.005, len(prices_df)))

        pred_df = ticker_features[["date", "ticker", "target_label"]].copy()
        pred_df = pred_df.rename(columns={"target_label": "predicted_direction"})
        pred_df["predicted_direction"] = pred_df["predicted_direction"].apply(lambda x: "UP" if x == 1 else "DOWN")
        pred_df["probability"] = 0.6

        config = BacktestConfig(initial_capital=10000, transaction_cost=0.001)
        engine = BacktestEngine(config)
        bt_result = engine.run(pred_df, prices_df, ticker)

        summary = bt_result["summary"]
        print(f"   {ticker}: return={summary.get('total_return', 0):+.1%}, "
              f"win_rate={summary.get('win_rate', 0):.1%}, "
              f"trades={summary.get('total_trades', 0)}")

    # Sentiment correlation
    print("\n5. Analyzing sentiment-price correlation...")
    for ticker in tickers:
        try:
            corr = predictor.get_sentiment_price_correlation(ticker)
            print(f"   {ticker}: best_lag={corr.get('best_lag', 'N/A')}, "
                  f"correlation={corr.get('best_correlation', 0):.3f}")
        except Exception as e:
            print(f"   {ticker}: ERROR - {e}")

    # Feature importance
    print("\n6. Feature importance analysis...")
    for ticker in tickers[:1]:  # Just show one
        try:
            importance_df = predictor.get_feature_importance_analysis(ticker, top_n=5)
            print(f"   {ticker} top features:")
            for _, row in importance_df.iterrows():
                print(f"      {row['rank']}. {row['feature']}: {row['importance']:.4f} ({row['category']})")
        except Exception as e:
            print(f"   ERROR - {e}")

    print("\n" + "=" * 60)
    print("✅ Demo mode completed!")
    print("=" * 60)


def main() -> None:
    """Main entry point."""
    args = parse_args()

    logger.info("=" * 60)
    logger.info("Market Sentiment Analyzer - Starting")
    logger.info("=" * 60)

    # Ensure required directories exist
    Path("data/raw/reddit").mkdir(parents=True, exist_ok=True)
    Path("data/raw/prices").mkdir(parents=True, exist_ok=True)
    Path("data/raw/trends").mkdir(parents=True, exist_ok=True)
    Path("data/cleaned").mkdir(parents=True, exist_ok=True)
    Path("data/features").mkdir(parents=True, exist_ok=True)
    Path("logs").mkdir(parents=True, exist_ok=True)

    # Demo mode
    if args.demo:
        run_demo_mode()
        return

    # Load configuration
    try:
        from src.config import settings
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        logger.info("Make sure you have a .env file with required credentials")
        logger.info("See .env.example for template")
        return

    # Override config with CLI args
    tickers = args.tickers or settings.tickers
    use_llm = not args.no_llm
    run_backtest = not args.no_backtest

    logger.info(f"Tickers: {tickers}")
    logger.info(f"Sentiment model: {args.model}")
    logger.info(f"LLM briefs: {'enabled' if use_llm else 'disabled'}")

    # Initialize database if requested
    if args.init_db:
        from src.storage import DatabaseManager
        db = DatabaseManager(settings.db_path)
        db.create_tables()
        logger.info("Database initialized successfully")
        print("\n✅ Database tables created!")
        return

    # Quick analysis mode
    if args.quick:
        if len(tickers) != 1:
            print("Error: --quick requires exactly one ticker with --tickers")
            return
        from src.pipeline import run_quick_analysis
        result = run_quick_analysis(tickers[0])
        print(f"\nQuick Analysis for {tickers[0]}:")
        print(f"  Price records: {result['price_records']}")
        print(f"  Sentiment records: {result['sentiment_records']}")
        if 'latest_price' in result:
            print(f"  Latest price: ${result['latest_price']['close']:.2f}")
        return

    # Run full pipeline
    try:
        from src.pipeline import run_full_pipeline, print_pipeline_report

        results = run_full_pipeline(
            tickers=tickers,
            use_llm=use_llm,
            sentiment_model=args.model,
            days=args.days,
            run_backtest=run_backtest,
        )

        # Print report
        print("\n" + print_pipeline_report(results))

        # Print GLM briefs if available
        if results.get("glm_briefs"):
            print("\n" + "=" * 60)
            print("AI-GENERATED DAILY BRIEFS")
            print("=" * 60)
            for brief in results["glm_briefs"]:
                if brief.get("type") == "portfolio_summary":
                    print("\n📊 PORTFOLIO SUMMARY:")
                    print(brief.get("brief", {}).get("summary", "No summary"))
                else:
                    print(f"\n📈 {brief.get('ticker', 'Unknown')}:")
                    print(brief.get("brief", "No brief"))

    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")
        raise


if __name__ == "__main__":
    main()
