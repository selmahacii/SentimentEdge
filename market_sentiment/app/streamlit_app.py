"""
Market Sentiment Analyzer - Streamlit Dashboard

Professional financial dashboard for visualizing:
- Price action and technical indicators
- Social media sentiment over time
- ML predictions and confidence
- Backtesting results
- AI-generated daily briefs

Usage:
    streamlit run app/streamlit_app.py

DISCLAIMER: This dashboard is for educational purposes only.
It does not constitute financial advice.
"""

import numpy as np
import pandas as pd
import streamlit as st
from datetime import datetime, timedelta
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Page configuration
st.set_page_config(
    page_title="Market Sentiment Analyzer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown("""
<style>
    .stMetric {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
    }
    .positive {
        color: #00c853;
    }
    .negative {
        color: #ff5252;
    }
    .neutral {
        color: #ffc107;
    }
    .confidence-high {
        color: #4caf50;
        font-weight: bold;
    }
    .confidence-medium {
        color: #ff9800;
        font-weight: bold;
    }
    .confidence-low {
        color: #f44336;
        font-weight: bold;
    }
    .brief-card {
        background-color: #f8f9fa;
        padding: 20px;
        border-radius: 10px;
        border-left: 4px solid #2196f3;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)


def get_sentiment_emoji(compound: float) -> str:
    """Get emoji for sentiment score."""
    if compound > 0.2:
        return "🟢"
    elif compound < -0.2:
        return "🔴"
    return "🟡"


def get_confidence_class(confidence: str) -> str:
    """Get CSS class for confidence level."""
    return f"confidence-{confidence}"


def render_sidebar() -> tuple:
    """Render the sidebar with controls."""
    st.sidebar.title("📊 Configuration")

    # Ticker selection
    st.sidebar.subheader("Stock Selection")
    default_tickers = ["AAPL", "TSLA", "NVDA"]
    tickers = st.sidebar.multiselect(
        "Select Tickers",
        options=["AAPL", "TSLA", "NVDA", "MSFT", "AMZN", "GOOGL", "META", "AMD"],
        default=default_tickers,
        help="Select stocks to analyze"
    )

    # Date range
    st.sidebar.subheader("Date Range")
    date_range = st.sidebar.slider(
        "Days to analyze",
        min_value=30,
        max_value=365,
        value=90,
        help="Historical data lookback period"
    )

    # Model selection
    st.sidebar.subheader("Model Settings")
    sentiment_model = st.sidebar.radio(
        "Sentiment Model",
        options=["VADER (fast)", "FinBERT (accurate)"],
        help="VADER is faster, FinBERT is more accurate for financial text"
    )

    # Run button
    run_analysis = st.sidebar.button("🚀 Run Analysis", type="primary")

    # Demo mode button
    demo_mode = st.sidebar.button("📊 Demo Mode", help="Use synthetic data for demonstration")

    # Info box
    st.sidebar.info(
        "Data from Yahoo Finance + Reddit.\n"
        "Educational use only.\n\n"
        "⚠️ Not financial advice."
    )

    # Last update
    st.sidebar.caption(f"Last update: {datetime.now().strftime('%Y-%m-%d %H:%M')}")

    return tickers, date_range, sentiment_model, run_analysis, demo_mode


def render_overview_tab(tickers: list, predictions: dict, sentiment_data: dict) -> None:
    """Render the Overview tab with metric cards."""
    st.header("📈 Market Overview")

    if not predictions:
        st.info("Click 'Run Analysis' to generate predictions")
        return

    # Create columns based on number of tickers
    cols = st.columns(min(len(tickers), 4))

    for i, ticker in enumerate(tickers[:4]):
        with cols[i]:
            pred = predictions.get(ticker, {})
            sent = sentiment_data.get(ticker, {})

            # Price metric
            current_price = pred.get("current_price", 0)
            daily_change_pct = pred.get("daily_change_pct", 0)

            st.metric(
                label=ticker,
                value=f"${current_price:,.2f}" if current_price else "N/A",
                delta=f"{daily_change_pct:+.2f}%" if daily_change_pct else None,
                delta_color="normal"
            )

            # Sentiment indicator
            vader = sent.get("vader_compound", 0)
            sentiment_emoji = get_sentiment_emoji(vader)
            sentiment_label = "Bullish" if vader > 0.2 else "Bearish" if vader < -0.2 else "Neutral"
            st.markdown(f"**Sentiment:** {sentiment_emoji} {sentiment_label}")

            # ML Prediction
            prediction = pred.get("prediction", "N/A")
            probability = pred.get("probability", 0)
            confidence = pred.get("confidence", "unknown")

            pred_color = "positive" if prediction == "UP" else "negative"
            st.markdown(f"**ML Prediction:** :{pred_color}[{prediction}] ({probability:.0%})")
            st.caption(f"Confidence: {confidence}")


def render_sentiment_tab(tickers: list, sentiment_data: dict, date_range: int) -> None:
    """Render the Sentiment Analysis tab."""
    st.header("🔍 Sentiment Analysis")

    if not sentiment_data:
        st.info("Run analysis to see sentiment data")
        return

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Sentiment Timeline")

        # Create sample sentiment timeline chart
        dates = pd.date_range(end=datetime.now(), periods=date_range, freq='D')

        chart_data = pd.DataFrame({
            'Date': dates,
        })

        for ticker in tickers[:3]:
            # Generate sample sentiment data
            np.random.seed(hash(ticker) % 2**32)
            chart_data[ticker] = np.cumsum(np.random.uniform(-0.05, 0.05, date_range))
            chart_data[ticker] = chart_data[ticker].clip(-1, 1)

        chart_data = chart_data.set_index('Date')

        st.line_chart(chart_data)

        st.caption("VADER compound sentiment score over time (-1 to +1)")

    with col2:
        st.subheader("Reddit Activity")

        # Activity bar chart
        activity_data = pd.DataFrame({
            'Ticker': tickers[:5],
            'Posts': [sentiment_data.get(t, {}).get('post_count', np.random.randint(20, 100))
                      for t in tickers[:5]],
            'Avg Sentiment': [abs(sentiment_data.get(t, {}).get('vader_compound', 0)) * 100
                             for t in tickers[:5]]
        })

        st.bar_chart(activity_data.set_index('Ticker'))

        # Sentiment breakdown
        st.subheader("Sentiment Breakdown")
        for ticker in tickers[:3]:
            sent = sentiment_data.get(ticker, {})
            vader = sent.get("vader_compound", 0)

            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.metric(f"{ticker}", f"{vader:.3f}")
            with col_b:
                posts = sent.get("post_count", 0)
                st.metric("Posts", f"{posts}")
            with col_c:
                trend = sent.get("trend", "stable")
                trend_emoji = "📈" if trend == "improving" else "📉" if trend == "declining" else "➡️"
                st.metric("Trend", f"{trend_emoji} {trend}")


def render_technical_tab(tickers: list, price_data: dict, date_range: int) -> None:
    """Render the Technical Analysis tab."""
    st.header("📊 Technical Analysis")

    selected_ticker = st.selectbox("Select Ticker", tickers)

    if selected_ticker not in price_data:
        st.info(f"No technical data available for {selected_ticker}")
        return

    data = price_data[selected_ticker]

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Price Chart")

        # Create candlestick-like chart using line chart
        dates = pd.date_range(end=datetime.now(), periods=date_range, freq='D')
        np.random.seed(hash(selected_ticker) % 2**32)

        prices = data.get("close_prices", 100 * np.cumprod(1 + np.random.normal(0.001, 0.02, date_range)))

        price_df = pd.DataFrame({
            'Date': dates,
            'Close': prices,
            'SMA_20': pd.Series(prices).rolling(20).mean().fillna(method='bfill'),
            'SMA_50': pd.Series(prices).rolling(50).mean().fillna(method='bfill'),
        }).set_index('Date')

        st.line_chart(price_df)

        st.caption(f"{selected_ticker} price with 20/50 day moving averages")

    with col2:
        st.subheader("Technical Indicators")

        # RSI
        rsi = data.get("rsi_14", np.random.uniform(30, 70))
        rsi_signal = "Overbought" if rsi > 70 else "Oversold" if rsi < 30 else "Neutral"
        rsi_color = "negative" if rsi > 70 else "positive" if rsi < 30 else "neutral"
        st.metric("RSI (14)", f"{rsi:.1f}", f"{rsi_signal}", delta_color="normal")

        # MACD
        macd = data.get("macd", 0)
        macd_signal = data.get("macd_signal", 0)
        macd_cross = "Bullish" if macd > macd_signal else "Bearish"
        st.metric("MACD Signal", f"{macd_cross}", f"MACD: {macd:.2f}")

        # Bollinger Bands
        bb_position = data.get("bb_position", 0.5)
        bb_signal = "Near Upper" if bb_position > 0.8 else "Near Lower" if bb_position < 0.2 else "Middle"
        st.metric("BB Position", f"{bb_position:.0%}", f"{bb_signal}")

        # Volume
        volume_ratio = data.get("volume_ratio", 1.0)
        vol_signal = "High" if volume_ratio > 1.5 else "Low" if volume_ratio < 0.5 else "Normal"
        st.metric("Volume Ratio", f"{volume_ratio:.2f}x", f"{vol_signal}")

    # Technical summary
    st.subheader("Technical Summary")

    summary_cols = st.columns(5)
    indicators = [
        ("RSI", rsi_signal, rsi_color),
        ("MACD", macd_cross, "positive" if macd_cross == "Bullish" else "negative"),
        ("BB", bb_signal, "neutral"),
        ("Volume", vol_signal, "neutral"),
        ("Trend", "Uptrend" if prices[-1] > prices[-20] else "Downtrend",
         "positive" if prices[-1] > prices[-20] else "negative"),
    ]

    for i, (name, signal, color) in enumerate(indicators):
        with summary_cols[i]:
            st.metric(name, signal)


def render_ml_tab(tickers: list, model_results: dict, backtest_results: dict) -> None:
    """Render the ML Predictions tab."""
    st.header("🤖 ML Predictions")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Model Performance")

        if not model_results:
            st.info("Train models to see performance metrics")
            return

        # Performance comparison
        perf_data = []
        for ticker, result in model_results.items():
            perf_data.append({
                'Ticker': ticker,
                'Accuracy': result.get('accuracy', 0),
                'AUC-ROC': result.get('auc_roc', 0),
                'CV Mean': result.get('cv_mean', 0),
            })

        perf_df = pd.DataFrame(perf_data).set_index('Ticker')

        st.bar_chart(perf_df)

        # Metrics table
        st.subheader("Detailed Metrics")
        st.dataframe(
            pd.DataFrame(model_results).T[['accuracy', 'auc_roc', 'cv_mean', 'cv_std']]
            .rename(columns={
                'accuracy': 'Accuracy',
                'auc_roc': 'AUC-ROC',
                'cv_mean': 'CV Mean',
                'cv_std': 'CV Std'
            })
            .style.format({
                'Accuracy': '{:.1%}',
                'AUC-ROC': '{:.3f}',
                'CV Mean': '{:.3f}',
                'CV Std': '{:.3f}',
            }),
            use_container_width=True
        )

    with col2:
        st.subheader("Feature Importance")

        # Show top features for first ticker
        if tickers and tickers[0] in model_results:
            top_features = model_results[tickers[0]].get('top_features', [])[:10]
            if top_features:
                feature_df = pd.DataFrame({
                    'Feature': [f[0] if isinstance(f, tuple) else f for f in top_features],
                    'Importance': [f[1] if isinstance(f, tuple) else i/len(top_features)
                                  for i, f in enumerate(top_features)]
                })
                st.bar_chart(feature_df.set_index('Feature'))

        st.subheader("Backtesting Results")

        if backtest_results:
            for ticker, bt in backtest_results.items():
                with st.expander(f"📊 {ticker} Backtest", expanded=False):
                    col_a, col_b, col_c, col_d = st.columns(4)
                    with col_a:
                        st.metric("Total Return", f"{bt.get('total_return', 0):+.1%}")
                    with col_b:
                        st.metric("Win Rate", f"{bt.get('win_rate', 0):.1%}")
                    with col_c:
                        st.metric("Sharpe Ratio", f"{bt.get('sharpe_ratio', 0):.2f}")
                    with col_d:
                        st.metric("Alpha", f"{bt.get('alpha', 0):+.1%}")

                    st.metric("Max Drawdown", f"{bt.get('max_drawdown', 0):.1%}")
        else:
            st.info("Run backtesting to see results")


def render_briefs_tab(briefs: list) -> None:
    """Render the AI Briefs tab."""
    st.header("📝 AI Daily Briefs")

    st.warning(
        "⚠️ **Educational Only** - These AI-generated briefs are for educational "
        "purposes only and do not constitute financial advice."
    )

    if not briefs:
        st.info("Run analysis with LLM enabled to generate AI briefs")
        return

    # Portfolio summary first
    portfolio_summary = next((b for b in briefs if b.get('type') == 'portfolio_summary'), None)

    if portfolio_summary:
        st.subheader("📊 Portfolio Summary")
        summary_content = portfolio_summary.get('brief', portfolio_summary.get('summary', 'No summary'))
        if isinstance(summary_content, dict):
            summary_text = summary_content.get('summary', 'No summary')
        else:
            summary_text = summary_content
        st.markdown(f"<div class='brief-card'>{summary_text}</div>", unsafe_allow_html=True)

    # Individual briefs
    st.subheader("Individual Stock Briefs")

    ticker_briefs = [b for b in briefs if b.get('type') != 'portfolio_summary']

    cols = st.columns(min(len(ticker_briefs), 3))

    for i, brief in enumerate(ticker_briefs):
        with cols[i % 3]:
            ticker = brief.get('ticker', 'Unknown')
            brief_text = brief.get('brief', 'No brief available')

            with st.expander(f"📈 {ticker} Brief", expanded=False):
                st.markdown(brief_text)

                # Metadata
                if 'tokens_used' in brief:
                    st.caption(f"Tokens: {brief['tokens_used']} | Model: {brief.get('model', 'N/A')}")


def generate_demo_data(tickers: list, date_range: int) -> tuple:
    """Generate demo data for dashboard demonstration."""
    np.random.seed(42)

    predictions = {}
    sentiment_data = {}
    price_data = {}
    model_results = {}
    backtest_results = {}
    briefs = []

    base_prices = {"AAPL": 178, "TSLA": 245, "NVDA": 450, "MSFT": 379, "AMZN": 178}
    base_vols = {"AAPL": 50e6, "TSLA": 80e6, "NVDA": 40e6, "MSFT": 20e6, "AMZN": 30e6}

    for ticker in tickers:
        base_price = base_prices.get(ticker, 100)

        # Predictions
        predictions[ticker] = {
            "current_price": base_price * (1 + np.random.uniform(-0.05, 0.05)),
            "daily_change_pct": np.random.uniform(-3, 3),
            "prediction": np.random.choice(["UP", "DOWN"]),
            "probability": np.random.uniform(0.55, 0.85),
            "confidence": np.random.choice(["high", "medium", "low"], p=[0.3, 0.5, 0.2]),
        }

        # Sentiment
        sentiment_data[ticker] = {
            "vader_compound": np.random.uniform(-0.4, 0.5),
            "post_count": int(np.random.uniform(20, 100)),
            "trend": np.random.choice(["improving", "declining", "stable"]),
        }

        # Price data for technical
        dates = pd.date_range(end=datetime.now(), periods=date_range, freq='D')
        prices = base_price * np.cumprod(1 + np.random.normal(0.001, 0.02, date_range))

        price_data[ticker] = {
            "close_prices": prices,
            "rsi_14": np.random.uniform(30, 70),
            "macd": np.random.normal(0, 1),
            "macd_signal": np.random.normal(0, 1),
            "bb_position": np.random.uniform(0.1, 0.9),
            "volume_ratio": np.random.uniform(0.5, 2.0),
        }

        # Model results
        model_results[ticker] = {
            "accuracy": np.random.uniform(0.52, 0.68),
            "auc_roc": np.random.uniform(0.55, 0.75),
            "cv_mean": np.random.uniform(0.52, 0.65),
            "cv_std": np.random.uniform(0.03, 0.08),
            "top_features": [
                ("vader_compound", np.random.uniform(0.1, 0.2)),
                ("rsi_14", np.random.uniform(0.08, 0.15)),
                ("volume_ratio", np.random.uniform(0.05, 0.12)),
                ("sentiment_momentum", np.random.uniform(0.04, 0.1)),
                ("macd", np.random.uniform(0.03, 0.08)),
            ],
        }

        # Backtest results
        backtest_results[ticker] = {
            "total_return": np.random.uniform(-0.15, 0.25),
            "win_rate": np.random.uniform(0.45, 0.65),
            "sharpe_ratio": np.random.uniform(0.5, 2.0),
            "max_drawdown": np.random.uniform(0.05, 0.25),
            "alpha": np.random.uniform(-0.1, 0.15),
        }

        # Brief
        briefs.append({
            "ticker": ticker,
            "brief": f"""## {ticker} — Daily Brief

**Situation**: {ticker} is trading at ${predictions[ticker]['current_price']:.2f}, {'up' if predictions[ticker]['daily_change_pct'] > 0 else 'down'} {abs(predictions[ticker]['daily_change_pct']):.1f}% today.

**Sentiment**: Reddit sentiment is {'positive' if sentiment_data[ticker]['vader_compound'] > 0 else 'negative'} (VADER: {sentiment_data[ticker]['vader_compound']:.3f}).

**Technical**: RSI at {price_data[ticker]['rsi_14']:.1f} suggests {'overbought' if price_data[ticker]['rsi_14'] > 70 else 'oversold' if price_data[ticker]['rsi_14'] < 30 else 'neutral'} conditions.

**Signal**: Model predicts {'UP' if predictions[ticker]['prediction'] == 'UP' else 'DOWN'} movement with {predictions[ticker]['probability']:.0%} probability ({predictions[ticker]['confidence']} confidence).

**Risk factors**: Sentiment can shift rapidly; monitor for news events.""",
        })

    # Portfolio summary
    briefs.append({
        "type": "portfolio_summary",
        "brief": {
            "summary": f"""Portfolio of {len(tickers)} stocks shows mixed signals.

Common themes: Tech sector sentiment is cautiously optimistic. Volume patterns suggest institutional interest in {tickers[0] if tickers else 'N/A'}.

Divergent signals: {tickers[0] if len(tickers) > 0 else 'N/A'} shows bullish sentiment while {tickers[1] if len(tickers) > 1 else 'N/A'} shows bearish divergence.

Overall sentiment: Neutral with slight bullish bias.

DISCLAIMER: For educational purposes only. Not financial advice."""
        }
    })

    return predictions, sentiment_data, price_data, model_results, backtest_results, briefs


def main() -> None:
    """Main dashboard entry point."""
    # Title
    st.title("📈 Market Sentiment Analyzer")
    st.markdown("*Predicting stock movements from social sentiment*")

    # Render sidebar
    tickers, date_range, sentiment_model, run_analysis, demo_mode = render_sidebar()

    # Store in session state
    st.session_state["tickers"] = tickers
    st.session_state["date_range"] = date_range

    # Initialize session state for data
    if "predictions" not in st.session_state:
        st.session_state["predictions"] = {}
    if "sentiment_data" not in st.session_state:
        st.session_state["sentiment_data"] = {}
    if "price_data" not in st.session_state:
        st.session_state["price_data"] = {}
    if "model_results" not in st.session_state:
        st.session_state["model_results"] = {}
    if "backtest_results" not in st.session_state:
        st.session_state["backtest_results"] = {}
    if "briefs" not in st.session_state:
        st.session_state["briefs"] = []

    # Handle demo mode
    if demo_mode:
        with st.spinner("Generating demo data..."):
            (st.session_state["predictions"],
             st.session_state["sentiment_data"],
             st.session_state["price_data"],
             st.session_state["model_results"],
             st.session_state["backtest_results"],
             st.session_state["briefs"]) = generate_demo_data(tickers, date_range)
        st.success("Demo data generated! Explore the tabs below.")

    # Handle run analysis
    if run_analysis and tickers:
        with st.spinner("Running analysis..."):
            try:
                # Try to run actual pipeline
                from src.pipeline import run_full_pipeline

                results = run_full_pipeline(
                    tickers=tickers,
                    use_llm=True,
                    sentiment_model="vader" if "VADER" in sentiment_model else "finbert",
                    days=date_range,
                    run_backtest=True,
                )

                # Parse results into session state
                st.session_state["model_results"] = results.get("model_results", {})
                st.session_state["backtest_results"] = results.get("backtest_results", {})
                st.session_state["briefs"] = results.get("glm_briefs", [])

                st.success(f"Analysis complete! Processed {len(results.get('tickers_processed', []))} tickers.")

            except Exception as e:
                st.warning(f"Pipeline execution failed: {str(e)[:100]}")
                st.info("Using demo data instead. Click 'Demo Mode' for a quick preview.")
                # Fall back to demo data
                (st.session_state["predictions"],
                 st.session_state["sentiment_data"],
                 st.session_state["price_data"],
                 st.session_state["model_results"],
                 st.session_state["backtest_results"],
                 st.session_state["briefs"]) = generate_demo_data(tickers, date_range)

    # Main tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Overview",
        "🔍 Sentiment",
        "📈 Technical",
        "🤖 ML Predictions",
        "📝 AI Briefs"
    ])

    with tab1:
        render_overview_tab(
            tickers,
            st.session_state["predictions"],
            st.session_state["sentiment_data"]
        )

    with tab2:
        render_sentiment_tab(
            tickers,
            st.session_state["sentiment_data"],
            date_range
        )

    with tab3:
        render_technical_tab(
            tickers,
            st.session_state["price_data"],
            date_range
        )

    with tab4:
        render_ml_tab(
            tickers,
            st.session_state["model_results"],
            st.session_state["backtest_results"]
        )

    with tab5:
        render_briefs_tab(st.session_state["briefs"])

    # Footer
    st.markdown("---")
    st.markdown(
        "*Market Sentiment Analyzer v1.0 | Educational Use Only | "
        "Not Financial Advice*"
    )


if __name__ == "__main__":
    main()
