"""
Backtesting Engine Module

Simulates trading strategy performance on historical data.
Provides realistic metrics with transaction costs and benchmark comparison.

Strategy: "Buy when model predicts UP, stay in cash when predicts DOWN"

Key Features:
- Realistic transaction costs (0.1% per trade)
- Maximum drawdown calculation
- Sharpe ratio computation
- Benchmark comparison (vs S&P 500 buy-and-hold)
- Alpha calculation
- Win rate and profit factor metrics

CRITICAL DISCLAIMERS:
- Backtest results are ALWAYS optimistic due to various biases
- Survivorship bias: Only analyzing stocks that still exist
- Lookahead bias risks: Data snooping during model selection
- Past performance does NOT guarantee future results
- This is for EDUCATIONAL PURPOSES ONLY

Evaluation Metrics:
- Total Return: (Final - Initial) / Initial
- Annualized Return: CAGR
- Sharpe Ratio: Risk-adjusted return
- Max Drawdown: Maximum peak-to-trough decline
- Win Rate: Profitable trades / Total trades
- Profit Factor: Sum(profits) / |Sum(losses)|
- Alpha: Strategy return - Benchmark return
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from loguru import logger
from typing import Optional
from dataclasses import dataclass


@dataclass
class BacktestConfig:
    """Configuration for backtesting simulation."""
    initial_capital: float = 10000.0
    transaction_cost: float = 0.001  # 0.1% per trade
    prediction_horizon: int = 3  # J+3 trading
    max_position_pct: float = 1.0  # 100% of capital per trade
    min_confidence: float = 0.5  # Minimum confidence to trade


@dataclass
class TradeResult:
    """Result of a single trade."""
    entry_date: datetime
    exit_date: datetime
    entry_price: float
    exit_price: float
    shares: int
    direction: str  # "UP" or "DOWN"
    gross_return: float
    transaction_cost: float
    net_return: float
    profit: float
    confidence: float
    predicted_direction: str


class BacktestEngine:
    """
    Backtesting engine for trading strategy simulation.
    
    Simulates a simple strategy:
    - When model predicts UP with sufficient confidence: Buy at open
    - Hold for prediction_horizon days
    - Sell at close on exit day
    - Transaction costs applied on both entry and exit
    
    This backtesting is EDUCATIONAL only. Real trading involves:
    - Slippage (execution at worse prices)
    - Market impact (large orders move prices)
    - Short selling constraints
    - Margin requirements
    - Liquidation risk
    
    Example:
        >>> engine = BacktestEngine(config)
        >>> results = engine.run(predictions_df, prices_df)
        >>> print(results['summary'])
    """
    
    def __init__(self, config: Optional[BacktestConfig] = None) -> None:
        """
        Initialize backtest engine.
        
        Args:
            config: BacktestConfig with simulation parameters
        """
        self.config = config or BacktestConfig()
        self.trades: list[TradeResult] = []
        self.equity_curve: list[dict] = []
        
        logger.info(
            f"BacktestEngine initialized: "
            f"${self.config.initial_capital:,.0f} capital, "
            f"{self.config.transaction_cost*100:.1f}% transaction cost"
        )
    
    def run(
        self,
        predictions_df: pd.DataFrame,
        prices_df: pd.DataFrame,
        ticker: Optional[str] = None
    ) -> dict:
        """
        Run backtesting simulation.
        
        Strategy Logic:
        1. For each prediction (date, predicted_direction, probability):
           - If predicted_direction == "UP" and probability >= min_confidence:
             - Buy at open price
             - Hold for prediction_horizon days
             - Sell at close price
             - Apply transaction costs
        
        Args:
            predictions_df: DataFrame with columns:
                - date: Prediction date
                - ticker: Stock symbol (if multi-ticker)
                - predicted_direction: "UP" or "DOWN"
                - probability: Confidence (0-1)
            prices_df: DataFrame with columns:
                - date: Trading date
                - ticker: Stock symbol (if multi-ticker)
                - open: Opening price
                - close: Closing price
            ticker: Optional ticker filter
            
        Returns:
            Dictionary with:
            - summary: Key metrics dict
            - trades: List of TradeResult objects
            - equity_curve: DataFrame of portfolio value over time
            - comparison: Dict comparing to buy-and-hold
        """
        # Reset state
        self.trades = []
        self.equity_curve = []
        
        # Prepare data
        pred_df = predictions_df.copy()
        price_df = prices_df.copy()
        
        # Filter by ticker if specified
        if ticker:
            if "ticker" in pred_df.columns:
                pred_df = pred_df[pred_df["ticker"] == ticker]
            if "ticker" in price_df.columns:
                price_df = price_df[price_df["ticker"] == ticker]
        
        # Ensure date columns are datetime
        pred_df["date"] = pd.to_datetime(pred_df["date"])
        price_df["date"] = pd.to_datetime(price_df["date"])
        
        # Sort by date
        pred_df = pred_df.sort_values("date").reset_index(drop=True)
        price_df = price_df.sort_values("date").reset_index(drop=True)
        
        # Create price lookup
        price_lookup = price_df.set_index("date")[["open", "close"]].to_dict("index")
        dates_available = set(price_df["date"])
        
        # Initialize portfolio
        capital = self.config.initial_capital
        cash = capital
        position = 0  # Number of shares held
        entry_price = 0
        entry_date = None
        
        # Track equity curve
        start_date = pred_df["date"].min()
        end_date = pred_df["date"].max()
        
        logger.info(
            f"Running backtest: {len(pred_df)} predictions, "
            f"date range: {start_date.date()} to {end_date.date()}"
        )
        
        # Process each prediction
        for idx, row in pred_df.iterrows():
            current_date = row["date"]
            predicted_dir = row.get("predicted_direction", row.get("prediction", "DOWN"))
            confidence = row.get("probability", 0.5)
            
            # Get prices for today
            if current_date not in price_lookup:
                continue
            
            today_prices = price_lookup[current_date]
            open_price = today_prices["open"]
            close_price = today_prices["close"]
            
            # If we have a position, check if we should exit
            if position > 0:
                # Exit after holding for prediction_horizon days
                days_held = (current_date - entry_date).days
                if days_held >= self.config.prediction_horizon:
                    # Sell at close
                    exit_price = close_price
                    gross_return = (exit_price - entry_price) / entry_price
                    tx_cost = 2 * self.config.transaction_cost  # Entry + exit
                    net_return = gross_return - tx_cost
                    
                    profit = position * exit_price * (1 - self.config.transaction_cost)
                    cash = profit
                    
                    # Record trade
                    trade = TradeResult(
                        entry_date=entry_date,
                        exit_date=current_date,
                        entry_price=entry_price,
                        exit_price=exit_price,
                        shares=position,
                        direction="UP",
                        gross_return=gross_return,
                        transaction_cost=tx_cost,
                        net_return=net_return,
                        profit=position * (exit_price - entry_price) * (1 - self.config.transaction_cost),
                        confidence=confidence,
                        predicted_direction=predicted_dir
                    )
                    self.trades.append(trade)
                    
                    position = 0
                    entry_price = 0
                    entry_date = None
            
            # If no position, check if we should enter
            if position == 0 and predicted_dir == "UP" and confidence >= self.config.min_confidence:
                # Buy at open
                entry_price = open_price
                shares_to_buy = int(cash * self.config.max_position_pct / entry_price)
                if shares_to_buy > 0:
                    position = shares_to_buy
                    entry_date = current_date
                    cash = cash - position * entry_price * (1 + self.config.transaction_cost)
            
            # Record equity
            portfolio_value = cash + position * close_price
            self.equity_curve.append({
                "date": current_date,
                "cash": cash,
                "position": position,
                "price": close_price,
                "portfolio_value": portfolio_value,
            })
        
        # Close any remaining position
        if position > 0:
            last_date = pred_df["date"].max()
            if last_date in price_lookup:
                exit_price = price_lookup[last_date]["close"]
                cash = position * exit_price * (1 - self.config.transaction_cost)
                position = 0
        
        # Calculate summary metrics
        summary = self._calculate_summary()
        
        # Calculate buy-and-hold comparison
        comparison = self._calculate_benchmark_comparison(price_df)
        
        return {
            "summary": summary,
            "trades": self.trades,
            "equity_curve": pd.DataFrame(self.equity_curve),
            "comparison": comparison,
            "config": {
                "initial_capital": self.config.initial_capital,
                "transaction_cost": self.config.transaction_cost,
                "prediction_horizon": self.config.prediction_horizon,
                "min_confidence": self.config.min_confidence,
            }
        }
    
    def _calculate_summary(self) -> dict:
        """Calculate summary metrics from trades."""
        if not self.trades:
            return {
                "total_trades": 0,
                "message": "No trades executed"
            }
        
        # Extract trade data
        gross_returns = [t.gross_return for t in self.trades]
        net_returns = [t.net_return for t in self.trades]
        profits = [t.profit for t in self.trades]
        
        # Calculate metrics
        total_trades = len(self.trades)
        winning_trades = sum(1 for r in net_returns if r > 0)
        losing_trades = sum(1 for r in net_returns if r <= 0)
        
        # Final portfolio value
        if self.equity_curve:
            final_value = self.equity_curve[-1]["portfolio_value"]
        else:
            final_value = self.config.initial_capital
        
        # Total return
        total_return = (final_value - self.config.initial_capital) / self.config.initial_capital
        
        # Win rate
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        # Profit factor
        total_profits = sum(p for p in profits if p > 0)
        total_losses = abs(sum(p for p in profits if p < 0))
        profit_factor = total_profits / total_losses if total_losses > 0 else float('inf')
        
        # Average win/loss
        avg_win = np.mean([r for r in net_returns if r > 0]) if winning_trades > 0 else 0
        avg_loss = np.mean([r for r in net_returns if r <= 0]) if losing_trades > 0 else 0
        
        # Calculate Sharpe ratio
        if len(net_returns) > 1:
            returns_series = pd.Series(net_returns)
            if returns_series.std() > 0:
                # Annualized Sharpe (assuming ~252 trading days)
                sharpe = returns_series.mean() / returns_series.std() * np.sqrt(252 / self.config.prediction_horizon)
            else:
                sharpe = 0
        else:
            sharpe = 0
        
        # Max drawdown from equity curve
        if self.equity_curve:
            equity_values = [e["portfolio_value"] for e in self.equity_curve]
            peak = equity_values[0]
            max_dd = 0
            for value in equity_values:
                if value > peak:
                    peak = value
                dd = (peak - value) / peak
                if dd > max_dd:
                    max_dd = dd
            max_drawdown = max_dd
        else:
            max_drawdown = 0
        
        return {
            "total_trades": total_trades,
            "winning_trades": winning_trades,
            "losing_trades": losing_trades,
            "win_rate": win_rate,
            "total_return": total_return,
            "final_value": final_value,
            "profit_factor": profit_factor,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "sharpe_ratio": sharpe,
            "max_drawdown": max_drawdown,
            "total_profits": total_profits,
            "total_losses": total_losses,
            "avg_trade_return": np.mean(net_returns),
            "median_trade_return": np.median(net_returns),
        }
    
    def _calculate_benchmark_comparison(self, price_df: pd.DataFrame) -> dict:
        """Calculate comparison with buy-and-hold strategy."""
        if price_df.empty:
            return {}
        
        # Buy-and-hold return
        first_close = price_df["close"].iloc[0]
        last_close = price_df["close"].iloc[-1]
        buy_hold_return = (last_close - first_close) / first_close
        
        # Strategy return
        if self.equity_curve:
            strategy_return = (self.equity_curve[-1]["portfolio_value"] - self.config.initial_capital) / self.config.initial_capital
        else:
            strategy_return = 0
        
        # Alpha
        alpha = strategy_return - buy_hold_return
        
        return {
            "buy_hold_return": buy_hold_return,
            "strategy_return": strategy_return,
            "alpha": alpha,
            "outperformed": alpha > 0,
            "first_date": str(price_df["date"].iloc[0]),
            "last_date": str(price_df["date"].iloc[-1]),
        }
    
    def run_with_confidence_levels(
        self,
        predictions_df: pd.DataFrame,
        prices_df: pd.DataFrame,
        confidence_levels: list[float] = [0.5, 0.6, 0.7, 0.8]
    ) -> pd.DataFrame:
        """
        Run backtests at different confidence thresholds.
        
        Tests how strategy performs when requiring higher confidence.
        Higher confidence → fewer trades → potentially better win rate.
        
        Args:
            predictions_df: Predictions DataFrame
            prices_df: Prices DataFrame
            confidence_levels: List of minimum confidence thresholds
            
        Returns:
            DataFrame comparing performance at each level
        """
        results = []
        
        for min_conf in confidence_levels:
            config = BacktestConfig(
                initial_capital=self.config.initial_capital,
                transaction_cost=self.config.transaction_cost,
                prediction_horizon=self.config.prediction_horizon,
                min_confidence=min_conf
            )
            
            engine = BacktestEngine(config)
            result = engine.run(predictions_df, prices_df)
            
            summary = result["summary"]
            comparison = result["comparison"]
            
            results.append({
                "min_confidence": min_conf,
                "total_trades": summary.get("total_trades", 0),
                "win_rate": summary.get("win_rate", 0),
                "total_return": summary.get("total_return", 0),
                "sharpe_ratio": summary.get("sharpe_ratio", 0),
                "max_drawdown": summary.get("max_drawdown", 0),
                "profit_factor": summary.get("profit_factor", 0),
                "alpha": comparison.get("alpha", 0),
            })
        
        return pd.DataFrame(results)


def run_backtest(
    predictions_df: pd.DataFrame,
    prices_df: pd.DataFrame,
    initial_capital: float = 10000.0,
    transaction_cost: float = 0.001
) -> dict:
    """
    Convenience function to run backtest with default settings.
    
    Args:
        predictions_df: DataFrame with predictions
        prices_df: DataFrame with price data
        initial_capital: Starting capital
        transaction_cost: Cost per trade (fraction)
        
    Returns:
        Dictionary with backtest results
    """
    config = BacktestConfig(
        initial_capital=initial_capital,
        transaction_cost=transaction_cost
    )
    
    engine = BacktestEngine(config)
    return engine.run(predictions_df, prices_df)


def format_backtest_report(results: dict) -> str:
    """
    Format backtest results as a readable report.
    
    Args:
        results: Results from BacktestEngine.run()
        
    Returns:
        Formatted string report
    """
    summary = results["summary"]
    comparison = results.get("comparison", {})
    config = results["config"]
    
    lines = [
        "=" * 60,
        "BACKTESTING REPORT",
        "=" * 60,
        "",
        "CONFIGURATION:",
        f"  Initial Capital: ${config['initial_capital']:,.2f}",
        f"  Transaction Cost: {config['transaction_cost']*100:.1f}%",
        f"  Prediction Horizon: {config['prediction_horizon']} days",
        f"  Min Confidence: {config['min_confidence']:.0%}",
        "",
        "TRADE STATISTICS:",
    ]
    
    if "total_trades" in summary:
        lines.extend([
            f"  Total Trades: {summary['total_trades']}",
            f"  Winning Trades: {summary['winning_trades']}",
            f"  Losing Trades: {summary['losing_trades']}",
            f"  Win Rate: {summary['win_rate']:.1%}",
            "",
            "RETURNS:",
            f"  Total Return: {summary['total_return']:+.1%}",
            f"  Final Value: ${summary['final_value']:,.2f}",
            f"  Avg Trade Return: {summary.get('avg_trade_return', 0):+.2%}",
            f"  Median Trade Return: {summary.get('median_trade_return', 0):+.2%}",
            "",
            "RISK METRICS:",
            f"  Sharpe Ratio: {summary['sharpe_ratio']:.2f}",
            f"  Max Drawdown: {summary['max_drawdown']:.1%}",
            f"  Profit Factor: {summary['profit_factor']:.2f}",
            "",
            "BENCHMARK COMPARISON:",
            f"  Buy & Hold Return: {comparison.get('buy_hold_return', 0):+.1%}",
            f"  Strategy Return: {comparison.get('strategy_return', 0):+.1%}",
            f"  Alpha: {comparison.get('alpha', 0):+.1%}",
            f"  Outperformed: {'Yes' if comparison.get('outperformed', False) else 'No'}",
        ])
    else:
        lines.append(f"  {summary.get('message', 'No trades executed')}")
    
    lines.extend([
        "",
        "=" * 60,
        "DISCLAIMER: This backtest is for EDUCATIONAL PURPOSES ONLY.",
        "Past performance does NOT guarantee future results.",
        "Backtest results are always optimistic due to various biases.",
        "=" * 60,
    ])
    
    return "\n".join(lines)


# ============================================================
# Module Test
# ============================================================

if __name__ == "__main__":
    from datetime import datetime, timedelta
    import numpy as np
    
    print("Backtesting Engine Test")
    print("=" * 60)
    
    # Create synthetic data
    np.random.seed(42)
    dates = [datetime.now() - timedelta(days=i) for i in range(100, 0, -1)]
    
    # Price data
    prices = 150 * np.cumprod(1 + np.random.normal(0.001, 0.02, 100))
    price_df = pd.DataFrame({
        "date": dates,
        "ticker": "TEST",
        "open": prices * (1 + np.random.uniform(-0.005, 0.005, 100)),
        "close": prices,
        "high": prices * 1.01,
        "low": prices * 0.99,
        "volume": np.random.randint(40000000, 60000000, 100),
    })
    
    # Prediction data (with some predictive signal)
    predictions = []
    for i, date in enumerate(dates[:-3]):  # Can't predict last 3 days
        # Random prediction with slight bias toward correct direction
        actual_direction = 1 if prices[min(i+3, len(prices)-1)] > prices[i] else 0
        # Model has ~60% accuracy
        if np.random.random() < 0.6:
            pred_dir = "UP" if actual_direction == 1 else "DOWN"
        else:
            pred_dir = "DOWN" if actual_direction == 1 else "UP"
        
        predictions.append({
            "date": date,
            "ticker": "TEST",
            "predicted_direction": pred_dir,
            "probability": np.random.uniform(0.5, 0.9),
        })
    
    pred_df = pd.DataFrame(predictions)
    
    print(f"\n1. Created test data:")
    print(f"   Prices: {len(price_df)} days")
    print(f"   Predictions: {len(pred_df)} predictions")
    
    # Run backtest
    print("\n2. Running backtest...")
    config = BacktestConfig(
        initial_capital=10000,
        transaction_cost=0.001,
        prediction_horizon=3,
        min_confidence=0.5
    )
    
    engine = BacktestEngine(config)
    results = engine.run(pred_df, price_df)
    
    # Print report
    print("\n" + format_backtest_report(results))
    
    # Test confidence levels
    print("\n3. Testing different confidence thresholds...")
    confidence_results = engine.run_with_confidence_levels(
        pred_df, price_df, 
        confidence_levels=[0.5, 0.6, 0.7]
    )
    print(confidence_results.to_string(index=False))
    
    # Show equity curve sample
    if not results["equity_curve"].empty:
        print("\n4. Equity curve sample (first 5 days):")
        print(results["equity_curve"].head().to_string())
    
    print("\n" + "=" * 60)
    print("✅ Backtesting engine test completed!")
