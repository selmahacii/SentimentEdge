"""
Market Sentiment Analyzer - Source Package

This package contains all core modules for the Market Sentiment Analyzer:
- collectors: Data collection from Reddit, Yahoo Finance, Google Trends
- sentiment: NLP sentiment analysis (VADER and FinBERT)
- features: Technical indicators and feature engineering
- storage: Database models and operations
- model: XGBoost ML model for prediction
- backtest: Backtesting engine
- insights: GLM-powered daily briefs
- pipeline: Full orchestration
"""

from src.config import settings

__version__ = "1.0.0"
__author__ = "Data Analyst Portfolio"
__all__ = ["settings"]
