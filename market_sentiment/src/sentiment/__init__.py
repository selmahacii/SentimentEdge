"""
Sentiment Analysis Package

This package provides sentiment analysis capabilities:

VADERAnalyzer:
- Fast, rule-based sentiment analysis
- Custom financial lexicon (bullish, bearish, moon, etc.)
- ~10,000 texts/second on CPU
- Good for real-time analysis and streaming

FinBERTAnalyzer:
- Transformer-based financial sentiment model
- More accurate for financial jargon
- ~100 texts/second on CPU
- Requires ~500MB download on first use
- Best for batch processing

Both analyzers:
- Weight sentiment by post upvotes (viral posts = more impact)
- Aggregate to daily sentiment scores
- Calculate sentiment momentum

Usage:
    # VADER (fast, real-time)
    from src.sentiment import VADERAnalyzer
    analyzer = VADERAnalyzer()
    result = analyzer.analyze_text("AAPL to the moon! 🚀")
    
    # FinBERT (accurate, batch)
    from src.sentiment import FinBERTAnalyzer
    analyzer = FinBERTAnalyzer()
    results = analyzer.analyze_batch(texts)

Model Selection Guide:
- Use VADER for: real-time dashboards, streaming data, quick analysis
- Use FinBERT for: research papers, model training, final analysis
- Use both for: comparison, validation, edge case detection
"""

from src.sentiment.vader_analyzer import VADERAnalyzer, FINANCIAL_LEXICON
from src.sentiment.finbert_analyzer import FinBERTAnalyzer

__all__ = [
    "VADERAnalyzer",
    "FinBERTAnalyzer",
    "FINANCIAL_LEXICON",
]
