"""
VADER Sentiment Analyzer Module

Fast, rule-based sentiment analysis optimized for social media text.
Extended with custom financial lexicon for stock market context.

VADER (Valence Aware Dictionary and sEntiment Reasoner) is specifically
designed for social media text and handles:
- Emojis and emoticons 🚀📈
- Capitalization (intensifiers) - "HUGE gains"
- Punctuation (!!!, ???)
- Negations (not good, wasn't bad)
- Degree modifiers (very, extremely, somewhat)

Features:
- Custom financial lexicon (bullish, bearish, moon, etc.)
- Upvote-weighted sentiment scoring
- Sentiment momentum calculation
- 1000x faster than transformer models

DISCLAIMER: VADER is rule-based and may miss sarcasm, irony, and 
complex financial jargon. Use FinBERT for higher accuracy.
"""

import math
import re
from datetime import datetime
from typing import Optional

import pandas as pd
from loguru import logger

# VADER sentiment analyzer
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    VADER_AVAILABLE = True
except ImportError:
    VADER_AVAILABLE = False
    logger.warning(
        "vaderSentiment not installed. Install with: pip install vaderSentiment"
    )


# ============================================================
# Custom Financial Lexicon
# ============================================================

# Financial terms and their sentiment scores
# Scale: -4 (most negative) to +4 (most positive)
# These are added to VADER's default lexicon
FINANCIAL_LEXICON = {
    # Strongly bullish terms
    "bullish": 3.0,
    "moon": 2.5,
    "mooning": 2.5,
    "to the moon": 3.0,
    "rocket": 2.0,
    "🚀": 2.5,  # Rocket emoji
    "moonshot": 2.5,
    "undervalued": 1.5,
    "oversold": 1.5,
    "breakout": 1.5,
    "rally": 1.5,
    "golden cross": 2.0,
    "all time high": 2.0,
    "ath": 2.0,
    "blue sky": 1.5,
    
    # Moderately bullish
    "calls": 1.0,
    "long": 0.5,
    "accumulate": 1.0,
    "accumulate": 1.0,
    "buy the dip": 1.5,
    "btd": 1.5,
    "discount": 0.5,
    "support": 0.5,
    "upgrade": 1.5,
    "outperform": 1.5,
    "overweight": 1.0,
    "strong buy": 2.0,
    "buy": 0.8,
    "hodl": 1.0,
    "diamond hands": 1.5,
    "💎🙌": 2.0,  # Diamond hands emoji
    "tendies": 1.0,
    "gains": 1.5,
    "profit": 1.0,
    "profit taking": 0.5,
    
    # Strongly bearish terms
    "bearish": -3.0,
    "crash": -2.5,
    "dump": -2.0,
    "dumping": -2.0,
    "pump and dump": -3.0,
    "rug pull": -3.5,
    "scam": -3.5,
    "bubble": -2.0,
    "overvalued": -1.5,
    "overbought": -1.5,
    "dead cat bounce": -2.0,
    "bagholder": -2.0,
    "bagholding": -2.0,
    
    # Moderately bearish
    "puts": -1.0,
    "short": -1.0,
    "shorting": -1.0,
    "short squeeze": 0.5,  # Can be positive for longs
    "sell": -0.8,
    "sell off": -1.5,
    "downgrade": -1.5,
    "underperform": -1.5,
    "underweight": -1.0,
    "strong sell": -2.0,
    "resistance": -0.5,
    "paper hands": -1.0,
    "losses": -1.5,
    "loss": -1.0,
    
    # Extreme/crash terms
    "rekt": -3.0,
    "wrecked": -2.5,
    "liquidated": -2.5,
    "margin call": -2.5,
    "bankruptcy": -3.5,
    "insolvency": -3.5,
    
    # Speculative/gambling terms
    "yolo": 0.5,  # Can be positive (excitement) or negative (recklessness)
    "degen": 0.0,  # Neutral - depends on context
    "gambling": -0.5,
    "lottery": -0.5,
    "bet": 0.0,
    
    # Market terms
    "volatility": 0.0,  # Neutral
    "uncertainty": -0.5,
    "fed": 0.0,
    "interest rate": 0.0,
    "inflation": -0.5,
    "recession": -2.0,
    "depression": -3.0,
    "correction": -1.0,
    "consolidation": 0.0,
    "sideways": 0.0,
    "choppy": -0.5,
}


class VADERAnalyzer:
    """
    VADER-based sentiment analyzer with financial lexicon extensions.
    
    This class provides fast sentiment analysis for stock market social
    media text. It extends VADER's lexicon with financial terms commonly
    used on Reddit (wallstreetbets, investing, etc.).
    
    Attributes:
        analyzer: SentimentIntensityAnalyzer instance
        financial_lexicon: Dictionary of financial term scores
        
    Example:
        >>> analyzer = VADERAnalyzer()
        >>> result = analyzer.analyze_text("AAPL to the moon! 🚀")
        >>> print(result)
        {'compound': 0.75, 'label': 'positive', ...}
    """
    
    # Sentiment thresholds
    POSITIVE_THRESHOLD = 0.05
    NEGATIVE_THRESHOLD = -0.05
    
    def __init__(self, custom_lexicon: Optional[dict] = None) -> None:
        """
        Initialize VADER analyzer with financial lexicon.
        
        Args:
            custom_lexicon: Optional dictionary to extend/override default lexicon
        """
        if not VADER_AVAILABLE:
            raise ImportError(
                "vaderSentiment is not installed. "
                "Install with: pip install vaderSentiment"
            )
        
        # Initialize VADER analyzer
        self.analyzer = SentimentIntensityAnalyzer()
        
        # Store financial lexicon
        self.financial_lexicon = FINANCIAL_LEXICON.copy()
        
        # Extend with custom lexicon if provided
        if custom_lexicon:
            self.financial_lexicon.update(custom_lexicon)
        
        # Update VADER's lexicon with financial terms
        # This merges our terms into VADER's existing lexicon
        self.analyzer.lexicon.update(self.financial_lexicon)
        
        logger.info(
            f"VADERAnalyzer initialized with {len(self.financial_lexicon)} "
            f"financial terms"
        )
    
    def _clean_text(self, text: str) -> str:
        """
        Clean text for sentiment analysis.
        
        Operations:
        - Remove URLs
        - Convert $TICKER to TICKER (VADER handles better)
        - Remove excessive whitespace
        - Keep emojis (VADER understands them)
        
        Args:
            text: Raw text string
            
        Returns:
            Cleaned text string
        """
        if not text:
            return ""
        
        # Remove URLs
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        
        # Remove Reddit-style links
        text = re.sub(r'/r/\w+', '', text)
        text = re.sub(r'/u/\w+', '', text)
        
        # Convert $TICKER to TICKER (remove $ sign but keep ticker)
        # VADER may interpret $ as currency (neutral/negative)
        text = re.sub(r'\$([A-Z]{1,5})\b', r'\1', text)
        
        # Remove excessive whitespace
        text = ' '.join(text.split())
        
        return text.strip()
    
    def analyze_text(self, text: str) -> dict:
        """
        Analyze sentiment of a single text.
        
        Returns compound score (-1 to +1) and individual components.
        
        Args:
            text: Text string to analyze
            
        Returns:
            Dictionary with:
            - compound: Normalized compound score (-1 to +1)
            - positive: Positive component (0 to 1)
            - negative: Negative component (0 to 1)
            - neutral: Neutral component (0 to 1)
            - label: "positive", "negative", or "neutral"
        """
        if not text or not isinstance(text, str):
            return {
                "compound": 0.0,
                "positive": 0.0,
                "negative": 0.0,
                "neutral": 1.0,
                "label": "neutral"
            }
        
        # Clean text
        cleaned_text = self._clean_text(text)
        
        # Get VADER scores
        scores = self.analyzer.polarity_scores(cleaned_text)
        
        # Determine label based on compound score
        compound = scores["compound"]
        if compound >= self.POSITIVE_THRESHOLD:
            label = "positive"
        elif compound <= self.NEGATIVE_THRESHOLD:
            label = "negative"
        else:
            label = "neutral"
        
        return {
            "compound": compound,
            "positive": scores["pos"],
            "negative": scores["neg"],
            "neutral": scores["neu"],
            "label": label
        }
    
    def analyze_batch(self, texts: list[str]) -> list[dict]:
        """
        Analyze sentiment for multiple texts.
        
        Args:
            texts: List of text strings
            
        Returns:
            List of sentiment dictionaries
        """
        return [self.analyze_text(text) for text in texts]
    
    def analyze_posts(
        self,
        posts: list[dict],
        text_key: str = "combined_text",
        score_key: str = "score"
    ) -> pd.DataFrame:
        """
        Analyze sentiment for Reddit posts.
        
        Applies upvote weighting: viral posts (high upvotes) have
        more market impact, so their sentiment is weighted higher.
        
        Weighting formula:
            weighted_compound = compound * log(max(score, 1) + 1)
        
        Log transformation prevents viral posts from dominating
        while still giving them more weight.
        
        Args:
            posts: List of post dictionaries
            text_key: Key for text field in post dict
            score_key: Key for upvote score field
            
        Returns:
            DataFrame with sentiment columns added:
            - vader_compound, vader_positive, vader_negative, vader_neutral
            - vader_label
            - weighted_compound
        """
        if not posts:
            return pd.DataFrame()
        
        results = []
        
        for post in posts:
            # Get text to analyze
            text = post.get(text_key, "")
            if not text:
                # Fallback to title if combined_text is empty
                text = post.get("title", "")
            
            # Analyze sentiment
            sentiment = self.analyze_text(text)
            
            # Get upvote score for weighting
            score = post.get(score_key, 0)
            
            # Calculate weighted compound
            # log(score + 1) scales the weight appropriately
            # score of 0 -> weight of 0
            # score of 100 -> weight of 2.0
            # score of 1000 -> weight of 3.0
            # score of 10000 -> weight of 4.0
            weight = math.log(max(score, 1) + 1)
            weighted_compound = sentiment["compound"] * weight
            
            # Combine post data with sentiment
            result = {
                **post,  # Include all original fields
                "vader_compound": sentiment["compound"],
                "vader_positive": sentiment["positive"],
                "vader_negative": sentiment["negative"],
                "vader_neutral": sentiment["neutral"],
                "vader_label": sentiment["label"],
                "weight": weight,
                "weighted_compound": weighted_compound,
            }
            results.append(result)
        
        df = pd.DataFrame(results)
        
        logger.info(
            f"Analyzed {len(df)} posts: "
            f"mean compound = {df['vader_compound'].mean():.3f}, "
            f"positive = {(df['vader_label'] == 'positive').sum()}, "
            f"negative = {(df['vader_label'] == 'negative').sum()}, "
            f"neutral = {(df['vader_label'] == 'neutral').sum()}"
        )
        
        return df
    
    def get_daily_sentiment(
        self,
        posts_df: pd.DataFrame,
        date_key: str = "created_at"
    ) -> pd.DataFrame:
        """
        Aggregate post sentiment to daily level.
        
        Groups posts by date and calculates:
        - Mean sentiment scores
        - Weighted mean sentiment
        - Post count
        - Sentiment momentum (change from previous day)
        
        Args:
            posts_df: DataFrame from analyze_posts()
            date_key: Key for date/timestamp field
            
        Returns:
            DataFrame with daily aggregated sentiment:
            - date, ticker
            - vader_compound (mean)
            - weighted_compound (mean)
            - post_count
            - sentiment_momentum
            - dominant_sentiment
        """
        if posts_df.empty:
            return pd.DataFrame()
        
        df = posts_df.copy()
        
        # Parse dates
        if date_key in df.columns:
            df["date"] = pd.to_datetime(df[date_key]).dt.date
            df["date"] = pd.to_datetime(df["date"])
        
        # Check required columns
        if "vader_compound" not in df.columns:
            logger.error("DataFrame missing vader_compound column. Run analyze_posts first.")
            return pd.DataFrame()
        
        # Group by date (and ticker if present)
        group_cols = ["date"]
        if "ticker" in df.columns:
            group_cols = ["date", "ticker"]
        
        daily = df.groupby(group_cols).agg(
            vader_compound=("vader_compound", "mean"),
            vader_positive=("vader_positive", "mean"),
            vader_negative=("vader_negative", "mean"),
            vader_neutral=("vader_neutral", "mean"),
            weighted_compound=("weighted_compound", "mean"),
            post_count=("vader_compound", "count"),
            avg_score=("score", "mean") if "score" in df.columns else ("vader_compound", "count"),
            total_comments=("num_comments", "sum") if "num_comments" in df.columns else ("vader_compound", "count"),
        ).reset_index()
        
        # Calculate sentiment momentum (change from previous day)
        if "ticker" in daily.columns:
            daily = daily.sort_values(["ticker", "date"])
            daily["sentiment_momentum"] = daily.groupby("ticker")["vader_compound"].diff()
        else:
            daily = daily.sort_values("date")
            daily["sentiment_momentum"] = daily["vader_compound"].diff()
        
        # Determine dominant sentiment for each day
        def get_dominant_label(row):
            if row["vader_compound"] >= self.POSITIVE_THRESHOLD:
                return "positive"
            elif row["vader_compound"] <= self.NEGATIVE_THRESHOLD:
                return "negative"
            else:
                return "neutral"
        
        daily["dominant_sentiment"] = daily.apply(get_dominant_label, axis=1)
        
        # Round numeric columns
        numeric_cols = ["vader_compound", "vader_positive", "vader_negative", 
                       "vader_neutral", "weighted_compound", "sentiment_momentum"]
        for col in numeric_cols:
            if col in daily.columns:
                daily[col] = daily[col].round(4)
        
        logger.info(
            f"Aggregated to {len(daily)} daily sentiment records"
        )
        
        return daily
    
    def get_sentiment_distribution(self, posts_df: pd.DataFrame) -> dict:
        """
        Get distribution statistics for sentiment.
        
        Args:
            posts_df: DataFrame from analyze_posts()
            
        Returns:
            Dictionary with distribution statistics
        """
        if posts_df.empty or "vader_compound" not in posts_df.columns:
            return {}
        
        return {
            "total_posts": len(posts_df),
            "mean_compound": posts_df["vader_compound"].mean(),
            "std_compound": posts_df["vader_compound"].std(),
            "median_compound": posts_df["vader_compound"].median(),
            "positive_count": (posts_df["vader_label"] == "positive").sum(),
            "negative_count": (posts_df["vader_label"] == "negative").sum(),
            "neutral_count": (posts_df["vader_label"] == "neutral").sum(),
            "positive_pct": (posts_df["vader_label"] == "positive").mean() * 100,
            "negative_pct": (posts_df["vader_label"] == "negative").mean() * 100,
            "neutral_pct": (posts_df["vader_label"] == "neutral").mean() * 100,
        }
    
    def explain_score(self, text: str) -> dict:
        """
        Explain which words contributed to the sentiment score.
        
        Useful for debugging and understanding VADER's scoring.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with score breakdown
        """
        cleaned_text = self._clean_text(text)
        words = cleaned_text.lower().split()
        
        # Find financial terms in text
        found_terms = {}
        for word in words:
            if word in self.financial_lexicon:
                found_terms[word] = self.financial_lexicon[word]
        
        # Get overall sentiment
        sentiment = self.analyze_text(text)
        
        return {
            "text": text[:100] + "..." if len(text) > 100 else text,
            "sentiment": sentiment,
            "financial_terms_found": found_terms,
            "total_financial_terms": len(found_terms),
        }


# ============================================================
# Convenience Functions
# ============================================================

def analyze_sentiment(text: str) -> dict:
    """
    Convenience function to analyze single text.
    
    Args:
        text: Text to analyze
        
    Returns:
        Sentiment dictionary
    """
    analyzer = VADERAnalyzer()
    return analyzer.analyze_text(text)


def analyze_posts_sentiment(posts: list[dict]) -> pd.DataFrame:
    """
    Convenience function to analyze multiple posts.
    
    Args:
        posts: List of post dictionaries
        
    Returns:
        DataFrame with sentiment analysis
    """
    analyzer = VADERAnalyzer()
    return analyzer.analyze_posts(posts)


# ============================================================
# Module Test
# ============================================================

if __name__ == "__main__":
    print("VADER Sentiment Analyzer Test")
    print("=" * 60)
    
    # Check if VADER is available
    if not VADER_AVAILABLE:
        print("❌ vaderSentiment not installed")
        print("   Install with: pip install vaderSentiment")
        exit(1)
    
    # Initialize analyzer
    analyzer = VADERAnalyzer()
    
    # Test single texts
    test_texts = [
        "AAPL to the moon! 🚀🚀🚀",
        "TSLA is going to crash, sell now!",
        "MSFT looks fairly valued at current levels",
        "NVDA earnings beat expectations, bullish!",
        "This stock is a scam, total pump and dump",
        "Buy the dip on AMD, great entry point",
        "Portfolio is down 50%, I'm rekt",
        "Diamond hands baby! 💎🙌",
    ]
    
    print("\n1. Single Text Analysis:")
    print("-" * 60)
    for text in test_texts:
        result = analyzer.analyze_text(text)
        print(f"\nText: {text[:50]}...")
        print(f"  Compound: {result['compound']:.3f}")
        print(f"  Label: {result['label']}")
    
    # Test batch analysis
    print("\n" + "=" * 60)
    print("2. Batch Analysis (Reddit-style posts):")
    print("-" * 60)
    
    test_posts = [
        {
            "post_id": "1",
            "ticker": "AAPL",
            "title": "Apple is undervalued!",
            "combined_text": "Apple is undervalued! Great entry point, buy now! 🚀",
            "score": 500,
            "num_comments": 45,
            "created_at": "2024-01-15T10:30:00",
        },
        {
            "post_id": "2",
            "ticker": "AAPL",
            "title": "AAPL bearish outlook",
            "combined_text": "I think AAPL is overvalued and will crash soon",
            "score": 50,
            "num_comments": 12,
            "created_at": "2024-01-15T14:00:00",
        },
        {
            "post_id": "3",
            "ticker": "TSLA",
            "title": "Tesla to the moon!",
            "combined_text": "TSLA going to $1000! Diamond hands! 💎🙌🚀",
            "score": 2000,
            "num_comments": 150,
            "created_at": "2024-01-15T09:00:00",
        },
    ]
    
    df = analyzer.analyze_posts(test_posts)
    print(f"\nAnalyzed {len(df)} posts")
    print("\nResults:")
    print(df[["ticker", "title", "vader_compound", "vader_label", "weighted_compound"]])
    
    # Test daily aggregation
    print("\n" + "=" * 60)
    print("3. Daily Sentiment Aggregation:")
    print("-" * 60)
    
    daily = analyzer.get_daily_sentiment(df)
    print(daily[["date", "ticker", "vader_compound", "post_count", "dominant_sentiment"]])
    
    # Test sentiment distribution
    print("\n" + "=" * 60)
    print("4. Sentiment Distribution:")
    print("-" * 60)
    
    dist = analyzer.get_sentiment_distribution(df)
    for key, value in dist.items():
        print(f"  {key}: {value}")
    
    # Test score explanation
    print("\n" + "=" * 60)
    print("5. Score Explanation:")
    print("-" * 60)
    
    explanation = analyzer.explain_score("TSLA is bullish! Diamond hands! 🚀📈")
    print(f"\nText: {explanation['text']}")
    print(f"Sentiment: {explanation['sentiment']['label']} ({explanation['sentiment']['compound']:.3f})")
    print(f"Financial terms found: {explanation['financial_terms_found']}")
    
    print("\n" + "=" * 60)
    print("✅ All VADER tests completed!")
