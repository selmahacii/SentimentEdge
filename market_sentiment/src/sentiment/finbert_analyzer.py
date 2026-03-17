"""
FinBERT Sentiment Analyzer Module

Transformer-based sentiment analysis model fine-tuned on financial text.
More accurate than VADER for financial jargon but significantly slower.

FinBERT is a BERT model trained on financial communications:
- Earnings call transcripts
- Financial news articles
- Analyst reports
- SEC filings

It outperforms VADER on:
- Financial jargon ("guidance", "headwinds", "tailwinds")
- Context-dependent meanings
- Complex sentence structures
- Formal vs informal financial language

Trade-off: ~100x slower than VADER, but more accurate.

Features:
- Pre-trained model: ProsusAI/finbert
- Batch processing for efficiency
- GPU acceleration when available
- Agreement analysis with VADER

Model Download: ~500MB on first use
Cached locally after first download.

DISCLAIMER: FinBERT requires significant compute resources.
Use VADER for real-time analysis, FinBERT for batch processing.
"""

import math
import re
from datetime import datetime
from typing import Optional

import pandas as pd
from loguru import logger

# Check for transformers availability
TRANSFORMERS_AVAILABLE = False
TORCH_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    pass

try:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    logger.warning(
        "transformers not installed. Install with: pip install transformers torch"
    )


class FinBERTAnalyzer:
    """
    FinBERT-based sentiment analyzer for financial text.
    
    Uses the ProsusAI/finbert model from HuggingFace, which is
    specifically trained on financial text and outperforms general
    sentiment models on financial documents.
    
    Model Details:
    - Architecture: BERT-base (12 layers, 110M parameters)
    - Training: Financial PhraseBank, FIQA, SFiad
    - Labels: positive, negative, neutral
    - Download size: ~500MB
    
    Attributes:
        model: FinBERT model
        tokenizer: BERT tokenizer
        device: 'cuda' or 'cpu'
        classifier: HuggingFace pipeline
    
    Example:
        >>> analyzer = FinBERTAnalyzer()
        >>> result = analyzer.analyze_text("Revenue exceeded expectations")
        >>> print(result)
        {'label': 'positive', 'score': 0.92}
    """
    
    # Model name on HuggingFace Hub
    MODEL_NAME = "ProsusAI/finbert"
    
    # Maximum sequence length for BERT
    MAX_LENGTH = 512
    
    # Label mapping to numeric scores
    LABEL_SCORES = {
        "positive": 1.0,
        "negative": -1.0,
        "neutral": 0.0,
    }
    
    def __init__(
        self,
        device: Optional[str] = None,
        batch_size: int = 16,
        use_fast_tokenizer: bool = True
    ) -> None:
        """
        Initialize FinBERT analyzer.
        
        Args:
            device: 'cuda', 'cpu', or None (auto-detect)
            batch_size: Number of texts to process at once
            use_fast_tokenizer: Use fast Rust-based tokenizer
            
        Note:
            First initialization downloads ~500MB model weights.
            Subsequent loads use cached model.
        """
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "transformers is not installed. "
                "Install with: pip install transformers torch"
            )
        
        # Determine device
        if device is None:
            if TORCH_AVAILABLE and torch.cuda.is_available():
                self.device = "cuda"
                logger.info("GPU detected, using CUDA")
            else:
                self.device = "cpu"
                logger.info("No GPU detected, using CPU")
        else:
            self.device = device
        
        self.batch_size = batch_size
        
        logger.warning(
            "FinBERT requires ~500MB download on first use. "
            "Model will be cached for future use."
        )
        
        # Load model and tokenizer
        logger.info(f"Loading FinBERT model: {self.MODEL_NAME}")
        
        try:
            # Use pipeline for simplicity
            self.classifier = pipeline(
                "sentiment-analysis",
                model=self.MODEL_NAME,
                tokenizer=self.MODEL_NAME,
                device=0 if self.device == "cuda" else -1,
                top_k=None,  # Return all labels with scores
            )
            
            logger.info("FinBERT model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load FinBERT model: {e}")
            raise
        
        # Store model info
        self.model_loaded = True
    
    def _clean_text(self, text: str) -> str:
        """
        Clean text for FinBERT analysis.
        
        Unlike VADER, FinBERT:
        - Doesn't need emoji handling (trained on formal text)
        - Benefits from keeping sentence structure
        - Has max length of 512 tokens
        
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
        
        # Remove excessive whitespace
        text = ' '.join(text.split())
        
        return text.strip()
    
    def _truncate_text(self, text: str, max_length: int = 512) -> str:
        """
        Truncate text to maximum token length.
        
        BERT models have a maximum of 512 tokens.
        We approximate: 1 token ≈ 4 characters.
        
        Args:
            text: Text to truncate
            max_length: Maximum number of tokens
            
        Returns:
            Truncated text
        """
        # Approximate characters (conservative)
        max_chars = max_length * 4
        
        if len(text) <= max_chars:
            return text
        
        # Truncate and add ellipsis
        return text[:max_chars - 3] + "..."
    
    def analyze_text(self, text: str) -> dict:
        """
        Analyze sentiment of a single text.
        
        Args:
            text: Text string to analyze
            
        Returns:
            Dictionary with:
            - label: "positive", "negative", or "neutral"
            - score: Confidence score (0 to 1)
            - compound: Numeric score (-1 to +1)
            - positive: Probability of positive
            - negative: Probability of negative
            - neutral: Probability of neutral
        """
        if not text or not isinstance(text, str):
            return {
                "label": "neutral",
                "score": 1.0,
                "compound": 0.0,
                "positive": 0.0,
                "negative": 0.0,
                "neutral": 1.0,
            }
        
        # Clean and truncate
        cleaned_text = self._clean_text(text)
        truncated_text = self._truncate_text(cleaned_text, self.MAX_LENGTH)
        
        if not truncated_text:
            return {
                "label": "neutral",
                "score": 1.0,
                "compound": 0.0,
                "positive": 0.0,
                "negative": 0.0,
                "neutral": 1.0,
            }
        
        # Get predictions
        try:
            results = self.classifier(truncated_text)
            
            # results is a list of dicts: [{"label": "positive", "score": 0.8}, ...]
            # Convert to format we want
            scores = {r["label"].lower(): r["score"] for r in results}
            
            # Get dominant label
            label = max(scores.keys(), key=lambda k: scores[k])
            score = scores[label]
            
            # Calculate compound score
            # Weighted average of positive and negative probabilities
            compound = (
                scores.get("positive", 0.0) - 
                scores.get("negative", 0.0)
            )
            
            return {
                "label": label,
                "score": score,
                "compound": compound,
                "positive": scores.get("positive", 0.0),
                "negative": scores.get("negative", 0.0),
                "neutral": scores.get("neutral", 0.0),
            }
            
        except Exception as e:
            logger.warning(f"Error analyzing text: {e}")
            return {
                "label": "neutral",
                "score": 1.0,
                "compound": 0.0,
                "positive": 0.0,
                "negative": 0.0,
                "neutral": 1.0,
            }
    
    def analyze_batch(
        self,
        texts: list[str],
        show_progress: bool = True
    ) -> list[dict]:
        """
        Analyze sentiment for multiple texts efficiently.
        
        Uses batch processing for better GPU utilization.
        Progress logging shows batch number.
        
        Args:
            texts: List of text strings
            show_progress: Whether to log progress
            
        Returns:
            List of sentiment dictionaries
        """
        if not texts:
            return []
        
        results = []
        total_batches = math.ceil(len(texts) / self.batch_size)
        
        logger.info(f"Processing {len(texts)} texts in {total_batches} batches")
        
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            batch_num = i // self.batch_size + 1
            
            if show_progress:
                logger.debug(f"Processing batch {batch_num}/{total_batches}")
            
            # Clean and truncate texts
            batch_cleaned = [
                self._truncate_text(self._clean_text(text), self.MAX_LENGTH)
                for text in batch
            ]
            
            # Filter empty texts
            valid_texts = [t for t in batch_cleaned if t]
            
            if valid_texts:
                try:
                    # Batch prediction
                    batch_results = self.classifier(valid_texts)
                    
                    # Process results
                    result_idx = 0
                    for j, original_text in enumerate(batch):
                        if batch_cleaned[j]:
                            # Get scores for this text
                            if isinstance(batch_results[result_idx], list):
                                scores = {
                                    r["label"].lower(): r["score"] 
                                    for r in batch_results[result_idx]
                                }
                            else:
                                # Single result
                                r = batch_results[result_idx]
                                scores = {r["label"].lower(): r["score"]}
                            
                            label = max(scores.keys(), key=lambda k: scores[k])
                            score = scores[label]
                            
                            compound = (
                                scores.get("positive", 0.0) - 
                                scores.get("negative", 0.0)
                            )
                            
                            results.append({
                                "label": label,
                                "score": score,
                                "compound": compound,
                                "positive": scores.get("positive", 0.0),
                                "negative": scores.get("negative", 0.0),
                                "neutral": scores.get("neutral", 0.0),
                            })
                            result_idx += 1
                        else:
                            results.append({
                                "label": "neutral",
                                "score": 1.0,
                                "compound": 0.0,
                                "positive": 0.0,
                                "negative": 0.0,
                                "neutral": 1.0,
                            })
                            
                except Exception as e:
                    logger.warning(f"Error in batch {batch_num}: {e}")
                    # Add neutral results for failed batch
                    for _ in batch:
                        results.append({
                            "label": "neutral",
                            "score": 1.0,
                            "compound": 0.0,
                            "positive": 0.0,
                            "negative": 0.0,
                            "neutral": 1.0,
                        })
            else:
                # All texts in batch were empty
                for _ in batch:
                    results.append({
                        "label": "neutral",
                        "score": 1.0,
                        "compound": 0.0,
                        "positive": 0.0,
                        "negative": 0.0,
                        "neutral": 1.0,
                    })
        
        logger.info(
            f"Processed {len(results)} texts: "
            f"positive = {sum(1 for r in results if r['label'] == 'positive')}, "
            f"negative = {sum(1 for r in results if r['label'] == 'negative')}, "
            f"neutral = {sum(1 for r in results if r['label'] == 'neutral')}"
        )
        
        return results
    
    def analyze_posts(
        self,
        posts: list[dict],
        text_key: str = "combined_text",
        score_key: str = "score"
    ) -> pd.DataFrame:
        """
        Analyze sentiment for Reddit posts using FinBERT.
        
        Same interface as VADERAnalyzer for easy comparison.
        Applies upvote weighting similarly.
        
        Args:
            posts: List of post dictionaries
            text_key: Key for text field in post dict
            score_key: Key for upvote score field
            
        Returns:
            DataFrame with sentiment columns added
        """
        if not posts:
            return pd.DataFrame()
        
        # Extract texts for batch processing
        texts = []
        for post in posts:
            text = post.get(text_key, "") or post.get("title", "")
            texts.append(text)
        
        # Batch analyze
        sentiments = self.analyze_batch(texts, show_progress=True)
        
        # Combine with post data
        results = []
        for post, sentiment in zip(posts, sentiments):
            score = post.get(score_key, 0)
            weight = math.log(max(score, 1) + 1)
            weighted_compound = sentiment["compound"] * weight
            
            result = {
                **post,
                "finbert_label": sentiment["label"],
                "finbert_score": sentiment["score"],
                "finbert_compound": sentiment["compound"],
                "finbert_positive": sentiment["positive"],
                "finbert_negative": sentiment["negative"],
                "finbert_neutral": sentiment["neutral"],
                "weight": weight,
                "weighted_compound": weighted_compound,
            }
            results.append(result)
        
        return pd.DataFrame(results)
    
    def compare_with_vader(
        self,
        posts_df: pd.DataFrame,
        vader_compound_col: str = "vader_compound",
        vader_label_col: str = "vader_label"
    ) -> pd.DataFrame:
        """
        Compare FinBERT results with VADER results on same texts.
        
        This is useful for:
        1. Validating model consistency
        2. Identifying edge cases
        3. Justifying model choice in METHODOLOGY.md
        
        Args:
            posts_df: DataFrame with both VADER and FinBERT results
            vader_compound_col: Column name for VADER compound scores
            vader_label_col: Column name for VADER labels
            
        Returns:
            DataFrame with comparison columns added:
            - agreement: 1 if same label, 0 if different
            - score_difference: Difference in compound scores
        """
        df = posts_df.copy()
        
        # Check required columns
        if vader_label_col not in df.columns:
            logger.error(f"Missing VADER label column: {vader_label_col}")
            return df
        
        if "finbert_label" not in df.columns:
            logger.error("Missing FinBERT label column. Run analyze_posts first.")
            return df
        
        # Calculate agreement
        df["agreement"] = (
            df[vader_label_col] == df["finbert_label"]
        ).astype(int)
        
        # Calculate score difference
        if vader_compound_col in df.columns:
            df["score_difference"] = abs(
                df[vader_compound_col] - df["finbert_compound"]
            )
        
        # Log agreement statistics
        agreement_rate = df["agreement"].mean()
        logger.info(
            f"VADER/FinBERT agreement rate: {agreement_rate:.1%} "
            f"({df['agreement'].sum()}/{len(df)} posts)"
        )
        
        if "score_difference" in df.columns:
            logger.info(
                f"Mean score difference: {df['score_difference'].mean():.3f}"
            )
        
        return df
    
    def get_daily_sentiment(
        self,
        posts_df: pd.DataFrame,
        date_key: str = "created_at"
    ) -> pd.DataFrame:
        """
        Aggregate post sentiment to daily level.
        
        Args:
            posts_df: DataFrame from analyze_posts()
            date_key: Key for date/timestamp field
            
        Returns:
            DataFrame with daily aggregated sentiment
        """
        if posts_df.empty:
            return pd.DataFrame()
        
        df = posts_df.copy()
        
        # Parse dates
        if date_key in df.columns:
            df["date"] = pd.to_datetime(df[date_key]).dt.date
            df["date"] = pd.to_datetime(df["date"])
        
        # Check required columns
        if "finbert_compound" not in df.columns:
            logger.error("DataFrame missing finbert_compound column. Run analyze_posts first.")
            return pd.DataFrame()
        
        # Group by date (and ticker if present)
        group_cols = ["date"]
        if "ticker" in df.columns:
            group_cols = ["date", "ticker"]
        
        daily = df.groupby(group_cols).agg(
            finbert_compound=("finbert_compound", "mean"),
            finbert_positive=("finbert_positive", "mean"),
            finbert_negative=("finbert_negative", "mean"),
            finbert_neutral=("finbert_neutral", "mean"),
            weighted_compound=("weighted_compound", "mean"),
            post_count=("finbert_compound", "count"),
            avg_score=("score", "mean") if "score" in df.columns else ("finbert_compound", "count"),
        ).reset_index()
        
        # Determine dominant sentiment
        def get_dominant_label(row):
            if row["finbert_positive"] > row["finbert_negative"]:
                if row["finbert_positive"] > row["finbert_neutral"]:
                    return "positive"
                return "neutral"
            elif row["finbert_negative"] > row["finbert_neutral"]:
                return "negative"
            return "neutral"
        
        daily["dominant_sentiment"] = daily.apply(get_dominant_label, axis=1)
        
        # Calculate momentum
        if "ticker" in daily.columns:
            daily = daily.sort_values(["ticker", "date"])
            daily["sentiment_momentum"] = daily.groupby("ticker")["finbert_compound"].diff()
        else:
            daily = daily.sort_values("date")
            daily["sentiment_momentum"] = daily["finbert_compound"].diff()
        
        # Round numeric columns
        numeric_cols = ["finbert_compound", "finbert_positive", "finbert_negative",
                       "finbert_neutral", "weighted_compound", "sentiment_momentum"]
        for col in numeric_cols:
            if col in daily.columns:
                daily[col] = daily[col].round(4)
        
        logger.info(f"Aggregated to {len(daily)} daily sentiment records")
        
        return daily


# ============================================================
# Convenience Functions
# ============================================================

def analyze_sentiment_finbert(text: str) -> dict:
    """
    Convenience function to analyze single text with FinBERT.
    
    Args:
        text: Text to analyze
        
    Returns:
        Sentiment dictionary
    """
    analyzer = FinBERTAnalyzer()
    return analyzer.analyze_text(text)


# ============================================================
# Module Test
# ============================================================

if __name__ == "__main__":
    print("FinBERT Sentiment Analyzer Test")
    print("=" * 60)
    
    # Check if transformers is available
    if not TRANSFORMERS_AVAILABLE:
        print("❌ transformers not installed")
        print("   Install with: pip install transformers torch")
        exit(1)
    
    # Initialize analyzer (downloads model on first use)
    print("\nInitializing FinBERT (may download ~500MB on first use)...")
    analyzer = FinBERTAnalyzer()
    
    # Test single texts
    test_texts = [
        "Revenue exceeded analyst expectations by 15%",
        "The company reported a significant loss this quarter",
        "Market conditions remain challenging with headwinds expected",
        "We maintain our buy rating with a price target of $200",
        "Concerns about rising interest rates impact margins",
        "Earnings per share increased 25% year over year",
        "The stock is undervalued relative to its peers",
        "Management provided cautious guidance for Q4",
    ]
    
    print("\n1. Single Text Analysis:")
    print("-" * 60)
    for text in test_texts:
        result = analyzer.analyze_text(text)
        print(f"\nText: {text[:50]}...")
        print(f"  Label: {result['label']}")
        print(f"  Confidence: {result['score']:.2%}")
        print(f"  Compound: {result['compound']:.3f}")
    
    # Test batch analysis
    print("\n" + "=" * 60)
    print("2. Batch Analysis:")
    print("-" * 60)
    
    batch_results = analyzer.analyze_batch(test_texts)
    
    print(f"\nAnalyzed {len(batch_results)} texts")
    
    # Summary
    labels = [r["label"] for r in batch_results]
    print(f"\nLabel distribution:")
    print(f"  Positive: {labels.count('positive')}")
    print(f"  Negative: {labels.count('negative')}")
    print(f"  Neutral: {labels.count('neutral')}")
    
    print("\n" + "=" * 60)
    print("✅ All FinBERT tests completed!")
