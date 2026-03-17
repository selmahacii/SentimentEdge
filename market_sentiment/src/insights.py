"""
GLM Insights Module

Generates AI-powered daily market briefs using GLM (ZhipuAI).
Provides concise, data-driven analysis with proper hedging language.

Features:
- Daily brief per ticker with structured format
- Portfolio summary across all tracked stocks
- Risk factor identification
- Factual, non-advisory language with disclaimers
- Model comparison: VADER vs FinBERT agreement

IMPORTANT: All briefs include disclaimer that this is for educational
purposes only and does not constitute financial advice.

Models Available:
- glm-4-flash: Fast, cost-effective for briefs
- glm-4: More capable for detailed analysis
"""

import os
from datetime import datetime
from typing import Optional

from loguru import logger

# ZhipuAI SDK
try:
    from zhipuai import ZhipuAI
    ZHIPUAI_AVAILABLE = True
except ImportError:
    ZHIPUAI_AVAILABLE = False
    logger.warning(
        "zhipuai not installed. Install with: pip install zhipuai"
    )


# ============================================================
# System Prompts
# ============================================================

MARKET_ANALYST_SYSTEM_PROMPT = """You are a quantitative market analyst specializing in sentiment-driven trading strategies. You analyze financial data and social media sentiment to generate concise, data-driven market briefs.

CRITICAL RULES:
1. Base analysis STRICTLY on provided data - no speculation beyond what's shown
2. Acknowledge uncertainty explicitly ("data suggests", "historically")
3. NEVER give financial advice or buy/sell recommendations
4. Use precise, hedged language throughout
5. Flag contradictory signals clearly
6. Keep total response under 200 words
7. This is for EDUCATIONAL PURPOSES ONLY

Your briefs follow this exact structure:
## {TICKER} — Daily Brief ({DATE})

**Situation**: [One sentence on current price action]

**Sentiment**: [One sentence on social sentiment signal, note alignment with price]

**Technical**: [One sentence on most relevant technical indicator]

**Signal**: [What the model detects, with hedged language]

**Risk factors**: [One line on what could invalidate this signal]"""

PORTFOLIO_SYSTEM_PROMPT = """You are a portfolio analyst summarizing multiple stock briefs. Create a concise portfolio overview that:
1. Identifies common themes across stocks
2. Highlights divergent signals
3. Notes overall sentiment direction
4. Remains educational, not advisory

Keep response under 150 words. Always include disclaimer."""


# ============================================================
# Prompt Builders
# ============================================================

def build_daily_brief_prompt(ticker: str, data: dict) -> tuple[str, str]:
    """
    Build the prompt for GLM daily brief generation.
    
    Args:
        ticker: Stock ticker symbol (e.g., "AAPL")
        data: Dictionary with price, sentiment, and prediction data:
            - price: dict with current_price, daily_change, daily_change_pct
            - sentiment: dict with vader_compound, sentiment_trend, post_count
            - technical: dict with rsi_14, rsi_signal, macd_signal, bb_position
            - volume: dict with volume_ratio
            - prediction: dict with direction, probability, confidence, key_signals
            - trends: dict with search_interest, interest_change
    
    Returns:
        Tuple of (system_prompt, user_prompt)
    """
    today = datetime.now().strftime("%Y-%m-%d")
    
    # Format price data
    price_section = format_price_section(data.get("price", {}))
    
    # Format sentiment data
    sentiment_section = format_sentiment_section(data.get("sentiment", {}))
    
    # Format technical data
    technical_section = format_technical_section(data.get("technical", {}))
    
    # Format prediction data
    prediction_section = format_prediction_section(data.get("prediction", {}))
    
    # Format trends data
    trends_section = format_trends_section(data.get("trends", {}))
    
    user_prompt = f"""Generate a daily market brief for {ticker}.

PRICE DATA (last 5 days):
{price_section}

TECHNICAL INDICATORS:
{technical_section}

SENTIMENT DATA (last 7 days):
{sentiment_section}

GOOGLE TRENDS:
{trends_section}

ML MODEL PREDICTION:
{prediction_section}

Generate a brief with EXACTLY this structure:

## {ticker} — Daily Brief ({today})

**Situation**: One sentence summarizing current price action.

**Sentiment**: One sentence on Reddit/social media signal. Note if sentiment and price are aligned or diverging.

**Technical**: One sentence on the most relevant technical indicator.

**Signal**: What the model detects. Use hedged language.
Example: "Data suggests mild bullish bias over next 3 days, but RSI indicates caution."

**Risk factors**: One line on what could invalidate this signal.

Keep total response under 200 words. No financial advice."""

    return MARKET_ANALYST_SYSTEM_PROMPT, user_prompt


def format_price_section(price_data: dict) -> str:
    """Format price data for prompt."""
    if not price_data:
        return "Price data unavailable"
    
    lines = []
    
    if "current_price" in price_data:
        current = price_data["current_price"]
        change = price_data.get("daily_change", 0)
        change_pct = price_data.get("daily_change_pct", 0)
        lines.append(f"Current: ${current:.2f} ({change_pct:+.2f}% / ${change:+.2f})")
    
    if "recent_prices" in price_data:
        lines.append("Recent closes: " + ", ".join(
            f"${p:.2f}" for p in price_data["recent_prices"]
        ))
    
    if "volatility" in price_data:
        lines.append(f"Volatility (10d): {price_data['volatility']:.2%}")
    
    return "\n".join(lines) if lines else "Price data unavailable"


def format_sentiment_section(sentiment_data: dict) -> str:
    """Format sentiment data for prompt."""
    if not sentiment_data:
        return "Sentiment data unavailable"
    
    lines = []
    
    if "vader_compound" in sentiment_data:
        compound = sentiment_data["vader_compound"]
        label = "positive" if compound > 0.05 else "negative" if compound < -0.05 else "neutral"
        lines.append(f"Average sentiment: {compound:.3f} ({label})")
    
    if "sentiment_trend" in sentiment_data:
        trend = sentiment_data["sentiment_trend"]
        lines.append(f"Sentiment trend: {trend}")
    
    if "post_count" in sentiment_data:
        avg_posts = sentiment_data.get("avg_posts", 10)
        lines.append(f"Mention volume: {sentiment_data['post_count']} posts (avg: {avg_posts})")
    
    if "vader_finbert_agreement" in sentiment_data:
        agreement = sentiment_data["vader_finbert_agreement"]
        lines.append(f"VADER/FinBERT agreement: {agreement:.0%}")
    
    return "\n".join(lines) if lines else "Sentiment data unavailable"


def format_technical_section(technical_data: dict) -> str:
    """Format technical indicators for prompt."""
    if not technical_data:
        return "Technical data unavailable"
    
    lines = []
    
    if "rsi_14" in technical_data:
        rsi = technical_data["rsi_14"]
        signal = technical_data.get("rsi_signal", "neutral")
        lines.append(f"RSI (14): {rsi:.1f} ({signal})")
    
    if "macd" in technical_data:
        macd = technical_data["macd"]
        macd_sig = technical_data.get("macd_signal", 0)
        crossover = "bullish crossover" if macd > macd_sig else "bearish crossover" if macd < macd_sig else "neutral"
        lines.append(f"MACD: {macd:.2f} vs Signal {macd_sig:.2f} ({crossover})")
    
    if "bb_position" in technical_data:
        bb = technical_data["bb_position"]
        bb_signal = "near upper band" if bb > 0.8 else "near lower band" if bb < 0.2 else "neutral"
        lines.append(f"Bollinger position: {bb:.0%} ({bb_signal})")
    
    if "volume_ratio" in technical_data:
        vol = technical_data["volume_ratio"]
        vol_signal = "unusually high" if vol > 2 else "below average" if vol < 0.5 else "normal"
        lines.append(f"Volume ratio: {vol:.1f}x average ({vol_signal})")
    
    return "\n".join(lines) if lines else "Technical data unavailable"


def format_prediction_section(prediction_data: dict) -> str:
    """Format ML prediction for prompt."""
    if not prediction_data:
        return "Prediction unavailable"
    
    lines = []
    
    direction = prediction_data.get("direction", "N/A")
    probability = prediction_data.get("probability", 0)
    confidence = prediction_data.get("confidence", "unknown")
    
    lines.append(f"Direction (J+3): {direction} (probability: {probability:.0%})")
    lines.append(f"Confidence: {confidence}")
    
    if "key_signals" in prediction_data:
        lines.append("Key signals:")
        for signal in prediction_data["key_signals"][:3]:
            lines.append(f"  - {signal}")
    
    return "\n".join(lines)


def format_trends_section(trends_data: dict) -> str:
    """Format Google Trends data for prompt."""
    if not trends_data:
        return "Trends data unavailable"
    
    lines = []
    
    if "search_interest" in trends_data:
        interest = trends_data["search_interest"]
        lines.append(f"Current interest: {interest:.0f}/100")
    
    if "interest_change" in trends_data:
        change = trends_data["interest_change"]
        lines.append(f"Change (7d): {change:+.1f}%")
    
    return "\n".join(lines) if lines else "Trends data unavailable"


# ============================================================
# GLM Client
# ============================================================

class GLMInsightGenerator:
    """
    GLM-powered market insight generator.
    
    Uses ZhipuAI's GLM models to generate daily market briefs
    based on sentiment and technical analysis data.
    
    Attributes:
        client: ZhipuAI client
        model: Default model to use
    
    Example:
        >>> generator = GLMInsightGenerator(api_key="...")
        >>> brief = generator.generate_daily_brief("AAPL", data)
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "glm-4-flash"
    ) -> None:
        """
        Initialize GLM insight generator.
        
        Args:
            api_key: ZhipuAI API key (defaults to ZHIPUAI_API_KEY env var)
            model: Model to use (glm-4-flash for speed, glm-4 for detail)
        """
        if not ZHIPUAI_AVAILABLE:
            raise ImportError(
                "zhipuai is not installed. "
                "Install with: pip install zhipuai"
            )
        
        self.api_key = api_key or os.getenv("ZHIPUAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "ZhipuAI API key required. Set ZHIPUAI_API_KEY environment variable "
                "or pass api_key parameter."
            )
        
        self.client = ZhipuAI(api_key=self.api_key)
        self.model = model
        
        logger.info(f"GLMInsightGenerator initialized with model: {model}")
    
    def generate_daily_brief(
        self,
        ticker: str,
        data: dict,
        temperature: float = 0.3,
        max_tokens: int = 500
    ) -> dict:
        """
        Generate a daily market brief for a ticker.
        
        Args:
            ticker: Stock ticker symbol
            data: Dictionary with all relevant data
            temperature: Sampling temperature (0.0-1.0, lower = more focused)
            max_tokens: Maximum tokens in response
            
        Returns:
            Dictionary with:
            - ticker: Stock symbol
            - brief: Generated brief text
            - date: Generation date
            - model: Model used
            - tokens_used: Total tokens
            - generation_time: Time taken
        """
        start_time = datetime.now()
        
        # Build prompt
        system_prompt, user_prompt = build_daily_brief_prompt(ticker, data)
        
        try:
            # Call GLM API
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            
            # Extract response
            brief_text = response.choices[0].message.content
            
            # Get token usage
            usage = response.usage
            tokens_used = usage.total_tokens if usage else 0
            
            generation_time = (datetime.now() - start_time).total_seconds()
            
            logger.info(
                f"Generated brief for {ticker}: "
                f"{tokens_used} tokens, {generation_time:.2f}s"
            )
            
            return {
                "ticker": ticker,
                "brief": brief_text,
                "date": datetime.now().isoformat(),
                "model": self.model,
                "tokens_used": tokens_used,
                "generation_time": generation_time,
                "temperature": temperature,
            }
            
        except Exception as e:
            logger.error(f"Failed to generate brief for {ticker}: {e}")
            return {
                "ticker": ticker,
                "brief": f"Error generating brief: {str(e)}",
                "date": datetime.now().isoformat(),
                "error": str(e),
            }
    
    def generate_portfolio_summary(
        self,
        briefs: list[dict],
        temperature: float = 0.3,
        max_tokens: int = 400
    ) -> dict:
        """
        Generate a portfolio-wide summary.
        
        Args:
            briefs: List of individual ticker briefs
            temperature: Sampling temperature
            max_tokens: Maximum tokens
            
        Returns:
            Dictionary with portfolio summary
        """
        if not briefs:
            return {"summary": "No briefs available", "error": "empty input"}
        
        # Combine briefs
        combined_briefs = "\n\n".join([
            f"### {b.get('ticker', 'Unknown')}\n{b.get('brief', 'No brief')}"
            for b in briefs
        ])
        
        user_prompt = f"""Summarize the following stock briefs into a portfolio overview:

{combined_briefs}

Provide:
1. Common themes (if any)
2. Divergent signals
3. Overall sentiment direction
4. Notable risks

Keep under 150 words. Include disclaimer."""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": PORTFOLIO_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            
            summary_text = response.choices[0].message.content
            tokens_used = response.usage.total_tokens if response.usage else 0
            
            return {
                "summary": summary_text,
                "tickers_covered": [b.get("ticker") for b in briefs],
                "model": self.model,
                "tokens_used": tokens_used,
                "date": datetime.now().isoformat(),
            }
            
        except Exception as e:
            logger.error(f"Failed to generate portfolio summary: {e}")
            return {
                "summary": f"Error generating summary: {str(e)}",
                "error": str(e),
            }
    
    def generate_all_briefs(
        self,
        tickers_data: dict[str, dict],
        temperature: float = 0.3
    ) -> tuple[list[dict], dict]:
        """
        Generate briefs for all tickers and portfolio summary.
        
        Args:
            tickers_data: Dict mapping ticker to data dict
            temperature: Sampling temperature
            
        Returns:
            Tuple of (briefs_list, portfolio_summary)
        """
        briefs = []
        
        for ticker, data in tickers_data.items():
            logger.info(f"Generating brief for {ticker}...")
            brief = self.generate_daily_brief(ticker, data, temperature=temperature)
            briefs.append(brief)
        
        logger.info("Generating portfolio summary...")
        summary = self.generate_portfolio_summary(briefs, temperature=temperature)
        
        return briefs, summary


# ============================================================
# Convenience Functions
# ============================================================

def generate_daily_brief(
    ticker: str,
    data: dict,
    model: str = "glm-4-flash"
) -> dict:
    """
    Convenience function to generate a single daily brief.
    
    Args:
        ticker: Stock ticker symbol
        data: Dictionary with all relevant data
        model: GLM model to use
        
    Returns:
        Dictionary with brief and metadata
    """
    generator = GLMInsightGenerator(model=model)
    return generator.generate_daily_brief(ticker, data)


def generate_portfolio_summary(briefs: list[dict]) -> str:
    """
    Convenience function to generate portfolio summary.
    
    Args:
        briefs: List of brief dictionaries
        
    Returns:
        Portfolio summary string
    """
    generator = GLMInsightGenerator()
    result = generator.generate_portfolio_summary(briefs)
    return result.get("summary", "Summary unavailable")


# ============================================================
# Module Test
# ============================================================

if __name__ == "__main__":
    import os
    
    print("GLM Insights Module Test")
    print("=" * 60)
    
    # Check for API key
    api_key = os.getenv("ZHIPUAI_API_KEY")
    
    if not api_key:
        print("\n⚠️ ZHIPUAI_API_KEY not set")
        print("   Set the environment variable or pass api_key parameter")
        print("   Get API key at: https://open.bigmodel.cn/")
        print("\n   Testing prompt generation only...")
    else:
        print("\n✓ ZHIPUAI_API_KEY found")
        print("   Testing full brief generation...")
    
    # Test data
    test_data = {
        "price": {
            "current_price": 178.50,
            "daily_change": 2.30,
            "daily_change_pct": 1.31,
            "volatility": 0.018,
        },
        "sentiment": {
            "vader_compound": 0.35,
            "sentiment_trend": "improving",
            "post_count": 45,
            "avg_posts": 30,
            "vader_finbert_agreement": 0.82,
        },
        "technical": {
            "rsi_14": 62.5,
            "rsi_signal": "neutral",
            "macd": 1.5,
            "macd_signal": 1.2,
            "bb_position": 0.65,
            "volume_ratio": 1.4,
        },
        "prediction": {
            "direction": "UP",
            "probability": 0.68,
            "confidence": "medium",
            "key_signals": [
                "Sentiment positive (+0.35)",
                "RSI neutral (62.5)",
                "Volume 1.4x average",
            ],
        },
        "trends": {
            "search_interest": 55,
            "interest_change": 8.5,
        },
    }
    
    # Test prompt generation
    print("\n1. Testing prompt generation...")
    system_prompt, user_prompt = build_daily_brief_prompt("AAPL", test_data)
    
    print("\n   User Prompt Preview:")
    print("   " + "-" * 50)
    for line in user_prompt.split("\n")[:15]:
        print(f"   {line}")
    print("   ...")
    
    if api_key and ZHIPUAI_AVAILABLE:
        # Test full generation
        print("\n2. Testing GLM generation...")
        
        generator = GLMInsightGenerator(api_key=api_key, model="glm-4-flash")
        
        result = generator.generate_daily_brief("AAPL", test_data)
        
        print("\n   Generated Brief:")
        print("   " + "-" * 50)
        for line in result.get("brief", "No brief").split("\n"):
            print(f"   {line}")
        
        print(f"\n   Metadata:")
        print(f"   Model: {result.get('model')}")
        print(f"   Tokens: {result.get('tokens_used')}")
        print(f"   Time: {result.get('generation_time', 0):.2f}s")
        
        # Test portfolio summary
        print("\n3. Testing portfolio summary...")
        
        briefs = [
            result,
            {"ticker": "TSLA", "brief": "TSLA shows bearish sentiment with -0.25 compound. Technical indicators suggest oversold conditions."},
        ]
        
        summary_result = generator.generate_portfolio_summary(briefs)
        
        print("\n   Portfolio Summary:")
        print("   " + "-" * 50)
        print(summary_result.get("summary", "No summary"))
    else:
        print("\n   ⚠️ Skipping GLM generation (no API key)")
    
    print("\n" + "=" * 60)
    print("✅ GLM Insights test completed!")
