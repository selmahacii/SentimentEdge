"""
Tests for Market Sentiment Analyzer

Run tests with:
    pytest tests/ -v

Skip tests requiring external dependencies with:
    pytest tests/ -v -m "not requires_db"
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


# ============================================================
# Configuration Tests
# ============================================================

class TestConfig:
    """Tests for configuration module."""
    
    def test_config_imports(self):
        """Test that config module can be imported."""
        from src.config import settings, ALL_FEATURE_COLUMNS
        assert settings is not None
        assert ALL_FEATURE_COLUMNS is not None
    
    def test_default_tickers(self):
        """Test default tickers are loaded."""
        from src.config import settings
        assert len(settings.tickers) > 0
        assert all(isinstance(t, str) for t in settings.tickers)
    
    def test_default_subreddits(self):
        """Test default subreddits are loaded."""
        from src.config import settings
        assert len(settings.subreddits) > 0
        assert "wallstreetbets" in settings.subreddits
    
    def test_lookback_days_range(self):
        """Test lookback_days is within valid range."""
        from src.config import settings
        assert 30 <= settings.lookback_days <= 730
    
    def test_prediction_horizon_range(self):
        """Test prediction_horizon is within valid range."""
        from src.config import settings
        assert 1 <= settings.prediction_horizon <= 10
    
    def test_sentiment_model_options(self):
        """Test sentiment_model is valid option."""
        from src.config import settings
        assert settings.sentiment_model in ["vader", "finbert"]
    
    def test_feature_columns_defined(self):
        """Test feature columns are properly defined."""
        from src.config import (
            PRICE_FEATURES, TECHNICAL_FEATURES, SENTIMENT_FEATURES,
            LAG_FEATURES, CROSS_FEATURES, ALL_FEATURE_COLUMNS, EXCLUDE_COLUMNS
        )
        
        # Check lists are not empty
        assert len(PRICE_FEATURES) > 0
        assert len(TECHNICAL_FEATURES) > 0
        assert len(SENTIMENT_FEATURES) > 0
        assert len(LAG_FEATURES) > 0
        assert len(CROSS_FEATURES) > 0
        
        # Check ALL_FEATURE_COLUMNS includes all
        total_expected = (
            len(PRICE_FEATURES) + len(TECHNICAL_FEATURES) +
            len(SENTIMENT_FEATURES) + len(LAG_FEATURES) + len(CROSS_FEATURES)
        )
        assert len(ALL_FEATURE_COLUMNS) == total_expected
        
        # Check EXCLUDE_COLUMNS contains important exclusions
        assert "target_label" in EXCLUDE_COLUMNS
        assert "target_return" in EXCLUDE_COLUMNS


# ============================================================
# Storage Model Tests (No DB Required)
# ============================================================

class TestStorageModels:
    """Tests for storage model definitions (no DB connection required)."""
    
    def test_storage_imports(self):
        """Test that storage module can be imported."""
        from src.storage import Base, PriceData, RedditPost, RedditDailySentiment
        from src.storage import GoogleTrend, FeatureMatrix, ModelPrediction
        
        assert Base is not None
        assert PriceData is not None
        assert RedditPost is not None
        assert RedditDailySentiment is not None
        assert GoogleTrend is not None
        assert FeatureMatrix is not None
        assert ModelPrediction is not None
    
    def test_price_data_table_name(self):
        """Test PriceData table name."""
        from src.storage import PriceData
        assert PriceData.__tablename__ == "price_data"
    
    def test_reddit_post_table_name(self):
        """Test RedditPost table name."""
        from src.storage import RedditPost
        assert RedditPost.__tablename__ == "reddit_posts"
    
    def test_feature_matrix_has_target_columns(self):
        """Test FeatureMatrix has target columns."""
        from src.storage import FeatureMatrix
        
        columns = [c.name for c in FeatureMatrix.__table__.columns]
        assert "target_label" in columns
        assert "target_return" in columns
    
    def test_database_manager_class(self):
        """Test DatabaseManager class exists."""
        from src.storage import DatabaseManager
        assert DatabaseManager is not None


# ============================================================
# Collector Tests (No API Required)
# ============================================================

class TestCollectors:
    """Tests for data collectors (no API calls)."""
    
    def test_price_collector_import(self):
        """Test PriceCollector can be imported."""
        from src.collectors.price_collector import PriceCollector
        assert PriceCollector is not None
    
    def test_reddit_collector_import(self):
        """Test RedditCollector can be imported."""
        from src.collectors.reddit_collector import RedditCollector
        assert RedditCollector is not None
    
    def test_trends_collector_import(self):
        """Test TrendsCollector can be imported."""
        from src.collectors.trends_collector import TrendsCollector
        assert TrendsCollector is not None
    
    def test_price_collector_init(self):
        """Test PriceCollector initialization."""
        from src.collectors.price_collector import PriceCollector
        
        collector = PriceCollector(
            tickers=["AAPL", "TSLA"],
            cache_dir="./data/raw"
        )
        assert collector.tickers == ["AAPL", "TSLA"]
    
    def test_reddit_collector_init_structure(self):
        """Test RedditCollector class structure."""
        from src.collectors.reddit_collector import RedditCollector
        
        # Check class has expected methods
        assert hasattr(RedditCollector, 'search_ticker_mentions')
        assert hasattr(RedditCollector, 'get_daily_mention_count')
        assert hasattr(RedditCollector, 'collect_all_tickers')
    
    def test_trends_collector_init_structure(self):
        """Test TrendsCollector class structure."""
        from src.collectors.trends_collector import TrendsCollector
        
        # Check class has expected methods
        assert hasattr(TrendsCollector, 'fetch_interest')
        assert hasattr(TrendsCollector, 'fetch_all')
        assert hasattr(TrendsCollector, 'get_interest_change')


# ============================================================
# Sentiment Analyzer Tests (No Model Required)
# ============================================================

class TestSentiment:
    """Tests for sentiment analysis modules."""
    
    def test_vader_import(self):
        """Test VADERAnalyzer can be imported."""
        from src.sentiment.vader_analyzer import VADERAnalyzer
        assert VADERAnalyzer is not None
    
    def test_finbert_import(self):
        """Test FinBERTAnalyzer can be imported."""
        from src.sentiment.finbert_analyzer import FinBERTAnalyzer
        assert FinBERTAnalyzer is not None


# ============================================================
# Feature Engineering Tests
# ============================================================

class TestFeatures:
    """Tests for feature engineering modules."""
    
    def test_technical_imports(self):
        """Test technical module imports."""
        from src.features.technical import compute_technical_features
        assert compute_technical_features is not None
    
    def test_fusion_imports(self):
        """Test fusion module imports."""
        from src.features.fusion import merge_all_features
        assert merge_all_features is not None


# ============================================================
# Integration Tests (Require DB)
# ============================================================

@pytest.mark.requires_db
class TestIntegration:
    """Integration tests requiring database."""
    
    def test_database_initialization(self, tmp_path):
        """Test database can be initialized."""
        from src.storage import init_db
        
        db_path = str(tmp_path / "test.db")
        db = init_db(db_path)
        
        assert db is not None
        assert Path(db_path).exists()
    
    def test_price_save_load(self, tmp_path):
        """Test price data save and load."""
        import pandas as pd
        from src.storage import init_db, save_prices, load_price_history
        
        db_path = str(tmp_path / "test.db")
        db = init_db(db_path)
        
        # Create test data
        test_df = pd.DataFrame({
            "date": [datetime.now()],
            "ticker": ["AAPL"],
            "open": [180.0],
            "high": [181.0],
            "low": [179.0],
            "close": [180.5],
            "volume": [50000000],
        })
        
        # Save
        count = save_prices(test_df, db)
        assert count == 1
        
        # Load
        loaded = load_price_history(db, "AAPL")
        assert len(loaded) == 1
        assert loaded["close"].iloc[0] == 180.5


# ============================================================
# Utility Functions
# ============================================================

def get_test_data_path() -> Path:
    """Get path to test data directory."""
    return Path(__file__).parent / "test_data"


def create_test_price_df() -> "pd.DataFrame":
    """Create a test price DataFrame."""
    import pandas as pd
    
    return pd.DataFrame({
        "date": [datetime.now() - timedelta(days=i) for i in range(5)],
        "ticker": ["AAPL"] * 5,
        "open": [180.0] * 5,
        "high": [181.0] * 5,
        "low": [179.0] * 5,
        "close": [180.5] * 5,
        "volume": [50000000] * 5,
    })


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
