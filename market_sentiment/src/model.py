"""
Machine Learning Model Module

XGBoost classifier for predicting stock price direction using proper
time series cross-validation (TimeSeriesSplit).

CRITICAL: Never use random train/test split for time series data!
Random splits cause data leakage - training on future data to predict past.
TimeSeriesSplit ensures we only train on past and test on future.

Model Features:
- XGBoost Gradient Boosting Classifier
- TimeSeriesSplit cross-validation (5 folds by default)
- Feature importance analysis
- Sentiment-price correlation analysis at different lags
- Prediction with confidence levels

Target:
- Binary classification: Will price be UP at J+3?
- target_label: 1 = price up, 0 = price down/flat

Evaluation Metrics:
- Accuracy, Precision, Recall, F1, AUC-ROC
- Feature importance ranking
- Sentiment correlation at lags 0, 1, 2, 3 days

DISCLAIMER: ML models for educational purposes only.
Past performance does not guarantee future results.
Model accuracy ~55-65% is realistic (better than random 50%).
"""

import numpy as np
import pandas as pd
from datetime import datetime
from loguru import logger
from typing import Optional
from scipy import stats

# ML libraries
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

# XGBoost
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    logger.warning("xgboost not installed. Install with: pip install xgboost")


class SentimentPredictor:
    """
    XGBoost-based stock direction predictor with proper time series validation.
    
    This class implements a machine learning pipeline for predicting stock
    price direction using sentiment and technical features.
    
    Key Design Decisions:
    1. TimeSeriesSplit instead of random split (prevents data leakage)
    2. Feature scaling fit only on training data
    3. Separate models per ticker (different dynamics)
    4. Feature importance for interpretability
    
    Attributes:
        db_path: Path to SQLite database
        model: Trained XGBoost classifier
        scaler: StandardScaler for feature normalization
        feature_columns: List of feature column names
    
    Example:
        >>> predictor = SentimentPredictor(db_path="./data/market_sentiment.db")
        >>> results = predictor.train("AAPL")
        >>> prediction = predictor.predict_tomorrow("AAPL")
    """
    
    # Default XGBoost parameters
    DEFAULT_PARAMS = {
        "n_estimators": 200,
        "max_depth": 4,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "use_label_encoder": False,
        "eval_metric": "logloss",
        "random_state": 42,
        "n_jobs": -1,  # Use all cores
    }
    
    # Columns that should NEVER be used as features
    EXCLUDE_COLUMNS = {
        "id", "ticker", "date", "target_label", "target_return",
        "future_close", "combined_text", "title", "selftext",
        "created_at", "fetched_at", "sentiment_data_available",
    }
    
    def __init__(
        self,
        db_path: Optional[str] = None,
        feature_df: Optional[pd.DataFrame] = None,
        params: Optional[dict] = None
    ) -> None:
        """
        Initialize the sentiment predictor.
        
        Args:
            db_path: Path to SQLite database (alternative to feature_df)
            feature_df: Pre-loaded feature DataFrame
            params: Custom XGBoost parameters
        """
        if not XGBOOST_AVAILABLE:
            raise ImportError(
                "xgboost is not installed. "
                "Install with: pip install xgboost"
            )
        
        self.db_path = db_path
        self.feature_df = feature_df
        self.model = None
        self.scaler = None
        self.feature_columns = []
        
        # Model parameters
        self.params = {**self.DEFAULT_PARAMS}
        if params:
            self.params.update(params)
        
        # Store results
        self.training_results = {}
        self.evaluation_results = {}
        
        logger.info("SentimentPredictor initialized")
    
    def load_features(self, ticker: Optional[str] = None) -> pd.DataFrame:
        """
        Load feature matrix from database or use provided DataFrame.
        
        Args:
            ticker: Optional ticker filter
            
        Returns:
            DataFrame with features
        """
        if self.feature_df is not None:
            df = self.feature_df.copy()
            if ticker:
                df = df[df["ticker"] == ticker.upper()]
            return df
        
        if self.db_path:
            from src.storage import load_feature_matrix
            df = load_feature_matrix(self.db_path, ticker=ticker)
            return df
        
        raise ValueError("Either db_path or feature_df must be provided")
    
    def get_feature_columns(self, df: pd.DataFrame) -> list[str]:
        """
        Get list of columns to use as features.
        
        Excludes identifiers, targets, and text columns.
        
        Args:
            df: DataFrame with all columns
            
        Returns:
            List of feature column names
        """
        feature_cols = [
            col for col in df.columns
            if col not in self.EXCLUDE_COLUMNS
            and df[col].dtype in ["float64", "int64", "float32", "int32"]
        ]
        
        # Remove any remaining non-numeric columns
        feature_cols = [col for col in feature_cols if df[col].dtype.kind in 'bifc']
        
        return feature_cols
    
    def prepare_data(
        self,
        ticker: str,
        test_size: float = 0.2
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
        """
        Prepare data for ML training with proper time series handling.
        
        CRITICAL: Data is sorted by date ascending and split chronologically.
        We NEVER shuffle - that would cause data leakage.
        
        Args:
            ticker: Stock ticker symbol
            test_size: Fraction of data for testing (chronological split)
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test, test_dates)
        """
        # Load features
        df = self.load_features(ticker)
        
        if df.empty:
            raise ValueError(f"No data found for ticker: {ticker}")
        
        # Sort by date ascending (CRITICAL for time series)
        df = df.sort_values("date").reset_index(drop=True)
        
        # Get feature columns
        self.feature_columns = self.get_feature_columns(df)
        
        # Drop rows with NaN in features or target
        valid_cols = self.feature_columns + ["target_label"]
        df_clean = df.dropna(subset=valid_cols).copy()
        
        if len(df_clean) < 50:
            raise ValueError(f"Insufficient data for {ticker}: {len(df_clean)} rows")
        
        logger.info(f"Preparing {len(df_clean)} samples for {ticker}")
        
        # Split features and target
        X = df_clean[self.feature_columns].values
        y = df_clean["target_label"].values
        
        # Chronological split (no shuffle!)
        split_idx = int(len(X) * (1 - test_size))
        
        X_train = X[:split_idx]
        X_test = X[split_idx:]
        y_train = y[:split_idx]
        y_test = y[split_idx:]
        
        # Get test dates for analysis
        test_dates = df_clean[["date", "ticker"]].iloc[split_idx:].copy()
        
        # Scale features (fit only on training data!)
        self.scaler = StandardScaler()
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)
        
        logger.info(
            f"Data prepared: {len(X_train)} train, {len(X_test)} test, "
            f"{len(self.feature_columns)} features"
        )
        
        return X_train, X_test, y_train, y_test, test_dates
    
    def train(
        self,
        ticker: str,
        n_splits: int = 5,
        test_size: float = 0.2
    ) -> dict:
        """
        Train XGBoost model with TimeSeriesSplit cross-validation.
        
        TimeSeriesSplit ensures we never train on future data:
        - Fold 1: Train [0:20%], Test [20%:40%]
        - Fold 2: Train [0:40%], Test [40%:60%]
        - Fold 3: Train [0:60%], Test [60%:80%]
        - Fold 4: Train [0:80%], Test [80%:100%]
        - Fold 5: Train on all, evaluate on holdout
        
        Args:
            ticker: Stock ticker symbol
            n_splits: Number of CV splits
            test_size: Final holdout test size
            
        Returns:
            Dictionary with training results:
            - accuracy: Holdout accuracy
            - auc_roc: AUC-ROC score
            - cv_scores: Cross-validation scores
            - feature_importance: Top features dict
            - classification_report: Detailed metrics
        """
        logger.info(f"Training model for {ticker}...")
        
        # Prepare data
        X_train, X_test, y_train, y_test, test_dates = self.prepare_data(
            ticker, test_size=test_size
        )
        
        # Initialize model
        self.model = xgb.XGBClassifier(**self.params)
        
        # TimeSeriesSplit cross-validation
        tscv = TimeSeriesSplit(n_splits=n_splits)
        
        cv_scores = []
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X_train)):
            X_fold_train = X_train[train_idx]
            y_fold_train = y_train[train_idx]
            X_fold_val = X_train[val_idx]
            y_fold_val = y_train[val_idx]
            
            # Train on fold
            self.model.fit(X_fold_train, y_fold_train, verbose=False)
            
            # Evaluate on fold validation
            y_pred = self.model.predict(X_fold_val)
            fold_acc = accuracy_score(y_fold_val, y_pred)
            cv_scores.append(fold_acc)
            
            logger.debug(f"Fold {fold + 1}: accuracy = {fold_acc:.3f}")
        
        # Train on full training set
        self.model.fit(X_train, y_train, verbose=False)
        
        # Evaluate on holdout test set
        y_pred = self.model.predict(X_test)
        y_proba = self.model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        try:
            auc_roc = roc_auc_score(y_test, y_proba)
        except ValueError:
            auc_roc = 0.5  # All same class
        
        # Feature importance
        importance_dict = dict(zip(
            self.feature_columns,
            self.model.feature_importances_
        ))
        importance_sorted = dict(
            sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
        )
        
        # Store results
        results = {
            "ticker": ticker,
            "n_train": len(X_train),
            "n_test": len(X_test),
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "auc_roc": auc_roc,
            "cv_scores": cv_scores,
            "cv_mean": np.mean(cv_scores),
            "cv_std": np.std(cv_scores),
            "feature_importance": importance_sorted,
            "top_features": list(importance_sorted.keys())[:10],
            "classification_report": classification_report(
                y_test, y_pred, output_dict=True
            ),
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
            "test_dates": test_dates,
            "y_test": y_test,
            "y_pred": y_pred,
            "y_proba": y_proba,
        }
        
        self.training_results[ticker] = results
        
        logger.info(
            f"Training complete for {ticker}: "
            f"accuracy={accuracy:.3f}, AUC={auc_roc:.3f}, "
            f"CV={np.mean(cv_scores):.3f}±{np.std(cv_scores):.3f}"
        )
        
        return results
    
    def evaluate_all_tickers(
        self,
        tickers: list[str],
        n_splits: int = 5
    ) -> pd.DataFrame:
        """
        Train and evaluate models for all tickers.
        
        Returns a comparison DataFrame for easy analysis.
        
        Args:
            tickers: List of ticker symbols
            n_splits: CV splits per ticker
            
        Returns:
            DataFrame with metrics per ticker
        """
        results_list = []
        
        for ticker in tickers:
            try:
                result = self.train(ticker, n_splits=n_splits)
                
                results_list.append({
                    "ticker": ticker,
                    "accuracy": result["accuracy"],
                    "auc_roc": result["auc_roc"],
                    "cv_mean": result["cv_mean"],
                    "cv_std": result["cv_std"],
                    "precision": result["precision"],
                    "recall": result["recall"],
                    "f1_score": result["f1_score"],
                    "top_feature": result["top_features"][0] if result["top_features"] else None,
                    "n_samples": result["n_train"] + result["n_test"],
                })
                
            except Exception as e:
                logger.warning(f"Failed to train {ticker}: {e}")
                continue
        
        df = pd.DataFrame(results_list)
        
        if not df.empty:
            df = df.sort_values("auc_roc", ascending=False)
            logger.info(f"Evaluated {len(df)} tickers successfully")
        
        return df
    
    def predict_tomorrow(
        self,
        ticker: str,
        latest_features: Optional[pd.DataFrame] = None
    ) -> dict:
        """
        Generate prediction for the next trading day.
        
        Uses the latest available features to predict if price
        will be UP or DOWN in 3 trading days.
        
        Args:
            ticker: Stock ticker symbol
            latest_features: Optional DataFrame with latest features
                            If None, loads from database
            
        Returns:
            Dictionary with:
            - ticker: Stock symbol
            - prediction: "UP" or "DOWN"
            - probability: Confidence probability
            - confidence: "high", "medium", or "low"
            - key_signals: Top 3 features driving prediction
        """
        # Check if model is trained
        if self.model is None or ticker not in self.training_results:
            # Train if not already done
            self.train(ticker)
        
        # Get latest features
        if latest_features is None:
            df = self.load_features(ticker)
            latest_features = df.sort_values("date").iloc[[-1]]
        
        # Extract features
        X = latest_features[self.feature_columns].values
        
        # Handle NaN values
        if np.any(np.isnan(X)):
            # Fill NaN with column means from training
            X = np.nan_to_num(X, nan=0.0)
        
        # Scale
        if self.scaler is not None:
            X = self.scaler.transform(X)
        
        # Predict
        prediction_proba = self.model.predict_proba(X)[0]
        prediction_class = self.model.predict(X)[0]
        
        # Determine confidence level
        confidence_proba = max(prediction_proba)
        if confidence_proba >= 0.7:
            confidence = "high"
        elif confidence_proba >= 0.55:
            confidence = "medium"
        else:
            confidence = "low"
        
        # Get key signals (top contributing features)
        feature_imp = self.training_results[ticker]["feature_importance"]
        
        # Get actual feature values for the prediction
        feature_values = latest_features[self.feature_columns].iloc[0]
        
        # Identify signals
        signals = []
        for feat in list(feature_imp.keys())[:5]:
            value = feature_values.get(feat, 0)
            if pd.notna(value):
                if "sentiment" in feat.lower() and abs(value) > 0.1:
                    signals.append(f"{feat}: {value:.3f}")
                elif "rsi" in feat.lower() and (value > 70 or value < 30):
                    signals.append(f"{feat}: {value:.1f} ({'overbought' if value > 70 else 'oversold'})")
                elif "volume" in feat.lower() and value > 1.5:
                    signals.append(f"{feat}: {value:.1f}x average")
        
        result = {
            "ticker": ticker,
            "prediction": "UP" if prediction_class == 1 else "DOWN",
            "probability": float(max(prediction_proba)),
            "probability_up": float(prediction_proba[1]),
            "probability_down": float(prediction_proba[0]),
            "confidence": confidence,
            "key_signals": signals[:3],
            "prediction_date": datetime.now().isoformat(),
        }
        
        logger.info(
            f"Prediction for {ticker}: {result['prediction']} "
            f"({result['probability']:.1%} confidence - {confidence})"
        )
        
        return result
    
    def get_sentiment_price_correlation(
        self,
        ticker: str,
        sentiment_col: str = "vader_compound"
    ) -> dict:
        """
        Analyze correlation between sentiment and future returns.
        
        This tests the core hypothesis: Does sentiment predict price?
        
        We calculate correlation between:
        - Day 0 sentiment vs Day 0 return (contemporaneous)
        - Day 0 sentiment vs Day 1 return (sentiment → next day)
        - Day 0 sentiment vs Day 2 return (sentiment → 2 days)
        - Day 0 sentiment vs Day 3 return (sentiment → 3 days)
        
        If correlation is highest at lag 1-3, it validates the hypothesis.
        
        Args:
            ticker: Stock ticker symbol
            sentiment_col: Column name for sentiment scores
            
        Returns:
            Dictionary with correlation at each lag:
            - lag_0: Same-day correlation
            - lag_1: Sentiment → next day return
            - lag_2: Sentiment → 2-day return
            - lag_3: Sentiment → 3-day return
            - best_lag: Lag with highest absolute correlation
            - conclusion: Interpretation string
        """
        df = self.load_features(ticker)
        
        if df.empty:
            return {"error": f"No data for {ticker}"}
        
        if sentiment_col not in df.columns or "daily_return" not in df.columns:
            return {"error": "Required columns missing"}
        
        # Sort by date
        df = df.sort_values("date").copy()
        
        # Calculate correlations at different lags
        results = {}
        correlations = []
        
        for lag in range(4):
            # Sentiment at day t vs return at day t+lag
            if lag == 0:
                # Same day
                corr_df = df[[sentiment_col, "daily_return"]].dropna()
            else:
                # Future returns
                df[f"return_lag{lag}"] = df["daily_return"].shift(-lag)
                corr_df = df[[sentiment_col, f"return_lag{lag}"]].dropna()
            
            if len(corr_df) > 10:
                corr, pvalue = stats.pearsonr(
                    corr_df[sentiment_col],
                    corr_df.iloc[:, 1]
                )
                results[f"lag_{lag}"] = {
                    "correlation": corr,
                    "p_value": pvalue,
                    "significant": pvalue < 0.05,
                    "n_samples": len(corr_df),
                }
                correlations.append((lag, abs(corr), corr, pvalue))
        
        # Find best lag
        if correlations:
            best_lag = max(correlations, key=lambda x: x[1])
            results["best_lag"] = int(best_lag[0])
            results["best_correlation"] = best_lag[2]
            results["best_p_value"] = best_lag[3]
            
            # Conclusion
            if best_lag[0] > 0 and best_lag[3] < 0.05:
                results["conclusion"] = (
                    f"Sentiment shows significant predictive power: "
                    f"correlation {best_lag[2]:.3f} at lag {best_lag[0]} (p={best_lag[3]:.4f})"
                )
            else:
                results["conclusion"] = (
                    f"Sentiment shows weak predictive power. "
                    f"Best correlation {best_lag[2]:.3f} at lag {best_lag[0]}"
                )
        
        logger.info(
            f"Sentiment-price correlation for {ticker}: "
            f"best lag = {results.get('best_lag', 'N/A')}, "
            f"corr = {results.get('best_correlation', 0):.3f}"
        )
        
        return results
    
    def get_feature_importance_analysis(
        self,
        ticker: str,
        top_n: int = 20
    ) -> pd.DataFrame:
        """
        Get detailed feature importance analysis.
        
        Args:
            ticker: Stock ticker symbol
            top_n: Number of top features to return
            
        Returns:
            DataFrame with feature importance ranking
        """
        if ticker not in self.training_results:
            raise ValueError(f"Model not trained for {ticker}")
        
        importance = self.training_results[ticker]["feature_importance"]
        
        df = pd.DataFrame([
            {"feature": k, "importance": v}
            for k, v in importance.items()
        ])
        
        df = df.sort_values("importance", ascending=False).head(top_n)
        df = df.reset_index(drop=True)
        df["rank"] = df.index + 1
        
        # Add feature category
        def categorize(feat):
            if "sentiment" in feat.lower() or "vader" in feat.lower():
                return "sentiment"
            elif "lag" in feat.lower():
                return "lag"
            elif any(x in feat.lower() for x in ["rsi", "macd", "bb_"]):
                return "technical"
            elif "volume" in feat.lower():
                return "volume"
            else:
                return "other"
        
        df["category"] = df["feature"].apply(categorize)
        
        return df[["rank", "feature", "importance", "category"]]
    
    def save_model(self, ticker: str, filepath: str) -> None:
        """
        Save trained model to file.
        
        Args:
            ticker: Ticker symbol
            filepath: Path to save model
        """
        import joblib
        
        if ticker not in self.training_results:
            raise ValueError(f"Model not trained for {ticker}")
        
        model_data = {
            "model": self.model,
            "scaler": self.scaler,
            "feature_columns": self.feature_columns,
            "params": self.params,
            "training_results": self.training_results[ticker],
        }
        
        joblib.dump(model_data, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, ticker: str, filepath: str) -> None:
        """
        Load trained model from file.
        
        Args:
            ticker: Ticker symbol
            filepath: Path to model file
        """
        import joblib
        
        model_data = joblib.load(filepath)
        
        self.model = model_data["model"]
        self.scaler = model_data["scaler"]
        self.feature_columns = model_data["feature_columns"]
        self.params = model_data["params"]
        self.training_results[ticker] = model_data["training_results"]
        
        logger.info(f"Model loaded from {filepath}")


# ============================================================
# Module Test
# ============================================================

if __name__ == "__main__":
    from datetime import datetime, timedelta
    import numpy as np
    
    print("XGBoost ML Model Test")
    print("=" * 60)
    
    if not XGBOOST_AVAILABLE:
        print("❌ xgboost not installed")
        print("   Install with: pip install xgboost")
        exit(1)
    
    # Create synthetic feature data for testing
    np.random.seed(42)
    dates = [datetime.now() - timedelta(days=i) for i in range(300, 0, -1)]
    
    # Simulate features
    n_samples = 300
    test_df = pd.DataFrame({
        "date": dates,
        "ticker": "TEST",
        "close": 150 * np.cumprod(1 + np.random.normal(0.001, 0.02, n_samples)),
        "volume": np.random.randint(40000000, 60000000, n_samples),
        "daily_return": np.random.normal(0.001, 0.02, n_samples),
        "rsi_14": np.random.uniform(30, 70, n_samples),
        "macd": np.random.normal(0, 1, n_samples),
        "macd_signal": np.random.normal(0, 1, n_samples),
        "bb_position": np.random.uniform(0, 1, n_samples),
        "volume_ratio": np.random.uniform(0.5, 2.0, n_samples),
        "vader_compound": np.random.uniform(-0.3, 0.3, n_samples),
        "vader_compound_lag1": np.random.uniform(-0.3, 0.3, n_samples),
        "vader_compound_lag2": np.random.uniform(-0.3, 0.3, n_samples),
        "sentiment_x_volume": np.random.uniform(-0.5, 0.5, n_samples),
        "search_interest": np.random.uniform(30, 70, n_samples),
    })
    
    # Create target (with some predictive signal)
    test_df["target_label"] = (
        test_df["vader_compound_lag1"] > 0
    ).astype(int)  # Lag1 sentiment predicts direction
    
    # Add some noise
    noise_idx = np.random.choice(n_samples, size=int(n_samples * 0.3), replace=False)
    test_df.loc[noise_idx, "target_label"] = 1 - test_df.loc[noise_idx, "target_label"]
    
    print(f"\n1. Test data created: {len(test_df)} samples")
    
    # Initialize predictor
    print("\n2. Initializing predictor...")
    predictor = SentimentPredictor(feature_df=test_df)
    
    # Train model
    print("\n3. Training model...")
    results = predictor.train("TEST")
    
    print(f"\n   Results:")
    print(f"   - Accuracy: {results['accuracy']:.3f}")
    print(f"   - AUC-ROC: {results['auc_roc']:.3f}")
    print(f"   - CV Score: {results['cv_mean']:.3f} ± {results['cv_std']:.3f}")
    
    # Feature importance
    print("\n4. Top 5 features:")
    for i, (feat, imp) in enumerate(list(results['feature_importance'].items())[:5]):
        print(f"   {i+1}. {feat}: {imp:.4f}")
    
    # Predict
    print("\n5. Making prediction...")
    prediction = predictor.predict_tomorrow("TEST")
    print(f"   Prediction: {prediction['prediction']}")
    print(f"   Probability: {prediction['probability']:.1%}")
    print(f"   Confidence: {prediction['confidence']}")
    
    # Sentiment correlation
    print("\n6. Sentiment-price correlation analysis:")
    corr = predictor.get_sentiment_price_correlation("TEST")
    for lag in range(4):
        if f"lag_{lag}" in corr:
            r = corr[f"lag_{lag}"]
            print(f"   Lag {lag}: r={r['correlation']:.3f}, p={r['p_value']:.4f}")
    print(f"   Conclusion: {corr.get('conclusion', 'N/A')}")
    
    print("\n" + "=" * 60)
    print("✅ XGBoost ML model test completed!")
