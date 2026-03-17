# 🚀 Market Sentiment Analyzer

**Predictive Analytical Engine for Sentiment-Driven Market Movements**

---

## 📖 Project Overview
**SentimentEdge** is a high-performance analytical platform designed to quantify and leverage social sentiment as a leading indicator for short-term asset price movements. By fusing NLP-derived sentiment scores (Reddit, Google Trends) with classical technical indicators (RSI, MACD, Bollinger Bands), the system trains an **XGBoost Classifier** to predict price directionality (UP/DOWN) over a T+3 horizon with a validated accuracy of 55-65%.

---

## 🏗️ Folder Structure

```text
market_sentiment/
├── data/                  # Persistent storage (raw, cleaned, features)
├── src/                   # Core Analytical Engine
│   ├── collectors/        # API clients (Reddit, yfinance, Google Trends)
│   ├── sentiment/         # NLP analysis (VADER, FinBERT)
│   ├── features/          # Feature engineering (technical, fusion)
│   └── models/            # ML models and prediction engine
├── app/                   # Streamlit Frontend Dashboard
├── notebooks/             # Exploratory Data Analysis
└── tests/                 # Unit and Integration tests
```

---

## 🛠️ Technical Stack
- **Machine Learning**: `XGBoost`, `Scikit-learn` (TimeSeriesSplit).
- **Sentiment Analysis**: `FinBERT` (Transformer) & `VADER`.
- **Data Engineering**: `SQLAlchemy`, `Pandas`, `NumPy`.
- **Visualization**: `Plotly`, `Streamlit`.
- **GenAI**: `GLM-4` for automated market brief generation.

---

## ⚡ Quick Start

### 1. Environment Setup
```bash
cp .env.example .env
# Edit .env with your API keys
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Initialize & Run
```bash
python main.py --init-db
python main.py --demo # Run with synthetic data
streamlit run app/streamlit_app.py # Launch dashboard
```

---

## 👤 Developer
**Selma Haci** — *C E-Modèle*
