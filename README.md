# Stock Price Direction — MLOps Pipeline

A complete, production-grade MLOps pipeline that ingests real daily stock market data, engineers predictive features, trains and tracks multiple machine learning models with MLflow, serves the best model as a REST API, monitors for data drift using Evidently, and retrains automatically every weekday via GitHub Actions CI/CD.

---

## What this project does

Every weekday at 8am UTC, or on every push to `main`, the pipeline runs automatically in the cloud:

1. Fetches the last 100 days of daily price data for **AAPL, MSFT, GOOGL, AMZN, TSLA** from the Alpha Vantage API
2. Engineers 7 financial features from raw OHLCV prices — moving average ratios, momentum, volatility, daily return, and gap
3. Creates a binary classification target: will tomorrow's closing price be higher than today's?
4. Trains 3 machine learning models and logs every run to **MLflow** — parameters, metrics, and model artifacts
5. Registers the best model and saves it as a deployable `.pkl` file
6. Runs **Evidently drift detection** comparing training data distribution against recent production data
7. Serves the best model as a **FastAPI REST endpoint** — send stock features, get a buy/sell prediction back

---

## Architecture

```
Alpha Vantage API (real daily stock data)
        ↓
    pipeline/ingest.py
        ↓
data/stock_prices.csv        ← 500 rows: 5 symbols × 100 days

        ↓
    pipeline/features.py
        ↓
data/features.csv            ← 405 rows with 7 engineered features + target

        ↓
    pipeline/train.py
        ↓
MLflow experiment tracking   ← 3 models logged: LogReg, RF-100, RF-200
models/RandomForest_100.pkl  ← best model registered

        ↓
    pipeline/monitor.py
        ↓
Evidently drift report       ← compares training vs recent data distributions
reports/drift_YYYYMMDD.html  ← full visual HTML report

        ↓
    pipeline/serve.py
        ↓
FastAPI REST endpoint         ← POST /predict → {prediction, direction, confidence}
```

**CI/CD pipeline (GitHub Actions):**
```
git push → install deps → ingest → features → train → tests → drift check → done
              ↑
    also triggers every weekday at 8am UTC automatically (cron: 0 8 * * 1-5)
```

---

## Technologies used

| Technology | Role |
|---|---|
| **Python 3.11** | Core language for all pipeline scripts |
| **Alpha Vantage API** | Free stock market data source — daily OHLCV prices |
| **pandas + numpy** | Data manipulation and feature engineering |
| **scikit-learn** | Machine learning models — Logistic Regression, Random Forest |
| **MLflow** | Experiment tracking, model registry, artifact storage |
| **FastAPI + uvicorn** | Model serving as a production REST API |
| **Evidently** | Data drift detection — compares training vs production distributions |
| **scipy** | Statistical tests for drift detection fallback |
| **GitHub Actions** | CI/CD — automated retraining pipeline every weekday |
| **pytest** | 8 automated tests covering data, features, model, and predictions |

---

## Project structure

```
stock-mlops/
│
├── pipeline/
│   ├── __init__.py          # makes pipeline a Python package
│   ├── ingest.py            # fetch daily prices from Alpha Vantage API
│   ├── features.py          # engineer 7 financial features + binary target
│   ├── train.py             # train 3 models, log all runs to MLflow
│   ├── serve.py             # FastAPI REST endpoint serving predictions
│   ├── monitor.py           # Evidently drift detection + HTML report
│   └── main.py              # orchestrates all 4 steps end to end
│
├── tests/
│   └── test_pipeline.py     # 8 tests: data files, features, model, predictions
│
├── .github/
│   └── workflows/
│       └── retrain.yml      # CI/CD — weekday cron + push trigger
│
├── data/                    # generated — gitignored
│   ├── stock_prices.csv     # raw OHLCV prices from API
│   └── features.csv         # engineered features ready for training
│
├── models/                  # generated — gitignored
│   └── RandomForest_100.pkl # best trained model
│
├── reports/                 # generated — gitignored
│   └── drift_YYYYMMDD.html  # Evidently visual drift report
│
├── mlruns/                  # generated — gitignored (MLflow tracking DB)
├── requirements.txt
├── .env.example
└── README.md
```

---

## How each component works in detail

### Data ingestion — `ingest.py`

Calls the Alpha Vantage `TIME_SERIES_DAILY` endpoint for each of the 5 symbols with `outputsize=compact` (last 100 days — free tier). A 15-second delay between calls respects the free tier rate limit of 5 calls per minute. Each symbol returns open, high, low, close, and volume for each trading day. All 5 symbols are combined into one CSV with 500 rows.

**Free tier limitation:** the compact output only returns the last 100 calendar days. The dataset is overwritten on every run — it always represents the most recent 100-day window. A production system would use the full output (premium) and append new rows incrementally.

### Feature engineering — `features.py`

Transforms raw dollar prices into normalised, comparable signals the model can learn from. All features are ratios or percentages so they are comparable across stocks at different price levels.

| Feature | Formula | What it captures |
|---|---|---|
| `daily_return` | (close − open) / open | How much the stock moved today |
| `daily_range` | (high − low) / open | How volatile today's session was |
| `ma_5` | 5-day rolling mean of close | Short-term price trend |
| `ma_20` | 20-day rolling mean of close | Long-term price trend |
| `ma_ratio` | ma_5 / ma_20 | Is short-term trend above/below long-term? |
| `volume_change` | volume.pct_change() | Unusual trading activity |
| `momentum_5` | close.pct_change(5) | Price direction over last 5 days |
| `volatility_10` | close.rolling(10).std() | Price uncertainty over 10 days |
| `gap` | (open − prev_close) / prev_close | Overnight price jump |

**Target variable:** `(tomorrow's close > today's close)` → 1 (up) or 0 (down). Balanced at ~51% up / 49% down across all 5 symbols.

Rows with NaN values (from rolling windows) are dropped, reducing 500 rows to ~405 usable rows.

### Experiment tracking — `train.py` + MLflow

Three models are trained and every run is logged to MLflow automatically:

| Model | Parameters | Typical accuracy |
|---|---|---|
| Logistic Regression | max_iter=1000 | ~46% |
| Random Forest | n_estimators=100 | ~47% |
| Random Forest | n_estimators=200, max_depth=5 | ~46% |

For each run MLflow records: model type, all hyperparameters, train/test split sizes, accuracy, precision, recall, F1 score, and the model artifact itself. Every run is reproducible — you can go back to any run and know exactly what data, code, and parameters produced it.

**Why ~47% accuracy?** Predicting stock price direction is genuinely hard — professional quantitative funds with massive data and complex models struggle to exceed 55% consistently. The low accuracy demonstrates a real MLOps principle: a model that does not meet a minimum quality threshold should not be deployed. In a production system you would set a gate — for example, only deploy if accuracy > 55%.

To view all experiments:
```bash
mlflow ui
# Open http://localhost:5000
```

### Model serving — `serve.py`

Loads the best trained model at server startup and exposes two endpoints:

`GET /health` — confirms the server is running and the model is loaded. Used by monitoring systems.

`POST /predict?symbol=AAPL` — accepts 7 feature values as JSON, runs them through the RandomForest model, and returns a prediction with confidence score.

**Example request:**
```bash
curl -X POST "http://localhost:8000/predict?symbol=AAPL" \
  -H "Content-Type: application/json" \
  -d '{
    "daily_return": 0.012,
    "daily_range": 0.018,
    "ma_ratio": 1.02,
    "volume_change": 0.15,
    "momentum_5": 0.03,
    "volatility_10": 5.2,
    "gap": 0.005
  }'
```

**Example response:**
```json
{
  "symbol": "AAPL",
  "prediction": 0,
  "direction": "DOWN ↓",
  "confidence": 0.59,
  "model_version": "RandomForest_100_v1"
}
```

### Drift detection — `monitor.py`

Uses Evidently to compare the statistical distribution of the 7 features between the training period (first 80% of data) and the recent production period (last 20%). Drift is detected per feature using the Kolmogorov-Smirnov test — a statistical test that measures whether two samples come from the same distribution.

If more than 50% of features drift significantly (p-value < 0.05), a drift alert is raised and the pipeline logs `exit code 1` to notify that retraining with fresh data is critical. An HTML report is saved to `reports/` with visual distribution comparisons for every feature.

**Why drift matters:** a model trained on calm market data from 2024 may degrade silently during a volatile 2025 market. Drift detection catches this before accuracy collapses — it is the difference between a model that works reliably in production and one that silently gives bad predictions for months.

### CI/CD pipeline — `retrain.yml`

GitHub Actions workflow with two triggers:

1. **Push to main** — runs the full pipeline immediately on every code change
2. **Weekday cron `0 8 * * 1-5`** — runs automatically every Monday-Friday at 8am UTC, before US market open at 9:30am ET

Steps in order:
1. Checkout code on a fresh Ubuntu runner
2. Install all Python dependencies including ODBC drivers
3. Ingest fresh stock data from Alpha Vantage
4. Engineer features from raw prices
5. Retrain all 3 models and log to MLflow
6. Run 8 automated tests — pipeline stops if any fail
7. Run drift detection — warning only, does not stop pipeline
8. Print summary

---

## How to run locally

**1. Clone the repo**
```bash
git clone https://github.com/ohawas21/stock-mlops.git
cd stock-mlops
```

**2. Create virtual environment**
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**3. Set up environment variables**
```bash
cp .env.example .env
# Edit .env and add your Alpha Vantage API key
```

Get a free API key at: https://www.alphavantage.co/support/#api-key

**4. Run the full pipeline**
```bash
python -m pipeline.main
```

This runs all 4 steps: ingest → features → drift check → train. Takes ~75 seconds due to API rate limiting.

**5. Start the prediction API**
```bash
python pipeline/serve.py
# API available at http://localhost:8000
# Docs at http://localhost:8000/docs
```

**6. View MLflow experiments**
```bash
mlflow ui
# Open http://localhost:5000
```

**7. Run tests**
```bash
pytest tests/ -v
```

---

## Environment variables

| Variable | Description |
|---|---|
| `ALPHA_VANTAGE_API_KEY` | Free API key from alphavantage.co — 25 calls/day, 5 calls/min |

---

## GitHub Actions secrets required

| Secret | Description |
|---|---|
| `ALPHA_VANTAGE_API_KEY` | Same key as above — stored securely in GitHub Secrets |

---

## MLOps concepts demonstrated

| Concept | How it is implemented |
|---|---|
| **Experiment tracking** | MLflow logs every training run with parameters, metrics, and artifacts |
| **Model registry** | Best model saved as versioned `.pkl` artifact |
| **Model serving** | FastAPI REST endpoint — any application can call `/predict` |
| **Data drift detection** | Evidently compares training vs production feature distributions |
| **Automated retraining** | GitHub Actions cron triggers full pipeline every weekday |
| **CI/CD for ML** | Push to main → tests → retrain → deploy automatically |
| **Reproducibility** | Every MLflow run records exact data, code, and parameters used |

---

## What I learned building this

- **MLOps vs DevOps** — the same CI/CD principles apply but ML adds data versioning, experiment tracking, model registry, and drift monitoring as additional layers
- **Experiment tracking with MLflow** — why logging every run matters: without it, you cannot answer "which model is in production and why was it chosen"
- **Feature engineering for financial data** — why raw prices are useless for ML and how normalised ratios make features comparable across different stocks and price levels
- **Model serving with FastAPI** — how a trained `.pkl` file becomes a live REST API that any application can call
- **Data drift** — why a model that works today can silently degrade as market conditions change, and how statistical tests detect this before accuracy collapses
- **Rate limiting** — how to handle API rate limits in production pipelines using sleep intervals between calls
- **CI/CD for ML** — why test order matters in ML pipelines (data and model must exist before tests that check them)

---

## Author

Omar Hawas — Applied AI student, TH Rosenheim