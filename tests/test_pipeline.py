import pandas as pd
import sys
import os
sys.path.insert(0, os.path.abspath("."))


def test_stock_prices_file_exists():
    """Test that the data file was created by ingest."""
    assert os.path.exists("data/stock_prices.csv"), \
        "stock_prices.csv not found — ingest step must run before tests"
    print("Test passed — stock_prices.csv exists")


def test_stock_prices_has_all_symbols():
    """Test that all 5 symbols are present."""
    df = pd.read_csv("data/stock_prices.csv")
    expected = {"AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"}
    actual   = set(df["symbol"].unique())
    assert expected == actual, f"Missing symbols: {expected - actual}"
    print(f"Test passed — symbols: {actual}")


def test_stock_prices_has_required_columns():
    """Test that all required columns exist."""
    df       = pd.read_csv("data/stock_prices.csv")
    required = ["symbol", "date", "open", "high", "low", "close", "volume"]
    for col in required:
        assert col in df.columns, f"Missing column: {col}"
    print("Test passed — all columns present")


def test_feature_engineering():
    """Test that feature engineering produces expected columns."""
    from pipeline.features import engineer_features
    df       = pd.read_csv("data/stock_prices.csv", parse_dates=["date"])
    features = engineer_features(df)
    expected = [
        "daily_return", "daily_range", "ma_ratio",
        "volume_change", "momentum_5", "volatility_10",
        "gap", "target"
    ]
    for col in expected:
        assert col in features.columns, f"Missing: {col}"
    assert len(features) > 0
    print(f"Test passed — {len(features)} rows")


def test_target_is_balanced():
    """Test target is not severely imbalanced."""
    from pipeline.features import engineer_features
    df    = pd.read_csv("data/stock_prices.csv", parse_dates=["date"])
    feats = engineer_features(df)
    ratio = feats["target"].mean()
    assert 0.3 < ratio < 0.7, f"Imbalanced: {ratio:.2f}"
    print(f"Test passed — ratio: {ratio:.2f}")


def test_no_nulls_in_features():
    """Test no nulls in feature columns."""
    from pipeline.features import engineer_features
    FEATURES = [
        "daily_return", "daily_range", "ma_ratio",
        "volume_change", "momentum_5", "volatility_10", "gap"
    ]
    df    = pd.read_csv("data/stock_prices.csv", parse_dates=["date"])
    feats = engineer_features(df)
    nulls = feats[FEATURES].isnull().sum()
    assert nulls.sum() == 0, f"Nulls found: {nulls[nulls > 0]}"
    print("Test passed — no nulls")


def test_model_file_exists():
    """Test trained model file exists."""
    assert os.path.exists("models/RandomForest_100.pkl"), \
        "Model not found — train step must run before tests"
    print("Test passed — model file exists")


def test_model_can_predict():
    """Test saved model loads and predicts."""
    import pickle
    with open("models/RandomForest_100.pkl", "rb") as f:
        model = pickle.load(f)
    X = pd.DataFrame([{
        "daily_return": 0.01, "daily_range": 0.02,
        "ma_ratio": 1.01,    "volume_change": 0.05,
        "momentum_5": 0.03,  "volatility_10": 5.0,
        "gap": 0.001
    }])
    pred = model.predict(X)
    assert pred[0] in [0, 1]
    print(f"Test passed — prediction: {pred[0]}")