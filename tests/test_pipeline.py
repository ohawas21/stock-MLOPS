import pandas as pd
import sys
import os
sys.path.insert(0, os.path.abspath("."))


def test_stock_prices_file_exists():
    """Test that the data file was created by ingest."""
    assert os.path.exists("data/stock_prices.csv"), \
        "stock_prices.csv not found — run ingest first"
    print("Test passed — stock_prices.csv exists")


def test_stock_prices_has_all_symbols():
    """Test that all 5 symbols are present in the data."""
    df = pd.read_csv("data/stock_prices.csv")
    expected = {"AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"}
    actual   = set(df["symbol"].unique())
    assert expected == actual, f"Missing symbols: {expected - actual}"
    print(f"Test passed — all symbols present: {actual}")


def test_stock_prices_has_required_columns():
    """Test that all required columns exist."""
    df = pd.read_csv("data/stock_prices.csv")
    required = ["symbol", "date", "open", "high", "low", "close", "volume"]
    for col in required:
        assert col in df.columns, f"Missing column: {col}"
    print("Test passed — all required columns present")


def test_feature_engineering():
    """Test that feature engineering produces expected columns."""
    from pipeline.features import engineer_features
    df       = pd.read_csv("data/stock_prices.csv", parse_dates=["date"])
    features = engineer_features(df)

    expected_cols = [
        "daily_return", "daily_range", "ma_ratio",
        "volume_change", "momentum_5", "volatility_10",
        "gap", "target"
    ]
    for col in expected_cols:
        assert col in features.columns, f"Missing column: {col}"

    assert len(features) > 0, "No rows produced"
    print(f"Test passed — {len(features)} rows, all feature columns present")


def test_target_is_balanced():
    """Test that target is reasonably balanced — not all 0s or 1s."""
    from pipeline.features import engineer_features
    df       = pd.read_csv("data/stock_prices.csv", parse_dates=["date"])
    features = engineer_features(df)
    ratio    = features["target"].mean()
    assert 0.3 < ratio < 0.7, \
        f"Target severely imbalanced: {ratio:.2f} — check data"
    print(f"Test passed — target ratio: {ratio:.2f} (well balanced)")


def test_no_nulls_in_features():
    """Test that there are no null values in the feature columns."""
    from pipeline.features import engineer_features
    FEATURES = [
        "daily_return", "daily_range", "ma_ratio",
        "volume_change", "momentum_5", "volatility_10", "gap"
    ]
    df       = pd.read_csv("data/stock_prices.csv", parse_dates=["date"])
    features = engineer_features(df)
    nulls    = features[FEATURES].isnull().sum()
    assert nulls.sum() == 0, f"Null values found:\n{nulls[nulls > 0]}"
    print("Test passed — no null values in features")


def test_model_file_exists():
    """Test that the trained model file exists after training."""
    assert os.path.exists("models/RandomForest_100.pkl"), \
        "Model file not found — run train.py first"
    print("Test passed — model file exists")


def test_model_can_predict():
    """Test that the saved model loads and makes predictions."""
    import pickle
    import pandas as pd

    with open("models/RandomForest_100.pkl", "rb") as f:
        model = pickle.load(f)

    # Dummy input — same shape as training features
    X = pd.DataFrame([{
        "daily_return":  0.01,
        "daily_range":   0.02,
        "ma_ratio":      1.01,
        "volume_change": 0.05,
        "momentum_5":    0.03,
        "volatility_10": 5.0,
        "gap":           0.001,
    }])

    prediction = model.predict(X)
    assert prediction[0] in [0, 1], "Prediction must be 0 or 1"
    print(f"Test passed — model predicts: {prediction[0]}")