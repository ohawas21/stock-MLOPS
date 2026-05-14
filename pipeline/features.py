import pandas as pd
import numpy as np

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Turn raw OHLCV price data into ML features.
    Target: will tomorrow's close be HIGHER than today's close?
    1 = price went up, 0 = price went down
    """
    all_symbols = []

    for symbol in df["symbol"].unique():
        s = df[df["symbol"] == symbol].copy()
        s = s.sort_values("date").reset_index(drop=True)

        # ── FEATURES ──────────────────────────────────────
        # 1. Daily return — how much did price change today
        s["daily_return"] = (s["close"] - s["open"]) / s["open"]

        # 2. Price range — how volatile was today
        s["daily_range"] = (s["high"] - s["low"]) / s["open"]

        # 3. Moving averages — trend over 5 and 20 days
        s["ma_5"]  = s["close"].rolling(5).mean()
        s["ma_20"] = s["close"].rolling(20).mean()

        # 4. MA ratio — is price above or below trend
        s["ma_ratio"] = s["ma_5"] / s["ma_20"]

        # 5. Volume change — unusual activity today
        s["volume_change"] = s["volume"].pct_change()

        # 6. Momentum — price change over last 5 days
        s["momentum_5"] = s["close"].pct_change(5)

        # 7. Volatility — standard deviation over 10 days
        s["volatility_10"] = s["close"].rolling(10).std()

        # 8. Gap — did the stock open higher/lower than yesterday's close
        s["gap"] = (s["open"] - s["close"].shift(1)) / s["close"].shift(1)

        # ── TARGET ────────────────────────────────────────
        # Will tomorrow's close be higher than today's?
        s["target"] = (s["close"].shift(-1) > s["close"]).astype(int)

        all_symbols.append(s)

    result = pd.concat(all_symbols, ignore_index=True)

    # Drop rows with NaN (from rolling windows and shift)
    result = result.dropna().reset_index(drop=True)

    print(f"  Features engineered: {len(result)} rows")
    print(f"  Feature columns: daily_return, daily_range, ma_5, ma_20, "
          f"ma_ratio, volume_change, momentum_5, volatility_10, gap")
    print(f"  Target distribution: "
          f"{result['target'].value_counts().to_dict()}")

    return result


if __name__ == "__main__":
    df = pd.read_csv("data/stock_prices.csv", parse_dates=["date"])
    features = engineer_features(df)
    features.to_csv("data/features.csv", index=False)
    print("\nSaved to data/features.csv")
    print(features[["symbol","date","close","target"]].tail(10).to_string())