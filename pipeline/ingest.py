import os
import time
import requests
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

SYMBOLS  = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
API_KEY  = os.environ.get("ALPHA_VANTAGE_API_KEY")
DELAY_S  = 15   # free tier: max 5 calls/min → 1 per 15s to be safe


def fetch_daily_prices(symbol: str) -> pd.DataFrame:
    """Fetch last 100 days of daily prices — free tier compatible."""
    url    = "https://www.alphavantage.co/query"
    params = {
        "function":   "TIME_SERIES_DAILY",
        "symbol":     symbol,
        "outputsize": "compact",   # last 100 days — free tier
        "apikey":     API_KEY,
    }

    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    data = response.json()

    # Handle API-level errors
    if "Time Series (Daily)" not in data:
        note = data.get(
            "Note",
            data.get("Information", data.get("Error Message", str(data)))
        )
        raise ValueError(note)

    records = []
    for date, values in data["Time Series (Daily)"].items():
        records.append({
            "symbol": symbol,
            "date":   date,
            "open":   float(values["1. open"]),
            "high":   float(values["2. high"]),
            "low":    float(values["3. low"]),
            "close":  float(values["4. close"]),
            "volume": int(values["5. volume"]),
        })

    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df


def run_ingest() -> pd.DataFrame:
    """
    Fetch last 100 days of prices for all symbols.
    Overwrites data/stock_prices.csv each run (free tier limitation —
    compact output only gives last 100 days).
    """
    if not API_KEY:
        raise ValueError(
            "ALPHA_VANTAGE_API_KEY not set — add it to your .env file"
        )

    print("Ingesting stock data...")
    print(f"  Symbols:  {SYMBOLS}")
    print(f"  API tier: free (compact — last 100 days)")
    print(f"  Delay:    {DELAY_S}s between calls (rate limit)")
    print()

    all_data    = []
    failed      = []

    for i, symbol in enumerate(SYMBOLS):
        print(f"  [{i+1}/{len(SYMBOLS)}] Fetching {symbol}...", end=" ")
        try:
            df = fetch_daily_prices(symbol)
            all_data.append(df)
            print(f"✓ {len(df)} days "
                  f"({df['date'].min().date()} → "
                  f"{df['date'].max().date()})")
        except Exception as e:
            print(f"✗ Failed: {e}")
            failed.append(symbol)

        # Rate limit — wait between every call except the last
        if i < len(SYMBOLS) - 1:
            print(f"  Waiting {DELAY_S}s...", end="\r")
            time.sleep(DELAY_S)
            print(" " * 20, end="\r")  # clear the waiting message

    if not all_data:
        raise RuntimeError(
            "No data fetched. Check your API key and daily limit "
            "(25 calls/day on free tier)."
        )

    # Combine and save
    combined = pd.concat(all_data, ignore_index=True)
    os.makedirs("data", exist_ok=True)
    combined.to_csv("data/stock_prices.csv", index=False)

    print()
    print(f"  ✓ Saved {len(combined)} rows → data/stock_prices.csv")
    print(f"  ✓ Symbols fetched: {combined['symbol'].unique().tolist()}")
    if failed:
        print(f"  ⚠ Failed symbols: {failed}")
    print()

    return combined


if __name__ == "__main__":
    run_ingest()