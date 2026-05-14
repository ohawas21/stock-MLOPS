import pandas as pd
import numpy as np
from evidently import Report
from evidently.presets import DataDriftPreset
import os
from datetime import datetime

FEATURES = [
    "daily_return", "daily_range", "ma_ratio",
    "volume_change", "momentum_5", "volatility_10", "gap"
]

DRIFT_THRESHOLD = 0.5  # alert if more than 50% of features drift


def load_reference_data() -> pd.DataFrame:
    """Training data — the baseline the model knows."""
    df = pd.read_csv("data/features.csv", parse_dates=["date"])
    cutoff = int(len(df) * 0.8)
    return df[:cutoff][FEATURES]


def load_current_data() -> pd.DataFrame:
    """Most recent data — what is arriving in production."""
    df = pd.read_csv("data/features.csv", parse_dates=["date"])
    cutoff = int(len(df) * 0.8)
    return df[cutoff:][FEATURES]


def run_drift_detection() -> dict:
    """
    Compare reference data (training) vs current data (production).
    Raises alert if significant drift detected.
    """
    print("Running drift detection...")
    print()

    reference = load_reference_data()
    current   = load_current_data()

    print(f"  Reference data: {len(reference)} rows (training period)")
    print(f"  Current data:   {len(current)} rows (recent production)")
    print()

    # Run Evidently drift report — new API
    report = Report([
        DataDriftPreset()
    ])

    my_eval = report.run(reference, current)

    # Extract results from new API format
    result_dict = my_eval.dict()

    # Count drifted features manually from results
    drifted_features = []
    feature_results  = {}

    try:
        # Navigate the new result structure
        for metric_result in result_dict.get("metrics", []):
            if "column_drift" in str(metric_result).lower():
                pass  # handled below

        # Simpler approach — use the summary
        summary = result_dict["metrics"][0]["value"]
        n_drifted = summary.get("number_of_drifted_columns", 0)
        n_total   = summary.get("number_of_columns", len(FEATURES))
        drift_ratio = n_drifted / n_total if n_total > 0 else 0

        # Per-column results
        col_results = summary.get("drift_by_columns", {})
        for col, stats in col_results.items():
            drifted = stats.get("drift_detected", False)
            score   = round(stats.get("drift_score", 0), 4)
            feature_results[col] = {
                "drift_detected": drifted,
                "drift_score":    score,
            }
            if drifted:
                drifted_features.append(col)

    except Exception:
        # Fallback — compute drift manually with scipy
        from scipy import stats as scipy_stats
        drift_ratio = 0
        for feature in FEATURES:
            stat, p_value = scipy_stats.ks_2samp(
                reference[feature].dropna(),
                current[feature].dropna()
            )
            drifted = p_value < 0.05
            feature_results[feature] = {
                "drift_detected": drifted,
                "drift_score":    round(stat, 4),
            }
            if drifted:
                drifted_features.append(feature)
        drift_ratio = len(drifted_features) / len(FEATURES)

    overall_drift = drift_ratio >= DRIFT_THRESHOLD

    # Print results
    print(f"  {'Feature':<20} {'Status':<12} {'Score'}")
    print(f"  {'-'*45}")
    for feature, stats in feature_results.items():
        status = "⚠  DRIFT" if stats["drift_detected"] else "✓  OK"
        print(f"  {feature:<20} {status:<12} {stats['drift_score']}")

    print()
    print(f"  Drifted features: {len(drifted_features)}/{len(FEATURES)}")
    print(f"  Drift ratio:      {drift_ratio:.1%}")

    if overall_drift:
        print()
        print("  ⚠  DRIFT ALERT — model retraining recommended")
        print(f"  Features that drifted: {drifted_features}")
    else:
        print()
        print("  ✓  No significant drift — model is stable")

    # Save HTML report
    os.makedirs("reports", exist_ok=True)
    timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = f"reports/drift_{timestamp}.html"
    my_eval.save_html(report_path)
    print(f"\n  Full HTML report → {report_path}")

    return {
        "overall_drift":    overall_drift,
        "drift_ratio":      drift_ratio,
        "drifted_features": drifted_features,
        "feature_results":  feature_results,
        "report_path":      report_path,
    }


if __name__ == "__main__":
    result = run_drift_detection()
    print()
    if result["overall_drift"]:
        print("ACTION REQUIRED: retrain the model with fresh data")
        exit(1)
    else:
        print("No action needed — continue monitoring")
        exit(0)