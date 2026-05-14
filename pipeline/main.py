import os
import sys
from dotenv import load_dotenv

load_dotenv()

def run_pipeline():
    """Run the complete MLOps pipeline end to end."""

    print("=" * 55)
    print("  STOCK MLOPS PIPELINE")
    print("=" * 55)
    print()

    # ── STEP 1 — INGEST ──────────────────────────────────
    print("STEP 1 — Ingest fresh stock data")
    try:
        from pipeline.ingest import run_ingest
        run_ingest()
    except Exception as e:
        print(f"  ✗ Ingest failed: {e}")
        sys.exit(1)

    # ── STEP 2 — FEATURES ────────────────────────────────
    print("STEP 2 — Engineer features")
    try:
        from pipeline.features import engineer_features
        import pandas as pd
        df = pd.read_csv("data/stock_prices.csv", parse_dates=["date"])
        features = engineer_features(df)
        features.to_csv("data/features.csv", index=False)
        print(f"  ✓ {len(features)} rows saved to data/features.csv\n")
    except Exception as e:
        print(f"  ✗ Feature engineering failed: {e}")
        sys.exit(1)

    # ── STEP 3 — DRIFT DETECTION ─────────────────────────
    print("STEP 3 — Drift detection")
    try:
        from pipeline.monitor import run_drift_detection
        result = run_drift_detection()
        if result["overall_drift"]:
            print("  ⚠ Drift detected — retraining is critical\n")
        else:
            print("  ✓ No drift — retraining as scheduled\n")
    except Exception as e:
        print(f"  ⚠ Drift detection failed (non-critical): {e}\n")

    # ── STEP 4 — TRAIN ───────────────────────────────────
    print("STEP 4 — Train models + log to MLflow")
    try:
        from pipeline.train import run_training
        best_run_id, best_accuracy = run_training()
        print(f"  ✓ Best accuracy: {best_accuracy:.4f}")
        print(f"  ✓ Best run ID:   {best_run_id}\n")
    except Exception as e:
        print(f"  ✗ Training failed: {e}")
        sys.exit(1)

    # ── DONE ─────────────────────────────────────────────
    print("=" * 55)
    print("  PIPELINE COMPLETE")
    print("=" * 55)
    print()
    print("Next steps:")
    print("  View experiments: mlflow ui → http://localhost:5000")
    print("  Serve model:      python pipeline/serve.py")
    print("  Check drift:      python pipeline/monitor.py")


if __name__ == "__main__":
    run_pipeline()