import os
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
import pickle
from dotenv import load_dotenv

load_dotenv()

FEATURES = [
    "daily_return", "daily_range", "ma_ratio",
    "volume_change", "momentum_5", "volatility_10", "gap"
]
TARGET = "target"


def load_data() -> tuple:
    """Load features and split into train/test."""
    df = pd.read_csv("data/features.csv", parse_dates=["date"])

    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=True
    )
    print(f"  Train: {len(X_train)} rows | Test: {len(X_test)} rows")
    return X_train, X_test, y_train, y_test


def evaluate(model, X_test, y_test) -> dict:
    """Calculate all metrics for a trained model."""
    y_pred = model.predict(X_test)
    return {
        "accuracy":  round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred), 4),
        "recall":    round(recall_score(y_test, y_pred), 4),
        "f1":        round(f1_score(y_test, y_pred), 4),
    }


def train_and_log(model, model_name: str, params: dict,
                  X_train, X_test, y_train, y_test):
    """Train one model and log everything to MLflow."""

    with mlflow.start_run(run_name=model_name):

        # Log parameters
        mlflow.log_params(params)
        mlflow.log_param("model_type", model_name)
        mlflow.log_param("train_size", len(X_train))
        mlflow.log_param("test_size", len(X_test))

        # Train
        model.fit(X_train, y_train)

        # Evaluate
        metrics = evaluate(model, X_test, y_test)

        # Log metrics
        mlflow.log_metrics(metrics)

        # Log the model itself
        mlflow.sklearn.log_model(model, "model")

        # Save locally as well
        os.makedirs("models", exist_ok=True)
        with open(f"models/{model_name}.pkl", "wb") as f:
            pickle.dump(model, f)

        print(f"  {model_name:30s} | "
              f"accuracy={metrics['accuracy']:.4f} | "
              f"f1={metrics['f1']:.4f}")

        return metrics["accuracy"], mlflow.active_run().info.run_id


def run_training():
    """Train multiple models, log all to MLflow, register the best."""
    print("Starting experiment tracking with MLflow...")
    print()

    mlflow.set_experiment("stock-price-direction")

    X_train, X_test, y_train, y_test = load_data()
    print()

    # Scale features for Logistic Regression
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    # ── Three models to compare ──────────────────────────────
    experiments = [
        (
            LogisticRegression(max_iter=1000),
            "LogisticRegression",
            {"max_iter": 1000},
            X_train_scaled, X_test_scaled
        ),
        (
            RandomForestClassifier(n_estimators=100, random_state=42),
            "RandomForest_100",
            {"n_estimators": 100, "random_state": 42},
            X_train, X_test
        ),
        (
            RandomForestClassifier(n_estimators=200, max_depth=5, random_state=42),
            "RandomForest_200_depth5",
            {"n_estimators": 200, "max_depth": 5, "random_state": 42},
            X_train, X_test
        ),
    ]

    print("Training models and logging to MLflow:")
    best_accuracy = 0
    best_run_id   = None
    best_name     = None

    for model, name, params, X_tr, X_te in experiments:
        accuracy, run_id = train_and_log(
            model, name, params,
            X_tr, X_te, y_train, y_test
        )
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_run_id   = run_id
            best_name     = name

    print()
    print(f"Best model: {best_name}")
    print(f"Best accuracy: {best_accuracy:.4f}")
    print(f"Best run ID: {best_run_id}")
    print()
    print("View all experiments:")
    print("  mlflow ui")
    print("  Then open http://localhost:5000")

    return best_run_id, best_accuracy


if __name__ == "__main__":
    run_training()