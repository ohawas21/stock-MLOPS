import os
import pickle
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

app = FastAPI(
    title="Stock Price Direction Predictor",
    description="Predicts whether a stock will close higher or lower tomorrow",
    version="1.0.0"
)

# Load the best model at startup
MODEL_PATH = "models/RandomForest_100.pkl"
model = None

@app.on_event("startup")
def load_model():
    global model
    if not os.path.exists(MODEL_PATH):
        print(f"Warning: model not found at {MODEL_PATH}")
        return
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    print(f"Model loaded from {MODEL_PATH}")


class StockFeatures(BaseModel):
    """Input features for prediction."""
    daily_return:  float
    daily_range:   float
    ma_ratio:      float
    volume_change: float
    momentum_5:    float
    volatility_10: float
    gap:           float


class Prediction(BaseModel):
    """Prediction output."""
    symbol:        str
    prediction:    int
    direction:     str
    confidence:    float
    model_version: str


@app.get("/")
def root():
    return {"status": "running", "model": MODEL_PATH}


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None
    }


@app.post("/predict", response_model=Prediction)
def predict(symbol: str, features: StockFeatures):
    """
    Predict whether a stock will close higher or lower tomorrow.
    Returns 1 (up) or 0 (down) with confidence score.
    """
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded"
        )

    # Build feature dataframe
    X = pd.DataFrame([{
        "daily_return":  features.daily_return,
        "daily_range":   features.daily_range,
        "ma_ratio":      features.ma_ratio,
        "volume_change": features.volume_change,
        "momentum_5":    features.momentum_5,
        "volatility_10": features.volatility_10,
        "gap":           features.gap,
    }])

    # Predict
    prediction = int(model.predict(X)[0])
    probabilities = model.predict_proba(X)[0]
    confidence = round(float(probabilities[prediction]), 4)

    return Prediction(
        symbol=symbol,
        prediction=prediction,
        direction="UP ↑" if prediction == 1 else "DOWN ↓",
        confidence=confidence,
        model_version="RandomForest_100_v1"
    )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)