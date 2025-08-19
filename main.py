from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, conint, PositiveInt
import joblib
import traceback
from typing import Optional
import os

# Define FastAPI app
app = FastAPI(title="House Price Prediction API")

# --- Load model on startup ---
MODEL_PATH = "models/model_selected_features.pkl"

try:
    model = joblib.load(MODEL_PATH)
    model_loaded = True
except Exception as e:
    print(f"Failed to load model: {e}")
    model_loaded = False


# --- Pydantic model for input ---
class HouseFeatures(BaseModel):
    area: PositiveInt = Field(..., example=2000, description="Square footage")
    bedrooms: conint(ge=1, le=10) = Field(..., example=3)
    bathrooms: conint(ge=1, le=10) = Field(..., example=2)
    stories: conint(ge=1, le=4) = Field(..., example=2)


# --- Pydantic model for response ---
class PredictionResponse(BaseModel):
    predicted_price: float
    model: str


# --- Root Health Check ---
@app.get("/")
def read_root():
    return {"status": "OK", "message": "House Price Prediction API is live."}


# --- Model Info ---
@app.get("/model-info")
def model_info():
    if not model_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {
        "model_path": MODEL_PATH,
        "model_type": type(model).__name__,
        "features": ["area", "bedrooms", "bathrooms", "stories"]
    }


# --- Predict Endpoint ---
@app.post("/predict", response_model=PredictionResponse)
def predict_price(features: HouseFeatures):
    if not model_loaded:
        raise HTTPException(status_code=503, detail="Model not available")

    try:
        input_data = [[
            features.area,
            features.bedrooms,
            features.bathrooms,
            features.stories
        ]]

        prediction = model.predict(input_data)[0]

        return {
            "predicted_price": round(float(prediction), 2),
            "model": os.path.basename(MODEL_PATH)
        }

    except Exception as e:
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail="Prediction failed")

