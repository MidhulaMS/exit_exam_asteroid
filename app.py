from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import numpy as np
import pickle
import json
import uvicorn
from tensorflow.keras.models import load_model
import os

app = FastAPI(title="Asteroid Predictor")

# Mount static files if needed, or just serve a template
# app.mount("/static", StaticFiles(directory="static"), name="static")

# Load model, scaler, features
MODEL_PATH = "model.h5"
SCALER_PATH = "scaler.pkl"
FEATURES_PATH = "features.json"
MEDIANS_PATH = "medians.json"

if os.path.exists(MODEL_PATH):
    model = load_model(MODEL_PATH)
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)
    with open(FEATURES_PATH, "r") as f:
        features_list = json.load(f)
    with open(MEDIANS_PATH, "r") as f:
        medians_dict = json.load(f)
else:
    model, scaler, features_list, medians_dict = None, None, [], {}

class PredictionRequest(BaseModel):
    H: float
    a: float
    e: float
    i: float
    neo: int
    pha: int
    q: float
    n: float

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    with open("index.html", "r", encoding="utf-8") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content)

@app.post("/predict")
async def predict(data: dict):
    if not model:
        return {"error": "Model not loaded"}
    
    # Create an input array with median values for all features
    input_data = np.zeros((1, len(features_list)))
    
    # Fill in the user-provided features, leave rest as medians
    for i, feature in enumerate(features_list):
        if feature in data:
            input_data[0, i] = float(data[feature])
        else:
            input_data[0, i] = float(medians_dict.get(feature, 0.0))
            
    # Scale input
    input_scaled = scaler.transform(input_data)
    
    # Predict
    pred = model.predict(input_scaled)
    
    return {"diameter": float(pred[0][0])}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
