# main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, List, Any
import joblib
import numpy as np
import pandas as pd

app = FastAPI(title="RandomForest Prediction API")

MODEL_PATH = "models/rf_pipeline.joblib"

# Pydantic models
class SingleRequest(BaseModel):
    features: Dict[str, float]   # {"sepal_length": 5.1, ...}

class BatchRequest(BaseModel):
    instances: List[Dict[str, float]]

class SingleResponse(BaseModel):
    prediction: str
    prediction_index: int
    probabilities: Dict[str, float]

class BatchResponse(BaseModel):
    predictions: List[str]
    prediction_indices: List[int]
    probabilities: List[Dict[str, float]]

# Globals filled at startup
model = None
feature_names = None
class_names = None

@app.on_event("startup")
def load_model():
    global model, feature_names, class_names
    saved = joblib.load(MODEL_PATH)
    model = saved["model"]
    feature_names = saved["feature_names"]
    class_names = saved.get("class_names")
    # basic sanity check
    if not hasattr(model, "predict"):
        raise RuntimeError("Loaded object is not a model with predict()")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict", response_model=SingleResponse)
def predict(req: SingleRequest):
    # ensure keys present
    try:
        row = [req.features[name] for name in feature_names]
    except KeyError as e:
        raise HTTPException(status_code=422, detail=f"Missing feature: {e.args[0]}")
    arr = np.array(row).reshape(1, -1)
    pred_index = int(model.predict(arr)[0])
    probs = model.predict_proba(arr)[0].tolist()
    # map probabilities to class names if available
    if class_names:
        prob_dict = {class_names[i]: float(probs[i]) for i in range(len(probs))}
        pred_label = class_names[pred_index]
    else:
        prob_dict = {str(i): float(p) for i, p in enumerate(probs)}
        pred_label = str(pred_index)
    return {"prediction": pred_label, "prediction_index": pred_index, "probabilities": prob_dict}

@app.post("/predict_batch", response_model=BatchResponse)
def predict_batch(req: BatchRequest):
    df = pd.DataFrame(req.instances)
    # enforce column order and presence:
    missing = [c for c in feature_names if c not in df.columns]
    if missing:
        raise HTTPException(status_code=422, detail=f"Missing columns: {missing}")
    df = df[feature_names]
    preds = model.predict(df)
    probs = model.predict_proba(df)
    if class_names:
        prob_list = [{class_names[i]: float(prob[i]) for i in range(len(prob))} for prob in probs]
        pred_labels = [class_names[int(p)] for p in preds]
    else:
        prob_list = [{str(i): float(prob[i]) for i in range(len(prob))} for prob in probs]
        pred_labels = [str(int(p)) for p in preds]
    return {"predictions": pred_labels, "prediction_indices": [int(p) for p in preds], "probabilities": prob_list}

# run with: uvicorn main:app --reload --port 8000
