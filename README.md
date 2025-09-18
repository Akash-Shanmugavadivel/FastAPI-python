# FastAPI RandomForest Prediction API

This project demonstrates a simple FastAPI web service for making predictions using a trained RandomForest model on the Iris dataset. It includes training, serving, and client code.

## Project Structure

```
ml_model/
│
├── train.py        # Train and save the RandomForest model
├── main.py         # FastAPI app for serving predictions
├── client.py       # Example client for making prediction requests
└── models/
    └── rf_pipeline.joblib  # Saved model (created by train.py)
```

## Setup

1. **Install dependencies**  
   Make sure you have Python 3.8+ and pip installed.  
   Install required packages:
   ```powershell
   pip install fastapi uvicorn scikit-learn pandas joblib requests
   ```

2. **Train the model**  
   Run the training script to create the model file:
   ```powershell
   python ml_model/train.py
   ```

3. **Start the API server**  
   Run the FastAPI app using Uvicorn:
   ```powershell
   uvicorn ml_model.main:app --reload --port 8000
   ```

## Usage

### Health Check

Check if the API is running:
```
GET http://127.0.0.1:8000/health
```

### Single Prediction

Send a POST request to `/predict` with feature values:
```json
POST http://127.0.0.1:8000/predict
{
  "features": {
    "sepal_length": 5.1,
    "sepal_width": 3.5,
    "petal_length": 1.4,
    "petal_width": 0.2
  }
}
```

### Batch Prediction

Send a POST request to `/predict_batch` with multiple instances:
```json
POST http://127.0.0.1:8000/predict_batch
{
  "instances": [
    {"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2},
    {"sepal_length": 6.2, "sepal_width": 2.8, "petal_length": 4.8, "petal_width": 1.8}
  ]
}
```

### Example Client

Run the provided client script to make a prediction:
```powershell
python ml_model/client.py
```

## Notes

- The model is trained on the Iris dataset and expects four features: `sepal_length`, `sepal_width`, `petal_length`, `petal_width`.
- The API returns predicted class, class index, and class probabilities.

## License

MIT License
