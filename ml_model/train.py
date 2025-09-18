# train.py
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import pandas as pd
import joblib
import os

data = load_iris()
# set friendly feature names so JSON keys are easy
feature_names = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
X = pd.DataFrame(data.data, columns=feature_names)
y = data.target
class_names = data.target_names.tolist()

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("rf", RandomForestClassifier(n_estimators=100, random_state=42))
])

pipeline.fit(X, y)

os.makedirs("models", exist_ok=True)
joblib.dump({
    "model": pipeline,
    "feature_names": feature_names,
    "class_names": class_names
}, "models/rf_pipeline.joblib")

print("Saved model to models/rf_pipeline.joblib")
