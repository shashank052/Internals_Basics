import pandas as pd
import mlflow
import mlflow.sklearn
import json
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor

# Load data
df = pd.read_csv("data/training_data.csv")

X = df.drop("suggestion_accept_rate", axis=1)
y = df["suggestion_accept_rate"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

mlflow.set_experiment("copilotbench-suggestion-accept-rate")

models = {
    "SVR": SVR(),
    "GradientBoosting": GradientBoostingRegressor(random_state=42)
}

results = []
best_rmse = float("inf")
best_model = None
best_model_name = ""

for name, model in models.items():
    with mlflow.start_run(run_name=name):
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        mae = mean_absolute_error(y_test, preds)
        rmse = mean_squared_error(y_test, preds, squared=False)
        r2 = r2_score(y_test, preds)

        mlflow.log_params(model.get_params())
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.set_tag("project_phase", "model_selection")

        mlflow.sklearn.log_model(model, name)

        results.append({
            "name": name,
            "mae": mae,
            "rmse": rmse,
            "r2": r2
        })

        if rmse < best_rmse:
            best_rmse = rmse
            best_model = model
            best_model_name = name

# Save model
os.makedirs("models", exist_ok=True)
joblib.dump(best_model, "models/best_model.pkl")

# Save JSON
os.makedirs("results", exist_ok=True)

output = {
    "experiment_name": "copilotbench-suggestion-accept-rate",
    "models": results,
    "best_model": best_model_name,
    "best_metric_name": "rmse",
    "best_metric_value": best_rmse
}

with open("results/step1_s1.json", "w") as f:
    json.dump(output, f, indent=4)