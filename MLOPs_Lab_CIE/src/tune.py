import pandas as pd
import json
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor

df = pd.read_csv("data/training_data.csv")

X = df.drop("suggestion_accept_rate", axis=1)
y = df["suggestion_accept_rate"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

param_grid = {
    "n_estimators": [50, 150],
    "learning_rate": [0.05, 0.1, 0.2],
    "max_depth": [3, 5, 10]
}

grid = GridSearchCV(
    GradientBoostingRegressor(random_state=42),
    param_grid,
    cv=3,
    scoring="neg_mean_squared_error"
)

grid.fit(X_train, y_train)

output = {
    "search_type": "grid",
    "n_folds": 3,
    "total_trials": len(grid.cv_results_["params"]),
    "best_params": grid.best_params_,
    "best_mae": -grid.best_score_,
    "best_cv_mae": -grid.best_score_,
    "parent_run_name": "tuning-copilotbench"
}

with open("results/step2_s2.json", "w") as f:
    json.dump(output, f, indent=4)