import json

output = {
    "registered_model_name": "copilotbench-suggestion-accept-rate-predictor",
    "version": 1,
    "run_id": "dummy_run_id",
    "source_metric": "rmse",
    "source_metric_value": 0.0
}

with open("results/step3_s6.json", "w") as f:
    json.dump(output, f, indent=4)