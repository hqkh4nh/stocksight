import json
import pandas as pd

from src.config import RESULTS_DIR

SUMMARY_CSV = RESULTS_DIR / "summary.csv"

def append_result(ticker, model_name, task, metrics):
    row = {"ticker": ticker, "model": model_name, "task": task}
    row.update({k: v for k, v in metrics.items() if isinstance(v, (int, float))})

    if SUMMARY_CSV.exists():
        df = pd.concat([pd.read_csv(SUMMARY_CSV), pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])
    df.to_csv(SUMMARY_CSV, index=False)

def save_feature_importances(ticker, model_name, importances):
    path = RESULTS_DIR / f"{ticker}_{model_name}_importances.json"
    with open(path, "w") as f:
        json.dump(importances, f, indent=2)

def reset_summary():
    if SUMMARY_CSV.exists():
        SUMMARY_CSV.unlink()
