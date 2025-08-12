import os
import json
import pandas as pd
import numpy as np
import requests
from sklearn.metrics import accuracy_score, f1_score
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, DataQualityPreset, TargetDriftPreset, ClassificationPreset
from evidently import ColumnMapping

# Paths & constants
DATA_PATH = "./artifacts/test.csv"
OUTPUT_DIR = "./artifacts"
URL = f"http://{os.getenv('SERVE_HOST','127.0.0.1')}:{os.getenv('SERVE_PORT','5000')}/invocations"
REPORT_NAME = "drift_after"

# ---- Helpers ----
def post_invocations_csv(url, X):
    """
    Send dataframe to MLflow model server using CSV format.
    This matches what worked in batch_infer.py.
    """
    csv_bytes = X.to_csv(index=False).encode("utf-8")
    headers = {"Content-Type": "text/csv"}
    r = requests.post(url, headers=headers, data=csv_bytes)
    print("DEBUG status", r.status_code, "body", r.text[:200])
    r.raise_for_status()

    obj = r.json()
    # H2O pyfunc returns {"predictions": [{"predict": <int>, ...}, ...]}
    if isinstance(obj, dict) and "predictions" in obj:
        preds = [row["predict"] for row in obj["predictions"]]
    else:
        preds = obj
    return pd.Series(preds)

def mapping(df):
    """
    Build Evidently ColumnMapping for reports.
    """
    num_cols = [
        'fixed acidity','volatile acidity','citric acid','residual sugar','chlorides',
        'free sulfur dioxide','total sulfur dioxide','density','pH','sulphates','alcohol'
    ]
    num_cols = [c for c in num_cols if c in df.columns]
    cat_cols = [c for c in ['type'] if c in df.columns]
    return ColumnMapping(
        target='quality',
        prediction='prediction',
        numerical_features=num_cols,
        categorical_features=cat_cols
    )

# ---- Main ----
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load data
    orig = pd.read_csv(DATA_PATH).dropna()
    y_true = orig['quality'].astype(int)
    X = orig.drop(columns=['quality'])

    # Baseline predictions
    yhat_base = post_invocations_csv(URL, X).astype(int)

    # Perturb features
    Xp = X.copy()
    if 'alcohol' in Xp.columns:
        Xp['alcohol'] = Xp['alcohol'] - 1.2
    if 'volatile acidity' in Xp.columns:
        Xp['volatile acidity'] = Xp['volatile acidity'] + 0.1

    # Perturbed predictions
    yhat_curr = post_invocations_csv(URL, Xp).astype(int)

    # Compute metrics
    base_acc = accuracy_score(y_true, yhat_base)
    base_f1 = f1_score(y_true, yhat_base, average='weighted')
    curr_acc = accuracy_score(y_true, yhat_curr)
    curr_f1 = f1_score(y_true, yhat_curr, average='weighted')

    # Save metrics JSON
    results = {
        "baseline": {"accuracy": base_acc, "f1_weighted": base_f1},
        "perturbed": {"accuracy": curr_acc, "f1_weighted": curr_f1},
        "delta": {
            "accuracy": curr_acc - base_acc,
            "f1_weighted": curr_f1 - base_f1
        }
    }
    with open(os.path.join(OUTPUT_DIR, "perturb_test_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"Baseline:   Accuracy={base_acc:.4f}, F1_weighted={base_f1:.4f}")
    print(f"Perturbed:  Accuracy={curr_acc:.4f}, F1_weighted={curr_f1:.4f}")

    # Generate Evidently drift report
    report = Report(metrics=[
        DataDriftPreset(),
        DataQualityPreset(),
        TargetDriftPreset(),
        ClassificationPreset()
    ])
    df_base = X.copy()
    df_base['quality'] = y_true
    df_base['prediction'] = yhat_base

    df_curr = Xp.copy()
    df_curr['quality'] = y_true
    df_curr['prediction'] = yhat_curr

    report.run(
        reference_data=df_base,
        current_data=df_curr,
        column_mapping=mapping(orig)
    )

    report_path = os.path.join(OUTPUT_DIR, f"{REPORT_NAME}.html")
    report.save_html(report_path)
    print(f"Saved drift report → {report_path}")

if __name__ == "__main__":
    main()
