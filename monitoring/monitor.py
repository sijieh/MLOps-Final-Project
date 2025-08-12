import os, json, pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, DataQualityPreset, TargetDriftPreset, ClassificationPreset
from evidently import ColumnMapping

DATA_PATH   = "./artifacts/test.csv"
OUTPUT_DIR  = "./artifacts"
PRED_FILE   = "./artifacts/preds.json"  # predictions from endpoint
BASE_NAME   = "baseline"                # baseline vs current = same data (no change)

os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_preds(path):
    with open(path) as f: return pd.Series(json.load(f)["pred"])

def mapping(df):
    num_cols = ['fixed acidity','volatile acidity','citric acid','residual sugar','chlorides',
                'free sulfur dioxide','total sulfur dioxide','density','pH','sulphates','alcohol']
    num_cols = [c for c in num_cols if c in df.columns]
    cat_cols = [c for c in ['type'] if c in df.columns]
    return ColumnMapping(target='quality', prediction='prediction',
                         numerical_features=num_cols, categorical_features=cat_cols)

def main():
    data = pd.read_csv(DATA_PATH).dropna()
    data_base = data.copy()
    data_curr = data.copy()

    # add endpoint predictions
    preds = load_preds(PRED_FILE).astype(int)
    data_base["prediction"] = preds
    data_curr["prediction"] = preds

    report = Report(metrics=[DataDriftPreset(), DataQualityPreset(), TargetDriftPreset(), ClassificationPreset()])
    report.run(reference_data=data_base, current_data=data_curr, column_mapping=mapping(data))
    report.save_html(f"{OUTPUT_DIR}/{BASE_NAME}.html")
    report.save_json(f"{OUTPUT_DIR}/{BASE_NAME}.json")
    print("Saved baseline report →", f"{OUTPUT_DIR}/{BASE_NAME}.html")

if __name__ == "__main__":
    main()
