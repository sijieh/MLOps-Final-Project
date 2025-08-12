import os, json, argparse, requests, pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from dotenv import load_dotenv
load_dotenv()

def post_invocations(url, X):
    payload = {"dataframe_split": {"columns": list(X.columns), "data": X.values.tolist()}}
    headers = {"Content-Type": "application/json"}
    r = requests.post(url, headers=headers, data=json.dumps(payload))
    print("DEBUG status", r.status_code, "body", r.text[:200])
    r.raise_for_status()

    obj = r.json()
    # MLflow H2O pyfunc returns {"predictions": [ { "predict": <int>, "p3":..., ... }, ... ]}
    if isinstance(obj, dict) and "predictions" in obj:
        preds = [row["predict"] for row in obj["predictions"]]
    else:
        # Fallback: plain list case
        preds = obj
    return pd.Series(preds)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="artifacts/test.csv")
    p.add_argument("--url", default=f"http://{os.getenv('SERVE_HOST','127.0.0.1')}:{os.getenv('SERVE_PORT','5000')}/invocations")
    p.add_argument("--out_preds", default="artifacts/preds.json")
    p.add_argument("--out_metrics", default="artifacts/metrics.json")
    p.add_argument("--target", default=os.getenv("TARGET","quality"))
    args = p.parse_args()

    df = pd.read_csv(args.input)
    y_true = df[args.target].astype(int)
    X = df.drop(columns=[args.target])  # IMPORTANT: do not send the target

    yhat = post_invocations(args.url, X).astype(int)
    acc = accuracy_score(y_true, yhat)
    f1  = f1_score(y_true, yhat, average="weighted")

    os.makedirs("artifacts", exist_ok=True)
    with open(args.out_preds, "w") as f: json.dump({"pred": yhat.tolist()}, f)
    with open(args.out_metrics, "w") as f: json.dump({"accuracy":acc, "f1_weighted":f1}, f, indent=2)

    print(f"Accuracy={acc:.4f}  F1_weighted={f1:.4f}")
    print("Saved → artifacts/preds.json, artifacts/metrics.json")

if __name__ == "__main__":
    main()
