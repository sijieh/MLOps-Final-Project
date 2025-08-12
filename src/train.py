import os
import argparse
import json
import mlflow
import mlflow.h2o
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from dotenv import load_dotenv
import h2o
from h2o.automl import H2OAutoML

load_dotenv()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=os.getenv("DATA_PATH", "./data/winequalityN.csv"))
    parser.add_argument("--target", default=os.getenv("TARGET", "quality"))
    parser.add_argument("--test_size", type=float, default=float(os.getenv("TEST_SIZE", 0.2)))
    parser.add_argument("--random_state", type=int, default=int(os.getenv("RANDOM_STATE", 42)))
    parser.add_argument("--max_runtime_secs", type=int, default=120)  # tweak for speed
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs("artifacts", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "./models/mlruns")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT_NAME", "wine-quality-automl"))

    # load data
    df = pd.read_csv(args.data).dropna()
    # ensure target is int (H2O multinomial classification)
    df[args.target] = df[args.target].astype(int)

    # split
    train_df, test_df = train_test_split(df, test_size=args.test_size, random_state=args.random_state, stratify=df[args.target])
    train_df.to_csv("artifacts/train.csv", index=False)
    test_df.to_csv("artifacts/test.csv", index=False)

    # H2O init
    h2o.init()
    train_h2o = h2o.H2OFrame(train_df)
    test_h2o  = h2o.H2OFrame(test_df)

    # set types
    if "type" in train_h2o.columns:
        train_h2o["type"] = train_h2o["type"].asfactor()
        test_h2o["type"] = test_h2o["type"].asfactor()
    train_h2o[args.target] = train_h2o[args.target].asfactor()
    test_h2o[args.target] = test_h2o[args.target].asfactor()

    x = [c for c in train_h2o.columns if c != args.target]
    y = args.target

    with mlflow.start_run() as run:
        run_id = run.info.run_id
        mlflow.log_params({
            "test_size": args.test_size,
            "random_state": args.random_state,
            "max_runtime_secs": args.max_runtime_secs,
            "seed": args.seed
        })

        # AutoML
        aml = H2OAutoML(max_runtime_secs=args.max_runtime_secs, seed=args.seed, balance_classes=False, sort_metric="logloss")
        aml.train(x=x, y=y, training_frame=train_h2o, leaderboard_frame=test_h2o)

        best = aml.leader
        print("Best model:", best.model_id)

        # offline metrics on test
        preds = best.predict(test_h2o).as_data_frame()["predict"].astype(int)
        y_true = test_df[args.target].astype(int)
        acc = accuracy_score(y_true, preds)
        f1  = f1_score(y_true, preds, average="weighted")
        mlflow.log_metrics({"accuracy": acc, "f1_weighted": f1})

        # log model (MLflow’s H2O flavor)
        mlflow.h2o.log_model(best, artifact_path="model")
        info = {
            "best_model_id": best.model_id,
            "run_id": run_id,
            "metrics": {"accuracy": acc, "f1_weighted": f1}
        }
        with open("models/h2o_model_info.json", "w") as f:
            json.dump(info, f, indent=2)
        print("Saved:", "models/h2o_model_info.json")

        print("\n=== Training done ===")
        print("MLflow run_id:", run_id)
        print("Tracking URI:", tracking_uri)

if __name__ == "__main__":
    main()
