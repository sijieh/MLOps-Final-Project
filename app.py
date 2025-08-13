from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel
import pandas as pd
import requests, json, os
from typing import List, Dict

SERVE_HOST = os.getenv("SERVE_HOST", "127.0.0.1")
SERVE_PORT = os.getenv("SERVE_PORT", "5000")
PRED_URL   = f"http://{SERVE_HOST}:{SERVE_PORT}/invocations"
DATA_CSV   = "./data/winequalityN.csv"
REQUESTS_LOG = "./artifacts/requests.csv"

app = FastAPI(title="Wine Quality – Demo Dashboard")

FEATURES = ["type","fixed acidity","volatile acidity","citric acid","residual sugar",
            "chlorides","free sulfur dioxide","total sulfur dioxide","density","pH",
            "sulphates","alcohol"]  # target 'quality' excluded

class Record(BaseModel):
    type: str
    fixed_acidity: float
    volatile_acidity: float
    citric_acid: float
    residual_sugar: float
    chlorides: float
    free_sulfur_dioxide: float
    total_sulfur_dioxide: float
    density: float
    pH: float
    sulphates: float
    alcohol: float

def _to_dataframe(records: List[Record]) -> pd.DataFrame:
    df = pd.DataFrame([r.model_dump() for r in records])
    # match training column names (spaces & casing)
    df.columns = ["type","fixed acidity","volatile acidity","citric acid","residual sugar",
                  "chlorides","free sulfur dioxide","total sulfur dioxide","density","pH",
                  "sulphates","alcohol"]
    return df[FEATURES]

def _post_invocations(df: pd.DataFrame) -> List[int]:
    payload = {"dataframe_split": {"columns": list(df.columns), "data": df.values.tolist()}}
    r = requests.post(PRED_URL, headers={"Content-Type": "application/json"},
                      data=json.dumps(payload), timeout=30)
    if r.status_code != 200:
        raise HTTPException(status_code=502, detail=f"MLflow serve error: {r.text[:200]}")
    obj = r.json()
    # MLflow/H2O pyfunc returns {"predictions":[{"predict":int,...},...]} or {"predictions":[int,...]}
    if isinstance(obj, dict) and "predictions" in obj:
        preds = obj["predictions"]
        if preds and isinstance(preds[0], dict):
            return [row.get("predict") for row in preds]
        return [int(v) for v in preds]
    raise HTTPException(status_code=500, detail="Unexpected prediction schema")

def _log_rows(df: pd.DataFrame, preds: List[int]):
    os.makedirs(os.path.dirname(REQUESTS_LOG), exist_ok=True)
    df2 = df.copy()
    df2["prediction"] = preds
    mode = "a" if os.path.exists(REQUESTS_LOG) else "w"
    header = not os.path.exists(REQUESTS_LOG)
    df2.to_csv(REQUESTS_LOG, index=False, mode=mode, header=header)

@app.get("/health")
def health():
    try:
        # some builds have /ping, otherwise try a tiny bogus request to /invocations
        requests.get(f"http://{SERVE_HOST}:{SERVE_PORT}/ping", timeout=2)
        return {"status": "up", "mlflow": f"{SERVE_HOST}:{SERVE_PORT}"}
    except Exception:
        return {"status": "unknown", "mlflow": f"{SERVE_HOST}:{SERVE_PORT}"}

@app.get("/data-stats")
def data_stats():
    if not os.path.exists(DATA_CSV):
        raise HTTPException(404, "Dataset not found at ./data/winequalityN.csv")
    df = pd.read_csv(DATA_CSV)
    return {
        "rows": len(df),
        "cols": list(df.columns),
        "target": "quality",
        "type_counts": df["type"].value_counts().to_dict(),
        "quality_counts": df["quality"].value_counts().sort_index().to_dict(),
    }

@app.post("/predict")
def predict(records: List[Record]):
    df = _to_dataframe(records)
    preds = _post_invocations(df)
    _log_rows(df, preds)
    return {"predictions": preds}

@app.get("/download-logs")
def download_logs():
    if os.path.exists(REQUESTS_LOG):
        return FileResponse(REQUESTS_LOG, filename="requests.csv")
    raise HTTPException(404, "No logs yet")

@app.get("/", response_class=HTMLResponse)
def home():
    return """<html><body>
    <h2>Wine Demo</h2>
    <p>POST JSON to <code>/predict</code> or check <code>/health</code> and <code>/data-stats</code>.</p>
    <p>Example payload:</p>
    <pre>[
  {"type":"white","fixed_acidity":7.0,"volatile_acidity":0.27,"citric_acid":0.36,"residual_sugar":20.7,
   "chlorides":0.045,"free_sulfur_dioxide":45,"total_sulfur_dioxide":170,"density":1.001,"pH":3.0,
   "sulphates":0.45,"alcohol":8.8}
]</pre></body></html>"""
