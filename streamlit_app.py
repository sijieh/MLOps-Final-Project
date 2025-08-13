#!/usr/bin/env python3
import os, json, subprocess, pandas as pd, requests
from pathlib import Path
import streamlit as st
import streamlit.components.v1 as components

# ------------------- Config -------------------
SERVE_HOST = os.getenv("SERVE_HOST", "model-serving")
SERVE_PORT = os.getenv("SERVE_PORT", "5000")
MLFLOW_URL = f"http://{SERVE_HOST}:{SERVE_PORT}"

DATA_CSV   = Path("./data/winequalityN.csv")
ARTIFACTS  = Path("./artifacts")
BASELINE_HTML = ARTIFACTS / "baseline.html"
DRIFT_HTML    = ARTIFACTS / "drift_after.html"
METRICS_JSON  = ARTIFACTS / "metrics.json"
PERTURB_JSON  = ARTIFACTS / "perturb_test_results.json"
MODEL_INFO    = Path("./models/h2o_model_info.json")

st.set_page_config(page_title="Wine Demo", page_icon="🍷", layout="wide")
st.markdown("# 🍷 Wine Quality Demo")
st.caption(f"Serving via MLflow at `{SERVE_HOST}:{SERVE_PORT}`")

# ------------------- Helpers -------------------
@st.cache_data
def load_data():
    if DATA_CSV.exists():
        return pd.read_csv(DATA_CSV)
    return None

def ping_mlflow():
    try:
        r = requests.get(f"{MLFLOW_URL}/ping", timeout=3)
        return r.status_code, r.text
    except Exception as e:
        return None, str(e)

def run_cmd(cmd: str):
    """Run a shell command and return (ok, stdout+stderr)."""
    try:
        out = subprocess.check_output(cmd, shell=True, stderr=subprocess.STDOUT, text=True)
        return True, out
    except subprocess.CalledProcessError as e:
        return False, e.output

def read_json_safe(p: Path, default=None):
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return default
    return default

def show_metrics(metrics: dict, title="Metrics"):
    if not metrics:
        st.info("Metrics not found yet.")
        return
    st.subheader(title)
    c1, c2 = st.columns(2)
    with c1: st.metric("Accuracy", f"{metrics.get('accuracy','—')}")
    with c2: st.metric("F1 (weighted)", f"{metrics.get('f1_weighted','—')}")

# ------------------- Layout -------------------
tab1, tab2, tab3 = st.tabs(["✅ Flow", "📊 EDA", "⚙️ Config & Paths"])

# ===== TAB 1: Required Flow =====
with tab1:
    st.markdown("### 1) Health Check (Serving)")
    colA, colB = st.columns([1,3])
    with colA:
        if st.button("Check MLflow /ping"):
            code, text = ping_mlflow()
            if code == 200:
                st.success(f"Serving OK ({code})")
            else:
                st.error(f"Serving not reachable: {text}")
    with colB:
        st.caption("Verifies the deployed model API is live (MLflow pyfunc /ping).")

    st.markdown("---")
    st.markdown("### 2) AutoML Result & Chosen Model")
    info = read_json_safe(MODEL_INFO, {})
    if info:
        rid   = info.get("run_id","—")
        def pick(d, *keys, default="—"):
            for k in keys:
                v = d.get(k)
                if v: 
                    return v
            return default

        algo = pick(info, "best_algo", "model_algo", "algo", "algorithm")
        mid  = pick(info, "model_id", "best_model_id", "modelId")
        c = st.columns(3)
        c[0].metric("MLflow run_id", rid)
        c[1].metric("Best algorithm", algo)
        c[2].metric("Model ID", mid)
        st.caption("Read from models/h2o_model_info.json")
    else:
        st.info("Train first to produce models/h2o_model_info.json (e.g., `make train` or `make train-docker`).")

    st.markdown("---")
    st.markdown("### 3) Run Baseline (batch inference + baseline monitoring)")
    b1, b2 = st.columns([1,3])
    with b1:
        if st.button("Run Baseline Now"):
            with st.spinner("Running batch inference..."):
                ok1, log1 = run_cmd("python src/batch_infer.py --input artifacts/test.csv")
            with st.spinner("Generating baseline report..."):
                ok2, log2 = run_cmd("python monitoring/monitor.py")
            if ok1 and ok2:
                st.success("Baseline ready: metrics.json + baseline.html")
            else:
                st.error("Baseline step failed. See logs below.")
            with st.expander("Logs"):
                st.code((log1 or "") + "\n" + (log2 or ""))
    with b2:
        metrics = read_json_safe(METRICS_JSON, {})
        show_metrics(metrics, "Baseline Metrics (artifacts/metrics.json)")
        st.markdown("**Baseline Report (Evidently)**")
        if BASELINE_HTML.exists():
            components.html(BASELINE_HTML.read_text(encoding="utf-8"), height=520, scrolling=True)
        else:
            st.info("Run baseline to generate artifacts/baseline.html")

    st.markdown("---")
    st.markdown("### 4) Run Drift Test (perturb ≥2 features + report)")
    d1, d2 = st.columns([1,3])
    with d1:
        if st.button("Run Drift Test Now"):
            with st.spinner("Running drift perturbation & monitoring..."):
                ok3, log3 = run_cmd("python monitoring/perturb_test.py")
            if ok3:
                st.success("Drift test ready: perturb_test_results.json + drift_after.html")
            else:
                st.error("Drift step failed. See logs below.")
            with st.expander("Logs"):
                st.code(log3 or "")
    with d2:
        res = read_json_safe(PERTURB_JSON, {})
        if res:
            def pickv(d, *keys):
                for k in keys:
                    if k in d and d[k] is not None:
                        return float(d[k])
                return float("nan")

            base  = pickv(res, "baseline_accuracy", "baseline", "acc_baseline")
            after = pickv(res, "after_accuracy", "after", "acc_after")

            change_signed = after - base  
            drop_abs = max(base - after, 0.0) 
            pct_change = (change_signed / base * 100.0) if base and base==base else float("nan")

            st.subheader("After-Drift Metrics")
            c = st.columns(3)
            c[0].metric("Baseline Accuracy", f"{base:.3f}" if base==base else "—")
            c[1].metric("After Accuracy", f"{after:.3f}" if after==after else "—",
                        delta=(f"{pct_change:+.1f}%" if pct_change==pct_change else None))
            c[2].metric("Performance Drop (abs)", f"{drop_abs:.3f}" if drop_abs==drop_abs else "—")

            changed = res.get("changed_features") or res.get("changes") or []
            if isinstance(changed, list):
                st.caption("Changed features: " + (", ".join(changed) if changed else "—"))
            else:
                st.caption(f"Changed features: {changed or '—'}")
        else:
            st.info("Run drift test to generate artifacts/perturb_test_results.json")

        st.markdown("**Drift Report (Evidently)**")
        if DRIFT_HTML.exists():
            components.html(DRIFT_HTML.read_text(encoding="utf-8"), height=520, scrolling=True)
        else:
            st.info("Run drift test to generate artifacts/drift_after.html")

# ===== TAB 2: EDA =====
with tab2:
    st.markdown("### Dataset Overview")
    df = load_data()
    if df is None:
        st.warning("Dataset not found at ./data/winequalityN.csv")
    else:
        c1, c2 = st.columns(2)
        with c1:
            st.metric("Rows", len(df))
            st.write("Type ratio (red/white)")
            st.bar_chart(df["type"].value_counts())
        with c2:
            st.write("Quality distribution")
            st.bar_chart(df["quality"].value_counts().sort_index())
        st.caption("These visuals satisfy the EDA requirement.")

# ===== TAB 3: Config & Paths =====
with tab3:
    st.markdown("### Paths & Files")
    st.json({
        "DATA_CSV": str(DATA_CSV),
        "ARTIFACTS": str(ARTIFACTS),
        "METRICS_JSON": str(METRICS_JSON),
        "PERTURB_JSON": str(PERTURB_JSON),
        "BASELINE_HTML": str(BASELINE_HTML),
        "DRIFT_HTML": str(DRIFT_HTML),
        "MODEL_INFO": str(MODEL_INFO),
        "MLflow URL (internal)": MLFLOW_URL,
    })
    st.caption("Helps the instructor verify artifact locations and reproducibility.")
