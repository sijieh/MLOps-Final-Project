# MLOps-Final-Project

## Project Overview
This project implements an end-to-end MLOps pipeline for wine quality prediction, including:
- Data ingestion & preprocessing
- Model training with H2O AutoML
- Model serving with MLflow
- Monitoring with Evidently AI
- Drift simulation & performance degradation testing
- Local & Dockerized deployment with Streamlit UI

## Tech Stack
- Modeling: H2O AutoML
- Serving: MLflow pyfunc
- Monitoring: Evidently AI
- UI: Streamlit
- Deployment: Docker Compose
- Orchestration: Makefile

## Technical Architecture
1. Data Source
- Dataset: Wine Quality Dataset (red & white wines) with 11 physicochemical features + wine type + quality score
- Target variable: quality (integer score)
- Features:
  - Numerical: fixed acidity, volatile acidity, citric acid, residual sugar, chlorides, free sulfur dioxide, total sulfur dioxide, density, pH, sulphates, alcohol
  - Categorical: wine type (red/white)

2. Model Training
- Framework: H2O AutoML
- Steps:
  1. Train/test split
  2. Launch AutoML search for best model
  3. Save best model in MLflow format
  4. Store model metadata (h2o_model_info.json)
     
3. Model Serving
- MLflow pyfunc server runs inside Docker (model-serving service)
- REST API Endpoints:
  - /invocations → batch prediction (POST JSON or CSV)
  - /ping → health check

4. Monitoring
- Evidently tracks:
  - Data drift
  - Target drift
  - Data quality metrics
  - Classification performance metrics (accuracy, F1)
- Baseline monitoring:
  - Reference data: training set
  - Current data: test set
- Drift simulation:
  - Perturb ≥ 2 features (alcohol -1.2, volatile acidity +0.1)
  - Re-run monitoring to detect performance drop

5. Deployment
- Local mode: Python virtualenv + Makefile commands
- Docker mode: Multi-service docker-compose setup
  - model-serving → MLflow server
  - streamlit → UI for predictions, batch uploads, drift tests
  - Shared models/ & artifacts/ via Docker volumes

### 1. Docker Setup (Recommended)
1. Build all services
```bash
docker compose build --no-cache
```

2. Train model inside Docker
```bash
make train-docker
```
**Note:** This will save the model into models/mlruns and models/h2o_model_info.json

3. Start Model Serving + Streamlit UI
```bash
docker compose up -d model-serving streamlit
```

4. Access
- UI: http://localhost:8501
- API: http://localhost:5001

### 2. Local Setup (No Docker)

1. Clone & Navigate
```bash
git clone https://github.com/yijiasong1002-uchi/MLOps-Final-Project.git
cd MLOps-Final-Project
```
2. Create Virtual Environment (Recommended)
```bash
python3 -m venv .venv
source .venv/bin/activate
```
3. Install Dependencies
```bash
pip install -r requirements.txt
```
4. Start MLflow Model Serving (Required)
```bash
make serve
```
**Note:** Ensure the service keeps running for the whole process

### Usage
5. Train the Model
**Note:** Open a new Terminal window, keep the old window (service) running, repeat Setup step 2-3 (cd to root file & Create Virtual Environment)
```bash
make train
```
The trained model will be stored in the models/mlruns/ directory.

6. Batch Inference
```bash
make infer
```
- Reads artifacts/test.csv
- Sends it to the MLflow model for predictions
- Outputs:
  - artifacts/preds.json
  - artifacts/metrics.json

7. Generate Baseline Monitoring Report
```bash
make monitor
```
- Uses the original dataset as both reference and current data
- Outputs:
  - artifacts/baseline.html
  - artifacts/baseline_metrics.json

8. Simulate Data Drift & Test
```bash
make perturb
```
- Artificially modifies:
  - alcohol → -1.2
  - volatile acidity → +0.1
- Detects performance degradation after drift
- Outputs:
  - artifacts/drift-after.html
  - artifacts/perturb_test_results.json

**Note:** Due to file size limitations, HTML reports need to be opened locally (the HTML reports are located inside **artifacts/**)

## How to View HTML Reports

### Method 1: Open Locally
1. After running all stpes above, the HTML files shoud be successfully locate within **artifacts/**
2. Open them in your web browser locally

### Method 2: GitHub Pages (Recommended)
Visit the live reports at: [GitHub Pages Site](https://yijiasong1002-uchi.github.io/MLOps-Final-Project/)

### Method 3: Alternative Hosting
- Upload to any static hosting service (Netlify, Vercel, etc.)
- Use online HTML viewers like htmlpreview.github.io
