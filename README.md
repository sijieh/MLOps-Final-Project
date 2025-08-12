# MLOps-Final-Project

## ML Monitoring with Evidently

### Files

- `monitor.py` - Baseline monitoring script
- `perturb_test.py` - Data drift simulation and testing
- `requirements.txt` - Required dependencies

### Setup

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
**Note:** Open a new Terminal window, keep the old window (service) running, repeat Setup step 2-3 (cd to root file & Create Virtual Environment)
1. Train the Model
```bash
make train
```

The trained model will be stored in the models/mlruns/ directory.

2. Batch Inference
```bash
make infer
```
- Reads artifacts/test.csv
- Sends it to the MLflow model for predictions
- Outputs:
  - artifacts/preds.json
  - artifacts/metrics.json

3. Generate Baseline Monitoring Report
```bash
make monitor
```
- Uses the original dataset as both reference and current data
- Outputs:
  - artifacts/baseline.html
  - artifacts/baseline_metrics.json

4. Simulate Data Drift & Test
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

**Note:** Due to file size limitations, HTML reports need to be downloaded and opened locally in your browser.


### Results

The monitoring system successfully detects significant performance degradation:

- **Baseline Accuracy:** 68.40%
- **After Drift Accuracy:** 49.92%
- **Performance Drop:** -18.48%

### Technical Details

#### Column Mapping
- **Numerical features:** 11 physicochemical properties
- **Categorical features:** Wine type (red/white)
- **Target:** Quality score
- **Predictions:** H2O AutoML model outputs

#### Monitoring Metrics
- Data drift detection
- Data quality analysis
- Target drift monitoring
- Classification performance tracking

## How to View HTML Reports

### Method 1: Download and Open Locally
1. Download the HTML files from the repository
2. Open them in your web browser locally

### Method 2: GitHub Pages (Recommended)
Visit the live reports at: [GitHub Pages Site](https://yijiasong1002-uchi.github.io/MLOps-Final-Project/)

### Method 3: Alternative Hosting
- Upload to any static hosting service (Netlify, Vercel, etc.)
- Use online HTML viewers like htmlpreview.github.io
