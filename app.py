#!/usr/bin/env python3

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel
import h2o
import pandas as pd
from datetime import datetime
import os
from typing import Dict, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Wine Quality ML Pipeline")
model = None
REQUESTS_LOG = "requests.csv"

class WineFeatures(BaseModel):
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

class PredictionResponse(BaseModel):
    predicted_quality: int
    confidence_scores: Dict[str, float]
    timestamp: str
    status: str

def init_h2o_and_model():
    global model
    try:
        if not h2o.connection():
            h2o.init()
        model_path = os.path.abspath("h2o_best_model/StackedEnsemble_BestOfFamily_4_AutoML_1_20250807_152640")
        if os.path.exists(model_path):
            logger.info(f"Loading model from: {model_path}")
            model = h2o.load_model(model_path)
            logger.info(f"Model loaded: {model.model_id}")
            return True
        logger.error(f"Model not found: {model_path}")
        return False
    except Exception as e:
        logger.error(f"Model loading failed: {e}")
        return False

@app.on_event("startup")
async def startup_event():
    init_h2o_and_model()

def log_request(features: Dict, prediction: Optional[int] = None):
    timestamp = datetime.now().isoformat()
    log_data = {'timestamp': timestamp, 'prediction': prediction, **features}
    df = pd.DataFrame([log_data])
    if os.path.exists(REQUESTS_LOG):
        df.to_csv(REQUESTS_LOG, mode='a', header=False, index=False)
    else:
        df.to_csv(REQUESTS_LOG, index=False)

@app.get("/", response_class=HTMLResponse)
async def root():
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Wine Quality ML Pipeline</title>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/3.9.1/chart.min.js"></script>
        <style>
            * { margin: 0; padding: 0; box-sizing: border-box; }
            body { 
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
                width: 100vw; height: 100vh; overflow: hidden;
                color: #2c3e50; font-weight: 400;
            }
            .container { 
                width: 100vw; height: 100vh; 
                display: grid; grid-template-rows: 80px 50px 1fr;
                padding: 0; margin: 0;
            }
            
            .header { 
                display: flex; align-items: center; justify-content: center;
                background: rgba(255,255,255,0.3); backdrop-filter: blur(10px);
                border-bottom: 1px solid rgba(255,255,255,0.2);
            }
            .header h1 { font-size: 2.5em; font-weight: 200; color: #34495e; }
            
            .nav { 
                display: flex; justify-content: center; align-items: center; gap: 15px;
                background: rgba(255,255,255,0.2); backdrop-filter: blur(5px);
            }
            .nav a { 
                color: #34495e; text-decoration: none; padding: 8px 20px;
                border-radius: 25px; background: rgba(255,255,255,0.4);
                transition: all 0.2s ease; font-size: 0.9em; font-weight: 500;
            }
            .nav a:hover { background: rgba(255,255,255,0.6); transform: translateY(-1px); }
            
            .main { 
                display: grid; grid-template-columns: 1fr 1fr; gap: 0;
                height: 100%; overflow: hidden;
            }
            
            .left, .right {
                padding: 30px; display: flex; flex-direction: column;
                background: rgba(255,255,255,0.1);
            }
            .left { border-right: 1px solid rgba(255,255,255,0.2); }
            .right { overflow-y: auto; }
            .right::-webkit-scrollbar { width: 4px; }
            .right::-webkit-scrollbar-thumb { background: rgba(52,73,94,0.2); border-radius: 2px; }
            .left { border-right: 1px solid rgba(255,255,255,0.2); }
            
            .section-title { 
                font-size: 1.4em; font-weight: 300; margin-bottom: 20px; 
                color: #2c3e50; text-align: center;
            }
            
            .form-grid { 
                display: grid; grid-template-columns: 1fr 1fr; gap: 12px;
                flex-grow: 1; align-content: start; overflow-y: auto;
            }
            .form-grid::-webkit-scrollbar { width: 4px; }
            .form-grid::-webkit-scrollbar-thumb { background: rgba(52,73,94,0.2); border-radius: 2px; }
            
            .input-group { display: flex; flex-direction: column; }
            .input-group label { 
                font-size: 0.85em; font-weight: 500; margin-bottom: 4px; 
                color: #34495e; opacity: 0.8;
            }
            .input-group input, .input-group select { 
                padding: 10px; border: none; border-radius: 8px;
                background: rgba(255,255,255,0.5); color: #2c3e50;
                font-size: 0.9em; transition: all 0.2s ease;
                border: 1px solid rgba(255,255,255,0.3);
            }
            .input-group input:focus, .input-group select:focus { 
                outline: none; background: rgba(255,255,255,0.7); 
                border-color: rgba(52,152,219,0.3); box-shadow: 0 0 0 3px rgba(52,152,219,0.1);
            }
            .input-group input::placeholder { color: rgba(44,62,80,0.5); }
            
            .predict-btn { 
                width: 100%; padding: 15px; margin-top: 20px;
                background: linear-gradient(45deg, #3498db, #2980b9);
                border: none; border-radius: 8px; color: white;
                font-size: 1.1em; font-weight: 500; cursor: pointer;
                transition: all 0.2s ease;
            }
            .predict-btn:hover { transform: translateY(-1px); box-shadow: 0 5px 15px rgba(52,152,219,0.3); }
            .predict-btn:disabled { opacity: 0.5; cursor: not-allowed; transform: none; box-shadow: none; }
            
            .status { 
                text-align: center; padding: 12px; margin-bottom: 15px;
                border-radius: 8px; background: rgba(255,255,255,0.4);
                font-size: 0.9em; font-weight: 500;
            }
            .status-ok { color: #27ae60; }
            .status-error { color: #e74c3c; }
            
            .result { 
                margin-top: 15px; padding: 15px; 
                background: rgba(255,255,255,0.4); border-radius: 8px;
            }
            .result h3 { font-size: 1.1em; font-weight: 500; margin-bottom: 8px; }
            
            .stats { flex-grow: 1; }
            .stat-grid { 
                display: grid; grid-template-columns: 1fr 1fr; gap: 15px;
                margin-bottom: 20px;
            }
            .stat-item { 
                text-align: center; padding: 15px; 
                background: rgba(255,255,255,0.4); border-radius: 8px;
            }
            .stat-value { 
                font-size: 1.8em; font-weight: 200; color: #3498db;
                display: block; margin-bottom: 5px;
            }
            .stat-label { font-size: 0.8em; color: #7f8c8d; font-weight: 500; }
            
            .charts { margin-top: 20px; }
            .chart-container { 
                background: rgba(255,255,255,0.3); 
                border-radius: 8px; padding: 15px; margin-bottom: 15px;
                height: 200px; position: relative;
            }
            .chart-title { 
                font-size: 0.9em; font-weight: 500; color: #2c3e50; 
                margin-bottom: 10px; text-align: center;
            }
            
            .modal { 
                display: none; position: fixed; top: 0; left: 0; 
                width: 100vw; height: 100vh; background: rgba(0,0,0,0.5);
                z-index: 1000; backdrop-filter: blur(5px);
            }
            .modal-content { 
                position: absolute; top: 50%; left: 50%; 
                transform: translate(-50%, -50%); 
                width: 80vw; height: 80vh; 
                background: rgba(255,255,255,0.95); 
                border-radius: 15px; padding: 20px;
                display: flex; flex-direction: column;
            }
            .modal-header { 
                display: flex; justify-content: space-between; 
                align-items: center; margin-bottom: 15px; 
                border-bottom: 1px solid rgba(0,0,0,0.1); 
                padding-bottom: 10px;
            }
            .modal-title { 
                font-size: 1.3em; font-weight: 300; color: #2c3e50; 
            }
            .modal-actions { display: flex; gap: 10px; }
            .modal-btn { 
                padding: 8px 16px; border: none; border-radius: 6px;
                cursor: pointer; font-size: 0.9em; transition: all 0.2s;
            }
            .btn-download { 
                background: #3498db; color: white; 
            }
            .btn-download:hover { background: #2980b9; }
            .btn-close { 
                background: #95a5a6; color: white; 
            }
            .btn-close:hover { background: #7f8c8d; }
            .modal-body { 
                flex-grow: 1; overflow-y: auto; 
                background: #f8f9fa; border-radius: 8px; 
                padding: 15px; font-family: 'Monaco', 'Menlo', monospace;
                font-size: 0.8em; line-height: 1.4; color: #2c3e50;
            }
            .modal-body::-webkit-scrollbar { width: 6px; }
            .modal-body::-webkit-scrollbar-thumb { 
                background: rgba(52,73,94,0.3); border-radius: 3px; 
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>Wine Quality ML</h1>
            </div>
            
            <div class="nav">
                <a href="/health">Health</a>
                <a href="/reports/baseline.html" target="_blank">Baseline</a>
                <a href="/reports/drift_after.html" target="_blank">Drift</a>
                <a href="/docs" target="_blank">API</a>
                <a href="#" onclick="showLogModal()">Logs</a>
            </div>
            
            <div class="main">
                <div class="left">
                    <h2 class="section-title">Prediction</h2>
                    <div class="status" id="statusBar">
                        <span id="statusText">Checking...</span>
                    </div>
                    
                    <form id="predictionForm">
                        <div class="form-grid">
                            <div class="input-group">
                                <label>Type</label>
                                <select name="type" required>
                                    <option value="">Select</option>
                                    <option value="white">White</option>
                                    <option value="red">Red</option>
                                </select>
                            </div>
                            <div class="input-group">
                                <label>Fixed Acidity</label>
                                <input type="number" name="fixed_acidity" step="0.1" placeholder="7.0" required>
                            </div>
                            <div class="input-group">
                                <label>Volatile Acidity</label>
                                <input type="number" name="volatile_acidity" step="0.01" placeholder="0.27" required>
                            </div>
                            <div class="input-group">
                                <label>Citric Acid</label>
                                <input type="number" name="citric_acid" step="0.01" placeholder="0.36" required>
                            </div>
                            <div class="input-group">
                                <label>Residual Sugar</label>
                                <input type="number" name="residual_sugar" step="0.1" placeholder="20.7" required>
                            </div>
                            <div class="input-group">
                                <label>Chlorides</label>
                                <input type="number" name="chlorides" step="0.001" placeholder="0.045" required>
                            </div>
                            <div class="input-group">
                                <label>Free SO2</label>
                                <input type="number" name="free_sulfur_dioxide" step="1" placeholder="45" required>
                            </div>
                            <div class="input-group">
                                <label>Total SO2</label>
                                <input type="number" name="total_sulfur_dioxide" step="1" placeholder="170" required>
                            </div>
                            <div class="input-group">
                                <label>Density</label>
                                <input type="number" name="density" step="0.0001" placeholder="1.0010" required>
                            </div>
                            <div class="input-group">
                                <label>pH</label>
                                <input type="number" name="pH" step="0.01" placeholder="3.00" required>
                            </div>
                            <div class="input-group">
                                <label>Sulphates</label>
                                <input type="number" name="sulphates" step="0.01" placeholder="0.45" required>
                            </div>
                            <div class="input-group">
                                <label>Alcohol</label>
                                <input type="number" name="alcohol" step="0.1" placeholder="8.8" required>
                            </div>
                        </div>
                        <button type="submit" class="predict-btn" id="predictBtn">Predict Quality</button>
                    </form>
                    
                    <div id="predictionResult" style="display:none;" class="result">
                        <h3>Result</h3>
                        <div id="resultContent"></div>
                    </div>
                </div>
                
                <div class="right">
                    <h2 class="section-title">Statistics</h2>
                    <div class="stats">
                        <div class="stat-grid">
                            <div class="stat-item">
                                <span class="stat-value" id="totalPredictions">0</span>
                                <div class="stat-label">Predictions</div>
                            </div>
                            <div class="stat-item">
                                <span class="stat-value" id="modelStatus">Unknown</span>
                                <div class="stat-label">Status</div>
                            </div>
                        </div>
                        
                        <div class="charts">
                            <div class="chart-container">
                                <div class="chart-title">Wine Quality Distribution</div>
                                <canvas id="qualityChart"></canvas>
                            </div>
                            <div class="chart-container">
                                <div class="chart-title">Wine Type Ratio</div>
                                <canvas id="typeChart"></canvas>
                            </div>
                            <div class="chart-container">
                                <div class="chart-title">Alcohol Content Distribution</div>
                                <canvas id="alcoholChart"></canvas>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        
        <div id="logModal" class="modal">
            <div class="modal-content">
                <div class="modal-header">
                    <h3 class="modal-title">Training Logs (flaml.log)</h3>
                    <div class="modal-actions">
                        <button class="modal-btn btn-download" onclick="downloadLog()">Download</button>
                        <button class="modal-btn btn-close" onclick="closeLogModal()">Close</button>
                    </div>
                </div>
                <div class="modal-body" id="logContent">
                    Loading...
                </div>
            </div>
        </div>
        
        <script>
            fetch('/health').then(r => r.json()).then(data => {
                const statusText = document.getElementById('statusText');
                const predictBtn = document.getElementById('predictBtn');
                const modelStatus = document.getElementById('modelStatus');
                
                if (data.model_loaded && data.h2o_connected) {
                    statusText.innerHTML = '<span class="status-ok">System Ready</span>';
                    modelStatus.textContent = 'Active';
                    predictBtn.disabled = false;
                } else {
                    statusText.innerHTML = '<span class="status-error">System Error</span>';
                    modelStatus.textContent = 'Inactive';
                    predictBtn.disabled = true;
                }
            }).catch(() => {
                document.getElementById('statusText').innerHTML = '<span class="status-error">Connection Failed</span>';
            });
            
            fetch('/stats').then(r => r.json()).then(data => {
                document.getElementById('totalPredictions').textContent = data.total_predictions;
            });
            
            fetch('/data-stats').then(r => r.json()).then(data => {
                if (data.success) {
                    createQualityChart(data.quality_distribution);
                    createTypeChart(data.type_distribution);
                    createAlcoholChart(data.alcohol_distribution);
                }
            }).catch(err => {
                console.log('Failed to load data statistics:', err);
            });
            
            function createQualityChart(data) {
                const ctx = document.getElementById('qualityChart').getContext('2d');
                new Chart(ctx, {
                    type: 'bar',
                    data: {
                        labels: data.labels,
                        datasets: [{
                            data: data.values,
                            backgroundColor: 'rgba(52, 152, 219, 0.6)',
                            borderColor: 'rgba(52, 152, 219, 0.8)',
                            borderWidth: 1
                        }]
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        plugins: { legend: { display: false } },
                        scales: {
                            y: { 
                                beginAtZero: true,
                                grid: { color: 'rgba(0,0,0,0.1)' },
                                ticks: { color: '#7f8c8d', font: { size: 10 } }
                            },
                            x: { 
                                grid: { display: false },
                                ticks: { color: '#7f8c8d', font: { size: 10 } }
                            }
                        }
                    }
                });
            }
            
            function createTypeChart(data) {
                const ctx = document.getElementById('typeChart').getContext('2d');
                new Chart(ctx, {
                    type: 'doughnut',
                    data: {
                        labels: data.labels,
                        datasets: [{
                            data: data.values,
                            backgroundColor: ['rgba(52, 152, 219, 0.7)', 'rgba(155, 89, 182, 0.7)'],
                            borderWidth: 0
                        }]
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        plugins: { 
                            legend: { 
                                position: 'bottom',
                                labels: { color: '#7f8c8d', font: { size: 10 } }
                            }
                        }
                    }
                });
            }
            
            function createAlcoholChart(data) {
                const ctx = document.getElementById('alcoholChart').getContext('2d');
                new Chart(ctx, {
                    type: 'line',
                    data: {
                        labels: data.labels,
                        datasets: [{
                            data: data.values,
                            borderColor: 'rgba(46, 204, 113, 0.8)',
                            backgroundColor: 'rgba(46, 204, 113, 0.1)',
                            fill: true,
                            tension: 0.4,
                            pointRadius: 2
                        }]
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        plugins: { legend: { display: false } },
                        scales: {
                            y: { 
                                beginAtZero: true,
                                grid: { color: 'rgba(0,0,0,0.1)' },
                                ticks: { color: '#7f8c8d', font: { size: 10 } }
                            },
                            x: { 
                                grid: { display: false },
                                ticks: { color: '#7f8c8d', font: { size: 10 } }
                            }
                        }
                    }
                });
            }
            
            document.getElementById('predictionForm').addEventListener('submit', async (e) => {
                e.preventDefault();
                
                const formData = new FormData(e.target);
                const data = Object.fromEntries(formData.entries());
                
                ['fixed_acidity', 'volatile_acidity', 'citric_acid', 'residual_sugar', 
                 'chlorides', 'free_sulfur_dioxide', 'total_sulfur_dioxide', 
                 'density', 'pH', 'sulphates', 'alcohol'].forEach(field => {
                    data[field] = parseFloat(data[field]);
                });
                
                try {
                    const response = await fetch('/predict', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify(data)
                    });
                    
                    const result = await response.json();
                    
                    if (response.ok) {
                        document.getElementById('resultContent').innerHTML = `
                            <p><strong>Quality:</strong> ${result.predicted_quality}/10</p>
                            <p><strong>Time:</strong> ${new Date(result.timestamp).toLocaleTimeString()}</p>
                        `;
                        document.getElementById('predictionResult').style.display = 'block';
                        
                        const current = parseInt(document.getElementById('totalPredictions').textContent);
                        document.getElementById('totalPredictions').textContent = current + 1;
                    } else {
                        alert('Failed: ' + result.detail);
                    }
                    
                } catch (error) {
                    alert('Error: ' + error.message);
                }
            });
            
            window.showLogModal = async () => {
                document.getElementById('logModal').style.display = 'block';
                try {
                    const response = await fetch('/log-content');
                    const data = await response.json();
                    if (response.ok) {
                        document.getElementById('logContent').textContent = data.content;
                    } else {
                        document.getElementById('logContent').textContent = 'Log file not found or empty.';
                    }
                } catch (error) {
                    document.getElementById('logContent').textContent = 'Failed to load log content.';
                }
            };
            
            window.closeLogModal = () => {
                document.getElementById('logModal').style.display = 'none';
            };
            
            window.downloadLog = () => {
                window.open('/download-flaml-log', '_blank');
            };
            
            document.getElementById('logModal').addEventListener('click', (e) => {
                if (e.target.id === 'logModal') {
                    closeLogModal();
                }
            });
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html)

@app.post("/predict", response_model=PredictionResponse)
async def predict(features: WineFeatures):
    data_dict = features.dict()
    
    if model is None:
        log_request(data_dict, None)
        return PredictionResponse(
            predicted_quality=6,
            confidence_scores={},
            timestamp=datetime.now().isoformat(),
            status="model_unavailable"
        )
    
    try:
        df = pd.DataFrame([data_dict])
        h2o_frame = h2o.H2OFrame(df)
        h2o_frame['type'] = h2o_frame['type'].asfactor()
        
        predictions = model.predict(h2o_frame)
        pred_df = predictions.as_data_frame()
        
        predicted_quality = int(pred_df['predict'].iloc[0])
        
        confidence_cols = [col for col in pred_df.columns if col.startswith('p')]
        confidence_scores = {}
        for col in confidence_cols:
            quality_level = col.replace('p', '')
            confidence_scores[f"quality_{quality_level}"] = float(pred_df[col].iloc[0])
        
        log_request(data_dict, predicted_quality)
        
        return PredictionResponse(
            predicted_quality=predicted_quality,
            confidence_scores=confidence_scores,
            timestamp=datetime.now().isoformat(),
            status="success"
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        log_request(data_dict, None)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "h2o_connected": h2o.connection() is not None
    }

@app.get("/stats")
async def get_stats():
    try:
        if os.path.exists(REQUESTS_LOG):
            df = pd.read_csv(REQUESTS_LOG)
            total_predictions = len(df)
        else:
            total_predictions = 0
        return {
            "total_predictions": total_predictions,
            "model_status": "active" if model is not None else "inactive",
            "model_type": "Stacked Ensemble" if model is not None else "not_loaded"
        }
    except:
        return {"total_predictions": 0, "model_status": "unknown", "model_type": "unknown"}

@app.get("/data-stats")
async def get_data_stats():
    try:
        if not os.path.exists("winequalityN.csv"):
            return {"success": False, "error": "Dataset not found"}
        
        df = pd.read_csv("winequalityN.csv")
        df = df.dropna()
        
        quality_counts = df['quality'].value_counts().sort_index()
        quality_distribution = {
            "labels": [str(i) for i in quality_counts.index],
            "values": quality_counts.values.tolist()
        }
        
        type_counts = df['type'].value_counts()
        type_distribution = {
            "labels": type_counts.index.tolist(),
            "values": type_counts.values.tolist()
        }
        
        alcohol_hist, alcohol_bins = pd.cut(df['alcohol'], bins=10, retbins=True)
        alcohol_counts = alcohol_hist.value_counts().sort_index()
        alcohol_distribution = {
            "labels": [f"{alcohol_bins[i]:.1f}-{alcohol_bins[i+1]:.1f}" for i in range(len(alcohol_bins)-1)],
            "values": alcohol_counts.values.tolist()
        }
        
        return {
            "success": True,
            "quality_distribution": quality_distribution,
            "type_distribution": type_distribution,
            "alcohol_distribution": alcohol_distribution,
            "total_samples": len(df)
        }
        
    except Exception as e:
        logger.error(f"Data stats error: {e}")
        return {"success": False, "error": str(e)}

@app.get("/log-content")
async def get_log_content():
    log_file = "flaml.log"
    try:
        if os.path.exists(log_file):
            with open(log_file, 'r', encoding='utf-8') as f:
                content = f.read()
            return {"content": content, "exists": True}
        else:
            return {"content": "Log file not found.", "exists": False}
    except Exception as e:
        return {"content": f"Error reading log file: {str(e)}", "exists": False}

@app.get("/download-flaml-log")
async def download_flaml_log():
    log_file = "flaml.log"
    if os.path.exists(log_file):
        return FileResponse(log_file, filename="flaml.log")
    raise HTTPException(status_code=404, detail="Log file not found")

@app.get("/reports/{filename}")
async def get_report(filename: str):
    if os.path.exists(filename):
        return FileResponse(filename)
    raise HTTPException(status_code=404, detail="Report not found")

@app.get("/download-logs")
async def download_logs():
    if os.path.exists(REQUESTS_LOG):
        return FileResponse(REQUESTS_LOG, filename="requests.csv")
    raise HTTPException(status_code=404, detail="No logs available")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)