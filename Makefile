train:
	python -m src.train --max_runtime_secs 120

serve:
	MLFLOW_TRACKING_URI=./models/mlruns \
	python -c "import json,os; j=json.load(open('models/h2o_model_info.json')); os.system(f'mlflow models serve -m runs:/{j[\"run_id\"]}/model --env-manager local -h {os.getenv(\"SERVE_HOST\",\"127.0.0.1\")} -p {os.getenv(\"SERVE_PORT\",\"5000\")}')"

infer:
	python -m src.batch_infer --input artifacts/test.csv

monitor:
	python monitoring/monitor.py

perturb:
	python monitoring/perturb_test.py
