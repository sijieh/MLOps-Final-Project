.PHONY: train serve infer monitor perturb train-docker infer-docker monitor-docker perturb-docker

# ---- Local host flow ----
train:
	python src/train.py

serve:
	MLFLOW_TRACKING_URI=./models/mlruns \
	python -c "import json,os; j=json.load(open('models/h2o_model_info.json')); os.system(f'mlflow models serve -m runs:/{j[\"run_id\"]}/model --env-manager local -h $${SERVE_HOST:-127.0.0.1} -p $${SERVE_PORT:-5000}')"

infer:
	python src/batch_infer.py

monitor:
	python monitoring/monitor.py

perturb:
	python monitoring/perturb_test.py

# ---- Docker flow ----
train-docker:
	docker compose run --rm trainer python src/train.py

infer-docker:
	docker compose run --rm trainer python src/batch_infer.py

monitor-docker:
	docker compose run --rm trainer python monitoring/monitor.py

perturb-docker:
	docker compose run --rm trainer python monitoring/perturb_test.py
