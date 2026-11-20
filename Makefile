PY=python
.PHONY: data ingest features train-baseline train-gnn rings evaluate app test
data:
	bash scripts/download_data.sh ibm
ingest:
	$(PY) -m mulering.data.ingest --config configs/config.yaml
features:
	$(PY) -m mulering.features.build --config configs/config.yaml
train-baseline:
	$(PY) -m mulering.models.baseline --config configs/config.yaml
train-gnn:
	$(PY) -m mulering.models.gnn --config configs/config.yaml
rings:
	$(PY) -m mulering.rings.extract --config configs/config.yaml
evaluate:
	$(PY) -m mulering.evaluation.run --config configs/config.yaml
app:
	streamlit run src/mulering/app/streamlit_app.py
test:
	pytest -q
