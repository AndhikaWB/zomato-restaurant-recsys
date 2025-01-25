SHELL := /bin/bash
CONDA_ENV := pytorch
.PHONY: qdrant streamlit

qdrant:
	qdrant --config-path $(CURDIR)/qdrant/local.yaml

streamlit:
	$(MAKE) qdrant & \
	source ~/.bashrc && \
	conda activate $(CONDA_ENV) && \
	streamlit run app.py