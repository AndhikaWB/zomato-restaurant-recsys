SHELL := /bin/bash
.PHONY: qdrant

qdrant:
	qdrant --config-path $(CURDIR)/qdrant/local.yaml