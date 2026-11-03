DATASET ?= toy
RETRIEVER ?= bm25
K ?= 10
DATA ?=
LIMIT ?= 0
PYTHON ?= $(if $(wildcard .venv/bin/python),.venv/bin/python,python3)

.PHONY: bench test gate
bench:
	$(PYTHON) -m agentmem.bench --dataset $(DATASET) --retriever $(RETRIEVER) --k $(K) --limit $(LIMIT) $(if $(DATA),--path $(DATA),)

test:
	$(PYTHON) -m tests.test_smoke
	$(PYTHON) -m tests.test_stage4
	$(PYTHON) -m tests.test_stage5
	$(PYTHON) -m tests.test_guardrails
	$(PYTHON) -m tests.test_package
	$(PYTHON) -m tests.test_index

gate:
	$(PYTHON) -m agentmem.gate_recall
