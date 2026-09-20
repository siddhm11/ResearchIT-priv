.PHONY: install run dev test test-live check

PYTHON ?= python3

install:
	$(PYTHON) -m pip install -r requirements-dev.txt

run:
	$(PYTHON) run.py

dev:
	DEV_RELOAD=1 $(PYTHON) run.py

test:
	$(PYTHON) -m pytest tests/ -m "not live" -q

test-live:
	$(PYTHON) -m pytest tests/ -m live -v

check:
	$(PYTHON) -m compileall -q app scripts tests
	$(PYTHON) -m pytest tests/ -m "not live" -q
