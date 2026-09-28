.PHONY: install run dev test test-live check lock

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

# Re-pin constraints.txt (the exact versions the Docker image installs) after
# editing requirements.txt. Resolves for the Space's platform, not this machine.
lock:
	uv pip compile requirements.txt --python-version 3.12 \
		--python-platform x86_64-unknown-linux-gnu \
		--extra-index-url https://download.pytorch.org/whl/cpu \
		--index-strategy unsafe-best-match \
		--custom-compile-command "make lock" -o constraints.txt
