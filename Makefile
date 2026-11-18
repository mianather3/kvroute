.PHONY: check test build bench bench-data plots

PYTHON ?= python3
VENV := .venv

check:
	cargo fmt --check && cargo clippy --workspace --all-targets -- -D warnings

test:
	cargo test --workspace

build:
	cargo build --workspace --release

# Seeded sweeps -> benchmarks/results/*.csv, then figures -> benchmarks/plots/*.png.
bench: bench-data plots

bench-data:
	scripts/run_benchmarks.sh

# Uses system matplotlib if present, else a local .venv.
plots:
	@if $(PYTHON) -c "import matplotlib" 2>/dev/null; then \
		$(PYTHON) scripts/plot_benchmarks.py; \
	else \
		[ -x $(VENV)/bin/python ] || $(PYTHON) -m venv $(VENV); \
		$(VENV)/bin/python -c "import matplotlib" 2>/dev/null || \
			$(VENV)/bin/pip install -q -r scripts/requirements.txt; \
		$(VENV)/bin/python scripts/plot_benchmarks.py; \
	fi
