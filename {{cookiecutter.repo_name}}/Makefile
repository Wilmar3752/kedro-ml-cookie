.PHONY: setup install lint format test \
        run \
        run-titanic run-titanic-preprocessing run-titanic-hypertune run-titanic-reporting \
        run-preprocessing run-hypertune run-reporting run-e2e \
        run-california run-california-preprocessing run-california-hypertune run-california-reporting \
        viz jupyter build clean

PYTHON = 3.11
VENV   = .venv

# ── Environment ───────────────────────────────────────────────────────────────

setup:
	uv venv --python $(PYTHON)

install:
	uv sync

# ── Quality ───────────────────────────────────────────────────────────────────

lint:
	uv run ruff check src/

format:
	uv run ruff format src/

test:
	uv run pytest tests/ --cov=src -v

# ── Titanic (binary classification) ──────────────────────────────────────────

run-titanic-preprocessing:
	uv run kedro run --pipeline titanic_preprocessing

run-titanic-hypertune:
	uv run kedro run --pipeline titanic_hypertune

run-titanic-reporting:
	uv run kedro run --pipeline titanic_reporting

run-titanic:
	uv run kedro run --pipeline e2e_titanic

# ── California Housing (regression) ──────────────────────────────────────────

run-california-preprocessing:
	uv run kedro run --pipeline california_preprocessing

run-california-hypertune:
	uv run kedro run --pipeline california_hypertune

run-california-reporting:
	uv run kedro run --pipeline california_reporting

run-california:
	uv run kedro run --pipeline e2e_california

# ── Both models end-to-end ────────────────────────────────────────────────────

run:
	uv run kedro run

# ── Backward-compatible aliases (titanic) ────────────────────────────────────

run-preprocessing:
	uv run kedro run --pipeline preprocessing

run-hypertune:
	uv run kedro run --pipeline hypertune

run-reporting:
	uv run kedro run --pipeline reporting

run-e2e:
	uv run kedro run --pipeline e2e_modelling

# ── Tools ─────────────────────────────────────────────────────────────────────

viz:
	uv run kedro viz run

jupyter:
	uv run jupyter lab

build:
	uv run kedro package

clean:
	rm -rf $(VENV) dist build .pytest_cache src/*.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
