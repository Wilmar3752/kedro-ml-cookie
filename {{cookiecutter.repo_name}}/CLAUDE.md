# CLAUDE.md — {{cookiecutter.project_name}}

## What This Project Is

End-to-end ML project built with Kedro. Includes two reference models (Titanic binary classification, California Housing regression) that demonstrate the full pipeline pattern: preprocessing → hyperparameter tuning → reporting.

## Stack

| Layer | Tool |
|---|---|
| Pipeline framework | Kedro 1.1.1 |
| MLflow integration | kedro-mlflow 2.0.0 |
| Model registry | MLflow 3.x (alias-based) |
| Hyperparameter tuning | Optuna 4.x |
| Base model | LightGBM 4.x |
| Feature engineering | feature-engine |
| Package manager | uv (never use pip) |
| Config | OmegaConfigLoader (YAML + globals) |

## Setup

```bash
make setup      # uv venv --python 3.11
make install    # uv sync
```

## Common Commands

```bash
# Run pipelines
make run                         # Both models end-to-end (default)
make run-titanic                 # e2e titanic
make run-california              # e2e california
make run-titanic-preprocessing   # Single stage
make run-titanic-hypertune
make run-titanic-reporting

# Dev
make lint        # ruff check src/
make format      # ruff format src/
make test        # pytest tests/ --cov=src -v
make viz         # kedro viz run
make jupyter     # jupyter lab

# MLflow UI
kedro mlflow ui  # localhost:5000
```

## Source Layout

```
src/
├── modelling/          # Kedro application (pipelines, nodes, settings)
│   ├── settings.py
│   ├── pipeline_registry.py
│   ├── titanic/        # Binary classification example
│   └── california/     # Regression example
└── packages/           # Kedro-independent reusable library
    ├── modelling/
    │   ├── evaluate/       # Metric functions
    │   ├── models/
    │   │   ├── supervised/ # BaseSklearnCompatibleModel
    │   │   ├── preprocessors/
    │   │   └── mlflow/     # MLflowModelStageManager
    │   └── transformers/
    ├── reporting/          # HTML reports (Plotly + Jinja2)
    └── python_utils/       # Object injection from YAML
```

`packages/` has no Kedro dependency — import it anywhere. `modelling/` is the Kedro app.

## Pipeline Structure

Each model follows this pattern:

```
{dataset}_preprocessing → {dataset}_hypertune → {dataset}_reporting
```

Each stage lives in `src/modelling/{dataset}/{stage}/` with `pipeline.py` + `nodes.py`. The master registry in `src/modelling/pipeline_registry.py` also exposes `e2e_{dataset}` and `__default__` (both models).

## Config Layout

```
conf/base/
├── globals.yml              # global_seed: 42, number_of_trials: 100
├── mlflow.yml               # Tracking URI, experiment name
├── catalog/modelling/{dataset}/
│   ├── preprocessing.yml
│   ├── hypertune.yml
│   └── reporting.yml
└── parameters/modelling/{dataset}/
    ├── data_processing.yml
    └── hypertune.yml
```

`conf/local/` is gitignored — use it for secrets and local overrides.

## Key Conventions

### MLflow Registry — Aliases, Not Stages

MLflow 3.x removed stages. This project uses aliases:
- `@champion` → production model
- `@challenger` → newly trained model

Never use `transition_model_version_stage()` — it is deprecated and removed.

Catalog pattern:
```yaml
titanic_registered_model:
  type: packages.modelling.models.mlflow.registry.MLflowModelStageManager
  model_name: titanic
  load_alias: champion
  save_alias: challenger
  pyfunc_predict_fn: predict
```

### Object Injection from YAML

Pipeline steps and Optuna hyperparameter search spaces are defined in `parameters/{dataset}/hypertune.yml` using dotted class paths. Optuna expressions like `"trial.suggest_int('n_estimators', 50, 500)"` are evaluated at tuning time. Do not hardcode these in Python — keep them in config.

### AutoConfigFeaturePreprocessors

`BaseSklearnCompatibleModel` auto-injects column lists (numeric, categorical, columns with NaNs) into feature-engine transformers at `fit()` time. Do not manually pass `variables` lists to these transformers in parameters — leave them empty (`variables: []`) and they are populated programmatically.

### Reproducibility

Global seed is `42` everywhere — CV splits, samplers, model `random_state`. Change it only via `conf/base/globals.yml`.

### Training Attributes Are Stripped Before MLflow Logging

`MLflowModelStageManager` removes `X_train`, `y_train`, `df_scores`, and `hypertune_results` before logging to MLflow. Do not rely on these attributes being present on a model loaded from the registry.

## Data Layers

```
data/01_raw/        # Committed raw CSVs — never modify
data/02 → 09/       # All gitignored, rebuilt by pipeline
data/06_models/     # Fitted .pkl artifacts (also in MLflow registry)
```

Never commit data beyond `01_raw/`. Never remove `.gitkeep` files.

## Adding a New Model

1. Create `src/modelling/{name}/` with `preprocessing/`, `hypertune/`, `reporting/` (each with `pipeline.py` + `nodes.py`)
2. Create `src/modelling/{name}/pipeline_registry.py` mirroring the titanic pattern
3. Add catalog files under `conf/base/catalog/modelling/{name}/`
4. Add parameter files under `conf/base/parameters/modelling/{name}/`
5. Register pipelines in the master `src/modelling/pipeline_registry.py`
6. Add Makefile targets following `run-{name}-{stage}` naming

## Critical Constraints

- **Never use pip** — always `uv add`, `uv sync`, `uv run`.
- **Do not add `versioned: true` to MLflow catalog entries** — `MLflowModelStageManager` handles versioning internally.
- **`mlruns/` is gitignored** — MLflow tracking is local by default. For shared tracking, set `mlflow_tracking_uri` in `conf/local/mlflow.yml`.
- **Optuna `load_if_exists: false`** by default — each run starts a fresh study. Set to `true` only when deliberately resuming.
