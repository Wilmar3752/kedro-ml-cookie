# CLAUDE.md — kedro-ml-cookie

## What This Repository Is

This is a **Cookiecutter template** for production-grade MLOps projects. It is NOT a directly runnable project. When instantiated via `cookiecutter`, everything inside `{{cookiecutter.repo_name}}/` becomes a real Kedro project. All work on business logic lives inside that directory.

```
kedro-ml-cookie/
├── cookiecutter.json              # Template variables
├── hooks/post_gen_project.py      # Post-generation hook (git init, uv sync)
└── {{cookiecutter.repo_name}}/    # The actual generated project
```

## Generating a Project

```bash
cookiecutter https://github.com/wilmar/kedro-ml-cookie
# or locally:
cookiecutter .
```

`project_name` is auto-derived from `repo_name` (spaces + title-cased). Do not add a separate prompt for it.

---

## Generated Project: Architecture Overview

### Stack
| Layer | Tool |
|---|---|
| Pipeline framework | Kedro 1.1.1 |
| MLflow integration | kedro-mlflow 2.0.0 |
| Model registry | MLflow 3.x (alias-based) |
| Hyperparameter tuning | Optuna 4.x |
| Base model | LightGBM 4.x |
| Feature engineering | feature-engine |
| Package manager | uv (never use pip) |
| Config | OmegaConfigLoader (Omega/YAML) |

### Source Layout

```
src/
├── modelling/          # Kedro application (pipelines, nodes, settings)
│   ├── settings.py
│   ├── pipeline_registry.py
│   ├── titanic/        # Binary classification example
│   └── california/     # Regression example
└── packages/           # Kedro-independent reusable library
    ├── modelling/
    │   ├── evaluate/       # Metric computation
    │   ├── models/
    │   │   ├── supervised/ # BaseSklearnCompatibleModel (700 lines)
    │   │   ├── preprocessors/
    │   │   └── mlflow/     # MLflowModelStageManager
    │   └── transformers/
    ├── reporting/          # HTML report generation (Plotly + Jinja2)
    └── python_utils/       # Object injection from YAML
```

`packages/` has no Kedro dependency and can be imported anywhere. `modelling/` is the Kedro app.

### Pipeline Structure (both Titanic and California follow this pattern)

```
preprocessing → hypertune → reporting
```

Each stage has its own `pipeline.py` and `nodes.py` under `src/modelling/{dataset}/{stage}/`.

Master registry in `src/modelling/pipeline_registry.py` registers:
- `{dataset}_preprocessing`, `{dataset}_hypertune`, `{dataset}_reporting`
- `e2e_{dataset}` (all three combined)
- `__default__` (both datasets end-to-end)

---

## Key Conventions

### MLflow Model Registry — Aliases, Not Stages

MLflow 3.x removed stages (`Staging`, `Production`). This project uses **aliases**:
- `@champion` → production model (previously "Production")
- `@challenger` → newly trained model (previously "Staging")

Never re-introduce `MlflowClient().transition_model_version_stage()` — it is deprecated.

`MLflowModelStageManager` in `src/packages/modelling/models/mlflow/registry.py` handles all registry operations as a Kedro catalog dataset.

Catalog entry pattern:
```yaml
titanic_registered_model:
  type: packages.modelling.models.mlflow.registry.MLflowModelStageManager
  model_name: titanic
  load_alias: champion
  save_alias: challenger
  pyfunc_predict_fn: predict
```

### Object Injection from YAML

Classes are instantiated dynamically from YAML using `load_object()` in `src/packages/python_utils/load/object_injection.py`. Optuna trial expressions (e.g., `"trial.suggest_int('n_estimators', 50, 500)"`) are evaluated at tuning time. This is the mechanism behind the `parameters/{dataset}/hypertune.yml` pipeline config.

### AutoConfigFeaturePreprocessors

`BaseSklearnCompatibleModel` automatically injects column lists (numeric, categorical, columns with NaNs) into feature-engine transformers at `fit()` time. Do not manually pass variable lists to these transformers in parameters — they are populated programmatically.

Supported: `CategoricalImputer`, `MeanMedianImputer`, `RareLabelEncoder`, `OrdinalEncoder`, `OneHotEncoder`, `AddMissingIndicator`.

### Reproducibility

Global seed is `42`, set via `conf/base/globals.yml` (`global_seed: 42`). All CV splits use `random_state=42`. The `packages/reproducibility/set_seed.py` module seeds `random` and `numpy`. Do not introduce new randomness without threading this seed through.

### Config Patterns

`OmegaConfigLoader` is configured in `settings.py`. Config resolution order:
- `conf/base/` (globals, catalog subdirs, parameters subdirs, mlflow)
- `conf/local/` (gitignored, for secrets/overrides)

`catalog/` and `parameters/` use glob patterns so subdirectory files are auto-discovered. When adding a new dataset, add catalog files under `conf/base/catalog/modelling/{name}/` — do not put everything in a single flat file.

---

## Common Commands (inside generated project)

```bash
# Setup
make setup       # uv venv --python 3.11
make install     # uv sync

# Run pipelines
make run-titanic              # e2e titanic
make run-california           # e2e california
make run                      # both models
make run-titanic-hypertune    # single stage

# Dev tools
make lint        # ruff check src/
make format      # ruff format src/
make test        # pytest tests/ --cov=src -v
make viz         # kedro viz run
make jupyter     # jupyter lab

# MLflow UI
kedro mlflow ui  # opens at localhost:5000
```

---

## Data Layer Conventions

```
data/01_raw/        # Committed raw CSVs — never modify
data/02_intermediate → 09_tracking/   # All gitignored, rebuilt by pipeline
data/06_models/     # Fitted model artifacts (also registered in MLflow)
```

Never commit data beyond `01_raw/`. Never remove `.gitkeep` files from empty layers.

---

## Adding a New Dataset/Model

1. Create `src/modelling/{name}/` with `preprocessing/`, `hypertune/`, `reporting/` subfolders (each with `pipeline.py` + `nodes.py`)
2. Create a `src/modelling/{name}/pipeline_registry.py` mirroring the titanic or california pattern
3. Add catalog files under `conf/base/catalog/modelling/{name}/`
4. Add parameter files under `conf/base/parameters/modelling/{name}/`
5. Register in the master `src/modelling/pipeline_registry.py`
6. Add Makefile targets following the `run-{name}-{stage}` naming convention

---

## Critical Constraints

- **Never use pip** — this project uses `uv` exclusively. Always `uv add`, `uv sync`, `uv run`.
- **Do not add `versioned: true` or `metadata:` to MLflow catalog entries** — `MLflowModelStageManager` handles versioning internally.
- **`mlruns/` is gitignored** — MLflow runs are local. For shared tracking, set `mlflow_tracking_uri` to a remote server in `conf/local/mlflow.yml`.
- **Optuna `load_if_exists: false`** — each training run starts a fresh study. Set to `true` only when deliberately resuming a study.
- **Training-only attributes are stripped before MLflow logging** — `X_train`, `y_train`, `df_scores`, `hypertune_results` are removed in `MLflowModelStageManager.save()`. Do not rely on them being present after loading from registry.
- **`_copy_without_render`** in `cookiecutter.json` excludes `*.py`, `*.yml`, `*.yaml`, `*.lock`, `*.html`, `*.csv` from Jinja rendering. Only use cookiecutter variables in filenames and in files not covered by this list.
