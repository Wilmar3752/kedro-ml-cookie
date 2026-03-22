# {{cookiecutter.project_name}}

{{cookiecutter.description}}

Demonstrates best practices for:
- YAML-driven sklearn pipelines with [Optuna](https://optuna.org/) hyperparameter tuning
- Stratified k-fold cross-validation with early stopping and pruning
- Auto-configured feature preprocessing (numeric imputation, OHE, scaling)
- SHAP-based model interpretability
- Self-contained HTML reports (no image files) with interactive Plotly charts

---

## Requirements

- [uv](https://docs.astral.sh/uv/) >= 0.9.21
- Python 3.11

Install uv if you don't have it:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

---

## Setup

```bash
git clone <repo-url>
cd {{cookiecutter.repo_name}}

uv sync
```

`uv sync` reads `pyproject.toml` and `uv.lock`, creates `.venv/`, and installs everything including dev dependencies. No manual activation needed — all commands use `uv run`.

---

## Running pipelines

### Individual stages

```bash
# 1. Load raw data, clean features, train/test split
uv run kedro run --pipeline preprocessing

# 2. Optuna CV search, final fit, evaluation, predictions
uv run kedro run --pipeline hypertune

# 3. Generate the three HTML reports
uv run kedro run --pipeline reporting
```

### End-to-end

```bash
# preprocessing + hypertune + reporting
uv run kedro run --pipeline e2e_modelling

# same as e2e_modelling
uv run kedro run
```

### Makefile shortcuts

```bash
make install           # uv sync
make run-preprocessing
make run-hypertune
make run-reporting
make run-e2e           # e2e_modelling
make run               # default
```

---

## Outputs

| Layer | Path | Content |
|---|---|---|
| `01_raw` | `data/01_raw/titanic.csv` | Raw Titanic CSV |
| `03_primary` | `data/03_primary/titanic_preprocessed.parquet` | Cleaned features |
| `05_model_input` | `data/05_model_input/titanic_train.parquet` | Train split |
| `05_model_input` | `data/05_model_input/titanic_test.parquet` | Test split |
| `06_models` | `data/06_models/titanic_model.pkl` | Fitted model (versioned) |
| `06_models` | `data/06_models/titanic_study.pkl` | Optuna study (versioned) |
| `07_model_output` | `data/07_model_output/predictions.csv` | Predictions + probabilities |
| `08_reporting` | `data/08_reporting/performance_report.html` | ROC, PR curve, confusion matrix, calibration, feature importance |
| `08_reporting` | `data/08_reporting/interpretability_report.html` | SHAP beeswarm, dependence plots |
| `08_reporting` | `data/08_reporting/hypertune_report.html` | Optuna optimization history and parameter analysis |
| `09_tracking` | `data/09_tracking/metrics.json` | Test-set metrics |

Open any `.html` file in a browser — each is fully self-contained with embedded Plotly JS and a sidebar navigation.

---

## Project structure

```
src/
├── titanic/                         # Kedro application
│   ├── pipelines/
│   │   ├── preprocessing/           # load → preprocess → split
│   │   ├── hypertune/               # train → evaluate → predict → cv_report → study
│   │   └── reporting/               # 3 HTML report nodes
│   └── pipeline_registry.py         # preprocessing, hypertune, reporting, e2e_modelling
└── packages/                        # Reusable library (independent of Kedro)
    ├── modelling/
    │   ├── models/supervised/       # BinaryClassifierSklearnPipeline (Optuna + CV)
    │   ├── models/preprocessors/    # AutoConfigFeaturePreprocessors
    │   └── evaluate/                # compute_binary_classification_metrics
    └── reporting/
        ├── charts/                  # Plotly chart functions (performance + SHAP)
        ├── rendering/html/          # Jinja2 HTML rendering engine
        └── reports/                 # titanic_performance / interpretability / hypertune report
```

---

## Configuration

Parameters live in `conf/base/parameters/`:

| File | Controls |
|---|---|
| `modelling/hypertune.yml` | Pipeline steps, Optuna trials, CV strategy, hyperparameter search space |
| `data_processing.yml` | Test size, random seed, target column |

The hyperparameter search space uses Optuna's `trial.suggest_*` syntax directly in YAML:

```yaml
# conf/base/parameters/modelling/hypertune.yml
pipeline:
  - class: lightgbm.LGBMClassifier
    kwargs:
      n_estimators: "trial.suggest_int('n_estimators', 100, 500)"
      learning_rate: "trial.suggest_float('learning_rate', 0.01, 0.3, log=True)"
      num_leaves: "trial.suggest_int('num_leaves', 20, 150)"
```

---

## Development

```bash
make test      # pytest with coverage
make lint      # ruff check
make format    # ruff format
make viz       # Kedro Viz — pipeline graph in the browser
make jupyter   # JupyterLab
```
