# kedro-ml-cookie

A [cookiecutter](https://cookiecutter.readthedocs.io/) template for end-to-end ML projects using [Kedro](https://kedro.org/) and [uv](https://docs.astral.sh/uv/).

Includes two working examples out of the box:

- **Titanic** — binary classification with LGBMClassifier, StratifiedKFold CV, ROC/PR charts, SHAP interpretability
- **California Housing** — regression with LGBMRegressor, KFold CV, actual-vs-predicted and residual charts

Each example uses the same pipeline structure: `preprocessing → hypertune → reporting`.

---

## Requirements

- [cookiecutter](https://cookiecutter.readthedocs.io/) >= 2.0
- [uv](https://docs.astral.sh/uv/) >= 0.9.21

```bash
pip install cookiecutter
curl -LsSf https://astral.sh/uv/install.sh | sh
```

---

## Usage

```bash
cookiecutter gh:your-org/kedro-ml-cookie
```

Or from a local clone:

```bash
cookiecutter path/to/kedro-ml-cookie
```

You will be prompted for:

| Variable | Default | Description |
|---|---|---|
| `project_name` | `My ML Project` | Human-readable project name |
| `repo_name` | `my-ml-project` | Directory name and Python package name |
| `description` | `End-to-end ML project using Kedro and uv` | One-line description |
| `author_name` | `Your Name` | Author |
| `python_version` | `3.11` | Python version for `requires-python` |

After generation, `uv sync` runs automatically to create the virtual environment.

---

## Generated project structure

```
<repo_name>/
├── conf/
│   └── base/
│       ├── catalog/modelling/{titanic,california}/   # per-model catalog YAMLs
│       ├── parameters/modelling/{titanic,california}/ # per-model parameter YAMLs
│       └── globals.yml                               # shared globals (n_trials, etc.)
├── data/
│   ├── 01_raw/          # titanic.csv, california_housing.csv (committed)
│   ├── 06_models/       # fitted model + Optuna study (versioned)
│   └── 08_reporting/    # 6 self-contained HTML reports
├── src/
│   ├── modelling/       # Kedro app (settings, pipeline_registry)
│   │   ├── titanic/     # preprocessing, hypertune, reporting pipelines
│   │   └── california/  # preprocessing, hypertune, reporting pipelines
│   └── packages/        # Reusable library (model classes, charts, reports)
│       ├── modelling/   # BinaryClassifierSklearnPipeline, RegressionSklearnPipeline
│       └── reporting/   # Plotly charts, Jinja2 HTML engine, report builders
├── pyproject.toml
├── Makefile
└── uv.lock
```

---

## Running the generated project

```bash
cd <repo_name>
uv run kedro run                          # default: all pipelines
uv run kedro run --pipeline e2e_titanic
uv run kedro run --pipeline e2e_california
```

See the generated `README.md` inside the project for full documentation.
