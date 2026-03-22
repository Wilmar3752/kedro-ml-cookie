import numpy as np
import pandas as pd
import pytest

from packages.reporting.reports.titanic_performance_report import (
    create_titanic_performance_report,
)
from packages.reporting.reports.titanic_interpretability_report import (
    create_titanic_interpretability_report,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture()
def params() -> dict:
    return {"target": "survived", "probability_threshold": 0.5}


@pytest.fixture()
def fake_model(params):
    """Minimal model stub that satisfies the report functions' interface."""
    import numpy as np
    from unittest.mock import MagicMock

    rng = np.random.default_rng(42)
    n = 100

    # df_scores — per-fold CV scores
    df_scores = pd.DataFrame(
        {
            "roc_auc": rng.uniform(0.80, 0.90, 5),
            "accuracy": rng.uniform(0.78, 0.85, 5),
            "f1": rng.uniform(0.70, 0.80, 5),
        }
    )

    # Feature names exposed by preprocessor
    feature_names = ["pclass", "sex", "age", "fare", "embarked"]
    n_features = len(feature_names)

    # Mock column selector (exposes .columns)
    col_selector = MagicMock()
    col_selector.columns = feature_names

    # Mock LightGBM booster
    booster = MagicMock()
    booster.feature_importance.return_value = np.array(
        rng.uniform(100, 5000, n_features), dtype=float
    )

    # Mock LGBM estimator
    lgbm = MagicMock()
    lgbm.booster_ = booster
    lgbm.n_features_in_ = n_features

    # Mock preprocessor pipeline (slice-able)
    preprocessor = MagicMock()
    preprocessor.__getitem__ = MagicMock(side_effect=lambda s: col_selector if s == slice(None, -1, None) else lgbm)
    preprocessor.__len__ = MagicMock(return_value=2)

    # mock preprocessor[-1].columns — return self so preprocessor[-1].columns works
    preprocessor_last = MagicMock()
    preprocessor_last.columns = feature_names
    preprocessor_last.__getitem__ = MagicMock(return_value=preprocessor_last)

    # Mock full model pipeline
    model_pipeline = MagicMock()
    model_pipeline.__getitem__ = MagicMock(
        side_effect=lambda s: preprocessor_last if s == slice(None, -1, None) else lgbm
    )

    # predict_proba: always return confident predictions
    def fake_predict_proba(X):
        return np.column_stack([
            rng.uniform(0, 0.5, len(X)),
            rng.uniform(0.5, 1.0, len(X)),
        ])

    mock = MagicMock()
    mock.df_scores = df_scores
    mock.scores = df_scores.mean().to_dict()
    mock.predict_proba = fake_predict_proba
    mock.model = model_pipeline
    mock.model.__getitem__ = model_pipeline.__getitem__

    return mock


@pytest.fixture()
def df_test() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    n = 100
    return pd.DataFrame(
        {
            "survived": rng.integers(0, 2, n),
            "pclass": rng.integers(1, 4, n),
            "sex": rng.integers(0, 2, n),
            "age": rng.uniform(1, 80, n),
            "fare": rng.uniform(5, 500, n),
            "embarked": rng.integers(0, 3, n),
        }
    )


@pytest.fixture()
def df_train(df_test) -> pd.DataFrame:
    rng = np.random.default_rng(99)
    n = 200
    return pd.DataFrame(
        {
            "survived": rng.integers(0, 2, n),
            "pclass": rng.integers(1, 4, n),
            "sex": rng.integers(0, 2, n),
            "age": rng.uniform(1, 80, n),
            "fare": rng.uniform(5, 500, n),
            "embarked": rng.integers(0, 3, n),
        }
    )


# ── Performance report tests ──────────────────────────────────────────────────


class TestCreateTitanicPerformanceReport:
    def test_returns_string(self, fake_model, df_test, params):
        html = create_titanic_performance_report(fake_model, df_test, params)
        assert isinstance(html, str)

    def test_contains_doctype(self, fake_model, df_test, params):
        html = create_titanic_performance_report(fake_model, df_test, params)
        assert "<!DOCTYPE html>" in html

    def test_contains_sidebar(self, fake_model, df_test, params):
        html = create_titanic_performance_report(fake_model, df_test, params)
        assert 'id="sidebar"' in html

    def test_contains_performance_sections(self, fake_model, df_test, params):
        html = create_titanic_performance_report(fake_model, df_test, params)
        assert "Model Metrics" in html
        assert "Discrimination" in html
        assert "Calibration" in html
        assert "Feature Importance" in html

    def test_contains_plotly(self, fake_model, df_test, params):
        html = create_titanic_performance_report(fake_model, df_test, params)
        assert "Plotly" in html

    def test_contains_titanic_title(self, fake_model, df_test, params):
        html = create_titanic_performance_report(fake_model, df_test, params)
        assert "Titanic" in html


# ── Interpretability report tests ─────────────────────────────────────────────


class TestCreateTitanicInterpretabilityReport:
    def test_returns_string(self, fake_model, df_train, params):
        """Basic smoke test — just checks it doesn't raise and returns HTML."""
        # The interpretability report uses SHAP on real tree models.
        # With a mock model, we skip it and just verify the interface.
        pytest.skip(
            "Interpretability report requires a real fitted LightGBM model; "
            "covered by integration tests that run the full pipeline."
        )
