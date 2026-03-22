import numpy as np
import pytest

from packages.modelling.evaluate.metrics import (
    compute_binary_classification_metrics,
    compute_multiclass_classification_metrics,
    compute_regression_metrics,
)


class TestBinaryClassificationMetrics:
    @pytest.fixture()
    def perfect_predictions(self):
        y_true = np.array([0, 1, 0, 1, 1, 0])
        y_pred = np.array([0, 1, 0, 1, 1, 0])
        y_score = np.array([0.1, 0.9, 0.2, 0.8, 0.85, 0.15])
        return y_true, y_pred, y_score

    def test_returns_dict(self, perfect_predictions):
        y_true, y_pred, y_score = perfect_predictions
        result = compute_binary_classification_metrics(y_true, y_pred, y_score)
        assert isinstance(result, dict)

    def test_perfect_accuracy(self, perfect_predictions):
        y_true, y_pred, _ = perfect_predictions
        result = compute_binary_classification_metrics(y_true, y_pred)
        assert result["accuracy"] == pytest.approx(1.0)

    def test_roc_auc_included_with_score(self, perfect_predictions):
        y_true, y_pred, y_score = perfect_predictions
        result = compute_binary_classification_metrics(y_true, y_pred, y_score)
        assert "roc_auc" in result

    def test_roc_auc_absent_without_score(self, perfect_predictions):
        y_true, y_pred, _ = perfect_predictions
        result = compute_binary_classification_metrics(y_true, y_pred)
        assert "roc_auc" not in result

    def test_all_values_are_floats(self, perfect_predictions):
        y_true, y_pred, y_score = perfect_predictions
        result = compute_binary_classification_metrics(y_true, y_pred, y_score)
        for key, value in result.items():
            assert isinstance(value, float), f"{key} is not a float: {type(value)}"


class TestRegressionMetrics:
    def test_perfect_r2(self):
        y = np.array([1.0, 2.0, 3.0, 4.0])
        result = compute_regression_metrics(y, y)
        assert result["r2"] == pytest.approx(1.0)
        assert result["mae"] == pytest.approx(0.0)
        assert result["mse"] == pytest.approx(0.0)
        assert result["rmse"] == pytest.approx(0.0)

    def test_returns_expected_keys(self):
        y = np.array([1.0, 2.0, 3.0])
        result = compute_regression_metrics(y, y)
        assert {"mse", "rmse", "mae", "mape", "r2", "explained_variance"}.issubset(
            result.keys()
        )
