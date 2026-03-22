import pytest

from packages.python_utils.load.object_injection import load_object, load_estimator


class TestLoadObject:
    def test_loads_kfold(self):
        obj = load_object(
            {
                "class": "sklearn.model_selection.KFold",
                "kwargs": {"n_splits": 3},
            }
        )
        from sklearn.model_selection import KFold
        assert isinstance(obj, KFold)
        assert obj.n_splits == 3

    def test_loads_object_without_kwargs(self):
        obj = load_object({"class": "sklearn.model_selection.KFold"})
        from sklearn.model_selection import KFold
        assert isinstance(obj, KFold)

    def test_raises_on_invalid_class(self):
        with pytest.raises(Exception):
            load_object({"class": "nonexistent.module.Class"})


class TestLoadEstimator:
    def test_loads_lgbm(self):
        obj = load_estimator(
            {
                "class": "lightgbm.LGBMClassifier",
                "kwargs": {"n_estimators": 10, "verbose": -1},
            }
        )
        assert hasattr(obj, "fit")
        assert hasattr(obj, "predict")
