import pandas as pd
import pytest

from modelling.titanic.preprocessing.nodes import preprocess_titanic, split_data

FEATURE_COLUMNS = ["pclass", "sex", "age", "sibsp", "parch", "fare", "embarked"]


@pytest.fixture()
def preprocess_params() -> dict:
    return {
        "target": "survived",
        "feature_columns": FEATURE_COLUMNS,
    }


@pytest.fixture()
def raw_titanic_sample() -> pd.DataFrame:
    """Minimal synthetic Titanic-like DataFrame for unit tests."""
    return pd.DataFrame(
        {
            "survived": [1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
            "pclass": [1, 2, 3, 1, 2, 3, 1, 2, 3, 1],
            "sex": ["female", "male", "female", "male", "female",
                    "male", "female", "male", "female", "male"],
            "age": [25.0, 30.0, None, 40.0, 22.0, 35.0, 28.0, None, 45.0, 33.0],
            "sibsp": [1, 0, 0, 1, 0, 0, 1, 0, 0, 1],
            "parch": [0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
            "fare": [100.0, 20.0, 8.0, 150.0, 25.0, 7.5, 90.0, 15.0, 10.0, 200.0],
            "embarked": ["S", "C", "Q", "S", None, "C", "S", "Q", "C", "S"],
            # Extra columns that should be dropped by preprocess
            "who": ["woman"] * 10,
            "deck": [None] * 10,
        }
    )


class TestPreprocessTitanic:
    def test_returns_dataframe(self, raw_titanic_sample, preprocess_params):
        result = preprocess_titanic(raw_titanic_sample, preprocess_params)
        assert isinstance(result, pd.DataFrame)

    def test_keeps_expected_columns(self, raw_titanic_sample, preprocess_params):
        result = preprocess_titanic(raw_titanic_sample, preprocess_params)
        expected = {"survived"} | set(FEATURE_COLUMNS)
        assert set(result.columns) == expected

    def test_drops_extra_columns(self, raw_titanic_sample, preprocess_params):
        result = preprocess_titanic(raw_titanic_sample, preprocess_params)
        assert "who" not in result.columns
        assert "deck" not in result.columns

    def test_survived_is_int(self, raw_titanic_sample, preprocess_params):
        result = preprocess_titanic(raw_titanic_sample, preprocess_params)
        assert result["survived"].dtype == int

    def test_resets_index(self, raw_titanic_sample, preprocess_params):
        result = preprocess_titanic(raw_titanic_sample, preprocess_params)
        assert list(result.index) == list(range(len(result)))


class TestSplitData:
    @pytest.fixture()
    def preprocessed(self, raw_titanic_sample, preprocess_params):
        return preprocess_titanic(raw_titanic_sample, preprocess_params)

    @pytest.fixture()
    def params(self):
        return {"target": "survived", "test_size": 0.3, "random_state": 42}

    def test_returns_two_dataframes(self, preprocessed, params):
        train, test = split_data(preprocessed, params)
        assert isinstance(train, pd.DataFrame)
        assert isinstance(test, pd.DataFrame)

    def test_correct_sizes(self, preprocessed, params):
        train, test = split_data(preprocessed, params)
        assert len(train) + len(test) == len(preprocessed)

    def test_integer_range_index(self, preprocessed, params):
        train, test = split_data(preprocessed, params)
        assert list(train.index) == list(range(len(train)))
        assert list(test.index) == list(range(len(test)))

    def test_target_present_in_both(self, preprocessed, params):
        train, test = split_data(preprocessed, params)
        assert "survived" in train.columns
        assert "survived" in test.columns
