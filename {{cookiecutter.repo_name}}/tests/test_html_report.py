import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure

from packages.reporting.rendering.html.report_generation import (
    ReportMetaData,
    generate_html_report,
)
from packages.reporting.reports.identifiers import Code, SectionHeader, Table
from packages.reporting.reports.titanic_report import create_titanic_html_report


@pytest.fixture()
def sample_model_report() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "metric": ["roc_auc", "accuracy", "f1"],
            "test_score": [0.85, 0.80, 0.75],
            "cv_mean": [0.84, 0.79, 0.74],
            "cv_std": [0.02, 0.02, 0.03],
            "cv_min": [0.82, 0.77, 0.71],
            "cv_max": [0.87, 0.82, 0.78],
        }
    )


@pytest.fixture()
def sample_cv_scores() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "roc_auc": rng.uniform(0.80, 0.87, 5),
            "accuracy": rng.uniform(0.77, 0.83, 5),
            "f1": rng.uniform(0.70, 0.78, 5),
        }
    )


@pytest.fixture()
def sample_model_metrics() -> dict:
    return {"roc_auc": 0.85, "accuracy": 0.80, "f1": 0.75}


class TestGenerateHtmlReport:
    def test_returns_string(self, sample_model_report, sample_cv_scores, sample_model_metrics):
        html = generate_html_report(
            report_structure={
                "Model Summary": sample_model_report,
            },
        )
        assert isinstance(html, str)

    def test_html_contains_doctype(self, sample_model_report):
        html = generate_html_report({"Summary": sample_model_report})
        assert "<!DOCTYPE html>" in html

    def test_html_contains_sidebar(self, sample_model_report):
        html = generate_html_report({"Summary": sample_model_report})
        assert 'id="sidebar"' in html

    def test_html_contains_plotly(self, sample_model_report):
        html = generate_html_report({"Summary": sample_model_report})
        assert "Plotly" in html

    def test_html_contains_table_data(self, sample_model_report):
        html = generate_html_report({"Summary": sample_model_report})
        assert "roc_auc" in html

    def test_report_meta_title(self):
        html = generate_html_report(
            report_structure={"Section": pd.DataFrame({"a": [1]})},
            report_meta_data={"title": "My Custom Report"},
        )
        assert "My Custom Report" in html

    def test_nested_sections(self, sample_model_report, sample_cv_scores):
        html = generate_html_report(
            report_structure={
                "Results": {
                    "Test Scores": sample_model_report,
                    "CV Scores": sample_cv_scores,
                }
            }
        )
        assert "Results" in html
        assert "Test Scores" in html
        assert "CV Scores" in html

    def test_section_header_with_description(self, sample_model_report):
        html = generate_html_report(
            report_structure={
                SectionHeader("My Section", description="Detailed description"): sample_model_report
            }
        )
        assert "My Section" in html
        assert "Detailed description" in html

    def test_table_identifier(self, sample_model_report):
        html = generate_html_report(
            report_structure={
                "Metrics": Table(sample_model_report, title="Performance Table")
            }
        )
        assert "Performance Table" in html

    def test_code_identifier(self):
        html = generate_html_report(
            report_structure={
                "Config": Code(code="print('hello')", language="python")
            }
        )
        assert "print" in html

    def test_matplotlib_figure(self):
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        ax.plot([0, 1], [0, 1])
        html = generate_html_report({"ROC Curve": fig})
        assert "data:image/png;base64," in html
        plt.close(fig)


class TestReportMetaData:
    def test_default_title(self):
        meta = ReportMetaData()
        assert meta.title == "Report"

    def test_custom_title(self):
        meta = ReportMetaData(title="Titanic Report")
        assert meta.title == "Titanic Report"

    def test_from_dict(self):
        meta = ReportMetaData.from_input({"title": "From Dict"})
        assert meta.title == "From Dict"

    def test_from_none(self):
        meta = ReportMetaData.from_input(None)
        assert isinstance(meta, ReportMetaData)

    def test_from_instance(self):
        original = ReportMetaData(title="Original")
        meta = ReportMetaData.from_input(original)
        assert meta is original


class TestCreateTitanicHtmlReport:
    def test_returns_string(self, sample_model_report, sample_cv_scores, sample_model_metrics):
        html = create_titanic_html_report(
            sample_model_report, sample_cv_scores, sample_model_metrics
        )
        assert isinstance(html, str)

    def test_contains_titanic_title(
        self, sample_model_report, sample_cv_scores, sample_model_metrics
    ):
        html = create_titanic_html_report(
            sample_model_report, sample_cv_scores, sample_model_metrics
        )
        assert "Titanic" in html

    def test_contains_performance_section(
        self, sample_model_report, sample_cv_scores, sample_model_metrics
    ):
        html = create_titanic_html_report(
            sample_model_report, sample_cv_scores, sample_model_metrics
        )
        assert "Model Performance" in html

    def test_contains_cv_section(
        self, sample_model_report, sample_cv_scores, sample_model_metrics
    ):
        html = create_titanic_html_report(
            sample_model_report, sample_cv_scores, sample_model_metrics
        )
        assert "Cross-Validation" in html

    def test_contains_metric_values(
        self, sample_model_report, sample_cv_scores, sample_model_metrics
    ):
        html = create_titanic_html_report(
            sample_model_report, sample_cv_scores, sample_model_metrics
        )
        assert "roc_auc" in html
