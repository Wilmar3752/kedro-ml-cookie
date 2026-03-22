"""SHAP-based interpretability charts.

All functions return objects compatible with the HTML rendering layer:
- matplotlib Figures (embedded as base64 PNG)
- plotly Figures (embedded as interactive HTML)
"""
import typing as tp

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import shap
from matplotlib.figure import Figure

matplotlib.use("Agg")


def plot_shap_beeswarm(
    shap_values: shap.Explanation,
    max_display: int = 20,
    title: str = "SHAP Feature Importance (Beeswarm)",
) -> Figure:
    """Matplotlib beeswarm plot — rendered as base64 PNG in the HTML report."""
    fig, ax = plt.subplots(figsize=(10, max(5, min(max_display, len(shap_values.feature_names)) * 0.4 + 2)))
    shap.plots.beeswarm(shap_values, max_display=max_display, show=False)
    fig = plt.gcf()
    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    return fig


def plot_shap_feature_importance(
    shap_values: shap.Explanation,
    top_n: int = 20,
    title: str = "SHAP Mean Absolute Feature Importance",
) -> go.Figure:
    """Plotly bar chart of mean |SHAP| values per feature."""
    feature_names = shap_values.feature_names
    mean_abs_shap = np.abs(shap_values.values).mean(axis=0)

    df = (
        pd.DataFrame({"feature": feature_names, "importance": mean_abs_shap})
        .sort_values("importance", ascending=False)
        .head(top_n)
        .sort_values("importance", ascending=True)
    )

    fig = go.Figure(
        go.Bar(
            x=df["importance"],
            y=df["feature"],
            orientation="h",
            marker_color="#FF9800",
            text=df["importance"].round(4),
            textposition="outside",
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="Mean |SHAP value|",
        yaxis_title="Feature",
        plot_bgcolor="white",
        paper_bgcolor="white",
        height=max(300, len(df) * 28),
        margin={"l": 160},
    )
    fig.update_xaxes(showgrid=True, gridcolor="LightGray")
    return fig


def plot_shap_dependence(
    shap_values: shap.Explanation,
    feature: str,
    title: tp.Optional[str] = None,
) -> go.Figure:
    """Plotly scatter plot of SHAP value vs feature value for a single feature."""
    feature_names = list(shap_values.feature_names)
    if feature not in feature_names:
        raise ValueError(f"Feature '{feature}' not in SHAP explanation.")

    idx = feature_names.index(feature)
    x_vals = shap_values.data[:, idx]
    y_vals = shap_values.values[:, idx]

    fig = go.Figure(
        go.Scatter(
            x=x_vals,
            y=y_vals,
            mode="markers",
            marker={"color": "#2196F3", "opacity": 0.5, "size": 5},
            hovertemplate=f"{feature}: %{{x:.3f}}<br>SHAP: %{{y:.4f}}<extra></extra>",
        )
    )
    fig.add_hline(y=0, line_dash="dash", line_color="gray", line_width=1)
    fig.update_layout(
        title=title or f"SHAP Dependence — {feature}",
        xaxis_title=feature,
        yaxis_title="SHAP value",
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    fig.update_xaxes(showgrid=True, gridcolor="LightGray")
    fig.update_yaxes(showgrid=True, gridcolor="LightGray")
    return fig
