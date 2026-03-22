import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def plot_predictions_distribution(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    title: str = "Prediction Probability Distribution",
) -> go.Figure:
    """Plotly histogram of survival probabilities split by true class."""
    survived_probs = y_probs[y_true == 1]
    not_survived_probs = y_probs[y_true == 0]

    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=not_survived_probs,
            name="Not Survived",
            opacity=0.65,
            nbinsx=30,
            marker_color="#F44336",
        )
    )
    fig.add_trace(
        go.Histogram(
            x=survived_probs,
            name="Survived",
            opacity=0.65,
            nbinsx=30,
            marker_color="#2196F3",
        )
    )
    fig.add_vline(x=0.5, line_dash="dash", line_color="black", annotation_text="Threshold 0.5")
    fig.update_layout(
        title=title,
        xaxis_title="Predicted Survival Probability",
        yaxis_title="Count",
        barmode="overlay",
        plot_bgcolor="white",
        paper_bgcolor="white",
        legend={"x": 0.7, "y": 0.9},
    )
    fig.update_xaxes(range=[0, 1], showgrid=True, gridcolor="LightGray")
    fig.update_yaxes(showgrid=True, gridcolor="LightGray")
    return fig
