import numpy as np
import plotly.graph_objects as go
from sklearn.calibration import calibration_curve


def plot_calibration_curve(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    n_bins: int = 10,
    strategy: str = "quantile",
    title: str = "Calibration Curve",
) -> go.Figure:
    """Plotly calibration curve — predicted probability vs fraction of positives."""
    prob_true, prob_pred = calibration_curve(y_true, y_probs, n_bins=n_bins, strategy=strategy)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=[0, 1], y=[0, 1],
            mode="lines",
            name="Perfect calibration",
            line={"dash": "dash", "color": "gray", "width": 1},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=prob_pred,
            y=prob_true,
            mode="lines+markers",
            name="Model calibration",
            line={"width": 2, "color": "#9C27B0"},
            marker={"size": 7},
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="Mean Predicted Probability",
        yaxis_title="Fraction of Positives",
        xaxis_range=[0, 1],
        yaxis_range=[0, 1],
        plot_bgcolor="white",
        paper_bgcolor="white",
        legend={"x": 0.05, "y": 0.9},
    )
    fig.update_xaxes(showgrid=True, gridcolor="LightGray")
    fig.update_yaxes(showgrid=True, gridcolor="LightGray")
    return fig
