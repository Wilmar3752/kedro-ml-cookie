"""Plotly charts for regression model performance."""
import numpy as np
import plotly.graph_objects as go


def plot_actual_vs_predicted(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Actual vs Predicted",
) -> go.Figure:
    """Scatter of actual vs predicted values with a perfect-fit diagonal."""
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    lo = min(y_true.min(), y_pred.min())
    hi = max(y_true.max(), y_pred.max())

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=y_true, y=y_pred, mode="markers",
        marker=dict(opacity=0.35, size=4, color="steelblue"),
        name="Predictions",
    ))
    fig.add_trace(go.Scatter(
        x=[lo, hi], y=[lo, hi], mode="lines",
        line=dict(color="red", dash="dash"),
        name="Perfect fit",
    ))
    fig.update_layout(
        title=title,
        xaxis_title="Actual",
        yaxis_title="Predicted",
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    return fig


def plot_residuals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Residuals vs Predicted",
) -> go.Figure:
    """Scatter of residuals (y_true − y_pred) against predicted values."""
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    residuals = y_true - y_pred

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=y_pred, y=residuals, mode="markers",
        marker=dict(opacity=0.35, size=4, color="steelblue"),
        name="Residuals",
    ))
    fig.add_hline(y=0, line_dash="dash", line_color="red")
    fig.update_layout(
        title=title,
        xaxis_title="Predicted",
        yaxis_title="Residual",
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    return fig


def plot_residual_distribution(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Residual Distribution",
) -> go.Figure:
    """Histogram of residuals."""
    residuals = np.asarray(y_true).ravel() - np.asarray(y_pred).ravel()

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=residuals, nbinsx=60,
        marker_color="steelblue", opacity=0.75,
        name="Residuals",
    ))
    fig.update_layout(
        title=title,
        xaxis_title="Residual",
        yaxis_title="Count",
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    return fig


def plot_distribution_comparison(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Actual vs Predicted Distribution",
) -> go.Figure:
    """Overlaid histograms of the actual and predicted distributions."""
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=y_true, nbinsx=60, name="Actual",
        marker_color="steelblue", opacity=0.6,
    ))
    fig.add_trace(go.Histogram(
        x=y_pred, nbinsx=60, name="Predicted",
        marker_color="darkorange", opacity=0.6,
    ))
    fig.update_layout(
        barmode="overlay",
        title=title,
        xaxis_title="Value",
        yaxis_title="Count",
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    return fig


def plot_prediction_error_colored(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Actual vs Predicted (colored by |error|)",
) -> go.Figure:
    """Scatter of actual vs predicted, colored by absolute error magnitude."""
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    abs_error = np.abs(y_true - y_pred)
    lo = min(y_true.min(), y_pred.min())
    hi = max(y_true.max(), y_pred.max())

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=y_true, y=y_pred, mode="markers",
        marker=dict(
            color=abs_error,
            colorscale="RdYlGn_r",
            colorbar=dict(title="|Error|"),
            opacity=0.5, size=4,
        ),
        name="Predictions",
    ))
    fig.add_trace(go.Scatter(
        x=[lo, hi], y=[lo, hi], mode="lines",
        line=dict(color="black", dash="dash"),
        name="Perfect fit",
    ))
    fig.update_layout(
        title=title,
        xaxis_title="Actual",
        yaxis_title="Predicted",
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    return fig


def plot_cumulative_error(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Cumulative Absolute Error",
) -> go.Figure:
    """Cumulative sum of sorted absolute errors, normalized to [0, 1]."""
    abs_errors = np.sort(np.abs(np.asarray(y_true).ravel() - np.asarray(y_pred).ravel()))
    cumulative = np.cumsum(abs_errors)
    n = len(abs_errors)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=np.arange(1, n + 1) / n,
        y=cumulative / cumulative[-1],
        mode="lines",
        line=dict(color="steelblue"),
        name="Cumulative |error|",
    ))
    fig.update_layout(
        title=title,
        xaxis_title="Fraction of samples (sorted by |error|)",
        yaxis_title="Cumulative error (normalized)",
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    return fig
