import numpy as np
import plotly.graph_objects as go
from sklearn.metrics import auc, precision_recall_curve


def plot_precision_recall_curve(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    title: str = "Precision-Recall Curve",
) -> go.Figure:
    """Interactive Plotly Precision-Recall curve with AUC score."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_probs)
    pr_auc = auc(recall, precision)
    baseline = float(np.mean(y_true))

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=recall,
            y=precision,
            mode="lines",
            name=f"PR Curve (AUC = {pr_auc:.3f})",
            line={"width": 2, "color": "#4CAF50"},
        )
    )
    fig.add_shape(
        type="line", x0=0, y0=baseline, x1=1, y1=baseline,
        line={"dash": "dash", "color": "gray", "width": 1},
    )
    fig.add_annotation(
        x=0.95, y=baseline + 0.03,
        text=f"Baseline ({baseline:.2f})",
        showarrow=False, font={"color": "gray", "size": 11},
    )
    fig.update_layout(
        title=title,
        xaxis_title="Recall",
        yaxis_title="Precision",
        xaxis_range=[0, 1],
        yaxis_range=[0, 1.05],
        plot_bgcolor="white",
        paper_bgcolor="white",
        legend={"x": 0.02, "y": 0.05},
    )
    fig.update_xaxes(showgrid=True, gridcolor="LightGray")
    fig.update_yaxes(showgrid=True, gridcolor="LightGray")
    return fig
