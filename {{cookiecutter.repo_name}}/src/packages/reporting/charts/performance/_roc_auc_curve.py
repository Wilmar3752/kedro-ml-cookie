import numpy as np
import plotly.graph_objects as go
from sklearn.metrics import auc, roc_curve


def plot_roc_auc_curve(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    title: str = "ROC Curve",
) -> go.Figure:
    """Interactive Plotly ROC curve with AUC score and threshold on hover."""
    fpr, tpr, thresholds = roc_curve(y_true, y_probs)
    roc_auc = auc(fpr, tpr)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=fpr,
            y=tpr,
            mode="lines",
            name=f"ROC (AUC = {roc_auc:.3f})",
            line={"width": 2, "color": "#2196F3"},
            customdata=thresholds,
            hovertemplate=(
                "<b>ROC Point</b><br>"
                "FPR: %{x:.3f}<br>TPR: %{y:.3f}<br>"
                "Threshold: %{customdata:.3f}<extra></extra>"
            ),
        )
    )
    fig.add_shape(
        type="line", x0=0, y0=0, x1=1, y1=1,
        line={"dash": "dash", "color": "gray", "width": 1},
    )
    fig.update_layout(
        title=title,
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        xaxis_range=[0, 1],
        yaxis_range=[0, 1.02],
        plot_bgcolor="white",
        paper_bgcolor="white",
        legend={"x": 0.6, "y": 0.15},
    )
    fig.update_xaxes(showgrid=True, gridcolor="LightGray")
    fig.update_yaxes(showgrid=True, gridcolor="LightGray")
    return fig
