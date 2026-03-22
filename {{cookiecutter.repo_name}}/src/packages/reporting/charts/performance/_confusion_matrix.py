import typing as tp

import numpy as np
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: tp.List[str] = ["Not Survived", "Survived"],
    title: str = "Confusion Matrix",
) -> go.Figure:
    """Plotly confusion matrix heatmap with TP/FP/FN/TN annotations."""
    cm = confusion_matrix(y_true, y_pred, labels=[1, 0])
    tn, fp, fn, tp_ = cm.ravel()[::-1]

    z = [[tp_, fn], [fp, tn]]
    text = [[f"TP: {tp_}", f"FN: {fn}"], [f"FP: {fp}", f"TN: {tn}"]]

    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            text=text,
            texttemplate="%{text}",
            colorscale="Blues",
            showscale=True,
            hoverinfo="text",
        )
    )
    fig.update_layout(
        title=title,
        xaxis={"title": "Predicted Label", "tickvals": [0, 1], "ticktext": class_names},
        yaxis={"title": "True Label", "tickvals": [0, 1], "ticktext": class_names},
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    fig.update_yaxes(autorange="reversed")
    return fig
