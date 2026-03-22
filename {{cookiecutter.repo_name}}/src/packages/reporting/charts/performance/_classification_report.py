import numpy as np
import plotly.graph_objects as go
from sklearn.metrics import classification_report


def plot_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list = ["Not Survived", "Survived"],
    title: str = "Classification Report",
) -> go.Figure:
    """Plotly heatmap of precision / recall / f1-score per class."""
    report = classification_report(y_true, y_pred, output_dict=True)
    class_labels = [k for k in report if k not in ("accuracy", "macro avg", "weighted avg")]
    metrics = ["precision", "recall", "f1-score"]

    z = np.array([[report[l][m] for m in metrics] for l in class_labels])
    # Map numeric labels to friendly names if possible
    y_labels = []
    for label in class_labels:
        try:
            idx = int(float(label))
            y_labels.append(class_names[idx] if idx < len(class_names) else label)
        except (ValueError, TypeError):
            y_labels.append(label)

    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=metrics,
            y=y_labels,
            colorscale="Blues",
            zmin=0, zmax=1,
            colorbar={"title": "Score"},
            hovertemplate="Class: %{y}<br>%{x}: %{z:.3f}<extra></extra>",
        )
    )
    for i, label in enumerate(y_labels):
        for j, metric in enumerate(metrics):
            val = z[i, j]
            fig.add_annotation(
                x=metric, y=label,
                text=f"{val:.3f}",
                showarrow=False,
                font={"color": "white" if val > 0.5 else "black", "size": 13},
            )
    fig.update_layout(
        title=title,
        xaxis_title="Metric",
        yaxis_title="Class",
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    return fig
