import typing as tp

import pandas as pd
import plotly.graph_objects as go


def plot_feature_importance(
    feature_names: tp.List[str],
    importances: tp.List[float],
    title: str = "Feature Importance (Gain)",
    top_n: int = 20,
) -> go.Figure:
    """Plotly horizontal bar chart of feature importances sorted by value."""
    df = (
        pd.DataFrame({"feature": feature_names, "importance": importances})
        .sort_values("importance", ascending=False)
        .head(top_n)
        .sort_values("importance", ascending=True)  # flip for horizontal bar readability
    )

    colors = [
        "#2196F3" if imp >= df["importance"].median() else "#607D8B"
        for imp in df["importance"]
    ]

    fig = go.Figure(
        go.Bar(
            x=df["importance"],
            y=df["feature"],
            orientation="h",
            marker_color=colors,
            text=df["importance"].round(1),
            textposition="outside",
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="Importance (Gain)",
        yaxis_title="Feature",
        plot_bgcolor="white",
        paper_bgcolor="white",
        height=max(300, len(df) * 28),
        margin={"l": 160},
    )
    fig.update_xaxes(showgrid=True, gridcolor="LightGray")
    return fig
