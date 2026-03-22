"""Titanic — hyperparameter tuning report (HTML)."""
import typing as tp

import optuna
import optuna.visualization as optuna_viz
import pandas as pd

from ..rendering.html.report_generation import ReportMetaData, generate_html_report
from .identifiers import SectionHeader, Table

MIN_PARAMS_FOR_RANK = 2
MIN_PARAMS_FOR_SLICE = 2


def _get_hyperparameter_importance(study: optuna.Study) -> pd.Series:
    """Return hyperparameter importances as a Series, empty if unavailable."""
    try:
        importances = optuna.importance.get_param_importances(study)
        return pd.Series(importances, name="importance").sort_values(ascending=False)
    except Exception:
        return pd.Series(dtype=float, name="importance")


def create_titanic_hypertune_report(
    study: optuna.Study,
    params: tp.Dict[str, tp.Any],
) -> str:
    """Generate a self-contained HTML hyperparameter tuning report.

    Sections
    --------
    1. Optimization History   — Objective value vs trial number + timeline.
    2. Hyperparameter Analysis — Importance ranking + slice plots.
    3. Exploration Space      — Parallel coordinates + rank plots.

    Args:
        study: Completed Optuna study object.
        params: Unused; kept for API consistency with the reference project.

    Returns:
        Self-contained HTML string.
    """
    importance = _get_hyperparameter_importance(study)

    # ── Always-available plots ────────────────────────────────────────────────
    fig_history = optuna_viz.plot_optimization_history(study)
    fig_history.update_layout(plot_bgcolor="white")

    fig_edf = optuna_viz.plot_edf(study)
    fig_edf.update_layout(plot_bgcolor="white")

    fig_timeline = optuna.visualization.plot_timeline(study)
    fig_timeline.update_layout(plot_bgcolor="white")

    report_structure: tp.Dict = {
        SectionHeader("Optimization History", description="How the best objective value evolved across trials."): {
            "Best value vs trial": fig_history,
            "Optimization timeline": fig_timeline,
            "Empirical distribution": fig_edf,
        },
    }

    # ── Hyperparameter-specific plots ─────────────────────────────────────────
    if len(importance) > 0:
        fig_importance = optuna_viz.plot_param_importances(study)
        fig_importance.update_layout(plot_bgcolor="white")

        fig_slice = optuna_viz.plot_slice(study)
        fig_slice.update_layout(plot_bgcolor="white")

        fig_parallel = optuna_viz.plot_parallel_coordinate(study)
        fig_parallel.update_layout(plot_bgcolor="white")

        importance_table = Table(
            importance.reset_index().rename(columns={"index": "parameter"}),
            title="Hyperparameter Importance",
            show_index=False,
            width=50,
        )

        report_structure[
            SectionHeader("Hyperparameter Importance", description="Relative importance of each hyperparameter.")
        ] = {
            "Importance table": importance_table,
            "Importance plot": fig_importance,
        }

        exploration_section: tp.Dict = {
            "Parallel coordinates": fig_parallel,
            "Slice (all parameters)": fig_slice,
        }

        if len(importance) >= MIN_PARAMS_FOR_RANK:
            top2 = importance.index[:2].tolist()
            fig_rank = optuna_viz.plot_rank(study, params=top2)
            fig_rank.update_layout(plot_bgcolor="white")
            exploration_section["Rank (top 2 params)"] = fig_rank

        if len(importance) >= MIN_PARAMS_FOR_SLICE:
            top_params = importance.index[:min(4, len(importance))].tolist()
            fig_slice_top = optuna_viz.plot_slice(study, params=top_params)
            fig_slice_top.update_layout(plot_bgcolor="white")
            exploration_section["Slice (top parameters)"] = fig_slice_top

        report_structure[
            SectionHeader("Exploration Space", description="How the search explored the hyperparameter space.")
        ] = exploration_section

    else:
        report_structure[
            SectionHeader("Hyperparameter Analysis")
        ] = {
            "Status": Table(
                pd.DataFrame({"Note": ["No hyperparameters were optimised — all trials used fixed values."]}),
                show_index=False,
            )
        }

    meta = ReportMetaData(title="Titanic Survival — Hyperparameter Tuning Report")
    return generate_html_report(
        report_structure=report_structure,
        report_meta_data=meta,
        max_table_of_content_depth=2,
    )
