"""Functions to generate a self-contained HTML report."""

from __future__ import annotations

import logging
import typing as tp
from datetime import datetime
from pathlib import Path

from jinja2 import DictLoader, Environment, FileSystemLoader, Template
from plotly.offline.offline import get_plotlyjs

from .renderables.protocols import THtmlRenderableDict
from .rendering import TSectionDescription, render_report
from .templates.templates_loader import DefaultTemplate

logger = logging.getLogger(__name__)
DEFAULT_ENCODING = "utf-8"
_TMetaData = tp.Union[None, tp.Dict[str, tp.Any], "ReportMetaData"]


def generate_html_report(
    report_structure: THtmlRenderableDict,
    render_path: tp.Union[str, Path, None] = None,
    sections_description: tp.Optional[TSectionDescription] = None,
    report_meta_data: _TMetaData = None,
    report_template_path: tp.Optional[str] = None,
    report_template_assets_path: tp.Optional[str] = None,
    max_table_of_content_depth: tp.Optional[int] = 3,
    max_level_of_header: tp.Optional[int] = 3,
) -> str:
    """Generate a self-contained HTML report from a nested dict structure.

    Args:
        report_structure: Ordered nested dict mapping section headers to content
            (matplotlib Figure, plotly Figure, pd.DataFrame, Code, or Table).
        render_path: Unused — kept for API compatibility with the reference project.
        report_meta_data: Optional metadata dict or ``ReportMetaData`` instance.
        sections_description: Optional mapping of section path tuples to descriptions.
        report_template_path: Path to a custom Jinja2 HTML template file.
        report_template_assets_path: Path to assets directory for the custom template.
        max_table_of_content_depth: Max header level shown in the sidebar TOC.
        max_level_of_header: Headers deeper than this level are omitted from output.

    Returns:
        Rendered HTML string.
    """
    rendering = render_report(
        report_structure,
        sections_description,
        max_table_of_content_depth,
        max_level_of_header,
    )

    jinja_template = _load_jinja_template(report_template_path, report_template_assets_path)

    return jinja_template.render(
        render=rendering.rendering_content,
        toc=rendering.table_of_content,
        meta=ReportMetaData.from_input(report_meta_data),
        plotly_js=get_plotlyjs(),
    )


def _load_jinja_template(
    template_path: tp.Optional[str],
    assets_path: tp.Optional[str],
) -> Template:
    if template_path is None:
        template = DefaultTemplate()
        env = _create_env(loader=DictLoader(template.assets) if template.assets else DictLoader({}))
        return env.from_string(template.source)

    if assets_path is None:
        assets_path = str(Path(template_path).parent)
    env = _create_env(loader=FileSystemLoader(assets_path))
    return env.from_string(Path(template_path).read_text(DEFAULT_ENCODING))


def _create_env(loader) -> Environment:
    return Environment(loader=loader, autoescape=False)  # nosec: internal template


class ReportMetaData:
    DATE_TIME_FORMAT = "%Y-%m-%d %H:%M"

    def __init__(
        self,
        title: str = "Report",
        creation_time: tp.Optional[tp.Union[str, datetime]] = None,
        **kwargs: tp.Any,
    ) -> None:
        self.title = title
        self.creation_time = self._parse_time(creation_time)
        self._kwargs = kwargs

    @staticmethod
    def from_input(metadata: _TMetaData) -> "ReportMetaData":
        if metadata is None:
            return ReportMetaData()
        if isinstance(metadata, ReportMetaData):
            return metadata
        return ReportMetaData(**metadata)

    def __getattr__(self, attribute: tp.Any) -> str:
        if isinstance(attribute, str) and attribute in self._kwargs:
            return self._kwargs[attribute]
        raise AttributeError(f"Attribute '{attribute}' not found on ReportMetaData.")

    def _parse_time(
        self,
        time_of_creation: tp.Optional[tp.Union[str, datetime]],
    ) -> datetime:
        if time_of_creation is None:
            return datetime.now().strftime(self.DATE_TIME_FORMAT)
        if isinstance(time_of_creation, str):
            try:
                return datetime.strptime(time_of_creation, self.DATE_TIME_FORMAT)
            except ValueError:
                logger.warning("Could not parse creation_time '%s'; using as-is.", time_of_creation)
                return time_of_creation
        return time_of_creation
