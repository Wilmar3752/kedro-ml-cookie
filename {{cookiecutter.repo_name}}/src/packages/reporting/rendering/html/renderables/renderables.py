from __future__ import annotations

import base64
import io
import re
import typing as tp
from abc import ABC, abstractmethod
from functools import cached_property

import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go

from .protocols import (
    SavefigCompatible,
    THtmlRenderableContent,
    THtmlRenderableDictKey,
    ToHtmlCompatible,
)
from ....reports.identifiers import Code, Table

TNumericPrefix = tp.Tuple[int, ...]

_CSS_SECTION = "section"
_CSS_SECTION_CONTENT = "section-content"
_CSS_SECTION_TITLE = "section-title"
_CSS_FIGURE_IMG = "figure-img"
_CSS_FIGURE_HTML = "figure-html"

INITIAL_LEVEL_HEADER = 1
_DEFAULT_IMAGE_PADDING_INCH = 0.5


class _SimpleHtmlTable:
    """Wraps a pandas DataFrame as a styled HTML table."""

    def __init__(
        self,
        data: tp.Union[pd.Series, pd.DataFrame],
        columns: tp.Optional[tp.List[str]] = None,
        precision: tp.Optional[int] = 4,
        title: tp.Optional[str] = None,
        width: float = 80,
        show_index: bool = True,
    ) -> None:
        self.data = data.to_frame() if isinstance(data, pd.Series) else data
        self.columns = columns
        self.precision = precision
        self.title = title
        self.width = width
        self.show_index = show_index

    def to_html(self) -> str:
        df = self.data[self.columns] if self.columns else self.data
        if self.precision is not None:
            df = df.round(self.precision)
        title_html = (
            f'<caption class="table-title">{self.title}</caption>'
            if self.title
            else ""
        )
        table_html = df.to_html(
            classes="report-table",
            index=self.show_index,
            border=0,
        )
        if title_html:
            table_html = table_html.replace("<table", f"<table", 1)
            table_html = table_html.replace(
                '<thead>', f'{title_html}<thead>', 1
            )
        width_style = f'style="width:{self.width}%"'
        table_html = table_html.replace("<table", f"<table {width_style}", 1)
        return table_html


class _SimpleHtmlCode:
    """Wraps code as a syntax-highlighted HTML block."""

    def __init__(self, code: str, language: tp.Optional[str] = None) -> None:
        self._code = code
        self._language = language

    def to_html(self) -> str:
        lang = f' class="language-{self._language}"' if self._language else ""
        return f'<div class="code-block"><pre><code{lang}>{self._code}</code></pre></div>'


def _convert_to_table(fig: Table) -> _SimpleHtmlTable:
    return _SimpleHtmlTable(
        data=fig.table,
        columns=fig.columns,
        precision=fig.precision,
        title=fig.title,
        width=fig.width,
        show_index=fig.show_index,
    )


def _convert_to_code(fig: Code) -> _SimpleHtmlCode:
    return _SimpleHtmlCode(code=fig.formatted_code, language=fig.language)


HTML_RENDERABLE_TO_HTML_COMPATIBLE_TYPES = (_SimpleHtmlTable, _SimpleHtmlCode, ToHtmlCompatible)
HTML_RENDERABLE_SAVEFIG_COMPATIBLE_TYPES = (SavefigCompatible,)


def convert_object_into_renderable(fig: THtmlRenderableContent) -> "HtmlRenderableFigure":
    """Dispatch a content object to the correct HtmlRenderable wrapper."""
    if isinstance(fig, pd.DataFrame):
        fig = _SimpleHtmlTable(data=fig)
    elif isinstance(fig, Code):
        fig = _convert_to_code(fig)
    elif isinstance(fig, Table):
        fig = _convert_to_table(fig)

    for cls in HTML_RENDERABLE_TO_HTML_COMPATIBLE_TYPES:
        if isinstance(fig, cls):
            return HtmlRenderableToHtmlCompatible(fig)
    for cls in HTML_RENDERABLE_SAVEFIG_COMPATIBLE_TYPES:
        if isinstance(fig, cls):
            return HtmlRenderableSavefigCompatible(fig)

    raise NotImplementedError(
        f"Type `{type(fig)}` does not implement SavefigCompatible or ToHtmlCompatible."
    )


class HtmlRenderableObject(ABC):
    @abstractmethod
    def is_visible(self, visibility_level: tp.Optional[int]) -> bool: ...

    @abstractmethod
    def to_html(
        self,
        previous_item: tp.Optional["HtmlRenderableObject"] = None,
        next_item: tp.Optional["HtmlRenderableObject"] = None,
    ) -> str: ...


class HtmlRenderableFigure(HtmlRenderableObject, ABC):
    def __init__(self, figure: THtmlRenderableContent) -> None:
        self._figure = figure

    @property
    def figure(self):
        return self._figure

    def is_visible(self, visibility_level: tp.Optional[int]) -> bool:
        return True

    def to_html(
        self,
        previous_item: tp.Optional[HtmlRenderableObject] = None,
        next_item: tp.Optional[HtmlRenderableObject] = None,
    ) -> str:
        encoded_fig = self._encode_to_str(self._figure)
        figure_html = self._wrap_src_to_html(encoded_fig)
        prefix = (
            ""
            if isinstance(previous_item, HtmlRenderableFigure)
            else f'<div class="{_CSS_SECTION_CONTENT}">'
        )
        suffix = "" if isinstance(next_item, HtmlRenderableFigure) else "</div></div>"
        return f"{prefix}{figure_html}{suffix}"

    @staticmethod
    @abstractmethod
    def _encode_to_str(figure: THtmlRenderableContent) -> str: ...

    @staticmethod
    @abstractmethod
    def _wrap_src_to_html(encoded_src: str) -> str: ...


class HtmlRenderableToHtmlCompatible(HtmlRenderableFigure):
    @staticmethod
    def _encode_to_str(figure) -> str:
        if isinstance(figure, go.Figure):
            html = figure.to_html(include_plotlyjs=False, full_html=False)
            return re.sub("<div>(.*)</div>", r"\1", html)
        return figure.to_html()

    @staticmethod
    def _wrap_src_to_html(encoded_src: str) -> str:
        return f'<div class="{_CSS_FIGURE_HTML}">{encoded_src}</div>'


class HtmlRenderableSavefigCompatible(HtmlRenderableFigure):
    @staticmethod
    def _encode_to_str(figure: SavefigCompatible) -> str:
        buf = io.BytesIO()
        if isinstance(figure, plt.Figure):
            figure.savefig(
                buf,
                format="png",
                pad_inches=_DEFAULT_IMAGE_PADDING_INCH,
                bbox_inches="tight",
            )
        else:
            figure.savefig(buf, format="png")
        return base64.b64encode(buf.getvalue()).decode()

    @staticmethod
    def _wrap_src_to_html(encoded_src: str) -> str:
        return (
            f'<img class="{_CSS_FIGURE_IMG}" alt="graph" '
            f'src="data:image/png;base64,{encoded_src}">'
        )


class HtmlRenderableHeader(HtmlRenderableObject):
    _INITIAL_H_TAG_LEVEL = 2

    def __init__(
        self,
        level: int,
        text: THtmlRenderableDictKey,
        unique_prefix: TNumericPrefix,
        description: tp.Optional[str],
    ) -> None:
        self._level = level
        self._text = str(text)
        self._description = description
        self._unique_prefix = "-".join(str(x) for x in unique_prefix)

    def __eq__(self, other: tp.Any) -> bool:
        if isinstance(other, HtmlRenderableHeader):
            return self.id == other.id
        return False

    def to_html(
        self,
        previous_item: tp.Optional[HtmlRenderableObject] = None,
        next_item: tp.Optional[HtmlRenderableObject] = None,
    ) -> str:
        h_tag = f"h{self._h_tag_level}"
        header_html = (
            f'<{h_tag} class="{_CSS_SECTION_TITLE}" id="header-{self.id}">'
            f"{self.text}"
            f"</{h_tag}>"
        )
        prefix = (
            ""
            if isinstance(next_item, HtmlRenderableHeader)
            else f'<div class="{_CSS_SECTION}">'
        )
        if self._description is None:
            return f"{prefix}{header_html}"
        tooltip = (
            f'\n<span class="tooltip" data-text="{self._description}">'
            f'<span class="info-icon">ℹ</span></span>'
        )
        return f'{prefix}\n<div class="inline">{header_html}{tooltip}</div>'

    @property
    def level(self) -> int:
        return self._level

    @property
    def text(self) -> str:
        return self._text

    @property
    def description(self) -> tp.Optional[str]:
        return self._description

    @cached_property
    def id(self) -> str:
        normalized = re.sub("[^0-9a-zA-Z]+", "-", self.text)
        return f"{self._unique_prefix}-{self._level}-{normalized}"

    def is_visible(self, visibility_level: tp.Optional[int]) -> bool:
        if visibility_level is None:
            return True
        return self._level <= visibility_level

    @property
    def _h_tag_level(self) -> int:
        level_diff = self._INITIAL_H_TAG_LEVEL - INITIAL_LEVEL_HEADER
        return self._level + level_diff


class HtmlRenderableTocElement(HtmlRenderableObject):
    def __init__(
        self,
        reference_header: HtmlRenderableHeader,
        children: tp.Iterable["HtmlRenderableTocElement"] = (),
    ) -> None:
        self._reference_header = reference_header
        self._children: tp.List["HtmlRenderableTocElement"] = list(children)

    def __eq__(self, other: tp.Any) -> bool:
        if not isinstance(other, HtmlRenderableTocElement):
            return False
        return (
            self.reference_header_id == other.reference_header_id
            and self.level == other.level
            and self.children == other.children
        )

    @property
    def text(self) -> str:
        return self._reference_header.text

    @property
    def reference_header(self) -> HtmlRenderableHeader:
        return self._reference_header

    @property
    def reference_header_id(self) -> str:
        return self._reference_header.id

    @property
    def level(self) -> int:
        return self._reference_header.level

    def is_visible(self, visibility_level: tp.Optional[int]) -> bool:
        if visibility_level is None:
            return True
        return self.level <= visibility_level and self._reference_header.is_visible(visibility_level)

    @property
    def children(self) -> tp.List["HtmlRenderableTocElement"]:
        return self._children

    def add_child(self, toc_element: "HtmlRenderableTocElement") -> None:
        self._children.append(toc_element)

    def to_html(
        self,
        previous_item: tp.Optional[HtmlRenderableObject] = None,
        next_item: tp.Optional[HtmlRenderableObject] = None,
    ) -> str:
        return ""
