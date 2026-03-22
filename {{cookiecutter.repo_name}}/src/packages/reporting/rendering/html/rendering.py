from __future__ import annotations

import typing as tp
from dataclasses import dataclass

from .renderables.protocols import THtmlRenderableDict, THtmlRenderableDictKey
from ...reports.identifiers import SectionHeader
from .renderables.renderables import (
    INITIAL_LEVEL_HEADER,
    HtmlRenderableHeader,
    HtmlRenderableObject,
    HtmlRenderableTocElement,
    TNumericPrefix,
    convert_object_into_renderable,
)

_TFigReference = tp.Tuple[THtmlRenderableDictKey, ...]
TSectionDescription = tp.Dict[_TFigReference, tp.Union[str, "TSectionDescription"]]


@dataclass
class Rendering:
    rendering_content: tp.List[HtmlRenderableObject]
    table_of_content: tp.List[HtmlRenderableTocElement]


def render_report(
    figures: THtmlRenderableDict,
    sections_description: tp.Optional[TSectionDescription],
    max_table_of_content_depth: tp.Optional[int],
    max_level_of_header: tp.Optional[int],
) -> Rendering:
    """Convert a nested dict report structure into a ``Rendering`` object.

    Args:
        figures: Ordered nested dict mapping section headers to content.
        sections_description: Optional mapping of section paths to descriptions.
        max_table_of_content_depth: Max header level to show in the TOC.
        max_level_of_header: Headers deeper than this level are hidden.

    Returns:
        ``Rendering`` with rendered objects and table-of-content tree.
    """
    rendered = _convert_section_into_renderables(
        section_data=figures,
        section_descriptions=sections_description or {},
        header_level=INITIAL_LEVEL_HEADER,
        section_prefix=(),
        section_numeric_prefix=(),
    )
    rendered = [obj for obj in rendered if obj.is_visible(max_level_of_header)]
    toc = _extract_table_of_content(rendered, max_table_of_content_depth)
    return Rendering(rendered, toc)


def _convert_section_into_renderables(
    section_data: THtmlRenderableDict,
    section_descriptions: TSectionDescription,
    header_level: int,
    section_prefix: _TFigReference,
    section_numeric_prefix: TNumericPrefix,
) -> tp.List[HtmlRenderableObject]:
    rendered_objects: tp.List[HtmlRenderableObject] = []
    for header_index, (header, content) in enumerate(section_data.items()):
        current_prefix = (*section_prefix, header)
        current_numeric_prefix = (*section_numeric_prefix, header_index)
        _handle_section_header(
            rendered_objects=rendered_objects,
            header_element=header,
            header_level=header_level,
            current_prefix=current_prefix,
            current_numeric_prefix=current_numeric_prefix,
            section_descriptions=section_descriptions,
        )
        if isinstance(content, dict):
            rendered_objects.extend(
                _convert_section_into_renderables(
                    section_data=content,
                    section_descriptions=section_descriptions,
                    header_level=header_level + 1,
                    section_prefix=current_prefix,
                    section_numeric_prefix=current_numeric_prefix,
                )
            )
        elif isinstance(content, list):
            rendered_objects.extend(convert_object_into_renderable(fig) for fig in content)
        else:
            rendered_objects.append(convert_object_into_renderable(content))
    return rendered_objects


def _extract_table_of_content(
    rendered_objects: tp.List[HtmlRenderableObject],
    max_depth: tp.Optional[int],
) -> tp.List[HtmlRenderableTocElement]:
    flat_toc = [
        HtmlRenderableTocElement(obj)
        for obj in rendered_objects
        if isinstance(obj, HtmlRenderableHeader)
    ]
    visible = [t for t in flat_toc if t.is_visible(max_depth)]
    if not visible:
        return []

    root = HtmlRenderableTocElement(HtmlRenderableHeader(-1, "", (), None))
    root.add_child(visible.pop(0))
    visit_stack = [root]
    for elem in visible:
        parent = visit_stack[-1]
        sibling = parent.children[-1]
        if elem.level > sibling.level:
            visit_stack.append(sibling)
            parent = sibling
        elif elem.level < sibling.level:
            while elem.level < sibling.level:
                parent = visit_stack.pop()
                sibling = parent.children[-1]
            visit_stack.append(parent)
        parent.add_child(elem)
    return root.children


def _handle_section_header(
    rendered_objects: tp.List,
    header_element,
    header_level: int,
    current_prefix: _TFigReference,
    current_numeric_prefix: TNumericPrefix,
    section_descriptions: TSectionDescription,
) -> None:
    if isinstance(header_element, str):
        rendered_objects.append(
            HtmlRenderableHeader(
                level=header_level,
                text=header_element,
                unique_prefix=current_numeric_prefix,
                description=section_descriptions.get(current_prefix),
            )
        )
        return
    if isinstance(header_element, SectionHeader):
        description = (
            header_element.description
            if header_element.description is not None
            else section_descriptions.get(current_prefix)
        )
        rendered_objects.append(
            HtmlRenderableHeader(
                level=header_level,
                text=header_element.header,
                unique_prefix=current_numeric_prefix,
                description=description,
            )
        )
        return
    raise NotImplementedError(
        f"Instances of {header_element.__class__.__name__} are not supported as section headers."
    )
