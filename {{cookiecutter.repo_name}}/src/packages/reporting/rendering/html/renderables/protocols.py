import typing as tp


@tp.runtime_checkable
class SavefigCompatible(tp.Protocol):
    def savefig(
        self,
        fname: tp.BinaryIO,
        *args: tp.Any,
        format: tp.Optional[str] = None,
        **kwargs: tp.Any,
    ) -> None:
        """Saves figure to a binary file-like object in the given format."""


@tp.runtime_checkable
class ToHtmlCompatible(tp.Protocol):
    def to_html(self, *args: tp.Optional[tp.Any], **kwargs: tp.Optional[tp.Any]) -> str:
        """Produces HTML representation for report rendering."""


DIRECTLY_HTML_RENDERABLE_TYPES = (ToHtmlCompatible, SavefigCompatible)
THtmlRenderableContent = tp.Union[ToHtmlCompatible, SavefigCompatible]
THtmlRenderableDictKey = tp.Union[str, int]
THtmlRenderableDictValue = tp.Union[
    THtmlRenderableDictKey,
    THtmlRenderableContent,
    tp.List[THtmlRenderableContent],
]
THtmlRenderableDict = tp.Dict[
    THtmlRenderableDictKey,
    tp.Union[THtmlRenderableDictValue, "THtmlRenderableDict"],
]
