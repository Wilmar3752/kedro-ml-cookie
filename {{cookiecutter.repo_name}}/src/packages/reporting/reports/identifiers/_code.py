import typing as tp
from functools import cached_property

from ._base import ReportElementIdentifier


class Code(ReportElementIdentifier):
    def __init__(
        self,
        code: tp.Optional[str] = None,
        language: tp.Optional[str] = None,
    ) -> None:
        super().__init__(report_element=code)
        self.code = code
        self.language = language

    @cached_property
    def formatted_code(self) -> str:
        return self.code or ""

    def _repr_html_(self) -> str:
        return f"<pre><code>{self.formatted_code}</code></pre>"
