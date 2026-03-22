import typing as tp

from ._base import ReportElementIdentifier


class SectionHeader(ReportElementIdentifier):
    def __init__(
        self,
        header: str,
        description: tp.Optional[str] = None,
    ) -> None:
        super().__init__(report_element=header)
        self.header = header
        self._description = description

    @property
    def description(self) -> tp.Optional[str]:
        return self._description

    def __repr__(self) -> str:
        desc = "" if self._description is None else f", description='{self._description}'"
        return f"SectionHeader(header='{self.header}'{desc})"

    def _repr_html_(self) -> str:
        return self.header
