import typing as tp


class ReportElementIdentifier(object):
    """Base class for identifying report elements.

    Informs the renderer that a given report element has to be rendered
    in a specific way.
    """

    def __init__(self, report_element: tp.Any) -> None:
        self._report_element = report_element

    @property
    def report_element(self):
        """Returns the unwrapped original object."""
        return self._report_element

    def __repr__(self) -> str:
        return f"ReportElementIdentifier(report_element={repr(self.report_element)})"
