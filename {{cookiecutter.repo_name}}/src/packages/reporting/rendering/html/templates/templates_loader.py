import typing as tp
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

TEMPLATES_PATH = Path(__file__).parent / "html_template"
DEFAULT_ENCODING = "utf-8"


@dataclass
class TemplateBase(ABC):
    @property
    @abstractmethod
    def source(self) -> str:
        """HTML template source string."""

    @property
    @abstractmethod
    def assets(self) -> tp.Dict[str, str]:
        """Dict of asset name → asset content (CSS/JS files)."""


class DefaultTemplate(TemplateBase):
    """Loads the default HTML template (no external JS dependencies)."""

    def __init__(self) -> None:
        self._template_name = "template.html"
        self._source = (TEMPLATES_PATH / self._template_name).read_text(DEFAULT_ENCODING)

    @property
    def source(self) -> str:
        return self._source

    @property
    def assets(self) -> tp.Dict[str, str]:
        # No external assets — template is fully self-contained
        return {}
