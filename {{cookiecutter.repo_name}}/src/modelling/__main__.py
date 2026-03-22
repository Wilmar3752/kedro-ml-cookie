from pathlib import Path

from kedro.framework.project import configure_project
from kedro.framework.cli import main


def main_entrypoint():
    configure_project(Path(__file__).parent.name)
    main()


if __name__ == "__main__":
    main_entrypoint()
