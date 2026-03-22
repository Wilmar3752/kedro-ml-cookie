import subprocess
import sys


def run(cmd: list[str]) -> int:
    return subprocess.call(cmd)


if __name__ == "__main__":
    print("Running uv sync...")
    rc = run(["uv", "sync"])
    if rc != 0:
        print(
            "Warning: 'uv sync' failed. Run it manually inside the project directory.",
            file=sys.stderr,
        )
