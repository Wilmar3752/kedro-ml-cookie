import subprocess
import sys


def run(cmd: list[str]) -> int:
    return subprocess.call(cmd)


if __name__ == "__main__":
    print("Initialising git repository...")
    rc = run(["git", "init"])
    if rc != 0:
        print("Warning: 'git init' failed.", file=sys.stderr)

    print("Running uv sync...")
    rc = run(["uv", "sync"])
    if rc != 0:
        print(
            "Warning: 'uv sync' failed. Run it manually inside the project directory.",
            file=sys.stderr,
        )
