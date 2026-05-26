import os
import subprocess
import sys
from pathlib import Path

PROJECT_DIR = Path.cwd()


def run(cmd: list[str]) -> int:
    return subprocess.call(cmd, cwd=PROJECT_DIR)


if __name__ == "__main__":
    print(f"Entering project directory: {PROJECT_DIR}", flush=True)
    os.chdir(PROJECT_DIR)

    print("Initialising git repository...", flush=True)
    rc = run(["git", "init"])
    if rc != 0:
        print("Warning: 'git init' failed.", file=sys.stderr, flush=True)

    print("Running uv sync...", flush=True)
    rc = run(["uv", "sync"])
    if rc != 0:
        print(
            "Warning: 'uv sync' failed. Run it manually inside the project directory.",
            file=sys.stderr,
            flush=True,
        )

    print(f"\n✅ Project ready! Navigate into it with:\n\n   cd {PROJECT_DIR.name}\n", flush=True)
