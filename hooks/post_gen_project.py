from pathlib import Path

PROJECT_DIR = Path.cwd()

if __name__ == "__main__":
    print(f"\n✅ Project ready! Navigate into it with:\n\n   cd {PROJECT_DIR.name}\n", flush=True)
