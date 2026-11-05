"""Run from any current directory: python -s <path>/run.py --help."""
import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

from jointcv.commands import main

if __name__ == "__main__":
    main()
