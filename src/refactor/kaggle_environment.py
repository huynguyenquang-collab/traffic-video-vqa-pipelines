"""Environment/package setup from the original Kaggle notebook."""

from __future__ import annotations

import subprocess
import sys

import numpy as np


DEFAULT_PACKAGES = [
    "unsloth",
    "triton",
    "tokenizers>=0.22.0,<=0.23.0",
    "rank_bm25",
    "deep_translator",
    "ultralytics",
]


def install_notebook_dependencies(packages: list[str] | None = None) -> None:
    """Install notebook dependencies and re-pin numpy metadata to the loaded version."""
    numpy_version = np.__version__
    print(f"In-memory numpy: {numpy_version}")
    for package in packages or DEFAULT_PACKAGES:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", package], check=True)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            "--force-reinstall",
            "--no-deps",
            f"numpy=={numpy_version}",
        ],
        check=True,
    )
    print(f"numpy metadata re-pinned to {numpy_version}.")


if __name__ == "__main__":
    install_notebook_dependencies()

