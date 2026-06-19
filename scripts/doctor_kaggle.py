from __future__ import annotations

import importlib
import os
import sys


def main() -> None:
    os.environ.setdefault("USE_TF", "0")
    os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
    os.environ.setdefault("USE_JAX", "0")

    checks = ["numpy", "torch", "sklearn", "matplotlib", "transformers", "trl", "unsloth"]
    print("Python:", sys.executable)
    print("Version:", sys.version.split()[0])
    for name in checks:
        module = importlib.import_module(name)
        version = getattr(module, "__version__", "unknown")
        print(f"{name}: {version}")

    import numpy as np

    major = int(np.__version__.split(".", 1)[0])
    if major >= 2:
        raise SystemExit(
            "NumPy 2.x detected. Kaggle's prebuilt sklearn/matplotlib often "
            "break with this ABI. Run scripts/bootstrap_kaggle.sh in a fresh session."
        )

    print("Doctor check: OK")


if __name__ == "__main__":
    main()
