from __future__ import annotations

import os
from typing import Any


def configure_runtime() -> None:
    """Apply small runtime compatibility fixes before importing ML stacks."""
    # Kaggle usually has TensorFlow/JAX installed. Transformers can auto-import
    # them while this project only needs PyTorch, which exposes avoidable ABI
    # conflicts in fast-moving notebook images.
    os.environ.setdefault("USE_TF", "0")
    os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
    os.environ.setdefault("USE_JAX", "0")
    patch_torch_pytree_register_constant()


def patch_torch_pytree_register_constant() -> None:
    """Backfill torch.utils._pytree.register_constant for mixed Kaggle stacks.

    Some newer libraries call this helper, while older Torch builds only expose
    register_pytree_node or _register_pytree_node. Treating the class as a leaf
    constant is enough for import-time registrations used by Transformers/TRL.
    """
    try:
        import torch.utils._pytree as pytree
    except Exception:
        return

    if hasattr(pytree, "register_constant"):
        return

    register = getattr(pytree, "register_pytree_node", None) or getattr(
        pytree, "_register_pytree_node", None
    )

    def register_constant(cls: type[Any]) -> type[Any]:
        if register is None:
            return cls

        def flatten_fn(obj: Any):
            return (), obj

        def unflatten_fn(children: Any, context: Any):
            return context

        kwargs = {
            "serialized_type_name": f"{cls.__module__}.{cls.__qualname__}",
        }
        try:
            register(cls, flatten_fn, unflatten_fn, **kwargs)
        except TypeError:
            try:
                register(cls, flatten_fn, unflatten_fn)
            except (KeyError, ValueError):
                pass
        except (KeyError, ValueError):
            pass
        return cls

    pytree.register_constant = register_constant
