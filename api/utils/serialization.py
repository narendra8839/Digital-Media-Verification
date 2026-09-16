"""JSON serialization and numeric sanitization utilities for API payloads."""

from typing import Any
import numpy as np


def sanitize_for_json(obj: Any) -> Any:
    """Recursively convert NumPy scalars, arrays, PyTorch tensors, and Path objects
    into standard Python JSON-serializable types.
    """
    # Check for PyTorch tensor
    if hasattr(obj, "detach") and hasattr(obj, "cpu") and hasattr(obj, "numpy"):
        try:
            return sanitize_for_json(obj.detach().cpu().numpy())
        except Exception:
            return str(obj)

    # Check for NumPy array
    if isinstance(obj, np.ndarray):
        return [sanitize_for_json(x) for x in obj.tolist()]

    # Check for NumPy scalar types
    if isinstance(obj, (np.floating, float)):
        val = float(obj)
        if np.isnan(val) or np.isinf(val):
            return None
        return val

    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)

    if isinstance(obj, (np.integer, int)):
        return int(obj)

    # Dictionaries
    if isinstance(obj, dict):
        return {str(k): sanitize_for_json(v) for k, v in obj.items()}

    # Lists, tuples, sets
    if isinstance(obj, (list, tuple, set)):
        return [sanitize_for_json(item) for item in obj]

    # Path objects
    if hasattr(obj, "__fspath__"):
        return str(obj)

    return obj
