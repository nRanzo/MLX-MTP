"""Numerical comparison helpers shared by Metal parity scripts and tests."""
from __future__ import annotations
import numpy as np

def max_abs_diff(actual, expected) -> float:
    return float(np.max(np.abs(np.asarray(actual, dtype=np.float32) - np.asarray(expected, dtype=np.float32))))

def assert_close(actual, expected, atol: float = 1e-4, name: str = "tensor") -> float:
    diff = max_abs_diff(actual, expected)
    if diff > atol:
        raise AssertionError(f"{name}: max_abs_diff={diff:.6g} exceeds atol={atol:.6g}")
    return diff
