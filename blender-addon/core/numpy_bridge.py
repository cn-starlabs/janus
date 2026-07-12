"""NumPy zero-copy views from FFI pointers."""

from __future__ import annotations

import ctypes

import numpy as np


def array_from_ptr(ptr: int, length: int, dtype=np.float64) -> np.ndarray:
    if length <= 0:
        return np.array([], dtype=dtype)
    c_array = (ctypes.c_double * length).from_address(ptr)
    return np.ctypeslib.as_array(c_array)
