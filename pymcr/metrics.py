"""
Metrics used in pyMCR

All functions must take C, ST, D_actual, D_calculated as NDArrays
and return a single NDArray as a result
"""
import numpy as np
from numpy.typing import NDArray


def mse(C: NDArray, ST: NDArray, D_actual: NDArray, D_calculated: NDArray) -> NDArray:
    """Mean square error"""
    return ((D_actual - D_calculated) ** 2).sum() / D_actual.size


def lof(C: NDArray, ST: NDArray, D_actual: NDArray, D_calculated: NDArray) -> NDArray:
    return 100 * np.sqrt(
        np.sum((D_actual - D_calculated) ** 2) / np.sum((D_actual**2))
    )
