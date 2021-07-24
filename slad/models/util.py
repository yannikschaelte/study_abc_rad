"""Utility functions."""

import numpy as np


def gk(A, B, c, g, k, n: int = 1):
    """Sample from the g-and-k distribution."""
    z = np.random.normal(size=n)
    e = np.exp(-g * z)
    return A + B * (1 + c * (1 - e) / (1 + e)) * (1 + z ** 2) ** k * z
