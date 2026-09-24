"""Mathematical utility functions."""

import numpy as np
from typing import List


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Safe cosine similarity between two vectors."""
    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)
    if a_norm < 1e-10 or b_norm < 1e-10:
        return 0.0
    return float(np.dot(a, b) / (a_norm * b_norm))


def l1_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sum(np.abs(a - b)))


def normalize_array(arr: np.ndarray) -> np.ndarray:
    mn, mx = arr.min(), arr.max()
    if mx - mn < 1e-10:
        return np.zeros_like(arr)
    return (arr - mn) / (mx - mn)


def hoeffding_bound(n: int, confidence: float = 0.95) -> float:
    """Hoeffding bound for values in [0,1]."""
    import math
    delta = 1.0 - confidence
    return math.sqrt(math.log(2.0 / delta) / (2.0 * n))


def bernstein_bound(n: int, variance: float, confidence: float = 0.95) -> float:
    """Bernstein concentration bound."""
    import math
    delta = 1.0 - confidence
    term1 = (2.0 * variance * math.log(2.0 / delta)) / n
    term2 = (2.0 * math.log(2.0 / delta)) / (3.0 * n)
    return math.sqrt(term1) + term2


def majority_vote(vectors: List[np.ndarray]) -> np.ndarray:
    """Per-field majority vote across binary vectors."""
    if not vectors:
        return np.array([])
    stacked = np.stack(vectors, axis=0)
    return (stacked.sum(axis=0) > len(vectors) / 2).astype(np.float64)
