"""Shared helpers for Round 3 analysis notebooks.

See DESIGN.md for the full specification.
"""
from __future__ import annotations

import math
import os
import re
from typing import Iterable

import numpy as np
import pandas as pd


# ---------- data loading ----------

def extract_day_from_filename(path: str) -> int:
    """Pull the integer day N out of a filename like 'prices_round_3_day_-1.csv'.

    Returns 0 when no 'day_<int>' segment is present.
    """
    m = re.search(r"day_(-?\d+)", os.path.basename(path))
    return int(m.group(1)) if m else 0


# ---------- tiny math utilities ----------

def safe_corr(x, y) -> float:
    """Pearson correlation that returns NaN on zero-variance or empty input.

    Aligns x and y, drops any row where either is NaN, then returns NaN if
    fewer than 2 rows remain or either side is constant.
    """
    z = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(z) < 2:
        return float("nan")
    if z["x"].std(ddof=0) == 0 or z["y"].std(ddof=0) == 0:
        return float("nan")
    return float(z["x"].corr(z["y"]))


def weighted_avg(values: Iterable[float], weights: Iterable[float]) -> float:
    """Volume-weighted average that returns NaN when all weights are zero/NaN."""
    values = np.asarray(list(values), dtype=float)
    weights = np.asarray(list(weights), dtype=float)
    mask = ~(np.isnan(values) | np.isnan(weights))
    values = values[mask]
    weights = weights[mask]
    total = weights.sum()
    if total == 0:
        return float("nan")
    return float((values * weights).sum() / total)
