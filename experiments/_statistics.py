"""Wilcoxon, McNemar, sign test wrappers. Each returns a dict.
Wilcoxon falls back to descriptive median ratio when n < 6.
"""

from __future__ import annotations

import math
import statistics
from typing import Sequence

from scipy.stats import wilcoxon, binomtest


def sign_test_one_sided(b: int, c: int) -> dict:
    """Exact one-sided sign test. b = second-config wins, c = first-config wins.
    Tests H_A: P(second wins) > 0.5. Returns dict with p, b, c, n, note.
    """
    n = b + c
    if n == 0:
        return {"p": 1.0, "b": 0, "c": 0, "n": 0, "note": "no discordant pairs"}
    result = binomtest(b, n, p=0.5, alternative="greater")
    return {"p": float(result.pvalue), "b": b, "c": c, "n": n, "note": ""}


_MIN_PAIRED_FOR_TEST = 6


def median_ratio(values_x: Sequence[float], values_y: Sequence[float]) -> float | None:
    pairs = [(x, y) for x, y in zip(values_x, values_y) if x > 0 and y > 0]
    if not pairs:
        return None
    ratios = [y / x for x, y in pairs]
    return statistics.median(ratios)


def wilcoxon_log_ratio(values_x: Sequence[float], values_y: Sequence[float]) -> dict:
    """Wilcoxon signed-rank on log(y/x). Drops nonpositive pairs.
    n < 6 returns the descriptive median ratio without a p-value.
    """
    pairs = [(x, y) for x, y in zip(values_x, values_y) if x > 0 and y > 0]
    n = len(pairs)
    if n == 0:
        return {"W": None, "p": None, "n": 0, "median_ratio": None,
                "note": "no positive paired values"}
    ratios = [y / x for x, y in pairs]
    med = statistics.median(ratios)
    if n < _MIN_PAIRED_FOR_TEST:
        return {"W": None, "p": None, "n": n, "median_ratio": med,
                "note": f"n={n} < {_MIN_PAIRED_FOR_TEST}, descriptive only"}
    log_ratios = [math.log(r) for r in ratios]
    if all(lr == 0.0 for lr in log_ratios):
        return {"W": 0.0, "p": 1.0, "n": n, "median_ratio": med,
                "note": "all paired values identical"}
    result = wilcoxon(log_ratios)
    return {"W": float(result.statistic), "p": float(result.pvalue),
            "n": n, "median_ratio": med, "note": ""}


def mcnemar_exact(b: int, c: int) -> dict:
    """Two-sided exact McNemar. Binomial on min(b, c) of (b + c) discordant pairs."""
    n = b + c
    if n == 0:
        return {"p": 1.0, "b": 0, "c": 0, "n": 0, "note": "no discordant pairs"}
    k = min(b, c)
    result = binomtest(k, n, p=0.5, alternative="two-sided")
    return {"p": float(result.pvalue), "b": b, "c": c, "n": n, "note": ""}
