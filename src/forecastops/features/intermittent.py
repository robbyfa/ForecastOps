"""Causal Croston–Syntetos–Boylan estimates (alpha fixed before evaluation)."""

import numpy as np

ALPHA = 0.1


def croston_sba(demand):
    # Initialize size at first nonzero demand; interval counts from series start.
    # No future positive observation is used to fill the initial zero period.
    result = np.zeros(len(demand), dtype=float)
    size, interval, gap = 0.0, 0.0, 0
    for i, value in enumerate(demand):
        gap += 1
        if value > 0:
            if interval == 0:
                size, interval = float(value), float(gap)
            else:
                size += ALPHA * (value - size)
                interval += ALPHA * (gap - interval)
            gap = 0
        if interval:
            result[i] = (1 - ALPHA / 2) * size / interval
    return result
