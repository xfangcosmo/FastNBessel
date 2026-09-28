"""Small helpers for preparing logarithmic FFTLog grids."""

from __future__ import annotations

from collections.abc import Sequence
from numbers import Integral

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _padding(value: int | Sequence[int], dimensions: int) -> tuple[int, ...]:
    if isinstance(value, Integral):
        result = (int(value),) * dimensions
    else:
        result = tuple(int(item) for item in value)
    if len(result) != dimensions or any(item < 0 for item in result):
        raise ValueError("pad must contain one non-negative integer per dimension")
    return result


def _extended_log_grid(grid: ArrayLike, pad: int) -> NDArray[np.float64]:
    values = np.asarray(grid, dtype=float)
    if values.ndim != 1 or values.size < 2 or np.any(values <= 0.0):
        raise ValueError("each grid must be a one-dimensional positive array")
    steps = np.diff(np.log(values))
    step = float(np.mean(steps))
    if np.any(np.diff(values) <= 0.0) or not np.allclose(
        steps, step, rtol=2.0e-10, atol=2.0e-13
    ):
        raise ValueError("each grid must be increasing and uniformly log-spaced")
    return values[0] * np.exp(step * np.arange(-pad, values.size + pad))


def log_zero_pad(
    x: ArrayLike, f: ArrayLike, pad: int
) -> tuple[NDArray[np.float64], NDArray]:
    """Add zero samples to both ends of a one-dimensional logarithmic grid.

    The logarithmic spacing is preserved.  Leading batch axes in ``f`` are
    unchanged, and its final axis must correspond to ``x``.
    """

    count = _padding(pad, 1)[0]
    grid = np.asarray(x, dtype=float)
    values = np.asarray(f)
    if values.ndim < 1 or values.shape[-1] != grid.size:
        raise ValueError("the final axis of f must match x")
    widths = [(0, 0)] * values.ndim
    widths[-1] = (count, count)
    return _extended_log_grid(grid, count), np.pad(values, widths)


def log_zero_pad_nd(
    x: Sequence[ArrayLike], f: ArrayLike, pad: int | Sequence[int]
) -> tuple[tuple[NDArray[np.float64], ...], NDArray]:
    """Zero-pad the final axes of a multidimensional logarithmic input."""

    grids = tuple(np.asarray(grid, dtype=float) for grid in x)
    if not grids:
        raise ValueError("x must contain at least one grid")
    counts = _padding(pad, len(grids))
    values = np.asarray(f)
    shape = tuple(grid.size for grid in grids)
    if values.ndim < len(grids) or values.shape[-len(grids) :] != shape:
        raise ValueError("the final axes of f must match the x grids")
    widths = [(0, 0)] * (values.ndim - len(grids))
    widths.extend((count, count) for count in counts)
    extended = tuple(
        _extended_log_grid(grid, count)
        for grid, count in zip(grids, counts, strict=True)
    )
    return extended, np.pad(values, widths)
