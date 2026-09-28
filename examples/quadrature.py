"""Independent quadrature helpers used only by the examples."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy.integrate import simpson
from scipy.interpolate import CubicSpline
from scipy.special import jv, jvp, spherical_jn

from nbessel import BinAverage


def _bessel(kind: str, order: int, argument: np.ndarray, derivative: int = 0):
    if kind == "spherical":
        if derivative == 0:
            return spherical_jn(order, argument)
        if derivative == 1:
            return spherical_jn(order, argument, derivative=True)
        # The spherical-Bessel differential equation supplies j_l''.
        if derivative == 2:
            safe = np.where(argument == 0.0, np.finfo(float).tiny, argument)
            value = spherical_jn(order, argument)
            first = spherical_jn(order, argument, derivative=True)
            return ((order * (order + 1) / safe**2) - 1.0) * value - 2.0 * first / safe
        raise ValueError("the example quadrature supports derivatives 0, 1, and 2")
    return jvp(order, argument, n=derivative) if derivative else jv(order, argument)


def _averaged_bessel(
    kind: str,
    order: int,
    argument: np.ndarray,
    derivative: int,
    average: BinAverage | None,
):
    if average is None:
        return _bessel(kind, order, argument, derivative)
    nodes, weights = np.polynomial.legendre.leggauss(16)
    half_width = 0.5 * (average.upper - average.lower)
    midpoint = 0.5 * (average.upper + average.lower)
    scale = midpoint + half_width * nodes
    radial_weight = (
        half_width
        * weights
        * average.dimension
        * scale ** (average.dimension - 1.0)
        / (average.upper**average.dimension - average.lower**average.dimension)
    )
    result = np.zeros_like(argument, dtype=float)
    for factor, weight in zip(scale, radial_weight, strict=True):
        result += weight * _bessel(kind, order, factor * argument, derivative)
    return result


def _bessel_product(kind, x, y, orders, ratios, derivatives, bins):
    result = np.ones_like(np.asarray(x, dtype=float) * np.asarray(y, dtype=float))
    for order, ratio, derivative, average in zip(
        orders, ratios, derivatives, bins, strict=True
    ):
        result *= _averaged_bessel(
            kind, int(order), ratio * x * y, int(derivative), average
        )
    return result


def simpson_1d(
    x: np.ndarray,
    f: np.ndarray,
    y: np.ndarray,
    orders: Sequence[int],
    ratios: Sequence[float] | None = None,
    *,
    family: str = "spherical",
    derivatives: Sequence[int] | None = None,
    bins: Sequence[BinAverage | None] | None = None,
    chunk: int = 32,
) -> np.ndarray:
    """Composite Simpson integration in ``ln(x)`` for many output values."""

    x = np.asarray(x, dtype=float)
    f = np.asarray(f)
    y = np.atleast_1d(y).astype(float)
    ratios = np.ones(len(orders)) if ratios is None else np.asarray(ratios)
    derivatives = (
        np.zeros(len(orders), dtype=int)
        if derivatives is None
        else np.asarray(derivatives)
    )
    bins = (None,) * len(orders) if bins is None else tuple(bins)
    output = np.empty(y.size, dtype=np.result_type(f, float))
    logx = np.log(x)
    for start in range(0, y.size, chunk):
        stop = min(start + chunk, y.size)
        values = f[None, :].astype(np.result_type(f, float), copy=True)
        values = np.broadcast_to(values, (stop - start, x.size)).copy()
        arguments = y[start:stop, None] * x[None, :]
        values *= _bessel_product(
            family,
            arguments,
            1.0,
            orders,
            ratios,
            derivatives,
            bins,
        )
        output[start:stop] = simpson(values, x=logx, axis=-1)
    return output


def simpson_2d_diagonal(
    x1: np.ndarray,
    x2: np.ndarray,
    f: np.ndarray,
    y1: np.ndarray,
    y2: np.ndarray,
    orders: Sequence[Sequence[int]],
    ratios: Sequence[Sequence[float]],
    *,
    families: Sequence[str] = ("spherical", "spherical"),
    bins: Sequence[Sequence[BinAverage | None]] | None = None,
) -> np.ndarray:
    """Tensor-product Simpson rule at paired ``(y1[i], y2[i])`` points."""

    output = np.empty(len(y1), dtype=np.result_type(f, float))
    grouped_bins = (
        ((None,) * len(orders[0]), (None,) * len(orders[1])) if bins is None else bins
    )
    for index, (first_y, second_y) in enumerate(zip(y1, y2, strict=True)):
        first = np.ones(x1.size)
        second = np.ones(x2.size)
        first *= _bessel_product(
            families[0],
            x1,
            first_y,
            orders[0],
            ratios[0],
            np.zeros(len(orders[0]), dtype=int),
            grouped_bins[0],
        )
        second *= _bessel_product(
            families[1],
            x2,
            second_y,
            orders[1],
            ratios[1],
            np.zeros(len(orders[1]), dtype=int),
            grouped_bins[1],
        )
        inner = simpson(f * second[None, :], x=np.log(x2), axis=1)
        output[index] = simpson(first * inner, x=np.log(x1))
    return output


def _simpson_weights(size: int, step: float):
    if size < 3 or size % 2 != 1:
        raise ValueError("a composite-Simpson grid must have an odd size >= 3")
    weights = np.ones(size)
    weights[1:-1:2] = 4.0
    weights[2:-1:2] = 2.0
    return weights * step / 3.0


def _hybrid_log_linear_grid(
    lower: float,
    upper: float,
    split: float,
    maximum_y: float,
    maximum_ratio: float,
    *,
    log_size: int,
    samples_per_period: int,
):
    if not lower < split < upper:
        raise ValueError("split must lie inside the integration interval")
    if log_size % 2 != 1:
        raise ValueError("log_size must be odd")
    low = np.geomspace(lower, split, log_size)
    low_weight = _simpson_weights(log_size, np.log(low[1] / low[0]))
    step = np.pi / (samples_per_period * maximum_y * maximum_ratio)
    intervals = int(np.ceil((upper - split) / step))
    intervals += intervals % 2
    high = np.linspace(split, upper, intervals + 1)
    high_weight = _simpson_weights(high.size, high[1] - high[0]) / high
    low_weight[-1] += high_weight[0]
    return np.concatenate((low, high[1:])), np.concatenate(
        (low_weight, high_weight[1:])
    )


def simpson_2d_diagonal_interpolated(
    x1: np.ndarray,
    x2: np.ndarray,
    f: np.ndarray,
    y1: np.ndarray,
    y2: np.ndarray,
    orders: Sequence[Sequence[int]],
    ratios: Sequence[Sequence[float]],
    *,
    families: Sequence[str] = ("spherical", "spherical"),
    bins: Sequence[Sequence[BinAverage | None]] | None = None,
    split: Sequence[float],
    log_size: int = 2049,
    samples_per_period: int = 12,
) -> np.ndarray:
    """Converged tensor-Simpson rule using log/linear hybrid grids.

    The smooth two-dimensional source is spline-interpolated from its stored
    logarithmic grid.  The high-frequency part is integrated on a linear grid
    fine enough to resolve the largest Bessel argument.  Contracting the two
    axes successively avoids constructing a dense two-dimensional quadrature
    grid.
    """

    x1 = np.asarray(x1, dtype=float)
    x2 = np.asarray(x2, dtype=float)
    f = np.asarray(f, dtype=float)
    y1 = np.asarray(y1, dtype=float)
    y2 = np.asarray(y2, dtype=float)
    grouped_bins = (
        ((None,) * len(orders[0]), (None,) * len(orders[1])) if bins is None else bins
    )
    q1, weight1 = _hybrid_log_linear_grid(
        x1[0],
        x1[-1],
        split[0],
        float(np.max(y1)),
        float(np.max(ratios[0])),
        log_size=log_size,
        samples_per_period=samples_per_period,
    )
    q2, weight2 = _hybrid_log_linear_grid(
        x2[0],
        x2[-1],
        split[1],
        float(np.max(y2)),
        float(np.max(ratios[1])),
        log_size=log_size,
        samples_per_period=samples_per_period,
    )
    source_on_q2 = CubicSpline(np.log(x2), f, axis=1)(np.log(q2))
    output = np.empty(y1.size)
    derivatives1 = np.zeros(len(orders[0]), dtype=int)
    derivatives2 = np.zeros(len(orders[1]), dtype=int)
    for index, (first_y, second_y) in enumerate(zip(y1, y2, strict=True)):
        second = _bessel_product(
            families[1],
            q2,
            second_y,
            orders[1],
            ratios[1],
            derivatives2,
            grouped_bins[1],
        )
        inner = source_on_q2 @ (weight2 * second)
        source_on_q1 = CubicSpline(np.log(x1), inner)(np.log(q1))
        first = _bessel_product(
            families[0],
            q1,
            first_y,
            orders[0],
            ratios[0],
            derivatives1,
            grouped_bins[0],
        )
        output[index] = np.dot(weight1 * first, source_on_q1)
    return output


def simpson_3d_points(
    grids: Sequence[np.ndarray],
    f: np.ndarray,
    points: np.ndarray,
    orders: Sequence[Sequence[int]],
    ratios: Sequence[Sequence[float]],
) -> np.ndarray:
    """Tensor-product Simpson rule at selected three-dimensional points."""

    result = []
    for point in points:
        legs = []
        for grid, y, grouped_orders, grouped_ratios in zip(
            grids, point, orders, ratios, strict=True
        ):
            value = np.ones(grid.size)
            for order, ratio in zip(grouped_orders, grouped_ratios, strict=True):
                value *= spherical_jn(order, ratio * grid * y)
            legs.append(value)
        integrand = (
            f * legs[0][:, None, None] * legs[1][None, :, None] * legs[2][None, None, :]
        )
        value = simpson(integrand, x=np.log(grids[2]), axis=2)
        value = simpson(value, x=np.log(grids[1]), axis=1)
        result.append(simpson(value, x=np.log(grids[0]), axis=0))
    return np.asarray(result)
