"""Implementation of the multiple-Bessel FFTLog algorithm.

The public API stores the sampled input in an :class:`NBessel` object; call
``spherical`` or ``cylindrical`` to evaluate the desired transform.  The
implementation contains no quadrature code and no optional compiled backend.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Integral, Real

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.fft import fft, fftn, ifft, next_fast_len
from scipy.special import loggamma


@dataclass(frozen=True)
class BinAverage:
    """Scale-invariant average over ``lower*y < y_bin < upper*y``.

    ``dimension`` selects the radial measure, for example 3 for a volume
    average and 2 for an area average.
    """

    lower: float
    upper: float
    dimension: float

    def __post_init__(self) -> None:
        if not np.all(np.isfinite((self.lower, self.upper, self.dimension))):
            raise ValueError("bin parameters must be finite")
        if self.lower <= 0.0 or self.upper <= self.lower:
            raise ValueError("bin edges must satisfy 0 < lower < upper")
        if self.dimension <= 0.0:
            raise ValueError("dimension must be positive")


def _log_grid(x: ArrayLike, name: str) -> tuple[NDArray[np.float64], float]:
    values = np.asarray(x, dtype=np.float64)
    if values.ndim != 1 or values.size < 4:
        raise ValueError(f"{name} must be a one-dimensional array of length >= 4")
    if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
        raise ValueError(f"{name} must contain finite positive values")
    steps = np.diff(np.log(values))
    step = float(np.mean(steps))
    if np.any(np.diff(values) <= 0.0) or not np.allclose(
        steps, step, rtol=2.0e-10, atol=2.0e-13
    ):
        raise ValueError(f"{name} must be increasing and uniformly log-spaced")
    return values, step


def _smooth_cutoff(indices: ArrayLike, maximum: float, width: float) -> NDArray:
    if not 0.0 <= width <= 1.0:
        raise ValueError("window widths must lie in [0, 1]")
    index = np.abs(np.asarray(indices, dtype=np.float64))
    if width == 0.0:
        return np.ones_like(index)
    start = maximum * (1.0 - width)
    result = np.ones_like(index)
    tapered = index > start
    theta = np.clip((maximum - index[tapered]) / (maximum - start), 0.0, 1.0)
    result[tapered] = theta - np.sin(2.0 * np.pi * theta) / (2.0 * np.pi)
    result[index >= maximum] = 0.0
    return result


def _expm1_over_x(x: NDArray[np.complex128]) -> NDArray[np.complex128]:
    result = np.empty_like(x)
    small = np.abs(x) < 1.0e-7
    value = x[small]
    result[small] = 1.0 + value / 2.0 + value**2 / 6.0 + value**3 / 24.0
    result[~small] = np.expm1(x[~small]) / x[~small]
    return result


def _bin_factor(z: NDArray[np.complex128], average: BinAverage) -> NDArray:
    exponent = average.dimension - z
    log_lower = np.log(average.lower)
    log_ratio = np.log(average.upper / average.lower)
    numerator = (
        np.exp(exponent * log_lower)
        * log_ratio
        * _expm1_over_x(exponent * log_ratio)
    )
    denominator = np.exp(average.dimension * log_lower) * np.expm1(
        average.dimension * log_ratio
    )
    return average.dimension * numerator / denominator


def _families(family: str | Sequence[str], count: int) -> tuple[str, ...]:
    if isinstance(family, str):
        result = (family.lower(),) * count
    else:
        result = tuple(str(item).lower() for item in family)
    aliases = {"j": "spherical", "spherical": "spherical", "cylindrical": "cylindrical", "jv": "cylindrical"}
    try:
        normalized = tuple(aliases[item] for item in result)
    except KeyError as error:
        raise ValueError("family must be 'spherical' or 'cylindrical'") from error
    if len(normalized) != count:
        raise ValueError("family must have one entry per Bessel function")
    return normalized


def _leg_arrays(
    orders: Sequence[int],
    ratios: Sequence[float] | None,
    derivatives: Sequence[int] | None,
    family: str | Sequence[str],
) -> tuple[NDArray[np.int64], NDArray, NDArray[np.int64], tuple[str, ...], NDArray]:
    raw_orders = np.asarray(orders)
    if raw_orders.ndim != 1 or raw_orders.size == 0 or not np.issubdtype(raw_orders.dtype, np.integer):
        raise ValueError("orders must be a non-empty sequence of integers")
    kinds = _families(family, raw_orders.size)
    signs = np.ones(raw_orders.size, dtype=np.float64)
    normalized = raw_orders.astype(np.int64, copy=True)
    for index, kind in enumerate(kinds):
        if kind == "spherical" and normalized[index] < 0:
            raise ValueError("spherical-Bessel orders must be non-negative")
        if kind == "cylindrical" and normalized[index] < 0:
            # J_{-n}(x)=(-1)^n J_n(x) for integer n.
            signs[index] = -1.0 if abs(int(normalized[index])) % 2 else 1.0
            normalized[index] = abs(normalized[index])
    scale = np.ones(normalized.size) if ratios is None else np.asarray(ratios, dtype=float)
    if scale.shape != normalized.shape or not np.all(np.isfinite(scale)) or np.any(scale <= 0.0):
        raise ValueError("ratios must be finite, positive, and match orders")
    derivative = np.zeros(normalized.size, dtype=np.int64) if derivatives is None else np.asarray(derivatives)
    if derivative.shape != normalized.shape or not np.issubdtype(derivative.dtype, np.integer) or np.any(derivative < 0):
        raise ValueError("derivatives must be non-negative integers matching orders")
    return normalized, scale, derivative.astype(np.int64), kinds, signs


def _split_biases(
    orders: NDArray[np.int64],
    derivatives: NDArray[np.int64],
    families: tuple[str, ...],
    bias: float,
    biases: Sequence[float] | None,
) -> NDArray:
    if not np.isfinite(bias):
        raise ValueError("bias must be finite")
    split = np.full(orders.size, bias / orders.size) if biases is None else np.asarray(biases, dtype=float)
    if split.shape != orders.shape or not np.all(np.isfinite(split)):
        raise ValueError("biases must be finite and match orders")
    if not np.isclose(np.sum(split), bias, rtol=1.0e-12, atol=1.0e-13):
        raise ValueError("biases must sum to bias")
    for index, (order, derivative, kind, value) in enumerate(
        zip(orders, derivatives, families, split, strict=True)
    ):
        lower = derivative - order if order >= derivative else -order
        upper = 2.0 if kind == "spherical" else 1.5
        if not lower < value < upper:
            raise ValueError(
                f"biases[{index}]={value:g} lies outside ({lower:g}, {upper:g})"
            )
    return split


def _mellin_leg(
    order: int,
    z: NDArray[np.complex128],
    family: str,
    derivative: int,
    sign: float,
) -> NDArray[np.complex128]:
    shifted = z - derivative
    if family == "spherical":
        log_value = (
            (shifted - 2.0) * np.log(2.0)
            + 0.5 * np.log(np.pi)
            + loggamma((order + shifted) / 2.0)
            - loggamma((3.0 + order - shifted) / 2.0)
        )
    else:
        log_value = (
            (shifted - 1.0) * np.log(2.0)
            + loggamma((order + shifted) / 2.0)
            - loggamma((2.0 + order - shifted) / 2.0)
        )
    result = sign * np.exp(log_value)
    if derivative:
        falling = np.ones_like(z)
        for offset in range(1, derivative + 1):
            falling *= z - offset
        result *= (-1) ** derivative * falling
    return np.asarray(result, dtype=np.complex128)


def _product_kernel(
    orders: Sequence[int],
    ratios: Sequence[float] | None,
    modes: NDArray[np.int64],
    frequency_step: float,
    *,
    family: str | Sequence[str],
    derivatives: Sequence[int] | None,
    bias: float,
    biases: Sequence[float] | None,
    mellin_size: int,
    oversampling: int,
    mellin_window: float,
    bins: Sequence[BinAverage | None] | None,
) -> NDArray[np.complex128]:
    ell, scale, derivative, kinds, signs = _leg_arrays(orders, ratios, derivatives, family)
    split = _split_biases(ell, derivative, kinds, bias, biases)
    averages = (None,) * ell.size if bins is None else tuple(bins)
    if len(averages) != ell.size or not all(item is None or isinstance(item, BinAverage) for item in averages):
        raise ValueError("bins must contain one BinAverage or None per Bessel function")
    if not isinstance(mellin_size, Integral) or mellin_size < 3 or mellin_size % 2 != 1:
        raise ValueError("mellin_size must be an odd integer >= 3")
    if not isinstance(oversampling, Integral) or oversampling < 1:
        raise ValueError("oversampling must be a positive integer")

    if ell.size == 1:
        z = split[0] + 1j * frequency_step * modes
        result = np.exp(-z * np.log(scale[0])) * _mellin_leg(
            int(ell[0]), z, kinds[0], int(derivative[0]), float(signs[0])
        )
        if averages[0] is not None:
            result *= _bin_factor(z, averages[0])
        return result

    half = mellin_size // 2
    requested = modes * int(oversampling)
    if np.any(np.abs(requested) > ell.size * half):
        raise ValueError("mellin_size is too small for the requested outer modes")
    internal_step = frequency_step / oversampling
    convolution_length = ell.size * (mellin_size - 1) + 1
    fft_size = next_fast_len(convolution_length)
    spectrum = np.ones(fft_size, dtype=np.complex128)
    positive_modes = np.arange(half + 1, dtype=float)

    for order, ratio, derivative_order, kind, sign, leg_bias, average in zip(
        ell, scale, derivative, kinds, signs, split, averages, strict=True
    ):
        z = leg_bias + 1j * internal_step * positive_modes
        positive = (
            np.exp(-z * np.log(ratio))
            * _mellin_leg(int(order), z, kind, int(derivative_order), float(sign))
            * _smooth_cutoff(positive_modes, float(half), mellin_window)
        )
        if average is not None:
            positive *= _bin_factor(z, average)
        leg = np.empty(mellin_size, dtype=np.complex128)
        leg[half:] = positive
        leg[:half] = np.conj(positive[:0:-1])
        spectrum *= fft(leg, fft_size)

    convolved = ifft(spectrum)[:convolution_length]
    normalization = (internal_step / (2.0 * np.pi)) ** (ell.size - 1)
    result = normalization * convolved[requested + ell.size * half]
    if not np.all(np.isfinite(result)):
        raise ValueError("non-finite Mellin kernel; change the contour biases")
    return result


class _Plan1D:
    def __init__(
        self,
        x: ArrayLike,
        orders: Sequence[int],
        ratios: Sequence[float] | None,
        *,
        family: str | Sequence[str],
        derivatives: Sequence[int] | None,
        bias: float,
        biases: Sequence[float] | None,
        mellin_size: int,
        oversampling: int,
        y0: float | None,
        window: float,
        mellin_window: float,
        bins: Sequence[BinAverage | None] | None,
    ) -> None:
        self.x, delta = _log_grid(x, "x")
        size = self.x.size
        self.y = (1.0 / self.x[-1] if y0 is None else float(y0)) * np.exp(delta * np.arange(size))
        if not np.all(np.isfinite(self.y)) or np.any(self.y <= 0.0):
            raise ValueError("y0 must be finite and positive")
        centered = np.arange(-(size // 2), size - size // 2, dtype=np.int64)
        unshifted = np.fft.fftfreq(size) * size
        frequency_step = 2.0 * np.pi / (size * delta)
        kernel = np.fft.ifftshift(
            _product_kernel(
                orders,
                ratios,
                centered,
                frequency_step,
                family=family,
                derivatives=derivatives,
                bias=bias,
                biases=biases,
                mellin_size=mellin_size,
                oversampling=oversampling,
                mellin_window=mellin_window,
                bins=bins,
            )
        )
        eta = frequency_step * unshifted
        phase = np.exp(-1j * eta * np.log(self.x[0] * self.y[0]))
        self._input_bias = self.x ** (-bias)
        self._output_bias = self.y ** (-bias) / size
        self._multiplier = _smooth_cutoff(unshifted, float(size // 2), window) * phase * kernel
        if size % 2 == 0:
            self._multiplier[size // 2] = 0.0

    def apply(self, f: ArrayLike) -> tuple[NDArray[np.float64], NDArray]:
        original = np.asarray(f)
        if original.ndim < 1 or original.shape[-1] != self.x.size:
            raise ValueError(f"the last axis of f must have length {self.x.size}")
        if not np.issubdtype(original.dtype, np.number) or not np.all(np.isfinite(original)):
            raise ValueError("f must contain finite numeric values")
        coefficients = fft(np.asarray(original, dtype=np.complex128) * self._input_bias, axis=-1)
        values = self._output_bias * fft(coefficients * self._multiplier, axis=-1)
        if not np.iscomplexobj(original):
            values = np.real_if_close(values, tol=1000)
            if np.iscomplexobj(values):
                scale = max(float(np.max(np.abs(values.real))), 1.0)
                if np.max(np.abs(values.imag)) > 5.0e-11 * scale:
                    raise RuntimeError("real transform retained a significant imaginary part")
                values = values.real
        return self.y.copy(), np.asarray(values)


class NBessel:
    """A sampled one-dimensional input ready for a multiple-Bessel transform.

    Examples
    --------
    ``r, F = NBessel(k, fk).spherical([0, 2], [1.0, 0.83])``
    """

    def __init__(self, x: ArrayLike, f: ArrayLike):
        self.x, _ = _log_grid(x, "x")
        self.f = np.asarray(f)
        if self.f.shape[-1:] != (self.x.size,):
            raise ValueError("the last axis of f must match x")

    def transform(
        self,
        orders: Sequence[int],
        ratios: Sequence[float] | None = None,
        *,
        family: str | Sequence[str] = "spherical",
        derivatives: Sequence[int] | None = None,
        bias: float = 1.01,
        biases: Sequence[float] | None = None,
        mellin_size: int = 8193,
        oversampling: int = 2,
        y0: float | None = None,
        window: float = 0.0,
        mellin_window: float = 0.0,
        bins: Sequence[BinAverage | None] | None = None,
    ) -> tuple[NDArray[np.float64], NDArray]:
        plan = _Plan1D(
            self.x,
            orders,
            ratios,
            family=family,
            derivatives=derivatives,
            bias=bias,
            biases=biases,
            mellin_size=mellin_size,
            oversampling=oversampling,
            y0=y0,
            window=window,
            mellin_window=mellin_window,
            bins=bins,
        )
        return plan.apply(self.f)

    def spherical(self, ell: Sequence[int], t: Sequence[float] | None = None, **kwargs):
        """Transform with spherical Bessel functions ``j_ell(t*x*y)``."""

        return self.transform(ell, t, family="spherical", **kwargs)

    def cylindrical(self, n: Sequence[int], t: Sequence[float] | None = None, **kwargs):
        """Transform with cylindrical Bessel functions ``J_n(t*x*y)``."""

        return self.transform(n, t, family="cylindrical", **kwargs)


def _broadcast(value, dimensions: int, name: str, converter):
    if isinstance(value, (Integral, Real)):
        return (converter(value),) * dimensions
    result = tuple(converter(item) for item in value)
    if len(result) != dimensions:
        raise ValueError(f"{name} must have one entry per dimension")
    return result


class NBesselND:
    """A sampled multidimensional input with grouped Bessel functions."""

    def __init__(self, x: Sequence[ArrayLike], f: ArrayLike):
        self.x = tuple(_log_grid(grid, f"x[{index}]")[0] for index, grid in enumerate(x))
        if not self.x:
            raise ValueError("x must contain at least one grid")
        self.f = np.asarray(f)
        self.shape = tuple(grid.size for grid in self.x)
        if self.f.ndim < len(self.x) or self.f.shape[-len(self.x):] != self.shape:
            raise ValueError("the final axes of f must match the x grids")

    def transform(
        self,
        orders: Sequence[Sequence[int]],
        ratios: Sequence[Sequence[float] | None] | None = None,
        *,
        family: str | Sequence[str | Sequence[str]] = "spherical",
        derivatives: Sequence[Sequence[int] | None] | None = None,
        biases: float | Sequence[float] = 1.01,
        leg_biases: Sequence[Sequence[float] | None] | None = None,
        mellin_size: int | Sequence[int] = 8193,
        oversampling: int | Sequence[int] = 2,
        y0: float | Sequence[float] | None = None,
        window: float | Sequence[float] = 0.0,
        mellin_window: float | Sequence[float] = 0.0,
        bins: Sequence[Sequence[BinAverage | None] | None] | None = None,
    ) -> tuple[tuple[NDArray[np.float64], ...], NDArray]:
        dimensions = len(self.x)
        grouped_orders = tuple(orders)
        if len(grouped_orders) != dimensions:
            raise ValueError("orders must contain one group per x grid")
        grouped_ratios = (None,) * dimensions if ratios is None else tuple(ratios)
        grouped_derivatives = (None,) * dimensions if derivatives is None else tuple(derivatives)
        grouped_leg_biases = (None,) * dimensions if leg_biases is None else tuple(leg_biases)
        grouped_bins = (None,) * dimensions if bins is None else tuple(bins)
        if isinstance(family, str):
            grouped_families = (family,) * dimensions
        else:
            grouped_families = tuple(family)
        for name, value in (
            ("ratios", grouped_ratios),
            ("derivatives", grouped_derivatives),
            ("leg_biases", grouped_leg_biases),
            ("bins", grouped_bins),
            ("family", grouped_families),
        ):
            if len(value) != dimensions:
                raise ValueError(f"{name} must contain one group per dimension")
        total_biases = _broadcast(biases, dimensions, "biases", float)
        mellin_sizes = _broadcast(mellin_size, dimensions, "mellin_size", int)
        oversamplings = _broadcast(oversampling, dimensions, "oversampling", int)
        windows = _broadcast(window, dimensions, "window", float)
        mellin_windows = _broadcast(mellin_window, dimensions, "mellin_window", float)
        y0s = (None,) * dimensions if y0 is None else _broadcast(y0, dimensions, "y0", float)

        y_grids = []
        input_bias = np.array(1.0)
        output_bias = np.array(1.0)
        multiplier = np.array(1.0 + 0.0j)
        for axis, grid in enumerate(self.x):
            _, delta = _log_grid(grid, f"x[{axis}]")
            size = grid.size
            y_start = 1.0 / grid[-1] if y0s[axis] is None else y0s[axis]
            y = y_start * np.exp(delta * np.arange(size))
            centered = np.arange(-(size // 2), size - size // 2, dtype=np.int64)
            unshifted = np.fft.fftfreq(size) * size
            frequency_step = 2.0 * np.pi / (size * delta)
            kernel = np.fft.ifftshift(
                _product_kernel(
                    grouped_orders[axis], grouped_ratios[axis], centered, frequency_step,
                    family=grouped_families[axis], derivatives=grouped_derivatives[axis],
                    bias=total_biases[axis], biases=grouped_leg_biases[axis],
                    mellin_size=mellin_sizes[axis], oversampling=oversamplings[axis],
                    mellin_window=mellin_windows[axis], bins=grouped_bins[axis],
                )
            )
            phase = np.exp(-1j * frequency_step * unshifted * np.log(grid[0] * y[0]))
            axis_shape = [1] * dimensions
            axis_shape[axis] = size
            axis_shape = tuple(axis_shape)
            input_bias = input_bias * grid.reshape(axis_shape) ** (-total_biases[axis])
            output_bias = output_bias * y.reshape(axis_shape) ** (-total_biases[axis])
            axis_multiplier = _smooth_cutoff(unshifted, float(size // 2), windows[axis]) * phase * kernel
            if size % 2 == 0:
                axis_multiplier[size // 2] = 0.0
            multiplier = multiplier * axis_multiplier.reshape(axis_shape)
            y_grids.append(y)

        axes = tuple(range(-dimensions, 0))
        samples = np.asarray(self.f, dtype=np.complex128)
        coefficients = fftn(samples * input_bias, axes=axes)
        values = output_bias / np.prod(self.shape) * fftn(coefficients * multiplier, axes=axes)
        if not np.iscomplexobj(self.f):
            values = np.real_if_close(values, tol=1000)
            if np.iscomplexobj(values):
                scale = max(float(np.max(np.abs(values.real))), 1.0)
                if np.max(np.abs(values.imag)) > 5.0e-11 * scale:
                    raise RuntimeError("real transform retained a significant imaginary part")
                values = values.real
        return tuple(np.asarray(y) for y in y_grids), np.asarray(values)

    def spherical(self, ell: Sequence[Sequence[int]], t=None, **kwargs):
        """Transform groups of spherical Bessel functions."""

        return self.transform(ell, t, family="spherical", **kwargs)

    def cylindrical(self, n: Sequence[Sequence[int]], t=None, **kwargs):
        """Transform groups of cylindrical Bessel functions."""

        return self.transform(n, t, family="cylindrical", **kwargs)


def transform(x: ArrayLike, f: ArrayLike, orders: Sequence[int], ratios=None, **kwargs):
    """Functional shortcut for :class:`NBessel`."""

    return NBessel(x, f).transform(orders, ratios, **kwargs)


def transform_nd(x: Sequence[ArrayLike], f: ArrayLike, orders, ratios=None, **kwargs):
    """Functional shortcut for :class:`NBesselND`."""

    return NBesselND(x, f).transform(orders, ratios, **kwargs)
