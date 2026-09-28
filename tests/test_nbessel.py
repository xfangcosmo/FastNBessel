from __future__ import annotations

import numpy as np
from numpy.testing import assert_allclose
from scipy.integrate import quad
from scipy.special import jv, spherical_jn

from nbessel import (
    BinAverage,
    NBessel,
    NBesselND,
    log_zero_pad,
    log_zero_pad_nd,
    transform,
)


def _quad(r, orders, ratios, power, family="spherical", derivatives=None):
    derivatives = [0] * len(orders) if derivatives is None else derivatives

    def integrand(logk):
        k = np.exp(logk)
        value = k**power * np.exp(-0.5 * k**2)
        for order, ratio, derivative in zip(orders, ratios, derivatives, strict=True):
            argument = ratio * k * r
            if family == "spherical":
                value *= spherical_jn(order, argument, derivative=bool(derivative))
            else:
                if derivative:
                    raise NotImplementedError
                value *= jv(order, argument)
        return value

    return quad(
        integrand, np.log(1e-8), np.log(25.0), epsabs=1e-10, epsrel=1e-9, limit=500
    )[0]


def test_two_and_three_bessel_examples_match_quadrature():
    k = np.geomspace(1e-7, 1e3, 512)
    cases = [
        ([0, 2], [1.0, 0.83], 2, 1.0, [0.5, 0.5]),
        ([0, 1, 2], [1.0, 0.8, 1.3], 3, 1.2, [0.4] * 3),
    ]
    for orders, ratios, power, bias, biases in cases:
        r, value = NBessel(k, k**power * np.exp(-0.5 * k**2)).spherical(
            orders,
            ratios,
            bias=bias,
            biases=biases,
            mellin_size=8193,
            y0=1e-2,
        )
        index = np.array([70, 120, 170])
        expected = np.array([_quad(r[i], orders, ratios, power) for i in index])
        assert_allclose(value[index], expected, rtol=5e-5, atol=4e-9)


def test_cylindrical_transform_and_negative_integer_order():
    x = np.geomspace(1e-7, 1e3, 512)
    f = x**3 * np.exp(-0.5 * x**2)
    theta, positive = NBessel(x, f).cylindrical(
        [0, 6], [1.0, 0.7], bias=0.7, biases=[0.35, 0.35], mellin_size=32769, y0=1e-2
    )
    _, negative = NBessel(x, f).cylindrical(
        [0, -6], [1.0, 0.7], bias=0.7, biases=[0.35, 0.35], mellin_size=32769, y0=1e-2
    )
    index = np.array([120, 130, 140])
    expected = np.array(
        [_quad(theta[i], [0, 6], [1.0, 0.7], 3, "cylindrical") for i in index]
    )
    assert_allclose(positive[index], expected, rtol=1e-4, atol=5e-9)
    assert_allclose(negative, positive, rtol=2e-13, atol=2e-13)


def test_derivative_and_bin_average_interfaces():
    k = np.geomspace(1e-7, 1e3, 512)
    f = k**2 * np.exp(-0.5 * k**2)
    r, derivative = transform(
        k, f, [2], derivatives=[1], bias=1.01, mellin_size=3, y0=1e-2
    )
    index = np.array([80, 130, 180])
    expected = np.array([_quad(r[i], [2], [1.0], 2, derivatives=[1]) for i in index])
    assert_allclose(derivative[index], expected, rtol=3e-5, atol=4e-9)

    average = BinAverage(0.9, 1.1, 3)
    _, binned = NBessel(k, f).spherical(
        [1], bins=[average], bias=1.0, mellin_size=3, y0=1e-2
    )
    nodes, weights = np.polynomial.legendre.leggauss(24)
    scale = 0.1 * nodes + 1.0
    radial_weight = weights * 0.1 * scale**2
    radial_weight /= np.sum(radial_weight)
    direct = np.array(
        [
            np.sum(
                radial_weight
                * np.array([_quad(r[i] * s, [1], [1.0], 2) for s in scale])
            )
            for i in index
        ]
    )
    assert_allclose(binned[index], direct, rtol=4e-5, atol=5e-9)


def test_log_zero_padding_helpers():
    x = np.geomspace(1.0e-3, 1.0e2, 16)
    values = np.arange(32.0).reshape(2, 16)
    padded_x, padded = log_zero_pad(x, values, 3)
    assert padded_x.size == 22
    assert padded.shape == (2, 22)
    assert_allclose(padded[:, 3:-3], values)
    assert_allclose(padded[:, :3], 0.0)
    assert_allclose(np.diff(np.log(padded_x)), np.diff(np.log(x))[0])

    source = np.outer(x, x)
    grids, padded_source = log_zero_pad_nd([x, x], source, [2, 4])
    assert tuple(grid.size for grid in grids) == (20, 24)
    assert padded_source.shape == (20, 24)
    assert_allclose(padded_source[2:-2, 4:-4], source)


def test_four_bessel_two_dimensional_separable_limit():
    k1 = np.geomspace(1e-6, 1e3, 64)
    k2 = np.geomspace(1e-5, 1e2, 48)
    f1 = k1**2 * np.exp(-0.5 * k1**2)
    f2 = k2**3 * np.exp(-0.5 * k2**2)
    arguments = dict(bias=1.0, biases=[0.5, 0.5], mellin_size=4097, y0=1e-2)
    r1, first = NBessel(k1, f1).spherical([0, 1], [1.0, 0.8], **arguments)
    r2, second = NBessel(k2, f2).spherical([0, 2], [1.0, 1.3], **arguments)
    (actual_r1, actual_r2), actual = NBesselND(
        [k1, k2], f1[:, None] * f2[None, :]
    ).spherical(
        [[0, 1], [0, 2]],
        [[1.0, 0.8], [1.0, 1.3]],
        biases=[1.0, 1.0],
        leg_biases=[[0.5, 0.5], [0.5, 0.5]],
        mellin_size=4097,
        y0=[1e-2, 1e-2],
    )
    assert_allclose(actual_r1, r1, rtol=0, atol=0)
    assert_allclose(actual_r2, r2, rtol=0, atol=0)
    assert_allclose(actual, first[:, None] * second[None, :], rtol=8e-14, atol=3e-15)


def test_batch_axes_are_preserved():
    k = np.geomspace(1e-6, 1e3, 64)
    f = k**2 * np.exp(-0.5 * k**2)
    _, values = NBessel(k, np.stack([f, 2 * f])).spherical([0], bias=1.0, mellin_size=3)
    assert values.shape == (2, 64)
    assert_allclose(values[1], 2 * values[0], rtol=2e-14, atol=2e-14)
