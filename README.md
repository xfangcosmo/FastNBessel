# FastNBessel

FastNBessel efficiently evaluates logarithmic-grid integrals containing an
arbitrary product of Bessel functions.  In one dimension it computes

\[
F(y)=\int_0^\infty \frac{dx}{x}\,f(x)
\prod_{a=1}^{N_B} j_{\ell_a}(t_a x y),
\]

and the same formalism extends to grouped multidimensional integrals.  Both
spherical Bessel functions and cylindrical Bessel functions are supported,
including argument derivatives and scale-invariant bin averages.

The interface is deliberately compact: construct an object from a
logarithmically sampled input, then call the desired transform method.

## Installation

FastNBessel requires Python 3.10 or newer.  Install the package from the
repository root with

```shell
python -m pip install .
```

For development tests and the executable notebooks, install the optional
dependencies with

```shell
python -m pip install '.[test,examples]'
```

## One-dimensional transform

```python
import numpy as np
from nbessel import NBessel

x = np.geomspace(1e-6, 1e3, 512)
f = x**2 * np.exp(-x**2 / 2)

calc = NBessel(x, f)
y, result = calc.spherical(
    [0, 2],
    [1.0, 0.83],
    bias=1.0,
    biases=[0.5, 0.5],
)
```

Use `calc.cylindrical(...)` for products of cylindrical Bessel functions.
Argument derivatives and scale-invariant bin averages are optional:

```python
from nbessel import BinAverage, log_zero_pad

y, result = calc.spherical(
    [2],
    derivatives=[1],
    bins=[BinAverage(0.9, 1.1, dimension=3)],
)

# Optional FFTLog padding preserves the logarithmic spacing.
padded_x, padded_f = log_zero_pad(x, f, 128)
```

## Multidimensional transform

```python
from nbessel import NBesselND

x1, x2 = np.meshgrid(x, x, indexing="ij")
source = (x1 * x2)**2 * np.exp(-(x1**2 + x2**2) / 2)

(y1, y2), result = NBesselND([x, x], source).spherical(
    [[0, 1], [0, 2]],
    [[1.0, 0.8], [1.0, 1.3]],
    biases=[1.0, 1.0],
    leg_biases=[[0.5, 0.5], [0.5, 0.5]],
)
```

The final input axes correspond to the supplied logarithmic grids.  Leading
batch axes are preserved.

## Numerical controls

The input model should first be evaluated or extrapolated beyond the scales
that contribute to the requested output range.  It can then be brought to zero
only near those distant endpoints and padded with `log_zero_pad` or
`log_zero_pad_nd`.  A narrow cutoff changes the integral; zero padding only
separates periodic FFTLog images after the physical range has converged.
Always repeat a calculation with a broader model interval and more padding.
For products of two or more Bessel functions, increasing `oversampling` from 2
to 4 is a useful high-accuracy setting.  Increase `mellin_size` when a larger
internal Mellin cutoff is required.

## Tests and examples

Run the unit tests with

```shell
python -m pytest
```

The `examples/` directory contains two executed tutorial notebooks with the
FastNBessel API calls and comparison plots directly in their cells.  It also
contains noninteractive reproduction scripts, independent direct-quadrature
comparisons, all fiducial input tables, and all plots used by the numerical
validation and cosmological applications:

- 2-, 3-, 4-, and 5-Bessel one-dimensional transforms;
- the `(N_B,d)=(2,2)`, `(4,2)`, and `(6,3)` multidimensional transforms;
- galaxy and shear three-point functions;
- galaxy 3PCF and general NPCF covariance kernels;
- spherical Fourier-Bessel bispectrum and survey-window kernels; and
- a repeated radial block in a two-loop calculation.

To regenerate every checked-in input, comparison, and notebook, follow
[`examples/README.md`](examples/README.md).

## License and contact

FastNBessel is open source under the [MIT license](LICENSE).  Please open a
[GitHub issue](https://github.com/xfangcosmo/FastNBessel/issues) if you find a
bug or have a question.  Contact: Xiao Fang (`xfangcosmo@gmail.com`).
