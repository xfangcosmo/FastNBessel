"""Run every cosmological application example and make two-panel plots."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np

from nbessel import BinAverage, NBessel, NBesselND, log_zero_pad_nd
import generate_fiducial_inputs as fiducial
from quadrature import simpson_1d, simpson_2d_diagonal_interpolated


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
FIGURES = ROOT / "figures"
PAPER_FIGURE_DIRECTORY = None

LABEL_SIZE = 13
TICK_SIZE = 11
TITLE_SIZE = 12
LEGEND_SIZE = 10
plt.rcParams.update(
    {
        "axes.labelsize": LABEL_SIZE,
        "axes.titlesize": TITLE_SIZE,
        "xtick.labelsize": TICK_SIZE,
        "ytick.labelsize": TICK_SIZE,
        "legend.fontsize": LEGEND_SIZE,
        "figure.titlesize": 13,
    }
)


def _grid_table(name: str):
    table = np.loadtxt(DATA / name)
    return table[1:, 0], table[0, 1:], table[1:, 1:]


def _indices_in_range(grid, lower, upper, count=128):
    available = np.flatnonzero((grid >= lower) & (grid <= upper))
    if available.size < count:
        return available
    return np.unique(
        np.round(np.linspace(available[0], available[-1], count)).astype(int)
    )


def _fractional_difference(fast, reference):
    """Return the fractional difference epsilon_frac."""
    fast = np.asarray(fast, dtype=float)
    reference = np.asarray(reference, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        difference = (fast - reference) / reference
    return np.where(reference != 0.0, difference, np.nan)


def _set_transform_scale(axis, *curves):
    """Use a logarithmic scale, retaining signs with symlog when necessary."""
    values = np.concatenate(
        [np.ravel(np.asarray(curve, dtype=float)) for curve in curves]
    )
    values = values[np.isfinite(values)]
    if values.size and np.all(values > 0.0):
        axis.set_yscale("log")
        return

    nonzero = np.abs(values[values != 0.0])
    if nonzero.size:
        linthresh = max(float(np.max(nonzero)) * 1e-4, np.finfo(float).tiny)
        axis.set_yscale("symlog", linthresh=linthresh, linscale=0.7)
        axis.axhline(0.0, color="0.75", lw=0.7)
        tick_locations = np.asarray(axis.get_yticks())
        nonzero_ticks = np.abs(tick_locations[tick_locations != 0.0])
        nearest_tick = float(np.min(nonzero_ticks))

        def sparse_symlog_labels(value, _position):
            """Keep zero and omit only the nearest signed decade labels."""

            magnitude = abs(value)
            if value == 0.0:
                return r"$0$"
            if np.isclose(magnitude, nearest_tick, rtol=1.0e-10, atol=0.0):
                return ""
            exponent = int(np.rint(np.log10(magnitude)))
            if not np.isclose(magnitude, 10.0**exponent, rtol=1.0e-10, atol=0.0):
                return ""
            sign = "-" if value < 0.0 else ""
            return rf"${sign}10^{{{exponent}}}$"

        axis.yaxis.set_major_formatter(FuncFormatter(sparse_symlog_labels))


def _set_fractional_limits(axis, difference, clip=5.0e-4):
    """Show the full residual range, clipping only zero-crossing excursions."""
    finite = np.abs(np.asarray(difference)[np.isfinite(difference)])
    maximum = float(np.max(finite)) if finite.size else 0.0
    limit = min(1.10 * maximum, clip) if maximum > 0.0 else 1.0e-12
    axis.set_ylim(-limit, limit)
    axis.axhline(0.0, color="0.55", lw=0.7)
    if limit > 1.5e-5:
        axis.axhline(1.0e-5, color="0.55", lw=0.6, ls=":")
        axis.axhline(-1.0e-5, color="0.55", lw=0.6, ls=":")
    axis.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))


def _plot(
    name,
    x,
    fast,
    reference,
    xlabel,
    ylabel,
    title,
    xscale="log",
    transform_scale="auto",
):
    FIGURES.mkdir(parents=True, exist_ok=True)
    x = np.asarray(x)
    fast = np.asarray(fast)
    reference = np.asarray(reference)
    residual = _fractional_difference(fast, reference)

    figure, axes = plt.subplots(
        2,
        1,
        figsize=(6.2, 5.2),
        sharex=True,
        layout="constrained",
        gridspec_kw={"height_ratios": [2.2, 1.0], "hspace": 0.12},
    )
    axes[0].plot(x, reference, color="black", lw=2.0, label="quadrature")
    axes[0].plot(x, fast, color="#d55e00", lw=1.4, ls="--", label="FastNBessel")
    axes[0].set_ylabel(ylabel)
    axes[0].set_title(title)
    axes[0].legend(frameon=False)
    if transform_scale == "auto":
        _set_transform_scale(axes[0], reference, fast)
    else:
        axes[0].set_yscale(transform_scale)
        axes[0].margins(y=0.08)
    axes[1].plot(x, residual, color="#0072b2", lw=1.4)
    axes[1].set_ylabel(r"$\epsilon_{\mathrm{frac}}$")
    axes[1].set_xlabel(xlabel)
    _set_fractional_limits(axes[1], residual)
    axes[1].grid(alpha=0.2)
    for axis in axes:
        axis.set_xscale(xscale)
        axis.tick_params(which="both", labelsize=TICK_SIZE)
    figure.get_layout_engine().set(w_pad=0.04, h_pad=0.04, hspace=0.04)
    figure.savefig(FIGURES / f"{name}.png", dpi=180, bbox_inches="tight")
    if PAPER_FIGURE_DIRECTORY is not None:
        pdf_path = Path(PAPER_FIGURE_DIRECTORY) / f"{name}.pdf"
        pdf_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(
            pdf_path,
            bbox_inches="tight",
            metadata={"Creator": "FastNBessel", "CreationDate": None, "ModDate": None},
        )
    plt.close(figure)
    absolute = np.abs(residual[np.isfinite(residual)])
    absolute_difference = np.abs(fast - reference)
    reference_peak = max(float(np.max(np.abs(reference))), np.finfo(float).tiny)
    return {
        "median_absolute_fractional_difference": float(np.median(absolute)),
        "percentile_90_absolute_fractional_difference": float(
            np.quantile(absolute, 0.90)
        ),
        "percentile_95_absolute_fractional_difference": float(
            np.quantile(absolute, 0.95)
        ),
        "maximum_absolute_fractional_difference": float(np.max(absolute)),
        "fraction_below_1e-5": float(np.mean(absolute < 1.0e-5)),
        "maximum_absolute_difference": float(np.max(absolute_difference)),
        "maximum_difference_over_reference_peak": float(
            np.max(absolute_difference) / reference_peak
        ),
    }


def galaxy_3pcf():
    k1, k2, source = _grid_table("galaxy_bispectrum_multipole.txt")
    endpoint1 = fiducial.endpoint_window(k1, 1e-5, 1e-4, 20.0, 50.0)
    endpoint2 = fiducial.endpoint_window(k2, 1e-5, 1e-4, 20.0, 50.0)
    source = source * endpoint1[:, None] * endpoint2[None, :]
    average = BinAverage(0.95, 1.05, 3)
    (padded_k1, padded_k2), padded_source = log_zero_pad_nd([k1, k2], source, 384)
    (r1, r2), fast_grid = NBesselND([padded_k1, padded_k2], padded_source).spherical(
        [[0], [0]],
        biases=[1.0, 1.0],
        mellin_size=3,
        bins=[[average], [average]],
        y0=[0.2, 0.2],
    )
    index = _indices_in_range(r1, 20.0, 150.0, 96)
    fast = fast_grid[index, index]
    reference = simpson_2d_diagonal_interpolated(
        k1,
        k2,
        source,
        r1[index],
        r2[index],
        [[0], [0]],
        [[1.0], [1.0]],
        bins=[[average], [average]],
        split=[0.2, 0.2],
    )
    return _plot(
        "galaxy_3pcf",
        r1[index],
        fast,
        reference,
        r"$r\ [h^{-1}{\rm Mpc}]$",
        r"$\overline{\zeta}_{g,00}(r,r)$",
        "Galaxy 3PCF: tree-level bispectrum monopole",
    )


def shear_3pcf():
    ell1, ell2, source = _grid_table("shear_bispectrum_multipole.txt")
    average = BinAverage(0.95, 1.05, 2)
    (theta1, theta2), fast_grid = NBesselND([ell1, ell2], source).cylindrical(
        [[-3], [-3]],
        biases=[0.0, 0.0],
        mellin_size=3,
        bins=[[average], [average]],
        y0=[1e-7, 1e-7],
    )
    arcmin = theta1 * 180.0 * 60.0 / np.pi
    index = _indices_in_range(arcmin, 2.0, 20.0, 96)
    fast = fast_grid[index, index]
    reference = simpson_2d_diagonal_interpolated(
        ell1,
        ell2,
        source,
        theta1[index],
        theta2[index],
        [[-3], [-3]],
        [[1.0], [1.0]],
        families=["cylindrical", "cylindrical"],
        bins=[[average], [average]],
        split=[100.0, 100.0],
    )
    return _plot(
        "shear_3pcf",
        arcmin[index],
        fast,
        reference,
        r"$\theta\ [{\rm arcmin}]$",
        r"$\Gamma_0^{\times,(0)}(\theta,\theta)$",
        "Shear 3PCF: thin-lens bispectrum moment",
        transform_scale="linear",
    )


def galaxy_3pcf_covariance():
    """Compare one four-Bessel block, not the sum over all Wick pairings."""

    k1, k2, source = _grid_table("gaussian_3pcf_covariance_source.txt")
    endpoint1 = fiducial.endpoint_window(k1, 1e-5, 1e-4, 20.0, 50.0)
    endpoint2 = fiducial.endpoint_window(k2, 1e-5, 1e-4, 20.0, 50.0)
    source = source * endpoint1[:, None] * endpoint2[None, :]
    (padded_k1, padded_k2), padded_source = log_zero_pad_nd([k1, k2], source, 192)
    orders = [[0, 2], [1, 1]]
    ratios = [[1.0, 0.85], [1.0, 0.9]]
    average = BinAverage(0.95, 1.05, 3)
    bins = [[average, average], [average, average]]
    (r1, r2), fast_grid = NBesselND([padded_k1, padded_k2], padded_source).spherical(
        orders,
        ratios,
        biases=[1.0, 1.0],
        leg_biases=[[0.5, 0.5], [0.5, 0.5]],
        mellin_size=65537,
        oversampling=4,
        bins=bins,
        y0=[0.2, 0.2],
    )
    index = _indices_in_range(r1, 3.0, 150.0, 96)
    fast = fast_grid[index, index]
    reference = simpson_2d_diagonal_interpolated(
        k1,
        k2,
        source,
        r1[index],
        r2[index],
        orders,
        ratios,
        bins=bins,
        split=[0.2, 0.2],
    )
    return _plot(
        "galaxy_3pcf_covariance",
        r1[index],
        fast,
        reference,
        r"$r\ [h^{-1}{\rm Mpc}]$",
        r"$C_{\zeta\zeta}^{\rm block}(r)$",
        "Gaussian anisotropic-3PCF covariance block",
    )


def npcf_covariance():
    table = np.loadtxt(DATA / "linear_matter_power.txt")
    k, raw_source = table[:, 0], table[:, 3]
    source = raw_source * fiducial.endpoint_window(k, 1e-7, 1e-6, 100.0, 1e3)
    ratios = [1.0, 0.8, 1.2]
    r, fast_grid = NBessel(k, source).spherical(
        [0, 1, 2],
        ratios,
        bias=1.2,
        biases=[0.4] * 3,
        mellin_size=65537,
        oversampling=4,
        y0=0.2,
    )
    index = _indices_in_range(r, 2.0, 180.0)
    dense_k = np.geomspace(1.0e-7, 1.0e3, 131073)
    dense_source = fiducial.npcf_transform_source(dense_k)
    reference = simpson_1d(dense_k, dense_source, r[index], [0, 1, 2], ratios)
    return _plot(
        "npcf_covariance",
        r[index],
        fast_grid[index],
        reference,
        r"$r\ [h^{-1}{\rm Mpc}]$",
        r"$f_{012}(r,0.8r,1.2r)$",
        "Example integral in NPCF covariance",
    )


def sfb_bispectrum():
    table = np.loadtxt(DATA / "sfb_radial_sources.txt")
    radius, source = table[:, 0], table[:, 6]
    ratios = [1.0, 0.8, 1.2]
    k, fast_grid = NBessel(radius, source).spherical(
        [0, 1, 2],
        ratios,
        bias=2.0,
        biases=[2.0 / 3.0] * 3,
        mellin_size=65537,
        oversampling=4,
        y0=2e-4,
    )
    index = _indices_in_range(k, 0.002, 0.15)
    dense_radius = np.geomspace(0.1, 1.0e4, 32769)
    dense_source = fiducial.sfb_radial_inputs(dense_radius)[5]
    reference = simpson_1d(dense_radius, dense_source, k[index], [0, 1, 2], ratios)
    return _plot(
        "sfb_bispectrum",
        k[index],
        fast_grid[index],
        reference,
        r"$k\ [h\,{\rm Mpc}^{-1}]$",
        r"${\cal J}^{012}(k,0.8k,1.2k)$",
        "Spherical Fourier-Bessel bispectrum kernel",
    )


def sfb_window():
    table = np.loadtxt(DATA / "sfb_radial_sources.txt")
    radius, bias_source, rsd_source = table[:, 0], table[:, 7], table[:, 8]
    ratio = 0.8
    common = dict(
        bias=1.0, biases=[0.5, 0.5], mellin_size=65537, oversampling=4, y0=2e-4
    )
    k, bias_fast = NBessel(radius, bias_source).spherical(
        [2, 2], [1.0, ratio], **common
    )
    _, rsd_fast = NBessel(radius, rsd_source).spherical(
        [2, 2], [1.0, ratio], derivatives=[0, 2], **common
    )
    index = _indices_in_range(k, 0.002, 0.15)
    prefactor = 2.0 * ratio * k[index] ** 2 / np.pi
    fast = prefactor * (bias_fast[index] + rsd_fast[index])
    dense_radius = np.geomspace(0.1, 1.0e4, 32769)
    dense_inputs = fiducial.sfb_radial_inputs(dense_radius)
    dense_bias_source, dense_rsd_source = dense_inputs[6], dense_inputs[7]
    bias_quad = simpson_1d(
        dense_radius, dense_bias_source, k[index], [2, 2], [1.0, ratio]
    )
    rsd_quad = simpson_1d(
        dense_radius,
        dense_rsd_source,
        k[index],
        [2, 2],
        [1.0, ratio],
        derivatives=[0, 2],
    )
    reference = prefactor * (bias_quad + rsd_quad)
    return _plot(
        "sfb_window",
        k[index],
        fast,
        reference,
        r"$k\ [h\,{\rm Mpc}^{-1}]$",
        r"${\cal W}_2(k,0.8k)$",
        "SFB survey window with density and RSD terms",
    )


def two_loop_kernel():
    table = np.loadtxt(DATA / "two_loop_source.txt")
    q, raw_source = table[:, 0], table[:, 2]
    source = raw_source * fiducial.endpoint_window(q, 1e-7, 1e-6, 100.0, 1e3)
    ratios = [1.0, 0.75, 1.25]
    r, fast_grid = NBessel(q, source).spherical(
        [0, 1, 2],
        ratios,
        bias=1.2,
        biases=[0.4] * 3,
        mellin_size=65537,
        oversampling=4,
        y0=0.2,
    )
    index = _indices_in_range(r, 2.0, 180.0)
    dense_q = np.geomspace(1.0e-7, 1.0e3, 131073)
    dense_source = fiducial.two_loop_transform_source(dense_q)
    reference = simpson_1d(dense_q, dense_source, r[index], [0, 1, 2], ratios)
    return _plot(
        "two_loop_kernel",
        r[index],
        fast_grid[index],
        reference,
        r"$r\ [h^{-1}{\rm Mpc}]$",
        r"${\cal F}_{012}(0.75r,1.25r;r)$",
        "Example integral in a two-loop calculation",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--paper-figure-dir",
        type=Path,
        help="also write vector PDF figures to this directory",
    )
    arguments = parser.parse_args()
    global PAPER_FIGURE_DIRECTORY
    PAPER_FIGURE_DIRECTORY = arguments.paper_figure_dir

    if not (DATA / "linear_matter_power.txt").exists():
        from generate_fiducial_inputs import main as generate

        generate()
    cases = [
        ("galaxy_3pcf", galaxy_3pcf),
        ("shear_3pcf", shear_3pcf),
        ("galaxy_3pcf_covariance", galaxy_3pcf_covariance),
        ("npcf_covariance", npcf_covariance),
        ("sfb_bispectrum", sfb_bispectrum),
        ("sfb_window", sfb_window),
        ("two_loop_kernel", two_loop_kernel),
    ]
    results = {}
    for name, function in cases:
        result = function()
        results[name] = result
        print(
            f"{name:28s} median={result['median_absolute_fractional_difference']:.3e} "
            f"p90={result['percentile_90_absolute_fractional_difference']:.3e} "
            f"p95={result['percentile_95_absolute_fractional_difference']:.3e} "
            f"below1e-5={result['fraction_below_1e-5']:.1%} "
            f"max={result['maximum_absolute_fractional_difference']:.3e}"
        )
    with (FIGURES / "application_results.json").open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
        handle.write("\n")


if __name__ == "__main__":
    main()
