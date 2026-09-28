"""Reproduce every numerical-validation example in the manuscript."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from time import perf_counter

# These limits must be set before importing numerical libraries.  They make
# the standalone benchmark a one-thread comparison on common BLAS/OpenMP
# runtimes; set_workers(1) below additionally fixes the FFT worker count.
for _thread_variable in (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "BLIS_NUM_THREADS",
):
    os.environ[_thread_variable] = "1"

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter
import numpy as np
from scipy.fft import set_workers

from nbessel import NBessel, NBesselND
from quadrature import simpson_1d, simpson_2d_diagonal, simpson_3d_points


ROOT = Path(__file__).resolve().parent
FIGURES = ROOT / "figures"
PAPER_FIGURE_DIRECTORY = None

LABEL_SIZE = 13
TICK_SIZE = 11
TITLE_SIZE = 12
LEGEND_SIZE = 10
ANNOTATION_SIZE = 9
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

QUADRATURE_TOLERANCE = 1.0e-5
QUADRATURE_FLOOR_FRACTION = 1.0e-5
ONE_D_QUADRATURE_SIZES = (257, 513, 1025, 2049, 4097, 8193, 16385, 32769)
ND_QUADRATURE_SIZES = {
    2: (17, 33, 65, 129, 257, 513),
    3: (9, 17, 33, 65, 129, 193, 257),
}

ONE_D_CASES = [
    dict(p=2, orders=[0, 2], ratios=[1.0, 0.83], power=2, bias=1.0, biases=[0.5, 0.5], mellin=16385),
    dict(p=3, orders=[0, 1, 2], ratios=[1.0, 0.8, 1.3], power=3, bias=1.2, biases=[0.4] * 3, mellin=16385),
    dict(p=4, orders=[0, 1, 2, 3], ratios=[1.0, 0.73, 1.11, 1.41], power=4, bias=1.6, biases=[0.4] * 4, mellin=32769),
    dict(p=5, orders=[0, 1, 2, 3, 4], ratios=[1.0, 0.69, 0.91, 1.17, 1.43], power=5, bias=2.0, biases=[0.4] * 5, mellin=32769),
]


def coupled_phi(grids):
    value = np.ones_like(grids[0])
    logarithms = []
    for grid in grids:
        value *= grid**2 * np.exp(-0.5 * grid**2)
        logarithms.append(np.log(grid))
    coupling = np.zeros_like(value)
    for first in range(len(grids)):
        for second in range(first + 1, len(grids)):
            coupling += (logarithms[first] - logarithms[second]) ** 2
    return value * (1.0 + 0.2 * np.exp(-coupling / (2.0 * 0.8**2)))


def _save(figure, name, pdf_path=None, layout_kwargs=None):
    if layout_kwargs is not None:
        figure.get_layout_engine().set(**layout_kwargs)
    FIGURES.mkdir(parents=True, exist_ok=True)
    figure.savefig(FIGURES / f"{name}.png", dpi=180, bbox_inches="tight")
    if pdf_path is None and PAPER_FIGURE_DIRECTORY is not None:
        pdf_path = Path(PAPER_FIGURE_DIRECTORY) / f"{name}.pdf"
    if pdf_path is not None:
        pdf_path = Path(pdf_path)
        pdf_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(
            pdf_path,
            bbox_inches="tight",
            metadata={"Creator": "FastNBessel", "CreationDate": None, "ModDate": None},
        )
    plt.close(figure)


def _fractional_difference(fast, reference):
    """Return the fractional difference epsilon_frac."""
    fast = np.asarray(fast, dtype=float)
    reference = np.asarray(reference, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        difference = (fast - reference) / reference
    return np.where(reference != 0.0, difference, np.nan)


def _set_transform_scale(axis, *curves):
    """Use a logarithmic scale, retaining signs with symlog when necessary."""
    values = np.concatenate([np.ravel(np.asarray(curve, dtype=float)) for curve in curves])
    values = values[np.isfinite(values)]
    if values.size and np.all(values > 0.0):
        axis.set_yscale("log")
        return

    nonzero = np.abs(values[values != 0.0])
    if nonzero.size:
        linthresh = max(float(np.max(nonzero)) * 1e-4, np.finfo(float).tiny)
        axis.set_yscale("symlog", linthresh=linthresh, linscale=0.7)
        axis.axhline(0.0, color="0.75", lw=0.6)
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
            if not np.isclose(
                magnitude, 10.0**exponent, rtol=1.0e-10, atol=0.0
            ):
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
    axis.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))


def one_dimensional_accuracy():
    k = np.geomspace(1e-7, 1e3, 512)
    dense_k = np.geomspace(1e-7, 1e3, 32769)
    figure, axes = plt.subplots(
        2,
        4,
        figsize=(13.5, 5.8),
        sharex="col",
        layout="constrained",
        gridspec_kw={"height_ratios": [2.1, 1.0], "hspace": 0.12},
    )
    metrics = {}
    for column, case in enumerate(ONE_D_CASES):
        f = k**case["power"] * np.exp(-0.5 * k**2)
        r, fast_grid = NBessel(k, f).spherical(
            case["orders"], case["ratios"], bias=case["bias"],
            biases=case["biases"], mellin_size=case["mellin"], y0=1e-2
        )
        candidates = np.flatnonzero((r >= 3e-2) & (r <= 30.0))
        index = np.unique(np.round(np.linspace(candidates[0], candidates[-1], 160)).astype(int))
        reference = simpson_1d(
            dense_k, dense_k**case["power"] * np.exp(-0.5 * dense_k**2),
            r[index], case["orders"], case["ratios"]
        )
        fast = fast_grid[index]
        difference = _fractional_difference(fast, reference)
        absolute = np.abs(difference[np.isfinite(difference)])
        metrics[f"p{case['p']}_median_absolute_fractional_difference"] = float(np.median(absolute))
        metrics[f"p{case['p']}_p95_absolute_fractional_difference"] = float(np.quantile(absolute, 0.95))
        metrics[f"p{case['p']}_maximum_absolute_fractional_difference"] = float(np.max(absolute))

        axes[0, column].plot(r[index], reference, color="black", lw=2.2, label="Simpson")
        axes[0, column].plot(r[index], fast, color="#d55e00", ls="--", lw=1.3, label="FastNBessel")
        axes[0, column].set_title(fr"$p={case['p']}$")
        _set_transform_scale(axes[0, column], reference, fast)
        axes[1, column].plot(r[index], difference, color="#0072b2")
        _set_fractional_limits(axes[1, column], difference)
        axes[1, column].set_xlabel(r"$r$")
        for row in range(2):
            axes[row, column].set_xscale("log")
        if column == 0:
            axes[0, column].set_ylabel(r"$I_p(r)$")
            axes[1, column].set_ylabel(r"$\epsilon_{\mathrm{frac}}$")
            axes[0, column].legend(frameon=False)
    figure.suptitle("One-dimensional multiple-Bessel validation")
    _save(
        figure,
        "validation_1d",
        layout_kwargs={
            "h_pad": 0.04,
            "w_pad": 0.04,
            "wspace": 0.03,
        },
    )
    return metrics


def multidimensional_accuracy():
    configurations = [
        dict(label=r"$(N_B,d)=(2,2)$", size=128, orders=[[0], [2]], ratios=[[1.0], [1.0]], y0=1e-2),
        dict(label=r"$(N_B,d)=(4,2)$", size=128, orders=[[0, 1], [0, 2]], ratios=[[1.0, 0.8], [1.0, 1.3]], y0=1e-2),
    ]
    figure, axes = plt.subplots(
        2, 3, figsize=(11.2, 5.8), sharex="col",
        layout="constrained",
        gridspec_kw={"height_ratios": [2.1, 1.0], "hspace": 0.12},
    )
    metrics = {}
    for column, config in enumerate(configurations):
        k = np.geomspace(1e-6, 1e3, config["size"])
        mesh = np.meshgrid(k, k, indexing="ij")
        group_biases = [None if len(group) == 1 else [0.5, 0.5] for group in config["orders"]]
        (r1, r2), fast_grid = NBesselND([k, k], coupled_phi(mesh)).spherical(
            config["orders"], config["ratios"], biases=[1.0, 1.0],
            leg_biases=group_biases, mellin_size=8193, y0=[config["y0"]] * 2
        )
        candidates = np.flatnonzero((r1 >= 4e-2) & (r1 <= 4.0))
        index = candidates
        dense_k = np.geomspace(1e-6, 1e3, 501)
        dense_mesh = np.meshgrid(dense_k, dense_k, indexing="ij")
        reference = simpson_2d_diagonal(
            dense_k, dense_k, coupled_phi(dense_mesh), r1[index], r2[index],
            config["orders"], config["ratios"]
        )
        fast = fast_grid[index, index]
        difference = _fractional_difference(fast, reference)
        absolute = np.abs(difference[np.isfinite(difference)])
        key = "2b2d" if column == 0 else "4b2d"
        metrics[f"{key}_median_absolute_fractional_difference"] = float(np.median(absolute))
        metrics[f"{key}_p95_absolute_fractional_difference"] = float(np.quantile(absolute, 0.95))
        metrics[f"{key}_maximum_absolute_fractional_difference"] = float(np.max(absolute))
        axes[0, column].plot(r1[index], reference, color="black", lw=2.2, label="tensor Simpson")
        axes[0, column].plot(r1[index], fast, color="#d55e00", ls="--", lw=1.3, label="FastNBessel")
        axes[0, column].set_title(config["label"] + r", $r_1=r_2$")
        _set_transform_scale(axes[0, column], reference, fast)
        axes[1, column].plot(r1[index], difference, color="#0072b2")
        _set_fractional_limits(axes[1, column], difference)

    # Six Bessel functions in three dimensions.  This case is especially
    # sensitive to low-frequency wrap-around because its small-r signal is
    # proportional to a high power of r.  A denser finite interval and a
    # shallower total bias suppress that periodic image.
    k = np.geomspace(1e-7, 30.0, 160)
    mesh = np.meshgrid(k, k, k, indexing="ij")
    orders = [[0, 1], [1, 2], [0, 2]]
    ratios = [[1.0, 0.8], [1.0, 1.2], [1.0, 0.9]]
    r, fast_grid = NBesselND([k] * 3, coupled_phi(mesh)).spherical(
        orders, ratios, biases=[0.5] * 3, leg_biases=[[0.25, 0.25]] * 3,
        mellin_size=32769, oversampling=4, y0=[2e-2] * 3
    )
    candidates = np.flatnonzero((r[0] >= 6e-2) & (r[0] <= 1.2))
    index = np.unique(np.round(np.linspace(candidates[0], candidates[-1], 16)).astype(int))
    points = np.column_stack([axis[index] for axis in r])
    dense_k = np.geomspace(1e-7, 30.0, 193)
    dense_mesh = np.meshgrid(dense_k, dense_k, dense_k, indexing="ij")
    reference = simpson_3d_points([dense_k] * 3, coupled_phi(dense_mesh), points, orders, ratios)
    fast = fast_grid[index, index, index]
    difference = _fractional_difference(fast, reference)
    absolute = np.abs(difference[np.isfinite(difference)])
    metrics["6b3d_median_absolute_fractional_difference"] = float(np.median(absolute))
    metrics["6b3d_p95_absolute_fractional_difference"] = float(np.quantile(absolute, 0.95))
    metrics["6b3d_maximum_absolute_fractional_difference"] = float(np.max(absolute))
    axes[0, 2].plot(points[:, 0], reference, color="black", lw=2.2, marker="o", ms=3, label="tensor Simpson")
    axes[0, 2].plot(points[:, 0], fast, color="#d55e00", ls="--", lw=1.3, label="FastNBessel")
    axes[0, 2].set_title(r"$(N_B,d)=(6,3)$, $r_1=r_2=r_3$")
    _set_transform_scale(axes[0, 2], reference, fast)
    axes[1, 2].plot(points[:, 0], difference, color="#0072b2", marker="o", ms=3)
    _set_fractional_limits(axes[1, 2], difference)

    for column in range(3):
        axes[0, column].set_xscale("log")
        axes[1, column].set_xscale("log")
        axes[1, column].set_xlabel(r"$r$")
    axes[0, 0].set_ylabel("transform")
    axes[1, 0].set_ylabel(r"$\epsilon_{\mathrm{frac}}$")
    axes[0, 0].legend(frameon=False)
    figure.suptitle("Multidimensional multiple-Bessel validation")
    _save(
        figure,
        "validation_multidimensional",
        layout_kwargs={
            "h_pad": 0.04,
            "w_pad": 0.04,
            "wspace": 0.03,
        },
    )
    return metrics


def _median_time(function, repetitions=3):
    function()
    samples = []
    for _ in range(repetitions):
        start = perf_counter()
        function()
        samples.append(perf_counter() - start)
    return float(np.median(samples))


def _quadrature_change(current, previous):
    """Maximum mixed relative change between successive quadrature grids."""

    current = np.asarray(current, dtype=float)
    previous = np.asarray(previous, dtype=float)
    peak = max(float(np.max(np.abs(current))), np.finfo(float).tiny)
    scale = np.maximum(np.abs(current), QUADRATURE_FLOOR_FRACTION * peak)
    return float(np.max(np.abs(current - previous) / scale))


def _refine_quadrature(evaluate, sample_sizes):
    """Refine a tensor-Simpson calculation until its result has converged."""

    previous = None
    last_error = np.inf
    for sample_size in sample_sizes:
        current = np.asarray(evaluate(sample_size))
        if previous is not None:
            last_error = _quadrature_change(current, previous)
            if last_error < QUADRATURE_TOLERANCE:
                return current, int(sample_size), last_error
        previous = current
    raise RuntimeError(
        "tensor-Simpson quadrature did not converge below "
        f"{QUADRATURE_TOLERANCE:g}; last change was {last_error:.3e}"
    )


def performance_and_convergence(performance_pdf=None):
    figure, axes = plt.subplots(
        1, 3, figsize=(13.5, 4.25), layout="constrained"
    )
    sizes = np.array([64, 128, 256, 512])
    timing_metrics = {}
    with set_workers(1):
        for case, color in zip(ONE_D_CASES, ["#0072b2", "#e69f00", "#009e73", "#cc79a7"], strict=True):
            quadrature_inputs = {}
            for sample_size in ONE_D_QUADRATURE_SIZES:
                dense = np.geomspace(1e-7, 1e3, sample_size)
                dense_f = dense**case["power"] * np.exp(-0.5 * dense**2)
                quadrature_inputs[sample_size] = (dense, dense_f)

            fast_times, quadrature_times = [], []
            for size in sizes:
                k = np.geomspace(1e-7, 1e3, int(size))
                f = k**case["power"] * np.exp(-0.5 * k**2)
                kwargs = dict(
                    bias=case["bias"], biases=case["biases"],
                    mellin_size=case["mellin"], y0=1e-2
                )
                fast_times.append(_median_time(lambda: NBessel(k, f).spherical(case["orders"], case["ratios"], **kwargs)))
                outputs = np.geomspace(3e-2, 30.0, int(size))

                def adaptive_quadrature():
                    return _refine_quadrature(
                        lambda sample_size: simpson_1d(
                            quadrature_inputs[sample_size][0],
                            quadrature_inputs[sample_size][1],
                            outputs,
                            case["orders"],
                            case["ratios"],
                        ),
                        ONE_D_QUADRATURE_SIZES,
                    )

                _, converged_size, convergence_error = adaptive_quadrature()
                quadrature_times.append(
                    _median_time(adaptive_quadrature, repetitions=2)
                )
                if size == sizes[-1]:
                    timing_metrics[f"p{case['p']}_simpson_samples"] = converged_size
                    timing_metrics[f"p{case['p']}_simpson_convergence"] = convergence_error
            axes[0].plot(sizes, fast_times, marker="o", color=color, label=fr"FastNBessel $p={case['p']}$")
            axes[0].plot(sizes, quadrature_times, marker="s", ls="--", color=color, alpha=0.75, label=fr"Simpson $p={case['p']}$")
            timing_metrics[f"p{case['p']}_fast_N512_seconds"] = fast_times[-1]
            timing_metrics[f"p{case['p']}_simpson_N512_seconds"] = quadrature_times[-1]

        # Multidimensional scaling; the quadrature curve is an extrapolation
        # from one measured tensor-Simpson point to all output grid points.
        for dimensions, orders, ratios, sizes_nd, color in [
            (2, [[0, 1], [0, 2]], [[1.0, 0.8], [1.0, 1.3]], np.array([24, 32, 48, 64]), "#0072b2"),
            (3, [[0, 1], [1, 2], [0, 2]], [[1.0, 0.8], [1.0, 1.2], [1.0, 0.9]], np.array([12, 16, 20, 24]), "#d55e00"),
        ]:
            quadrature_inputs = {}
            for sample_size in ND_QUADRATURE_SIZES[dimensions]:
                dense = np.geomspace(1e-6, 1e2, sample_size)
                dense_mesh = np.meshgrid(*([dense] * dimensions), indexing="ij")
                quadrature_inputs[sample_size] = (dense, coupled_phi(dense_mesh))

            point = np.full((1, dimensions), 0.3)

            def adaptive_quadrature_point():
                def evaluate(sample_size):
                    dense, dense_values = quadrature_inputs[sample_size]
                    if dimensions == 2:
                        return simpson_2d_diagonal(
                            dense, dense, dense_values,
                            point[:, 0], point[:, 1], orders, ratios,
                        )
                    return simpson_3d_points(
                        [dense] * 3, dense_values, point, orders, ratios
                    )

                return _refine_quadrature(
                    evaluate, ND_QUADRATURE_SIZES[dimensions]
                )

            _, converged_size, convergence_error = adaptive_quadrature_point()
            one_point_time = _median_time(
                adaptive_quadrature_point, repetitions=2
            )
            key = "4b2d" if dimensions == 2 else "6b3d"
            timing_metrics[f"{key}_simpson_samples_per_axis"] = converged_size
            timing_metrics[f"{key}_simpson_convergence"] = convergence_error
            timing_metrics[f"{key}_simpson_one_point_seconds"] = one_point_time

            fast_times = []
            for size in sizes_nd:
                k = np.geomspace(1e-6, 1e2, int(size))
                mesh = np.meshgrid(*([k] * dimensions), indexing="ij")
                samples = coupled_phi(mesh)
                fast_times.append(_median_time(lambda: NBesselND([k] * dimensions, samples).spherical(
                    orders, ratios, biases=[1.0] * dimensions,
                    leg_biases=[[0.5, 0.5]] * dimensions,
                    mellin_size=4097, y0=[2e-2] * dimensions
                ), repetitions=2))
            estimated_quad = one_point_time * sizes_nd.astype(float) ** dimensions
            label = r"$(4,2)$" if dimensions == 2 else r"$(6,3)$"
            axes[1].plot(sizes_nd, fast_times, marker="o", color=color, label="FastNBessel " + label)
            axes[1].plot(sizes_nd, estimated_quad, marker="s", ls="--", color=color, label="Simpson estimate " + label)

    # Mellin convergence for the p=2 test.
    case = ONE_D_CASES[0]
    k = np.geomspace(1e-7, 1e3, 512)
    f = k**2 * np.exp(-0.5 * k**2)
    r_index = np.array([70, 110, 150, 190])
    baseline_r, _ = NBessel(k, f).spherical(
        case["orders"], case["ratios"], bias=case["bias"], biases=case["biases"],
        mellin_size=32769, y0=1e-2
    )
    dense = np.geomspace(1e-7, 1e3, 32769)
    reference = simpson_1d(dense, dense**2 * np.exp(-0.5 * dense**2), baseline_r[r_index], case["orders"], case["ratios"])
    mellin_sizes = np.array([2049, 4097, 8193, 16385, 32769])
    errors, times = [], []
    with set_workers(1):
        for mellin_size in mellin_sizes:
            function = lambda m=int(mellin_size): NBessel(k, f).spherical(
                case["orders"], case["ratios"], bias=case["bias"], biases=case["biases"],
                mellin_size=m, y0=1e-2
            )
            _, values = function()
            fraction = _fractional_difference(values[r_index], reference)
            errors.append(np.nanmax(np.abs(fraction)))
            times.append(_median_time(function))
    axes[2].plot(times, errors, marker="o", color="#0072b2")
    for x, y, size in zip(times, errors, mellin_sizes, strict=True):
        axes[2].annotate(
            f"{size}",
            (x, y),
            xytext=(3, 4),
            textcoords="offset points",
            fontsize=ANNOTATION_SIZE,
        )

    axes[0].set_xlabel("outer grid size $N$")
    axes[0].set_ylabel("wall time [s]")
    axes[0].set_title("1D: all four validation cases")
    axes[0].legend(
        frameon=False, ncol=2, loc="upper left", borderaxespad=0.2
    )
    axes[1].set_xlabel("points per input axis")
    axes[1].set_title("Multidimensional scaling")
    axes[1].legend(frameon=False)
    axes[2].set_xlabel("wall time [s]")
    axes[2].set_ylabel(r"maximum $|\epsilon_{\mathrm{frac}}|$", labelpad=0)
    axes[2].set_title("Mellin-grid convergence")
    for axis in axes:
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.grid(alpha=0.18)
    axes[0].set_xticks(sizes, [str(size) for size in sizes], minor=False)
    axes[0].set_xticks([], minor=True)
    nd_ticks = [12, 16, 24, 32, 48, 64]
    axes[1].set_xticks(nd_ticks, [str(size) for size in nd_ticks], minor=False)
    axes[1].set_xticks([], minor=True)
    axes[2].xaxis.set_major_locator(LogLocator(base=10, numticks=4))
    axes[2].xaxis.set_minor_formatter(NullFormatter())
    lower, upper = axes[0].get_ylim()
    axes[0].set_ylim(lower, upper * 75.0)
    figure.suptitle("Single-thread performance and convergence")
    _save(
        figure,
        "validation_performance",
        performance_pdf,
        layout_kwargs={"w_pad": 0.06, "wspace": 0.10},
    )
    return timing_metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--performance-pdf",
        type=Path,
        help="also write the performance figure as a PDF at this path",
    )
    parser.add_argument(
        "--paper-figure-dir",
        type=Path,
        help="also write all validation figures as vector PDFs to this directory",
    )
    arguments = parser.parse_args()
    global PAPER_FIGURE_DIRECTORY
    PAPER_FIGURE_DIRECTORY = arguments.paper_figure_dir

    metrics = {}
    metrics.update(one_dimensional_accuracy())
    metrics.update(multidimensional_accuracy())
    metrics.update(performance_and_convergence(arguments.performance_pdf))
    FIGURES.mkdir(parents=True, exist_ok=True)
    (FIGURES / "validation_results.json").write_text(json.dumps(metrics, indent=2) + "\n")
    for key, value in metrics.items():
        if isinstance(value, (int, np.integer)):
            print(f"{key:42s} {value:d}")
        else:
            print(f"{key:42s} {value:.6e}")


if __name__ == "__main__":
    main()
