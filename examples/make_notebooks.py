"""Create the executable example notebooks kept with the package.

The notebooks deliberately contain the public FastNBessel calls themselves.
The standalone ``run_*.py`` programs remain useful for unattended figure
generation, but a reader should not need to open those programs to learn the
package interface.
"""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import nbformat as nbf


ROOT = Path(__file__).resolve().parent


def markdown(source: str):
    return nbf.v4.new_markdown_cell(dedent(source).strip())


def code(source: str):
    return nbf.v4.new_code_cell(dedent(source).strip())


def write_notebook(name: str, cells):
    notebook = nbf.v4.new_notebook()
    notebook["cells"] = cells
    notebook["metadata"] = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3"},
    }
    nbf.write(notebook, ROOT / name)


def numerical_notebook():
    return [
        markdown(
            r"""
            # FastNBessel numerical validation and API examples

            This notebook shows the public package calls used for the numerical
            examples in the paper.  It covers products of two through five
            spherical Bessel functions, derivatives, bin averages, cylindrical
            Bessel functions, and the multidimensional API.  The independent
            Simpson routines come from the local `quadrature.py` file and are
            not part of the installed package.

            Install the repository with `python -m pip install '.[examples]'`
            before running the notebook from this `examples/` directory.
            """
        ),
        code(
            r"""
            from time import perf_counter

            import matplotlib.pyplot as plt
            import numpy as np
            from scipy.fft import set_workers

            from nbessel import BinAverage, NBessel, NBesselND
            from quadrature import simpson_1d, simpson_2d_diagonal, simpson_3d_points

            plt.rcParams.update({
                "axes.labelsize": 12,
                "axes.titlesize": 12,
                "xtick.labelsize": 10,
                "ytick.labelsize": 10,
                "legend.fontsize": 10,
            })
            """
        ),
        markdown(
            r"""
            ## Plotting and test helpers

            These small helpers only select output points, construct the smooth
            nonseparable multidimensional test input, and format plots.  Every
            FastNBessel transform is called explicitly in the sections below.
            """
        ),
        code(
            r"""
            def fractional_difference(fast, reference):
                fast = np.asarray(fast, dtype=float)
                reference = np.asarray(reference, dtype=float)
                with np.errstate(divide="ignore", invalid="ignore"):
                    residual = (fast - reference) / reference
                return np.where(reference != 0.0, residual, np.nan)


            def indices_in_range(grid, lower, upper, count=None):
                available = np.flatnonzero((grid >= lower) & (grid <= upper))
                if count is None or available.size <= count:
                    return available
                return np.unique(
                    np.round(np.linspace(available[0], available[-1], count)).astype(int)
                )


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


            def transform_scale(axis, *curves):
                values = np.concatenate([np.ravel(curve) for curve in curves])
                values = values[np.isfinite(values)]
                if values.size and np.all(values > 0.0):
                    axis.set_yscale("log")
                else:
                    nonzero = np.abs(values[values != 0.0])
                    threshold = max(np.max(nonzero) * 1.0e-4, np.finfo(float).tiny)
                    axis.set_yscale("symlog", linthresh=threshold)
                    axis.axhline(0.0, color="0.75", lw=0.6)
            """
        ),
        markdown(
            r"""
            ## One-dimensional products: $N_B=2,3,4,5$

            `NBessel(x, f)` stores one logarithmically sampled input.  The
            `orders` and `ratios` arrays passed to `.spherical(...)` contain one
            entry per Bessel factor.  The returned array contains the transform
            on the logarithmic output grid `r`.
            """
        ),
        code(
            r"""
            one_dimensional_cases = [
                dict(orders=[0, 2], ratios=[1.0, 0.83], power=2,
                     bias=1.0, leg_biases=[0.5, 0.5], mellin_size=16385),
                dict(orders=[0, 1, 2], ratios=[1.0, 0.8, 1.3], power=3,
                     bias=1.2, leg_biases=[0.4] * 3, mellin_size=16385),
                dict(orders=[0, 1, 2, 3], ratios=[1.0, 0.73, 1.11, 1.41], power=4,
                     bias=1.6, leg_biases=[0.4] * 4, mellin_size=32769),
                dict(orders=[0, 1, 2, 3, 4], ratios=[1.0, 0.69, 0.91, 1.17, 1.43], power=5,
                     bias=2.0, leg_biases=[0.4] * 5, mellin_size=32769),
            ]

            x = np.geomspace(1.0e-7, 1.0e3, 512)
            dense_x = np.geomspace(1.0e-7, 1.0e3, 32769)
            figure, axes = plt.subplots(
                2, 4, figsize=(13.5, 5.5), sharex="col",
                gridspec_kw={"height_ratios": [2.1, 1.0]}, layout="constrained",
            )
            one_dimensional_metrics = {}

            for column, case in enumerate(one_dimensional_cases):
                source = x**case["power"] * np.exp(-0.5 * x**2)

                # This is the complete FastNBessel call for this example.
                r, fast_all = NBessel(x, source).spherical(
                    case["orders"],
                    case["ratios"],
                    bias=case["bias"],
                    biases=case["leg_biases"],
                    mellin_size=case["mellin_size"],
                    y0=1.0e-2,
                )

                index = indices_in_range(r, 3.0e-2, 30.0, 160)
                reference = simpson_1d(
                    dense_x,
                    dense_x**case["power"] * np.exp(-0.5 * dense_x**2),
                    r[index],
                    case["orders"],
                    case["ratios"],
                )
                fast = fast_all[index]
                residual = fractional_difference(fast, reference)
                finite = np.abs(residual[np.isfinite(residual)])
                one_dimensional_metrics[f"N_B={len(case['orders'])}"] = {
                    "median |epsilon_frac|": float(np.median(finite)),
                    "95th percentile": float(np.quantile(finite, 0.95)),
                }

                axes[0, column].plot(r[index], reference, color="black", lw=2, label="Simpson")
                axes[0, column].plot(r[index], fast, "--", color="#d55e00", label="FastNBessel")
                axes[1, column].plot(r[index], residual, color="#0072b2")
                axes[0, column].set_title(fr"$N_B={len(case['orders'])}$")
                transform_scale(axes[0, column], reference, fast)
                axes[1, column].axhline(0.0, color="0.6", lw=0.7)
                axes[1, column].set_xlabel(r"$r$")
                axes[0, column].set_xscale("log")
                axes[1, column].set_xscale("log")

            axes[0, 0].set_ylabel(r"$I_{N_B}(r)$")
            axes[1, 0].set_ylabel(r"$\epsilon_{\mathrm{frac}}$")
            axes[0, 0].legend(frameon=False)
            plt.show()
            one_dimensional_metrics
            """
        ),
        markdown(
            r"""
            ## Derivatives, bin averages, and cylindrical Bessel functions

            Derivative orders and `BinAverage` objects are specified one per
            Bessel factor.  Cylindrical transforms use the same object and
            numerical controls through `.cylindrical(...)`.
            """
        ),
        code(
            r"""
            source = x**2 * np.exp(-0.5 * x**2)
            calculator = NBessel(x, source)

            r, first_derivative = calculator.spherical(
                [2], derivatives=[1], bias=1.01, mellin_size=3, y0=1.0e-2
            )
            _, volume_average = calculator.spherical(
                [1],
                bins=[BinAverage(0.9, 1.1, dimension=3)],
                bias=1.0,
                mellin_size=3,
                y0=1.0e-2,
            )
            _, cylindrical = calculator.cylindrical(
                [0, 2],
                [1.0, 0.83],
                bias=1.0,
                biases=[0.5, 0.5],
                mellin_size=16385,
                y0=1.0e-2,
            )

            visible = (r >= 3.0e-2) & (r <= 30.0)
            plt.figure(figsize=(7, 4))
            plt.semilogx(r[visible], first_derivative[visible], label=r"$j_2'$ transform")
            plt.semilogx(r[visible], volume_average[visible], label="volume-averaged $j_1$")
            plt.semilogx(r[visible], cylindrical[visible], label=r"$J_0J_2$ transform")
            plt.xlabel(r"$r$")
            plt.ylabel("transform")
            plt.legend(frameon=False)
            plt.show()
            """
        ),
        markdown(
            r"""
            ## Multidimensional transforms: $(N_B,d)=(2,2)$ and $(4,2)$

            `NBesselND` receives one input grid per integration variable.  Each
            nested `orders` or `ratios` list describes the Bessel functions
            attached to that variable.  The final axes of the sampled source
            must match the input grids.
            """
        ),
        code(
            r"""
            two_dimensional_cases = [
                dict(label=r"$(N_B,d)=(2,2)$", orders=[[0], [2]],
                     ratios=[[1.0], [1.0]], leg_biases=[None, None]),
                dict(label=r"$(N_B,d)=(4,2)$", orders=[[0, 1], [0, 2]],
                     ratios=[[1.0, 0.8], [1.0, 1.3]],
                     leg_biases=[[0.5, 0.5], [0.5, 0.5]]),
            ]

            figure, axes = plt.subplots(2, 2, figsize=(9, 5.5), sharex="col", layout="constrained")
            multidimensional_metrics = {}

            for column, case in enumerate(two_dimensional_cases):
                x2d = np.geomspace(1.0e-6, 1.0e3, 128)
                mesh = np.meshgrid(x2d, x2d, indexing="ij")
                source = coupled_phi(mesh)

                # One group of Bessel functions is supplied for each input axis.
                (r1, r2), fast_grid = NBesselND([x2d, x2d], source).spherical(
                    case["orders"],
                    case["ratios"],
                    biases=[1.0, 1.0],
                    leg_biases=case["leg_biases"],
                    mellin_size=8193,
                    y0=[1.0e-2, 1.0e-2],
                )

                index = indices_in_range(r1, 4.0e-2, 4.0)
                dense_x2d = np.geomspace(1.0e-6, 1.0e3, 501)
                dense_mesh = np.meshgrid(dense_x2d, dense_x2d, indexing="ij")
                reference = simpson_2d_diagonal(
                    dense_x2d,
                    dense_x2d,
                    coupled_phi(dense_mesh),
                    r1[index],
                    r2[index],
                    case["orders"],
                    case["ratios"],
                )
                fast = fast_grid[index, index]
                residual = fractional_difference(fast, reference)
                finite = np.abs(residual[np.isfinite(residual)])
                multidimensional_metrics[case["label"]] = {
                    "median |epsilon_frac|": float(np.median(finite)),
                    "95th percentile": float(np.quantile(finite, 0.95)),
                }

                axes[0, column].plot(r1[index], reference, color="black", lw=2, label="tensor Simpson")
                axes[0, column].plot(r1[index], fast, "--", color="#d55e00", label="FastNBessel")
                axes[1, column].plot(r1[index], residual, color="#0072b2")
                axes[0, column].set_title(case["label"] + r", $r_1=r_2$")
                transform_scale(axes[0, column], reference, fast)
                axes[1, column].axhline(0.0, color="0.6", lw=0.7)
                axes[0, column].set_xscale("log")
                axes[1, column].set_xscale("log")
                axes[1, column].set_xlabel(r"$r$")

            axes[0, 0].set_ylabel("transform")
            axes[1, 0].set_ylabel(r"$\epsilon_{\mathrm{frac}}$")
            axes[0, 0].legend(frameon=False)
            plt.show()
            multidimensional_metrics
            """
        ),
        markdown(
            r"""
            ## Six Bessel functions in three dimensions: $(N_B,d)=(6,3)$

            This is the largest validation example in the paper.  Two Bessel
            functions are attached to each of the three integration variables.
            Only a small set of diagonal output points is evaluated by direct
            tensor-product quadrature.
            """
        ),
        code(
            r"""
            x3d = np.geomspace(1.0e-7, 30.0, 160)
            mesh3d = np.meshgrid(x3d, x3d, x3d, indexing="ij")
            source3d = coupled_phi(mesh3d)
            orders3d = [[0, 1], [1, 2], [0, 2]]
            ratios3d = [[1.0, 0.8], [1.0, 1.2], [1.0, 0.9]]

            (r1, r2, r3), fast_grid3d = NBesselND([x3d] * 3, source3d).spherical(
                orders3d,
                ratios3d,
                biases=[0.5] * 3,
                leg_biases=[[0.25, 0.25]] * 3,
                mellin_size=32769,
                oversampling=4,
                y0=[2.0e-2] * 3,
            )

            index3d = indices_in_range(r1, 6.0e-2, 1.2, 16)
            points3d = np.column_stack([axis[index3d] for axis in (r1, r2, r3)])
            dense_x3d = np.geomspace(1.0e-7, 30.0, 193)
            dense_mesh3d = np.meshgrid(dense_x3d, dense_x3d, dense_x3d, indexing="ij")
            reference3d = simpson_3d_points(
                [dense_x3d] * 3,
                coupled_phi(dense_mesh3d),
                points3d,
                orders3d,
                ratios3d,
            )
            fast3d = fast_grid3d[index3d, index3d, index3d]
            residual3d = fractional_difference(fast3d, reference3d)

            figure, axes = plt.subplots(2, 1, figsize=(6.5, 5.5), sharex=True, layout="constrained")
            axes[0].plot(points3d[:, 0], reference3d, "o-", color="black", label="tensor Simpson")
            axes[0].plot(points3d[:, 0], fast3d, "--", color="#d55e00", label="FastNBessel")
            axes[1].plot(points3d[:, 0], residual3d, "o-", color="#0072b2")
            transform_scale(axes[0], reference3d, fast3d)
            axes[0].legend(frameon=False)
            axes[0].set_ylabel("transform")
            axes[1].set_ylabel(r"$\epsilon_{\mathrm{frac}}$")
            axes[1].set_xlabel(r"$r_1=r_2=r_3$")
            axes[1].set_xscale("log")
            axes[1].axhline(0.0, color="0.6", lw=0.7)
            plt.show()

            finite3d = np.abs(residual3d[np.isfinite(residual3d)])
            {
                "median |epsilon_frac|": float(np.median(finite3d)),
                "95th percentile": float(np.quantile(finite3d, 0.95)),
            }
            """
        ),
        markdown(
            r"""
            ## Compact timing and Mellin-convergence example

            The publication timing figure is produced by
            `run_numerical_validation.py`, which also refines the quadrature
            until its change is below $10^{-5}$.  Here the relevant package
            calls are shown directly: the left panel varies the outer FFTLog
            size, and the right panel varies the internal Mellin grid.
            """
        ),
        code(
            r"""
            def median_time(function, repetitions=3):
                function()
                samples = []
                for _ in range(repetitions):
                    start = perf_counter()
                    function()
                    samples.append(perf_counter() - start)
                return float(np.median(samples))


            outer_sizes = np.array([64, 128, 256, 512])
            timing_results = {}
            figure, axes = plt.subplots(1, 2, figsize=(10.5, 4), layout="constrained")

            with set_workers(1):
                for case in one_dimensional_cases:
                    elapsed = []
                    for size in outer_sizes:
                        timing_x = np.geomspace(1.0e-7, 1.0e3, int(size))
                        timing_source = timing_x**case["power"] * np.exp(-0.5 * timing_x**2)

                        def evaluate():
                            return NBessel(timing_x, timing_source).spherical(
                                case["orders"],
                                case["ratios"],
                                bias=case["bias"],
                                biases=case["leg_biases"],
                                mellin_size=case["mellin_size"],
                                y0=1.0e-2,
                            )

                        elapsed.append(median_time(evaluate, repetitions=2))
                    timing_results[f"N_B={len(case['orders'])}"] = elapsed
                    axes[0].plot(outer_sizes, elapsed, "o-", label=fr"$N_B={len(case['orders'])}$")

                reference_case = one_dimensional_cases[0]
                reference_x = np.geomspace(1.0e-7, 1.0e3, 512)
                reference_source = reference_x**2 * np.exp(-0.5 * reference_x**2)
                reference_r, _ = NBessel(reference_x, reference_source).spherical(
                    reference_case["orders"], reference_case["ratios"],
                    bias=reference_case["bias"], biases=reference_case["leg_biases"],
                    mellin_size=32769, y0=1.0e-2,
                )
                check = np.array([70, 110, 150, 190])
                reference_values = simpson_1d(
                    dense_x, dense_x**2 * np.exp(-0.5 * dense_x**2),
                    reference_r[check], reference_case["orders"], reference_case["ratios"],
                )
                mellin_sizes = np.array([2049, 4097, 8193, 16385, 32769])
                errors = []
                mellin_times = []
                for mellin_size in mellin_sizes:
                    def evaluate_mellin():
                        return NBessel(reference_x, reference_source).spherical(
                            reference_case["orders"], reference_case["ratios"],
                            bias=reference_case["bias"], biases=reference_case["leg_biases"],
                            mellin_size=int(mellin_size), y0=1.0e-2,
                        )

                    _, values = evaluate_mellin()
                    errors.append(np.nanmax(np.abs(fractional_difference(values[check], reference_values))))
                    mellin_times.append(median_time(evaluate_mellin, repetitions=2))

            axes[0].set_xscale("log")
            axes[0].set_yscale("log")
            axes[0].set_xlabel("outer grid size $N$")
            axes[0].set_ylabel("wall time [s]")
            axes[0].legend(frameon=False)
            axes[0].grid(alpha=0.2)
            axes[1].loglog(mellin_times, errors, "o-", color="#0072b2")
            for x_time, error, size in zip(mellin_times, errors, mellin_sizes):
                axes[1].annotate(str(size), (x_time, error), xytext=(3, 3), textcoords="offset points")
            axes[1].set_xlabel("wall time [s]")
            axes[1].set_ylabel(r"maximum $|\epsilon_{\mathrm{frac}}|$")
            axes[1].grid(alpha=0.2)
            plt.show()
            """
        ),
    ]


def cosmology_notebook():
    return [
        markdown(
            r"""
            # Cosmological applications of FastNBessel

            Every section below contains the actual `NBessel` or `NBesselND`
            call for the corresponding application in the paper.  The input
            tables use a transparent flat $\Lambda$CDM model with
            $\Omega_m=0.315$, $\Omega_b=0.049$, $h=0.674$, $n_s=0.965$, and
            $\sigma_8=0.811$.  The direct-quadrature calculations are included
            only as independent checks; they are not part of FastNBessel.

            Install the repository with `python -m pip install '.[examples]'`
            before running the notebook from this `examples/` directory.
            """
        ),
        code(
            r"""
            from pathlib import Path

            import matplotlib.pyplot as plt
            import numpy as np

            from nbessel import BinAverage, NBessel, NBesselND, log_zero_pad_nd
            import generate_fiducial_inputs as fiducial
            from quadrature import simpson_1d, simpson_2d_diagonal_interpolated

            DATA = Path("data")
            if not (DATA / "linear_matter_power.txt").exists():
                fiducial.main()
            """
        ),
        markdown(
            r"""
            ## Shared plotting helpers

            The helpers below contain no FastNBessel calculations.  They load a
            tabulated two-dimensional source, choose a diagonal output slice,
            and draw the transform and its fractional difference
            $\epsilon_{\mathrm{frac}}$.
            """
        ),
        code(
            r"""
            def grid_table(name):
                table = np.loadtxt(DATA / name)
                return table[1:, 0], table[0, 1:], table[1:, 1:]


            def indices_in_range(grid, lower, upper, count=128):
                available = np.flatnonzero((grid >= lower) & (grid <= upper))
                if available.size <= count:
                    return available
                return np.unique(
                    np.round(np.linspace(available[0], available[-1], count)).astype(int)
                )


            def comparison_plot(x, fast, reference, xlabel, ylabel, title, transform_scale="auto"):
                with np.errstate(divide="ignore", invalid="ignore"):
                    residual = np.where(reference != 0.0, (fast - reference) / reference, np.nan)
                figure, axes = plt.subplots(
                    2, 1, figsize=(6.5, 5.5), sharex=True, layout="constrained",
                    gridspec_kw={"height_ratios": [2.2, 1.0]},
                )
                axes[0].plot(x, reference, color="black", lw=2, label="quadrature")
                axes[0].plot(x, fast, "--", color="#d55e00", label="FastNBessel")
                axes[0].set_ylabel(ylabel)
                axes[0].set_title(title)
                axes[0].legend(frameon=False)
                if transform_scale == "linear":
                    axes[0].margins(y=0.08)
                elif np.all(np.asarray(reference) > 0.0) and np.all(np.asarray(fast) > 0.0):
                    axes[0].set_yscale("log")
                else:
                    nonzero = np.abs(np.concatenate([reference, fast]))
                    nonzero = nonzero[nonzero > 0.0]
                    axes[0].set_yscale("symlog", linthresh=np.max(nonzero) * 1.0e-4)
                    axes[0].axhline(0.0, color="0.75", lw=0.6)
                axes[1].plot(x, residual, color="#0072b2")
                axes[1].axhline(0.0, color="0.6", lw=0.7)
                axes[1].set_ylabel(r"$\epsilon_{\mathrm{frac}}$")
                axes[1].set_xlabel(xlabel)
                axes[1].set_xscale("log")
                axes[1].grid(alpha=0.2)
                plt.show()
                absolute = np.abs(residual[np.isfinite(residual)])
                return {
                    "median |epsilon_frac|": float(np.median(absolute)),
                    "95th percentile": float(np.quantile(absolute, 0.95)),
                    "fraction below 1e-5": float(np.mean(absolute < 1.0e-5)),
                }
            """
        ),
        markdown("## Fiducial matter power spectrum"),
        code(
            r"""
            power = np.loadtxt(DATA / "linear_matter_power.txt")
            plt.figure(figsize=(6.5, 4))
            plt.loglog(power[:, 0], power[:, 1], label="$z=0$")
            plt.loglog(power[:, 0], power[:, 2], label="$z=0.7$")
            plt.xlabel(r"$k\ [h\,{\rm Mpc}^{-1}]$")
            plt.ylabel(r"$P_{\rm L}(k)\ [(h^{-1}{\rm Mpc})^3]$")
            plt.legend(frameon=False)
            plt.show()
            """
        ),
        markdown(
            r"""
            ## Galaxy three-point correlation function

            The bispectrum monopole is a two-dimensional input.  There is one
            spherical Bessel function for each integration variable, so this is
            the $(N_B,d)=(2,2)$ limit of the general interface.  The stored
            input is undamped.  A distant endpoint window is applied only above
            $20\,h\,{\rm Mpc}^{-1}$, followed by explicit zero padding; changing
            these boundaries is a useful stability test.
            """
        ),
        code(
            r"""
            k1, k2, galaxy_source = grid_table("galaxy_bispectrum_multipole.txt")
            endpoint1 = fiducial.endpoint_window(k1, 1e-5, 1e-4, 20.0, 50.0)
            endpoint2 = fiducial.endpoint_window(k2, 1e-5, 1e-4, 20.0, 50.0)
            galaxy_source = galaxy_source * endpoint1[:, None] * endpoint2[None, :]
            radial_bin = BinAverage(0.95, 1.05, dimension=3)
            (padded_k1, padded_k2), padded_galaxy_source = log_zero_pad_nd(
                [k1, k2], galaxy_source, 384
            )
            (r1, r2), galaxy_grid = NBesselND(
                [padded_k1, padded_k2], padded_galaxy_source
            ).spherical(
                [[0], [0]],
                biases=[1.0, 1.0],
                mellin_size=3,
                bins=[[radial_bin], [radial_bin]],
                y0=[0.2, 0.2],
            )
            galaxy_index = indices_in_range(r1, 20.0, 150.0, 96)
            galaxy_fast = galaxy_grid[galaxy_index, galaxy_index]

            galaxy_quad = simpson_2d_diagonal_interpolated(
                k1, k2, galaxy_source,
                r1[galaxy_index], r2[galaxy_index],
                [[0], [0]], [[1.0], [1.0]],
                bins=[[radial_bin], [radial_bin]], split=[0.2, 0.2],
            )
            comparison_plot(
                r1[galaxy_index], galaxy_fast, galaxy_quad,
                r"$r\ [h^{-1}{\rm Mpc}]$", r"$\overline{\zeta}_{g,00}(r,r)$",
                "Galaxy 3PCF: tree-level bispectrum monopole",
            )
            """
        ),
        markdown(
            r"""
            ## Shear three-point correlation function

            The shear example uses cylindrical Bessel functions.  Negative
            integer orders are accepted and reduced using
            $J_{-n}(x)=(-1)^nJ_n(x)$.  Here each angular separation is averaged
            over a scale-invariant bin with edges $0.95\theta$ and $1.05\theta$.
            """
        ),
        code(
            r"""
            ell1, ell2, shear_source = grid_table("shear_bispectrum_multipole.txt")
            angular_bin = BinAverage(0.95, 1.05, dimension=2)
            (theta1, theta2), shear_grid = NBesselND([ell1, ell2], shear_source).cylindrical(
                [[-3], [-3]],
                biases=[0.0, 0.0],
                mellin_size=3,
                bins=[[angular_bin], [angular_bin]],
                y0=[1.0e-7, 1.0e-7],
            )
            theta_arcmin = theta1 * 180.0 * 60.0 / np.pi
            shear_index = indices_in_range(theta_arcmin, 2.0, 20.0, 96)
            shear_fast = shear_grid[shear_index, shear_index]

            shear_quad = simpson_2d_diagonal_interpolated(
                ell1, ell2, shear_source,
                theta1[shear_index], theta2[shear_index],
                [[-3], [-3]], [[1.0], [1.0]],
                families=["cylindrical", "cylindrical"],
                bins=[[angular_bin], [angular_bin]], split=[100.0, 100.0],
            )
            comparison_plot(
                theta_arcmin[shear_index], shear_fast, shear_quad,
                r"$\theta\ [{\rm arcmin}]$", r"$\Gamma_0^{\times,(0)}(\theta,\theta)$",
                "Shear 3PCF: thin-lens bispectrum moment", transform_scale="linear",
            )
            """
        ),
        markdown(
            r"""
            ## Anisotropic 3PCF covariance: $(N_B,d)=(4,2)$

            Two spherical Bessel functions depend on each wavenumber.  The
            nested `leg_biases` list gives the Mellin-contour split within each
            integration variable.  The plotted transform contains the smooth
            clustering term; constant shot-noise contact terms are not included.
            """
        ),
        code(
            r"""
            k1, k2, covariance_source = grid_table("gaussian_3pcf_covariance_source.txt")
            endpoint1 = fiducial.endpoint_window(k1, 1e-5, 1e-4, 20.0, 50.0)
            endpoint2 = fiducial.endpoint_window(k2, 1e-5, 1e-4, 20.0, 50.0)
            covariance_source = covariance_source * endpoint1[:, None] * endpoint2[None, :]
            (padded_k1, padded_k2), padded_covariance_source = log_zero_pad_nd(
                [k1, k2], covariance_source, 192
            )
            covariance_orders = [[0, 2], [1, 1]]
            covariance_ratios = [[1.0, 0.85], [1.0, 0.9]]
            covariance_bin = BinAverage(0.95, 1.05, dimension=3)
            covariance_bins = [[covariance_bin] * 2, [covariance_bin] * 2]
            (r1, r2), covariance_grid = NBesselND(
                [padded_k1, padded_k2], padded_covariance_source
            ).spherical(
                covariance_orders,
                covariance_ratios,
                biases=[1.0, 1.0],
                leg_biases=[[0.5, 0.5], [0.5, 0.5]],
                mellin_size=65537,
                oversampling=4,
                bins=covariance_bins,
                y0=[0.2, 0.2],
            )
            covariance_index = indices_in_range(r1, 3.0, 150.0, 96)
            covariance_fast = covariance_grid[covariance_index, covariance_index]

            covariance_quad = simpson_2d_diagonal_interpolated(
                k1, k2, covariance_source,
                r1[covariance_index], r2[covariance_index],
                covariance_orders, covariance_ratios,
                bins=covariance_bins, split=[0.2, 0.2],
            )
            comparison_plot(
                r1[covariance_index], covariance_fast, covariance_quad,
                r"$r\ [h^{-1}{\rm Mpc}]$", r"$C_{\zeta\zeta}^{\rm block}(r)$",
                "Gaussian anisotropic-3PCF covariance block",
            )
            """
        ),
        markdown(
            r"""
            ## Three-Bessel kernel for a general $N$-point covariance

            All three Bessel functions share the same integration variable, so
            this application uses the one-dimensional `NBessel` class.
            """
        ),
        code(
            r"""
            power_table = np.loadtxt(DATA / "linear_matter_power.txt")
            k = power_table[:, 0]
            npcf_source = power_table[:, 3] * fiducial.endpoint_window(
                k, 1e-7, 1e-6, 100.0, 1e3
            )
            npcf_ratios = [1.0, 0.8, 1.2]
            r, npcf_all = NBessel(k, npcf_source).spherical(
                [0, 1, 2],
                npcf_ratios,
                bias=1.2,
                biases=[0.4] * 3,
                mellin_size=65537,
                oversampling=4,
                y0=0.2,
            )
            npcf_index = indices_in_range(r, 2.0, 180.0)
            dense_k = np.geomspace(1.0e-7, 1.0e3, 131073)
            npcf_quad = simpson_1d(
                dense_k, fiducial.npcf_transform_source(dense_k), r[npcf_index],
                [0, 1, 2], npcf_ratios,
            )
            comparison_plot(
                r[npcf_index], npcf_all[npcf_index], npcf_quad,
                r"$r\ [h^{-1}{\rm Mpc}]$", r"$f_{012}(r,0.8r,1.2r)$",
                "Example integral in NPCF covariance",
            )
            """
        ),
        markdown(
            r"""
            ## Spherical Fourier--Bessel bispectrum

            The radial source is transformed with three spherical Bessel
            functions whose arguments are in the fixed ratio $1:0.8:1.2$.
            """
        ),
        code(
            r"""
            sfb_table = np.loadtxt(DATA / "sfb_radial_sources.txt")
            radius, sfb_source = sfb_table[:, 0], sfb_table[:, 6]
            sfb_ratios = [1.0, 0.8, 1.2]
            k, sfb_all = NBessel(radius, sfb_source).spherical(
                [0, 1, 2],
                sfb_ratios,
                bias=2.0,
                biases=[2.0 / 3.0] * 3,
                mellin_size=65537,
                oversampling=4,
                y0=2.0e-4,
            )
            sfb_index = indices_in_range(k, 0.002, 0.15)
            dense_radius = np.geomspace(0.1, 1.0e4, 32769)
            sfb_quad = simpson_1d(
                dense_radius, fiducial.sfb_radial_inputs(dense_radius)[5],
                k[sfb_index], [0, 1, 2], sfb_ratios,
            )
            comparison_plot(
                k[sfb_index], sfb_all[sfb_index], sfb_quad,
                r"$k\ [h\,{\rm Mpc}^{-1}]$", r"${\cal J}^{012}(k,0.8k,1.2k)$",
                "Spherical Fourier-Bessel bispectrum kernel",
            )
            """
        ),
        markdown(
            r"""
            ## SFB survey window with a second Bessel derivative

            The density and redshift-space-distortion terms use the same
            two-Bessel kernel.  The latter sets `derivatives=[0, 2]`, so the
            second Bessel function is differentiated twice with respect to its
            argument.
            """
        ),
        code(
            r"""
            radius = sfb_table[:, 0]
            density_source, rsd_source = sfb_table[:, 7], sfb_table[:, 8]
            ratio = 0.8
            transform_options = dict(
                bias=1.0,
                biases=[0.5, 0.5],
                mellin_size=65537,
                oversampling=4,
                y0=2.0e-4,
            )
            k, density_fast = NBessel(radius, density_source).spherical(
                [2, 2], [1.0, ratio], **transform_options
            )
            _, rsd_fast = NBessel(radius, rsd_source).spherical(
                [2, 2], [1.0, ratio], derivatives=[0, 2], **transform_options
            )
            window_index = indices_in_range(k, 0.002, 0.15)
            prefactor = 2.0 * ratio * k[window_index] ** 2 / np.pi
            window_fast = prefactor * (density_fast[window_index] + rsd_fast[window_index])

            dense_radius = np.geomspace(0.1, 1.0e4, 32769)
            dense_sfb = fiducial.sfb_radial_inputs(dense_radius)
            density_quad = simpson_1d(
                dense_radius, dense_sfb[6], k[window_index], [2, 2], [1.0, ratio]
            )
            rsd_quad = simpson_1d(
                dense_radius, dense_sfb[7], k[window_index], [2, 2], [1.0, ratio],
                derivatives=[0, 2],
            )
            window_quad = prefactor * (density_quad + rsd_quad)
            comparison_plot(
                k[window_index], window_fast, window_quad,
                r"$k\ [h\,{\rm Mpc}^{-1}]$", r"${\cal W}_2(k,0.8k)$",
                "SFB survey window with density and RSD terms",
            )
            """
        ),
        markdown(
            r"""
            ## Repeated radial block in a two-loop calculation

            This final example is another one-dimensional three-Bessel
            transform, now with radial arguments in the ratio $1:0.75:1.25$.
            """
        ),
        code(
            r"""
            loop_table = np.loadtxt(DATA / "two_loop_source.txt")
            q, loop_source = loop_table[:, 0], loop_table[:, 2]
            loop_source = loop_source * fiducial.endpoint_window(
                q, 1e-7, 1e-6, 100.0, 1e3
            )
            loop_ratios = [1.0, 0.75, 1.25]
            r, loop_all = NBessel(q, loop_source).spherical(
                [0, 1, 2],
                loop_ratios,
                bias=1.2,
                biases=[0.4] * 3,
                mellin_size=65537,
                oversampling=4,
                y0=0.2,
            )
            loop_index = indices_in_range(r, 2.0, 180.0)
            dense_q = np.geomspace(1.0e-7, 1.0e3, 131073)
            loop_quad = simpson_1d(
                dense_q, fiducial.two_loop_transform_source(dense_q), r[loop_index],
                [0, 1, 2], loop_ratios,
            )
            comparison_plot(
                r[loop_index], loop_all[loop_index], loop_quad,
                r"$r\ [h^{-1}{\rm Mpc}]$", r"${\cal F}_{012}(0.75r,1.25r;r)$",
                "Example integral in a two-loop calculation",
            )
            """
        ),
    ]


def main():
    write_notebook("numerical_validation.ipynb", numerical_notebook())
    write_notebook("cosmological_applications.ipynb", cosmology_notebook())
    print("Wrote numerical_validation.ipynb and cosmological_applications.ipynb")


if __name__ == "__main__":
    main()
