"""Generate all text inputs used by the cosmological application examples.

The model is a deliberately transparent flat-Lambda-CDM setup.  It uses the
Eisenstein-Hu no-wiggle transfer fit, normalizes the spectrum to
sigma8, and evaluates tree-level density/galaxy bispectra.  The files are
examples of the inputs expected by FastNBessel, not precision predictions for
a survey.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.integrate import cumulative_trapezoid, simpson


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"

# Fiducial flat-Lambda-CDM parameters (Planck-like).
OMEGA_M = 0.315
OMEGA_B = 0.049
H = 0.674
N_S = 0.965
SIGMA8 = 0.811
C_OVER_100 = 2997.92458  # Mpc/h

# Illustrative constant-density galaxy sample used only for the Gaussian
# anisotropic-3PCF covariance block.
GALAXY_LINEAR_BIAS = 2.0
GALAXY_NUMBER_DENSITY = 3.0e-4  # (h/Mpc)^3
SURVEY_VOLUME = 5.0e9  # (Mpc/h)^3 = 5 (Gpc/h)^3


def expansion(z):
    return np.sqrt(OMEGA_M * (1.0 + z) ** 3 + 1.0 - OMEGA_M)


def omega_m_z(z):
    return OMEGA_M * (1.0 + z) ** 3 / expansion(z) ** 2


def growth(z):
    om = omega_m_z(z)
    ol = 1.0 - om
    g = 2.5 * om / (om ** (4.0 / 7.0) - ol + (1.0 + om / 2.0) * (1.0 + ol / 70.0))
    om0 = OMEGA_M
    ol0 = 1.0 - om0
    g0 = 2.5 * om0 / (om0 ** (4.0 / 7.0) - ol0 + (1.0 + om0 / 2.0) * (1.0 + ol0 / 70.0))
    return g / ((1.0 + z) * g0)


def transfer_no_wiggle(k):
    """Eisenstein-Hu no-wiggle transfer, with k in h/Mpc."""

    omega_m = OMEGA_M * H**2
    k_eq = 7.46e-2 * omega_m  # 1/Mpc, T_CMB=2.7255 K
    q = (k * H) / (13.41 * k_eq)
    log_term = np.log(np.e + 1.8 * q)
    coefficient = 14.2 + 731.0 / (1.0 + 62.5 * q)
    return log_term / (log_term + coefficient * q**2)


def _top_hat(x):
    small = np.abs(x) < 1e-3
    result = np.empty_like(np.asarray(x, dtype=float))
    result[small] = 1.0 - x[small] ** 2 / 10.0 + x[small] ** 4 / 280.0
    result[~small] = (
        3.0 * (np.sin(x[~small]) - x[~small] * np.cos(x[~small])) / x[~small] ** 3
    )
    return result


def _power_amplitude():
    k = np.geomspace(1e-6, 1e3, 100001)
    shape = k**N_S * transfer_no_wiggle(k) ** 2
    variance = simpson(k**3 * shape * _top_hat(8.0 * k) ** 2, x=np.log(k)) / (
        2.0 * np.pi**2
    )
    return SIGMA8**2 / variance


POWER_AMPLITUDE = _power_amplitude()


def linear_power(k, z=0.0):
    return POWER_AMPLITUDE * k**N_S * transfer_no_wiggle(k) ** 2 * growth(z) ** 2


def f2_kernel(k1, k2, mu):
    return 5.0 / 7.0 + 0.5 * mu * (k1 / k2 + k2 / k1) + 2.0 * mu**2 / 7.0


def matter_bispectrum(k1, k2, mu, z):
    k3 = np.sqrt(np.maximum(k1**2 + k2**2 + 2.0 * k1 * k2 * mu, 1e-30))
    p1, p2, p3 = linear_power(k1, z), linear_power(k2, z), linear_power(k3, z)
    mu23 = -(k2 + k1 * mu) / k3
    mu31 = -(k1 + k2 * mu) / k3
    return 2.0 * (
        f2_kernel(k1, k2, mu) * p1 * p2
        + f2_kernel(k2, k3, mu23) * p2 * p3
        + f2_kernel(k3, k1, mu31) * p3 * p1
    )


def galaxy_bispectrum(k1, k2, mu, z, b1=GALAXY_LINEAR_BIAS, b2=0.3):
    k3 = np.sqrt(np.maximum(k1**2 + k2**2 + 2.0 * k1 * k2 * mu, 1e-30))
    p1, p2, p3 = linear_power(k1, z), linear_power(k2, z), linear_power(k3, z)
    mu23 = -(k2 + k1 * mu) / k3
    mu31 = -(k1 + k2 * mu) / k3

    def z2(a, b, angle):
        return b1 * f2_kernel(a, b, angle) + b2 / 2.0

    return (
        2.0
        * b1**2
        * (
            z2(k1, k2, mu) * p1 * p2
            + z2(k2, k3, mu23) * p2 * p3
            + z2(k3, k1, mu31) * p3 * p1
        )
    )


def angle_moment(function, k1, k2, z, mode=0, nodes=48):
    """Gauss--Legendre angular moment without forming a large 3D array."""

    mu, weight = np.polynomial.legendre.leggauss(nodes)
    k1, k2 = np.broadcast_arrays(
        np.asarray(k1, dtype=float), np.asarray(k2, dtype=float)
    )
    result = np.zeros_like(k1)
    for node, node_weight in zip(mu, weight, strict=True):
        angular_weight = np.cos(mode * np.arccos(node)) if mode else 1.0
        result += 0.5 * node_weight * angular_weight * function(k1, k2, node, z)
    return result


def galaxy_bispectrum_inputs(k, z=0.7):
    """Return the bispectrum monopole and the undamped 2D transform source."""

    k1, k2 = np.meshgrid(np.asarray(k), np.asarray(k), indexing="ij")
    monopole = angle_moment(galaxy_bispectrum, k1, k2, z)
    source = (k1 * k2) ** 3 * monopole / (2.0 * np.pi**2) ** 2
    return monopole, source


def shear_bispectrum_inputs(ell, z=0.7):
    """Return a unit-normalized thin-lens matter-bispectrum monopole."""

    ell1, ell2 = np.meshgrid(np.asarray(ell), np.asarray(ell), indexing="ij")
    z_grid = np.linspace(0.0, z, 2001)
    chi = C_OVER_100 * np.trapz(1.0 / expansion(z_grid), z_grid)
    monopole = angle_moment(matter_bispectrum, ell1 / chi, ell2 / chi, z) / chi**4
    source = ell1**2 * ell2**2 * monopole / (2.0 * np.pi) ** 3
    return monopole, source


def gaussian_covariance_inputs(k, z=0.7, *, include_shot_noise=False):
    """Return one angle-averaged three-power-spectrum covariance block.

    This is the identity cross-triangle Wick pairing used to validate the
    four-Bessel transform, not the sum over all six pairings in the complete
    Gaussian 3PCF covariance.
    """

    k1, k2 = np.meshgrid(np.asarray(k), np.asarray(k), indexing="ij")
    mu, weight = np.polynomial.legendre.leggauss(48)
    shot_noise = 1.0 / GALAXY_NUMBER_DENSITY if include_shot_noise else 0.0
    p1 = GALAXY_LINEAR_BIAS**2 * linear_power(k1, z) + shot_noise
    p2 = GALAXY_LINEAR_BIAS**2 * linear_power(k2, z) + shot_noise
    ppp = np.zeros_like(k1)
    for node, node_weight in zip(mu, weight, strict=True):
        k3 = np.sqrt(np.maximum(k1**2 + k2**2 + 2.0 * k1 * k2 * node, 1e-30))
        p3 = GALAXY_LINEAR_BIAS**2 * linear_power(k3, z) + shot_noise
        ppp += 0.5 * node_weight * p1 * p2 * p3
    source = (k1 * k2) ** 3 * ppp / (SURVEY_VOLUME * (2.0 * np.pi**2) ** 2)
    return ppp, source


def npcf_transform_source(k, z=0.7):
    k = np.asarray(k)
    return k**3 * linear_power(k, z) / (2.0 * np.pi**2)


def two_loop_transform_source(k, z=0.7):
    k = np.asarray(k)
    return k**3 * linear_power(k, z) / (2.0 * np.pi**2)


def endpoint_window(x, low_stop, low_full, high_full, high_stop):
    """Raised-cosine endpoint window on a logarithmic grid."""

    x = np.asarray(x, dtype=float)
    logx = np.log(x)
    result = np.ones_like(x)
    low = x < low_full
    phase = np.clip(
        (logx[low] - np.log(low_stop)) / np.log(low_full / low_stop), 0.0, 1.0
    )
    result[low] = 0.5 - 0.5 * np.cos(np.pi * phase)
    high = x > high_full
    phase = np.clip(
        (np.log(high_stop) - logx[high]) / np.log(high_stop / high_full), 0.0, 1.0
    )
    result[high] = 0.5 - 0.5 * np.cos(np.pi * phase)
    result[(x <= low_stop) | (x >= high_stop)] = 0.0
    return result


@lru_cache(maxsize=1)
def _distance_table():
    z = np.linspace(0.0, 6.0, 40001)
    chi = C_OVER_100 * np.concatenate(
        ([0.0], cumulative_trapezoid(1.0 / expansion(z), z))
    )
    return z, chi


def _radial_selection(radius):
    z_background, chi_background = _distance_table()
    z = np.interp(radius, chi_background, z_background)
    return z, np.exp(-0.5 * ((z - 0.7) / 0.18) ** 2)


@lru_cache(maxsize=1)
def sfb_window_normalization():
    radius = np.geomspace(1.0e-4, 2.0e4, 65537)
    _, window = _radial_selection(radius)
    return float(simpson(window * radius**2, x=radius))


def sfb_radial_inputs(radius):
    """Return the background columns and three SFB transform sources."""

    radius = np.asarray(radius)
    z_radius, window = _radial_selection(radius)
    window = window / sfb_window_normalization()
    d = growth(z_radius)
    growth_rate = omega_m_z(z_radius) ** 0.55
    bias = 1.5 + 0.5 * z_radius
    z_template = bias * (2.0 / 7.0) + 0.15
    bispectrum = radius**3 * window * d**2 * z_template
    density = radius**3 * window * d * bias
    rsd = -(radius**3) * window * d * growth_rate
    return z_radius, window, d, bias, growth_rate, bispectrum, density, rsd


def save_table(path: Path, columns, names: str, description: str):
    header = description + "\ncolumns: " + names
    np.savetxt(path, np.column_stack(columns), fmt="%.17e", header=header)


def save_square_grid(path: Path, grid, values, description: str):
    """Save a square 2D input with its two identical axes in one text file."""

    grid = np.asarray(grid)
    values = np.asarray(values)
    if values.shape != (grid.size, grid.size):
        raise ValueError("values must be square and match grid")
    table = np.empty((grid.size + 1, grid.size + 1))
    table[0, 0] = np.nan
    table[0, 1:] = grid
    table[1:, 0] = grid
    table[1:, 1:] = values
    np.savetxt(
        path,
        table,
        fmt="%.12e",
        header=description
        + "\nfirst row: x2 grid; first column: x1 grid; body: transform source",
    )


def main():
    DATA.mkdir(parents=True, exist_ok=True)
    z_eff = 0.7

    k = np.geomspace(1e-7, 1e3, 2048)
    pk0 = linear_power(k, 0.0)
    pk = linear_power(k, z_eff)
    npcf_source = npcf_transform_source(k, z_eff)
    save_table(
        DATA / "linear_matter_power.txt",
        (k, pk0, pk, npcf_source),
        "k_h_Mpc P_lin_z0_(Mpc/h)^3 P_lin_z0p7_(Mpc/h)^3 k3_P_z0p7/(2pi2)",
        f"Flat LCDM: Omega_m={OMEGA_M}, Omega_b={OMEGA_B}, h={H}, ns={N_S}, sigma8={SIGMA8}.",
    )

    # The stored sources contain the undamped physical model.  The reproduction
    # script applies only distant endpoint windows and then adds explicit zeros.
    k2d = 1.0e-5 * np.exp(0.01 * np.arange(1544))
    _, galaxy_source = galaxy_bispectrum_inputs(k2d, z_eff)
    save_square_grid(
        DATA / "galaxy_bispectrum_multipole.txt",
        k2d,
        galaxy_source,
        "Tree-level real-space galaxy bispectrum monopole at z=0.7; b1=2.0, b2=0.3.",
    )

    # A thin-lens shear-transform example.  H_0 is the projected matter-
    # bispectrum monopole at chi(z=0.7), with the overall lensing-efficiency
    # prefactor set to unity.
    ell = np.geomspace(0.5, 1.0e6, 513)
    _, shear_source = shear_bispectrum_inputs(ell, z_eff)
    save_square_grid(
        DATA / "shear_bispectrum_multipole.txt",
        ell,
        shear_source,
        "Unit-normalized thin-lens projected matter-bispectrum monopole at z=0.7.",
    )

    # One representative Gaussian 3PCF covariance block, angle-averaged over
    # the third side of the Fourier triangle.  The complete covariance has six
    # cross-triangle Wick pairings; the transform test deliberately isolates
    # the identity pairing.
    k_cov = 1.0e-5 * np.exp(0.02 * np.arange(773))
    _, covariance_source = gaussian_covariance_inputs(
        k_cov, z_eff, include_shot_noise=False
    )
    save_square_grid(
        DATA / "gaussian_3pcf_covariance_source.txt",
        k_cov,
        covariance_source,
        (
            "Clustering-only Gaussian 3PCF covariance block at z=0.7; "
            f"b1={GALAXY_LINEAR_BIAS}, Vs={SURVEY_VOLUME:.6e} (Mpc/h)^3."
        ),
    )

    # SFB inputs are sampled on a logarithmic distance grid.  z(chi) is found
    # by monotonic interpolation of the background distance relation.
    radius = np.geomspace(0.1, 1.0e4, 8193)
    (
        z_radius,
        window,
        d,
        bias,
        growth_rate,
        sfb_bispectrum_source,
        sfb_bias_source,
        sfb_rsd_source,
    ) = sfb_radial_inputs(radius)
    save_table(
        DATA / "sfb_radial_sources.txt",
        (
            radius,
            z_radius,
            window,
            d,
            bias,
            growth_rate,
            sfb_bispectrum_source,
            sfb_bias_source,
            sfb_rsd_source,
        ),
        "r_Mpc_h z W D b f source_bispectrum source_window_bias source_window_rsd",
        "Gaussian radial selection centered at z=0.7 for the SFB examples.",
    )

    loop_source = two_loop_transform_source(k, z_eff)
    save_table(
        DATA / "two_loop_source.txt",
        (k, pk, loop_source),
        "q_h_Mpc P_lin_z0p7_(Mpc/h)^3 q3_P/(2pi2)",
        "Representative undamped source for the repeated three-Bessel two-loop radial transform.",
    )

    print("Wrote fiducial input tables to examples/data")


if __name__ == "__main__":
    main()
