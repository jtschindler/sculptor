"""Shared constants (speed of light, package data directory) and the
numba-compiled resolution broadening used by `FitModel.eval`."""

import importlib.resources

import numpy as np
from astropy import constants as const
from astropy import units as u
from numba import njit, prange

# Speed of light in km/s and in Angstroem/s
c_km_s = const.c.to("km/s").value
c_AngstroemPerSecond = const.c.to(u.AA / u.s).value

# Percentiles of the 1-sigma interval of a normal distribution
SIGMA_PERCENTILES = (15.87, 84.13)

# Package data directory (iron templates, HOL data), ending in "/"
datadir = str(importlib.resources.files("sculptor") / "data") + "/"


@njit(cache=True)
def gaussian(x, amp, cen, sigma, shift):
    """Normalized Gaussian (numba-compiled).

    :param x: Evaluation points
    :param amp: Integral of the Gaussian
    :param cen: Center
    :param sigma: Standard deviation
    :param shift: Offset added to the center
    :return: amp / (sqrt(2 pi) sigma) * exp(-(x - cen - shift)^2 / (2 sigma^2))
    """

    central = cen + shift
    return (amp / (np.sqrt(2 * np.pi) * sigma)) * np.exp(
        -((x - central) ** 2) / (2 * sigma**2)
    )


@njit(cache=True)
def _resolution_convolution(idx, dispersion, fluxden, resolution, fwhm_lim):
    fwhm = dispersion[idx] / resolution
    sigma = fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))

    lo = dispersion[idx] - fwhm_lim * fwhm
    hi = dispersion[idx] + fwhm_lim * fwhm
    lo_i = np.searchsorted(dispersion, lo, side="right")
    hi_i = np.searchsorted(dispersion, hi, side="left")

    flux_sum = 0.0
    profile_sum = 0.0
    for j in range(lo_i, hi_i):
        p = gaussian(dispersion[j], 1.0, dispersion[idx], sigma, 0.0)
        flux_sum += p * fluxden[j]
        profile_sum += p

    return flux_sum / profile_sum


@njit(cache=True, parallel=True)
def _broaden_spectrum_jit(dispersion, fluxden, resolution, fwhm_lim=5.0):
    n = dispersion.shape[0]
    broadened_fluxden = np.zeros(n)

    for idx in prange(n):
        broadened_fluxden[idx] = _resolution_convolution(
            idx, dispersion, fluxden, resolution[idx], fwhm_lim
        )

    return broadened_fluxden


def broaden_spectrum(dispersion, fluxden, resolution, fwhm_lim=5.0):
    """Convolve `fluxden` with a per-pixel resolution-dependent Gaussian.

    Data loaded from FITS files (e.g. via astropy) is commonly big-endian,
    but numba's nopython mode has no typing overload for non-native byte
    order, so the jitted kernel below would raise a TypingError on such
    input. Casting to native-order, contiguous float64 here makes this
    function work regardless of the caller's array byte order/layout.
    """

    dispersion = np.ascontiguousarray(dispersion, dtype=np.float64)
    fluxden = np.ascontiguousarray(fluxden, dtype=np.float64)
    resolution = np.ascontiguousarray(resolution, dtype=np.float64)

    return _broaden_spectrum_jit(dispersion, fluxden, resolution, fwhm_lim)
