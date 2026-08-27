# Updated with the help of Claude
# Run src
import numpy as np
from numba import njit, prange


@njit(cache=True)
def gaussian(x, amp, cen, sigma, shift):
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
def broaden_spectrum(dispersion, fluxden, resolution, fwhm_lim=5.0):
    n = dispersion.shape[0]
    broadened_fluxden = np.zeros(n)

    for idx in prange(n):
        broadened_fluxden[idx] = _resolution_convolution(
            idx, dispersion, fluxden, resolution[idx], fwhm_lim
        )

    return broadened_fluxden
