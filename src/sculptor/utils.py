
import numpy as np
from numba import jit, prange


@jit
def gaussian(x, amp, cen, sigma, shift):
    """ 1-D Gaussian function"""
    central = cen + shift

    return (amp / (np.sqrt(2*np.pi) * sigma)) * np.exp(-(x-central)**2 /
                                                       (2*sigma**2))

@jit
def _resolution_convolution(idx, dispersion, fluxden, resolution, fwhm_lim):

    fwhm = dispersion[idx] / resolution  # fwhm in pixel units
    sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))  # sigma in pixel units

    # Mask wavelength range within 5 fwhm
    index_mask = (dispersion > dispersion[idx] - fwhm_lim * fwhm) & \
                 (dispersion < dispersion[idx] + fwhm_lim * fwhm)

    flux_to_convolve = fluxden[index_mask]

    profile = gaussian(dispersion[index_mask], 1.0,
                       dispersion[idx], sigma, 0)

    flux_sum = np.sum(profile * flux_to_convolve)

    profile_sum = np.sum(profile)


    return flux_sum / profile_sum


@jit
def broaden_spectrum(dispersion, fluxden, resolution, fwhm_lim=5):

    broadened_fluxden = np.zeros(len(dispersion))

    if isinstance(resolution, (int, float)):
        resolution = np.ones(len(dispersion)) * resolution

    for idx in prange(len(dispersion)):
        broadened_fluxden[idx] = _resolution_convolution(
            idx, dispersion, fluxden, resolution[idx], fwhm_lim)

    return broadened_fluxden
