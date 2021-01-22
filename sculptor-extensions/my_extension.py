#!/usr/bin/env python

import os
import numpy as np
import scipy as sp

from astropy import constants as const
from astropy.modeling.blackbody import blackbody_lambda

from sculptor import speconed as sod

from lmfit import Model, Parameters

c_km_s = const.c.to('km/s').value


def add_redshift_param(redshift, params, prefix):

    z_min = min(redshift*1.05, 1080)
    z_max = max(1.0, redshift * 0.95)

    params.add(prefix+'z', value=redshift, min=z_min, max=z_max, vary=False)


# ------------------------------------------------------------------------------
# Model functions
# ------------------------------------------------------------------------------

def power_law_at_2500A(x, amp, slope, z):
    """ Power law anchored at 2500 Angstroem

    Parameters:
    -----------
    :param x: ndarray
        x-Axis of the power law
    :param amp: float
        Amplitude of the power law anchored at 2500 (Angstroem)
    :param slope: float
        Slope of the power law

    Returns:
    --------
    :return: ndarray
    """

    return amp * (x / (2500. * (z+1.)))**slope


def gaussian_fwhm_km_s_z(x, z, amp, cen, fwhm_km_s, shift_km_s):
    """ Calculate 1-D Gaussian using fwhm for in km/s instead sigma

    :param x: ndarray
        x-Axis of the Gaussian
    :param amp: float
        Amplitude of the Gaussian
    :param cen: float
        Central x-value of the Gaussian
    :param fwhm_km_s: float
        Full Width at Half Maximum (FWHM) of the Gaussian in km/s
    :param shift_km_s: float
        x-Axis shift of the Gaussian in km/s
    :return: ndarray
    """

    cen = cen * (1+z)

    delta_cen = shift_km_s / c_km_s * cen
    central = cen + delta_cen
    fwhm = fwhm_km_s / c_km_s * central
    sigma = fwhm / np.sqrt(8*np.log(2))

    return (amp / (np.sqrt(2*np.pi) * sigma)) * np.exp(-(x-central)**2 / (2*sigma**2))


# ------------------------------------------------------------------------------
# Model setups
# ------------------------------------------------------------------------------


def setup_power_law_at_2500A(prefix, **kwargs):
    """ Set up the power law model anchored at 2500 Angstroem

    :param prefix: string
        Model prefix
    :param kwargs:
    :return:
    """

    redshift = kwargs.pop('redshift', 0)
    amplitude = kwargs.pop('amplitude', 1e-17)

    params = Parameters()

    add_redshift_param(redshift, params, prefix)

    params.add(prefix + 'amp', value=amplitude)
    params.add(prefix + 'slope', value=-1.5, min=-2.5, max=-0.3)

    model = Model(power_law_at_2500A, prefix=prefix)

    return model, params


def setup_gaussian_fwhm_km_s_z(prefix, **kwargs):

    params = Parameters()

    redshift = kwargs.pop('redshift', 0)
    amplitude = kwargs.pop('amplitude', 1e-14)
    cenwave = kwargs.pop('cenwave', 1215)
    fwhm = kwargs.pop('fwhm', 2500)
    vshift = kwargs.pop('vshift', 0)


    add_redshift_param(redshift, params, prefix)

    params.add(prefix + 'amp', value=amplitude)
    params.add(prefix + 'cen', value=cenwave, min=0, max=25000)
    params.add(prefix + 'fwhm_km_s', value=fwhm, min=0, max=2E+4)
    params.add(prefix + 'shift_km_s', value=vshift, vary=False)

    model = Model(gaussian_fwhm_km_s_z, prefix=prefix)

    return model, params

# ------------------------------------------------------------------------------
# Masks
# ------------------------------------------------------------------------------


qso_cont_VP06 = {'name': 'QSO Cont. VP06' ,
                 'rest_frame': True,
                 'mask_ranges': [[1265, 1290], [1340, 1375], [1425, 1470],
                                 [1680, 1705], [1905, 2050]]}


# ------------------------------------------------------------------------------
# Model and mask lists and dicts
# The variables need to always be included for sculptor to find and add the
# user-defined models and masks to the GUI
# ------------------------------------------------------------------------------


""" list of str: List of model names"""
model_func_list = ['Power Law @2500A (amp, slope, z)',
                   'Gaussian (amp, cen, FWHM, vshift, z)'
                   ]
""" dict: Dictionary of model functions"""
model_func_dict = {'power_law_at_2500A': power_law_at_2500A,
                   'gaussian_fwhm_km_s_z': gaussian_fwhm_km_s_z,
                   }
""" dict: Dictionary of model setup function names"""
model_setup_list = [setup_power_law_at_2500A,
                    setup_gaussian_fwhm_km_s_z,
                    ]
""" dict: Dictionary of mask presets"""
mask_presets = {'QSO Cont. VP06': qso_cont_VP06,
                }
