#!/usr/bin/env python

"""An example Sculptor Extension

This module defines models, masks and analysis routines as an example to
create your own Sculptor extension.

At first we define the basic model functions and their setups. The setup
functions initialize LMFIT models and parameters using the basic model
functions defined here.

When youy define a mask you need to specify the name under which it will
appear in the Sculptor GUI, the rest_frame keyword and the mask ranges to
that will be included in the SpecModel masking or excluded in the SpecFit
masking. With rest_frame=True the mask regions will automatically be adjusted
for the object redshift specified in the SpecFit class. With rest-frame=False
the mask will not be redshifted.

"""

import numpy as np
from lmfit import Model, Parameters
from astropy import constants as const

from sculptor import speconed as sod

# ------------------------------------------------------------------------------
# Model functions
# ------------------------------------------------------------------------------

def my_model(x, z, amp, cen, fwhm_km_s, shift_km_s):
    """ Gaussian line model as an example for a model

    The central wavelength of the Gaussian line model is determined by the
    central wavelength cen, the redshift, z, and the velocity shift
    shift_km_s (in km/s). These parameters are degenerate in a line fit and
    it is adviseable to fix two of them (to predetermined values e.g., the
    redshift or the central wavelength).

    The width of the line is set by the FWHM in km/s.

    The Gaussian is not normalized.

    :param x: Dispersion of the continuum model
    :type x: np.ndarray
    :param z: Redshift
    :type z: float
    :param amp: Amplitude of the Gaussian
    :type amp: float
    :param cen: Central wavelength
    :type cen: float
    :param fwhm_km_s: FWHM of the Gaussian in km/s
    :type fwhm_km_s: float
    :param shift_km_s: Doppler velocity shift of the central wavelength
    :type shift_km_s: float
    :return: Gaussian line model
    :rtype: np.ndarray

    """

    c_km_s = const.c.to('km/s').value

    cen = cen * (1+z)

    delta_cen = shift_km_s / c_km_s * cen
    central = cen + delta_cen
    fwhm = fwhm_km_s / c_km_s * central
    sigma = fwhm / np.sqrt(8*np.log(2))

    return amp * np.exp(-(x-central)**2 / (2*sigma**2))

# ------------------------------------------------------------------------------
# Model setups
# ------------------------------------------------------------------------------


def setup_my_model(prefix, **kwargs):
    """Example of a model setup function for the Gaussian emission line model.

    The 'prefix' argument needs to be included. You can use a variety of
    keyword arguments as you can see below.

    :param prefix: Model prefix
    :type prefix: string
    :param kwargs: Keyword arguments
    :return: LMFIT model and parameters
    :rtype: (lmfit.Model, lmfit.Parameters)

    """

    params = Parameters()

    redshift = kwargs.pop('redshift', 0)
    amplitude = kwargs.pop('amplitude', 100)
    cenwave = kwargs.pop('cenwave', 1450)
    fwhm = kwargs.pop('fwhm', 2500)
    vshift = kwargs.pop('vshift', 0)

    z_min = min(redshift * 1.05, 1080)
    z_max = max(1.0, redshift * 0.95)

    params.add(prefix + 'z', value=redshift, min=z_min, max=z_max, vary=False)

    params.add(prefix + 'amp', value=amplitude)
    params.add(prefix + 'cen', value=cenwave)
    params.add(prefix + 'fwhm_km_s', value=fwhm)
    params.add(prefix + 'shift_km_s', value=vshift, vary=False)

    model = Model(my_model, prefix=prefix)

    return model, params


# ------------------------------------------------------------------------------
# Masks
# ------------------------------------------------------------------------------

""" When youy define a mask you need to specify the name under which it will 
appear in the Sculptor GUI, the rest_frame keyword and the mask ranges to 
that will be included in the SpecModel masking or excluded in the SpecFit 
masking. With rest_frame=True the mask regions will automatically be adjusted 
for the object redshift specified in the SpecFit class. With rest-frame=False 
the mask will not be redshifted."""

my_mask = {'name': 'My mask',
                 'rest_frame': True,
                 'mask_ranges': [[1265, 1290], [1340, 1375], [1425, 1470],
                                 [1680, 1705], [1905, 2050]]}


# ------------------------------------------------------------------------------
# Model and mask lists and dicts
# The variables need to always be included for sculptor to find and add the
# user-defined models and masks to the GUI
# ------------------------------------------------------------------------------

""" dict: Dictionary of model functions"""
model_func_dict = {'my_model': my_model}

""" list of str: List of model names"""
model_func_list = ['My Model']

""" dict: Dictionary of model setup function names"""
model_setup_list = [setup_my_model]
""" dict: Dictionary of mask presets"""
mask_presets = {'My mask': my_mask,
                }
