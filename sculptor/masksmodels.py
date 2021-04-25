#!/usr/bin/env python

import os
import glob
import importlib
import numpy as np
import pkg_resources
from astropy import constants as const

from lmfit import Model, Parameters

c_km_s = const.c.to('km/s').value

# ------------------------------------------------------------------------------
# Model functions
# ------------------------------------------------------------------------------

def constant(x, amp):
    """ Constant model

    :param x: Dispersion
    :type x: np.ndarray
    :param amp: Amplitude of the constant model
    :type amp: float
    :return: Constant model
    :rtype: np.ndarray
    """

    return amp + 0 * x


def power_law(x, amp, slope):
    """

    :param x: Dispersion
    :type x: np.ndarray
    :param amp: Amplitude of the power law
    :type amp: float
    :param slope: Slope of the power law
    :type slope: float
    :return: Power law model
    :rtype: np.ndarray
    """

    return amp*(x**slope)


def gaussian(x, amp, cen, sigma, shift):
    """ Basic Gaussian line model

    The Gaussian is not normalized.

    :param x: Dispersion
    :type x: np.ndarray
    :param amp: Amplitude of the Gaussian
    :type amp: float
    :param cen: Central dispersion of the Gaussian
    :type cen: float
    :param sigma: Width of the Gaussian in sigma
    :type sigma: float
    :param shift: Shift of the Gaussian in dispersion units
    :type shift: float
    :return: Gaussian line model
    :rtype: np.ndarray
    """

    central = cen + shift

    return amp * np.exp(-(x-central)**2 / (2*sigma**2))


def lorentzian(x, amp, cen, gamma, shift):
    """ Basic Lorentzian line model

    :param x: Dispersion
    :type x: np.ndarray
    :param amp: Amplitude of the Lorentzian
    :type amp: float
    :param cen: Central dispersion of the Lorentzian
    :type cen: float
    :param gamma: Lorentzian Gamma parameter
    :param shift: Shift of the Lorentzian in dispersion units
    :type shift: float
    :return: Gaussian line model
    :rtype: np.ndarray
    """

    central = cen + shift

    return amp * 1 / (np.pi * gamma * (1 + ((x-central) / gamma)**2))


def gausshermite4(x, a, b, c, d):
    """ Calculate 4th order Gauss-Hermite function

    :param x:
    :param a:
    :param b:
    :param c:
    :param d:
    :return:
    """

    return np.polynomial.hermite.Hermite(coef=[a, b, c, d]).__call__(x)


# ------------------------------------------------------------------------------
# Model functions
# ------------------------------------------------------------------------------


def setup_power_law(prefix,  **kwargs):

    amplitude = kwargs.pop('amplitude', 2)

    params = Parameters()

    params.add(prefix + 'amp', value=amplitude)
    params.add(prefix + 'slope', value=-1.5, min=-2.5, max=2.5)

    model = Model(power_law, prefix=prefix)

    return model, params


def setup_gaussian(prefix, **kwargs):

    amplitude = kwargs.pop('amplitude', 5)

    params = Parameters()

    params.add(prefix + 'amp', value=amplitude)
    params.add(prefix + 'cen')
    params.add(prefix + 'sigma')
    params.add(prefix + 'shift', value=0, vary=False)

    model = Model(gaussian, prefix=prefix)

    return model, params


def setup_constant(prefix, **kwargs):

    amplitude = kwargs.pop('amplitude', 5)

    params = Parameters()

    params.add(prefix + 'amp', value=amplitude)

    model = Model(constant, prefix=prefix)

    return model, params


def setup_lorentzian(prefix, **kwargs):

    amplitude = kwargs.pop('amplitude', 5)

    params = Parameters()

    params.add(prefix + 'amp', value=amplitude)
    params.add(prefix + 'cen')
    params.add(prefix + 'gamma')
    params.add(prefix + 'shift', value=0, vary=False)

    model = Model(lorentzian, prefix=prefix)

    return model, params


""" list of str: List of model names"""
model_func_list = ['Constant (amp)',
                   'Power Law (amp, slope)',
                   'Gaussian (amp, cen, sigma, shift)',
                   'Lorentzian (amp, cen, gamma, shift)'
                   ]
""" dict: Dictionary of model functions"""
model_func_dict = {'constant': constant,
                   'power_law': power_law,
                   'gaussian': gaussian,
                   'lorentzian': lorentzian
                   }
""" dict: Dictionary of model setup function names"""
model_setup_list = [setup_constant,
                    setup_power_law,
                    setup_gaussian,
                    setup_lorentzian
                    ]
""" dict: Dictionary of mask presets"""
mask_presets = {}


extension_path = pkg_resources.resource_filename('sculptor',
                                                 '../sculptor_extensions')

file_names = glob.glob(extension_path+'/*.py')

module_names = [os.path.basename(f)[:-3] for f in file_names if
                os.path.isfile(f) and not f.endswith('__init__.py')]

for module_name in module_names:
    print('[INFO] Import "sculptor_extensions" package: {}'.format(
        module_name))
    module = importlib.import_module('sculptor_extensions.{}'.format(
        module_name))

    if hasattr(module, 'model_func_list') and \
            hasattr(module, 'model_func_dict') and \
            hasattr(module, 'model_setup_list'):

        if len(module.model_func_list) == len(module.model_setup_list):
            model_func_list.extend(module.model_func_list)
            model_func_dict.update(module.model_func_dict)
            model_setup_list.extend(module.model_setup_list)

    if hasattr(module, 'mask_presets'):
        mask_presets.update(module.mask_presets)