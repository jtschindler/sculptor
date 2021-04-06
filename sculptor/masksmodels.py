#!/usr/bin/env python

import os
import glob
import importlib
import numpy as np
import pkg_resources
from astropy import constants as const

from lmfit import Model, Parameters

c_km_s = const.c.to('km/s').value


def constant(x, amp):
    """ Return constant

    :param x:
    :param amp:
    :return:
    """

    return amp + 0 * x


def power_law(x, amp, slope):
    """ Calculate power law

    Parameters:
    -----------
    :param x: ndarray
        x-Axis of the power law
    :param amp: float
        Amplitude of the power law
    :param slope: float
        Slope of the power law

    Returns:
    --------
    :return: ndarray
    """

    return amp*(x**slope)


def gaussian(x, amp, cen, sigma, shift):
    """ Calculate 1-D Gaussian

    :param x: ndarray
         x-Axis of the Gaussian
    :param amp: float
        Amplitude of the Gaussian
    :param cen: float
        Central x-value of the Gaussian
    :param sigma: float
        Standard deviation of the Gaussian
    :param shift: float
        x-Axis shift of the Gaussian
    :return: ndarray

    """

    central = cen + shift
    # return (amp / (np.sqrt(2*np.pi) * sigma)) * np.exp(-(x-central)**2 / (2*sigma**2))
    return amp * np.exp(-(x-central)**2 / (2*sigma**2))


def lorentzian(x, amp, cen, gamma, shift):
    """ Calculate 1-D Lorentzian (normalised)

    :param x:
    :param amp:
    :param cen:
    :param gamma:
    :param shift:
    :return:
    """


    central = cen + shift

    return amp * 1 / (np.pi * gamma * (1 + ((x-central) / (gamma))**2))


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


def setup_power_law(prefix, amplitude=2, **kwargs):

    params = Parameters()

    params.add(prefix + 'amp', value=amplitude)
    params.add(prefix + 'slope', value=-1.5, min=-2.5, max=2.5)

    model = Model(power_law, prefix=prefix)

    return model, params


def setup_gaussian(prefix, amplitude=5, **kwargs):

    params = Parameters()

    params.add(prefix + 'amp', value=amplitude)
    params.add(prefix + 'cen')
    params.add(prefix + 'sigma')
    params.add(prefix + 'shift', value=0, vary=False)

    model = Model(gaussian, prefix=prefix)

    return model, params


def setup_constant(prefix, amplitude=1, **kwargs):

    params = Parameters()

    params.add(prefix + 'amp', value=amplitude)

    model = Model(constant, prefix=prefix)

    return model, params


def setup_lorentzian(prefix, amplitude=5, **kwargs):

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
                                                 '../sculptor-extensions')

file_names = glob.glob(extension_path+'/*.py')

module_names = [os.path.basename(f)[:-3] for f in file_names if
                os.path.isfile(f) and not f.endswith('__init__.py')]

for module_name in module_names:
    print('[INFO] Import "sculptor-extensions" packages: {}'.format(
        module_name))
    module = importlib.import_module('sculptor-extensions.{}'.format(
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