#!/usr/bin/env python

"""The Sculptor Quasar Extension

This module defines models, masks and analysis routines specific for the
analysis of type-I QSOs (quasars, type-I AGN).

At first we define the basic model functions and their setups. The setup
functions initialize LMFIT models and parameters using the basic model
functions defined here.

Complex models can be constructed by combining multiple of the basic model
functions. For example, we define a setup function to initialize a model for
the Hbeta and [OIII] lines consistent of six Gaussian emission lines.

"""

import os
import numpy as np
import scipy as sp

from astropy import constants as const
from astropy import units as u
from astropy.modeling.models import BlackBody

from sculptor import speconed as sod

from lmfit import Model, Parameters

from linetools import utils as lu

import pkg_resources

datadir = pkg_resources.resource_filename('sculptor', 'data/')

c_km_s = const.c.to('km/s').value
c_AngstroemPerSecond = const.c.to(u.AA/u.s).value

# ------------------------------------------------------------------------------
# Model functions
# ------------------------------------------------------------------------------

def power_law_at_2500(x, amp, slope, z):
    """ Power law model anchored at 2500 AA

    This model is defined for a spectral dispersion axis in Angstroem.

    :param x: Dispersion of the power law
    :type x: np.ndarray
    :param amp: Amplitude of the power law (at 2500 A)
    :type amp: float
    :param slope: Slope of the power law
    :type slope: float
    :param z: Redshift
    :type z: float
    :return: Power law model
    :rtype: np.ndarray

    """

    return amp * (x / (2500. * (z+1.))) ** slope


def balmer_continuum_model(x, z, amp_be, Te, tau_be, lambda_be):
    """ Model of the Balmer continuum (Dietrich 2003)

    This model is defined for a spectral dispersion axis in Angstroem.

    This functions implements the Balmer continuum model presented in
    Dietrich 2003. The model follows a blackbody below the Balmer edge (
    3646A) and is zero above.

    The amplitude parameter amp_be defines the amplitude at the balmer edge.
    The strength of the Balmer continuum can be estimated from the fluxden
    density at 3675A after subtraction of the power-law continuum component
    for reference see Grandi et al.(1982), Wills et al.(1985) or Verner et al.(
    1999).

    At wavelengths of 3646A higher order Balmer lines are merging. This has
    not been included in this model and thus it will produce a sharp break at
    the Balmer edge.

    :param x: Dispersion of the Balmer continuum
    :type x: np.ndarray
    :param z: Redshift
    :type z: float
    :param amp_be: Amplitude of the Balmer continuum at the Balmer edge
    :type amp_be: float
    :param Te: Electron temperature
    :type Te: float
    :param tau_be: Optical depth at the Balmer edge
    :type tau_be: float
    :param lambda_be: Wavelength of the Balmer edge
    :type lambda_be: float
    :return: Balmer continuum model
    :rtype: np.ndarray

    """

    x = x / (1.+z)

    bb = BlackBody(temperature=Te*u.K)
    black_body_lambda = bb(x*u.AA).value * c_AngstroemPerSecond / x**2

    fluxden = amp_be * black_body_lambda * (
                1. - np.exp(-tau_be * (x / lambda_be) ** 3))

    fluxden[x >= lambda_be] = fluxden[x >= lambda_be] * 0

    fluxden *= 1e-20

    return fluxden


def power_law_at_2500_plus_fractional_bc(x, amp, slope, z, f, Te, tau_be,
                                       lambda_be):
    """ QSO continuum model consisting of a power law anchored at 2500 A and
    a balmer continuum contribution.

    This model is defined for a spectral dispersion axis in Angstroem.

    The amplitude of the Balmer continuum is set to be a fraction of the
    power law component at the Balmer edge (3646A) using the variabe f.

    :param x: Dispersion of the continuum model
    :type x: np.ndarray
    :param amp: Amplitude of the power law (at 2500 A)
    :type amp: float
    :param slope: Slope of the power law
    :type slope: float
    :param z: Redshift
    :type z: float
    :param f: Amplitude of the Balmer continuum as a fraction of the power \
        law component
    :type f: float
    :param Te: Electron temperature
    :type Te: float
    :param tau_be: Optical depth at the Balmer edge
    :type tau_be: float
    :param lambda_be: Wavelength of the Balmer edge
    :type lambda_be: float
    :return: QSO continuum model
    :rtype: np.ndarray

    """

    pl_fluxden = amp * (x / (2500. * (z + 1.))) ** slope

    pl_fluxden_at_be = amp * ((lambda_be * (z + 1.)) / (2500. * (z + 1.))) **\
                       slope

    x = x / (1. + z)

    bb = BlackBody(temperature=Te*u.K)
    black_body_lambda = bb(x*u.AA).value * c_AngstroemPerSecond / x**2
    black_body_lambda_be = bb(lambda_be*u.AA).value * c_AngstroemPerSecond / \
                            lambda_be**2

    bc_fluxden_at_lambda_be = black_body_lambda_be * (
                1. - np.exp(-tau_be))

    bc_fluxden = f * pl_fluxden_at_be / bc_fluxden_at_lambda_be * \
                black_body_lambda * (
                1. - np.exp(-tau_be * (x / lambda_be) ** 3))

    bc_fluxden[x >= lambda_be] = bc_fluxden[x >= lambda_be] * 0

    return pl_fluxden + bc_fluxden


def power_law_at_2500_plus_bc(x, amp, slope, z, amp_be, Te, tau_be,
                                       lambda_be):
    """ QSO continuum model consisting of a power law anchored at 2500 A and
    a balmer continuum contribution.

    This model is defined for a spectral dispersion axis in Angstroem.

    The amplitude of the Balmer continuum is set independently of the
    power law component by the amplitude of the balmer continuum at the
    balmer edge amp_be.

    :param x: Dispersion of the continuum model
    :type x: np.ndarray
    :param amp: Amplitude of the power law (at 2500 A)
    :type amp: float
    :param slope: Slope of the power law
    :type slope: float
    :param z: Redshift
    :type z: float
    :param amp_be: Amplitude of the Balmer continuum at the Balmer edge
    :type amp_be: float
    :param Te: Electron temperature
    :type Te: float
    :param tau_be: Optical depth at the Balmer edge
    :type tau_be: float
    :param lambda_be: Wavelength of the Balmer edge
    :type lambda_be: float
    :return: QSO continuum model
    :rtype: np.ndarray

    """

    pl_fluxden = amp * (x / (2500. * (z + 1.))) ** slope

    x = x / (1. + z)

    bb = BlackBody(temperature=Te*u.K)
    black_body_lambda = bb(x*u.AA).value * c_AngstroemPerSecond / x**2
    black_body_lambda_be = bb(lambda_be*u.AA).value * c_AngstroemPerSecond / \
                            lambda_be**2

    bc_fluxden_0 = amp_be / (black_body_lambda_be *
                                        (1. - np.exp(-tau_be)))

    bc_fluxden = bc_fluxden_0 * black_body_lambda * \
                 (1. - np.exp(-tau_be * (x / lambda_be) ** 3))

    bc_fluxden[x >= lambda_be] = bc_fluxden[x >= lambda_be] * 0

    return pl_fluxden + bc_fluxden


def line_model_gaussian(x, z, amp, cen, fwhm_km_s, shift_km_s):
    """ Gaussian line model

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

    cen = cen * (1+z)

    delta_cen = shift_km_s / c_km_s * cen
    central = cen + delta_cen
    fwhm = fwhm_km_s / c_km_s * central
    sigma = fwhm / np.sqrt(8*np.log(2))

    return amp * np.exp(-(x-central)**2 / (2*sigma**2))


def template_model(x, amp, z, fwhm, intr_fwhm, templ_disp=None,
                   templ_fluxden=None,
                   templ_disp_unit_str=None,
                   templ_fluxden_unit_str=None):
    """Template model

    :param x: Dispersion of the template model
    :type x: np.ndarray
    :param amp: Amplitude of the template model
    :type amp: float
    :param z: Redshift
    :type z: float
    :param fwhm: FWHM the template should be broadened to
    :type fwhm: float
    :param intr_fwhm: Intrinsic FWHM of the template
    :type intr_fwhm: float
    :param templ_disp: Dispersion axis of the template. This must match the \
        same dispersion unit as the model
    :type templ_disp: np.ndarray
    :param templ_fluxden: Flux density of the template.
    :type templ_fluxden: templ_fluxden: np.ndarray
    :param templ_disp_unit_str: Dispersion unit of the template as a string
        in astropy cds format.
    :type templ_fluxden_unit_str: str
    :param templ_fluxden_unit: Flux density unit of the template as a string
        in astropy cds format.
    :type templ_disp_unit_str: str
    :return: Template model as a Scipy interpolation

    """

    if templ_disp_unit_str is not None and type(templ_disp_unit_str) is str:
        dispersion_unit = u.Unit(templ_disp_unit_str, format='cds')
    else:
        dispersion_unit = None
    if templ_fluxden_unit_str is not None and \
            type(templ_fluxden_unit_str)is str:
        fluxden_unit = u.Unit(templ_fluxden_unit_str, format='cds')
    else:
        fluxden_unit = None

    # Initialize a SpecOneD model for the template
    template_spec = sod.SpecOneD(dispersion=templ_disp, fluxden=templ_fluxden,
                                 dispersion_unit=dispersion_unit,
                                 fluxden_unit=fluxden_unit)

    # Calculate the FWHM for the convolution
    convol_fwhm = np.sqrt(fwhm**2-intr_fwhm**2)
    # Convolve the template spectrum to the goal FWHM
    spec = template_spec.broaden_by_gaussian(convol_fwhm)

    # Shift the dispersion axis to the specified redshift. This does not
    # change the amplitude of the template spectrum as is normally done for
    # cosmological redshifting. As we want to determine the amplitude it is a
    # free parameter anyway.
    spec.dispersion = spec.dispersion * (1.+z)

    # Return interpolation function
    f = sp.interpolate.interp1d(spec.dispersion, spec.fluxden, kind='linear',
                                bounds_error=False, fill_value=(0, 0))

    return f(x)*amp


def CIII_complex_model_func(x, z, cen, cen_alIII, cen_siIII,
                          amp, fwhm_km_s, shift_km_s,
                          amp_alIII, fwhm_km_s_alIII, shift_km_s_alIII,
                          amp_siIII, fwhm_km_s_siIII, shift_km_s_siIII):
    """Model function for the CIII] emission line complex, consisting of the
    Gaussian line models with a combined redshift parameter.

    The width of the line is set by the FWHM in km/s.

    The Gaussians are not normalized.

    :param x: Dispersion of the template model
    :type x: np.ndarray
    :param z: Redshift for CIII], AlIII, SiIII]
    :type z: float
    :param cen: CIII] central wavelength
    :type cen: float
    :param cen_alIII: AlIII central wavelength
    :type cen_alIII: float
    :param cen_siIII:  SiIII] central wavelength
    :type cen_siIII: float
    :param amp: Amplitude of the CIII] line
    :type amp: float
    :param fwhm_km_s:  Full Width at Half Maximum (FWHM) of CIII] in km/s
    :type fwhm_km_s: float
    :param shift_km_s: Doppler velocity shift of the central wavelength
    :type shift_km_s: float
    :param amp_alIII: Amplitude of the AlIII line
    :param fwhm_km_s_alIII:  Full Width at Half Maximum (FWHM) of AlIII in km/s
    :type fwhm_km_s_alIII: float
    :param shift_km_s_alIII: Doppler velocity shift of the central wavelength
    :type shift_km_s: float
    :param amp_siIII: Amplitude of the SiIII] line
    :param fwhm_km_s_siIII: Full Width at Half Maximum (FWHM) of SiIII] in km/s
    :type fwhm_km_s_siIII: float
    :param shift_km_s_siIII: Doppler velocity shift of the central wavelength
    :type shift_km_s: float
    :return: CIII] complex model

    """

    cIII_cen = cen * (1 + z)
    siIII_cen = cen_siIII * (1 + z)
    alIII_cen = cen_alIII * (1 + z)

    cIII_delta_cen = shift_km_s / c_km_s * cIII_cen
    siIII_delta_cen = shift_km_s_siIII / c_km_s * siIII_cen
    alIII_delta_cen = shift_km_s_alIII / c_km_s * alIII_cen


    central = cIII_cen + cIII_delta_cen
    fwhm = fwhm_km_s / c_km_s * central
    sigma = fwhm / np.sqrt(8 * np.log(2))
    gauss_cIII = amp * np.exp(
        -(x - central) ** 2 / (2 * sigma ** 2))

    central = siIII_cen + siIII_delta_cen
    fwhm = fwhm_km_s_siIII / c_km_s * central
    sigma = fwhm / np.sqrt(8 * np.log(2))
    gauss_siIII = amp_siIII * np.exp(
        -(x - central) ** 2 / (2 * sigma ** 2))

    central = alIII_cen + alIII_delta_cen
    fwhm = fwhm_km_s_alIII / c_km_s * central
    sigma = fwhm / np.sqrt(8 * np.log(2))
    gauss_alIII = amp_alIII * np.exp(
        -(x - central) ** 2 / (2 * sigma ** 2))

    return gauss_cIII + gauss_alIII + gauss_siIII

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# Model setups
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------


def add_redshift_param(redshift, params, prefix):
    """ Add the redshift parameter to the LMFIT parameters.

    :param redshift: Redshift
    :type redshift: float
    :param params: Model parameters
    :type params: lmfit.Parameters
    :param prefix: Model prefix
    :type prefix: string
    :return:

    """

    z_min = min(redshift*1.05, 1080)
    z_max = max(1.0, redshift * 0.95)

    params.add(prefix+'z', value=redshift, min=z_min, max=z_max, vary=False)


def setup_power_law_at_2500(prefix, **kwargs):
    """Initialize the power law model anchored at 2500 A.

    :param prefix: Model prefix
    :type prefix: string
    :param kwargs: Keyword arguments
    :return: LMFIT model and parameters
    :rtype: (lmfit.Model, lmfit.Parameters)

    """

    redshift = kwargs.pop('redshift', 0)
    amplitude = kwargs.pop('amplitude', 1)

    params = Parameters()

    add_redshift_param(redshift, params, prefix)

    params.add(prefix + 'amp', value=amplitude)
    params.add(prefix + 'slope', value=-1.5, min=-2.5, max=-0.3)

    model = Model(power_law_at_2500, prefix=prefix)

    return model, params


def setup_power_law_at_2500_plus_fractional_bc(prefix, **kwargs):
    """Initialize the quasar continuum model consistent of a power law \
        anchored at 2500A and a balmer continuum contribution.

    The Balmer continuum amplitude at the Balmer edge is set to be a fraction of
    the power law component.

    :param prefix: Model prefix
    :type prefix: string
    :param kwargs: Keyword arguments
    :return: LMFIT model and parameters
    :rtype: (lmfit.Model, lmfit.Parameters)

    """

    redshift = kwargs.pop('redshift', 0)
    amplitude = kwargs.pop('amplitude', 1)

    params = Parameters()
    add_redshift_param(redshift, params, prefix)
    params.add(prefix + 'amp', value=amplitude)
    params.add(prefix + 'slope', value=-1.5, min=-2.5, max=-0.3)
    params.add(prefix + 'f', value=0.1, min=0.0, max=0.4, vary=False)
    params.add(prefix + 'Te', value=15000, min=10000, max=20000, vary=False)
    params.add(prefix + 'tau_be', value=1.0, min=0.1, max=2.0, vary=False)
    params.add(prefix + 'lambda_be', value=3646, vary=False)

    model = Model(power_law_at_2500_plus_fractional_bc, prefix=prefix)

    return model, params


def setup_power_law_at_2500_plus_bc(prefix, **kwargs):
    """Initialize the quasar continuum model consistent of a power law \
        anchored at 2500A and a balmer continuum contribution.

    :param prefix: Model prefix
    :type prefix: string
    :param kwargs: Keyword arguments
    :return: LMFIT model and parameters
    :rtype: (lmfit.Model, lmfit.Parameters)

    """

    redshift = kwargs.pop('redshift', 0)
    amplitude = kwargs.pop('amplitude', 1)

    params = Parameters()
    add_redshift_param(redshift, params, prefix)

    params.add(prefix + 'amp', value=amplitude)
    params.add(prefix + 'slope', value=-1.5, min=-2.5, max=-0.3)
    params.add(prefix + 'amp_be', value=0.5, min=0.0, max=10, vary=True)
    params.add(prefix + 'Te', value=15000, min=10000, max=20000, vary=False)
    params.add(prefix + 'tau_be', value=1.0, min=0.1, max=2.0, vary=False)
    params.add(prefix + 'lambda_be', value=3646, vary=False)

    model = Model(power_law_at_2500_plus_bc, prefix=prefix)

    return model, params


def setup_line_model_gaussian(prefix, **kwargs):
    """Initialize the Gaussian line model.

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

    add_redshift_param(redshift, params, prefix)

    params.add(prefix + 'amp', value=amplitude)
    params.add(prefix + 'cen', value=cenwave)
    params.add(prefix + 'fwhm_km_s', value=fwhm)
    params.add(prefix + 'shift_km_s', value=vshift, vary=False)

    model = Model(line_model_gaussian, prefix=prefix)

    return model, params

# ------------------------------------------------------------------------------
# Iron template models
# ------------------------------------------------------------------------------


def setup_iron_template_model(prefix, template_filename,
                              templ_disp_unit, templ_fluxden_unit,
                              fwhm=2500,
                              redshift=0,
                              amplitude=1, intr_fwhm=900,
                              dispersion_limits=None):
    """Initialize an iron template model

    :param prefix: Model prefix
    :type prefix: string
    :param template_filename: Filename of the iron template
    :type template_filename: string
    :param fwhm: FWHM the template should be broadened to
    :type fwhm: float
    :param redshift: Redshift
    :type redshift: float
    :param amplitude: Amplitude of the template model
    :type amplitude: float
    :param intr_fwhm: Intrinsic FWHM of the template
    :type intr_fwhm: float
    :param dispersion_limits:
    :type dispersion_limits: (float, float)
    :return: LMFIT model and parameters
    :rtype: (lmfit.Model, lmfit.Parameters)

    """

    # Initialize parameters
    params = Parameters()

    # Load template model from Sculptor data directory
    template = np.genfromtxt(datadir+'iron_templates/'+template_filename)

    # Convert astropy units to string to be able to save them with LMFIT
    if type(templ_disp_unit) is u.Quantity:
        templ_disp_unit_str = templ_disp_unit.unit.to_string('cds')
        templ_dist_unit_factor = templ_disp_unit.value
    elif type(templ_disp_unit) is u.Unit or type(templ_disp_unit) is \
            u.CompositeUnit:
        templ_disp_unit_str = templ_disp_unit.to_string('cds')
    else:
        raise ValueError('[ERROR] Template flux density unit type is not an '
                         'astropy unit or astropy quantity. Current datatype: {}'.format(type(templ_disp_unit)))

    if type(templ_fluxden_unit) is u.Quantity:
        templ_fluxden_unit_str = templ_fluxden_unit.unit.to_string('cds')
        templ_fluxden_unit_factor = templ_fluxden_unit.value
    elif type(templ_fluxden_unit) is u.Unit or type(templ_fluxden_unit) is \
            u.CompositeUnit:
        templ_fluxden_unit_str = templ_fluxden_unit.to_string('cds')
    else:
        raise ValueError('[ERROR] Template flux density unit type is not an '
                         'astropy unit or astropy quantity. Current datatype: {}'.format(type(templ_fluxden_unit)))

    # Apply dispersion limits
    if dispersion_limits is not None:
        wav_min = dispersion_limits[0]
        wav_max = dispersion_limits[1]

        idx_min = np.argmin(np.abs(template[:, 0] - wav_min))
        idx_max = np.argmin(np.abs(template[:, 0] - wav_max))

        model = Model(template_model,
                      param_names=['amp', 'z', 'fwhm', 'intr_fwhm'],
                      templ_disp=template[idx_min:idx_max, 0],
                      templ_fluxden=template[idx_min:idx_max, 1],
                      templ_disp_unit_str=templ_disp_unit_str,
                      templ_fluxden_unit_str=templ_fluxden_unit_str,
                      prefix=prefix)
    else:
        model = Model(template_model,
                      param_names=['amp', 'z', 'fwhm', 'intr_fwhm'],
                      templ_disp=template,
                      templ_fluxden=template,
                      templ_disp_unit_str=templ_disp_unit_str,
                      templ_fluxden_unit_str=templ_fluxden_unit_str,
                      prefix=prefix)

    add_redshift_param(redshift, params, prefix)
    params.add(prefix + 'fwhm', value=fwhm, min=0, max=10000, vary=False)
    params.add(prefix + 'amp', value=amplitude, min=1, max=1e+3)
    params.add(prefix + 'intr_fwhm', value=intr_fwhm, vary=False)

    return model, params


def setup_subdivided_iron_template(templ_list, fwhm=2500, redshift=0,
                                   amplitude=1):
    """Setup iron template models from a predefined list of templates and \
    dispersion ranges.

    :param templ_list: List of template names for which models will be set up
    :type: list
    :param fwhm: Goal FWHM of the template model
    :type fwhm: float
    :param redshift: Redshift of the template model
    :type redshift: float
    :param amplitude: Amplitude of the template model
    :type amplitude: float
    :return: Return a list of LMFIT models and a list of LMFIT parameters
    :rtype: (list, list)

    """

    # Empty parameters and mode lists
    param_list = []
    model_list = []

    templ_disp_unit = u.AA
    templ_fluxden_unit = u.erg/u.s/u.cm**2/u.AA

    if 'UV01_VW01' in templ_list:
        # 1200-1560 Vestergaard 2001
        model, params = setup_iron_template_model(
            'UV01_VW01_', 'Fe_UVtemplt_A.asc',
            templ_disp_unit,
            templ_fluxden_unit,
            fwhm=fwhm,
            redshift=redshift,
            amplitude=amplitude, intr_fwhm=900, dispersion_limits=[1200, 1560])
        param_list.append(params)
        model_list.append(model)
    if 'UV02_VW01' in templ_list or not templ_list:
        # 1560-1875 Vestergaard 2001
        model, params = setup_iron_template_model(
            'UV02_VW01_', 'Fe_UVtemplt_A.asc',
            templ_disp_unit,
            templ_fluxden_unit,
            fwhm=fwhm, redshift=redshift,
            amplitude=amplitude, intr_fwhm=900, dispersion_limits=[1560, 1875])
        param_list.append(params)
        model_list.append(model)
    if 'UV03_VW01' in templ_list or not templ_list:
        # 1875-2200 Vestergaard 2001
        model, params = setup_iron_template_model(
            'UV03_VW01_', 'Fe_UVtemplt_A.asc',
            templ_disp_unit,
            templ_fluxden_unit,
            fwhm=fwhm, redshift=redshift,
            amplitude=amplitude, intr_fwhm=900, dispersion_limits=[1875, 2200])
        param_list.append(params)
        model_list.append(model)
    if 'FeIIIUV34_VW01' in templ_list or not templ_list:
        # Vestergaard 2001 - FeIII UV34 template
        model, params = setup_iron_template_model(
            'FeIIIUV34_VW01_', 'Fe3UV34modelB2.asc',
            templ_disp_unit,
            templ_fluxden_unit,
            fwhm=fwhm,
            redshift=redshift,
            amplitude=amplitude, intr_fwhm=900, dispersion_limits=[1875, 2200])
        param_list.append(params)
        model_list.append(model)
    if 'UV04_T06' in templ_list or not templ_list:
        # 2200-2660 Tsuzuki 2006
        model, params = setup_iron_template_model(
            'UV04_T06_', 'Tsuzuki06.txt',
            templ_disp_unit,
            templ_fluxden_unit,
            fwhm=fwhm, redshift=redshift,
            amplitude=amplitude, intr_fwhm=900, dispersion_limits=[2200, 2660])
        param_list.append(params)
        model_list.append(model)
    if 'UV05_T06' in templ_list or not templ_list:
        # 2660-3000 Tsuzuki 2006
        model, params = setup_iron_template_model(
            'UV05_T06_', 'Tsuzuki06.txt',
            templ_disp_unit,
            templ_fluxden_unit,
            fwhm=fwhm, redshift=redshift,
            amplitude=amplitude, intr_fwhm=900, dispersion_limits=[2660, 3000])
        param_list.append(params)
        model_list.append(model)
    if 'UV06_T06' in templ_list or not templ_list:
        # 3000-3500 Tsuzuki 2006
        model, params = setup_iron_template_model(
            'UV06_T06_', 'Tsuzuki06.txt',
            templ_disp_unit,
            templ_fluxden_unit,
            fwhm=fwhm, redshift=redshift,
            amplitude=amplitude, intr_fwhm=900, dispersion_limits=[3000, 3500])
        param_list.append(params)
        model_list.append(model)
    if 'OPT01_BG92' in templ_list or not templ_list:
        # 4400-4700 Boronson & Green 1992
        model, params = setup_iron_template_model(
            'OPT01_BG92_', 'Fe_OPT_BR92_linear.txt',
            templ_disp_unit,
            templ_fluxden_unit,
            fwhm=fwhm,
            redshift=redshift, amplitude=amplitude, intr_fwhm=900,
            dispersion_limits=[3700, 4700])
        param_list.append(params)
        model_list.append(model)
    if 'OPT02_BG92' in templ_list or not templ_list:
        # 4700-5100 Boronson & Green 1992
        model, params = setup_iron_template_model(
            'OPT02_BG92_', 'Fe_OPT_BR92_linear.txt',
            templ_disp_unit,
            templ_fluxden_unit,
            fwhm=fwhm,
            redshift=redshift, amplitude=amplitude, intr_fwhm=900,
            dispersion_limits=[4700, 5100])
        param_list.append(params)
        model_list.append(model)
    if 'OPT03_BG92' in templ_list or not templ_list:
        # 5100-5600 Boronson & Green 1992
        model, params = setup_iron_template_model(
            'OPT03_BG92_', 'Fe_OPT_BR92_linear.txt',
            templ_disp_unit,
            templ_fluxden_unit,
            fwhm=fwhm,
            redshift=redshift, amplitude=amplitude, intr_fwhm=900,
            dispersion_limits=[5100, 5600])
        param_list.append(params)
        model_list.append(model)
    # Vestergaard templates
    if 'UV04_VW01' in templ_list or not templ_list:
        # 2200-2660 Vestergaard 2001
        model, params = setup_iron_template_model(
            'UV04_VW01_', 'Fe_UVtemplt_A.asc',
            templ_disp_unit,
            templ_fluxden_unit,
            fwhm=fwhm, redshift=redshift,
            amplitude=amplitude, intr_fwhm=900, dispersion_limits=[2200, 2660])
        param_list.append(params)
        model_list.append(model)
    if 'UV05_VW01' in templ_list or not templ_list:
        # 2660-3000 Vestergaard 2001
        model, params = setup_iron_template_model(
            'UV05_VW01_', 'Fe_UVtemplt_A.asc',
            templ_disp_unit,
            templ_fluxden_unit,
            fwhm=fwhm, redshift=redshift,
            amplitude=amplitude, intr_fwhm=900, dispersion_limits=[2660, 3000])
        param_list.append(params)
        model_list.append(model)
    if 'UV06_VW01' in templ_list or not templ_list:
        # 3000-3500 Vestergaard 2001
        model, params = setup_iron_template_model(
            'UV06_VW01_', 'Fe_UVtemplt_A.asc',
            templ_disp_unit,
            templ_fluxden_unit,
            fwhm=fwhm, redshift=redshift,
            amplitude=amplitude, intr_fwhm=900, dispersion_limits=[3000, 3500])
        param_list.append(params)
        model_list.append(model)

    return model_list, param_list


def setup_iron_template_T06(prefix, **kwargs):
    """ Setup the Tsuzuki 2006 iron template model

    The dispersion axis for this model is in Angstroem.

    :param prefix: The input parameter exists for conformity with the \
        Sculptor models, but will be ignored. The prefix is automatically set \
        by the setup function.
    :type prefix: string
    :param kwargs:
    :return: LMFIT model and parameters
    :rtype: (lmfit.Model, lmfit.Parameters)

    """

    templ_disp_unit = u.AA
    templ_fluxden_unit = u.erg / u.s / u.cm ** 2 / u.AA

    redshift = kwargs.pop('redshift', 0)
    amplitude = kwargs.pop('amplitude', 1)
    fwhm = kwargs.pop('fwhm', 2500)
    # Tsuzuki 2006
    model, params = setup_iron_template_model(
        'FeII_T06_', 'Tsuzuki06.txt', templ_disp_unit,
        templ_fluxden_unit, fwhm=fwhm, redshift=redshift,
        amplitude=amplitude, intr_fwhm=900)

    return model, params


def setup_iron_template_UV_VW01(prefix, **kwargs):
    """ Setup the Vestergaard & Wilkes 2001 iron template model around CIV \
        (1200-2200A)

    The dispersion axis for this model is in Angstroem.

    :param prefix: The input parameter exists for conformity with the \
        Sculptor models, but will be ignored. The prefix is automatically set \
        by the setup function.
    :type prefix: string
    :param kwargs:
    :return: LMFIT model and parameters
    :rtype: (lmfit.Model, lmfit.Parameters)

    """

    templ_disp_unit = u.AA
    templ_fluxden_unit = u.erg / u.s / u.cm ** 2 / u.AA

    redshift = kwargs.pop('redshift', 0)
    amplitude = kwargs.pop('amplitude', 1)
    fwhm = kwargs.pop('fwhm', 2500)
    # 1200-2200 Vestergaard 2001
    model, params = setup_iron_template_model(
        'FeIIUV_VW01_', 'Fe_UVtemplt_A.asc', templ_disp_unit,
        templ_fluxden_unit,
        fwhm=fwhm, redshift=redshift,
        amplitude=amplitude, intr_fwhm=900, dispersion_limits=[1200, 2200])

    return model, params


def setup_split_iron_template_UV_VW01(prefix, **kwargs):
    """ Setup the Vestergaard & Wilkes 2001 iron template model subdivided \
        into three separate models at 1200-1560, 1560-1875, 1875-2200.

    The dispersion axis for this model is in Angstroem.

    :param prefix: The input parameter exists for conformity with the \
        Sculptor models, but will be ignored. The prefix is automatically set \
        by the setup function.
    :type prefix: string
    :param kwargs:
    :return: Return a list of LMFIT models and a list of LMFIT parameters
    :rtype: (list, list)

    """

    redshift = kwargs.pop('redshift', 0)
    amplitude = kwargs.pop('amplitude', 1)
    fwhm = kwargs.pop('fwhm', 2500)
    templ_list = ['UV01_VW01', 'UV02_VW01', 'UV03_VW01']

    model_list, param_list = setup_subdivided_iron_template(templ_list,
                                                   fwhm=fwhm,
                                                   redshift=redshift,
                                                   amplitude=amplitude,
                                                   )

    return model_list, param_list


def setup_iron_template_MgII_T06(prefix, **kwargs):
    """ Setup the Tsuzuki 2006 iron template model around MgII (2200-3500A)

    The dispersion axis for this model is in Angstroem.

    :param prefix: The input parameter exists for conformity with the \
        Sculptor models, but will be ignored. The prefix is automatically set \
        by the setup function.
    :type prefix: string
    :param kwargs:
    :return: LMFIT model and parameters
    :rtype: (lmfit.Model, lmfit.Parameters)

    """

    templ_disp_unit = u.AA
    templ_fluxden_unit = u.erg / u.s / u.cm ** 2 / u.AA

    redshift = kwargs.pop('redshift', 0)
    amplitude = kwargs.pop('amplitude', 1)
    fwhm = kwargs.pop('fwhm', 2500)
    # 2200-3500 Tsuzuki 2006
    model, params = setup_iron_template_model(
        'FeIIMgII_T06_', 'Tsuzuki06.txt', templ_disp_unit,
        templ_fluxden_unit, fwhm=fwhm, redshift=redshift,
        amplitude=amplitude, intr_fwhm=900, dispersion_limits=[2200, 3500])

    return model, params


def setup_split_iron_template_MgII_T06(prefix, **kwargs):
    """ Setup the Tsuzuki 2006 iron template model subdivided into three \
        separate models at 2200-2660, 2660-3000, 3000-3500.

    The dispersion axis for this model is in Angstroem.

    :param prefix: The input parameter exists for conformity with the \
        Sculptor models, but will be ignored. The prefix is automatically set \
        by the setup function.
    :type prefix: string
    :param kwargs:
    :return: Return a list of LMFIT models and a list of LMFIT parameters
    :rtype: (list, list)

    """

    redshift = kwargs.pop('redshift', 0)
    amplitude = kwargs.pop('amplitude', 1)
    fwhm = kwargs.pop('fwhm', 2500)
    templ_list = ['UV04_T06', 'UV05_T06', 'UV06_T06']

    model_list, param_list = setup_subdivided_iron_template(
        templ_list, fwhm=fwhm, redshift=redshift, amplitude=amplitude)

    return model_list, param_list


def setup_iron_template_MgII_VW01(prefix, **kwargs):
    """ Setup the Vestergaard & Wilkes 2001 iron template model around MgII \
        (2200-3500A)

    The dispersion axis for this model is in Angstroem.

    :param prefix: The input parameter exists for conformity with the \
        Sculptor models, but will be ignored. The prefix is automatically set \
        by the setup function.
    :type prefix: string
    :param kwargs:
    :return: LMFIT model and parameters
    :rtype: (lmfit.Model, lmfit.Parameters)

    """

    templ_disp_unit = u.AA
    templ_fluxden_unit = u.erg / u.s / u.cm ** 2 / u.AA

    redshift = kwargs.pop('redshift', 0)
    amplitude = kwargs.pop('amplitude', 1)
    fwhm = kwargs.pop('fwhm', 2500)
    # 2200-3500 Vestergaard 2001
    model, params = setup_iron_template_model(
        'FeIIMgII_VW01_', 'Fe_UVtemplt_A.asc', templ_disp_unit,
        templ_fluxden_unit, fwhm=fwhm, redshift=redshift,
        amplitude=amplitude, intr_fwhm=900, dispersion_limits=[2200, 3500])

    return model, params


def setup_split_iron_template_MgII_VW01(prefix, **kwargs):
    """ Setup the Vestergaard & Wilkes 2001 iron template model subdivided \
        into three separate models at 2200-2660, 2660-3000, 3000-3500.

    The dispersion axis for this model is in Angstroem.

    :param prefix: The input parameter exists for conformity with the \
        Sculptor models, but will be ignored. The prefix is automatically set \
        by the setup function.
    :type prefix: string
    :param kwargs:
    :return: Return a list of LMFIT models and a list of LMFIT parameters
    :rtype: (list, list)

    """

    redshift = kwargs.pop('redshift', 0)
    amplitude = kwargs.pop('amplitude', 1)
    fwhm = kwargs.pop('fwhm', 2500)
    templ_list = ['UV04_VW01', 'UV05_VW01', 'UV06_VW01']

    model_list, param_list = setup_subdivided_iron_template(
        templ_list, fwhm=fwhm, redshift=redshift, amplitude=amplitude)
    return model_list, param_list


def setup_iron_template_OPT_BG92(prefix, **kwargs):
    """ Setup the Boroson & Green 1992 iron template model around Hbeta \
        (3700-7480A)

    The dispersion axis for this model is in Angstroem.

    :param prefix: The input parameter exists for conformity with the \
        Sculptor models, but will be ignored. The prefix is automatically set \
        by the setup function.
    :type prefix: string
    :param kwargs:
    :return: LMFIT model and parameters
    :rtype: (lmfit.Model, lmfit.Parameters)

    """

    templ_disp_unit = u.AA
    templ_fluxden_unit = u.erg / u.s / u.cm ** 2 / u.AA

    redshift = kwargs.pop('redshift', 0)
    amplitude = kwargs.pop('amplitude', 1)
    fwhm = kwargs.pop('fwhm', 2500)
    # 3700-5600 Boronson & Green 1992
    model, params = setup_iron_template_model(
        'FeIIOPT_BG92_', 'Fe_OPT_BR92_linear.txt', templ_disp_unit,
        templ_fluxden_unit, fwhm=fwhm, redshift=redshift,
        amplitude=amplitude, intr_fwhm=900, dispersion_limits=[3700, 7480])

    return model, params


def setup_split_iron_template_OPT_BG92(prefix, **kwargs):
    """ Setup the Boroson & Green 1992 iron template model subdivided into \
        three separate models at 3700-4700, 4700-5100, 5100-5600.

    The dispersion axis for this model is in Angstroem.

    :param prefix: The input parameter exists for conformity with the \
        Sculptor models, but will be ignored. The prefix is automatically set \
        by the setup function.
    :type prefix: string
    :param kwargs:
    :return: Return a list of LMFIT models and a list of LMFIT parameters
    :rtype: (list, list)

    """

    redshift = kwargs.pop('redshift', 0)
    amplitude = kwargs.pop('amplitude', 1)
    fwhm = kwargs.pop('fwhm', 2500)
    templ_list = ['OPT01_BG92', 'OPT02_BG92', 'OPT03_BG92']

    model_list, param_list = setup_subdivided_iron_template(
        templ_list, fwhm=fwhm, redshift=redshift, amplitude=amplitude)

    return model_list, param_list

# ------------------------------------------------------------------------------
# Emission line models
# ------------------------------------------------------------------------------


def setup_line_model_SiIV_2G(prefix, **kwargs):
    """ Set up a 2 component Gaussian line model for the SiIV emission line.

    This setup models the broad SiIV line emission as seen in type-I quasars
    and AGN. Due to the broad nature of the line SiIV, the SiIV doublet is
    assumed to be unresolved and blended with the OIV emission line.

    This model is defined for a spectral dispersion axis in Angstroem.

    :param prefix: The input parameter exists for conformity with the \
        Sculptor models, but will be ignored. The prefix is automatically set \
        by the setup function.
    :type prefix: string
    :param kwargs:
    :return: Return a list of LMFIT models and a list of LMFIT parameters
    :rtype: (list, list)

    """

    redshift = kwargs.pop('redshift', 0)
    amplitude = kwargs.pop('amplitude', 1)

    # CIV 2G model params
    prefixes = ['SiIV_A_', 'SiIV_B_']
    amplitudes = np.array([2, 2]) * amplitude
    fwhms = [2500, 2500]
    # SiIV + OIV] http://classic.sdss.org/dr6/algorithms/linestable.html
    cenwaves = [1399.8, 1399.8]
    vshifts = [0, 0]

    param_list = []
    model_list = []

    # Build parameter and model list
    for idx, prefix in enumerate(prefixes):
        model, params = setup_line_model_gaussian(prefix,
                                                  redshift=redshift,
                                                  amplitude=amplitudes[idx],
                                                  cenwave=cenwaves[idx],
                                                  fwhm=fwhms[idx],
                                                  vshift=vshifts[idx])
        param_list.append(params)
        model_list.append(model)

    # Set parameter properties
    for (params, prefix) in zip(param_list, prefixes):
        params[prefix + 'cen'].set(vary=False)
        params[prefix + 'amp'].set(min=amplitude/100, max=amplitude*1000)
        params[prefix + 'fwhm_km_s'].set(min=300, max=20000)
        params[prefix + 'shift_km_s'].set(vary=False, min=-200, max=200)

    return model_list, param_list


def setup_line_model_CIV_2G(prefix, **kwargs):
    """ Set up a 2 component Gaussian line model for the CIV emission line.

    This model is defined for a spectral dispersion axis in Angstroem.

    This setup models the broad CIV line emission as seen in type-I quasars
    and AGN. Due to the broad nature of the line CIV the CIV doublet is assumed
    to be unresolved.

    :param prefix: The input parameter exists for conformity with the \
        Sculptor models, but will be ignored. The prefix is automatically set \
        by the setup function.
    :type prefix: string
    :param kwargs:
    :return: Return a list of LMFIT models and a list of LMFIT parameters
    :rtype: (list, list)

    """

    redshift = kwargs.pop('redshift', 0)
    amplitude = kwargs.pop('amplitude', 1)

    # CIV 2G model params
    prefixes = ['CIV_A_', 'CIV_B_']
    amplitudes = np.array([1, 1]) * amplitude
    fwhms = [2500, 2500]
    cenwaves = [1549.06, 1549.06]  # Vanden Berk 2001
    vshifts = [0, 0]

    param_list = []
    model_list = []

    # Build parameter and model list
    for idx, prefix in enumerate(prefixes):

        model, params = setup_line_model_gaussian(prefix,
                                                  redshift=redshift,
                                                  amplitude=amplitudes[idx],
                                                  cenwave=cenwaves[idx],
                                                  fwhm=fwhms[idx],
                                                  vshift=vshifts[idx])
        param_list.append(params)
        model_list.append(model)

    # Set parameter properties
    for (params, prefix) in zip(param_list, prefixes):
        params[prefix + 'cen'].set(vary=False)
        params[prefix + 'amp'].set(min=amplitude/100, max=amplitude*1000)
        params[prefix + 'fwhm_km_s'].set(min=300, max=20000)
        params[prefix + 'shift_km_s'].set(vary=False, min=-200, max=200)

    return model_list, param_list


def setup_line_model_MgII_2G(prefix, **kwargs):
    """ Set up a 2 component Gaussian line model for the MgII emission line.

    This model is defined for a spectral dispersion axis in Angstroem.

    This setup models the broad MgII line emission as seen in type-I quasars
    and AGN. Due to the broad nature of the line MgII the MgII doublet is
    assumed to be unresolved.

    :param prefix: The input parameter exists for conformity with the \
        Sculptor models, but will be ignored. The prefix is automatically set \
        by the setup function.
    :type prefix: string
    :param kwargs:
    :return: Return a list of LMFIT models and a list of LMFIT parameters
    :rtype: (list, list)

    """

    redshift = kwargs.pop('redshift', 0)
    amplitude = kwargs.pop('amplitude', 5)

    prefixes = ['MgII_A_', 'MgII_B_']

    # MgII 2G model params
    fwhms = [2500, 1000]
    amplitudes = np.array([5, 2]) * amplitude
    cenwaves = [2798.75, 2798.75]  # Vanden Berk 2001
    vshifts = [0, 0]

    param_list = []
    model_list = []

    # Build parameter and model list
    for idx, prefix in enumerate(prefixes):
        model, params = setup_line_model_gaussian(prefix,
                                                   redshift=redshift,
                                                   amplitude=amplitudes[idx],
                                                   cenwave=cenwaves[idx],
                                                   fwhm=fwhms[idx],
                                                   vshift=vshifts[idx])
        param_list.append(params)
        model_list.append(model)

    # Set parameter properties
    for (params, prefix) in zip(param_list, prefixes):
        print(params, prefix)
        params[prefix + 'cen'].set(vary=False)
        params[prefix + 'amp'].set(min=amplitude/100, max=amplitude*1000)
        params[prefix + 'fwhm_km_s'].set(min=300, max=20000)
        params[prefix + 'shift_km_s'].set(vary=False, min=-200, max=200)

    return model_list, param_list


def setup_line_model_HbOIII_6G(prefix, **kwargs):
    """ Set up a 6 component Gaussian line model for the HBeta and [OIII]
    emission lines.

    This model is defined for a spectral dispersion axis in Angstroem.

    :param prefix: The input parameter exists for conformity with the \
        Sculptor models, but will be ignored. The prefix is automatically set \
        by the setup function.
    :type prefix: string
    :param kwargs:
    :return: Return a list of LMFIT models and a list of LMFIT parameters
    :rtype: (list, list)

    """

    redshift = kwargs.pop('redshift', 0)
    amplitude = kwargs.pop('amplitude', 5)

    prefixes = ['HBeta_A_', 'HBeta_B_', 'OIII4960_A_', 'OIII4960_B_',
                'OIII5008_A_', 'OIII5008_B_']

    # Hbeta + O[III] 6G model params
    amplitudes = np.array([4, 2, 1, 0.5, 2, 1]) * amplitude
    fwhms = [2500, 900, 900, 900, 1200, 1200]
    # Line wavelengths from Vanden Berk 2001 Table 2 (in Angstroem)
    cenwaves = [4862.68, 4862.68, 4960.30, 4960.30, 5008.24, 5008.24]
    vshifts = [0, 0, 0, 0, 0, 0]

    param_list = []
    model_list = []

    # Build parameter and model list
    for idx, prefix in enumerate(prefixes):
        model, params = setup_line_model_gaussian(prefix,
                                                   redshift=redshift,
                                                   amplitude=amplitudes[idx],
                                                   cenwave=cenwaves[idx],
                                                   fwhm=fwhms[idx],
                                                   vshift=vshifts[idx])
        param_list.append(params)
        model_list.append(model)

    # Set parameter properties
    for (params, prefix) in zip(param_list, prefixes):
        print(params, prefix)
        params[prefix + 'cen'].set(vary=False)
        params[prefix + 'amp'].set(min=amplitude / 100, max=amplitude * 1000)
        params[prefix + 'fwhm_km_s'].set(min=300, max=20000)
        params[prefix + 'shift_km_s'].set(vary=False, min=-200, max=200)

    return model_list, param_list


def setup_line_model_CIII_complex(prefix, **kwargs):
    """ Set up a 3 component Gaussian line model for the CIII], AlIII and
    SiIII] emission lines.

    Note that a special model function exists as all three Gaussian line
    models share a common redshift parameter.

    This model is defined for a spectral dispersion axis in Angstroem.

    :param prefix: The input parameter exists for conformity with the \
        Sculptor models, but will be ignored. The prefix is automatically set \
        by the setup function.
    :type prefix: string
    :param kwargs:
    :return:

    """
    redshift = kwargs.pop('redshift', 0)
    amplitude = kwargs.pop('amplitude', 1)

    prefix = 'CIII_'

    amp_cIII = 21.19 * amplitude
    amp_alIII = 0.4 * amplitude
    amp_siIII = 0.16 * amplitude

    params = Parameters()
    # Line wavelengths from Vanden Berk 2001 Table 2 (in Angstroem)
    params.add(prefix + 'cen', value=1908.73, vary=False)
    params.add(prefix + 'cen_siIII', value=1892.03, vary=False)
    params.add(prefix + 'cen_alIII', value=1857.40, vary=False)

    add_redshift_param(redshift, params, prefix)

    params.add(prefix + 'amp', value=amp_cIII, vary=True,
               min=amp_cIII / 100, max=amp_cIII * 1000)
    params.add(prefix + 'amp_alIII', value=amp_alIII, vary=True,
               min=amp_alIII / 100, max=amp_alIII * 1000)
    params.add(prefix + 'amp_siIII', value=amp_siIII, vary=True,
               min=amp_siIII / 100, max=amp_siIII * 1000)

    params.add(prefix + 'fwhm_km_s', value=2000, vary=True, min=500, max=1e+4)
    params.add(prefix + 'fwhm_km_s_alIII', value=800, vary=True, min=500,
               max=7e+3)
    params.add(prefix + 'fwhm_km_s_siIII', value=300, vary=True, min=500,
               max=7e+3)

    params.add(prefix + 'shift_km_s', value=0, vary=False)
    params.add(prefix + 'shift_km_s_alIII', value=0, vary=False)
    params.add(prefix + 'shift_km_s_siIII', value=0, vary=False)

    model = Model(CIII_complex_model_func, prefix='CIII_')

    return model, params

# ------------------------------------------------------------------------------
# Masks
# ------------------------------------------------------------------------------

# QSO continuum windows, see Vestergaard & Peterson 2006
qso_cont_VP06 = {'name': 'QSO Cont. VP06' ,
                 'rest_frame': True,
                 'mask_ranges':[[1265, 1290], [1340, 1375], [1425, 1470],
                                [1680, 1705], [1905, 2050]]}

# QSO continuum + iron windows, see Shen+2011
qso_contfe_MgII_Shen11 = {'name': 'QSO Cont. MgII Shen11',
                          'rest_frame': True,
                          'mask_ranges':[[2200, 2700], [2900, 3090]]}

qso_contfe_HBeta_Shen11 = {'name': 'QSO Cont. HBeta Shen11',
                           'rest_frame': True,
                           'mask_ranges':[[4435, 4700], [5100, 5535]]}

qso_contfe_HAlpha_Shen11 = {'name': 'QSO Cont. HAlpha Shen11',
                            'rest_frame': True,
                            'mask_ranges':[[6000, 6250], [6800, 7000]]}

qso_contfe_CIV_Shen11 = {'name': 'QSO Cont. CIV Shen11',
                         'rest_frame': True,
                         'mask_ranges':[[1445, 1465], [1700, 1705]]}

# Dictionary with all masks
mask_presets = {'QSO Cont.W. VP06': qso_cont_VP06,
                'QSO Fe+Cont.W. CIV Shen11': qso_contfe_CIV_Shen11,
                'QSO Fe+Cont.W. MgII Shen11': qso_contfe_MgII_Shen11,
                'QSO Fe+Cont.W. HBeta Shen11': qso_contfe_HBeta_Shen11,
                'QSO Fe+Cont.W. HAlpha Shen11': qso_contfe_HAlpha_Shen11}

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# Model function and setup lists
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------

# Dictionary listing all models for load/save functionality
model_func_dict = {'power_law_at_2500': power_law_at_2500,
                   'power_law_at_2500_plus_bc':
                       power_law_at_2500_plus_bc,
                   'power_law_at_2500_plus_fractional_bc':
                   power_law_at_2500_plus_fractional_bc,
                   'template_model': template_model,
                   'line_model_gaussian': line_model_gaussian,
                   'CIII_complex_model_func': CIII_complex_model_func}


# Model function name list and setup function list.
# Note that both lists need to contain the same number of elements
model_func_list = ['Power Law (2500A)',
                   'Power Law (2500A) + BC',
                   'Power Law (2500A) + BC (fractional)',
                   'Line model Gaussian',
                   'SiIV (2G components)',
                   'CIV (2G components)',
                   'MgII (2G components)',
                   'HBeta+[OIII] (6G components)',
                   'CIII] complex (3G components)'
                   ]

model_setup_list = [setup_power_law_at_2500,
                    setup_power_law_at_2500_plus_bc,
                    setup_power_law_at_2500_plus_fractional_bc,
                    setup_line_model_gaussian,
                    setup_line_model_SiIV_2G,
                    setup_line_model_CIV_2G,
                    setup_line_model_MgII_2G,
                    setup_line_model_HbOIII_6G,
                    setup_line_model_CIII_complex]

# Test if iron templates are present and then add the model and setup functions.
if os.path.isfile(datadir+'iron_templates/'+'Fe_UVtemplt_A.asc'):
    print('[INFO] FeII iron template of Vestergaard & Wilkes 2001 found. If '
          'you will be using these templates in your model fit and '
          'publication, please add the citation to the original work, '
          'ADS bibcode: 2001ApJS..134....1V')
    model_funcs = [
                   # 'FeII template (V01, cont)',
                   'FeII template 1200-2200 (VW01, cont)',
                   'FeII template 1200-2200 (VW01, split)',
                   'FeII template 2200-3500 (VW01, cont)',
                   'FeII template 2200-3500 (VW01, split)']
    model_setups = [
                    # setup_iron_template_VW01,
                    setup_iron_template_UV_VW01,
                    setup_split_iron_template_UV_VW01,
                    setup_iron_template_MgII_VW01,
                    setup_split_iron_template_MgII_VW01
        ]
    model_func_list.extend(model_funcs)
    model_setup_list.extend(model_setups)
else:
    print('[WARNING] FeII iron template of Vestergaard & Wilkes 2001 NOT '
          'found. If you want to use the pre-defined iron models, please '
          'contact Marianne Vestergaard at mvester@nbi.ku.dk and add the '
          'Fe_UVtemplt_A.asc file to the sculptor/data/iron_templates folder.')
# if os.path.isfile(datadir+'iron_templates/'+'Fe3UV34modelB2.asc'):
#     print('[INFO] FeIII iron template of Vestergaard & Wilkes 2001 found. If '
#           'you will be using these templates in your model fit and '
#           'publication, please add the citation to the original work, '
#           'ADS bibcode: 2001ApJS..134....1V')
#     model_func_list.extend()
# else:
#     print('[WARNING] FeIII iron template of Vestergaard & Wilkes 2001 NOT '
#           'found. If you want to use the pre-defined iron models, please '
#           'contact Marianne Vestergaard at mvester@nbi.ku.dk and add the '
#           'Fe3UV34modelB2.asc file to the sculptor/data/iron_templates folder.')

if os.path.isfile(datadir+'iron_templates/'+'Tsuzuki06.txt'):
    print('[INFO] FeII iron template of Tsuzuki et al. 2006 found. If '
          'you will be using these templates in your model fit and '
          'publication, please add the citation to the original work, '
          'ADS bibcode: 2006ApJ...650...57T')

    model_funcs = ['FeII template 2200-3500 (T06, cont)',
                   'FeII template 2200-3500 (T06, split)']
    model_setups = [setup_iron_template_MgII_T06,
                    setup_split_iron_template_MgII_T06]
    model_func_list.extend(model_funcs)
    model_setup_list.extend(model_setups)
else:
    print('[WARNING] FeII iron template of Tsuzuki et al. 2006 NOT found. '
          'The template should come with Sculptor. Please contact the '
          'Sculptor developer through github.')

if os.path.isfile(datadir+'iron_templates/'+'Fe_OPT_BR92_linear.txt'):
    print('[INFO] FeII iron template of Boroson & Green 1992 found. If '
          'you will be using these templates in your model fit and '
          'publication, please add the citation to the original work, '
          'ADS bibcode: 1992ApJS...80..109B')
    model_funcs = ['FeII template 3700-7480 (BG92, cont)',
                  'FeII template 3700-5600 (BG92, split)']
    model_setups = [setup_iron_template_OPT_BG92,
                    setup_split_iron_template_OPT_BG92]
    model_func_list.extend(model_funcs)
    model_setup_list.extend(model_setups)
else:
    print('[WARNING] FeII iron template of Boroson & Green 1992 NOT found. '
          'The template should come with Sculptor. Please contact the '
          'Sculptor developer through github.')


# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# Analysis routines
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------


def calc_velocity_shifts(z, z_sys, relativistic=True):
    """Calculate the velocity difference of a feature redshift with respect
    to the systemic redshift.

    This function is currently simply a wrapper around the linetools
    functions calculating the velocity difference.

    :param z: The redshift of the spectroscopic feature (e.g., absorption or \
        emission line).
    :type z: float
    :param z_sys: The systemic redshift
    :type z_sys: float
    :param relativistic: Boolean indicating whether the doppler velocity is \
        calculated assuming relativistic velocities.
    :type: bool
    :return: Returns the velocity difference in km/s.
    :rtype: u.Quantity

    """

    return lu.dv_from_z(z, z_sys, rel=relativistic)


def calc_eddington_luminosity(bh_mass):
    """Calculate the Eddington luminosity for a given black hole mass.

    :param bh_mass: Black hole mass as an astropy Quantity
    :type bh_mass: u.Quantity
    :return: Returns the Eddington luminosity in cgs units (erg/s).
    :rtype: u.Quantity

    """

    # Convert the black hole mass to solar masses
    bh_mass = bh_mass.to(u.Msun)

    factor = (4 * np.pi * const.G * const.c * const.m_p) / const.sigma_T
    factor = factor.to(units.erg / units.s / units.Msun)

    return factor * bh_mass


def calc_eddington_ratio(lbol, bh_mass):
    """ Calculate the Eddington ratio for a provided bolometric luminosity
    and black hole mass.

    :param lbol: Bolometric luminosity
    :type lbol: u.Quantity
    :param bh_mass: Black hole mass
    :type bh_mass: u.Quantity
    :return:

    """

    edd_lum = calc_Edd_luminosity(bh_mass)
    return lbol / edd_lum


def se_bhmass_hbeta_vp06(hbeta_fwhm, cont_lwav, cont_wav=5100*u.A):
    """Calculate the single-epoch virial BH mass based on the Hbeta FWHM and
    monochromatic continuum luminosity at 5100A.

    This relationship is taken from Vestergaard & Peterson 2006, ApJ 641, 689

    Note that the Hbeta line width to establish this single-epoch virial
    estimator was established by using the FWHM of only the broad component.
    The line width was corrected for the spectral resolution as described in
    Peterson et al. 2004.

    The relationship is based on line width measurements of quasars published in
    Boroson & Green 1992 and Marziani 2003.

    "The sample standard deviation of the weighted average zero point offset,
    which shows the intrinsic scatter in the saample is +-0.43 dex. This
    value is more representative of the uncertainty zer point than is the
    formal error."

    :param hbeta_fwhm: FWHM of the Hbeta line in km/s
    :type hbeta_fwhm: astropy.units.Quantity
    :param cont_lwav: Monochromatic continuum luminosity at 5100A in erg/s/A
    :type cont_lwav: astropy.units.Quantity
    :param cont_wav: Wavelength of the monochromatic continuum luminosity \
        (default = 5100A).
    :type cont_wav: astropy.units.Quantity
    :return: Returns a tuple of the BH mass estimate based on the Hbeta FWHM \
        and a reference string for the single-epoch scaling relationship.
    :rtype: astropy.units.Quantity, string

    """

    if cont_wav.value == 5100:

        reference = 'Hbeta_VP06_fwhm'

        bhmass = 10 ** 6.91 * (hbeta_fwhm / (1000.*u.km/u.s))**2 \
                 * (cont_wav * cont_lwav / (10**44*u.erg/u.s))**0.5 * u.Msun

        return bhmass, reference

    else:
        raise ValueError('[ERROR] The monochromatic continuum luminosity '
                         'should be estimated at 5100A.')


def se_bhmass_civ_vp06_fwhm(civ_fwhm, cont_lwav, cont_wav):
    """Calculate the single-epoch virial BH mass based on the CIV FWHM and
    monochromatic continuum luminosity at 1350A.

    The monochromatic continuum luminosity at 1450A can be used without error or
    penalty in accuracy.

    This relationship is taken from Vestergaard & Peterson 2006, ApJ 641, 689

    The FWHM of the CIV line was measured with the methodology described in
    Peterson et al. 2004. The line width measurements to establish the CIV
    single-epoch relation are corrected for resolution effects as described in
    Peterson et al. 2004.

    "The sample standard deviation of the weighted average zero point offset,
    which shows the intrinsic scatter in the saample is +-0.36 dex. This
    value is more representative of the uncertainty zer point than is the
    formal error."

    :param civ_fwhm: FWHM of the CIV line in km/s
    :type civ_fwhm: astropy.units.Quantity
    :param cont_lwav: Monochromatic continuum luminosity at 1350A/1450A in \
        erg/s/A.
    :type cont_lwav: astropy.units.Quantity
    :param cont_wav: Wavelength of the monochromatic continuum luminosity in A.
    :type cont_wav: astropy.units.Quantity
    :return: Returns a tuple of the BH mass estimate based on the CIV FWHM \
        and a reference string for the single-epoch scaling relationship.
    :rtype: astropy.units.Quantity, string

    """

    if cont_wav.value == 1450 or cont_wav.value == 1350:

        reference = 'CIV_VP06_fwhm'

        bhmass = 10**6.66 * (civ_fwhm / (1000.*u.kms/u.s))**2 \
                 * (cont_wav * cont_lwav / (10**44*u.erg/u.s))**0.53 * u.Msun

        return bhmass, reference

    else:
        raise ValueError('[ERROR] The monochromatic continuum luminosity '
                         'should be estimated at either 1350A or 1450A.')


def se_bhmass_civ_vp06_sigma(civ_sigma, cont_lwav, cont_wav):
    """Calculate the single-epoch virial BH mass based on the CIV line
    dispersion (sigma) and monochromatic continuum luminosity at 1350A.

    The monochromatic continuum luminosity at 1450A can be used without error or
    penalty in accuracy.

    This relationship is taken from Vestergaard & Peterson 2006, ApJ 641, 689

    The FWHM of the CIV line was measured with the methodology described in
    Peterson et al. 2004. The line width measurements to establish the CIV
    single-epoch relation are corrected for resolution effects as described in
    Peterson et al. 2004.

    Peterson et al. (2004) note a number of advantages to using sigma rathern
    than the FWHM as the line width measure.

    "The sample standard deviation of the weighted average zero point offset,
    which shows the intrinsic scatter in the saample is +-0.33 dex. This
    value is more representative of the uncertainty zer point than is the
    formal error."

    :param civ_sigma: Line dispersion (sigma) of the CIV line in km/s
    :type civ_sigma: astropy.units.Quantity
    :param cont_lwav: Monochromatic continuum luminosity at 1350A/1450A in \
        erg/s/A.
    :type cont_lwav: astropy.units.Quantity
    :param cont_wav: Wavelength of the monochromatic continuum luminosity in A.
    :type cont_wav: astropy.units.Quantity
    :return: Returns a tuple of the BH mass estimate based on the CIV sigma \
        and a reference string for the single-epoch scaling relationship.
    :rtype: astropy.units.Quantity, string

    """

    if cont_wav.value == 1450 or cont_wav.value == 1350:

        reference = 'CIV_VP06_sigma'

        bhmass = 10**6.73 * (civ_sigma / (1000.*u.kms/u.s))**2 \
                 * (cont_wav * cont_lwav / (10**44*u.erg/u.s))**0.53 * u.Msun

        return bhmass, reference

    else:
        raise ValueError('[ERROR] The monochromatic continuum luminosity '
                         'should be estimated at either 1350A or 1450A.')


def se_bhmass_mgii_vo09_fwhm(mgii_fwhm, cont_lwav, cont_wav):
    """Calculate the single-epoch virial BH mass based on the MgII FWHM
    and monochromatic continuum luminosity at 1350A, 2100A, 3000A, or 5100A.

    This relationship is taken from Vestergaard & Osmer 2009, ApJ 641, 689

    To determine the FWHM of the MgII line, the authors modeled the FeII
    emission beneath the MgII line with the Vestergaard & Wilkes 2001 and the
    Boroson & Green 1992 iron templates.

    Most of the MgII lines were modeled with a single Gaussian component,
    in cases of high-quality spectra two Gaussian components were used. For
    the single-Gaussian components the authors adopted the measurements of
    the FWHM and uncertainties tabulated by Forster et al. (their Table 5).
    For multi-Gaussian components the FWHM of the full modeled profile was
    measured.

    As the subtraction of the underlying FeII continuum can have systematic
    effects on the measurement of the MgII FWHM and therefore the BH mass
    estimate it is advised to always employ the same continuum construction
    procedure as in the reference sample that established the single-epoch
    virial relationship.

    "The absolute 1 sigma scatter in the zero points is 0.55dex,
    which includes the factor ~2.9 uncertainties of the reverberation mapping
    masses to which these mass estimation relations are anchored (see
    Vestergaard & Peterson 2006 and Onken et al. 2004 for details)"

    :param mgii_fwhm: FWHM of the MgII line in km/s
    :type mgii_fwhm: astropy.units.Quantity
    :param cont_lwav: Monochromatic continuum luminosity in erg/s/A.
    :type cont_lwav: astropy.units.Quantity
    :param cont_wav: Wavelength of the monochromatic continuum luminosity in A.
    :type cont_wav: astropy.units.Quantity
    :return: Returns a tuple of the BH mass estimate based on the MgII FWHM \
        and a reference string for the single-epoch scaling relationship.
    :rtype: astropy.units.Quantity, string

    """

    zero_points = {1350: 6.72, 2100: 6.79, 3000: 6.86, 5100: 6.96}

    if cont_wav.value in [1350, 2100, 3000, 5100]:

        reference = 'MgII_VO09_fwhm_{}'.format(cont_wav.value)

        zp = zero_points[cont_wav.value]

        bhmass = 10**zp * (mgii_fwhm / (1000.*u.km/u.s))**2 \
                 * (cont_wav * cont_lwav / (10**44*u.erg/u.s))**0.5 * u.Msun

        return bhmass, reference

    else:
        raise ValueError('[ERROR] The monochromatic continuum luminosity '
                         'should be estimated at 1350A, 2100A, 3000A, or '
                         '5100A.')


def se_bhmass_mgii_s11_fwhm(mgii_fwhm, cont_lwav, cont_wav):
    """Calculate the single-epoch virial BH mass based on the MgII FWHM
    and monochromatic continuum luminosity at 3000A.

    This relationship is taken from Shen et al. 2011, ApJ, 194, 45

    To model the FeII contribution beneath MgII line the authors use
    empirical FeII templates from Borosn & Grenn 1992, Vestergaard & Wilkes
    2001, and Salviander 2007.

    Salviander modified the Vestergaard & Wilkes (2001) template in the
    region of 2780–2830A centered on MgII, where the Vestergaard & Wilkes (2001)
    template is set to zero. For this region, they incorporate a theoretical
    FeII model ofSigut & Pradhan (2003) scaled to match the Vestergaard &
    Wilkes(2001) template at neighboring wavelengths.

    As the subtraction of the underlying FeII continuum can have systematic
    effects on the measurement of the MgII FWHM and therefore the BH mass
    estimate it is advised to always employ the same continuum construction
    procedure as in the reference sample that established the single-epoch
    virial relationship.

    :param mgii_fwhm: FWHM of the MgII line in km/s
    :type mgii_fwhm: astropy.units.Quantity
    :param cont_lwav: Monochromatic continuum luminosity in erg/s/A.
    :type cont_lwav: astropy.units.Quantity
    :param cont_wav: Wavelength of the monochromatic continuum luminosity in A.
    :type cont_wav: astropy.units.Quantity
    :return: Returns a tuple of the BH mass estimate based on the MgII FWHM \
        and a reference string for the single-epoch scaling relationship.
    :rtype: astropy.units.Quantity, string

    """


    if cont_wav.value == 3000:

        reference = 'MgII_S11_fwhm_3000'

        zp = 6.74

        bhmass = 10**zp * (mgii_fwhm / (1000.*u.km/u.s))**2 \
                 * (cont_wav * cont_lwav / (10**44*u.erg/u.s))**0.62 * u.Msun

        return bhmass, reference

    else:
        raise ValueError('[ERROR] The monochromatic continuum luminosity '
                         'should be estimated at 3000A.')


def calc_bolometric_luminosity(cont_lwav, cont_wav, reference='Shen2011'):
    """Calculate the bolometric luminosity from the monochromatic continuum
    luminosity (erg/s/A) using bolometric correction factors from the
    literature.

    The following bolometric corrections are available
    cont_wav = 1350, reference = Shen2011
    cont_wav = 3000, reference = Shen2011
    cont_wav = 5100, reference = Shen2011

    The Shen et al. 2011 (ApJ, 194, 45) bolometric corrections are based on the
    composite spectral energy distribution (SED) in Richards et al. 2006 (
    ApJ, 166,470).

    :param cont_lwav: Monochromatic continuum luminosity in erg/s/A.
    :type cont_lwav: astropy.units.Quantity
    :param cont_wav: Wavelength of the monochromatic continuum luminosity in A.
    :type cont_wav: astropy.units.Quantity
    :param reference: A reference string to select from the available \
        bolometric corrections.
    :type reference: string
    :return: Returns a tuple of the bolometric luminosity in erg/s and a \
        reference string indicating the publication and continuum wavelength \
        of the bolometric correction.
    :rtype: astropy.units.Quantity, string

    """

    if cont_wav.value == 1350 and reference == 'Shen2011':
        reference = 'Shen2011_1350'
        return cont_lwav * cont_wav * 3.81, reference

    if cont_wav.value == 3000 and reference == 'Shen2011':
        reference = 'Shen2011_3000'
        return cont_lwav * cont_wav * 5.15, reference

    if cont_wav.value == 5100 and reference == 'Shen2011':
        reference = 'Shen2011_5100'
        return cont_lwav * cont_wav * 9.26, reference
    else:
        raise ValueError('[ERROR] No bolometric correction available for the '
                         'supplied combination of continuum wavelength and '
                         'reference.')


def correct_CIV_fwhm_for_blueshift(civ_fwhm, blueshift):
    """Correct the CIV FWHM for the CIV blueshifts using the relation
    determined in Coatman et al. (2017, MNRAS, 465, 2120).

    The correction follows Eq. 4 of Coatman et al. (2017).

    :param civ_fwhm: FWHM of the CIV line in km/s
    :type civ_fwhm: astropy.units.Quantity
    :param blueshift: Blueshift of the CIV line in km/s
    :type blueshift: astropy.units.Quantity
    :return: Corrected FWHM in km/s
    :rtype: astropy.units.Quantity

    """

    return civ_fwhm / (0.41 * blueshift/(1000*u.km/u.s) + 0.62)


def se_bhmass_civ_c17_fwhm(civ_fwhm, cont_lwav, cont_wav):
    """Calculate the single-epoch virial BH mass based on the CIV FWHM and
    monochromatic continuum luminosity at 1350A.

    This relationship follows Eq.6 of from Coatman et al. (2017, MNRAS, 465,
    2120)

    The FWHM of the CIV line was corrected by the CIV blueshift using their
    Eq. 4. In this study the CIV line was modeled with sixth order
    Gauss-Hermite (GH) polynomials using the normalization of van der Marel
    & Franx (1993) and the functional forms of Cappelari et al. (2002). GH
    polynomials were chose because they are flexble enough to model the often
    very asymmetric CIV line profile.

    The authors state that using commonly employed three Gaussian components,
    rather than the GH polynomials, resulted in only marginal differences in
    the line parameters.

    :param civ_fwhm: FWHM of the CIV line in km/s
    :type civ_fwhm: astropy.units.Quantity
    :param cont_lwav: Monochromatic continuum luminosity at 1350A in erg/s/A.
    :type cont_lwav: astropy.units.Quantity
    :param cont_wav: Wavelength of the monochromatic continuum luminosity in A.
    :type cont_wav: astropy.units.Quantity
    :return: Returns a tuple of the BH mass estimate based on the CIV FWHM \
        and a reference string for the single-epoch scaling relationship.
    :rtype: astropy.units.Quantity, string

    """

    if cont_wav.value == 1350:

        reference = 'CIV_C17_fwhm'

        bhmass = 10 ** 6.71 * (civ_fwhm / (1000.*u.kms/u.s))**2 \
                 * (cont_wav * cont_lwav / (10**44*u.erg/u.s))**0.53 * u.Msun

        return bhmass, reference

    else:
        raise ValueError('[ERROR] The monochromatic continuum luminosity '
                         'should be estimated at 1350A.')

