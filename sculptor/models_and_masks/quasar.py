#!/usr/bin/env python

import os
import numpy as np
import scipy as sp

from astropy import constants as const
from astropy.modeling.blackbody import blackbody_lambda

from astrotools.speconed import speconed as sod

from lmfit import Model, Parameters


datadir = os.path.split(__file__)[0]
datadir = os.path.split(datadir)[0] + '/data/'

c_km_s = const.c.to('km/s').value


def add_redshift_param(redshift, params, prefix):

    z_min = min(redshift*1.05, 1080)
    z_max = max(1.0, redshift * 0.95)

    params.add(prefix+'z', value=redshift, min=z_min, max=z_max, vary=False)

# ------------------------------------------------------------------------------
# Model functions
# ------------------------------------------------------------------------------

def power_law_at_2500A(x, amp, slope, z):
    """ Power law anchored at 2500 (Angstroem)

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


def balmer_continuum_model(x, z, flux_BE, T_e, tau_BE, lambda_BE):
    """

    :param x:
    :param z:
    :param flux_BE:
    :param T_e:
    :param tau_BE:
    :param lambda_BE:
    :return:
    """

    # Dietrich 2003
    # lambda <= 3646A, flux_BE = normalized estimate for Balmer continuum
    # The strength of the Balmer continuum can be estimated from the flux
    # density at 3675A after subtraction of the power-law continuum component
    # for reference see Grandi82, Wills85 or Verner99
    # at >= 3646A higher order balmer lines are merging  -> templates Dietrich03

    # The Balmer continuum does currently not include any blended high order
    # Balmer lines this has to be DONE!!!

    x = x / (1.+z)

    flux = flux_BE * blackbody_lambda(x, T_e).value * (
                1. - np.exp(-tau_BE * (x / lambda_BE) ** 3))

    flux[x >= lambda_BE] = flux[x>= lambda_BE] * 0

    flux *= 1e-20

    return flux


def power_law_at_2500A_plus_flexible_BC(x, amp, slope, z, f, T_e, tau_BE,
                                       lambda_BE):
    """
    Power law anchored at 2500 (Angstroem) plus a Balmer continuum model with
    a fixed flux of 30% of the power law flux at 3645A.
    :param x:
    :param amp:
    :param slope:
    :param z:
    :param T_e:
    :param tau_BE:
    :param lambda_BE:
    :return:
    """

    pl_flux =  amp * (x / (2500. * (z + 1.))) ** slope

    pl_flux_at_BE = amp * ((lambda_BE * (z + 1.)) / (2500. * (z + 1.))) ** slope

    x = x / (1. + z)

    bc_flux_at_lambda_BE = blackbody_lambda(lambda_BE, T_e).value * (
                1. - np.exp(-tau_BE))

    bc_flux = f * pl_flux_at_BE / bc_flux_at_lambda_BE * \
              blackbody_lambda(x, T_e).value * (
                1. - np.exp(-tau_BE * (x / lambda_BE) ** 3))

    bc_flux[x >= lambda_BE] = bc_flux[x >= lambda_BE] * 0

    return pl_flux + bc_flux


def power_law_at_2500A_plus_manual_BC(x, amp, slope, z, amp_BE, T_e, tau_BE,
                                       lambda_BE):
    """
    Power law anchored at 2500 (Angstroem) plus a Balmer continuum model with
    a fixed flux of 30% of the power law flux at 3645A.
    :param x:
    :param amp:
    :param slope:
    :param z:
    :param T_e:
    :param tau_BE:
    :param lambda_BE:
    :return:
    """



    pl_flux =  amp * (x / (2500. * (z + 1.))) ** slope

    # pl_flux_at_BE = amp * ((lambda_BE * (z + 1.)) / (2500. * (z + 1.))) ** slope


    x = x / (1. + z)

    F_BC0 = amp_BE /(blackbody_lambda(lambda_BE, T_e).value * (
                1. - np.exp(-tau_BE)))

    bc_flux = F_BC0 * \
              blackbody_lambda(x, T_e).value * (
                1. - np.exp(-tau_BE * (x / lambda_BE) ** 3))

    bc_flux[x >= lambda_BE] = bc_flux[x >= lambda_BE] * 0

    return pl_flux + bc_flux


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



def template_model(x, amp, z, fwhm, intr_fwhm, templ_disp=None,
                   templ_flux=None):

    template_spec = sod.SpecOneD(dispersion=templ_disp, flux=templ_flux,
                                 unit='f_lam')
    # artifical broadening
    convol_fwhm = np.sqrt(fwhm**2-intr_fwhm**2)
    spec = template_spec.convolve_loglam(convol_fwhm)
    # shift in redshift
    spec = spec.redshift(z)
    # return interpolation function
    f = sp.interpolate.interp1d(spec.dispersion, spec.flux, kind='linear',
                                bounds_error=False, fill_value=(0, 0))

    return f(x)*amp



def CIII_complex_model_func(x, z, cen, cen_alIII, cen_siIII,
                          amp, fwhm_km_s, shift_km_s,
                          amp_alIII, fwhm_km_s_alIII, shift_km_s_alIII,
                          amp_siIII, fwhm_km_s_siIII, shift_km_s_siIII):
    """

    :param x: ndarray
         x-Axis of the Gaussian
    :param z:  float
        Redshift for CIII], AlIII, SiIII]
    :param amp_cIII: float
        Amplitude of the Gaussian
    :param fwhm_km_s_cIII: float
        Full Width at Half Maximum (FWHM) of the Gaussian in km/s
    :param shift_km_s_cIII: float
        x-Axis shift of the Gaussian in km/s
    :param amp_alIII: float
        Amplitude of the Gaussian
    :param fwhm_km_s_alIII: float
        Full Width at Half Maximum (FWHM) of the Gaussian in km/s
    :param shift_km_s_alIII: float
        x-Axis shift of the Gaussian in km/s
    :param amp_siIII: float
        Amplitude of the Gaussian
    :param fwhm_km_s_siIII: float
        Full Width at Half Maximum (FWHM) of the Gaussian in km/s
    :param shift_km_s_siIII: float
            x-Axis shift of the Gaussian in km/s
    :return: float
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
    gauss_cIII = (amp / (np.sqrt(2 * np.pi) * sigma)) * np.exp(
        -(x - central) ** 2 / (2 * sigma ** 2))

    central = siIII_cen + siIII_delta_cen
    fwhm = fwhm_km_s_siIII / c_km_s * central
    sigma = fwhm / np.sqrt(8 * np.log(2))
    gauss_siIII = (amp_siIII / (np.sqrt(2 * np.pi) * sigma)) * np.exp(
        -(x - central) ** 2 / (2 * sigma ** 2))

    central = alIII_cen + alIII_delta_cen
    fwhm = fwhm_km_s_alIII / c_km_s * central
    sigma = fwhm / np.sqrt(8 * np.log(2))
    gauss_alIII = (amp_alIII / (np.sqrt(2 * np.pi) * sigma)) * np.exp(
        -(x - central) ** 2 / (2 * sigma ** 2))

    return gauss_cIII + gauss_alIII + gauss_siIII


# ------------------------------------------------------------------------------
# Model setups
# ------------------------------------------------------------------------------


def setup_power_law_at_2500A(prefix, **kwargs):

    redshift = kwargs.pop('redshift', 0)
    amplitude = kwargs.pop('amplitude', 1e-17)

    params = Parameters()

    add_redshift_param(redshift, params, prefix)

    params.add(prefix + 'amp', value=amplitude)
    params.add(prefix + 'slope', value=-1.5, min=-2.5, max=-0.3)

    model = Model(power_law_at_2500A, prefix=prefix)

    return model, params


def setup_balmer_continuum_model():
    pass


def setup_power_law_at_2500A_plus_flexible_BC(prefix, **kwargs):

    redshift = kwargs.pop('redshift', 0)
    amplitude = kwargs.pop('amplitude', 1e-17)

    params = Parameters()
    add_redshift_param(redshift, params, prefix)
    params.add(prefix + 'amp', value=amplitude)
    params.add(prefix + 'slope', value=-1.5, min=-2.5, max=-0.3)
    params.add(prefix + 'f', value=0.1, min=0.0, max=0.4, vary=True)
    params.add(prefix + 'T_e', value=15000, min=10000, max=20000, vary=False)
    params.add(prefix + 'tau_BE', value=1.0, min=0.1, max=2.0, vary=False)
    params.add(prefix + 'lambda_BE', value=3646, vary=False)

    model = Model(power_law_at_2500A_plus_flexible_BC, prefix=prefix)

    return model, params


def setup_power_law_at_2500A_plus_manual_BC(prefix, **kwargs):

    redshift = kwargs.pop('redshift', 0)
    amplitude = kwargs.pop('amplitude', 1e-17)

    params = Parameters()
    add_redshift_param(redshift, params, prefix)

    params.add(prefix + 'amp', value=amplitude)
    params.add(prefix + 'slope', value=-1.5, min=-2.5, max=-0.3)
    params.add(prefix + 'amp_BE', value=1e-17, min=0.0, max=1e-12, vary=True)
    params.add(prefix + 'T_e', value=15000, min=10000, max=20000, vary=False)
    params.add(prefix + 'tau_BE', value=1.0, min=0.1, max=2.0, vary=False)
    params.add(prefix + 'lambda_BE', value=3646, vary=False)

    model = Model(power_law_at_2500A_plus_manual_BC, prefix=prefix)

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



def setup_template_model(template_filename=None, fwhm=2500, redshift=0,
                        prefix=None, amplitude=1e-4,
                         intr_fwhm=900, wav_limits=None):

    params = Parameters()

    template = np.genfromtxt(datadir+'iron_templates/'+template_filename)

    # Set wavelength limits
    if wav_limits is not None:
        wav_min = wav_limits[0]
        wav_max = wav_limits[1]

        idx_min = np.argmin(np.abs(template[:, 0] - wav_min))
        idx_max = np.argmin(np.abs(template[:, 0] - wav_max))

        model = Model(template_model,
                      templ_disp=template[idx_min:idx_max, 0],
                      templ_flux=template[idx_min:idx_max, 1],
                      prefix=prefix)

    else:
        model = Model(template_model,
                      templ_disp=template[:, 0],
                      templ_flux=template[:, 1],
                      prefix=prefix)

    add_redshift_param(redshift, params, prefix)
    params.add(prefix + 'fwhm', value=fwhm, min=0, max=10000, vary=False)
    params.add(prefix + 'amp', value=amplitude, min=1.0e-10, max=1.0)
    params.add(prefix + 'intr_fwhm', value=intr_fwhm, vary=False)

    return model, params



def setup_subdivided_iron_template(fwhm=None, redshift=None, amplitude=None,
                             templ_list=[], intr_fwhm=900):

    param_list = []
    model_list = []
    if 'UV01' in templ_list or not templ_list:
        # 1200-1560 Vestergaard 2001
        model, params = setup_template_model(
            template_filename='Fe_UVtemplt_A.asc', fwhm=fwhm, redshift=redshift,
            prefix='UV01_', amplitude=amplitude, wav_limits=[1200, 1560],
            intr_fwhm=intr_fwhm)
        param_list.append(params)
        model_list.append(model)
    if 'UV02' in templ_list or not templ_list:
        # 1560-1875 Vestergaard 2001
        model, params = setup_template_model(
            template_filename='Fe_UVtemplt_A.asc', fwhm=fwhm, redshift=redshift,
            prefix='UV02_', amplitude=amplitude, wav_limits=[1560, 1875],
            intr_fwhm=intr_fwhm)
        param_list.append(params)
        model_list.append(model)
    if 'UV03' in templ_list or not templ_list:
        # 1875-2200 Vestergaard 2001
        model, params = setup_template_model(
            template_filename='Fe_UVtemplt_A.asc', fwhm=fwhm, redshift=redshift,
            prefix='UV03_', amplitude=amplitude, wav_limits=[1875, 2200],
            intr_fwhm=intr_fwhm)
        param_list.append(params)
        model_list.append(model)
    if 'FeIIIUV34' in templ_list or not templ_list:
        # Vestergaard 2001 - FeIII UV34 template
        model, params = setup_template_model(
            template_filename='Fe3UV34modelB2.asc', fwhm=fwhm, redshift=redshift,
            prefix='FeIIIUV34_', amplitude=amplitude, wav_limits=[1875,2200],
            intr_fwhm=intr_fwhm)
        param_list.append(params)
        model_list.append(model)
    if 'UV04_T06' in templ_list or not templ_list:
        # 2200-2660 Tsuzuki 2006
        model, params = setup_template_model(
            template_filename='Tsuzuki06.txt', fwhm=fwhm, redshift=redshift,
            prefix='UV04_T06_', amplitude=amplitude, wav_limits=[2200, 2660],
            intr_fwhm=intr_fwhm)
        param_list.append(params)
        model_list.append(model)
    if 'UV05_T06' in templ_list or not templ_list:
        # 2660-3000 Tsuzuki 2006
        model, params = setup_template_model(
            template_filename='Tsuzuki06.txt', fwhm=fwhm, redshift=redshift,
            prefix='UV05_T06_', amplitude=amplitude, wav_limits=[2660, 3000],
            intr_fwhm=intr_fwhm)
        param_list.append(params)
        model_list.append(model)
    if 'UV06_T06' in templ_list or not templ_list:
        # 3000-3500 Tsuzuki 2006
        model, params = setup_template_model(
            template_filename='Tsuzuki06.txt', fwhm=fwhm, redshift=redshift,
            prefix='UV06_T06_', amplitude=amplitude, wav_limits=[3000, 3500],
            intr_fwhm=intr_fwhm)
        param_list.append(params)
        model_list.append(model)
    if 'OPT01' in templ_list or not templ_list:
        # 4400-4700 Boronson & Green 1992
        model, params = setup_template_model(
            template_filename='Fe_OPT_BR92_linear.txt', fwhm=fwhm,
            redshift=redshift,
            prefix='OPT01_', amplitude=amplitude, wav_limits=[3700, 4700],
            intr_fwhm=intr_fwhm)
        param_list.append(params)
        model_list.append(model)
    if 'OPT02' in templ_list or not templ_list:
        # 4700-5100 Boronson & Green 1992
        model, params = setup_template_model(
            template_filename='Fe_OPT_BR92_linear.txt', fwhm=fwhm,
            redshift=redshift,
            prefix='OPT02_', amplitude=amplitude, wav_limits=[4700, 5100],
            intr_fwhm=intr_fwhm)
        param_list.append(params)
        model_list.append(model)
    if 'OPT03' in templ_list or not templ_list:
        # 5100-5600 Boronson & Green 1992
        model, params = setup_template_model(
            template_filename='Fe_OPT_BR92_linear.txt', fwhm=fwhm,
            redshift=redshift,
            prefix='OPT03_', amplitude=amplitude, wav_limits=[5100, 5600],
            intr_fwhm=intr_fwhm)
        param_list.append(params)
        model_list.append(model)
    # Vestergaard templates
    if 'UV04_V01' in templ_list or not templ_list:
        # 2200-2660 Vestergaard 2001
        model, params = setup_template_model(
            template_filename='Fe_UVtemplt_A.asc', fwhm=fwhm, redshift=redshift,
            prefix='UV04V01_', amplitude=amplitude, wav_limits=[2200, 2660],
            intr_fwhm=intr_fwhm)
        param_list.append(params)
        model_list.append(model)
    if 'UV05_V01' in templ_list or not templ_list:
        # 2660-3000 Vestergaard 2001
        model, params = setup_template_model(
            template_filename='Fe_UVtemplt_A.asc', fwhm=fwhm, redshift=redshift,
            prefix='UV05V01_', amplitude=amplitude, wav_limits=[2660, 3000],
            intr_fwhm=intr_fwhm)
        param_list.append(params)
        model_list.append(model)
    if 'UV06_V01' in templ_list or not templ_list:
        # 3000-3500 Vestergaard 2001
        model, params = setup_template_model(
            template_filename='Fe_UVtemplt_A.asc', fwhm=fwhm, redshift=redshift,
            prefix='UV06V01_', amplitude=amplitude, wav_limits=[3000, 3500],
            intr_fwhm=intr_fwhm)

        param_list.append(params)
        model_list.append(model)

    return model_list, param_list

def setup_iron_template_V01(prefix, **kwargs):

    redshift = kwargs.pop('redshift', 0)
    amplitude = kwargs.pop('amplitude', 1e-4)
    intr_fwhm = kwargs.pop('intr_fwhm', 900)
    # 1200-2200 Vestergaard 2001
    model, params = setup_template_model(
        template_filename='Fe_UVtemplt_A.asc', fwhm=3500, redshift=redshift,
        prefix='FeIIV01_', amplitude=amplitude,
            intr_fwhm=intr_fwhm)

    return model, params

def setup_iron_template_T06(prefix, **kwargs):

    redshift = kwargs.pop('redshift', 0)
    amplitude = kwargs.pop('amplitude', 1e-4)
    intr_fwhm = kwargs.pop('intr_fwhm', 900)
    # 1200-2200 Vestergaard 2001
    model, params = setup_template_model(
        template_filename='Tsuzuki06.txt', fwhm=3500, redshift=redshift,
        prefix='FeIIT06_', amplitude=amplitude,
            intr_fwhm=intr_fwhm)

    return model, params

def setup_iron_template_MgII_T06(prefix, **kwargs):

    redshift = kwargs.pop('redshift', 0)
    amplitude = kwargs.pop('amplitude', 1e-4)
    intr_fwhm = kwargs.pop('intr_fwhm', 900)
    # 2200-3500 Tsuzuki 2006
    model, params = setup_template_model(
        template_filename='Tsuzuki06.txt', fwhm=2500, redshift=redshift,
        prefix='FeIIMgII_', amplitude=amplitude, wav_limits=[2200, 3500],
        intr_fwhm=intr_fwhm)

    return model, params

def setup_iron_template_MgII_V01(prefix, **kwargs):

    redshift = kwargs.pop('redshift', 0)
    amplitude = kwargs.pop('amplitude', 1e-4)
    intr_fwhm = kwargs.pop('intr_fwhm', 900)
    # 2200-3500 Vestergaard 2001
    model, params = setup_template_model(
        template_filename='Fe_UVtemplt_A.asc', fwhm=2500, redshift=redshift,
        prefix='FeIIMgII_', amplitude=amplitude, wav_limits=[2200, 3500],
        intr_fwhm=intr_fwhm)

    return model, params

def setup_iron_template_UV_V01(prefix, **kwargs):

    redshift = kwargs.pop('redshift', 0)
    amplitude = kwargs.pop('amplitude', 1e-4)
    intr_fwhm = kwargs.pop('intr_fwhm', 900)
    # 1200-2200 Vestergaard 2001
    model, params = setup_template_model(
        template_filename='Fe_UVtemplt_A.asc', fwhm=3500, redshift=redshift,
        prefix='FeIICIV_', amplitude=amplitude, wav_limits=[1200, 2200],
        intr_fwhm=intr_fwhm)

    return model, params

def setup_iron_template_OPT_BG92(prefix, **kwargs):

    redshift = kwargs.pop('redshift', 0)
    amplitude = kwargs.pop('amplitude', 1e-4)
    intr_fwhm = kwargs.pop('intr_fwhm', 900)
    # 3700-5600 Boronson & Green 1992
    model, params = setup_template_model(
        template_filename='Fe_OPT_BR92_linear.txt', fwhm=3500,
        redshift=redshift,
        prefix='FeIIHb_', amplitude=amplitude, wav_limits=[3700, 5600],
        intr_fwhm=intr_fwhm)

    return model, params


def setup_iron_template_FeIIUV(prefix, **kwargs):

    redshift = kwargs.pop('redshift', 0)
    amplitude = kwargs.pop('amplitude', 1e-4)
    intr_fwhm = kwargs.pop('intr_fwhm', 900)
    templ_list = ['UV01', 'UV02', 'UV03']

    model, params = setup_subdivided_iron_template(fwhm=2500,
                                                   redshift=redshift,
                                                   amplitude=amplitude,
                                                   templ_list=templ_list,
                                                   intr_fwhm=intr_fwhm)

    return model, params


def setup_split_iron_template_FeIIUV(prefix, **kwargs):

    redshift = kwargs.pop('redshift', 0)
    amplitude = kwargs.pop('amplitude', 1e-4)
    intr_fwhm = kwargs.pop('intr_fwhm', 900)
    templ_list = ['UV01', 'UV02', 'UV03']

    model, params = setup_subdivided_iron_template(fwhm=2500,
                                                   redshift=redshift,
                                                   amplitude=amplitude,
                                                   templ_list=templ_list,
                                                   intr_fwhm=intr_fwhm)

    return model, params

def setup_split_iron_template_MgII_V01(prefix, **kwargs):

    redshift = kwargs.pop('redshift', 0)
    amplitude = kwargs.pop('amplitude', 1e-4)
    intr_fwhm = kwargs.pop('intr_fwhm', 900)
    templ_list = ['UV04_V01', 'UV05_V01', 'UV06_V01']

    model, params = setup_subdivided_iron_template(fwhm=2500,
                                                   redshift=redshift,
                                                   amplitude=amplitude,
                                                   templ_list=templ_list,
                                                   intr_fwhm=intr_fwhm)

    return model, params

def setup_split_iron_template_MgII_T06(prefix, **kwargs):

    redshift = kwargs.pop('redshift', 0)
    amplitude = kwargs.pop('amplitude', 1e-4)
    intr_fwhm = kwargs.pop('intr_fwhm', 900)
    templ_list = ['UV04_T06', 'UV05_T06', 'UV06_T06']

    model, params = setup_subdivided_iron_template(fwhm=2500,
                                                   redshift=redshift,
                                                   amplitude=amplitude,
                                                   templ_list=templ_list,
                                                   intr_fwhm=intr_fwhm)

    return model, params

def setup_split_iron_template_OPT(prefix, **kwargs):

    redshift = kwargs.pop('redshift', 0)
    amplitude = kwargs.pop('amplitude', 1e-4)
    intr_fwhm = kwargs.pop('intr_fwhm', 900)
    templ_list = ['OPT01', 'OPT02', 'OPT03']

    model, params = setup_subdivided_iron_template(fwhm=3500,
                                                   redshift=redshift,
                                                   amplitude=amplitude,
                                                   templ_list=templ_list,
                                                   intr_fwhm=intr_fwhm)

    return model, params


# Emission Line models

# SiIV line
def setup_line_model_SiIV_2G(prefix, **kwargs):

    redshift = kwargs.pop('redshift', 0)
    amplitude = kwargs.pop('amplitude', 1e-16)

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
        model, params = setup_gaussian_fwhm_km_s_z(prefix,
                                                   redshift=redshift,
                                                   amplitude=amplitudes[idx],
                                                   cenwave=cenwaves[idx],
                                                   fwhm=fwhms[idx],
                                                   vshift=vshifts[idx]
                                                   )
        param_list.append(params)
        model_list.append(model)

    # Set parameter properties
    for (params, prefix) in zip(param_list, prefixes):
        print(params, prefix)
        params[prefix + 'cen'].set(vary=False)
        params[prefix + 'amp'].set(min=1.0e-19, max=1.0e-10)
        params[prefix + 'fwhm_km_s'].set(min=300, max=20000)
        params[prefix + 'shift_km_s'].set(vary=False, min=-200, max=200)

    return model_list, param_list


def setup_line_model_SiIV_1G(prefix, **kwargs):

    redshift = kwargs.pop('redshift', 0)
    amplitude = kwargs.pop('amplitude', 1e-16)

    # CIV 2G model params
    prefixes = ['SiIV_A_']
    amplitudes = np.array([2]) * amplitude
    fwhms = [2500]
    # SiIV + OIV] http://classic.sdss.org/dr6/algorithms/linestable.html
    cenwaves = [1399.8]
    vshifts = [0]

    param_list = []
    model_list = []

    # Build parameter and model list
    for idx, prefix in enumerate(prefixes):
        model, params = setup_gaussian_fwhm_km_s_z(prefix,
                                                   redshift=redshift,
                                                   amplitude=amplitudes[idx],
                                                   cenwave=cenwaves[idx],
                                                   fwhm=fwhms[idx],
                                                   vshift=vshifts[idx]
                                                   )
        param_list.append(params)
        model_list.append(model)

    # Set parameter properties
    for (params, prefix) in zip(param_list, prefixes):
        print(params, prefix)
        params[prefix + 'cen'].set(vary=False)
        params[prefix + 'amp'].set(min=1.0e-19, max=1.0e-10)
        params[prefix + 'fwhm_km_s'].set(min=300, max=20000)
        params[prefix + 'shift_km_s'].set(vary=False, min=-200, max=200)

    return model_list, param_list

# CIV line
def setup_line_model_CIV_2G(prefix, **kwargs):

    redshift = kwargs.pop('redshift', 0)
    amplitude = kwargs.pop('amplitude', 1e-16)

    # CIV 2G model params
    prefixes = ['CIV_A_', 'CIV_B_']
    amplitudes = np.array([2, 2]) * amplitude
    fwhms = [2500, 2500]
    cenwaves = [1549.06, 1549.06] # Vanden Berk 2001
    vshifts = [0, 0]

    param_list = []
    model_list = []

    # Build parameter and model list
    for idx, prefix in enumerate(prefixes):

        model, params = setup_gaussian_fwhm_km_s_z(prefix,
                                                   redshift=redshift,
                                                   amplitude=amplitudes[idx],
                                                   cenwave=cenwaves[idx],
                                                   fwhm=fwhms[idx],
                                                   vshift=vshifts[idx]
                                                   )
        param_list.append(params)
        model_list.append(model)

    # Set parameter properties
    for (params, prefix) in zip(param_list, prefixes):
        print(params, prefix)
        params[prefix + 'cen'].set(vary=False)
        params[prefix + 'amp'].set(min=1.0e-19, max=1.0e-10)
        params[prefix + 'fwhm_km_s'].set(min=300, max=20000)
        params[prefix + 'shift_km_s'].set(vary=False, min=-200, max=200)

    return model_list, param_list


def setup_line_model_CIV_1G(prefix, **kwargs):

    redshift = kwargs.pop('redshift', 0)
    amplitude = kwargs.pop('amplitude', 1e-16)

    # CIV 2G model params
    prefixes = ['CIV_A_']
    amplitudes = np.array([2]) * amplitude
    fwhms = [2500]
    cenwaves = [1549.06]  # Vanden Berk 2001
    vshifts = [0]

    param_list = []
    model_list = []

    # Build parameter and model list
    for idx, prefix in enumerate(prefixes):
        model, params = setup_gaussian_fwhm_km_s_z(prefix,
                                                   redshift=redshift,
                                                   amplitude=amplitudes[idx],
                                                   cenwave=cenwaves[idx],
                                                   fwhm=fwhms[idx],
                                                   vshift=vshifts[idx]
                                                   )
        param_list.append(params)
        model_list.append(model)

    # Set parameter properties
    for (params, prefix) in zip(param_list, prefixes):
        print(params, prefix)
        params[prefix + 'cen'].set(vary=False)
        params[prefix + 'amp'].set(min=1.0e-19, max=1.0e-10)
        params[prefix + 'fwhm_km_s'].set(min=300, max=20000)
        params[prefix + 'shift_km_s'].set(vary=False, min=-200, max=200)

    return model_list, param_list


def setup_line_model_MgII_2G(prefix, **kwargs):


    redshift = kwargs.pop('redshift', 0)
    amplitude = kwargs.pop('amplitude', 1e-16)

    prefixes = ['MgII_A_', 'MgII_B_']

    # MgII 2G model params
    fwhms = [2500, 1000]
    amplitudes = np.array([5, 2]) * amplitude
    cenwaves = [2798.75, 2798.75] # Vanden Berk 2001
    vshifts = [0, 0]

    param_list = []
    model_list = []

    # Build parameter and model list
    for idx, prefix in enumerate(prefixes):
        model, params = setup_gaussian_fwhm_km_s_z(prefix,
                                                   redshift=redshift,
                                                   amplitude=amplitudes[idx],
                                                   cenwave=cenwaves[idx],
                                                   fwhm=fwhms[idx],
                                                   vshift=vshifts[idx]
                                                   )
        param_list.append(params)
        model_list.append(model)

    # Set parameter properties
    for (params, prefix) in zip(param_list, prefixes):
        print(params, prefix)
        params[prefix + 'cen'].set(vary=False)
        params[prefix + 'amp'].set(min=1.0e-19, max=1.0e-10)
        params[prefix + 'fwhm_km_s'].set(min=300, max=20000)
        params[prefix + 'shift_km_s'].set(vary=False, min=-200, max=200)

    return model_list, param_list


def setup_line_model_MgII_1G(prefix, **kwargs):
    redshift = kwargs.pop('redshift', 0)
    amplitude = kwargs.pop('amplitude', 1e-16)

    prefixes = ['MgII_A_']

    # MgII 1G model params
    fwhms = [2500]
    amplitudes = np.array([2]) * amplitude
    cenwaves = [2798.75]  # Vanden Berk 2001
    vshifts = [0]

    param_list = []
    model_list = []

    # Build parameter and model list
    for idx, prefix in enumerate(prefixes):
        model, params = setup_gaussian_fwhm_km_s_z(prefix,
                                                   redshift=redshift,
                                                   amplitude=amplitudes[idx],
                                                   cenwave=cenwaves[idx],
                                                   fwhm=fwhms[idx],
                                                   vshift=vshifts[idx]
                                                   )
        param_list.append(params)
        model_list.append(model)

    # Set parameter properties
    for (params, prefix) in zip(param_list, prefixes):
        print(params, prefix)
        params[prefix + 'cen'].set(vary=False)
        params[prefix + 'amp'].set(min=1.0e-19, max=1.0e-10)
        params[prefix + 'fwhm_km_s'].set(min=300, max=20000)
        params[prefix + 'shift_km_s'].set(vary=False, min=-200, max=200)

    return model_list, param_list


def setup_line_model_HbOIII_6G(prefix, **kwargs):

    redshift = kwargs.pop('redshift', 0)
    amplitude = kwargs.pop('amplitude', 1e-16)

    prefixes = ['HBeta_A_', 'HBeta_B_', 'OIII4960_A_', 'OIII4960_B_',
                'OIII5008_A_', 'OIII5008_B_']

    # Hbeta + O[III] 6G model params
    amplitudes = np.array([4, 2, 1, 0.5, 2, 1]) * amplitude
    fwhms = [2500, 900, 900, 900, 1200, 1200]
    cenwaves = [4862.68, 4862.68, 4960.30, 4960.30, 5008.24, 5008.24]
    vshifts = [0, 0, 0, 0, 0, 0]

    param_list = []
    model_list = []

    # Build parameter and model list
    for idx, prefix in enumerate(prefixes):
        model, params = setup_gaussian_fwhm_km_s_z(prefix,
                                                   redshift=redshift,
                                                   amplitude=amplitudes[idx],
                                                   cenwave=cenwaves[idx],
                                                   fwhm=fwhms[idx],
                                                   vshift=vshifts[idx]
                                                   )
        param_list.append(params)
        model_list.append(model)


    # Set parameter properties
    for (params, prefix) in zip(param_list, prefixes):
        print(params, prefix)
        params[prefix + 'cen'].set(vary=False)
        params[prefix + 'amp'].set(min=1.0e-19, max=1.0e-10)
        params[prefix + 'fwhm_km_s'].set(min=300, max=20000)
        params[prefix + 'shift_km_s'].set(vary=False, min=-200, max=200)

    return model_list, param_list


def setup_line_model_HbOIII_4G(prefix, **kwargs):

    redshift = kwargs.pop('redshift', 0)
    amplitude = kwargs.pop('amplitude', 1e-16)

    prefixes = ['HBeta_A_', 'HBeta_B_', 'OIII4960_A_', 'OIII5008_A_']

    # Hbeta + O[III] 4G model params
    amplitudes = np.array([4, 2, 1, 2]) * amplitude
    fwhms = [2500, 900, 900, 1200]
    cenwaves = [4862.68, 4862.68, 4960.30, 5008.24]
    vshifts = [0, 0, 0, 0]

    param_list = []
    model_list = []

    # Build parameter and model list
    for idx, prefix in enumerate(prefixes):
        model, params = setup_gaussian_fwhm_km_s_z(prefix,
                                                   redshift=redshift,
                                                   amplitude=amplitudes[idx],
                                                   cenwave=cenwaves[idx],
                                                   fwhm=fwhms[idx],
                                                   vshift=vshifts[idx]
                                                   )
        param_list.append(params)
        model_list.append(model)


    # Set parameter properties
    for (params, prefix) in zip(param_list, prefixes):
        print(params, prefix)
        params[prefix + 'cen'].set(vary=False)
        params[prefix + 'amp'].set(min=1.0e-19, max=1.0e-10)
        params[prefix + 'fwhm_km_s'].set(min=300, max=20000)
        params[prefix + 'shift_km_s'].set(vary=False, min=-200, max=200)

    return model_list, param_list


def setup_line_model_Hb_2G(prefix, **kwargs):

    redshift = kwargs.pop('redshift', 0)
    amplitude = kwargs.pop('amplitude', 1e-16)

    prefixes = ['HBeta_A_', 'HBeta_B_']

    # Hbeta + O[III] 4G model params
    amplitudes = np.array([4, 2]) * amplitude
    fwhms = [2500, 900]
    cenwaves = [4862.68, 4862.68]
    vshifts = [0, 0]

    param_list = []
    model_list = []

    # Build parameter and model list
    for idx, prefix in enumerate(prefixes):
        model, params = setup_gaussian_fwhm_km_s_z(prefix,
                                                   redshift=redshift,
                                                   amplitude=amplitudes[idx],
                                                   cenwave=cenwaves[idx],
                                                   fwhm=fwhms[idx],
                                                   vshift=vshifts[idx]
                                                   )
        param_list.append(params)
        model_list.append(model)


    # Set parameter properties
    for (params, prefix) in zip(param_list, prefixes):
        print(params, prefix)
        params[prefix + 'cen'].set(vary=False)
        params[prefix + 'amp'].set(min=1.0e-19, max=1.0e-10)
        params[prefix + 'fwhm_km_s'].set(min=300, max=20000)
        params[prefix + 'shift_km_s'].set(vary=False, min=-200, max=200)

    return model_list, param_list



def setup_line_model_Ha_2G(prefix, **kwargs):

    redshift = kwargs.pop('redshift', 0)
    amplitude = kwargs.pop('amplitude', 1e-16)

    prefixes = ['HAlpha_A_', 'HAlpha_B_']

    amplitudes = np.array([2, 2]) * amplitude
    fwhms = [10000, 5000]
    cenwaves = [6564.61, 6564.61]
    vshifts = [0, 0]

    param_list = []
    model_list = []

    # Build parameter and model list
    for idx, prefix in enumerate(prefixes):
        model, params = setup_gaussian_fwhm_km_s_z(prefix,
                                                   redshift=redshift,
                                                   amplitude=amplitudes[idx],
                                                   cenwave=cenwaves[idx],
                                                   fwhm=fwhms[idx],
                                                   vshift=vshifts[idx]
                                                   )
        param_list.append(params)
        model_list.append(model)

    # Set parameter properties
    for (params, prefix) in zip(param_list, prefixes):
        print(params, prefix)
        params[prefix + 'cen'].set(vary=False)
        params[prefix + 'amp'].set(min=1.0e-19, max=1.0e-10)
        params[prefix + 'fwhm_km_s'].set(min=300, max=20000)
        params[prefix + 'shift_km_s'].set(vary=False, min=-200, max=200)

    return model_list, param_list


def setup_line_model_Ha_3G(prefix, **kwargs):

    redshift = kwargs.pop('redshift', 0)
    amplitude = kwargs.pop('amplitude', 1e-16)

    prefixes = ['HAlpha_A_', 'HAlpha_B_', 'HAlpha_narrow_']

    amplitudes = np.array([10, 5, 2]) * amplitude
    fwhms = [10000, 5000, 500]
    cenwaves = [6564.61, 6564.61, 6564.61]
    vshifts = [0, 0, 0]

    param_list = []
    model_list = []

    # Build parameter and model list
    for idx, prefix in enumerate(prefixes):
        model, params = setup_gaussian_fwhm_km_s_z(prefix,
                                                   redshift=redshift,
                                                   amplitude=amplitudes[idx],
                                                   cenwave=cenwaves[idx],
                                                   fwhm=fwhms[idx],
                                                   vshift=vshifts[idx]
                                                   )
        param_list.append(params)
        model_list.append(model)

    # Set parameter properties
    for (params, prefix) in zip(param_list, prefixes):
        print(params, prefix)
        params[prefix + 'cen'].set(vary=False)
        params[prefix + 'amp'].set(min=1.0e-19, max=1.0e-10)
        params[prefix + 'fwhm_km_s'].set(min=300, max=20000)
        params[prefix + 'shift_km_s'].set(vary=False, min=-200, max=200)


    # Adjusting narrow component
    param_list[2]['HAlpha_narrow_' + 'fwhm_km_s'].set(min=50, max=2000)

    return model_list, param_list


def setup_line_model_HeII_1G(prefix, **kwargs):

    redshift = kwargs.pop('redshift', 0)
    amplitude = kwargs.pop('amplitude', 1e-16)

    prefixes = ['HeII_']

    amplitudes = np.array([2]) * amplitude
    fwhms = [2500]
    cenwaves = [1640.42] # Vanden Berk 2001
    vshifts = [0, 0]

    param_list = []
    model_list = []

    # Build parameter and model list
    for idx, prefix in enumerate(prefixes):
        model, params = setup_gaussian_fwhm_km_s_z(prefix,
                                                   redshift=redshift,
                                                   amplitude=amplitudes[idx],
                                                   cenwave=cenwaves[idx],
                                                   fwhm=fwhms[idx],
                                                   vshift=vshifts[idx]
                                                   )
        param_list.append(params)
        model_list.append(model)

    # Set parameter properties
    for (params, prefix) in zip(param_list, prefixes):
        print(params, prefix)
        params[prefix + 'cen'].set(vary=False)
        params[prefix + 'amp'].set(min=1.0e-19, max=1.0e-10)
        params[prefix + 'fwhm_km_s'].set(min=300, max=20000)
        params[prefix + 'shift_km_s'].set(vary=False, min=-200, max=200)

    return model_list, param_list


def setup_line_model_CIII_1G(prefix, **kwargs):

    redshift = kwargs.pop('redshift', 0)
    amplitude = kwargs.pop('amplitude', 1e-16)

    prefixes = ['CIII_']

    amplitudes = np.array([2]) * amplitude
    fwhms = [2500]
    cenwaves = [1908.73] # Vanden Berk 2001
    vshifts = [0, 0]

    param_list = []
    model_list = []

    # Build parameter and model list
    for idx, prefix in enumerate(prefixes):
        model, params = setup_gaussian_fwhm_km_s_z(prefix,
                                                   redshift=redshift,
                                                   amplitude=amplitudes[idx],
                                                   cenwave=cenwaves[idx],
                                                   fwhm=fwhms[idx],
                                                   vshift=vshifts[idx]
                                                   )
        param_list.append(params)
        model_list.append(model)

    # Set parameter properties
    for (params, prefix) in zip(param_list, prefixes):
        print(params, prefix)
        params[prefix + 'cen'].set(vary=False)
        params[prefix + 'amp'].set(min=1.0e-19, max=1.0e-10)
        params[prefix + 'fwhm_km_s'].set(min=300, max=20000)
        params[prefix + 'shift_km_s'].set(vary=False, min=-200, max=200)

    return model_list, param_list



def setup_line_model_CIII(prefix, **kwargs):
    
    redshift = kwargs.pop('redshift', 0)
    amplitude = kwargs.pop('amplitude', 1e-16)
    
    prefix ='CIII_'

    amp_cIII = 21.19 * amplitude
    amp_alIII = 0.4 * amplitude
    amp_siIII = 0.16 * amplitude
        
    params = Parameters()

    params.add(prefix+'cen', value=1908.73, vary=False)
    params.add(prefix+'cen_siIII', value=1892.03, vary=False)
    params.add(prefix+'cen_alIII', value=1857.40, vary=False)

    add_redshift_param(redshift, params, prefix)

    params.add(prefix+'amp', value=amp_cIII, vary=True, min=1e-19, max=1e-10)
    params.add(prefix+'amp_alIII', value=amp_alIII, vary=True, min=1e-19,
               max=1e-10)
    params.add(prefix+'amp_siIII', value=amp_siIII, vary=True, min=1e-19,
               max=1e-10)

    params.add(prefix+'fwhm_km_s', value=2000, vary=True, min=500, max=1e+4)
    params.add(prefix+'fwhm_km_s_alIII', value=400, vary=True, min=500, max=7e+3)
    params.add(prefix+'fwhm_km_s_siIII', value=300, vary=True, min=500, max=7e+3)

    params.add(prefix+'shift_km_s', value=0, vary=False)
    params.add(prefix+'shift_km_s_alIII', value=0, vary=False)
    params.add(prefix+'shift_km_s_siIII', value=0, vary=False)


    elmodel = Model(CIII_complex_model_func, prefix='CIII_')

    return elmodel, params




# ------------------------------------------------------------------------------
# Masks
# ------------------------------------------------------------------------------

 # QSO continuum windows, see Vestergaard & Peterson 2006
qso_cont_VP06 = {'name': 'QSO Cont. VP06' ,
                 'rest_frame': True,
                 'mask_ranges':[[1265, 1290], [1340, 1375], [1425, 1470],
                                [1680, 1705], [1905, 2050]]}

qso_contfe_MgII_Shen11 = {'name': 'QSO Cont. MgII Shen11' ,
                 'rest_frame': True,
                 'mask_ranges':[[2200, 2700], [2900, 3090]]}

qso_contfe_HBeta_Shen11 = {'name': 'QSO Cont. HBeta Shen11' ,
                 'rest_frame': True,
                 'mask_ranges':[[4435, 4700], [5100, 5535]]}

qso_contfe_HAlpha_Shen11 = {'name': 'QSO Cont. HAlpha Shen11' ,
                 'rest_frame': True,
                 'mask_ranges':[[6000, 6250], [6800, 7000]]}

qso_contfe_CIV_Shen11 = {'name': 'QSO Cont. CIV Shen11' ,
                 'rest_frame': True,
                 'mask_ranges':[[1445, 1465], [1700, 1705]]}


xshooter_nir_atmospheric_windows = {'name': 'XSHOOTER telluric regions' ,
                                    'rest_frame': False,
                                    'mask_ranges':[[5000, 10250],
                                                  [13450, 14300],
                                                  [18000, 19400]]}

xshooter_surge_continuum_windows = {'name': 'XSHOOTER surge continuum regions' ,
                                    'rest_frame': True,
                                    'mask_ranges':[[1445, 1465],
                                                   [1700, 1705],
                                                   [2155, 2400],
                                                   [2480, 2675],
                                                   [2925, 3100]]}

xshooter_surge_line_windows = {'name': 'XSHOOTER surge line regions' ,
                                    'rest_frame': True,
                                    'mask_ranges':[[2700, 2900],
                                        [1470, 1600]]}


qso_cont_windows = {'name': 'QSO Continuum windows',
                    'rest_frame': True,
                    'mask_ranges':[[1445, 1465],
                                   [1700, 1705],
                                   [2155, 2400],
                                   [2480, 2675],
                                   [2925, 3100]]}


qso_pca_1200_3100_model = {'name': 'PypeIt PCA Boundaries 1200-3100',
                    'rest_frame': True,
                    'mask_ranges':[[0, 1200],
                                  [3100, 99999]]}


J2211_cont_windows = {'name': 'J2211 Continuum windows',
                    'rest_frame': True,
                    'mask_ranges':[[1276, 1286],
                                   [1564, 1628],
                                   [1764, 1805],
                                   [2155, 2269],
                                   [2360, 2400],
                                   [2653, 2750],
                                   [2846, 3090]]}

# Dictionary with all masks
mask_presets = {'QSO Cont. VP06': qso_cont_VP06,
                'XSHOOTER telluric regions': xshooter_nir_atmospheric_windows,
                'XSHOOTER surge continuum regions':
                    xshooter_surge_continuum_windows,
                'XSHOOTER surge line regions': xshooter_surge_line_windows,
                'QSO Cont. CIV Shen11': qso_contfe_CIV_Shen11,
                'QSO Cont. MgII Shen11': qso_contfe_MgII_Shen11,
                'QSO Cont. HBeta Shen11': qso_contfe_HBeta_Shen11,
                'QSO Cont. HAlpha Shen11': qso_contfe_HAlpha_Shen11,
                'QSO Continuum windows': qso_cont_windows,
                'PypeIt PCA Boundaries 1200-3100': qso_pca_1200_3100_model,
                'J2211 Continuum windows': J2211_cont_windows}

# ------------------------------------------------------------------------------
# LISTS TO IMPORT IN SPECMODEL
# ------------------------------------------------------------------------------

# Model function name list and setup function list for SpecModelWidget
model_func_list = ['Power Law (2500A)',
                   'Power Law (2500A) + BC (flexible)',
                   'Power Law (2500A) + BC (manual)',
                   'FeII template (V01, cont)',
                   'FeII template (T06, cont)',
                   'FeII template 1200-2200 (V01, cont)',
                   'FeII template 1200-2200 (V01, split)',
                   'FeII template 2200-3500 (V01, cont)',
                   'FeII template 2200-3500 (V01, split)',
                   'FeII template 2200-3500 (T06, cont)',
                   'FeII template 2200-3500 (T06, split)',
                   'FeII template 3700-5600 (BG92, cont)',
                   'FeII template 3700-5600 (BG92, split)',
                   'Gaussian (FWHM in km/z, z)',
                   'SiIV (2G, FWHM in km/s, z)',
                   'SiIV (1G, FWHM in km/s, z)',
                   'CIV (2G, FWHM in km/s, z)',
                   'CIV (1G, FWHM in km/s, z)',
                   'MgII (2G, FWHM in km/s, z)',
                   'MgII (1G, FWHM in km/s, z)',
                   'HBeta+[OIII] (6G, FWHM in km/s, z)',
                   'HBeta+[OIII] (4G, FWHM in km/s, z)',
                   'HBeta (2G, FWHM in km/s, z)',
                   'HAlpha (3G, FWHM in km/s, z)',
                   'HAlpha (2G, FWHM in km/s, z)',
                   'HeII (1G, FWHM in km/s, z)',
                   'CIII (1G, FWHM in km/s, z)',
                   'CIII complex (FWHM in km/s, z)'
                   ]

model_setup_list = [setup_power_law_at_2500A,
                    setup_power_law_at_2500A_plus_flexible_BC,
                    setup_power_law_at_2500A_plus_manual_BC,
                    setup_iron_template_V01,
                    setup_iron_template_T06,
                    setup_iron_template_UV_V01,
                    setup_split_iron_template_FeIIUV,
                    setup_iron_template_MgII_V01,
                    setup_split_iron_template_MgII_V01,
                    setup_iron_template_MgII_T06,
                    setup_split_iron_template_MgII_T06,
                    setup_iron_template_OPT_BG92,
                    setup_split_iron_template_OPT,
                    setup_gaussian_fwhm_km_s_z,
                    setup_line_model_SiIV_2G,
                    setup_line_model_SiIV_1G,
                    setup_line_model_CIV_2G,
                    setup_line_model_CIV_1G,
                    setup_line_model_MgII_2G,
                    setup_line_model_MgII_1G,
                    setup_line_model_HbOIII_6G,
                    setup_line_model_HbOIII_4G,
                    setup_line_model_Hb_2G,
                    setup_line_model_Ha_3G,
                    setup_line_model_Ha_2G,
                    setup_line_model_HeII_1G,
                    setup_line_model_CIII_1G,
                    setup_line_model_CIII
                    ]

# Dictionary listing all models for load/save functionality
model_func_dict = {'power_law_at_2500A': power_law_at_2500A,
                   'power_law_at_2500A_plus_flexible_BC':
                       power_law_at_2500A_plus_flexible_BC,
                   'power_law_at_2500A_plus_manual_BC':
                       power_law_at_2500A_plus_manual_BC,
                   # 'setup_iron_template_UV_V01' : template_model,
                   # 'setup_split_iron_template_FeIIUV': template_model,
                   # 'setup_iron_template_MgII_V01': template_model,
                   # 'setup_split_iron_template_MgII_V01': template_model,
                   # 'setup_iron_template_MgII_T06': template_model,
                   # 'setup_split_iron_template_MgII_T06': template_model,
                   # 'setup_iron_template_OPT_BG92': template_model,
                   # 'setup_split_iron_template_OPT': template_model,
                   'template_model': template_model,
                   'gaussian_fwhm_km_s_z': gaussian_fwhm_km_s_z,
                   'CIII_complex_model_func': CIII_complex_model_func}
































