
import numpy as np
import matplotlib.pyplot as plt
from astropy import constants as const
import emcee
import corner
from tqdm import tqdm
import matplotlib.transforms as mtransforms
from astropy.cosmology import FlatLambdaCDM
from astropy import units as u
from astropy import units
from astropy.table import QTable

import scipy as sp

import time

import extinction

from IPython import embed

c_km_s = const.c.to('km/s').value

from speconed import speconed as sod
from sculptor import analysis as scana

# ------------------------------------------------------------------------------
# Model functions
# ------------------------------------------------------------------------------


def power_law_at2500(x, redsh, amp, slope, **kwargs):
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

    return amp * ((x / (2500 * (redsh+1)))**slope)


def ext_power_law_at2500(x, redsh, amp, slope, a_v, **kwargs):

    flux = amp * ((x / (2500 * (redsh+1)))**slope)

    ext = extinction.calzetti00(x / (redsh + 1), a_v, 4.05)

    return extinction.apply(ext, flux)


def scattered_power_law_at2500(x, redsh, amp, slope, f_sc, **kwargs):

    flux = amp * ((x / (2500 * (redsh + 1))) ** slope)

    return 10**f_sc * flux


def template_model(x, redsh, amp, fwhm, intr_fwhm,  templ_disp=None,
                   templ_fluxden=None,
                   templ_disp_unit_str=None,
                   templ_fluxden_unit_str=None, **kwargs):

    if templ_disp_unit_str is not None and type(templ_disp_unit_str) is str:
        dispersion_unit = u.Unit(templ_disp_unit_str, format='cds')
    else:
        dispersion_unit = None
    if templ_fluxden_unit_str is not None and \
            type(templ_fluxden_unit_str) is str:
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
    spec.dispersion = spec.dispersion * (1.+redsh)

    # Return interpolation function
    f = sp.interpolate.interp1d(spec.dispersion, spec.fluxden, kind='linear',
                                bounds_error=False, fill_value=(0, 0))

    return f(x)*amp


def line_model_gaussian(x, redsh, flux, cen, fwhm_km_s, **kwargs):
    """ Gaussian line model

    The central wavelength of the Gaussian line model is determined by the
    central wavelength cen and the redshift, z. These parameters are degenerate
    in a line fit and it is adviseable to fix one of them (to predetermined
    values e.g., the redshift or the central wavelength).

    The width of the line is set by the FWHM in km/s.

    The Gaussian is normalized.

    :param x: Dispersion of the continuum model
    :type x: np.ndarray
    :param z: Redshift
    :type z: float
    :param flux: Amplitude of the Gaussian
    :type flux: float
    :param cen: Central wavelength
    :type cen: float
    :param fwhm_km_s: FWHM of the Gaussian in km/s
    :type fwhm_km_s: float
    :return: Gaussian line model
    :rtype: np.ndarray

    """

    # Redshift central wavelength
    cen = cen * (1. + redsh)

    # Calculate sigma from fwhm
    fwhm = fwhm_km_s / c_km_s * cen
    sigma = fwhm / np.sqrt(8*np.log(2))

    return flux/(sigma*np.sqrt(2*np.pi)) * np.exp(-(x-cen)**2 / (2*sigma**2))


# ------------------------------------------------------------------------------
# High-level analysis routines
# ------------------------------------------------------------------------------

def analyze_mcmc_results(model, flat_chain, cont_dict,
                         emfeat_dictlist, redshift, cosmology, dispersion=None,
                         dispersion_unit=None, fluxden_unit=None, width=10):

    print('[INFO] Analyzing MCMC results')



    pass


def _mcmc_analyze(model, flat_chain, cont_dict, emfeat_dictlist, redshift,
                  cosmology, dispersion=None, dispersion_unit=None,
                  fluxden_unit=None, width=10, emfeat_meas=None, a_v=None,
                  ext_law='calzetti00'):


    cont_component = cont_dict['components']
    cont_wavelengths = cont_dict['rest_frame_wavelengths']

    result_table = None

    for idx in tqdm(range(flat_chain.shape[0])):

        params = flat_chain[idx, :]

        result_dict = {}


        cont_result_dict = analyze_continuum(model, cont_component, params,
                                       cont_wavelengths, cosmology, redshift=redshift,
                                       dispersion_unit=dispersion_unit,
                                       fluxden_unit=fluxden_unit,
                                       dispersion=dispersion, width=width)

        for emfeat_dict in emfeat_dictlist:

            feature_name = emfeat_dict['feature_name']
            em_components = emfeat_dict['components']
            rest_frame_wavelength = emfeat_dict['rest_frame_wavelength']
            if 'disp_range' in emfeat_dict:
                disp_range = emfeat_dict['disp_range']
            else:
                disp_range = None
            if 'fwhm_method' in emfeat_dict:
                fwhm_method = emfeat_dict['fwhm_method']
            else:
                fwhm_method = 'spline'



            emfeat_result_dict = analyze_emission_feature(model, feature_name, em_components,
                                                    params,
                                                    rest_frame_wavelength,
                                                    cont_components=cont_component,
                                                    redshift=redshift,
                                                    dispersion=dispersion,
                                                      emfeat_meas=emfeat_meas,
                                                      disp_range=disp_range,
                                                      cosmology=cosmology,
                                                      fwhm_method=fwhm_method,
                                                    dispersion_unit=dispersion_unit,
                                                    fluxden_unit=fluxden_unit,
                                                    a_v=a_v, ext_law=ext_law)


            if emfeat_result_dict is not None:
                result_dict.update(emfeat_result_dict)

        if cont_result_dict is not None:
            result_dict.update(cont_result_dict)

        if result_table is None:
            # Initialize QTable for results
            # Initialize column names
            result_table = QTable(names=result_dict.keys())
            # Initialize column units
            for column in result_table.columns:
                if (isinstance(result_dict[column], units.Quantity) or
                        isinstance(result_dict[column], units.IrreducibleUnit) or
                        isinstance(result_dict[column], units.CompositeUnit)):
                    result_table[column].unit = result_dict[column].unit

        result_table.add_row(result_dict)

    return result_table



emfeat_measures_default = ['peak_fluxden',
                           'peak_redsh',
                           'EW',
                           'FWHM',
                           'flux',
                           'lum']

def analyze_emission_feature(model, feature_name, components, params_values,
                             rest_frame_wavelength, cont_components=None,
                             redshift=None, dispersion=None,
                             emfeat_meas=None, disp_range=None,
                             cosmology=None, fwhm_method='spline',
                             dispersion_unit=None, fluxden_unit=None,
                             a_v=None, ext_law='calzetti00'):
    """Calculate measurements of an emission feature for a spectral fit (
    SpecFit object).

    At present this analysis assumes that the spectra are in the following
    units:
    flux density - erg/s/cm^2/AA
    dispersion - AA

    :param specfit: SpecFit class object to extract the model flux from
    :type specfit: sculptor.specfit.SpecFit
    :param feature_name: Name of the emission feature, which will be used to
        name the resulting measurements in the output dictionary.
    :type feature_name: string
    :param model_names: List of model names to create the emission feature flux
        from.
    :type model_names: list
    :param rest_frame_wavelength: Rest-frame wavelength of the emission feature
    :type rest_frame_wavelength: float
    :param cont_model_names:  List of model names to create the continuum
        flux model from. The continuum spectrum is for example used in the
        calculation of some emission feature properties (e.g. equivalent width).
    :type cont_model_names: list
    :param redshift: Redshift of the object. This keyword argument defaults
        to 'None', in which case the redshift from the SpecFit object is used.
    :type redshift: float
    :param dispersion: This keyword argument allows to input a dispersion
        axis (e.g., wavelengths) for which the model fluxes are calculated. The
        value defaults to 'None', in which case the dispersion from the SpecFit
        spectrum is being used.
    :type dispersion: np.array
    :param emfeat_meas: This keyword argument allows to specify the list of
        emission feature measurements.
        Currently possible measurements are ['peak_fluxden', 'peak_redsh', 'EW',
        'FWHM', 'flux']. The value defaults to 'None' in which all measurements
        are calculated
    :type emfeat_meas: list
    :param disp_range: 2 element list holding the lower and upper dispersion
        boundaries for the integration
    :type disp_range: list
    :param cosmology: Cosmology for calculating luminosities
    :type cosmology: astropy.cosmology class
    :param fwhm_method: Method to use for calculating the FWHM. Possible
        values are 'sign' or 'spline' (default).
    :type fwhm_method: string
    :return: Dictionary with measurement results (with units)
    :rtype: dict
    """

    # Build the model flux
    model_fluxden = model.eval(dispersion, params_values, components=components)

    if a_v is not None:
        if ext_law == 'calzetti00':
            ext = extinction.calzetti00(dispersion / (1 + redshift), a_v, 4.05)
        else:
            raise ValueError('No valid extinction law specified')
        model_fluxden = extinction.remove(ext, model_fluxden)

    model_spec = sod.SpecOneD(dispersion=dispersion, fluxden=model_fluxden,
                              dispersion_unit=dispersion_unit,
                              fluxden_unit=fluxden_unit)

    if cont_components is not None:
        cont_fluxden = model.eval(dispersion, params_values, components=cont_components)

        if a_v is not None:
            if ext_law == 'calzetti00':
                ext = extinction.calzetti00(dispersion/ (1 + redshift), a_v, 4.05)
            else:
                raise ValueError('No valid extinction law specified')
            cont_fluxden = extinction.remove(ext, cont_fluxden)

        cont_spec = sod.SpecOneD(dispersion=dispersion, fluxden=cont_fluxden,
                              dispersion_unit=dispersion_unit,
                              fluxden_unit=fluxden_unit)

    # Start analysis
    result_dict = {}

    if emfeat_meas is None:
        emfeat_meas = emfeat_measures_default

    if redshift is None:
        raise ValueError('No redshift specified for the analysis.')

    if cosmology is None:
        cosmology = FlatLambdaCDM(H0=70, Om0=0.3, Tcmb0=2.725)
        print('[WARNING] No Cosmology specified. Assuming default '
              'FlatLambdaCDM cosmology with H0=70, Om0=0.3, Tcmb0=2.725.')

    if 'peak_fluxden' in emfeat_meas:
        # Calculate the peak flux density
        result_dict.update({feature_name + '_peak_fluxden':
                                np.max(model_spec.fluxden) *
                                model_spec.fluxden_unit})

    if 'peak_redsh' in emfeat_meas:
        # Calculate the peak redshift
        result_dict.update({feature_name + '_peak_redsh':
                                scana.get_peak_redshift(model_spec,
                                                  rest_frame_wavelength)})
    if 'EW' in emfeat_meas and cont_components is not None:
        # Calculate the rest-frame equivalent width
        ew = scana.get_equivalent_width(cont_spec, model_spec,
                                  redshift=redshift)
        result_dict.update({feature_name + '_EW': ew})

    if 'FWHM' in emfeat_meas:
        # Calculate the (composite) line model FWHM.
        fwhm = scana.get_fwhm(model_spec, method=fwhm_method)
        result_dict.update({feature_name + '_FWHM': fwhm})

        if np.isnan(fwhm):
            print('[WARNING] FWHM could not be calculated for feature '
                  '{}'.format(feature_name))

    if 'flux' in emfeat_meas:
        # Calculate the integrated line flux
        flux = scana.get_integrated_flux(model_spec, disp_range=disp_range)
        result_dict.update({feature_name + '_flux': flux})

    if 'lum' in emfeat_meas:
        # Calculate the integrated line luminosity
        lum = scana.calc_integrated_luminosity(model_spec,
                                         redshift=redshift, cosmology=cosmology)
        result_dict.update({feature_name + '_lum': lum})

    if 'nonparam' in emfeat_meas:
        # Calculate basic non-parametric line measurements
        # Note that this might be computationally expensive!

        v50, v05, v10, v90, v95, v_res_at_line, freq_v50, wave_v50, z_v50 = \
            scana.get_nonparametric_measurements(model_spec, rest_frame_wavelength,
                                           redshift, disp_range=disp_range)

        result_dict.update({feature_name + '_v50': v50})
        result_dict.update({feature_name + '_v05': v05})
        result_dict.update({feature_name + '_v10': v10})
        result_dict.update({feature_name + '_v90': v90})
        result_dict.update({feature_name + '_v95': v95})
        result_dict.update({feature_name + '_vres_at_line': v_res_at_line})
        result_dict.update({feature_name + '_freq_v50': freq_v50})
        result_dict.update({feature_name + '_wave_v50': wave_v50})
        result_dict.update({feature_name + '_redsh_v50': z_v50})

    return result_dict

cont_measures_default = ['fluxden_avg',
                         'Lwav',
                         'appmag']

def analyze_continuum(model, components, params_values, rest_frame_wavelengths,
                      cosmology, redshift=None, dispersion_unit=None,
                      fluxden_unit=None,
                      dispersion=None, cont_meas=None, width=10):
    """Calculate measurements of the continuum at a range of specified
    wavelengths for a spectral fit (SpecFit object).

    At present this analysis assumes that the spectra are in the following
    units:
    flux density - erg/s/cm^2/AA
    dispersion - AA

    :param specfit: SpecFit class object to extract the model flux from
    :type specfit: sculptor.specfit.SpecFit
    :param model_names: List of model names to create the emission feature flux
        from.
    :type model_names: list(string)
    :param rest_frame_wavelengths: Rest-frame wavelength of the emission feature
    :type rest_frame_wavelengths: list(float)
    :param cosmology: Cosmology for calculation of absolute properties
    :type cosmology: astropy.cosmology.Cosmology
    :param redshift: Redshift of the object. This keyword argument defaults
        to 'None', in which case the redshift from the SpecFit object is used.
    :type redshift: float
    :param dispersion: This keyword argument allows to input a dispersion
        axis (e.g., wavelengths) for which the model fluxes are calculated. The
        value defaults to 'None', in which case the dispersion from the SpecFit
        spectrum is being used.
    :type dispersion: np.array
    :param cont_meas: This keyword argument allows to specify the list of
        emission feature measurements.
        Currently possible measurements are ['peak_fluxden', 'peak_redsh', 'EW',
        'FWHM', 'flux']. The value defaults to 'None' in which all measurements
        are calculated
    :type cont_meas: list(string)
    :param width: Window width in dispersion units to calculate the average
        flux density in.
    :type width: [float, float]
    :return: Dictionary with measurement results (with units)
    :rtype: dict

    """

    # Build the continuum model spectrum
    cont_flux = model.eval(dispersion, params_values, components=components)
    cont_spec = sod.SpecOneD(dispersion=dispersion, fluxden=cont_flux,
                                dispersion_unit=dispersion_unit,
                                fluxden_unit=fluxden_unit)
    # Define unit for flux density
    fluxden_unit = cont_spec.fluxden_unit

    # Analyze the continuum
    if cont_meas is None:
        cont_meas = cont_measures_default


    # Start analysis
    result_dict = {}

    for wave in rest_frame_wavelengths:

        wave_name = str(wave)
        fluxden_avg = scana.get_average_fluxden(cont_spec, wave,
                                          redshift=redshift,
                                          width=width)

        if 'fluxden_avg' in cont_meas:
            result_dict.update({wave_name + '_fluxden_avg':
                                    fluxden_avg})

        if 'Lwav' in cont_meas:
            lwav = scana.calc_lwav_from_fwav(fluxden_avg,
                                       redshift=redshift,
                                       cosmology=cosmology)
            result_dict.update({wave_name + '_Lwav': lwav})

        if 'appmag' in cont_meas:
            appmag = scana.calc_apparent_mag_from_fluxden(fluxden_avg,
                                                    wave * (1. + redshift) * units.AA)
            result_dict.update({wave_name + '_appmag': appmag})

        # if 'absmag' in cont_meas:
        #     absmag = calc_absolute_mag_from_monochromatic_luminosity(
        #             lwav, wave*units.AA)
        #     result_dict.update({wave_name + '_absmag': absmag})

    return result_dict
