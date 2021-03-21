#!/usr/bin/env python

import os
import glob
import numpy as np
import pandas as pd
from sculptor import specfit as sf
from sculptor import speconed as sod
from astropy import constants as const
from astropy import units
from scipy.interpolate import UnivariateSpline
import matplotlib.pyplot as plt

pc_in_cm = units.pc.to(units.cm)
c_km_s = const.c.to('km/s').value
L_sun_erg_s = const.L_sun.to('erg/s').value

# ------------------------------------------------------------------------------
# High-level analysis routines
# ------------------------------------------------------------------------------

emfeat_measures_default = ['peak_fluxden',
                           'peak_redsh',
                           'EW',
                           'FWHM',
                           'flux']


def analyze_emission_feature(specfit, feature_name, model_names,
                             rest_frame_wavelength, cont_model_names=None,
                             redshift=None, dispersion=None,
                             emfeat_meas=None, disp_range=None):
    """Calculate measurements of an emission feature for a spectral fit (
    SpecFit object).

    At present this analysis assumes that the spectra are in the following
    units:
    flux density - erg/s/cm^2/AA
    dispersipn - AA

    :param specfit: SpecFit class object to extract the model flux from
    :type sf.SpecFit
    :param feature_name: Name of the emission feature, which will be used to
    name the resulting measurements in the output dictionary.
    :type string
    :param model_names: List of model names to create the emission feature flux
    from.
    :type list
    :param rest_frame_wavelength: Rest-frame wavelength of the emission feature
    :type float
    :param cont_model_names:  List of model names to create the continuum
    flux model from. The continuum spectrum is for example used in the
    calculation of some emission feature properties (e.g. equivalent width).
    :type sod.SpecOneD
    :param redshift: Redshift of the object. This keyword argument defaults
    to 'None', in which case the redshift from the SpecFit object is used.
    :type float
    :param dispersion: This keyword argument allows to input a dispersion
    axis (e.g., wavelengths) for which the model fluxes are calculated. The
    value defaults to 'None', in which case the dispersion from the SpecFit
    spectrum is being used.
    :type np.array
    :param emfeat_meas: This keyword argument allows to specify the list of
    emission feature measurements.
    Currently possible measurements are ['peak_fluxden', 'peak_redsh', 'EW',
    'FWHM', 'flux']. The value defaults to 'None' in which all measurements
    are calculated
    :type list
    :param disp_range: 2 element list holding the lower and upper dispersion
    boundaries for the integration
    :type list
    :return: Dictionary with measurement results (without units)
    :rtype dict
    """

    # Build the model flux
    model_spec = build_model_flux(specfit, model_names, dispersion=dispersion)
    if cont_model_names is not None:
        cont_spec = build_model_flux(specfit, cont_model_names,
                                     dispersion=dispersion)

    # Start analysis
    result_dict = {}

    if emfeat_meas is None:
        emfeat_meas = emfeat_measures_default

    if redshift is None:
        redshift = specfit.redshift

    if 'peak_fluxden' in emfeat_meas:
        result_dict.update({feature_name+'_peak_fluxden':
                            np.max(model_spec.fluxden)})

    if 'peak_redsh' in emfeat_meas:
        result_dict.update({feature_name + '_peak_redsh':
                            get_peak_redshift(model_spec,
                                                    rest_frame_wavelength)})
    if 'EW' in emfeat_meas and cont_model_names is not None:
        ew = get_equivalent_width(cont_spec, model_spec,
                                        redshift=redshift)
        result_dict.update({feature_name + '_EW': ew})

    if 'FWHM' in emfeat_meas:
        fwhm = get_fwhm(model_spec)
        result_dict.update({feature_name+'_FWHM': fwhm})

    if 'flux' in emfeat_meas:
        flux = get_integrated_flux(model_spec, disp_range=disp_range)
        result_dict.update({feature_name+'_flux': flux})

    return result_dict


cont_measures_default = ['fluxden_avg',
                         'Lwav',
                         'appmag',
                         'absmag']


def analyze_continuum(specfit, model_names, rest_frame_wavelengths,
                      cosmology, redshift=None,
                      dispersion=None, cont_meas=None, width=10):
    """Calculate measurements of the continuum at a range of specified
    wavelengths for a spectral fit (SpecFit object).

    At present this analysis assumes that the spectra are in the following
    units:
    flux density - erg/s/cm^2/AA
    dispersipn - AA

    :param specfit: SpecFit class object to extract the model flux from
    :type sf.SpecFit
    :param model_names: List of model names to create the emission feature flux
    from.
    :type list
    :param rest_frame_wavelengths: Rest-frame wavelength of the emission feature
    :type float
    :param cosmology: Astropy Cosmology object
    :type astropy.cosmology.Cosmology
    :param redshift: Redshift of the object. This keyword argument defaults
    to 'None', in which case the redshift from the SpecFit object is used.
    :type float
    :param dispersion: This keyword argument allows to input a dispersion
    axis (e.g., wavelengths) for which the model fluxes are calculated. The
    value defaults to 'None', in which case the dispersion from the SpecFit
    spectrum is being used.
    :type np.array
    :param cont_meas: This keyword argument allows to specify the list of
    emission feature measurements.
    Currently possible measurements are ['peak_fluxden', 'peak_redsh', 'EW',
    'FWHM', 'flux']. The value defaults to 'None' in which all measurements
    are calculated
    :param width: Window width in dispersion units to calculate the average
    flux density in.
    :type list
    :return: Dictionary with measurement results (without units)
    :rtype dict
    """

    # Build the continuum model spectrum
    cont_spec = build_model_flux(specfit, model_names, dispersion=dispersion)

    # Define unit for flux density
    fluxden_unit = units.erg / units.s / units.cm ** 2 / units.AA

    # Analyze the continuum
    if cont_meas is None:
        cont_meas = cont_measures_default
    if redshift is None:
        redshift = specfit.redshift

    # Start analysis
    result_dict = {}

    for wave in rest_frame_wavelengths:

        wave_name = str(wave)

        if 'fluxden_avg' in cont_meas:
            fluxden_avg = get_average_fluxden(cont_spec, wave,
                                              redshift=redshift,
                                              width=width)
            result_dict.update({wave_name+'_fluxden_avg': fluxden_avg})

        if 'Lwav' in cont_meas:
            lwav = calc_Lwav_from_fwav(fluxden_avg, fluxden_unit,
                                       redshift=redshift,
                                       cosmology=cosmology)
            result_dict.update({wave_name+'_Lwav': lwav.value})

        if 'appmag' in cont_meas:
            appmag = calc_apparent_mag_from_fluxden(fluxden_avg,
                                                    fluxden_unit,
                                                    wave*(1.+redshift)*units.AA)
            result_dict.update({wave_name + '_appmag': appmag.value})

        if 'absmag' in cont_meas:
            absmag = calc_absolute_mag_from_monochromatic_luminosity(
                    lwav.value, lwav.unit, wave*units.AA)
            result_dict.update({wave_name + '_absmag': absmag.value})


    return result_dict


# EXPERIMENTAL FROM HERE

# emission_feature_listdict = [{'feature_name': None,
#                          'model_names' : None,
#                          'rest_frame_wavelength': None}]
# continuum_listdict = {'model_names': None,
#                       'rest_frame_wavelengths': None}

def analyse_fit(specfit,
                continuum_dict,
                emission_feature_dictlist,
                # absorption_feature_dictlist,
                redshift,
                cosmology,
                emfeat_meas=None,
                cont_meas=None,
                dispersion=None,
                width=10
                ):


    result_dict = {}

    # Analyze continuum properties
    cont_result = analyze_continuum(specfit,
                                    continuum_dict['model_names'],
                                    continuum_dict['rest_frame_wavelengths'],
                                    cosmology,
                                    redshift=redshift,
                                    dispersion=dispersion,
                                    cont_meas=cont_meas,
                                    width=width
                                    )

    result_dict.update(cont_result)

    # Analyze emission features
    for emission_feat_dict in emission_feature_dictlist:
        if 'disp_range' in emission_feat_dict:
            disp_range = emission_feat_dict['disp_range']
        else:
            disp_range = None
        emfeat_result = analyze_emission_feature(specfit,
                                                 emission_feat_dict[
                                                     'feature_name'],
                                                 emission_feat_dict[
                                                     'model_names'],
                                                 emission_feat_dict[
                                                     'rest_frame_wavelength'],
                                                 cont_model_names=
                                                 continuum_dict['model_names'],
                                                 redshift=redshift,
                                                 dispersion=dispersion,
                                                 emfeat_meas=emfeat_meas,
                                                 disp_range = disp_range
                                                 )
        result_dict.update(emfeat_result)


    # Analyze absorption features
    # TODO

    return result_dict


def analyse_mcmc_results(foldername, specfit,
                continuum_dict,
                emission_feature_dictlist,
                redshift,
                cosmology,
                emfeat_meas=None,
                cont_meas=None,
                dispersion=None,
                width=10, concatenate=False):

    # In the future integrate absorption featues absorption_feature_dictlist,
    # Only makes sense to analyze models together that were sampled together
    # for the continuum in the emission feature analysis use the best fit cont
    # parameters from the MCMC analysis

    mcmc_filelist = glob.glob(os.path.join(foldername,  '*_mcmc_chain.hdf5'))

    print('[INFO] Starting MCMC analysis')

    cont_analyzed = False

    # Iterate through the MCMC output files
    for mcmc_file in mcmc_filelist:
        mcmc_df = pd.read_hdf(mcmc_file)
        mcmc_columns = list(mcmc_df.columns)
        print('[INFO] Working on output file {}'.format(mcmc_file))

        # Iterate through emission feature model lists
        cont_models = continuum_dict['model_names']

        for emfeat_dict in emission_feature_dictlist:
            emfeat_models = emfeat_dict['model_names']
            cont_emfeat_models = emfeat_models + cont_models

            for idx, specmodel in enumerate(specfit.specmodels):
                prefix_list = [model.prefix for model in specmodel.model_list]

                if set(cont_emfeat_models).issubset(prefix_list):
                    params_list_keys = []
                    for params in specmodel.params_list:
                        params_list_keys.extend(list(params.keys()))
                    if set(params_list_keys).issubset(mcmc_columns):
                        print('[INFO] Analyzing continuum and emission '
                              'feature {}'.format(
                            emfeat_dict['feature_name']))
                        _mcmc_analyze(specfit, idx, mcmc_df,
                                      continuum_dict,
                                      emfeat_dict,
                                      redshift,
                                      cosmology,
                                      mode='both',
                                      emfeat_meas=emfeat_meas,
                                      cont_meas=cont_meas,
                                      dispersion=dispersion,
                                      width=width,
                                      concatenate=concatenate
                                      )
                else:
                    if set(emfeat_models).issubset(prefix_list):
                        params_list_keys = []
                        for params in specmodel.params_list:
                            params_list_keys.extend(list(params.keys()))
                        if set(params_list_keys).issubset(mcmc_columns):
                            print('[INFO] Analyzing emission feature {}'.format(
                                emfeat_dict['feature_name']))
                            _mcmc_analyze(specfit, idx, mcmc_df,
                                          continuum_dict,
                                          emfeat_dict,
                                          redshift,
                                          cosmology,
                                          mode='emfeat',
                                          emfeat_meas=emfeat_meas,
                                          cont_meas=cont_meas,
                                          dispersion=dispersion,
                                          width=width,
                                          concatenate=concatenate
                                          )

                    if set(cont_models).issubset(prefix_list) \
                            and not cont_analyzed:
                        params_list_keys = []
                        for params in specmodel.params_list:
                            params_list_keys.extend(list(params.keys()))
                        if set(params_list_keys).issubset(mcmc_columns):
                            print('[INFO] Analyzing continuum')
                            _mcmc_analyze(specfit, idx, mcmc_df,
                                          continuum_dict,
                                          emfeat_dict,
                                          redshift,
                                          cosmology,
                                          mode='cont',
                                          emfeat_meas=emfeat_meas,
                                          cont_meas=cont_meas,
                                          dispersion=dispersion,
                                          width=width,
                                          concatenate=concatenate
                                          )
                            cont_analyzed = True


def _mcmc_analyze(specfit, specmodel_index, mcmc_df, continuum_dict,
                  emission_feature_dict, redshift, cosmology, mode,
                  emfeat_meas=None, cont_meas=None, dispersion=None,
                  width=10,
                  concatenate=False):

    # Set up continuum model variables
    cont_model_names = continuum_dict['model_names']

    result_df = None

    for idx in mcmc_df.index:
        # Work on a copy of the SpecFit object, not on the original
        fit = specfit.copy()

        # Update specific SpecModel for the emission feature analysis
        mcmc_series = mcmc_df.loc[idx, :]
        fit.specmodels[
            specmodel_index].update_param_values_from_input_series(mcmc_series)

        result_dict = {}
        emfeat_result_dict = None
        cont_result_dict = None

        if mode == 'both' or mode == 'emfeat':
            feature_name = emission_feature_dict['feature_name']
            model_names = emission_feature_dict['model_names']
            rest_frame_wavelength = emission_feature_dict['rest_frame_wavelength']
            if 'disp_range' in emission_feat_dict:
                disp_range = emission_feat_dict['disp_range']
            else:
                disp_range = None

            emfeat_result_dict = \
                analyze_emission_feature(specfit, feature_name,
                                         model_names, rest_frame_wavelength,
                                         cont_model_names=cont_model_names,
                                         redshift=redshift,
                                         dispersion=dispersion,
                                         emfeat_meas=emfeat_meas,
                                         disp_range = disp_range)
        if mode == 'both' or mode == 'cont':
            cont_result_dict = \
                analyze_continuum(specfit,
                                  continuum_dict['model_names'],
                                  continuum_dict['rest_frame_wavelengths'],
                                  cosmology,
                                  redshift=redshift,
                                  dispersion=dispersion,
                                  cont_meas=cont_meas,
                                  width=width)

        if cont_result_dict is not None:
            result_dict.update(cont_result_dict)
        if emfeat_result_dict is not None:
            result_dict.update(emfeat_result_dict)

        if result_df is None:
            result_df = pd.DataFrame(data=result_dict, index=range(1))
        else:
            result_df = result_df.append(result_dict, ignore_index=True)

    if concatenate:
        result = pd.concat([mcmc_df, result_df], axis=1)
    else:
        result = result_df

    if cont_result_dict is None:
        result.to_csv('mcmc_analysis_{}.csv'.format(feature_name))
    elif emfeat_result_dict is None:
        result.to_csv('mcmc_analysis_cont.csv')
    else:
        result.to_csv('mcmc_analysis_cont_{}.csv'.format(feature_name))


def analyse_resampled_results(specfit, resampled_df, continuum_dict,
                              emission_feature_dictlist, redshift, cosmology,
                              emfeat_meas=None, cont_meas=None, dispersion=None,
                              width=10, concatenate=False):

    # Test if all necessary columns are inlcuded in the resampled file
    fit_columns = resampled_df.columns
    all_specfit_params = []

    for specmodel in specfit.specmodels:
        params_list = specmodel.params_list
        for params in params_list:
            for param in params:
                all_specfit_params.append(param)

    if set(all_specfit_params).issubset(fit_columns):
        print('[INFO] Resampled results contain necessary column information.')
        print('[INFO] Proceeding with analysis.')
        _resampled_analyze(specfit, resampled_df, continuum_dict,
                           emission_feature_dictlist, redshift, cosmology,
                           emfeat_meas=emfeat_meas, cont_meas=cont_meas,
                           dispersion=dispersion,
                           width=width, concatenate=concatenate)
    else:
        print('[ERROR] Resampled results do NOT contain necessary column '
              'information.')
        print('[ERROR] Please double check if the correct file was supplied.')



def _resampled_analyze(specfit, resampled_df, continuum_dict,
                              emission_feature_dictlist, redshift, cosmology,
                              emfeat_meas=None, cont_meas=None, dispersion=None,
                              width=10, concatenate=False):

    result_df = None

    for idx in resampled_df.index:

        # Work on a copy of the original SpecFit object
        fit = specfit.copy()

        # Update all SpecModels for the analysis
        resampled_series = resampled_df.loc[idx, :]
        for specmodel in fit.specmodels:
            specmodel.update_param_values_from_input_series(resampled_series)

        result_dict = analyse_fit(specfit,
                                  continuum_dict,
                                  emission_feature_dictlist,
                                  # absorption_feature_dictlist,
                                  redshift,
                                  cosmology,
                                  emfeat_meas=emfeat_meas,
                                  cont_meas=cont_meas,
                                  dispersion=dispersion,
                                  width=width
                                  )

        if result_df is None:
            result_df = pd.DataFrame(data=result_dict, index=range(1))
        else:
            result_df = result_df.append(result_dict, ignore_index=True)

    if concatenate:
        result = pd.concat([mcmc_df, result_df], axis=1)
    else:
        result = result_df

    result.to_csv('resampled_analysis.csv')

# ------------------------------------------------------------------------------
# Model Spectrum Generation
# ------------------------------------------------------------------------------


# build model flux
def build_model_flux(specfit, model_list, dispersion=None):
    """
    Build the model flux from a specified list of models that exist in the
    SpecModels of the SpecFit object.

    The dispersion axis for the model flux can be specified as a keyword
    argument.

    :param specfit: SpecFit class object to extract the model flux from
    :type sf.SpecFit
    :param model_list: List of model names to create the model flux from. The
    models must be exist in the SpecModel objects inside the SpecFit object.
    :type list
    :param dispersion: New dispersion to create the model flux for
    :type np.array
    :return: SpecOneD objects with the model flux
    :rtype: sod.SpecOneD
    """

    if dispersion is None:
        dispersion = specfit.spec.dispersion

    model_fluxden = np.zeros_like(dispersion)

    for specmodel in specfit.specmodels:
        for idx, model in enumerate(specmodel.model_list):
            if model.prefix in model_list:
                model_fluxden += model.eval(specmodel.params_list[idx],
                                            x=dispersion)

    model_spec = sod.SpecOneD(dispersion=dispersion, fluxden=model_fluxden,
                              unit='f_lambda')

    return model_spec


# ------------------------------------------------------------------------------
# General model measurements
# ------------------------------------------------------------------------------

def get_integrated_flux(input_spec, disp_range=None):
    """
    Calculate the integrated flux of a spectrum.

    The keyword argument disp_range allows to specify the dispersion
    boundaries for the integration. The standard numpy.trapz function is used
    for the integration.

    :param input_spec: SpecOneD object holding the spectrum to integrate
    :type sod.SpecOneD
    :param disp_range: 2 element list holding the lower and upper dispersion
    boundaries for the integration
    :type list
    :return: integrated flux
    :rtype: float
    """

    if disp_range is not None:
        spec = input_spec.trim_dispersion(disp_range, inplace=False)
    else:
        spec = input_spec.copy()

    return np.trapz(spec.fluxden, x=spec.dispersion)


def get_average_fluxden(input_spec, dispersion, width=10, redshift=0):
    """
    Calculate the average flux density of a spectrum in a window centered at
    the specified dispersion and with a given width.

    The central dispersion and width can be specified in the rest-frame and
    the redshifted to the observed frame using the redsh keyword argument.

    :param input_spec: Input spectrum
    :type sod.SpecOneD
    :param dispersion: Central dispersion
    :type float
    :param width: Width of the dispersion window
    :type float
    :param redsh: Redshift argument to redshift the dispersion window into
    the observed frame.
    :return: Average flux density
    :rtype: float
    """

    disp_range = [(dispersion - width / 2.)*(1.+redshift),
                  (dispersion + width / 2.)*(1.+redshift)]
    
    return input_spec.average_fluxden(disp_range=disp_range)
        

# ------------------------------------------------------------------------------
# Emission line model measurements
# ------------------------------------------------------------------------------

def get_peak_redshift(input_spec, rest_wave):
    """
    Calculate the redshift of the flux density peak in the spectrum by
    specifying the expected rest frame wavelength of the emission feature.

    :param input_spec: Input spectrum
    :type sod.SpecOneD
    :param rest_wave: Rest-frame wavelength of the expected emission feature.
    :return: Redshift of the peak flux density
    :rtype: float
    """

    return input_spec.peak_dispersion()/rest_wave - 1


def get_equivalent_width(cont_spec, line_spec, disp_range=None,
                         redshift=0):
    """
    Calculate the rest-frame equivalent width of a spectral feature.

    Warning: this function currently only works for spectra in wavelength
    units. For spectra in frequency units the conversion to rest-frame will
    be incorrect.

    :param cont_spec: Continuum spectrum
    :type sod.SpecOneD
    :param line_spec: Spectrum with the feature (e.g. emission line)
    :type sod.SpecOneD
    :param disp_range: Dispersion range (2 element list of floats) over which
    the equivalent width will be calculated.
    :type list
    :param redshift: Redshift of the astronomical source
    :return: Rest-frame equivalent width
    :rtype: float
    """

    # TODO: Implement frequency unit support.

    rest_dispersion = cont_spec.dispersion / (1+redshift)
    rest_cont_flux = cont_spec.fluxden * (1+redshift)
    rest_line_flux = line_spec.fluxden * (1+redshift)

    if disp_range is not None:
        l_idx = np.argmin(np.abs(rest_dispersion - limits[0]))
        u_idx = np.argmin(np.abs(rest_dispersion - limits[1]))

        ew = np.trapz((rest_line_flux[l_idx:u_idx])/rest_cont_flux[l_idx:u_idx],
                      rest_dispersion[l_idx:u_idx])
    else:
        ew = np.trapz((rest_line_flux) / rest_cont_flux,
                      rest_dispersion)

    return ew


def get_fwhm(input_spec, disp_range=None, resolution=None):
    """
    Calculate the FWHM (in km/s) of an emission feature from the spectrum.

    The user can specify a dispersion range to limit the FWHM calculation to
    this part of the spectrum. If a resolution (R) is specified the FWHM is
    corrected for the broadening due to the resolution.

    The function will subtract a flux density value of half of the maximum
    and then find the two roots (flux density = 0) of the new flux density axis.
    If the emission feature has multiple components more than two roots can
    be found in which case the a np.NaN value will be returned.

    :param input_spec: Input spectrum
    :type sod.SpecOneD
    :param disp_range: Dispersion range to which the calculation is limited.
    :param resolution: Resolution in R = Lambda/Delta Lambda
    :type float
    :return: FWHM of the spectral feature
    :rtype: float
    """

    if disp_range is not None:
        spec = input_spec.trim_dispersion(disp_range, inplace=False)
    else:
        spec = input_spec.copy()

    fluxden = spec.fluxden
    dispersion = spec.dispersion

    spline = UnivariateSpline(dispersion,
                              fluxden - np.max(fluxden) / 2.,
                              s=0)

    roots = spline.roots()

    if len(roots) > 2 or len(roots) < 2:
        print('[ERROR] Found {} roots. Cannot determine FWHM'.format(len(
            roots)))
        print('[ERROR] Setting FWHM to NaN')
        # plt.cla()
        # plt.plot(dispersion, fluxden - np.max(fluxden) / 2., 'r-',
        #          lw=2)
        # plt.plot(dispersion, dispersion*0, 'k--')
        # plt.show()
        return np.NaN
    else:

        max_idx = np.where(fluxden == np.max(fluxden))
        max_wav = dispersion[max_idx]

        # plt.plot(dispersion, fluxden)
        # plt.plot(dispersion, spline(dispersion))
        # plt.show()

        fwhm = (abs(roots[0] - max_wav) + abs(roots[1] - max_wav)) / max_wav * \
               c_km_s

        if resolution is None:
            return fwhm[0]
        else:
            print('[INFO] FWHM is corrected for the provided '
                  'resolution of R={}'.format(resolution))
            resolution_km_s = c_km_s/resolution
            return np.sqrt(fwhm ** 2 - resolution_km_s ** 2)[0]

# non-parametric width measurement

def get_centroid_wavelength():
    pass

def get_centroid_redshift():
    pass

# ------------------------------------------------------------------------------
# Astrophysical spectral measurements
# ------------------------------------------------------------------------------

def calc_Lwav_from_fwav(fluxden, fluxden_unit, redshift, cosmology):
    """Calculate the monochromatic luminosity from the monochromatic flux
    density.

    :param fluxden: Monochromatic flux density at a given wavelength.
    :type float
    :param fluxden_unit: Unit of the flux density.
    :type astropy.units.Quantity
    :param redshift: Redshift of the source.
    :type float
    :param cosmology: Cosmology as an astropy Cosmology object
    :type astropy.cosmology.Cosmology
    :return: Monochromatic luminosity in units of erg s^-1 Angstroem^-1
    :rtype: astropy.units.Quantity
    """

    # Calculate luminosity distance in Mpc
    lum_distance = cosmology.luminosity_distance(redshift)

    return (fluxden * fluxden_unit * \
           lum_distance**2 * 4 * np.pi * (1. + redshift)).decompose(
        bases=[units.erg, units.s, units.AA])


def calc_integrated_luminosity(input_spec, fluxden_unit, redshift,
                                    cosmology, disp_range=None):
    """Calculate the integrated model spectrum luminosity. 
    
    :param input_spec: Input spectrum
    :type sod.SpecOneD
    :param fluxden_unit: Unit of the flux density.
    :type astropy.units.Quantity
    :param redshift: Redshift of the source.
    :type float
    :param cosmology: Cosmology as an astropy Cosmology object
    :type astropy.cosmology.Cosmology
    :param disp_range: 2 element list holding the lower and upper dispersion
    boundaries for the integration
    :type list
    :return: Return the integrated luminosity for (dispersion range in) the
    input spectrum.
    :rtype: astropy.units.Quantity
    """

    # Cconvert to rest-frame flux and dispersion
    rest_dispersion = input_spec.dispersion / (1. + redshift)
    rest_fluxden = calc_Lwav_from_fwav(input_spec.fluxden,
                                       fluxden_unit, redshift,
                                       cosmology)
    rest_frame_spec = sod.SpecOneD(dispersion=rest_dispersion,
                                fluxden=rest_fluxden.value,
                        unit='f_lam')

    # Only integrate part of the model spectrum if disp_range is specified
    if disp_range is not None:
        rest_frame_spec.trim_dispersion(disp_range, inplace=True)

    # Integrate the model flux
    integrated_line_luminosity = np.trapz(rest_frame_spec.fluxden,
                                          x=rest_frame_spec.dispersion)
    
    return integrated_line_luminosity * rest_fluxden.unit * units.AA

def calc_apparent_mag_from_fluxden(fluxden, fluxden_unit, dispersion):
    """Calculate the apparent AB magnitude from the monochromatic flux
    density at a specified dispersion value.

    :param fluxden: Monochromatic flux density at a given wavelength.
    :type float
    :param fluxden_unit: Unit of the flux density.
    :type astropy.units.Quantity
    :param dispersion: Dispersion value (usually in wavelength).
    :type float
    :return: Returns apparent AB magnitude.
    :rtype: float
    """

    f_nu = fluxden * fluxden_unit * (dispersion)**2 / const.c

    value = (f_nu/units.ABflux).decompose()

    return -2.5*np.log10(value)

# TODO: In a future update that includes astrophysical unit support the
#  following functions should be updated.
# TODO: In a future update consider allowing to specify a custom k-correction
#  function to allow for arbitrary k-corrections.

def calc_absolute_mag_from_fluxden(fluxden, fluxden_unit, dispersion,
                                   cosmology, redshift, kcorrection=True,
                                   a_nu=0):
    """Calculate the absolute AB magnitude from the monochromatic flux density
    at a given dispersion value.

    :param fluxden: Monochromatic flux density at a given wavelength.
    :type float
    :param fluxden_unit: Unit of the flux density.
    :type astropy.units.Quantity
    :param dispersion: Dispersion value (usually in wavelength).
    :type float
    :param cosmology: Cosmology as an astropy Cosmology object.
    :type astropy.cosmology.Cosmology
    :param redshift: Redshift of the source.
    :type float
    :param kcorrection: Boolean to indicate whether the magnitude should be
    k-corrected assuming a power law spectrum. This keyword argument defaults
    to 'True'.
    :type bool
    :param a_nu: Power law slope as a function of frequency for the
    k-correction. This defaults to '0', appropriate for monochromatic
    measurements.
    :type float
    :return: Absolute AB magnitude (monochromatic)
    :rtype: float
    """

    abmag = calc_apparent_mag_from_fluxden(fluxden, fluxden_unit, dispersion)

    return calc_absolute_mag_from_apparent_mag(abmag, cosmology, redshift,
                                        kcorrection, a_nu)


def calc_absolute_mag_from_monochromatic_luminosity(l_wav, l_wav_unit,
                                                    wavelength):
    """Calculate the absolute monochromatic magnitude from the monochromatic
    luminosity per wavelegnth.

    :param l_wav: Monochromatic luminosity per wavelength
    :type float
    :param l_wav_unit: Unit of the monochromatic luminosity
    :type astropy.units.Quantity
    :param wavelength: Wavelength of the monochromatic luminosity
    :type float
    :return: Absolute monochromatic magnitude
    :rtype: float
    """
    l_nu = l_wav * l_wav_unit * (wavelength)**2 / const.c / 4 / \
                  np.pi

    value = (l_nu / units.ABflux / (10 * units.pc)**2).decompose()

    return -2.5 * np.log10(value)


def calc_absolute_mag_from_apparent_mag(appmag, cosmology, redshift,
                                        kcorrection=False, a_nu=0):
    """Calculate the absolute magnitude from the apparent magnitude using a
    power law k-correction.

    :param appmag: Apparent AB magnitude
    :param cosmology: Cosmology as an astropy Cosmology object.
    :type astropy.cosmology.Cosmology
    :param redshift: Redshift of the source.
    :type float
    :param kcorrection: Boolean to indicate whether the magnitude should be
    k-corrected assuming a power law spectrum. This keyword argument defaults
    to 'True'.
    :param a_nu: Power law slope as a function of frequency for the
    k-correction. This defaults to '0', appropriate for monochromatic
    measurements.
    :return: Absolute AB magnitude (monochromatic)
    :rtype: float
    """

    if kcorrection == True:
        kcorr = k_correction_pl(redshift, a_nu)
    else:
        kcorr = 0

    lum_dist = cosmology.luminosity_distance(redshift)
    distmod = cosmology.distmod(redshift)

    print(lum_dist, distmod, kcorr)

    return appmag - distmod.value - kcorr

def k_correction_pl(redshift, a_nu):
    """Calculate the k-correction for a power law spectrum with spectral
    index (per frequency) a_nu.

    :param redshift: Cosmological redshift of the source
    :type float
    :param a_nu: Power law index (per frequency)
    :type float
    :return: K-correction
    :rtype: float
    """
    # Hogg 1999, eq. 27
    return -2.5 * (1. + a_nu) * np.log10(1.+redshift)

# ------------------------------------------------------------------------------
# Quasar/AGN specific function - Will be moved to separate module
# ------------------------------------------------------------------------------


def calc_velocity_shifts():

    # calculate blueshift compared to given rest_wavelength
    dv = lu.dv_from_z(line_z, redshift, rel=True).value

    pass


def calc_iron_over_mgII_ratio():

    feII_over_mgII = result_dict['FeII_flux'] / result_dict['MgII_flux']
    result_dict['FeII_over_MgII'] = feII_over_mgII
    feII_over_mgII_L = result_dict['FeII_L'] / result_dict['MgII_L']
    result_dict['FeII_over_MgII_L'] = feII_over_mgII_L

    pass


def calc_eddington_ratio():

    edd_lum = calc_Edd_luminosity(bhmass)
    edd_lum_name = name + '_EddLumR_' + str(wave) + '_' + ref
    result_dict[edd_lum_name] = L_bol / edd_lum

    pass


def calc_black_hole_mass(emline_width, emline_name, cont_Lwav, cont_wav,
                         verbosity=2, emline_L=None, scaling_relation=None):


    zp = None
    reference = None
    result_dict = {}

    if emline_name == 'MgII':
        if scaling_relation is None or scaling_relation == 'VW09':
            # See equation(1) in Vestergaard & Osmer 2009 for MgII
            reference = 'VW09'

            if cont_wav == 1350:
                zp = 6.72
            elif cont_wav == 2100:
                zp = 6.79
            elif cont_wav == 3000:
                zp = 6.86
            elif cont_wav == 5100:
                zp = 6.96

        else:
            if verbosity > 1:
                print('[ERROR] Continuum wavelength of {} does not '
                      'allow for BH mass calculation from the {} '
                      'emission line using the {} scaling relation.'.format(
                       cont_wav, emline_name, reference))
        if zp is not None:
            bhmass = 10 ** zp * (emline_width / 1000.) ** 2 * (
                        cont_wav * cont_Lwav / 10 ** 44) ** (0.5)
            result_dict.update({'BHmass_{}_{}'.format(reference, str(cont_Lwav))
                               : bhmass})

    elif emline_name == 'Hbeta':
        if scaling_relation is None or scaling_relation == 'VO06':
            # See equations (5, 7) in Vestergaard & Peterson for Hbeta and CIV
            reference = "VO06"

            if cont_wav == 5100:
                bhmass = 10**6.91 * (emline_width/1000.)**2 \
                        * (cont_wav * cont_Lwav / 10**44)**0.5
                result_dict.update(
                    {'BHmass_{}_{}'.format(reference, str(cont_Lwav))
                     : bhmass})

            else:
                if verbosity > 1:
                    print('[ERROR] Continuum wavelength of {} does not '
                          'allow for BH mass calculation from the {} '
                          'emission line using the {} scaling relation.'.format(
                        cont_wav, emline_name, reference))

        if scaling_relation is None or scaling_relation == 'MJ02':
            reference = 'MJ02'

            if wav == 5100:
                zp = np.log10(4.74E+6)
                bhmass = 10 ** zp * (emline_width / 1000.) ** 2 * (
                        cont_wav * cont_Lwav / 10 ** 44) ** 0.61
                result_dict.update(
                    {'BHmass_{}_{}'.format(reference, str(cont_Lwav))
                     : bhmass})
            else:
                if verbosity > 1:
                    print('[ERROR] Continuum wavelength of {} does not '
                          'allow for BH mass calculation from the {} '
                          'emission line using the {} scaling relation.'.format(
                        cont_wav, emline_name, reference))


    elif emline_name == 'CIV':
        if scaling_relation is None or scaling_relation == 'VO06':
            # See equations (5, 7) in Vestergaard & Peterson for Hbeta and CIV
            reference = 'VO06'

            if wav == 1350:
                bhmass = 10**6.66 * (emline_width/1000.)**2 *\
                         (cont_wav * cont_Lwav / 10**44)**0.53
                result_dict.update(
                    {'BHmass_{}_{}'.format(reference, str(cont_Lwav))
                     : bhmass})

        else:
            if verbosity > 1:
                print('[ERROR] Continuum wavelength of {} does not '
                      'allow for BH mass calculation from the {} '
                      'emission line using the {} scaling relation.'.format(
                       cont_wav, emline_name, reference))

        if scaling_relation is None or scaling_relation == 'Co17':
            reference = 'Co17'

            if wav == 1350:
                bhmass = 10 ** 6.71 * (emline_width / 1000.) ** 2 * (
                        cont_wav * cont_Lwav / 10 ** 44) ** 0.53
                result_dict.update(
                    {'BHmass_{}_{}'.format(reference, str(cont_Lwav))
                     : bhmass})

            else:
                if verbosity > 1:
                    print('[ERROR] Continuum wavelength of {} does not '
                          'allow for a BH mass calculation from the {} '
                          'emission line using the {} scaling relation.'.format(
                        cont_wav, emline_name, reference))

    elif emline_name == 'Halpha':
        reference = 'GH05'

        if scaling_relation is None or scaling_relation == 'GH05':

            if (not np.isnan(emline_L) or emline_L is not None):
                bhmass =  2.0 * 1e+6 * (emline_width / 1000) ** 2.06 \
                          * (emline_L /10 ** 42) ** 0.55
                result_dict.update(
                    {'BHmass_{}_{}'.format(reference, str(cont_Lwav))
                     : bhmass})

        else:
            if verbosity > 1:
                print('[ERROR] Emission line luminosity {} does not'
                      'allow for a BH mass calculation from the {} '
                      'emission line using the {} scaling relation.'.format(
                    emline_L, emline_name, reference))

    return result_dict


def calc_bolometric_luminosity(cont_Lwav, cont_wav, reference=None):

    result_dict = {}

    if cont_wav == 3000 and (reference is None or reference == 'Shen2011'):
        # Should the reference be Shen2001 or Richards2006
        Lbol = cont_Lwav * 5.15
        result_dict.update({'Lbol_{}_{}'.format(reference, str(cont_wav)):
                                Lbol})

    if reference is None or reference == 'Ne19':
        # Derive bolometric luminosities for quasars using the Netzer 2019
        # bolometric corrections.Values are taken from Table 1 in Netzer 2019.
        # Possibly only valid for fainter AGN!

        if cont_wav in [1400, 3000, 5100]:

            if cont_wav == 1400:
                c = 7
                d = -0.1
            elif cont_wav == 3000:
                c = 19
                d = -0.2
            elif cont_wav == 5100:
                c = 40
                d = -0.2

            k_bol = c * (cont_Lwav / 10 ** 42) ** d

            Lbol = cont_Lwav * k_bol
            result_dict.update({'Lbol_{}_{}'.format(reference, str(cont_wav)):
                                    Lbol})

        else:
            print('[ERROR] Bolometric correction for specified wavelength not '
                  'available.')

    # Add Shen 2020 bolometric corrections (?)

    return result_dict


def calc_Edd_luminosity(bh_mass):

    factor = (4 * np.pi * const.G * const.c * const.m_p) / const.sigma_T
    factor = factor.to(units.erg / units.s / units.Msun).value

    return factor * bh_mass

def correct_CIV_fwhm(fwhm, blueshift):
    """
    Correct the CIV FWHM according to Coatman et al. 2017 Eq. 4

    :param fwhm: float
        CIV FWHM in km/s
    :param blueshift: float
        CIV blueshift in km/s (not velocity shift, use correct sign)
    :return:
    """

    return fwhm / (0.41 * blueshift/1000 + 0.62)


#
# def calc_QSO_Lbol_Ne19(L_wav, wav):
#     """
#
#     :param L_wav:
#     :param wav:
#     :return:
#     """
#
#     if wav in [1400, 3000, 5100]:
#
#         if wav == 1400:
#             c = 7
#             d = -0.1
#         elif wav == 3000:
#             c = 19
#             d = -0.2
#         elif wav == 5100:
#             c = 40
#             d = -0.2
#
#         k_bol = c * (L_wav/ 10 ** 42)**d
#
#         return L_wav * k_bol
#
#     else:
#         print('[ERROR] Bolometric correction for specified wavelength not '
#               'available.')




#
# def calc_CIV_BHmass_Co17(L_wav, wav, fwhm, verbosity=1):
#
#     reference = "Co17"
#
#     if wav == 1350:
#         return 10 ** 6.71 * (fwhm / 1000.) ** 2 * (
#                     wav * L_wav / 10 ** 44) ** (0.53), reference
#     else:
#         if verbosity > 1:
#             print("Specified wavelength does not allow for BH mass "
#                   "calculation with Hbeta", wav)

#
# def calc_BHmass_VP06VO09(L_wav, wav, fwhm, line="MgII", verbosity=2):
#
#     """ Calculate black hole mass according to the empirical relations
#     of Vestergaard & Peterson 2006 and Vestergaard & Osmer 2009
#
#
#     See equations (5, 7) in Vestergaard & Peterson for Hbeta and CIV
#
#     Parameters:
#     -----------
#     L_wav : float
#         Monochromatic luminosity at the given wavelength of wav in erg s^-1
#         Angstroem^-1
#
#     wav : float or int
#         Wavelength for the given monochromatic luminosity
#
#     fwhm : float
#         Full width at half maximum of the emission line specified in line
#
#     line : string
#         Name of the emission line to calculate the black hole mass from. Possible
#         line names are MgII, Hbeta, CIV
#
#     Returns:
#     --------
#
#
#     BHmass : float
#     Black hole mass in units for solar masses
#
#     """
#     zp = None
#
#     if line == "MgII":
#         reference = "VW09"
#
#         if wav == 1350:
#             zp = 6.72
#
#         elif wav == 2100:
#             zp = 6.79
#
#         elif wav == 3000:
#             zp = 6.86
#
#         elif wav == 5100:
#             zp = 6.96
#
#         else:
#             if verbosity > 1:
#                 print("Specified wavelength does not allow for BH mass "
#                        "calculation with MgII", wav)
#         if zp is not None:
#             return 10**zp * (fwhm/1000.)**2 * (wav * L_wav / 10**44)**(0.5), \
#                reference
#         # else:
#         #     return None, None
#
#     elif line == "Hbeta":
#         reference = "VO06"
#
#         if wav == 5100:
#             return 10**6.91 * (fwhm/1000.)**2 * (wav * L_wav / 10**44)**(0.5), \
#                    reference
#
#         else:
#             if verbosity > 1:
#                 print ("Specified wavelength does not allow for BH mass "
#                        "calculation with Hbeta", wav)
#             # return None, None
#
#     elif line == "CIV":
#         reference = "VO06"
#
#         if wav == 1350:
#             return 10**6.66 * (fwhm/1000.)**2 * (wav * L_wav / 10**44)**(0.53),\
#                    reference
#
#         else:
#             if verbosity > 1:
#                 print ("Specified wavelength does not allow for BH mass "
#                        "calculation with CIV", wav)
#
#     else:
#         if verbosity >1:
#             print("[Warning] No relation exists to calculate the BH mass for "
#                   "the specified line ({}) and wavelength ({}): ".format(
#                 line, wav))
#
#     return None, None
#
# def calc_BH_masses(L_wav, wav, fwhm, line="MgII", verbosity=2):
#
#     """ Calculate black hole mass according to the empirical relations
#     of Vestergaard & Peterson 2006 and Vestergaard & Osmer 2009
#
#     See equation (1) in Vestergaard & Osmer 2009 for MgII
#     See equations (5, 7) in Vestergaard & Peterson for Hbeta and CIV
#
#     Parameters:
#     -----------
#     L_wav : float
#         Monochromatic luminosity at the given wavelength of wav in erg s^-1
#         Angstroem^-1
#
#     wav : float or int
#         Wavelength for the given monochromatic luminosity
#
#     fwhm : float
#         Full width at half maximum of the emission line specified in line
#
#     line : string
#         Name of the emission line to calculate the black hole mass from. Possible
#         line names are MgII, Hbeta, CIV
#
#     Returns:
#     --------
#
#
#     BHmass : float
#     Black hole mass in units for solar masses
#
#     """
#     zp = None
#     b = None
#
#     if line == "MgII":
#         reference = "VW09"
#
#         if wav == 1350:
#             zp = 6.72
#
#         elif wav == 2100:
#             zp = 6.79
#
#         elif wav == 3000:
#             zp = [6.86, 6.74]
#             b = [0.5, 0.62]
#             reference = ['VW09', 'S11']
#
#         elif wav == 5100:
#             zp = 6.96
#
#         else:
#             if verbosity > 1:
#                 print("Specified wavelength does not allow for BH mass "
#                        "calculation with MgII", wav)
#         if zp is not None and b is None:
#             return 10**zp * (fwhm/1000.)**2 * (wav * L_wav / 10**44)**(0.5), \
#                reference
#         elif zp is not None and b is not None:
#
#             bhmass_a = 10**zp[0] * (fwhm/1000.)**2 * (wav * L_wav / 10**44)**(
#                 b[0])
#             bhmass_b = 10 ** zp[1] * (fwhm / 1000.) ** 2 * (
#                         wav * L_wav / 10 ** 44) ** (
#                            b[1])
#             bhmass = [bhmass_a, bhmass_b]
#
#             return bhmass, reference
#
#
#
#     elif line == "Hbeta":
#         reference = "VO06"
#
#         if wav == 5100:
#             zp = [6.91, np.log10(4.74E+6)]
#             b = [0.5, 0.61]
#             reference = ["VO06", "MJ02"]
#             # return 10**6.91 * (fwhm/1000.)**2 * (wav * L_wav / 10**44)**(0.5), \
#             #        reference
#         # elif
#         else:
#             if verbosity > 1:
#                 print ('[INFO] Specified wavelength {} does not allow for BH '
#                        'mass calculation with Hbeta'.format(wav))
#             # return None, None
#
#
#         if zp is not None and b is None:
#             return 10**zp * (fwhm/1000.)**2 * (wav * L_wav / 10**44)**(0.5), \
#                reference
#         elif zp is not None and b is not None:
#
#             bhmass_a = 10**zp[0] * (fwhm/1000.)**2 * (wav * L_wav / 10**44)**(
#                 b[0])
#             bhmass_b = 10 ** zp[1] * (fwhm / 1000.) ** 2 * (
#                         wav * L_wav / 10 ** 44) ** (
#                            b[1])
#             bhmass = [bhmass_a, bhmass_b]
#
#             return bhmass, reference
#
#     elif line == "CIV":
#         reference = "VO06"
#
#         if wav == 1350:
#             if not np.isnan(fwhm):
#                 return 10**6.66 * (fwhm/1000.)**2 * (wav * L_wav / 10**44)**(0.53),\
#                        reference
#             else:
#                 return np.NaN, reference
#
#         else:
#             if verbosity > 1:
#                 print ("Specified wavelength does not allow for BH mass "
#                        "calculation with CIV", wav)
#
#     else:
#         if verbosity > 1:
#             print("[Warning] No relation exists to calculate the BH mass for "
#                   "the specified line ({}) and wavelength ({}): ".format(
#                 line, wav))
#
#
#     return None, None
#
#
# def calc_Halpha_BH_mass(L_Halpha, FWHM_Halpha):
#
#     # Implementation after Greene & Ho 2005 Equation 6
#     reference = "GH05"
#
#     if not np.isnan(FWHM_Halpha) or not np.isnan(L_Halpha):
#
#         return 2.0 * 1e+6 * (FWHM_Halpha/1000)**2.06 * (L_Halpha/
#                                                        10**42)**0.55, \
#                reference
#
#







