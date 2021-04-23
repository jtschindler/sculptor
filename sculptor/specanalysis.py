#!/usr/bin/env python

"""
The SpecAnalysis module provides a range of functions for spectral analsysis
in the context of the Sculptor packages SpecOneD, SpecFit, and SpecModel
classes.
"""


import os
import glob
import numpy as np
import pandas as pd
from sculptor import specfit as sf
from sculptor import speconed as sod
from astropy.table import QTable, hstack
from astropy import constants as const
from astropy import units
from astropy.cosmology import FlatLambdaCDM

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
                           'flux',
                           'lum']


def analyze_emission_feature(specfit, feature_name, model_names,
                             rest_frame_wavelength, cont_model_names=None,
                             redshift=None, dispersion=None,
                             emfeat_meas=None, disp_range=None, cosmology=None):
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
    :return: Dictionary with measurement results (with units)
    :rtype: dict
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

    if cosmology is None:
        cosmology = FlatLambdaCDM(H0=70, Om0=0.3, Tcmb0=2.725)
        print('[WARNING] No Cosmology specified. Assuming default '
              'FlatLambdaCDM cosmology with H0=70, Om0=0.3, Tcmb0=2.725.')

    if 'peak_fluxden' in emfeat_meas:
        # Calculate the peak flux density
        result_dict.update({feature_name+'_peak_fluxden':
                            np.max(model_spec.fluxden) *
                            model_spec.fluxden_unit})

    if 'peak_redsh' in emfeat_meas:
        # Calculate the peak redshift
        result_dict.update({feature_name + '_peak_redsh':
                            get_peak_redshift(model_spec,
                                              rest_frame_wavelength)})
    if 'EW' in emfeat_meas and cont_model_names is not None:
        # Calculate the rest-frame equivalent width
        ew = get_equivalent_width(cont_spec, model_spec,
                                        redshift=redshift)
        result_dict.update({feature_name + '_EW': ew})

    if 'FWHM' in emfeat_meas:
        # Calculate the (composite) line model FWHM.
        fwhm = get_fwhm(model_spec)
        result_dict.update({feature_name+'_FWHM': fwhm})

    if 'flux' in emfeat_meas:
        # Calculate the integrated line flux
        flux = get_integrated_flux(model_spec, disp_range=disp_range)
        result_dict.update({feature_name+'_flux': flux})

    if 'lum' in emfeat_meas:
        # Calculate the integrated line luminosity
        lum = calc_integrated_luminosity(model_spec,
                                         redshift=redshift, cosmology=cosmology)
        result_dict.update({feature_name + '_lum': lum})

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
    :type specfit: sculptor.specfit.SpecFit
    :param model_names: List of model names to create the emission feature flux
        from.
    :type model_names: list(string)
    :param rest_frame_wavelengths: Rest-frame wavelength of the emission feature
    :type rest_frame_wavelengths: list(float)
    :param cosmology: Astropy Cosmology object
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
    cont_spec = build_model_flux(specfit, model_names, dispersion=dispersion)

    # Define unit for flux density
    fluxden_unit = cont_spec.fluxden_unit

    # Analyze the continuum
    if cont_meas is None:
        cont_meas = cont_measures_default
    if redshift is None:
        redshift = specfit.redshift

    # Start analysis
    result_dict = {}

    for wave in rest_frame_wavelengths:

        wave_name = str(wave)
        fluxden_avg = get_average_fluxden(cont_spec, wave,
                                          redshift=redshift,
                                          width=width)

        if 'fluxden_avg' in cont_meas:
            result_dict.update({wave_name+'_fluxden_avg':
                                fluxden_avg*cont_spec.fluxden_unit})

        if 'Lwav' in cont_meas:
            lwav = calc_lwav_from_fwav(fluxden_avg,
                                       redshift=redshift,
                                       cosmology=cosmology)
            result_dict.update({wave_name+'_Lwav': lwav})

        if 'appmag' in cont_meas:
            appmag = calc_apparent_mag_from_fluxden(fluxden_avg,
                                                    wave*(1.+redshift)*units.AA)
            result_dict.update({wave_name + '_appmag': appmag})

        if 'absmag' in cont_meas:
            absmag = calc_absolute_mag_from_monochromatic_luminosity(
                    lwav, wave*units.AA)
            result_dict.update({wave_name + '_absmag': absmag})


    return result_dict


# TODO: Finalize the analyze, mcmc, and resample analysis functions.

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
    """
    THIS FUNCTION IS CURRENTLY UNDER ACTIVE DEVELOPMENT.
    """


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
    """
    THIS FUNCTION IS CURRENTLY UNDER ACTIVE DEVELOPMENT.
    """

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
    """
    THIS FUNCTION IS CURRENTLY UNDER ACTIVE DEVELOPMENT.
    """

    # Set up continuum model variables
    cont_model_names = continuum_dict['model_names']

    # result_df = None
    result_table = None

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
            if 'disp_range' in emission_feature_dict:
                disp_range = emission_feature_dict['disp_range']
            else:
                disp_range = None

            emfeat_result_dict = \
                analyze_emission_feature(specfit, feature_name,
                                         model_names, rest_frame_wavelength,
                                         cont_model_names=cont_model_names,
                                         redshift=redshift,
                                         dispersion=dispersion,
                                         emfeat_meas=emfeat_meas,
                                         disp_range=disp_range,
                                         cosmology=cosmology)
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

        # if result_df is None:
        #     result_df = pd.DataFrame(data=result_dict, index=range(1))
        # else:
        #     result_df = result_df.append(result_dict, ignore_index=True)

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

    if concatenate:
        mcmc_table = QTable.from_pandas(mcmc_df)
        result = hstack([mcmc_table, result_table])
        # result = pd.concat([mcmc_df, result_df], axis=1)
    else:
        result = result_table

    if cont_result_dict is None:
        # result.to_csv('mcmc_analysis_{}.csv'.format(feature_name))
        result.write('mcmc_analysis_{}.csv'.format(feature_name),
                           format='ascii.ecsv', overwrite=True)
    elif emfeat_result_dict is None:
        # result.to_csv('mcmc_analysis_cont.csv')
        result.write('mcmc_analysis_cont.csv',
                           format='ascii.ecsv', overwrite=True)
    else:
        # result.to_csv('mcmc_analysis_cont_{}.csv'.format(feature_name))
        result.write('mcmc_analysis_cont_{}.csv'.format(feature_name),
                           format='ascii.ecsv', overwrite=True)


def analyse_resampled_results(specfit, resampled_df, continuum_dict,
                              emission_feature_dictlist, redshift, cosmology,
                              emfeat_meas=None, cont_meas=None, dispersion=None,
                              width=10, concatenate=False):
    """
    THIS FUNCTION IS CURRENTLY UNDER ACTIVE DEVELOPMENT.
    """

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
    """
    THIS FUNCTION IS CURRENTLY UNDER ACTIVE DEVELOPMENT.
    """
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
    :type specfit: sculptor.specfit.SpecFit
    :param model_list: List of model names to create the model flux from. The
        models must be exist in the SpecModel objects inside the SpecFit object.
    :type model_list: list(string)
    :param dispersion: New dispersion to create the model flux for
    :type dispersion: np.array
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
                              dispersion_unit=specfit.spec.dispersion_unit,
                              fluxden_unit=specfit.spec.fluxden_unit)

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
    :type input_spec: sculptor.speconed.SpecOneD
    :param disp_range: 2 element list holding the lower and upper dispersion
        boundaries for the integration
    :type disp_range: [float, float]
    :return: Integrated flux
    :rtype: astropy.units.Quantity

    """

    if disp_range is not None:
        spec = input_spec.trim_dispersion(disp_range, inplace=False)
    else:
        spec = input_spec.copy()

    return np.trapz(spec.fluxden, x=spec.dispersion) * spec.dispersion_unit *\
           spec.fluxden_unit


def get_average_fluxden(input_spec, dispersion, width=10, redshift=0):
    """
    Calculate the average flux density of a spectrum in a window centered at
    the specified dispersion and with a given width.

    The central dispersion and width can be specified in the rest-frame and
    then are redshifted to the observed frame using the redsh keyword argument.

    Warning: this function currently only works for spectra in wavelength
    units. For spectra in frequency units the conversion to rest-frame will
    be incorrect.

    :param input_spec: Input spectrum
    :type input_spec: sculptor.speconed.SpecOneD
    :param dispersion: Central dispersion
    :type dispersion: float
    :param width: Width of the dispersion window
    :type width: float
    :param redshift: Redshift argument to redshift the dispersion window into
        the observed frame.
    :type redshift: float
    :return: Average flux density
    :rtype: astropy.units.Quantity

    """

    disp_range = [(dispersion - width / 2.)*(1.+redshift),
                  (dispersion + width / 2.)*(1.+redshift)]

    unit = input_spec.fluxden_unit

    return input_spec.average_fluxden(dispersion_range=disp_range) * unit
        

# ------------------------------------------------------------------------------
# Emission line model measurements
# ------------------------------------------------------------------------------

def get_peak_redshift(input_spec, rest_wave):
    """
    Calculate the redshift of the flux density peak in the spectrum by
    specifying the expected rest frame wavelength of the emission feature.

    :param input_spec: Input spectrum
    :type input_spec: sculptor.speconed.SpecOneD
    :param rest_wave: Rest-frame wavelength of the expected emission feature.
    :return: Redshift of the peak flux density
    :rtype: float
    """
    # TODO: Implement frequency unit support.

    return input_spec.peak_dispersion()/rest_wave - 1


def get_equivalent_width(cont_spec, line_spec, disp_range=None,
                         redshift=0):
    """
    Calculate the rest-frame equivalent width of a spectral feature.

    Warning: this function currently only works for spectra in wavelength
    units. For spectra in frequency units the conversion to rest-frame will
    be incorrect.

    :param cont_spec: Continuum spectrum
    :type cont_spec: sculptor.speconed.SpecOneD
    :param line_spec: Spectrum with the feature (e.g. emission line)
    :type line_spec: sculptor.speconed.SpecOneD
    :param disp_range: Dispersion range (2 element list of floats) over which
        the equivalent width will be calculated.
    :type disp_range: [float, float]
    :param redshift: Redshift of the astronomical source
    :return: Rest-frame equivalent width
    :rtype: astropy.units.Quantity
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

    return ew * cont_spec.dispersion_unit


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
    :type input_spec: sculptor.speconed.SpecOneD
    :param disp_range: Dispersion range to which the calculation is limited.
    :type disp_range: [float, float]
    :param resolution: Resolution in R = Lambda/Delta Lambda
    :type resolution: float
    :return: FWHM of the spectral feature
    :rtype: astropy.units.Quantity
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
            return fwhm[0] * units.km/units.s
        else:
            print('[INFO] FWHM is corrected for the provided '
                  'resolution of R={}'.format(resolution))
            resolution_km_s = c_km_s/resolution
            return np.sqrt(fwhm ** 2 - resolution_km_s ** 2)[0] * \
                   units.km/units.s

# TODO: Add non-parametric width measurements

# def get_centroid_wavelength():
#     pass
#
#
# def get_centroid_redshift():
#     pass

# ------------------------------------------------------------------------------
# Astrophysical spectral measurements
# ------------------------------------------------------------------------------


def calc_lwav_from_fwav(fluxden, redshift, cosmology):
    """Calculate the monochromatic luminosity from the monochromatic flux
    density.

    :param fluxden: Monochromatic flux density at a given wavelength.
    :type fluxden: astropy.units.Unit or astropy.units.Quantity or
          astropy.units.CompositeUnit or astropy.units.IrreducibleUnit
    :param redshift: Redshift of the source.
    :type redshift: float
    :param cosmology: Cosmology as an astropy Cosmology object
    :type cosmology: astropy.cosmology.Cosmology
    :return: Monochromatic luminosity in units of erg s^-1 Angstroem^-1
    :rtype: astropy.units.Quantity
    """

    # Calculate luminosity distance in Mpc
    lum_distance = cosmology.luminosity_distance(redshift)

    return (fluxden * \
           lum_distance**2 * 4 * np.pi * (1. + redshift)).decompose(
        bases=[units.erg, units.s, units.AA])


def calc_integrated_luminosity(input_spec, redshift,
                                    cosmology, disp_range=None):
    """Calculate the integrated model spectrum luminosity. 
    
    :param input_spec: Input spectrum
    :type input_spec: sculptor.speconed.SpecOneD
    :param redshift: Redshift of the source.
    :type redshift: float
    :param cosmology: Cosmology as an astropy Cosmology object
    :type cosmology: astropy.cosmology.Cosmology
    :param disp_range: 2 element list holding the lower and upper dispersion
        boundaries for the integration
    :type disp_range: [float, float]
    :return: Return the integrated luminosity for (dispersion range in) the
        input spectrum.
    :rtype: astropy.units.Quantity
    """

    # Cconvert to rest-frame flux and dispersion
    rest_dispersion = input_spec.dispersion / (1. + redshift)
    rest_fluxden = calc_lwav_from_fwav(
        input_spec.fluxden*input_spec.fluxden_unit,
        redshift, cosmology)
    rest_frame_spec = sod.SpecOneD(dispersion=rest_dispersion,
                                   dispersion_unit=input_spec.dispersion_unit,
                                   fluxden=rest_fluxden.value,
                                   fluxden_unit=rest_fluxden.unit)

    # Only integrate part of the model spectrum if disp_range is specified
    if disp_range is not None:
        rest_frame_spec.trim_dispersion(disp_range, inplace=True)

    # Integrate the model flux
    integrated_line_luminosity = np.trapz(rest_frame_spec.fluxden,
                                          x=rest_frame_spec.dispersion)

    return integrated_line_luminosity * \
           rest_frame_spec.fluxden_unit * \
           rest_frame_spec.dispersion_unit


def calc_apparent_mag_from_fluxden(fluxden, dispersion):
    """Calculate the apparent AB magnitude from the monochromatic flux
    density at a specified dispersion value.

    :param fluxden: Monochromatic flux density at a given wavelength.
    :type fluxden: astropy.units.Quantity
    :param dispersion: Dispersion value (usually in wavelength).
    :type dispersion: float
    :return: Returns apparent AB magnitude.
    :rtype: astropy.units.Quantity
    """

    f_nu = fluxden * dispersion**2 / const.c

    value = (f_nu/units.ABflux).decompose()

    return -2.5*np.log10(value) * units.mag


# TODO: In a future update that includes astrophysical unit support the
#  following functions should be updated.

# TODO: In a future update consider allowing to specify a custom k-correction
#  function to allow for arbitrary k-corrections.


def calc_absolute_mag_from_fluxden(fluxden, dispersion,
                                   cosmology, redshift, kcorrection=True,
                                   a_nu=0):
    """Calculate the absolute AB magnitude from the monochromatic flux density
    at a given dispersion value.

    :param fluxden: Monochromatic flux density at a given wavelength.
    :type fluxden: astropy.units.Quantity
    :param dispersion: Dispersion value (usually in wavelength).
    :type dispersion: float
    :param cosmology: Cosmology as an astropy Cosmology object.
    :type cosmology: astropy.cosmology.Cosmology
    :param redshift: Redshift of the source.
    :type redshift: float
    :param kcorrection: Boolean to indicate whether the magnitude should be
        k-corrected assuming a power law spectrum. This keyword argument
        defaults to 'True'.
    :type kcorrection: bool
    :param a_nu: Power law slope as a function of frequency for the
        k-correction. This defaults to '0', appropriate for monochromatic
        measurements.
    :type a_nu: float
    :return: Absolute AB magnitude (monochromatic)
    :rtype: astropy.units.Quantity
    """

    abmag = calc_apparent_mag_from_fluxden(fluxden, dispersion)

    return calc_absolute_mag_from_apparent_mag(abmag, cosmology, redshift,
                                        kcorrection, a_nu)


def calc_absolute_mag_from_monochromatic_luminosity(l_wav, wavelength):
    """Calculate the absolute monochromatic magnitude from the monochromatic
    luminosity per wavelegnth.

    :param l_wav: Monochromatic luminosity per wavelength
    :type l_wav: astropy.units.Quantity
    :param wavelength: Wavelength of the monochromatic luminosity
    :type wavelength: float
    :return: Absolute monochromatic magnitude
    :rtype: astropy.units.Quantity
    """
    l_nu = l_wav * (wavelength)**2 / const.c / 4 / \
                  np.pi

    value = (l_nu / units.ABflux / (10 * units.pc)**2).decompose()

    return -2.5 * np.log10(value) * units.mag


def calc_absolute_mag_from_apparent_mag(appmag, cosmology, redshift,
                                        kcorrection=True, a_nu=0):
    """Calculate the absolute magnitude from the apparent magnitude using a
    power law k-correction.

    :param appmag: Apparent AB magnitude
    :type appmag: float
    :param cosmology: Cosmology as an astropy Cosmology object.
    :type cosmology: astropy.cosmology.Cosmology
    :param redshift: Redshift of the source.
    :type redshift: float
    :param kcorrection: Boolean to indicate whether the magnitude should be
        k-corrected assuming a power law spectrum. This keyword argument
        defaults to 'True'.
    :type kcorrection: bool
    :param a_nu: Power law slope as a function of frequency for the
        k-correction. This defaults to '0', appropriate for monochromatic
        measurements.
    :type a_nu: float
    :return: Absolute AB magnitude (monochromatic)
    :rtype: astropy.units.Quantity
    """

    if kcorrection:
        kcorr = k_correction_pl(redshift, a_nu)
    else:
        kcorr = 0 * units.mag

    distmod = cosmology.distmod(redshift)

    return appmag - distmod - kcorr


def k_correction_pl(redshift, a_nu):
    """Calculate the k-correction for a power law spectrum with spectral
    index (per frequency) a_nu.

    :param redshift: Cosmological redshift of the source
    :type redshift: float
    :param a_nu: Power law index (per frequency)
    :type a_nu: float
    :return: K-correction
    :rtype: float
    """
    # Hogg 1999, eq. 27
    return -2.5 * (1. + a_nu) * np.log10(1.+redshift) * units.mag

