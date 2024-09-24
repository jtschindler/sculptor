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
from tqdm import tqdm
from sculptor import specfit as sf
from sculptor import speconed as sod
from astropy.table import QTable, hstack
from astropy import constants as const
from astropy import units
from astropy.cosmology import FlatLambdaCDM

from scipy.interpolate import UnivariateSpline
from scipy.integrate import cumtrapz
import matplotlib.pyplot as plt

from IPython import embed

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
                             emfeat_meas=None, disp_range=None,
                             cosmology=None, fwhm_method='spline'):
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
        fwhm = get_fwhm(model_spec, method=fwhm_method)
        result_dict.update({feature_name+'_FWHM': fwhm})

        if np.isnan(fwhm):
            print('[WARNING] FWHM could not be calculated for feature '
                  '{}'.format(feature_name))

    if 'flux' in emfeat_meas:
        # Calculate the integrated line flux
        flux = get_integrated_flux(model_spec, disp_range=disp_range)
        result_dict.update({feature_name+'_flux': flux})

    if 'lum' in emfeat_meas:
        # Calculate the integrated line luminosity
        lum = calc_integrated_luminosity(model_spec,
                                         redshift=redshift, cosmology=cosmology)
        result_dict.update({feature_name + '_lum': lum})

    if 'nonparam' in emfeat_meas:
        # Calculate basic non-parametric line measurements
        # Note that this might be computationally expensive!

        v50, v05, v10, v90, v95, v_res_at_line, freq_v50, wave_v50, z_v50 = \
            get_nonparametric_measurements(model_spec, rest_frame_wavelength,
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


def analyze_continuum(specfit, model_names, rest_frame_wavelengths,
                      cosmology, redshift=None,
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
                                fluxden_avg})

        if 'Lwav' in cont_meas:
            lwav = calc_lwav_from_fwav(fluxden_avg,
                                       redshift=redshift,
                                       cosmology=cosmology)
            result_dict.update({wave_name+'_Lwav': lwav})

        if 'appmag' in cont_meas:
            appmag = calc_apparent_mag_from_fluxden(fluxden_avg,
                                                    wave*(1.+redshift)*units.AA)
            result_dict.update({wave_name + '_appmag': appmag})

        # if 'absmag' in cont_meas:
        #     absmag = calc_absolute_mag_from_monochromatic_luminosity(
        #             lwav, wave*units.AA)
        #     result_dict.update({wave_name + '_absmag': absmag})


    return result_dict


# TODO: Finalize the analyze, mcmc, and resample analysis functions.

# emission_feature_listdict = [{'feature_name': None,
#                          'model_names' : None,
#                          'rest_frame_wavelength': None}]
# continuum_listdict = {'model_names': None,
#                       'rest_frame_wavelengths': None}

# def analyse_fit(specfit,
#                 continuum_dict,
#                 emission_feature_dictlist,
#                 # absorption_feature_dictlist,
#                 redshift,
#                 cosmology,
#                 emfeat_meas=None,
#                 cont_meas=None,
#                 dispersion=None,
#                 width=10
#                 ):
#     """
#     THIS FUNCTION IS CURRENTLY UNDER ACTIVE DEVELOPMENT.
#     """
#
#
#     result_dict = {}
#
#     # Analyze continuum properties
#     cont_result = analyze_continuum(specfit,
#                                     continuum_dict['model_names'],
#                                     continuum_dict['rest_frame_wavelengths'],
#                                     cosmology,
#                                     redshift=redshift,
#                                     dispersion=dispersion,
#                                     cont_meas=cont_meas,
#                                     width=width
#                                     )
#
#     result_dict.update(cont_result)
#
#     # Analyze emission features
#     for emission_feat_dict in emission_feature_dictlist:
#         if 'disp_range' in emission_feat_dict:
#             disp_range = emission_feat_dict['disp_range']
#         else:
#             disp_range = None
#         emfeat_result = analyze_emission_feature(specfit,
#                                                  emission_feat_dict[
#                                                      'feature_name'],
#                                                  emission_feat_dict[
#                                                      'model_names'],
#                                                  emission_feat_dict[
#                                                      'rest_frame_wavelength'],
#                                                  cont_model_names=
#                                                  continuum_dict['model_names'],
#                                                  redshift=redshift,
#                                                  dispersion=dispersion,
#                                                  emfeat_meas=emfeat_meas,
#                                                  disp_range = disp_range
#                                                  )
#         result_dict.update(emfeat_result)
#
#
#     # Analyze absorption features
#     # TODO
#
#     return result_dict


def analyze_mcmc_results(foldername, specfit, continuum_dict,
                         emission_feature_dictlist, redshift, cosmology,
                         emfeat_meas=None, cont_meas=None, dispersion=None,
                         width=10, concatenate=False):
    """Analyze MCMC model fit results of specified continuum/feature models.

    Results will be written to an enhanced csv file in the same folder,
    where the MCMC flat chain data resides.

    **Important:** Only model functions that are sampled together can be \
    analyzed together. This means that only model functions from ONE \
    SpecModel can also be analyzed together. \
    Additionally, only model functions for which all variable parameters \
    have sampled by the MCMC fit are analyzed.

    The following parameters should be specified in the *continuum_listdict*:

    * 'model_names' - list of model function prefixes for the full continuum \
        model
    * 'rest_frame_wavelengths' - list of rest-frame wavelengths (float) for \
        which fluxes, luminosities and magnitudes should be calculated

    The other arguments for the *SpecAnalysis.analyze_continuum* are provided \
        to the MCMC analysis function separately.

    The following parameters should be specified in the \
        *emission_feature_listdict*:

    * 'feature_name' - name of the emission feature, which will be used to \
        name the resulting measurements in the output file.
    * 'model_names' - list of model names to create the emission feature model \
        flux from.
    * 'rest_frame_wavelength' - rest-frame wavelength of the emission feature.

    Additionally, one can specify:

    * 'disp_range' - 2 element list holding the lower and upper dispersion \
        boundaries flux density integration.

    :param foldername: Path to the folder with the MCMC flat chain hdf5 files.
    :type foldername: string
    :param specfit: Sculptor model fit (SpecFit object) containing the
        information about the science spectrum, the SpecModels and parameters.
    :type specfit: sculptor.specfit.SpecFit
    :param continuum_dict: The *continuum_listdict* holds the arguments for
        the *SpecAnalysis.analyze_continuum* function that will be called by
        this procedure.
    :type continuum_dict: dictionary
    :param emission_feature_dictlist: The *emission_feature_listdict* hold the
        arguments for the *SpecAnalysis.analyze_emission_feature* functions that
        will be called by this procedure.
    :type emission_feature_dictlist: list of dictionary
    :param redshift: Source redshift
    :type redshift: float
    :param cosmology: Cosmology for calculation of absolute properties
    :type cosmology: astropy.cosmology.Cosmology
    :param emfeat_meas: This keyword argument allows to specify the list of
        emission feature measurements.
        Currently possible measurements are ['peak_fluxden', 'peak_redsh', 'EW',
        'FWHM', 'flux']. The value defaults to 'None' in which all measurements
        are calculated
    :type emfeat_meas: list(string)
    :param cont_meas: This keyword argument allows to specify the list of
        emission feature measurements.
        Currently possible measurements are ['peak_fluxden', 'peak_redsh', 'EW',
        'FWHM', 'flux']. The value defaults to 'None' in which all measurements
        are calculated
    :type cont_meas: list(string)
    :param dispersion: This keyword argument allows to input a dispersion
        axis (e.g., wavelengths) for which the model fluxes are calculated. The
        value defaults to 'None', in which case the dispersion from the SpecFit
        spectrum is being used.
    :type dispersion: np.array
    :param width: Window width in dispersion units to calculate the average
        flux density in.
    :type width: [float, float]
    :param concatenate: Boolean to indicate whether the MCMC flat chain and
        the analysis results should be concatenated before written to file.
        (False = Only writes analysis results to file; True = Writes analysis
        results and MCMC flat chain parameter values to file)
    :type concatenate: bool
    :return: None
    """

    # Automatically identify all MCMC chains available in the given folder.
    mcmc_filelist = glob.glob(os.path.join(foldername,  '*_mcmc_chain.hdf5'))

    print('[INFO] Starting MCMC analysis')

    # Flag - Has the continuum been analyzed?
    cont_analyzed = False

    # Iterate through the MCMC flat chain files and identify if any specified
    # continuum or features should analyzed
    for mcmc_file in mcmc_filelist:
        mcmc_df = pd.read_hdf(mcmc_file)
        mcmc_columns = list(mcmc_df.columns)
        print('[INFO] Working on output file {}'.format(mcmc_file))

        # Set the continuum model names from the input dictionary
        cont_models = continuum_dict['model_names']

        # Iterate through emission feature model lists
        for emfeat_dict in emission_feature_dictlist:

            # Set the emission feature model names
            emfeat_models = emfeat_dict['model_names']
            # Set a list of both continuum and emission feature model names
            cont_emfeat_models = emfeat_models + cont_models

            # Iterate through the SpecModel objects
            for idx, specmodel in enumerate(specfit.specmodels):
                # Retrieve model function prefixes available in this SpecModel
                prefix_list = [model.prefix for model in specmodel.model_list]

                # IF continuum and emission features requested for the
                # analysis are available in THIS SpecModel begin a joint
                # analysis of the continuum and emission feature.
                # ONLY MODELS THAT WERE SAMPLED TOGETHER ARE ANALYZED TOGETHER!
                if set(cont_emfeat_models).issubset(prefix_list):
                    params_list_keys = []
                    for params in specmodel.params_list:
                        params_list_keys.extend(list(params.keys()))
                    # ONLY IF all parameters have been sampled by the MCMC
                    # continue with the analysis.
                    if set(params_list_keys).issubset(mcmc_columns):
                        print('[INFO] Analyzing continuum and emission '
                              'feature {}'.format(
                            emfeat_dict['feature_name']))
                        _mcmc_analyze(foldername, specfit, idx, mcmc_df,
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
                    # IF only the emission features requested for the analysis
                    # are available in THIS SpecModel analyze the emission
                    # feature.
                    if set(emfeat_models).issubset(prefix_list):
                        params_list_keys = []
                        for params in specmodel.params_list:
                            params_list_keys.extend(list(params.keys()))
                        # ONLY IF all parameters have been sampled by the MCMC
                        # continue with the analysis.
                        if set(params_list_keys).issubset(mcmc_columns):
                            print('[INFO] Analyzing emission feature {}'.format(
                                emfeat_dict['feature_name']))
                            _mcmc_analyze(foldername, specfit, idx, mcmc_df,
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

                    # IF only the continuum models that are requested to be
                    # analyzed (and haven't been analyzed before) are available
                    # in THIS SpecModel analyze the continuum.
                    if set(cont_models).issubset(prefix_list) \
                            and not cont_analyzed:
                        params_list_keys = []
                        for params in specmodel.params_list:
                            params_list_keys.extend(list(params.keys()))
                        # ONLY IF all parameters have been sampled by the MCMC
                        # continue with the analysis.
                        if set(params_list_keys).issubset(mcmc_columns):
                            print('[INFO] Analyzing continuum')
                            _mcmc_analyze(foldername, specfit, idx, mcmc_df,
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


def _mcmc_analyze(foldername, specfit, specmodel_index, mcmc_df, continuum_dict,
                  emission_feature_dict, redshift, cosmology, mode,
                  emfeat_meas=None, cont_meas=None, dispersion=None,
                  width=10,
                  concatenate=False):
    """Analyze the MCMC SpecModel fit and save the posterior distributions of
    the analyzed parameters.

    :param foldername: Path to the folder where the MCMC analysis should be
        saved to.
    :type foldername: string
    :param specfit: Sculptor model fit (SpecFit object) containing the
        information about the science spectrum, the SpecModels and parameters.
    :type specfit: sculptor.specfit.SpecFit
    :param specmodel_index: Integer that indicates which SpecModel the model
        functions for the analysis belong to.
    :type specmodel_index: int
    :param mcmc_df: Dataframe holding the MCMC flat chain results for the
        model function parameters.
    :type mcmc_df: pandas.DataFrame
    :param continuum_dict:
    :param continuum_dict: The *continuum_listdict* holds the arguments for
        the *SpecAnalysis.analyze_continuum* function that will be called by
        this procedure.
    :type continuum_dict: dictionary
    :param emission_feature_dictlist: The *emission_feature_listdict* hold the
        arguments for the *SpecAnalysis.analyze_emission_feature* functions that
        will be called by this procedure.
    :type emission_feature_dictlist: dictionary
    :param redshift: Source redshift
    :type redshift: float
    :param cosmology: Cosmology for calculation of absolute properties
    :type cosmology: astropy.cosmology.Cosmology
    :param mode: A string indicating whether an emission feature by itself
        will be analyzed (mode='emfeat'), the continuum will be analyzed (
        mode='cont') or both will be analyzed (mode='both').
    :type mode: string
    :param emfeat_meas: This keyword argument allows to specify the list of
        emission feature measurements.
        Currently possible measurements are ['peak_fluxden', 'peak_redsh', 'EW',
        'FWHM', 'flux']. The value defaults to 'None' in which all measurements
        are calculated
    :type emfeat_meas: list(string)
    :param cont_meas: This keyword argument allows to specify the list of
        emission feature measurements.
        Currently possible measurements are ['peak_fluxden', 'peak_redsh', 'EW',
        'FWHM', 'flux']. The value defaults to 'None' in which all measurements
        are calculated
    :type cont_meas: list(string)
    :param dispersion: This keyword argument allows to input a dispersion
        axis (e.g., wavelengths) for which the model fluxes are calculated. The
        value defaults to 'None', in which case the dispersion from the SpecFit
        spectrum is being used.
    :type dispersion: np.array
    :param width: Window width in dispersion units to calculate the average
        flux density in.
    :type width: [float, float]
    :param concatenate: Boolean to indicate whether the MCMC flat chain and
        the analysis results should be concatenated before written to file.
        (False = Only writes analysis results to file; True = Writes analysis
        results and MCMC flat chain parameter values to file)
    :type concatenate: bool
    :return:
    """

    # Set up continuum model variables
    cont_model_names = continuum_dict['model_names']

    # Set up the table variable for the results
    result_table = None

    emfeat_result_dict = None
    cont_result_dict = None
    feature_name = None

    for i in tqdm(range(len(mcmc_df.index))):
        idx = mcmc_df.index[i]
        # Work on a copy of the SpecFit object, not on the original
        fit = specfit.copy()

        # Update the specific SpecModel for the emission feature analysis
        mcmc_series = mcmc_df.loc[idx, :]
        fit.specmodels[
            specmodel_index].update_param_values_from_input_series(mcmc_series)

        result_dict = {}
        emfeat_result_dict = None
        cont_result_dict = None

        # Analyze the emission feature model function(s)
        if mode == 'both' or mode == 'emfeat':
            feature_name = emission_feature_dict['feature_name']
            model_names = emission_feature_dict['model_names']
            rest_frame_wavelength = emission_feature_dict['rest_frame_wavelength']
            if 'disp_range' in emission_feature_dict:
                disp_range = emission_feature_dict['disp_range']
            else:
                disp_range = None
            if 'fwhm_method' in emission_feature_dict:
                fwhm_method = emission_feature_dict['fwhm_method']
            else:
                fwhm_method = 'spline'

            emfeat_result_dict = \
                analyze_emission_feature(specfit,
                                         feature_name,
                                         model_names,
                                         rest_frame_wavelength,
                                         cont_model_names=cont_model_names,
                                         redshift=redshift,
                                         dispersion=dispersion,
                                         emfeat_meas=emfeat_meas,
                                         disp_range=disp_range,
                                         cosmology=cosmology,
                                         fwhm_method=fwhm_method)
        # Analyze the continuum model
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
    else:
        result = result_table

    if cont_result_dict is None:
        result.write('{}/mcmc_analysis_{}.csv'.format(foldername, feature_name),
                           format='ascii.ecsv', overwrite=True, delimiter=',')
    elif emfeat_result_dict is None:
        result.write('{}/mcmc_analysis_cont.csv'.format(foldername),
                           format='ascii.ecsv', overwrite=True, delimiter=',')
    else:
        result.write('{}/mcmc_analysis_cont_{}.csv'.format(
            foldername, feature_name), format='ascii.ecsv', overwrite=True,
            delimiter=',')


def analyze_resampled_results(specfit, foldername, resampled_df_name,
                              continuum_dict,
                              emission_feature_dictlist, redshift, cosmology,
                              emfeat_meas=None, cont_meas=None, dispersion=None,
                              width=10, concatenate=False):
    """Analyze resampled model fit results for all specified continuum and
    feature models.

    Results will be written to an enhanced csv file in the same folder,
    where the resampled raw data resides.

    The following parameters should be specified in the *continuum_listdict*:

    * 'model_names' - list of model function prefixes for the full continuum \
        model
    * 'rest_frame_wavelengths' - list of rest-frame wavelengths (float) for \
        which fluxes, luminosities and magnitudes should be calculated

    The other arguments for the *SpecAnalysis.analyze_continuum* are provided \
        to the MCMC analysis function separately.

    The following parameters should be specified in the \
        *emission_feature_listdict*:

    * 'feature_name' - name of the emission feature, which will be used to \
        name the resulting measurements in the output file.
    * 'model_names' - list of model names to create the emission feature model \
        flux from.
    * 'rest_frame_wavelength' - rest-frame wavelength of the emission feature.

    Additionally, one can specify:

    * 'disp_range' - 2 element list holding the lower and upper dispersion \
        boundaries flux density integration.

    :param specfit: Sculptor model fit (SpecFit object) containing the
        information about the science spectrum, the SpecModels and parameters.
    :type specfit: sculptor.specfit.SpecFit
    :param foldername: Path to the folder with the resampled raw hdf5 file.
    :type foldername: string
    :param resampled_df_name: Filename of the resampled raw DataFrame saved
        in hdf5 format.
    :type resampled_df_name: str
    :param continuum_dict: The *continuum_listdict* holds the arguments for
        the *SpecAnalysis.analyze_continuum* function that will be called by
        this procedure.
    :type continuum_dict: dictionary
    :param emission_feature_dictlist: The *emission_feature_listdict* hold the
        arguments for the *SpecAnalysis.analyze_emission_feature* functions that
        will be called by this procedure.
    :type emission_feature_dictlist: list of dictionary
    :param redshift: Source redshift
    :type redshift: float
    :param cosmology: Cosmology for calculation of absolute properties
    :type cosmology: astropy.cosmology.Cosmology
    :param emfeat_meas: This keyword argument allows to specify the list of
        emission feature measurements.
        Currently possible measurements are ['peak_fluxden', 'peak_redsh', 'EW',
        'FWHM', 'flux']. The value defaults to 'None' in which all measurements
        are calculated
    :type emfeat_meas: list(string)
    :param cont_meas: This keyword argument allows to specify the list of
        emission feature measurements.
        Currently possible measurements are ['peak_fluxden', 'peak_redsh', 'EW',
        'FWHM', 'flux']. The value defaults to 'None' in which all measurements
        are calculated
    :type cont_meas: list(string)
    :param dispersion: This keyword argument allows to input a dispersion
        axis (e.g., wavelengths) for which the model fluxes are calculated. The
        value defaults to 'None', in which case the dispersion from the SpecFit
        spectrum is being used.
    :type dispersion: np.array
    :param width: Window width in dispersion units to calculate the average
        flux density in.
    :type width: [float, float]
    :param concatenate: Boolean to indicate whether the MCMC flat chain and
        the analysis results should be concatenated before written to file.
        (False = Only writes analysis results to file; True = Writes analysis
        results and MCMC flat chain parameter values to file)
    :type concatenate: bool
    :return: None
    """

    # Test if all necessary columns are inlcuded in the resampled file
    resampled_df = pd.read_hdf(foldername+'/'+resampled_df_name, 'data')
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
        # If all parameters are found, continue with analysis
        _resampled_analyze(specfit, resampled_df, foldername, continuum_dict,
                           emission_feature_dictlist, redshift, cosmology,
                           emfeat_meas=emfeat_meas, cont_meas=cont_meas,
                           dispersion=dispersion,
                           width=width, concatenate=concatenate)
    else:
        print('[ERROR] Resampled results do NOT contain necessary column '
              'information.')
        print('[ERROR] Please double check if the correct file was supplied.')


def _resampled_analyze(specfit, resampled_df, foldername, continuum_dict,
                              emission_feature_dictlist, redshift, cosmology,
                              emfeat_meas=None, cont_meas=None, dispersion=None,
                              width=10, concatenate=False):
    """Analyze the resampled model fit and save the posterior distributions of
    the analyzed parameters.

    :param specfit: Sculptor model fit (SpecFit object) containing the
        information about the science spectrum, the SpecModels and parameters.
    :type specfit: sculptor.specfit.SpecFit
    :param resampled_df: The resampled raw DataFrame
    :type resampled_df: pd.DataFrame
    :param foldername: Path to the folder with the resampled raw hdf5 file.
        Used for saving the analyzed results.
    :type foldername: string
    :param continuum_dict: The *continuum_listdict* holds the arguments for
        the *SpecAnalysis.analyze_continuum* function that will be called by
        this procedure.
    :type continuum_dict: dictionary
    :param emission_feature_dictlist: The *emission_feature_listdict* hold the
        arguments for the *SpecAnalysis.analyze_emission_feature* functions that
        will be called by this procedure.
    :type emission_feature_dictlist: list of dictionary
    :param redshift: Source redshift
    :type redshift: float
   :param cosmology: Cosmology for calculation of absolute properties
    :type cosmology: astropy.cosmology.Cosmology
    :param emfeat_meas: This keyword argument allows to specify the list of
        emission feature measurements.
        Currently possible measurements are ['peak_fluxden', 'peak_redsh', 'EW',
        'FWHM', 'flux']. The value defaults to 'None' in which all measurements
        are calculated
    :type emfeat_meas: list(string)
    :param cont_meas: This keyword argument allows to specify the list of
        emission feature measurements.
        Currently possible measurements are ['peak_fluxden', 'peak_redsh', 'EW',
        'FWHM', 'flux']. The value defaults to 'None' in which all measurements
        are calculated
    :type cont_meas: list(string)
    :param dispersion: This keyword argument allows to input a dispersion
        axis (e.g., wavelengths) for which the model fluxes are calculated. The
        value defaults to 'None', in which case the dispersion from the SpecFit
        spectrum is being used.
    :type dispersion: np.array
    :param width: Window width in dispersion units to calculate the average
        flux density in.
    :type width: [float, float]
    :param concatenate: Boolean to indicate whether the MCMC flat chain and
        the analysis results should be concatenated before written to file.
        (False = Only writes analysis results to file; True = Writes analysis
        results and MCMC flat chain parameter values to file)
    :type concatenate: bool
    :return: None
    """
    
    # Set up the table variable for the results
    result_table = None

    emfeat_result_dict = None
    cont_result_dict = None
    feature_name = None

    for i in tqdm(range(len(resampled_df.index))):
        idx = resampled_df.index[i]

        # Work on a copy of the original SpecFit object
        fit = specfit.copy()

        # Update all SpecModels for the analysis
        resampled_series = resampled_df.loc[idx, :]
        for specmodel in fit.specmodels:
            specmodel.update_param_values_from_input_series(resampled_series)

        result_dict = {}
        emfeat_result_dict = None
        cont_result_dict = None

        # Continuum analysis
        cont_result_dict = \
            analyze_continuum(specfit,
                              continuum_dict['model_names'],
                              continuum_dict['rest_frame_wavelengths'],
                              cosmology,
                              redshift=redshift,
                              dispersion=dispersion,
                              cont_meas=cont_meas,
                              width=width)

        # Emission line analysis
        emfeat_result_dict = {}
        for emission_feature_dict in emission_feature_dictlist:
            cont_model_names = continuum_dict['model_names']
            feature_name = emission_feature_dict['feature_name']
            model_names = emission_feature_dict['model_names']
            rest_frame_wavelength = emission_feature_dict[
                'rest_frame_wavelength']
            if 'disp_range' in emission_feature_dict:
                disp_range = emission_feature_dict['disp_range']
            else:
                disp_range = None
            if 'fwhm_method' in emission_feature_dict:
                fwhm_method = emission_feature_dict['fwhm_method']
            else:
                fwhm_method = 'spline'

            single_emfeat_result_dict = \
                analyze_emission_feature(specfit,
                                         feature_name,
                                         model_names,
                                         rest_frame_wavelength,
                                         cont_model_names=cont_model_names,
                                         redshift=redshift,
                                         dispersion=dispersion,
                                         emfeat_meas=emfeat_meas,
                                         disp_range=disp_range,
                                         cosmology=cosmology,
                                         fwhm_method=fwhm_method)

            emfeat_result_dict.update(single_emfeat_result_dict)


        if cont_result_dict is not None:
            result_dict.update(cont_result_dict)
        if emfeat_result_dict is not None:
            result_dict.update(emfeat_result_dict)

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
        resampled_table = QTable.from_pandas(resampled_df)
        result = hstack([resampled_table, result_table])
    else:
        result = result_table

    if cont_result_dict is None and len(emission_feature_dictlist) == 1:
        result.write('{}/resampled_analysis_{}.csv'.format(foldername,
                                                       feature_name),
                           format='ascii.ecsv', overwrite=True, delimiter=',')
    elif cont_result_dict is None and len(emission_feature_dictlist) > 1:
        result.write('{}/resampled_analysis_lines.csv'.format(foldername),
                           format='ascii.ecsv', overwrite=True, delimiter=',')
    elif emfeat_result_dict is None:
        result.write('{}/resampled_analysis_cont.csv'.format(foldername),
                           format='ascii.ecsv', overwrite=True, delimiter=',')
    elif cont_result_dict is not None and len(emission_feature_dictlist) == 1:
        result.write('{}/resampled_analysis_cont_{}.csv'.format(
            foldername, feature_name), format='ascii.ecsv', overwrite=True,
            delimiter=',')
    else:
        result.write('{}/resampled_analysis_cont_lines.csv'.format(
            foldername), format='ascii.ecsv', overwrite=True,
            delimiter=',')

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

                # Check params for global parameters and update them
                for param in specmodel.params_list[idx]:
                    expr = specmodel.params_list[idx][param].expr
                    if expr is not None:
                        expr_val = specmodel.params_list[idx][expr].value
                        if expr in specmodel.params_list[idx]:
                            specmodel.params_list[idx][param].expr = None
                            specmodel.params_list[idx][param].value = expr_val
                        else:
                            raise ValueError('Global parameter {} not found in SpecModel'.format(expr))

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
        l_idx = np.argmin(np.abs(rest_dispersion - disp_range[0]))
        u_idx = np.argmin(np.abs(rest_dispersion - disp_range[1]))

        ew = np.trapz((rest_line_flux[l_idx:u_idx])/rest_cont_flux[l_idx:u_idx],
                      rest_dispersion[l_idx:u_idx])
    else:
        ew = np.trapz((rest_line_flux) / rest_cont_flux,
                      rest_dispersion)

    return ew * cont_spec.dispersion_unit


def get_fwhm(input_spec, disp_range=None, resolution=None, method='spline'):
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
    :param method: Method to use in retrieving the FWHM. There are two
        methods available. The default method 'spline' uses a spline to interpolate
        the original spectrum and find the zero points using a root finding
        algorithm on the spline. The second method 'sign'  finds sign changes in
        the half peak flux subtracted spectrum.
    :type method: string
    :return: FWHM of the spectral feature
    :rtype: astropy.units.Quantity
    """

    if disp_range is not None:
        spec = input_spec.trim_dispersion(disp_range, inplace=False)
    else:
        spec = input_spec.copy()

    fluxden = spec.fluxden
    dispersion = spec.dispersion

    if method == 'spline':
        spline = UnivariateSpline(dispersion,
                                  fluxden - np.max(fluxden) / 2.,
                                  s=0)
        roots = spline.roots()
    elif method == 'sign':
        roots_idx = np.where(np.diff(np.sign(fluxden-np.max(fluxden)/2.)))[0]
        roots = dispersion[roots_idx]
    else:
        raise ValueError('[ERROR] Value for "method" keyword argument not '
                        'recgonized. Possible values are "sign" or "spline".')

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


def get_nonparametric_measurements(input_spec, line_rest_wavel, redshift,
                                   disp_range=None):
    """
    Measure the velocities at different ratios of the total emission line flux.

    These velocity measurements are referenced in the literature by (e.g.)
    Whittle+1985, Liu+2013, Zakamska & Greene 2014.

    This function calculates the cumulative integral of the emission line
    flux and then determines the closest dispersion values in velocity space to
    the 5%, 10%, 50%, 90% and 95% total flux ratios.

    :param input_spec: Input spectrum
    :type input_spec: sculptor.speconed.SpecOneD
    :param line_rest_wavel: rest-frame wavelength fo the line in Angstroem
    :type line_rest_wavel: float
    :param redshift: Redshift of the source
    :type redshift: float
    :param disp_range: Observed-frame dispersion range to which the
        calculation is limited.
    :type disp_range: [float, float]
    :return: median velocity, 5% velocity, 10% velocity, 90% velocity,
        95% velocity, velocity resolution at median velocity, frequency of
        median velocity, wavelength of median velocity, redshift of median
        velocity
    :rtype: astropy.units.Quantity, astropy.units.Quantity,
        astropy.units.Quantity, astropy.units.Quantity,
        astropy.units.Quantity, astropy.units.Quantity,
        astropy.units.Quantity, astropy.units.Quantity, astropy.units.Quantity
    """

    # Convert input spectrum to frequency
    input_spec.to_fluxden_per_unit_frequency_cgs()

    # Total flux
    flux_total = get_integrated_flux(input_spec, disp_range=disp_range)

    # Trim spectrum to disp_range if not None
    if disp_range is not None:
        input_spec.trim_dispersion(disp_range, inplace=True)

    # Calculate velocity dispersion axis
    line_wavel = (1 + redshift) * line_rest_wavel
    line_cen_freq = (const.c / (line_wavel * units.AA))
    freq_to_vel = units.doppler_optical(line_cen_freq)
    velocity_disp = (input_spec.dispersion * input_spec.dispersion_unit).to(
        units.km / units.s, equivalencies=freq_to_vel)

    # Calculate cumulative flux
    flux_fraction = cumtrapz(input_spec.fluxden, input_spec.dispersion,
                             initial=0)
    flux_fraction *= input_spec.dispersion_unit * input_spec.fluxden_unit
    flux_fraction /= flux_total

    # Determine velocities for flux fractions
    idx_90 = np.argmin(abs(flux_fraction - 0.1))
    idx_10 = np.argmin(abs(flux_fraction - 0.9))
    idx_95 = np.argmin(abs(flux_fraction - 0.05))
    idx_05 = np.argmin(abs(flux_fraction - 0.95))
    idx_50 = np.argmin(abs(flux_fraction - 0.5))

    v90 = velocity_disp[idx_90]
    v10 = velocity_disp[idx_10]
    v95 = velocity_disp[idx_95]
    v05 = velocity_disp[idx_05]
    v50 = velocity_disp[idx_50]
    v_res_at_line = np.abs(velocity_disp[idx_50-1]-velocity_disp[idx_50+1])/2.

    # Calculate the centroid frequency
    freq_v50 = input_spec.dispersion[idx_50] * input_spec.dispersion_unit
    # Calculate the centroid wavelength
    wave_v50 = (freq_v50).to(units.AA, equivalencies= units.spectral())
    # Calculate the centroid redshift
    z_v50 = wave_v50.value/line_rest_wavel - 1.

    return v50, v05, v10, v90, v95, v_res_at_line, freq_v50, wave_v50, z_v50



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


def calc_absolute_mag_from_fluxden_alt(fluxden, dispersion,
                                   cosmology, redshift, kcorrection=True,
                                   a_nu=0):
    """
    Calculate the absolute AB magnitude from the monochromatic flux density at a
    given wavelength value.
    
    :param fluxden: 
    :param dispersion: 
    :param cosmology: 
    :param redshift: 
    :param kcorrection: 
    :param a_nu: 
    :return: 
    """
    
    f_nu = fluxden * dispersion**2 / const.c
    
    dist_lum = cosmology.luminosity_distance(redshift)
    
    l_nu = f_nu / (1+redshift) * dist_lum**2 
    
    absmag = -2.5 * np.log10(l_nu / units.ABflux / (10 * units.pc)**2).decompose()
    
    return absmag
    


def calc_absolute_mag_from_monochromatic_luminosity(l_wav, wavelength, redshift):
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

