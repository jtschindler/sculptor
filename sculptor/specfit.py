#!/usr/bin/env python

"""
This module introduces the SpecFit class and its functionality. The SpecFit
class is designed to facilitate complex model fits to astronomical spectra.

It is initialized with the supplied astronomical spectrum and can hold
multiple SpecModel objects, which themselves hold the fit models and their
parameters.
"""

import os
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.transforms as mtransforms

from lmfit import Parameters

from sculptor import speconed as sod

from sculptor.specmodel import SpecModel



class SpecFit:
    """ Base class for fitting of astronomical spectroscopic data.

    The SpecFit class takes a SpecOneD object of an astronomical spectrum and
    allows complex models to be fit to it using the LMFIT module.

    SpecModel objects will be added to the SpecFit class to hold information
    on the different models and parameters. Each SpecModel object will be
    consecutively fit to the astronomical spectrum.

    Attributes:
        spec (SpecOneD): Astronomical spectrum as a SpecOneD object
        xlim (list of float): Wavelength limits for plotting
        ylim (list of float): Flux density limits for plotting
        redshift (float): Cosmological redshift of the astronomical object
        fitting_method (str): Fitting method (default: 'Levenberg-Marquardt')
        colors (numpy.ndarray of floats): Float values to set colors for
            plotting
        super_params (lmfit.parameters): Parameter list of "Super
            Parameters", which are global for the specfit class and are added
            as "Global Parameters" to all SpecModels
        specmodels (list of SpecModel): List of SpecModel objects added to the
            SpecFit class.

        """

    def __init__(self, spectrum=None, redshift=0):
        """Init method for the SpecFit class

        :param  (SpecOneD) spectrum: Astronomical spectrum in form of a \
        SpecOneD object
        :param (float) redshift: Cosmological redshift of the astronomical \
        object
        :return : None
        """

        if isinstance(spectrum, sod.SpecOneD):
            self.spec = spectrum.copy()
            self.xlim = [min(self.spec.dispersion),
                         max(self.spec.dispersion)]
            self.ylim = [0, max(self.spec.fluxden)]
        else:
            self.spec = None
            self.xlim = [0, 1]
            self.ylim = [0, 1]

        self.redshift = redshift
        self.fitting_method = 'Levenberg-Marquardt'
        self.colors = np.array([0.5])
        self.super_params = Parameters()
        self.specmodels = []

        self.emcee_kws = dict(steps=1000,
                              burn=300,
                              thin=20,
                              nwalkers=50,
                              workers=1,
                              is_weighted=True,
                              progress=False,
                              seed=1234)


    def copy(self):
        """
        Copy the SpecFit object

        :return: SpecFit
        """

        specfit = SpecFit(spectrum=self.spec,
                          redshift=self.redshift)


        specfit.fitting_method = self.fitting_method
        specfit.colors = self.colors
        specfit.super_params = self.super_params.copy()

        specfit.specmodels = self.specmodels.copy()

        return specfit

    def add_specmodel(self):
        """
        Add a SpecModel to the SpecFit class

        :return: None
        """

        if hasattr(self, 'spec'):

            if len(self.specmodels) == 0:
                specmodel = SpecModel(self,
                                      spectrum=self.spec,
                                      redshift=self.redshift)

            elif hasattr(self.specmodels[-1], 'spec'):
                spec = self.specmodels[-1].spec.copy()

                # Propagate SpecModel results to new SpecModel
                if hasattr(self.specmodels[-1], 'model_fluxden'):
                    if self.specmodels[-1].propagate == "subtract":
                        spec.fluxden -= self.specmodels[-1].model_fluxden
                    elif self.specmodels[-1].propagate == "divide":
                        spec.fluxden /= self.specmodels[-1].model_fluxden
                        spec.fluxden_unit /= self.specmodels[
                            -1].spec.fluxden_unit

                specmodel = SpecModel(self,
                                      spectrum=spec,
                                      redshift=self.redshift)

            else:
                specmodel = SpecModel(self, redshift=self.redshift)


        else:
            specmodel = SpecModel(self, redshift=self.redshift)

        # Add current super parameters to global parameters
        specmodel.global_params.update(self.super_params)

        self.specmodels.append(specmodel)

        self._update_colors()

    def delete_specmodel(self, index=None):
        """
        Delete the latest SpecModel or the one indicated by the index keyword
        argument from the SpecFit class.

        :param int index: Index of the SpecModel to delete in specmodels
        :return: None
        """

        # If index is None last SpecModel will be removed otherwise the
        # SpecModel specified by the index will be removed.
        if index is None:
            specmodel_to_delete = self.specmodels.pop()
        else:
            specmodel_to_delete = self.specmodels.pop(index)

        # Delete the SpecModel object
        del specmodel_to_delete
        if len(self.specmodels) > 0:
            self._update_colors()

    def update_specmodels(self):
        """
        Update SpecFit parameters in all SpecModels

        """

        for specmodel in self.specmodels:
            specmodel.redshift = self.redshift


    def update_specmodel_spectra(self):
        """
        Update all SpecModel spectra

        This function updates the SpecModel spectra consecutively. Model fits
        from each SpecModel will be automatically subtracted/divided.

        Note: Not only the dispersion and the fluxden, but also the mask will be
        updated.

        :return:  None
        """

        if self.spec is not None and len(self.specmodels) > 0:

            # Copy SpecFit Spectrum to first specmodel
            self.specmodels[0].spec = self.spec.copy()

            # If there is more than 1 SpecModel
            if len(self.specmodels) > 1:
                # Propagate Spectrum
                for idx in range(len(self.specmodels[1:])):
                    previous_spec = self.specmodels[idx].spec.copy()
                    previous_model_fluxden = self.specmodels[idx].model_fluxden
                    previous_fluxden_unit = self.specmodels[
                        idx].spec.fluxden_unit

                    # Propagate SpecModel fit results to next SpecModel
                    if self.specmodels[idx].propagate == "subtract":

                        self.specmodels[idx + 1].spec = previous_spec.copy()
                        self.specmodels[
                            idx + 1].spec.fluxden -= previous_model_fluxden

                    elif self.specmodels[idx].propagate == "divide":

                        self.specmodels[idx + 1].spec = previous_spec.copy()
                        self.specmodels[
                            idx + 1].spec.fluxden /= previous_model_fluxden
                        self.specmodels[
                            idx + 1].spec.fluxden_unit /= previous_fluxden_unit
                    else:
                        raise ValueError(
                            '[ERROR] Specmodels propagations allows '
                            'subtraction ("subtract") or division ('
                            '"divide"). No other modes are supported. '
                            'Please check the SpecModel.propagate '
                            'attribute.')


    def add_super_param(self, param_name, value=None, vary=True,
                         min=-np.inf, max=np.inf, expr=None):
        """ Adding a "Super Parameter" to the SpecFit object.

        :param str param_name: Name of the super parameter
        :param float,optional value: Initial value of the super parameter
        :param bool,optional vary: Boolean to indicate whether the super \
        parameter should be varied during the fit
        :param float,optional min: Minimum value for the super parameter
        :param float,optional max: Maximum value for the super parameter
        :param str,optional expr: Optional expression for the super \
        parameter

        :return: None
        """

        self.super_params.add(param_name, value=value, vary=vary, min=min,
                              max=max, expr=expr)

        if len(self.specmodels) > 0:
            for specmodel in self.specmodels:
                specmodel.add_global_param(param_name, value=value,
                                           vary=vary,
                                           min=min,
                                           max=max,
                                           expr=expr)

    def remove_super_param(self, param_name):
        """ Remove "Super Parameter" from SpecFit object.

        :param str param_name: Parameter name of the super parameter to \
        remove.
        :return: None
        """

        if param_name in self.super_params:
            self.super_params.pop(param_name)

        if len(self.specmodels) > 0:
            for specmodel in self.specmodels:
                specmodel.remove_global_param(param_name)

    def fit(self, save_results=False, foldername='.'):
        """ Fit all SpecModels consecutively 
        
        :param bool save_results: Boolean to indicate whether fit results \
        will be saved.
        :param str,optional foldername: If "save_results==True" the fit \
        results will be saved to the folder specified in foldername. This \
        variable defaults to the current folder. If set to "None" fit \
        results will not be saved.
        :return: None 
        """

        # Update the initial SpecModel spectrum
        self.specmodels[0].spec = self.spec.copy()

        # For each SpecModel in specmodels
        for idx, specmodel in enumerate(self.specmodels):
            # Update the main spectrum mask
            specmodel.spec.mask = self.spec.mask
            # Fit the model
            specmodel.fit()

            if save_results and foldername is not None:
                specmodel.save_fit_report(foldername, specmodel_id=str(idx) + '_FitAll')

            specmodel.update_params_from_fit_result()

            # Propagate SpecModel fit results to next SpecModel
            if specmodel.propagate == "subtract":

                if idx+1 < len(self.specmodels):
                    self.specmodels[idx+1].spec.fluxden = \
                        specmodel.spec.fluxden - specmodel.model_fluxden

            elif specmodel.propagate == "divide":

                if idx + 1 < len(self.specmodels):
                    self.specmodels[idx+1].spec.fluxden = \
                     specmodel.spec.fluxden / specmodel.model_fluxden
                    self.specmodels[idx+1].spec.fluxden_unit = \
                    specmodel.spec.fluxden_unit / specmodel.spec.fluxden_unit

            else:
                raise ValueError('[ERROR] Specmodels propagations allows '
                                 'subtraction ("subtract") or division ('
                                 '"divide"). No other modes are supported. '
                                 'Please check the SpecModel.propagate '
                                 'attribute.')

    def load(self, foldername):
        """ Load a full spectral fit (SpecFit) from a folder

        This function overwrites all SpecModels, SpecFit parameters, and the
        astronomical spectrum.
        
        :param str foldername: Folder from which the SpecFit class will be \
        loaded.
        :return: None 
        """

        # Load main SpecFit data

        # Load the spectrum as a SpecOneD object
        self.spec = sod.SpecOneD()
        self.spec.read_from_hdf(foldername+'/spectrum.hdf5')

        #  Load SpecFit meta data
        meta = pd.read_hdf(foldername + '/fit.hdf5', key='specfit_meta')

        n_specmodels = int(meta.loc['n_specmodels', 0])

        # Compatibility if statements for older versions without these
        # meta parameters.
        if 'redshift' in meta.index:
            self.redshift = float(meta.loc['redshift', 0])

        # For backward_compatibility of dispersion/flux density limits loading
        if 'xlim' in meta.index:
            self.xlim = meta.loc['xlim', 0]
            print('[WARNING] You are loading data from a '
                                     'saved fit created with beta '
                  'version 0.2b0. This '
                                     'format will be deprecated with '
                                     'release 1.0.0')
        else:
            self.xlim = [float(meta.loc['xlim_0',0]),
                         float(meta.loc['xlim_1',0])]
        if 'ylim' in meta.index:
            self.ylim = meta.loc['ylim', 0]
            print('[WARNING] You are loading data from a '
                                     'saved fit created with beta '
                  'version 0.2b0. This '
                                     'format will be deprecated with '
                                     'release 1.0.0')
        else:
            self.ylim = [float(meta.loc['ylim_0', 0]),
                         float(meta.loc['ylim_1', 0])]

        if 'fitting_method' in meta.index:
            self.fitting_method = meta.loc['fitting_method', 0]

        # Load global parameters
        if os.path.isfile(foldername + '/super_params.json'):
            sp_file = open(foldername + '/super_params.json', 'r')
            self.super_params.load(sp_file)
            sp_file.close()

            # TODO update functions for SuperParams
            # self.update_model_params_for_global_params()

        # Initialize and load SpecModels
        self.specmodels = []
        for idx in range(n_specmodels):
            specmodel = SpecModel(self)
            specmodel.load(foldername, specmodel_id=idx)
            self.specmodels.append(specmodel)

        self._update_colors()

    def save(self, foldername):
        """ Save the spectral fit (SpecFit) to a folder.
        
        :param str foldername: Folder to which the SpecFit class will be \
            saved.
        :return: None 
        """

        # Check if folder exists otherwise create it
        if not os.path.exists(foldername):
            os.makedirs(foldername)

        # Remove existing spectrum file in the folder
        if os.path.isfile(foldername+'/spectrum.hdf5'):
            os.remove(foldername+'/spectrum.hdf5')

        # Save the spectrum as a hdf5 file
        self.spec.save_to_hdf(foldername+'/spectrum.hdf5')

        # Create an hdf5 file with information on the SpecFit meta
        data = []
        columns = []

        data.append([len(self.specmodels)])
        columns.append('n_specmodels')
        data.append([self.xlim[0]])
        columns.append('xlim_0')
        data.append([self.ylim[0]])
        columns.append('ylim_0')
        data.append([self.xlim[1]])
        columns.append('xlim_1')
        data.append([self.ylim[1]])
        columns.append('ylim_1')
        data.append([self.redshift])
        columns.append('redshift')
        data.append([self.fitting_method])
        columns.append('fitting_method')

        # Remove existing fit file in the folder
        if os.path.isfile(foldername + '/fit.hdf5'):
            os.remove(foldername + '/fit.hdf5')

        df = pd.DataFrame(np.array(data), index=columns)
        df.to_hdf(foldername + '/fit.hdf5', key='specfit_meta')

        # Save super parameters
        if len(self.super_params) > 0:
            sp_file = open(foldername + '/super_params.json', 'w')
            self.super_params.dump(sp_file)
            sp_file.close()

        # Save SpecModels
        for idx, specmodel in enumerate(self.specmodels):
            specmodel.save(foldername, idx)

    def import_spectrum(self, filename, filetype='IRAF'):
        """ Import an astronomical spectrum into SpecFit class

        Currently the allowed 'filetype' options are:
        'IRAF', 'PypeIt', 'SpecOneD', 'SDSS'

        Note that the SpecFit class can be initialized a SpecOneD spectrum
        object, that can be constructed manually from the spectral format of
        choice.
        
        :param str filename: Full file name of the astronomical spectrum
        :param str filetype: String specifying the type of the spectrum to \
            select the appropriate read method.

        :return: None
        """

        spec = sod.SpecOneD()

        if filetype == 'IRAF':
            spec.read_from_fits(filename)
        elif filetype == 'PypeIt':
            spec.read_pypeit_fits(filename)
        elif filetype == 'SpecOneD':
            spec.read_from_hdf(filename)
        elif filetype == 'SDSS':
            spec.read_sdss_fits(filename)

        self.spec = spec.copy()

        # Update all SpecModels
        for specmodel in self.specmodels:
            specmodel.__init__(self, spectrum=self.spec, redshift=self.redshift)
            specmodel.reset_plot_limits()
            specmodel.reset_fit_mask()

    def normalize_spectrum_by_error(self):
        """Normalize the flux density, flux density error and object model
        arrays of the spectrum by the median value of the flux density error
        array.

        The flux density unit will be scaled accordingly. Hence,
        this normalization does not affect the physical values of the flux
        density and only serves to normalize the values in the flux density
        array.

        This enables more efficient calculations on the flux density array by
        avoiding small numerical values.

        :return:
        """

        self.spec.normalize_fluxden_by_error(inplace=True)
        self.update_specmodel_spectra()

    def normalize_spectrum_to_factor(self, factor):
        """Normalize the flux density, flux density error and object model
        arrays of the spectrum to the specified unit factor.

        The flux density unit will be scaled accordingly. Hence,
        this normalization does not affect the physical values of the flux
        density and only serves to normalize the values in the flux density
        array.

        This enables more efficient calculations on the flux density array by
        avoiding small numerical values.

        :param factor:
        :return:
        """

        self.spec.normalize_fluxden_to_factor(factor, inplace=True)
        self.update_specmodel_spectra()

    def normalize_spectrum_by_factor(self, factor):
        """Normalize the flux density, flux density error and object model
        arrays of the spectrum by the specified numerical factor.

        The flux density unit will be scaled accordingly. Hence,
        this normalization does not affect the physical values of the flux
        density and only serves to normalize the values in the flux density
        array.

        This enables more efficient calculations on the flux density array by
        avoiding small numerical values.

        :param factor:
        :return:
        """

        self.spec.normalize_fluxden_by_factor(factor, inplace=True)
        self.update_specmodel_spectra()

    def resample(self, n_samples=100, save_result_plots=True,
                 foldername='.', seed=1234):
        """
        Resample and fit the spectrum.

        Resample the spectral flux on a pixel by pixel basis by assuming a
        Gaussian distribution of flux values around the measured flux value
        with a sigma equal to the flux uncertainty.

        Fit all SpecModels to the resampled spectrum and record the best-fit
        values of all fit parameters. The fits are initialized with the
        current parameter values from all SpecModels.

        All n_samples results for each parameter are saved in a hdf5 file
        with the filename 'resampled_fitting_results_[n_samples]_raw.hdf5'.
        Median, lower (15.9 percentile) and upper (84.1 percentile) values are
        calculated from each parameter distribution and saved in a csv/hdf5
        file with the name 'resampled_fitting_results_[n_samples].hdf5/.csv'.

        If fit result plots are enabled ('save_result_plots=True') then the
        best-fie value distributions for each parameters, including their
        median, lower and upper values are saved to
        '[foldername]/[parameter name]_results.pdf'.

        :param int n_samples: Number of samples to draw
        :param bool save_result_plots: Boolean indicating whether \
        histograms for all parameters should be saved in the specified folder.
        :param str foldername: Path to the folder where the result plots \
        will be saved. This defaults to '.'.
        :param int seed: Random seed for initializing the numpy random \
        number generator

        :return: None
        """

        spec = self.spec.copy()

        if not hasattr(spec, 'fluxden_err'):
            raise ValueError("The spectrum does not have usable fluxden errors.")

        if len(self.specmodels) == 0:
            raise ValueError("No SpecModel to fit.")

        n_results = len(self.get_result_dict())

        # Build the re-sampled fluxes
        flux_array = np.zeros(shape=(n_samples, len(spec.dispersion)))

        # Initialize result array
        result_array = np.zeros(shape=(n_samples, n_results))
        np.random.seed(seed)

        for idx, flux_value in enumerate(spec.fluxden):
            flux_err_value = spec.fluxden_err[idx]
            sampled_flux = np.random.normal(flux_value, flux_err_value,
                                            n_samples)

            flux_array[:, idx] = sampled_flux

        # Fit the resampled spectrum
        for idx in range(n_samples):

            # Create a new SpecFit Object from the current one.
            new_specfit = self.copy()
            # Update the spectrum in the new SpecFit object
            new_specfit.spec.fluxden = flux_array[idx, :]
            new_specfit.update_specmodel_spectra()

            # Fit the new SpecFit object
            new_specfit.fit()
            # Retrieve fit results
            result_dict = self.get_result_dict()
            # Store them in the output array
            result_array[idx, :] = np.array(list(result_dict.values()))

        resampled_df = pd.DataFrame(data=result_array,
                               columns=result_dict.keys())

        resampled_df.to_hdf(foldername + '/resampled_fitting_results' + str(
            n_samples) + '_raw.hdf5', 'data')

        # Evaluate raw resampled fitting results and save median/+-1sigma
        # fit results
        result_df = np.zeros(shape=(3, n_results))

        result_df[0, :] = np.nanpercentile(resampled_df.values, 50, axis=0)
        result_df[1, :] = np.nanpercentile(resampled_df.values, 15.9, axis=0)
        result_df[2, :] = np.nanpercentile(resampled_df.values, 84.1, axis=0)

        result_df = pd.DataFrame(data=result_df,
                                 index=['median', 'lower',
                                        'upper'],
                                 columns=result_dict.keys())
        result_df.to_hdf(foldername + '/resampled_fitting_results' + str(
            n_samples) + '.hdf5', 'data')
        result_df.to_csv(foldername + '/resampled_fitting_results' + str(
            n_samples) + '.csv')


        # Save a plot showing the posterior distribution of all fitting
        # parameters
        if save_result_plots:
            for col in result_df.columns:
                fig = plt.figure(figsize=(8, 8))
                ax = fig.add_subplot(111)

                median = result_df.loc['median', col]
                lower = result_df.loc['lower', col]
                upper = result_df.loc['upper', col]

                ax.hist(resampled_df.loc[:, col], bins=round(n_samples / 2.))
                ax.axvline(median, c='k', ls='--', lw=2)
                ax.axvline(lower, c='k', ls='-.', lw=2)
                ax.axvline(upper, c='k', ls='-.', lw=2)

                med = '{:.4e}'.format(median)
                dp = '{:+.4e}'.format((lower - median))
                dm = '{:+.4e}'.format((upper - median))

                ax.set_title(med + ' ' + dp + ' ' + dm)
                ax.set_xlabel(r'{}'.format(col),
                              fontsize=15)

                plt.savefig(foldername + '/{}_results.pdf'.format(col))

    def get_result_dict(self):
        """Get the best-fit parameter values and return them as a dictionary

        :return: (dict) result_dict
            Dictionary with best-fit parameter values.
        """

        result_dict = {}

        # TODO Finish and test this
        for idx, model in enumerate(self.specmodels):
            model_index = idx
            for param in model.fit_result.params:
                value = model.fit_result.params[param].value
                result_dict.update({param: value})

        return result_dict


    def plot(self):
        """ Plot the astronomical spectrum with all SpecModels

        :return: None
        """

        fig = plt.figure()
        fig.subplots_adjust(hspace=0.4)
        gs = gridspec.GridSpec(4, 1, figure=fig)
        ax_main = fig.add_subplot(gs[:3, 0])
        ax_resid = fig.add_subplot(gs[3:4, 0], sharex=ax_main)

        ax_main.clear()
        ax_resid.clear()

        if self.spec is not None:
            self._plot_specfit(ax_main, ax_resid)

        ax_main.set_ylim(self.ylim)
        ax_main.set_xlim(self.xlim)
        ax_resid.set_xlim(self.xlim)

        plt.show()

    def _update_colors(self):
        """ Update the colors appropriately based on the number of SpecModels

        :return: None
        """

        n = len(self.specmodels)
        if n > 0:
            self.colors = np.arange(n) / n + 1/(2 * n)

            for idx, specmodel in enumerate(self.specmodels):
                specmodel.color = plt.cm.viridis(self.colors[idx])

    def _plot_specfit(self, ax_main, ax_resid):
        """ Internal plotting function to plot all SpecModels

        :param (matplotlib.axes.Axes) ax_main: Axis for main plot
        :param (matplotlib.axes.Axes) ax_resid: Axis for residual plot

        :return: None
        """

        spec = self.spec.copy()

        # Plot the spectrum fluxden error
        if spec.fluxden_err is not None:
            ax_main.plot(spec.dispersion[spec.mask],
                         spec.fluxden_err[spec.mask],
                         'grey')
        ax_main.plot(spec.dispersion[spec.mask], spec.fluxden[spec.mask],
                     'k')

        trans = mtransforms.blended_transform_factory(
            ax_main.transData, ax_main.transAxes)

        # Plot spectrum mask
        mask = np.ones_like(self.spec.fluxden)
        mask[self.spec.mask] = -1
        ax_main.fill_between(self.spec.dispersion, 0, 1,
                             where=(mask > 0),
                             facecolor='0.5', alpha=0.3,
                             transform=trans)

        # Plot the SpecModel masks and best-fit fluxes
        for idx, specmodel in enumerate(self.specmodels):

            color = plt.cm.viridis(self.colors[idx])

            if hasattr(specmodel, 'model_fluxden'):
                ax_main.plot(specmodel.spec.dispersion,
                             specmodel.model_fluxden, color=color, lw=3)

            mask = np.ones_like(self.spec.fluxden)
            mask[np.invert(specmodel.mask)] = -1
            ax_main.fill_between(self.spec.dispersion, 0, 1,
                                 where=(mask > 0),
                                 facecolor=color, alpha=0.3,
                                 transform=trans)

        # Plot the full SpecFit model
        model_fluxden = np.zeros_like(spec.dispersion)
        for specmodel in self.specmodels:
            if hasattr(specmodel, 'model_fluxden'):
                model_fluxden = model_fluxden + specmodel.model_fluxden

        ax_main.plot(spec.dispersion, model_fluxden, color='red', lw=3)

        # Plot the residual
        ax_resid.plot(self.spec.dispersion, self.spec.fluxden - model_fluxden,
                      color='k')

        ax_resid.plot(self.spec.dispersion, self.spec.fluxden*0, color='r',
                      ls=':', lw=2)

        if self.redshift is not None:
            def forward(x):
                return x / (1. + self.redshift)

            def inverse(x):
                return x * (1. + self.redshift)

            ax_main_rest = ax_main.secondary_xaxis('top',
                                                   functions=(forward, inverse))
            ax_main_rest.set_xlabel(r'Rest-frame dispersion ({})'.format(
                self.spec.dispersion_unit.to_string(format='latex')),
                fontsize=14)

        ax_resid.set_xlabel(r'Dispersion ({})'.format(
            self.spec.dispersion_unit.to_string(format='latex')), fontsize=14)

        ax_resid.set_ylabel(r'Fit residual', fontsize=14)

        ax_main.set_ylabel(r'Flux density ({})'.format(
            self.spec.fluxden_unit.to_string(format='latex')), fontsize=14)

        ylim = [0, max(spec.fluxden)]

        ax_main.set_ylim(ylim)





