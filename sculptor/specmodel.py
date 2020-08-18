
import os
import glob
import corner
import importlib
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms

from lmfit import Model, Parameters, fit_report
from lmfit.model import save_model, load_model, save_modelresult, \
    load_modelresult

from sculptor import speconed as sod



# For a full list of fitting method
# https://lmfit.github.io/lmfit-py/fitting.html
fitting_methods = {'Levenberg-Marquardt': 'leastsq',
                   'Nelder-Mead': 'nelder',
                   'Maximum likelihood via Monte-Carlo Markov Chain':'emcee',
                   'Least-Squares minimization': 'least_squares',
                   'Differential evolution': 'differential_evolution',
                   'Brute force method': 'brute',
                   'Basinhopping': 'basinhopping',
                   'Adaptive Memory Programming for Global Optimization':
                       'ampgo',
                   'L-BFGS-B': 'lbfgsb',
                   'Powell': 'powell',
                   'Conjugate-Gradient':'cg',
                   # 'Newton-CG':'newton',
                   'Cobyla': 'cobyla',
                   'BFGS': 'bfgs',
                   'Truncated Newton': 'tnc',
                   # 'Newton-CG trust-region':'trust-ncg',
                   # 'Nearly Exact trust-region':'trust-exact',
                   'Newton GLTR trust-region': 'trust-krylov',
                   'Trust-region for constrained obtimization': 'trust-constr',
                   # 'Dog-leg trust-region':'dogleg',
                   'Sequential Linear Squares Programming':'slsqp',
                   'Simplicial Homology Global Optimization':'shgo',
                   'Dual Annealing Optimization':'dual_annealing'
                   }
"""dict: Dictionary of fitting methods 

Fitting methods available for fitting in SpecFit based on the list of methods in 
LMFIT.
"""



# Make all models and masks available to the SpecModel class
model_func_list = []
""" list of str: List of model names"""
model_func_dict = {}
""" dict: Dictionary of model functions"""
model_setup_list = []
""" dict: Dictionary of model setup function names"""
mask_presets = {}
""" dict: Dictionary of mask presets"""

file_names = glob.glob(os.path.join(os.path.split(__file__)[
                                        0] + '/models_and_masks/*.py'))
module_names = [os.path.basename(f)[:-3] for f in file_names if
                os.path.isfile(f) and not f.endswith('__init__.py')]

for module_name in module_names:
    print('[INFO] Import "models_and_masks" module: {}'.format(module_name))
    module = importlib.import_module('sculptor.models_and_masks.{}'.format(
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


class SpecModel:
    """ Class holding information on models for the SpecFit class


    Attributes:
        specfit (SpecFit): Associated SpecFit object
        xlim (list of float): Wavelength limits for plotting
        ylim (list of float): Flux density limits for plotting
        spec (SpecOneD): Astronomical spectrum as a SpecOneD object
        redshift (float): Cosmological redshift of the astronomical object
        use_weights (bool): Boolean to indicate whether flux errors will be
            used as weights for the fit.
        model_list (list of Models): List of LMFIT models
        params_list (list of Parameters): List of LMFIT Parameters for all
            LMFIT models.
        global_params (Parameters): Global parameters to be added to the all
            models in the Specmodel. Their main use is to provide variables
            and constraints for multiple individual models.
        color (str): Color to use in the SpecModel plot.
        model (Model): LMFIT SpecModel model. The global model including all
            models in the model_list.
        params (Parameters): LMFIT SpecModel parameters. The global parameter
            list including all parameters from all models.
        fit_result(ModelResult): LMFIT ModelResult for the fit to the SpecModel

    """

    def __init__(self, specfit, spectrum=None, redshift=0):
        """ Initialisation of the SpecModel object """


        self.xlim = [0, 1]
        self.ylim = [0, 1]

        # Copy the spectrum (SpecOneD object) to the SpecModel object
        # if isinstance(spectrum, sod.SpecOneD)
        self.specfit = specfit

        if isinstance(spectrum, sod.SpecOneD):
            self.spec = spectrum.copy()
            # Set mask describing the regions included in the fit for this model
            self.reset_fit_mask()
            self.reset_plot_limits()
            self.model_flux = np.zeros_like(self.spec.flux)

        else:
            self.mask = None

        self.redshift = redshift

        # Boolean to indicate whether the flux uncertainties will be used as
        # weights in the fit
        self.use_weights = True

        self.name = 'SpecModel'
        # list of functional models
        self.model_list = []
        # list of parameters for the models
        self.params_list = []

        self.global_params = Parameters()

        self.color = 'orange'

        self.model = None
        self.params = None
        self.fit_result = None


    def _copy(self, specfit):
        # TODO Test this function and make sure it copies and not links

        specmodel = SpecModel(specfit, spectrum=self.spec,
                              redshift=self.redshift)

        specmodel.mask = self.mask

        specmodel.redshift = self.redshift

        # Boolean to indicate whether the flux uncertainties will be used as
        # weights in the fit
        specmodel.use_weights = self.use_weights

        specmodel.name = self.name
        # list of functional models
        specmodel.model_list = self.model_list.copy()
        # list of parameters for the models
        specmodel.params_list = self.params_list.copy()

        specmodel.global_params = self.global_params.copy()

        specmodel.color = self.color

        specmodel.build_model()

        specmodel.fit_result = self.fit_result

        return specmodel

    def add_model(self, model, params):
        """ Add a model to the SpecModel object

        :param (Model) model: LMFIT Model to be added to the model list.
        :param (Parameters) params: LMFIT Parameters to be added
        to the
            parameter list.

        :return: None
        """
        # TODO FIX THIS SHIT IS BROKEN!!!!
        print('MODEL LIST!!!')
        print(self.model_list)
        print(len(self.model_list))

        prefix_list = []
        for existing_model in self.model_list:
            prefix_list.append(existing_model.prefix)
        print('PREFIX LIST', prefix_list)

        if len(self.model_list) > 0:

            if isinstance(model, list) and isinstance(params, list):
                if model[0].prefix in prefix_list:
                    print('[WARNING] Model with same name exists. \
                                #                 Model could not be added.')
                else:
                    self.model_list.extend(model)
                    self.params_list.extend(params)
            else:
                if model.prefix in prefix_list:
                    print('[WARNING] Model with same name exists. \
                                #                 Model could not be added.')
                else:
                    self.model_list.append(model)
                    self.params_list.append(params)

        else:
            if isinstance(model, list) and isinstance(params, list):
                self.model_list.extend(model)
                self.params_list.extend(params)

            else:
                self.model_list.append(model)
                self.params_list.append(params)


    def delete_model(self, index=None):
        """ Delete model (Model, Parameters) from the SpecModel object.

        :param (int) index: Index of model to remove from model_list and
            Parameters to remove from params_list (default index=="None"). If
            the index is None the last model will be removed.

        :return: None
        """

        if len(self.model_list) > 0:
            if index is None:
                model_to_delete = self.model_list.pop()
                params_to_delete = self.params_list.pop()
            else:
                model_to_delete = self.model_list.pop(index)
                params_to_delete = self.params_list.pop(index)

            # Delete the Model and Parameter objects
            del model_to_delete
            del params_to_delete


    def add_wavelength_range_to_fit_mask(self, disp_x1, disp_x2):
        """ Adding a wavelength region to the fit mask.

        The dispersion region between the two dispersion values will be added
        to the fit mask.

        :param (float) disp_x1: Dispersion value 1
        :param (float) disp_x2: Dispersion value 2
        :return:
        """

        print('Manual mask range', disp_x1, disp_x2)

        if hasattr(self, 'spec'):
            mask_between = np.sort(np.array([disp_x1, disp_x2]))
            lo_index = np.argmin(np.abs(self.spec.dispersion - mask_between[0]))
            up_index = np.argmin(np.abs(self.spec.dispersion - mask_between[1]))

            self.mask[lo_index:up_index] = True

    def reset_fit_mask(self):
        """ Reset the fit mask based on the supplied astronomical spectrum."""

        self.mask = np.zeros_like(self.spec.dispersion, dtype='bool')

    def add_mask_preset_to_fit_mask(self, mask_preset_key):
        """ Adding a preset mask from the models_and_masks module to the fit.

        :param mask_preset_key (str): Name of the preset mask in the
            mask_preset dictionary.

        :return: None
        """

        mask_preset = mask_presets[mask_preset_key]

        if mask_preset['rest_frame']:
            one_p_z = 1 + self.redshift
        else:
            one_p_z = 1

        for mask_range in mask_preset['mask_ranges']:

            wave_a = mask_range[0] * one_p_z
            wave_b = mask_range[1] * one_p_z

            self.add_wavelength_range_to_fit_mask(wave_a, wave_b)

    def add_global_param(self, param_name, value=None, vary=True,
                         min=-np.inf, max=np.inf, expr=None):
        """ Adding a "Global Parameter" to the SpecModel object

        :param (str) param_name: Name of the global parameter
        :param (float, optional) value: Initial value of the
        global
        parameter
        :param (bool, optional) vary: Boolean to indicate whether the global
        parameter
            should be varied during the fit
        :param (float, optional) min: Minimum value for the global parameter
        :param (float, optional) max: Maximum value for the global parameter
        :param (str, optional) expr: Optional expression for the global
            parameter

        :return: None
        """

        self.global_params.add(param_name, value=value, vary=vary, min=min,
                               max=max, expr=expr)

        self.update_model_params_for_global_params()


    def remove_global_param(self, param_name):
        """ Remove "Global Parameter" from SpecModel object

        Parameters
        :param (str) param_name: Parameter name of the global parameter to
            remove.

        :return: None
        """

        if param_name in self.global_params:
            self.global_params.pop(param_name)

        for params in self.params_list:
            if param_name in params:
                params.pop(param_name)


    def build_model(self):
        """ Build the Specmodel model and parameters for the fit

        :return: None
        """

        # If at least one model exists
        if len(self.model_list) > 0 and len(self.params_list) > 0:
            # Instantiate the SpecModel model parameters
            self.params = Parameters()

            # Add super parameters (Test!)
            self.params.update(self.specfit.super_params)

            # Add global parameters (includes super parameters)
            self.params.update(self.global_params)

            for params in self.params_list:
                self.params.update(params)

            # Build the full SpecModel model
            # Instantiate the model with the first model in the list
            self.model = self.model_list[0]
            # Add all other models to the global model
            for model in self.model_list[1:]:
                self.model += model

            # Evaluate the model with the initial parameters
            self.model_flux = self.model.eval(self.params,
                                              x=self.spec.dispersion)


    def fit(self):
        """ Fit the SpecModel to the astronomical spectrum

        :return: None
        """

        # (re-) build the model from the model and params lists
        self.build_model()

        fit_mask = np.logical_and(self.mask, self.spec.mask)

        emcee_kws = self.specfit.emcee_kws

        if len(self.spec.flux[fit_mask]) > 0:

            if self.use_weights:
                weights = 1./self.spec.flux_err[fit_mask]**2

                if fitting_methods[self.specfit.fitting_method] == 'emcee':
                    emcee_kws['is_weighted'] = True
                    self.fit_result = self.model.fit(self.spec.flux[fit_mask],
                                                     self.params,
                                                     x=self.spec.dispersion[
                                                         fit_mask],
                                                     weights=weights,
                                                     method=fitting_methods[
                                                         self.specfit.fitting_method],
                                                     fit_kws =emcee_kws)

                else:
                    self.fit_result = self.model.fit(self.spec.flux[fit_mask],
                                                     self.params,
                                                     x=self.spec.dispersion[
                                                         fit_mask],
                                                     weights=weights,
                                                     method=fitting_methods[
                                                         self.specfit.fitting_method])



            else:
                emcee_kws['is_weighted'] = False
                if fitting_methods[self.specfit.fitting_method] == 'emcee':
                    emcee_kws['is_weighted'] = True
                    self.fit_result = self.model.fit(self.spec.flux[fit_mask],
                                                     self.params,
                                                     x=self.spec.dispersion[
                                                         fit_mask],
                                                     method=fitting_methods[
                                                         self.specfit.fitting_method],
                                                     fit_kws=emcee_kws)

                else:
                    self.fit_result = self.model.fit(self.spec.flux[fit_mask],
                                                     self.params,
                                                     x=self.spec.dispersion[
                                                         fit_mask],
                                                     method=fitting_methods[
                                                         self.specfit.fitting_method])

            self.model_flux = self.model.eval(self.fit_result.params,
                                              x=self.spec.dispersion)


            self.update_params_from_fit_result()


    def update_model_params_for_global_params(self):
        """ Global parameters are added to the Model parameters.

        :return: None
        """

        # Iterate through all global params
        for param in self.global_params:
            # And through all Parameters in the params_list
            for model_params in self.params_list:
                # Check if the param is not in the model Parameters
                if param not in model_params:
                    # Add param to model Parameters
                    model_params.add(self.global_params[param])


    def update_params_from_fit_result(self):
        """Update all parameter values in the parameter list based on the
        fit result.

        Individual model parameter, global parameters and even the super
        parameters of the associated SpecFit object will be updated based on
        the fit.

        :return: None
        """

        if self.fit_result is not None:
            # Update super params
            for param in self.specfit.super_params:

                temp_val = self.fit_result.params[param].value
                self.specfit.super_params[param].value = temp_val

                temp_val = self.fit_result.params[param].expr
                self.specfit.super_params[param].expr = temp_val

                temp_val = self.fit_result.params[param].min
                self.specfit.super_params[param].min = temp_val

                temp_val = self.fit_result.params[param].max
                self.specfit.super_params[param].max = temp_val

                temp_val = self.fit_result.params[param].vary
                self.specfit.super_params[param].vary = temp_val

            # Update global params
            for param in self.global_params:

                temp_val = self.fit_result.params[param].value
                self.global_params[param].value = temp_val

                temp_val = self.fit_result.params[param].expr
                self.global_params[param].expr = temp_val

                temp_val = self.fit_result.params[param].min
                self.global_params[param].min = temp_val

                temp_val = self.fit_result.params[param].max
                self.global_params[param].max = temp_val

                temp_val = self.fit_result.params[param].vary
                self.global_params[param].vary = temp_val

            # Update parameters for all models in model_list
            for idx, model in enumerate(self.model_list):
                # For each model retrieve the parameter set
                params = self.params_list[idx]
                # Iterate through each parameter and update it
                for jdx, param in enumerate(params):

                    temp_val = self.fit_result.params[param].value
                    self.params_list[idx][param].value = temp_val

                    temp_val = self.fit_result.params[param].expr
                    self.params_list[idx][param].expr = temp_val

                    temp_val = self.fit_result.params[param].min
                    self.params_list[idx][param].min = temp_val

                    temp_val = self.fit_result.params[param].max
                    self.params_list[idx][param].max = temp_val

                    temp_val = self.fit_result.params[param].vary
                    self.params_list[idx][param].vary = temp_val

        else:
            raise ValueError("No fit result available")

    def save_fit_report(self, foldername, specmodel_id=None, show=False):
        """ Save the fit report to a file in the specified folder

        :param (str) foldername: Specified folder in which the fit report will
            be saved.
        :param (str) specmodel_id: Unique SpecModel identifier used in creating
            the filename for the fit report.
        :param (bool) show:  Boolean to indicate whether the fit report
        should
            also be printed to the screen.

        :return: None
        """

        if self.fit_result is not None:
            # Check if folder exists otherwise create it
            if not os.path.exists(foldername):
                os.makedirs(foldername)

            if specmodel_id is None:
                specmodel_id = self.name

            # Save fit result to folder
            savefile = open(foldername +'/specmodel_{}_fit_report.txt'.format(
                specmodel_id), 'w')
            report = fit_report(self.fit_result)
            savefile.write(report)
            savefile.close()

            if show:
                print(report)

        else:
            print('[WARNING] The SpecModel has not been fitted, yet.')

    def save(self, foldername, specmodel_id=0):
        """ Save the SpecModel object to a specified folder

        :param (str) foldername: Specified folder in which the SpecModel will
            be saved.
        :param (str) specmodel_id: Unique SpecModel identifier used in creating
            the filenames for the save files.

        :return: None
        """

        # Check if folder exists otherwise create it
        if not os.path.exists(foldername):
            os.makedirs(foldername)

        # Check if any fit result and model files exist
        # If some files with same ID exist, remove them!
        model_list = glob.glob(foldername + '/{}_*model.json'.format(
            specmodel_id))

        for file in model_list:
            print("Removing old model file: {}".format(file))
            os.remove(file)

        # Save fit result if it exists
        if self.fit_result is not None:
            print("Saving fit result")
            save_modelresult(self.fit_result,
                             foldername + '/{}_fitresult.json'.format(
                                 specmodel_id))

        # Save specmodel global parameters
        if len(self.global_params) > 0:
            gp_file = open(foldername + '/{}_global_params.json'.format(
                specmodel_id), 'w')
            self.global_params.dump(gp_file)
            gp_file.close()

        # Save all models in SpecModel
        for model in self.model_list:
            model_file = foldername+'/{}_{}_model.json'.format(specmodel_id,
                                                               model.prefix)
            print("Saving new model file: {}".format(model_file))
            save_model(model, model_file)

        # Save SpecModel data to fit.hdf5 file
        # Previous data is overwritten
        save_file = foldername+'/fit.hdf5'
        key = 'specmodel_{}'.format(specmodel_id)

        df = self._specmodel_data_to_df()
        df.to_hdf(save_file, key=key)

        df = self._specmodel_meta_to_df()
        df.to_hdf(save_file, key=key+'_meta')


    def _specmodel_data_to_df(self):
        """ Create a DataFrame with all SpecModel data

        This function is used internally to save the SpecModel data to a file.

        :return: df (pandas.DataFrame)
        """

        data = []
        columns = []

        data.append(self.spec.dispersion)
        columns.append('spec_dispersion')
        data.append(self.spec.flux)
        columns.append('spec_flux')
        data.append(self.spec.mask)
        columns.append('spec_mask')
        if hasattr(self.spec, 'flux_err'):
            data.append(self.spec.flux_err)
            columns.append('spec_fluxerror')

        data.append(self.mask)
        columns.append('mask')

        if hasattr(self, 'model_flux'):
            data.append(self.model_flux)
            columns.append('model_flux')


        df = pd.DataFrame(np.array(data).T, columns=columns)

        return df


    def _specmodel_meta_to_df(self):
        """ Create a DataFrame with all SpecModel meta data

        This function is used internally to save the SpecModel meta data to a
        file.

        :return: df (pandas.DataFrame)
        """

        data = []
        columns = []

        data.append([self.use_weights])
        columns.append('use_weights')
        data.append([self.name])
        columns.append('name')
        data.append([self.xlim])
        columns.append('xlim')
        data.append([self.ylim])
        columns.append('ylim')
        data.append([self.redshift])
        columns.append('redshift')

        df = pd.DataFrame(np.array(data), index=columns)

        return df


    def load(self, foldername, specmodel_id):
        """ Load a SpecModel from the specified folder.

        :param (str) foldername: Specified folder in which the SpecModel will
            be saved.
        :param (str) specmodel_id: Unique SpecModel identifier used in creating
            the filenames for the save files.

        :return: None
        """

        # Read in data frames
        key = 'specmodel_{}'.format(specmodel_id)
        data = pd.read_hdf('{}/fit.hdf5'.format(foldername), key=key)
        meta = pd.read_hdf('{}/fit.hdf5'.format(foldername), key=key+'_meta')

        # Initialize new spectrum from data frame
        self.spec = sod.SpecOneD(dispersion=data['spec_dispersion'].values,
                                 flux=data['spec_flux'].values,
                                 mask=np.array(data['spec_mask'].values,
                                 dtype=bool),
                                 unit='f_lam')

        if hasattr(data, 'spec_fluxerror'):
            self.spec.flux_err = data['spec_fluxerror'].values

        # Read in the specmodel mask
        self.mask = np.array(data['mask'].values, dtype=bool)

        # Read in the model flux from the data frame if available
        if hasattr(data, 'model_flux'):
            self.model_flux = data['model_flux'].values

        # Read in meta data from meta data frame
        if meta.loc['use_weights', 0] == True:
            self.use_weights = True
        else:
            self.use_weights = False

        self.name = meta.loc['name', 0]
        self.xlim = meta.loc['xlim', 0]
        self.ylim = meta.loc['ylim', 0]

        if 'redshift' in meta.index:
            self.redshift = meta.loc['redshift', 0]

        # Load models and params into the model_list and params_list
        model_list = glob.glob(foldername + '/{}_*model.json'.format(
            specmodel_id))

        self.model_list = []
        self.params_list = []

        for model_name in model_list:
            # Load model
            model = load_model(model_name, funcdefs=model_func_dict)
            # Initialize parameters
            params = Parameters()
            pars = model.make_params()
            for p in pars:
                params.add(pars[p])


            self.model_list.append(model)
            self.params_list.append(params)

        # Load global parameters
        if os.path.isfile(foldername + '/{}_global_params.json'.format(specmodel_id)):
            gp_file = open(foldername + '/{}_global_params.json'.format(specmodel_id), 'r')
            self.global_params.load(gp_file)
            gp_file.close()

            self.update_model_params_for_global_params()

        # Load the model fit results
        if os.path.isfile(foldername+'/{}_fitresult.json'.format(specmodel_id)):
            # Load the result into the class
            self.fit_result = load_modelresult(
                foldername+'/{}_fitresult.json'.format(specmodel_id),
                funcdefs=model_func_dict)
            # Update the parameters of the model functions with the fit results
            self.update_params_from_fit_result()


    def reset_plot_limits(self):
        """ Reset the plot limits based on the astronomical spectrum.

        :return: None
        """

        if hasattr(self, 'spec'):
            self.xlim = [min(self.spec.dispersion),
                         max(self.spec.dispersion)]
            self.ylim = [0,
                         max(self.spec.flux) * 1.05]

    def plot(self):
        """ Plot the SpecModel

        :return: None
        """

        fig = plt.figure()
        ax_main = fig.add_subplot(1,1,1)

        ax_main.clear()

        self._plot_specmodel(ax_main)

        plt.show()

    def _plot_specmodel(self, ax_main):
        """  Internal plotting function to plot the SpecModel

        :param (matplotlib.axes.Axes) ax_main: Axis for main plot

        :return: None
        """

        spec = self.spec.copy()

        # Plot the spectrum flux error
        if spec.flux_err is not None:
            ax_main.plot(spec.dispersion[spec.mask],
                         spec.flux_err[spec.mask],
                         'grey')
        ax_main.plot(spec.dispersion[spec.mask], spec.flux[spec.mask],
                     'k')

        # Plot model flux
        if hasattr(self, 'model_flux'):
            ax_main.plot(self.spec.dispersion,
                         self.model_flux, color=self.color, lw=3)

        trans = mtransforms.blended_transform_factory(
            ax_main.transData, ax_main.transAxes)

        # Plot spectrum mask
        mask = np.ones_like(self.spec.flux)
        mask[self.spec.mask] = -1
        ax_main.fill_between(self.spec.dispersion, 0, 1,
                             where=(mask > 0),
                             facecolor='0.5', alpha=0.3,
                             transform=trans)

        # Plot SpecModel mask
        mask = np.ones_like(self.spec.flux)
        mask[np.invert(self.mask)] = -1
        ax_main.fill_between(self.spec.dispersion, 0, 1,
                             where=(mask > 0),
                             facecolor=self.color, alpha=0.3,
                             transform=trans)

        ax_main.set_xlabel(r'$\rm{Observed-frame\ wavelength}\ (\rm{'
                           r'\AA})$')

        # ax_main.set_ylabel(r'$\rm{Flux\ density}\ f_{\lambda}\ ('
        #                    r'\rm{'
        #                    r'erg}\,\rm{s}^{'
        #                    r'-1}\,\rm{cm}^{-2}\,\rm{\AA}^{'
        #                    r'-1})$')

        ax_main.set_ylabel(r'$\rm{Flux\ density}\ \rm{(arbitrary\ units)}$')