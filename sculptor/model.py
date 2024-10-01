
import os
import emcee
import itertools
import numpy as np
import pandas as pd
import chainconsumer as cc

from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.transforms as mtransforms

from speconed import speconed as sod

from IPython import embed

from sculptor import plot as scp
from sculptor import colors as tolc

cmap = tolc.tol_cmap(colormap='rainbow_PuRd')

# Definition of the likelihood function

def log_likelihood_chi_square(theta, model, x, y, yerr):
    """ Calculate the chi-square log likelihood for the model.

    :param theta:
    :param model:
    :param x:
    :param y:
    :param yerr:
    :return:
    """

    sigma2 = yerr ** 2

    logl = -0.5 * np.sum((y - model.eval(x, theta)) ** 2 /
                         sigma2 + np.log(sigma2))

    return logl


def log_prior(theta, parameters):
    """ Calculate the log prior for the model parameters.

    :param theta:
    :param parameters:
    :return:
    """

    sum = 0

    # Possibly banish this into the model!
    for idx, param in enumerate(parameters):

        prior = parameters[param].prior
        # Make sure that a prior was populated
        if prior is not None:
            sum += prior.logprior(theta[idx])

        if not np.isfinite(sum):
            return sum

    return sum


def log_probability(theta, x, model, y, yerr):
    """ Calculate the log probability for the model.

    :param theta:
    :param x:
    :param model:
    :param y:
    :param yerr:
    :return:
    """
    lp = log_prior(theta, model.params_variable)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood_chi_square(theta, model, x, y, yerr)

class FitModel(object):

    def __init__(self, components=None, parameters=None, spectrum=None,
                 redshift=None):
        """ Initialize the FitModel class.

        :param components:
        :param parameters:
        :param spectrum:
        :param redshift:
        """

        if parameters is None:
            parameters = {}
        if components is None:
            components = []
        self.components = components
        self.parameters = parameters
        self.redshift = redshift

        self.mcmc_model_posterior = None
        self.sampler = None
        self.nsteps = None
        self.nwalkers = None
        self.flat_chain = None

        self.seed = 1234
        self.rng = np.random.default_rng(self.seed)

        if isinstance(spectrum, sod.SpecOneD):
            self.spec = spectrum.copy()
            # Set mask describing the regions included in the fit for this model
            self.reset_fit_mask()
            self.model_fluxden = np.zeros_like(self.spec.fluxden)

        else:
            # The good pixel mask (gpm) for the fit
            self.gpm_fit = None
            # The model flux density from the fit
            self.model_fluxden = None


    def add_component(self, component: list):
        """ Add a component to the model.

        :param component:
        :return:
        """

        self.components.append(component)

    def add_parameters(self, parameters: dict):
        """ Add a parameter to the model.

        :param parameters:
        :return:
        """

        self.parameters.update(parameters)


    def eval(self, x, params_values, components=None):
        """ Evaluate the model at the given x values."""

        # If no components are specified, use all components
        if components is None:
            components = self.components

        params = {}
        params.update(zip(self.params_variable.keys(), params_values))
        params.update(zip(self.params_constant.keys(), self.params_constant_val))

        y = np.zeros_like(x, dtype=float)

        # Iterate over the components and evaluate them using a parameter
        # mapping
        for component in components:

            pmap = component.param_mapping
            args = component.function_args

            mapped = map(params.get, map(pmap.get, args))

            y += component.eval(x, mapped)

        return y


    def get_params_to_sample(self):

        params_variable = dict()
        params_constant = dict()
        params_constant_val = []

        for param in self.parameters:
            if self.parameters[param].vary:
                params_variable.update({param: self.parameters[param]})
            else:
                params_constant.update({param: self.parameters[param]})
                params_constant_val.append(self.parameters[param].value)

        self.params_variable = params_variable
        self.vardim = len(self.params_variable)
        self.params_constant = params_constant
        self.params_constant_val = params_constant_val

    def initialize_emcee(self, nwalkers, log_probability=log_probability,
                         spec=None):

        self.get_params_to_sample()

        if spec is None:

            if self.spec is None:
                raise ValueError('No spectrum provided for the model.')

            spec = self.spec

        wave = spec.dispersion[self.gpm_fit]
        flux = spec.fluxden[self.gpm_fit]
        flux_err = spec.fluxden_err[self.gpm_fit]

        sampler = emcee.EnsembleSampler(nwalkers, self.vardim, log_probability,
                                        args=(wave, self, flux, flux_err))

        self.sampler = sampler

    def initialize_positions(self):

        self.get_params_to_sample()

        pos = np.zeros((self.nwalkers, self.vardim))

        print('[INFO] Retrieving the initial guess from the parameter priors')

        for idx, par_name in enumerate(self.params_variable):

            param = self.params_variable[par_name]

            if param.prior is not None:
                print('[INFO] Sampled prior for {}'.format(par_name))
                pos[:, idx] = param.prior.sample(self.nwalkers, rng=self.rng)

            else:
                print('[INFO] No prior specified for {}'.format(par_name))
                print('[INFO] Sampling Gaussian around input value {}'.format(param.value))
                print('[INFO] with width of 20% of the absolute value.')
                print('[WARNING] This is a hack!')
                value = param.value
                pos[:, idx] = self.rng.normal(loc=value, scale=0.2 * np.abs(value),
                                               size=self.nwalkers)


        return pos

    def run_emcee(self, nsteps, nwalkers, log_probability=log_probability,
                  spec=None, pos=None, progress=True):
        """ Run the emcee sampler.

        :param nsteps:
        :param nwalkers:
        :param log_probability:
        :param spec:
        :param pos:
        :param progress:
        :return:
        """

        self.nsteps = nsteps
        self.nwalkers = nwalkers

        self.initialize_emcee(nwalkers, log_probability=log_probability,
                              spec=spec)

        if pos is None:
            pos = self.initialize_positions()
        else:
            if pos.shape[0] != nwalkers or pos.shape[1] != self.vardim:
                raise ValueError('[ERROR] Provided position array has wrong shape.')

        self.sampler.run_mcmc(pos, nsteps, progress=progress)

    # -----------------------------------------------------------------------
    # Functions related to the good pixel fit mask (gpm_fit)
    # -----------------------------------------------------------------------
    def reset_fit_mask(self):
        """Reset the fit mask based on the supplied astronomical spectrum.

        :return: None
        """

        self.gpm_fit = np.zeros_like(self.spec.dispersion, dtype='bool')


    def add_wavelength_range_to_fit_mask(self, disp_x1, disp_x2):
        """ Adding a wavelength region to the fit mask.

        The dispersion region between the two dispersion values will be added
        to the fit mask.

        :param (float) disp_x1: Dispersion value 1
        :param (float) disp_x2: Dispersion value 2
        :return:
        """

        print('[INFO] Manual mask range', disp_x1, disp_x2)

        if hasattr(self, 'spec'):
            mask_between = np.sort(np.array([disp_x1, disp_x2]))
            lo_index = np.argmin(np.abs(self.spec.dispersion - mask_between[0]))
            up_index = np.argmin(np.abs(self.spec.dispersion - mask_between[1]))

            self.gpm_fit[lo_index:up_index] = True

    def remove_wavelength_range_from_fit_mask(self, disp_x1, disp_x2):
        """ Removing a wavelength region to the fit mask.

        The dispersion region between the two dispersion values will be removed
        from the fit mask.

        :param (float) disp_x1: Dispersion value 1
        :param (float) disp_x2: Dispersion value 2
        :return:
        """

        print('[INFO] Manual mask range', disp_x1, disp_x2)

        if hasattr(self, 'spec'):
            mask_between = np.sort(np.array([disp_x1, disp_x2]))
            lo_index = np.argmin(np.abs(self.spec.dispersion - mask_between[0]))
            up_index = np.argmin(np.abs(self.spec.dispersion - mask_between[1]))

            self.gpm_fit[lo_index:up_index] = False

    # -----------------------------------------------------------------------
    # Functions related to the model evaluation
    # -----------------------------------------------------------------------

    def get_mode_model(self, discard=2000):

        # Get the maximum likelihood model
        max_likelihood = np.argmax(self.sampler.get_log_prob(discard=discard, flat=True))
        theta_max = self.sampler.get_chain(discard=discard, flat=True)[max_likelihood]
        model_max = self.eval(self.spec.dispersion, theta_max)

        print('[INFO] Maximum likelihood model: ', theta_max)
        print('[INFO] Variable parameter names: ', self.params_variable.keys())

        return model_max

    def get_mcmc_model_results(self, discard=2000, components=None):

        if self.sampler is not None:
             self.flat_chain = self.sampler.get_chain(discard=discard, flat=True)
        elif self.flat_chain is None:
            raise ValueError('No MCMC chain available. Run the MCMC first.')

        dispersion = self.spec.dispersion
        model_fluxden = np.zeros((self.flat_chain.shape[0], len(dispersion)))

        for idx in tqdm(range(self.flat_chain.shape[0])):
            model_fluxden[idx, :] = self.eval(dispersion,
                                              self.flat_chain[idx, :],
                                              components=components)

        med_model = np.median(model_fluxden, axis=0)
        low_model = np.percentile(model_fluxden, 15.87, axis=0)
        upp_model = np.percentile(model_fluxden, 84.13, axis=0)

        self.mcmc_model_posterior = [med_model, low_model, upp_model]

    def plot_mcmc_result(self, discard=2000, show_fit_mask=True,
                         show_ml_model=True, ylim=None, xlim=None,
                         save=False, save_dir='.', show_components=False,
                         save_name='fit_result.pdf', save_data=False):

        scp.set_presentation_defaults()

        # Define the figure
        plt.clf()
        fig = plt.figure(figsize=(14, 8))

        # Define the gridspec
        gs = gridspec.GridSpec(2, 1,
                               height_ratios=[3, 1])  # 3:1 ratio for main plot to residuals

        ax_main = fig.add_subplot(gs[0])  # Main plot
        ax_resid = fig.add_subplot(gs[1], sharex=ax_main)  # Residuals plot

        # Generate the axis transform
        trans = mtransforms.blended_transform_factory(
            ax_main.transData, ax_main.transAxes)

        # Plot the gpm fit mask
        if show_fit_mask:
            mask = np.ones_like(self.spec.fluxden)
            mask[np.invert(self.gpm_fit)] = -1
            ax_main.fill_between(self.spec.dispersion, 0, 0.05,
                                 where=(mask >= 0),
                                 facecolor='0.5', alpha=0.3,
                                 transform=trans)

        # Plot the spectrum flux density error
        if self.spec.fluxden_err is not None:
            ax_main.step(self.spec.dispersion[self.spec.mask],
                         self.spec.fluxden_err[self.spec.mask],
                         'grey', where='mid', lw=1.5)
        # Plot the spectrum flux density
        ax_main.step(self.spec.dispersion[self.spec.mask],
                     self.spec.fluxden[self.spec.mask],
                     'k', where='mid', lw=1.5)

        # Plot the MCMC model results
        print('[INFO] Evaluating the MCMC model results.')
        n_flat_chain = (self.nsteps - discard) * self.nwalkers
        if (self.mcmc_model_posterior is None or
                self.flat_chain.shape[0] < n_flat_chain):
            self.get_mcmc_model_results(discard=discard)

        med_model, low_model, upp_model = self.mcmc_model_posterior

        if save_data:
            filename = os.path.join(save_dir, 'mcmc_model_results.csv')
            np.savetxt(filename, np.vstack([self.spec.dispersion, med_model, low_model, upp_model]).T,
                       delimiter=',', header='dispersion, med_model, low_model, upp_model')

        # Plot the median model
        ax_main.step(self.spec.dispersion, med_model, color=scp.dblue, where='mid',
                     label='Median model', lw=2)
        # Plot the 1-sigma model region
        ax_main.fill_between(self.spec.dispersion, low_model, upp_model,
                             color=scp.dblue, alpha=0.3, step='mid')

        if show_components:
            comp_names = [comp.name for comp in self.components]
            comp_dict = dict(zip(comp_names, self.components))
            comp_array = np.zeros((self.flat_chain.shape[0],
                                   len(comp_names),
                                   self.spec.dispersion.shape[0]))

            print('[INFO] Evaluating the components in the MCMC chain.')
            for idx in tqdm(range(self.flat_chain.shape[0])):

                for ldx, comp in enumerate(comp_names):

                    comp_array[idx, ldx, :] = self.eval(
                        self.spec.dispersion, self.flat_chain[idx, :],
                        components=[comp_dict[comp]])

            for ldx, comp in enumerate(comp_names):
                c = cmap(ldx / len(comp_names))
                med_comp = np.median(comp_array[:, ldx, :], axis=0)
                ax_main.step(self.spec.dispersion, med_comp, where='mid',
                             color=c, lw=1.5, label=comp)

                if save_data:
                    filename = os.path.join(save_dir, 'mcmc_model_results_{}.csv'.format(comp))
                    np.savetxt(filename, np.vstack([self.spec.dispersion, med_comp]).T,
                               delimiter=',', header='dispersion, med_comp')

        # Plot the maximum likelihood model
        if show_ml_model and self.sampler is not None:
            self.model_fluxden = self.get_mode_model(discard=discard)
            ax_main.step(self.spec.dispersion,
                         self.model_fluxden, color=scp.vermillion, where='mid',
                         label='ML model', ls='--', lw=2)

            if save_data:
                filename = os.path.join(save_dir, 'mcmc_model_results_ml.csv')
                np.savetxt(filename, np.vstack([self.spec.dispersion, self.model_fluxden]).T,
                           delimiter=',', header='dispersion, model_flux')

        ax_main.plot(self.spec.dispersion, self.spec.dispersion * 0, 'k:',
                     lw=1.5)

        # Residual plot
        ax_resid.fill_between(self.spec.dispersion, -self.spec.fluxden_err,
                              self.spec.fluxden_err, color='0.5', alpha=0.3,
                              step='mid')

        # Plot the residuals
        residuals = self.spec.fluxden[self.spec.mask] - med_model[self.spec.mask]
        ax_resid.step(self.spec.dispersion[self.spec.mask], residuals,
                      color=scp.dblue, where='mid', lw=1.5, label='Median residuals')
        residuals = self.spec.fluxden[self.spec.mask] - self.model_fluxden[self.spec.mask]
        ax_resid.step(self.spec.dispersion[self.spec.mask], residuals,
                      color=scp.vermillion, where='mid', lw=1.5, label='ML residuals')

        ax_resid.plot(self.spec.dispersion, self.spec.dispersion * 0, 'k:',
                     lw=1.5)

        ax_resid.set_ylabel('Residuals', fontsize=14)

        # Add the rest-frame/observed wavelength axis and labels
        if self.redshift:
            ax_main.tick_params(axis='x', which='both', top=False)

            ax_resid.set_xlabel('Observed wavelength ({})'.format(
                self.spec.dispersion_unit.to_string(format='latex')), fontsize=14)

            def wave_to_restwave(wav):
                return wav / (1 + self.redshift)

            def restwave_to_wave(restwav):
                return restwav * (1 + self.redshift)

            secax = ax_main.secondary_xaxis('top',
                                            functions=(wave_to_restwave,
                                                       restwave_to_wave))
            secax.set_xlabel('Rest-frame wavelength ({})'.format(
                self.spec.dispersion_unit.to_string(format='latex')),
                fontsize=14)

        else:
            ax_resid.set_xlabel('Wavelength ({})'.format(
                self.spec.dispersion_unit.to_string(format='latex')),
                fontsize=14)

        ax_main.set_ylabel('Flux density ({})'.format(
            self.spec.fluxden_unit.to_string(format='latex')), fontsize=14)

        # Get the y-axis limits
        if ylim is None:
            y_max = 1.1 * np.max(self.spec.fluxden[self.gpm_fit])
            if self.spec.fluxden_err is not None:
                y_min = -np.max(self.spec.fluxden_err[self.gpm_fit])
            else:
                y_min = 0

            ylim = [y_min, y_max]

        # Set the axis limits
        ax_main.set_ylim(ylim)
        if xlim is not None:
            ax_main.set_xlim(xlim)

        yerr_lim = 2 * np.max(self.spec.fluxden_err[self.gpm_fit])
        ax_resid.set_ylim([-yerr_lim, yerr_lim])

        ax_main.legend()
        ax_resid.legend(ncol=2)

        if save:
            filename = os.path.join(save_dir, save_name)
            plt.savefig(filename, bbox_inches='tight')
        else:
            plt.show()

    def plot_posterior_corner(self, discard=2000, parameters=None, save=False,
                              save_dir='.', save_name='corner_plot.pdf'):

        chain = self.sampler.get_chain(discard=discard, flat=True)

        # Get the parameter names
        param_names = list(self.params_variable.keys())

        # Replace underscores
        param_names_readable = [name.replace('_', ' ') for name in param_names]

        # Create a pandas DataFrame with the chain
        df_chain = pd.DataFrame(chain, columns=param_names_readable)

        if parameters is None:
            parameters = param_names_readable
        else:
            parameters = [name.replace('_', ' ') for name in parameters]

        # Instantiate the chain consumer
        c = cc.ChainConsumer()
        chain = cc.Chain(samples=df_chain, name='MCMC chain')
        c.add_chain(chain)

        if save:
            filename = os.path.join(save_dir, save_name)
            c.plotter.plot(columns=parameters,
                           filename=filename)
        else:
            c.plotter.plot(columns=parameters)



    def plot_posterior_histograms(self, discard=2000, parameters=None):

        pass

    def evaluate_parameters(self, discard=2000):

        chain = self.sampler.get_chain(discard=discard, flat=True)

        # Get the parameter names
        param_names = list(self.params_variable.keys())
        # Replace underscores
        param_names_readable = [name.replace('_', ' ') for name in param_names]

        # Create a pandas DataFrame with the chain
        df_chain = pd.DataFrame(chain, columns=param_names_readable)

        # Instantiate the chain consumer
        c = cc.ChainConsumer()
        c.add_chain(df_chain)

        # Create individual corner plots
        combs = itertools.combinations(param_names_readable, 2)

        # TODO: Fine tune the corner plot settings to
        #  - include the parameter name nicely
        #  - include the maximum likelihood value
        #  - display the parameter name properly

        for comb in combs:
            filename = './corner_plot_{}__{}.pdf'.format(comb[0].replace(' ', '_'),
                                                   comb[1].replace(' ', '_'))
            c.plotter.plot(parameters=comb,
                           filename=filename)



