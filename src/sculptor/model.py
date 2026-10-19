import logging
import os
import time

import chainconsumer as cc
import emcee
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import numpy as np
import pandas as pd
from matplotlib import gridspec
from matplotlib.lines import Line2D
from speconed import speconed as sod
from tqdm import tqdm

from sculptor import plot as scp
from sculptor import utils as scut
from sculptor.persistence import PickleMixin

logger = logging.getLogger(__name__)

cmap = scp.component_cmap

# Definition of the likelihood function


def log_likelihood_chi_square(theta, model, x, y, yerr):
    """Gaussian log-likelihood of the data given the model, over the fit mask.

    log L = -1/2 sum[(y - m)^2 / yerr^2 + ln(yerr^2)], summed over the
    fitted pixels, `model.get_fit_mask()`: the fit regions (`gpm_fit`) minus
    the pixels masked in the spectrum (`spec.mask`).

    :param theta: Values of the variable parameters, in the order of
        `model.params_variable`
    :type theta: np.ndarray
    :param model: The model
    :type model: FitModel
    :param x: Dispersion axis
    :type x: np.ndarray
    :param y: Observed flux density
    :type y: np.ndarray
    :param yerr: Flux density uncertainty
    :type yerr: np.ndarray
    :return: Log-likelihood
    :rtype: float
    """
    # print(f"PID: {os.getpid()} computing log-likelihood")
    fit_mask = model.get_fit_mask()

    # Calculate the full evaluated model at the given x values
    model_flux = model.eval(x, theta)

    # Define sigma ** 2
    sigma2 = yerr[fit_mask] ** 2

    logl = -0.5 * np.sum(
        (y[fit_mask] - model_flux[fit_mask]) ** 2 / sigma2 + np.log(sigma2)
    )

    return logl


def log_prior(theta, parameters):
    """Sum of the log priors of the variable parameters.

    Parameters without a prior contribute 0. Returns as soon as the sum is
    not finite (a value outside a prior's support).

    :param theta: Values of the variable parameters, in the order of
        `model.params_variable`
    :type theta: np.ndarray
    :param parameters: The model's variable parameters
        (`FitModel.params_variable`), name -> FitParameter
    :type parameters: dict
    :return: Log prior
    :rtype: float
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
    """Log posterior probability (log prior + log-likelihood), the function
    sampled by emcee.

    :param theta: Values of the variable parameters, in the order of
        `model.params_variable`
    :type theta: np.ndarray
    :param x: Dispersion axis
    :type x: np.ndarray
    :param model: The model
    :type model: FitModel
    :param y: Observed flux density
    :type y: np.ndarray
    :param yerr: Flux density uncertainty
    :type yerr: np.ndarray
    :return: Log probability; -inf outside the prior support
    :rtype: float
    """
    # The priors are looked up in model.params_variable, which eval() only
    # fills on first use; fill it here, or the first call (on a model that
    # has not been evaluated yet) silently skips the prior.
    if len(model.params_variable) == 0:
        model.get_params_to_sample()

    lp = log_prior(theta, model.params_variable)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood_chi_square(theta, model, x, y, yerr)


def _evaluate_single_sample(args):
    """
    Evaluate the model flux for a single parameter sample.

    :param args: tuple of (fitmodel, dispersion, params, components)
    :type args: tuple
    :return: evaluated model flux
    :rtype: np.ndarray
    """

    fitmodel, dispersion, params, components = args
    return fitmodel.eval(dispersion, params, components=components)


class FitModel(PickleMixin):
    """A spectral model: a sum of components, its parameters, the spectrum
    to fit and the fit results.

    The model flux density is the sum of the components, optionally
    broadened by the instrumental resolution (`eval`). Fits use emcee
    (`run_emcee`); results are read through `get_flat_chain` and the
    plotting/analysis methods.

    The fit uses the pixels inside the fit regions (`gpm_fit`, set with
    `add_wavelength_range_to_fit_mask`) that are not masked in the spectrum
    (`spec.mask`); see `get_fit_mask`.
    """

    def __init__(
        self,
        components=None,
        parameters=None,
        spectrum=None,
        redshift=None,
        resolution=None,
    ):
        """:param components: Model components
        :type components: list(sculptor.component.FitComponent)
        :param parameters: Parameter name -> FitParameter, for all components
            (including shared parameters)
        :type parameters: dict
        :param spectrum: Spectrum to fit; a copy is stored in `spec` and an
            empty fit mask (`gpm_fit`) is created
        :type spectrum: speconed.speconed.SpecOneD
        :param redshift: Redshift of the source, used for the rest-frame axis
            in plots
        :type redshift: float
        :param resolution: Instrumental resolution R = lambda / delta lambda,
            scalar or one value per pixel; if set, `eval` broadens the model by
            it
        :type resolution: float or np.ndarray
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
        # Log-probabilities matching flat_chain, and the burn-in it was built
        # with (None if unknown, e.g. a chain assigned by hand).
        self.flat_log_prob = None
        self.flat_chain_discard = None
        # Burn-in the cached mcmc_model_posterior was built with.
        self.posterior_discard = None

        self.seed = 1234
        self.rng = np.random.default_rng(self.seed)

        self.params_variable = {}
        self.params_constant = {}
        self.params_constant_val = []
        self.vardim = None

        self.resolution = resolution

        self.red_chisq = None
        self.bic = None

        # The spectrum, the good pixel mask (gpm) for the fit and the model
        # flux density from the fit
        self.spec = None
        self.gpm_fit = None
        self.model_fluxden = None

        if spectrum is not None:
            if not isinstance(spectrum, sod.SpecOneD):
                raise TypeError(
                    "spectrum must be a speconed.SpecOneD, "
                    f"not {type(spectrum).__name__}."
                )
            self.spec = spectrum.copy()
            # Set mask describing the regions included in the fit for this model
            self.reset_fit_mask()
            self.model_fluxden = np.zeros_like(self.spec.fluxden)

    def _require_spectrum(self, action):
        """Raise a ValueError if no spectrum is set."""

        # getattr: models pickled before `spec` was always set lack it.
        if getattr(self, "spec", None) is None:
            raise ValueError(
                f"No spectrum set, cannot {action}. Pass `spectrum=` to "
                "FitModel or assign `model.spec` first."
            )

    def add_component(self, component):
        """Add a component to the model.

        :param component: The component
        :type component: sculptor.component.FitComponent
        """

        self.components.append(component)

    def add_parameters(self, parameters):
        """Add parameters to the model (existing names are replaced).

        :param parameters: Parameter name -> FitParameter
        :type parameters: dict
        """

        self.parameters.update(parameters)

    def eval(self, x, params_values, components=None, broaden_by_resolution=True):
        """Evaluate the model at the given x values."""

        # If no components are specified, use all components
        if components is None:
            components = self.components

        # If the parameter dictionaries have not been populated, do so now
        if len(self.params_variable) == 0:
            self.get_params_to_sample()

        params = {}
        params.update(zip(self.params_variable.keys(), params_values, strict=True))
        params.update(
            zip(self.params_constant.keys(), self.params_constant_val, strict=True)
        )

        y = np.zeros_like(x, dtype=float)

        # Iterate over the components and evaluate them using a parameter
        # mapping
        for component in components:
            mapped = map(params.get, component.get_arg_keys())

            y += component.eval(x, mapped)

        # If resolution is provided, convolve the model with the resolution
        if self.resolution is not None and broaden_by_resolution:
            resolution = self.resolution
            if isinstance(resolution, (int, float)):
                resolution = np.full_like(x, resolution, dtype=np.float64)

            y = scut.broaden_spectrum(x, y, resolution, fwhm_lim=5.0)

        return y

    def get_params_to_sample(self):
        """Split the parameters into variable (`params_variable`, sampled) and
        constant ones (`params_constant`, values in `params_constant_val`),
        and set `vardim` to the number of variable parameters.
        """

        self.params_variable = {}
        self.params_constant = {}
        self.params_constant_val = []

        for param in self.parameters:
            if self.parameters[param].vary:
                self.params_variable.update({param: self.parameters[param]})
            else:
                self.params_constant.update({param: self.parameters[param]})
                self.params_constant_val.append(self.parameters[param].value)

        self.vardim = len(self.params_variable)

    def initialize_emcee(
        self, nwalkers, log_probability=log_probability, spec=None, pool=None, **kwargs
    ):
        """Create the emcee sampler (`self.sampler`).

        Clears everything derived from a previous chain (cached flat chain,
        log-probabilities and posterior model).

        :param nwalkers: Number of walkers
        :type nwalkers: int
        :param log_probability: Function to sample, with the signature of
            `sculptor.model.log_probability`
        :type log_probability: callable
        :param spec: Spectrum to fit instead of `self.spec`
        :type spec: speconed.speconed.SpecOneD
        :param pool: Pool for parallel evaluation (e.g. `multiprocessing.Pool`)
        :param kwargs: Passed on to `emcee.EnsembleSampler`
        :raises ValueError: if neither `spec` nor `self.spec` is set
        """

        self.get_params_to_sample()

        # A new sampler invalidates everything derived from the old chain.
        self.flat_chain = None
        self.flat_log_prob = None
        self.flat_chain_discard = None
        self.mcmc_model_posterior = None
        self.posterior_discard = None

        if spec is None:
            self._require_spectrum("initialize emcee")
            spec = self.spec

        # TODO: Test if there is really no way to reduce the number of pixels
        #  here, if we want to take the resolution into account
        #  Moved application of the fit mask to the log_chi_quasared function
        # wave = spec.dispersion[self.gpm_fit]
        # flux = spec.fluxden[self.gpm_fit]
        # flux_err = spec.fluxden_err[self.gpm_fit]

        wave = spec.dispersion
        flux = spec.fluxden
        flux_err = spec.fluxden_err

        sampler = emcee.EnsembleSampler(
            nwalkers,
            self.vardim,
            log_probability,
            args=(wave, self, flux, flux_err),
            pool=pool,
            **kwargs,
        )

        self.sampler = sampler

    def initialize_positions(self):
        """Initial walker positions, drawn from each variable parameter's prior.

        A parameter without a prior is drawn from a Gaussian around its value
        with a width of 20% of its absolute value (a warning is logged).
        Uses `self.nwalkers` and the model's random generator `self.rng`.

        :return: Positions, shape (nwalkers, number of variable parameters)
        :rtype: np.ndarray
        """

        self.get_params_to_sample()

        pos = np.zeros((self.nwalkers, self.vardim))

        logger.info("Retrieving the initial guess from the parameter priors")

        for idx, par_name in enumerate(self.params_variable):
            param = self.params_variable[par_name]

            if param.prior is not None:
                logger.info(f"Sampled prior for {par_name}")
                pos[:, idx] = param.prior.sample(self.nwalkers, rng=self.rng)

            else:
                logger.warning(
                    f"No prior specified for {par_name}: sampling the initial "
                    f"walker positions from a Gaussian around {param.value} "
                    "with a width of 20% of its absolute value."
                )
                value = param.value
                pos[:, idx] = self.rng.normal(
                    loc=value, scale=0.2 * np.abs(value), size=self.nwalkers
                )

        return pos

    def run_emcee(
        self,
        nsteps,
        nwalkers,
        log_probability=log_probability,
        spec=None,
        pos=None,
        progress=True,
        pool=None,
    ):
        """Run the emcee sampler.

        :param nsteps: Number of steps
        :type nsteps: int
        :param nwalkers: Number of walkers
        :type nwalkers: int
        :param log_probability: Function to sample (see `initialize_emcee`)
        :type log_probability: callable
        :param spec: Spectrum to fit instead of `self.spec`
        :type spec: speconed.speconed.SpecOneD
        :param pos: Initial positions, shape (nwalkers, number of variable
            parameters); drawn from the priors if not given
            (`initialize_positions`)
        :type pos: np.ndarray
        :param progress: Show emcee's progress bar
        :type progress: bool
        :param pool: Pool for parallel evaluation
        :raises ValueError: if `pos` has the wrong shape
        """

        self.nsteps = nsteps
        self.nwalkers = nwalkers

        self.initialize_emcee(
            nwalkers, log_probability=log_probability, spec=spec, pool=pool
        )

        if pos is None:
            pos = self.initialize_positions()
        else:
            if pos.shape[0] != nwalkers or pos.shape[1] != self.vardim:
                raise ValueError("Provided position array has wrong shape.")

        self.sampler.run_mcmc(pos, nsteps, progress=progress)

    def get_components_by_prefix(self, prefixes):
        """Return the components whose name starts with any of the prefixes.

        :param prefixes: Prefix, or list of prefixes, to match component
            names against
        :type prefixes: str or list(str)
        :return: List of matching FitComponent instances, in model order
        :rtype: list
        """

        if isinstance(prefixes, str):
            prefixes = [prefixes]
        prefixes = tuple(prefixes)

        return [comp for comp in self.components if comp.name.startswith(prefixes)]

    def get_all_prefix_components(self, prefix):
        """
        Return a list of components whose names start with the given prefix.

        :param prefix: String prefix to match component names
        :return: List of matching FitComponent instances
        """
        return self.get_components_by_prefix([prefix])

    def calculate_chi_sq(self):
        """
        Calculate chi square statistic using median model from the fit and observed data.
        Only considers the fitted pixels (`get_fit_mask()`).

        :return chi_sq, dof, reduced_chi_sq: Chi-square, degree of freedom, reduced chi-square
        :rtype: tuple
        """
        self._require_spectrum("calculate the chi-square")
        if self.mcmc_model_posterior is None:
            raise ValueError(
                "No posterior model available. Run get_mcmc_model_results() first."
            )
        fit_mask = self.get_fit_mask()
        flux_exp = self.spec.fluxden[fit_mask]
        median_model = self.mcmc_model_posterior[0]
        flux_model = median_model[fit_mask]
        flux_exp_err = self.spec.fluxden_err[fit_mask]
        dof = len(flux_model) - len(self.params_variable)

        chi_sq = np.sum(((flux_exp - flux_model) / flux_exp_err) ** 2)

        reduced_chi_sq = chi_sq / dof
        self.red_chisq = reduced_chi_sq
        return chi_sq, dof, reduced_chi_sq

    def compute_bic(self, discard=2000):
        """
        Compute the Bayesian Information Criterion (BIC) for a given model.
        Only considers the fitted pixels (`get_fit_mask()`).

        :param discard: Number of burn-in steps to discard
        :type discard: int
        :return bic: BIC factor
        """

        self._require_spectrum("compute the BIC")
        flat_chain = self.get_flat_chain(discard=discard)

        x = self.spec.dispersion
        y = self.spec.fluxden
        yerr = self.spec.fluxden_err

        logls = []
        for theta in tqdm(flat_chain, desc="Computing BIC"):
            logl = log_likelihood_chi_square(theta, self, x, y, yerr)
            logls.append(logl)

        max_logl = np.max(logls)
        n_params = len(self.params_variable)
        n_data = np.sum(self.get_fit_mask())

        bic = n_params * np.log(n_data) - 2 * max_logl
        self.bic = bic
        return bic

    # -----------------------------------------------------------------------
    # Functions related to the good pixel fit mask (gpm_fit)
    # -----------------------------------------------------------------------
    def get_fit_mask(self):
        """Pixels used in the fit: inside the fit regions (`gpm_fit`) and not
        masked in the spectrum (`spec.mask`, True = good pixel).

        Pixels masked in the spectrum (e.g. bad or NaN pixels) are excluded
        from the likelihood, the chi-square and the BIC even if they lie in a
        fit region. Adopted from the upstream `kat` branch (4ee90b4).

        :return: Boolean mask over `spec.dispersion`
        :rtype: np.ndarray
        """

        spec_mask = getattr(self.spec, "mask", None)
        if spec_mask is None:
            return self.gpm_fit
        return self.gpm_fit & spec_mask

    def reset_fit_mask(self):
        """Reset the fit mask based on the supplied astronomical spectrum.

        :return: None
        """

        self._require_spectrum("reset the fit mask")
        self.gpm_fit = np.zeros_like(self.spec.dispersion, dtype="bool")

    def add_wavelength_range_to_fit_mask(self, disp_x1, disp_x2):
        """Adding a wavelength region to the fit mask.

        The dispersion region between the two dispersion values will be added
        to the fit mask.

        :param (float) disp_x1: Dispersion value 1
        :param (float) disp_x2: Dispersion value 2
        :return:
        """

        logger.info(f"Manual mask range: {disp_x1} - {disp_x2}")

        self._require_spectrum("add a range to the fit mask")
        if self.gpm_fit is None:
            self.reset_fit_mask()

        mask_between = np.sort(np.array([disp_x1, disp_x2]))
        lo_index = np.argmin(np.abs(self.spec.dispersion - mask_between[0]))
        up_index = np.argmin(np.abs(self.spec.dispersion - mask_between[1]))

        self.gpm_fit[lo_index:up_index] = True

    def remove_wavelength_range_from_fit_mask(self, disp_x1, disp_x2):
        """Removing a wavelength region to the fit mask.

        The dispersion region between the two dispersion values will be removed
        from the fit mask.

        :param (float) disp_x1: Dispersion value 1
        :param (float) disp_x2: Dispersion value 2
        :return:
        """

        logger.info(f"Manual mask range: {disp_x1} - {disp_x2}")

        self._require_spectrum("remove a range from the fit mask")
        if self.gpm_fit is None:
            self.reset_fit_mask()

        mask_between = np.sort(np.array([disp_x1, disp_x2]))
        lo_index = np.argmin(np.abs(self.spec.dispersion - mask_between[0]))
        up_index = np.argmin(np.abs(self.spec.dispersion - mask_between[1]))

        self.gpm_fit[lo_index:up_index] = False

    # -----------------------------------------------------------------------
    # Functions related to the model evaluation
    # -----------------------------------------------------------------------

    def get_flat_chain(self, discard=2000):
        """Return the flattened MCMC chain after discarding the burn-in.

        This is the one place the chain is read from. With a sampler, the
        chain (and its log-probabilities) are read from it and cached in
        `flat_chain` / `flat_log_prob`. Without one (e.g. a chain assigned
        by hand), the cached `flat_chain` is returned; it cannot be
        re-trimmed, so asking for a different `discard` than it was built
        with raises a ValueError.

        :param discard: Number of burn-in steps to discard
        :type discard: int
        :return: Flat chain, shape (n_samples, n_variable_params)
        :rtype: np.ndarray
        """

        if self.sampler is not None:
            if self.flat_chain is None or self.flat_chain_discard != discard:
                self.flat_chain = self.sampler.get_chain(discard=discard, flat=True)
                self.flat_log_prob = self.sampler.get_log_prob(
                    discard=discard, flat=True
                )
                self.flat_chain_discard = discard
            return self.flat_chain

        if self.flat_chain is None:
            raise ValueError("No MCMC chain available. Run the MCMC first.")

        built_with = getattr(self, "flat_chain_discard", None)
        if built_with is not None and built_with != discard:
            raise ValueError(
                f"flat_chain was built with discard={built_with} and there is "
                f"no sampler to re-apply discard={discard}. Use "
                f"discard={built_with}."
            )
        return self.flat_chain

    def get_flat_log_prob(self, discard=2000):
        """Return the log-probabilities matching `get_flat_chain(discard)`.

        Computed from the chain if they are not cached (e.g. a chain
        assigned by hand).
        """

        flat_chain = self.get_flat_chain(discard=discard)
        if getattr(self, "flat_log_prob", None) is None:
            self._require_spectrum("compute log-probabilities")
            self.flat_log_prob = np.array(
                [
                    log_probability(
                        theta,
                        self.spec.dispersion,
                        self,
                        self.spec.fluxden,
                        self.spec.fluxden_err,
                    )
                    for theta in flat_chain
                ]
            )
        return self.flat_log_prob

    def get_mode_model(self, discard=2000):
        """Model at the posterior sample with the highest log-probability.

        Called the "maximum likelihood" (ML) model in the plots; strictly it is
        the maximum of the posterior among the samples, as the log-probability
        includes the priors.

        :param discard: Number of burn-in steps to discard
        :type discard: int
        :return: Model flux density on `self.spec.dispersion`
        :rtype: np.ndarray
        """

        self._require_spectrum("evaluate the mode model")

        # Get the maximum likelihood model
        flat_chain = self.get_flat_chain(discard=discard)
        max_likelihood = np.argmax(self.get_flat_log_prob(discard=discard))
        theta_max = flat_chain[max_likelihood]
        model_max = self.eval(self.spec.dispersion, theta_max)

        logger.info(f"Maximum likelihood model: {theta_max}")
        logger.info(f"Variable parameter names: {list(self.params_variable)}")

        return model_max

    def _posterior_model_bands(self, discard=2000, components=None, pool=None):
        """Median and 1-sigma bands of the model over the posterior samples.

        No side effects; `get_mcmc_model_results` stores the result.

        :return: median, lower (15.87%) and upper (84.13%) model flux density
        :rtype: tuple(np.ndarray, np.ndarray, np.ndarray)
        """

        self._require_spectrum("evaluate the posterior model")
        flat_chain = self.get_flat_chain(discard=discard)
        dispersion = self.spec.dispersion

        param_list = [(self, dispersion, params, components) for params in flat_chain]

        if pool:
            logger.info("Evaluating MCMC model with multiprocessing.")
            start_time = time.time()
            model_fluxden = list(
                tqdm(
                    pool.imap(_evaluate_single_sample, param_list),
                    total=len(param_list),
                )
            )
            end_time = time.time()
            logger.info(
                "MCMC model evaluation completed in "
                f"{end_time - start_time:.2f} seconds."
            )

        else:
            model_fluxden = [_evaluate_single_sample(p) for p in tqdm(param_list)]

        model_fluxden = np.array(model_fluxden)

        med_model = np.median(model_fluxden, axis=0)
        low_model = np.percentile(model_fluxden, 15.87, axis=0)
        upp_model = np.percentile(model_fluxden, 84.13, axis=0)

        return med_model, low_model, upp_model

    def get_mcmc_model_results(self, discard=2000, components=None, pool=None):
        """Evaluate the model for every posterior sample and store the median and
        the 1-sigma band (15.87 and 84.13 percentiles) in
        `mcmc_model_posterior` (and the `discard` used in `posterior_discard`).

        :param discard: Number of burn-in steps to discard
        :type discard: int
        :param components: Only evaluate these components (default: all)
        :type components: list(sculptor.component.FitComponent)
        :param pool: Pool for parallel evaluation (anything with `imap`)
        """

        self.mcmc_model_posterior = list(
            self._posterior_model_bands(
                discard=discard, components=components, pool=pool
            )
        )
        self.posterior_discard = discard

    def plot_mcmc_result(
        self,
        discard=2000,
        show_fit_mask=True,
        show_ml_model=True,
        ylim=None,
        xlim=None,
        resid_ylim=None,
        save=False,
        save_dir=".",
        show_components=False,
        save_name="fit_result.pdf",
        save_data=False,
        pool=None,
    ):
        """Plot the spectrum with the posterior model and the residuals.

        Shows the median model with its 1-sigma band, optionally the
        maximum-likelihood model, the per-component medians and the fit mask,
        and the reduced chi-square in the legend.

        :param discard: Number of burn-in steps to discard
        :type discard: int
        :param show_fit_mask: Shade the fitted regions
        :type show_fit_mask: bool
        :param show_ml_model: Also plot the maximum-likelihood model and its
            residuals (see `get_mode_model`)
        :type show_ml_model: bool
        :param ylim: y-axis limits of the spectrum panel
        :type ylim: [float, float]
        :param xlim: x-axis limits
        :type xlim: [float, float]
        :param resid_ylim: y-axis limits of the residual panel
        :type resid_ylim: [float, float]
        :param save: Save the figure instead of showing it
        :type save: bool
        :param save_dir: Directory for the figure and data files
        :type save_dir: str
        :param show_components: Plot the posterior median of every component
        :type show_components: bool
        :param save_name: File name of the figure
        :type save_name: str
        :param save_data: Also write the plotted models as CSV files
        :type save_data: bool
        :param pool: Pool for parallel evaluation of the posterior models
        :return: Residuals of the median model and the matching dispersion
            (both within the spectrum's mask)
        :rtype: tuple(np.ndarray, np.ndarray)
        """

        self._require_spectrum("plot the MCMC result")

        scp.set_presentation_defaults()

        # Define the figure
        # plt.clf()
        fig = plt.figure(figsize=(14, 8))

        # Define the gridspec
        gs = gridspec.GridSpec(
            2, 1, height_ratios=[3, 1]
        )  # 3:1 ratio for main plot to residuals

        ax_main = fig.add_subplot(gs[0])  # Main plot
        ax_resid = fig.add_subplot(gs[1], sharex=ax_main)  # Residuals plot

        # Generate the axis transform
        trans = mtransforms.blended_transform_factory(
            ax_main.transData, ax_main.transAxes
        )

        # Plot the gpm fit mask
        if show_fit_mask:
            mask = np.ones_like(self.spec.fluxden)
            mask[np.invert(self.get_fit_mask())] = -1
            ax_main.fill_between(
                self.spec.dispersion,
                0,
                0.05,
                where=(mask >= 0),
                facecolor="0.5",
                alpha=0.3,
                transform=trans,
            )

        # Plot the spectrum flux density error
        if self.spec.fluxden_err is not None:
            ax_main.step(
                self.spec.dispersion[self.spec.mask],
                self.spec.fluxden_err[self.spec.mask],
                "grey",
                where="mid",
                lw=1.5,
            )
        # Plot the spectrum flux density
        ax_main.step(
            self.spec.dispersion[self.spec.mask],
            self.spec.fluxden[self.spec.mask],
            "k",
            where="mid",
            lw=1.5,
        )

        # Plot the MCMC model results
        logger.info("Evaluating the MCMC model results.")
        if (
            self.mcmc_model_posterior is None
            or getattr(self, "posterior_discard", None) != discard
        ):
            self.get_mcmc_model_results(discard=discard, pool=pool)

        med_model, low_model, upp_model = self.mcmc_model_posterior

        if save_data:
            filename = os.path.join(save_dir, "mcmc_model_results.csv")
            np.savetxt(
                filename,
                np.vstack([self.spec.dispersion, med_model, low_model, upp_model]).T,
                delimiter=",",
                header="dispersion, med_model, low_model, upp_model",
            )

        # Plot the median model
        ax_main.step(
            self.spec.dispersion,
            med_model,
            color=scp.dblue,
            where="mid",
            label="Median model",
            lw=2,
        )
        # Plot the 1-sigma model region
        ax_main.fill_between(
            self.spec.dispersion,
            low_model,
            upp_model,
            color=scp.dblue,
            alpha=0.3,
            step="mid",
        )

        if show_components:
            logger.info("Evaluating the components in the MCMC chain.")
            n_comp = len(self.components)
            for ldx, component in enumerate(self.components):
                comp = component.name
                c = cmap(ldx / n_comp)
                med_comp, _, _ = self._posterior_model_bands(
                    discard=discard, components=[component], pool=pool
                )
                ax_main.step(
                    self.spec.dispersion,
                    med_comp,
                    where="mid",
                    color=c,
                    lw=1.5,
                    label=comp,
                )

                if save_data:
                    filename = os.path.join(save_dir, f"mcmc_model_results_{comp}.csv")
                    np.savetxt(
                        filename,
                        np.vstack([self.spec.dispersion, med_comp]).T,
                        delimiter=",",
                        header="dispersion, med_comp",
                    )

        # Plot the maximum likelihood model
        if show_ml_model:
            self.model_fluxden = self.get_mode_model(discard=discard)
            ax_main.step(
                self.spec.dispersion,
                self.model_fluxden,
                color=scp.vermillion,
                where="mid",
                label="ML model",
                ls="--",
                lw=2,
            )

            if save_data:
                filename = os.path.join(save_dir, "mcmc_model_results_ml.csv")
                np.savetxt(
                    filename,
                    np.vstack([self.spec.dispersion, self.model_fluxden]).T,
                    delimiter=",",
                    header="dispersion, model_flux",
                )

        ax_main.plot(self.spec.dispersion, self.spec.dispersion * 0, "k:", lw=1.5)

        # Residual plot
        ax_resid.fill_between(
            self.spec.dispersion,
            -self.spec.fluxden_err,
            self.spec.fluxden_err,
            color="0.5",
            alpha=0.8,
            step="mid",
        )

        # Plot the residuals
        residuals = self.spec.fluxden[self.spec.mask] - med_model[self.spec.mask]
        ax_resid.step(
            self.spec.dispersion[self.spec.mask],
            residuals,
            color=scp.dblue,
            where="mid",
            lw=1.5,
            label="Median residuals",
            zorder=-100,
        )
        if show_ml_model:
            residuals_ml = (
                self.spec.fluxden[self.spec.mask] - self.model_fluxden[self.spec.mask]
            )
            ax_resid.step(
                self.spec.dispersion[self.spec.mask],
                residuals_ml,
                color=scp.vermillion,
                where="mid",
                lw=1.5,
                label="ML residuals",
                zorder=-100,
            )

        ax_resid.plot(self.spec.dispersion, self.spec.dispersion * 0, "k:", lw=1.5)

        ax_resid.set_ylabel("Residuals", fontsize=20)

        # Add the rest-frame/observed wavelength axis and labels
        if self.redshift:
            ax_main.tick_params(axis="x", which="both", top=False)

            ax_resid.set_xlabel(
                "Observed wavelength ({})".format(
                    self.spec.dispersion_unit.to_string(format="latex")
                ),
                fontsize=14,
            )

            def wave_to_restwave(wav):
                """Observed to rest-frame wavelength."""

                return wav / (1 + self.redshift)

            def restwave_to_wave(restwav):
                """Rest-frame to observed wavelength."""

                return restwav * (1 + self.redshift)

            secax = ax_main.secondary_xaxis(
                "top", functions=(wave_to_restwave, restwave_to_wave)
            )
            secax.set_xlabel(
                "Rest-frame wavelength ({})".format(
                    self.spec.dispersion_unit.to_string(format="latex")
                ),
                fontsize=20,
            )

        else:
            ax_resid.set_xlabel(
                "Wavelength ({})".format(
                    self.spec.dispersion_unit.to_string(format="latex")
                ),
                fontsize=20,
            )

        # Add reduced-chi2 value on the plot for the fit
        _chi_sq, _dof, reduced_chi_sq = self.calculate_chi_sq()
        # bic = self.compute_bic()

        stats_handle = Line2D(
            [0], [0], color="none", label=f"reduced chi-sq = {reduced_chi_sq:.2f}"
        )
        handles, labels = ax_main.get_legend_handles_labels()
        handles.append(stats_handle)
        labels.append(stats_handle.get_label())

        ax_main.set_ylabel(
            "Flux density ({})".format(
                self.spec.fluxden_unit.to_string(format="latex")
            ),
            fontsize=14,
        )

        # Get the y-axis limits
        if ylim is None:
            y_max = 1.1 * np.max(self.spec.fluxden[self.get_fit_mask()])
            if self.spec.fluxden_err is not None:
                y_min = -np.max(self.spec.fluxden_err[self.get_fit_mask()])
            else:
                y_min = 0

            ylim = [y_min, y_max]

        # Set the axis limits
        ax_main.set_ylim(ylim)
        if xlim is not None:
            ax_main.set_xlim(xlim)

        if resid_ylim is None:
            yerr_lim = 2 * np.max(self.spec.fluxden_err[self.get_fit_mask()])
            ax_resid.set_ylim([-yerr_lim, yerr_lim])
        else:
            ax_resid.set_ylim(resid_ylim)

        ax_main.legend(handles=handles, labels=labels, fontsize=12)
        ax_resid.legend(ncol=2)

        if save:
            filename = os.path.join(save_dir, save_name)
            plt.savefig(filename, bbox_inches="tight")
        else:
            plt.show()

        return residuals, self.spec.dispersion[self.spec.mask]

    def plot_posterior_corner(
        self,
        discard=2000,
        parameters=None,
        save=False,
        save_dir=".",
        save_name="corner_plot.pdf",
    ):
        """Corner plot of the posterior (chainconsumer), and its summary.

        :param discard: Number of burn-in steps to discard
        :type discard: int
        :param parameters: Parameter names to plot (default: all variable
            parameters)
        :type parameters: list(str)
        :param save: Save the figure instead of showing it
        :type save: bool
        :param save_dir: Directory for the figure
        :type save_dir: str
        :param save_name: File name of the figure
        :type save_name: str
        :return: chainconsumer summary (parameter name -> bounds)
        :rtype: dict
        """

        chain = self.get_flat_chain(discard=discard)

        # Get the parameter names
        param_names = list(self.params_variable.keys())

        # Replace underscores
        param_names_readable = [name.replace("_", " ") for name in param_names]

        # Create a pandas DataFrame with the chain
        df_chain = pd.DataFrame(chain, columns=param_names_readable)

        if parameters is None:
            parameters = param_names_readable
        else:
            parameters = [name.replace("_", " ") for name in parameters]

        # Instantiate the chain consumer
        c = cc.ChainConsumer()
        chain = cc.Chain(samples=df_chain, name="MCMC chain")
        c.add_chain(chain)

        logger.info(f"Posterior summary:\n{c.analysis.get_summary()}")

        if save:
            filename = os.path.join(save_dir, save_name)
            c.plotter.plot(columns=parameters, filename=filename)
        else:
            c.plotter.plot(columns=parameters)

        return c.analysis.get_summary()

    def plot_posterior_param_histograms(
        self, discard, save=False, save_dir=".", save_name_prefix="posterior_hist"
    ):
        """
        Plot posterior histograms for MCMC samples from a given model, generating separate plots for each parameter.

        :param discard: number of steps to discard as burn-in
        :type discard: int
        :param save: whether to save the plots instead of displaying them
        :type save: boolean
        :param save_dir: directory to save the plots
        :type save_dir: str
        :param save_name_prefix: prefix for the saved file names
        :type save_name_prefix: str
        """

        flat_chain = self.get_flat_chain(discard=discard)
        param_names = list(self.params_variable.keys())

        if not param_names:
            raise ValueError("No parameters found.")

        if save:
            os.makedirs(save_dir, exist_ok=True)

        for i, param in enumerate(param_names):
            samples = flat_chain[:, i]

            median = np.percentile(samples, 50)
            lower = np.percentile(samples, 15.9)
            upper = np.percentile(samples, 84.1)

            plt.figure(figsize=(6, 4))
            plt.hist(samples, bins=100, color="skyblue", edgecolor="black", alpha=0.7)
            plt.axvline(
                median, color="k", linestyle="--", lw=2, label=f"Median: {median:.3f}"
            )
            plt.axvline(
                lower, color="r", linestyle="-.", lw=2, label=f"Lower: {lower:.3f}"
            )
            plt.axvline(
                upper, color="r", linestyle="-.", lw=2, label=f"Upper: {upper:.3f}"
            )

            plt.title(f"Posterior Hist: {param}")
            plt.xlabel(param)
            plt.legend()

            if save:
                plt.savefig(os.path.join(save_dir, f"{save_name_prefix}_{param}.pdf"))
                plt.close()
            else:
                plt.show()
