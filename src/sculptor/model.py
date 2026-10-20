import logging
import os
import time
import warnings

import chainconsumer as cc
import emcee
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import numpy as np
import pandas as pd
from matplotlib import gridspec
from matplotlib.lines import Line2D
from scipy.ndimage import median_filter
from speconed import speconed as sod
from tqdm import tqdm

from sculptor import plot as scp
from sculptor import prior as scpri
from sculptor import quick_fit as scqf
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

    total = 0

    for idx, param in enumerate(parameters):
        prior = parameters[param].prior
        if prior is not None:
            total += prior.logprior(theta[idx])

        if not np.isfinite(total):
            return total

    return total


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
    # The priors are looked up in model.params_variable; refresh it, or a
    # model that was not evaluated yet (or whose parameters changed since)
    # would use a stale or empty set of priors.
    model.get_params_to_sample()

    lp = log_prior(theta, model.params_variable)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood_chi_square(theta, model, x, y, yerr)


def _short_runs(mask, max_width):
    """Keep only the runs of True in `mask` that are at most `max_width`
    elements long."""

    padded = np.concatenate([[False], mask, [False]])
    edges = np.flatnonzero(np.diff(padded.astype(int)))
    out = np.zeros_like(mask)
    for start, stop in zip(edges[::2], edges[1::2], strict=True):
        if stop - start <= max_width:
            out[start:stop] = True
    return out


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
    (`run_emcee`), or an optimizer for a quick best fit (`run_quick_fit`);
    both use the same log-probability, and `fit` selects one. MCMC results
    are read through `get_flat_chain` and the plotting/analysis methods.

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

        # Result of the last quick fit (`run_quick_fit`)
        self.quick_fit_result = None

        # The spectrum, the good pixel mask (gpm) for the fit and the model
        # flux density from the fit
        self.spec = None
        self.gpm_fit = None
        # Pixels kept by the last sigma clipping (`sigma_clip`), None if none
        # was done
        self.clip_mask = None
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
        """Evaluate the model (the sum of its components) at `x`.

        The variable parameters take the values in `params_values`; the fixed
        ones their current `value`. The split into variable and fixed
        parameters is rebuilt on every call (`get_params_to_sample`), so
        changes to a parameter's `value` or `vary` take effect immediately.

        :param x: Dispersion axis
        :type x: np.ndarray
        :param params_values: Values of the variable parameters, in the order
            of `params_variable` (the order of `parameters` with `vary=True`)
        :type params_values: sequence of float
        :param components: Only evaluate these components (default: all)
        :type components: list(sculptor.component.FitComponent)
        :param broaden_by_resolution: Broaden by `resolution`, if it is set
        :type broaden_by_resolution: bool
        :return: Model flux density
        :rtype: np.ndarray
        :raises ValueError: if `params_values` does not have one value per
            variable parameter
        """

        if components is None:
            components = self.components

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
        elif self.gpm_fit is not None and len(spec.dispersion) != len(self.gpm_fit):
            raise ValueError(
                f"`spec` has {len(spec.dispersion)} pixels but the fit mask "
                f"(from `self.spec`) has {len(self.gpm_fit)}; the likelihood "
                "applies the model's fit mask to `spec`."
            )

        # The full spectrum is passed (the resolution broadening needs the
        # neighboring pixels); the fit mask is applied in the likelihood.
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
        # Seed emcee's stretch moves too, so that runs are reproducible.
        sampler.random_state = np.random.RandomState(self.seed).get_state()

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

    def run_quick_fit(
        self,
        log_probability=log_probability,
        spec=None,
        method="L-BFGS-B",
        options=None,
        update_values=False,
        clip=None,
    ):
        """Quick fit: find the maximum of the posterior (MAP) with an optimizer.

        Maximizes the same `log_probability` the MCMC samples (priors plus
        likelihood over the fit mask), starting from the parameters' current
        values. Uniform priors become bounds. The result, with Gauss-Newton
        uncertainties, is stored in `quick_fit_result` and returned. Analysis
        functions take its `values` like a posterior sample, e.g.
        `analyze_mcmc_results(model, result.values[None, :], ...)`.

        :param log_probability: Function to maximize (see `initialize_emcee`)
        :type log_probability: callable
        :param spec: Spectrum to fit instead of `self.spec` (same pixels)
        :type spec: speconed.speconed.SpecOneD
        :param method: `scipy.optimize.minimize` method that supports bounds
        :type method: str
        :param options: Passed on to `scipy.optimize.minimize`
        :type options: dict
        :param update_values: Also set the variable parameters' `value` to
            the best fit (e.g. to start the next fit from there)
        :type update_values: bool
        :param clip: Sigma-clip and refit until the clip mask no longer
            changes: keyword arguments of `sigma_clip`, plus
            `max_iterations` (default 5); e.g. `{"low": 3, "grow": 2}`
        :type clip: dict
        :return: The fit result
        :rtype: sculptor.quick_fit.QuickFitResult
        :raises ValueError: if no spectrum is set, `spec` has a different
            number of pixels, a variable parameter has no value, or `clip`
            is combined with `spec`
        """

        if clip is not None and spec is not None:
            raise ValueError("`clip` uses `self.spec`; do not pass `spec`.")
        if spec is None:
            self._require_spectrum("run the quick fit")
            spec = self.spec
        elif self.gpm_fit is not None and len(spec.dispersion) != len(self.gpm_fit):
            raise ValueError(
                f"`spec` has {len(spec.dispersion)} pixels but the fit mask "
                f"(from `self.spec`) has {len(self.gpm_fit)}."
            )

        def _fit():
            return scqf.quick_fit(
                self,
                log_probability,
                spec.dispersion,
                spec.fluxden,
                spec.fluxden_err,
                method=method,
                options=options,
            )

        result = _fit()
        if clip is not None:
            clip = dict(clip)
            for _ in range(clip.pop("max_iterations", 5)):
                if not self.sigma_clip(values=result.values, **clip):
                    break
                result = _fit()
        self.quick_fit_result = result
        logger.info(str(result))

        if update_values:
            for name, value in zip(result.param_names, result.values, strict=True):
                self.parameters[name].value = float(value)

        return result

    def positions_around_quick_fit(self, nwalkers, spread=0.1):
        """Initial walker positions in a small Gaussian ball around the quick
        fit, to start an MCMC run from the best fit.

        Each parameter is drawn with a width of `spread` times its quick-fit
        uncertainty (or `spread` times 1% of its value where the uncertainty
        is not finite) and kept inside its uniform prior.

        :param nwalkers: Number of walkers
        :type nwalkers: int
        :param spread: Width of the ball in units of the uncertainties
        :type spread: float
        :return: Positions, shape (nwalkers, number of variable parameters)
        :rtype: np.ndarray
        :raises ValueError: if there is no quick-fit result for the current
            variable parameters
        """

        result = getattr(self, "quick_fit_result", None)
        self.get_params_to_sample()
        if result is None or result.param_names != list(self.params_variable):
            raise ValueError(
                "No quick-fit result for the current variable parameters. "
                "Run run_quick_fit() first."
            )

        width = np.where(
            np.isfinite(result.errors) & (result.errors > 0),
            result.errors,
            0.01 * np.abs(result.values),
        )
        width = np.where(width > 0, width, 1e-8)
        pos = result.values + spread * width * self.rng.standard_normal(
            (nwalkers, result.values.size)
        )

        for idx, param in enumerate(self.params_variable.values()):
            prior = param.prior
            if isinstance(prior, scpri.UniformPrior):
                # Reflect at the bounds, so that walkers stay inside
                # the prior without piling up on the bound.
                low, upp = prior.low, prior.upp
                col = pos[:, idx]
                col = np.where(col < low, 2 * low - col, col)
                col = np.where(col > upp, 2 * upp - col, col)
                pos[:, idx] = np.clip(col, low, upp)

        return pos

    def fit(self, method="quick", **kwargs):
        """Fit the model with the quick fit or MCMC.

        One entry point for scripts and the GUI. `"quick"` calls
        `run_quick_fit(**kwargs)`; `"mcmc"` calls `run_emcee(**kwargs)`
        (`nsteps` and `nwalkers` required). With `"quick+mcmc"` a quick fit
        runs first and the MCMC walkers start around it
        (`positions_around_quick_fit`); pass the quick-fit options as
        `quick_fit_kwargs`.

        :param method: "quick", "mcmc" or "quick+mcmc"
        :type method: str
        :param kwargs: Passed on to `run_quick_fit` or `run_emcee`
        :return: The quick-fit result for "quick", else None
        :rtype: sculptor.quick_fit.QuickFitResult or None
        :raises ValueError: for an unknown method
        """

        if method == "quick":
            return self.run_quick_fit(**kwargs)
        if method == "mcmc":
            self.run_emcee(**kwargs)
            return None
        if method == "quick+mcmc":
            self.run_quick_fit(**kwargs.pop("quick_fit_kwargs", {}))
            if kwargs.get("pos") is not None:
                raise ValueError("quick+mcmc sets `pos`; do not pass it.")
            kwargs["pos"] = self.positions_around_quick_fit(kwargs["nwalkers"])
            self.run_emcee(**kwargs)
            return None
        raise ValueError(
            f"Unknown fit method {method!r}; use 'quick', 'mcmc' or 'quick+mcmc'."
        )

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
        """Deprecated alias of `get_components_by_prefix`.

        :param prefix: Prefix to match component names against
        :type prefix: str
        :return: List of matching FitComponent instances
        :rtype: list
        """
        warnings.warn(
            "get_all_prefix_components is deprecated; use get_components_by_prefix",
            DeprecationWarning,
            stacklevel=2,
        )
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
        """Pixels used in the fit: inside the fit regions (`gpm_fit`), not
        masked in the spectrum (`spec.mask`, True = good pixel) and not
        sigma-clipped (`clip_mask`, see `sigma_clip`).

        Pixels masked in the spectrum (e.g. bad or NaN pixels) are excluded
        from the likelihood, the chi-square and the BIC even if they lie in a
        fit region. Adopted from the upstream `kat` branch (4ee90b4).

        :return: Boolean mask over `spec.dispersion`
        :rtype: np.ndarray
        """

        mask = self._fit_region_mask()
        clip_mask = getattr(self, "clip_mask", None)
        if clip_mask is None:
            return mask
        return mask & clip_mask

    def _fit_region_mask(self):
        """Fit regions minus the spectrum's mask, before sigma clipping."""

        spec_mask = getattr(self.spec, "mask", None)
        if spec_mask is None:
            return self.gpm_fit
        return self.gpm_fit & spec_mask

    def sigma_clip(
        self,
        low=3.0,
        high=None,
        grow=0,
        values=None,
        max_fraction=0.2,
        scale="local",
        window=101,
        high_max_width=None,
    ):
        """Clip outlying pixels from the fit, given a model fit.

        Recomputed from scratch on every call over the fit regions: a pixel
        clipped before comes back if the model now explains it. Pixels with
        (flux - model) / (error * s) below -`low` or above `high` are
        clipped, plus `grow` pixels on each side (e.g. the wings of an
        absorber). The result is stored in `clip_mask` and applied by
        `get_fit_mask`.

        The scale s corrects for errors that do not match the scatter:
        "local" (default) is the robust standard deviation (1.4826 x median
        absolute deviation) of the error-normalized residuals in a running
        window of `window` fitted pixels, "robust" the same over all fitted
        pixels, "errors" 1 (trust the errors). s is never below 1, so
        clipping is never stricter than the errors say.

        Clipping only negative outliers (`high=None`, the default) suits
        emission-line fits: absorbers are removed, while unmodeled emission
        stays in the residuals for the model to explain.

        :param low: Clip below -`low` sigma (None: no lower clipping)
        :type low: float
        :param high: Clip above `high` sigma (None: no upper clipping)
        :type high: float
        :param grow: Also clip this many pixels on each side of a clipped one
        :type grow: int
        :param values: Variable parameter values of the model to compare with
            (default: the last quick fit)
        :type values: np.ndarray
        :param max_fraction: If more than this fraction of the fitted pixels
            would be clipped, log a warning and keep the previous mask (the
            model is probably far off)
        :type max_fraction: float
        :param scale: "local", "robust" or "errors", see above
        :type scale: str
        :param window: Window of the "local" scale, in fitted pixels
        :type window: int
        :param high_max_width: Only clip positive outliers in runs of at
            most this many pixels (hot pixels, cosmic rays), so that the
            core of an emission line the model does not explain yet is not
            clipped; None clips all of them
        :type high_max_width: int
        :return: Whether the clip mask changed
        :rtype: bool
        :raises ValueError: if no spectrum is set, there are no values or
            `scale` is unknown
        """

        self._require_spectrum("sigma-clip")
        if scale not in ("local", "robust", "errors"):
            raise ValueError(
                f"Unknown scale {scale!r}; use 'local', 'robust' or 'errors'."
            )
        if values is None:
            result = getattr(self, "quick_fit_result", None)
            if result is None:
                raise ValueError("No model values; run run_quick_fit() first.")
            values = result.values

        spec = self.spec
        region = self._fit_region_mask()
        with np.errstate(divide="ignore", invalid="ignore"):
            norm = (spec.fluxden - self.eval(spec.dispersion, values)) / (
                spec.fluxden_err
            )
        if scale != "errors" and region.any():
            fitted = norm[region]
            if scale == "robust":
                mad = np.median(np.abs(fitted - np.median(fitted)))
            else:
                local = median_filter(fitted, size=window, mode="nearest")
                mad = median_filter(np.abs(fitted - local), size=window, mode="nearest")
            norm = norm.copy()
            norm[region] = fitted / np.maximum(1.4826 * mad, 1.0)

        bad = np.zeros_like(region)
        if low is not None:
            bad |= norm < -low
        if high is not None:
            high_bad = (norm > high) & region
            if high_max_width is not None:
                high_bad = _short_runs(high_bad, high_max_width)
            bad |= high_bad
        bad &= region
        if grow > 0:
            bad = np.convolve(bad, np.ones(2 * grow + 1), mode="same") > 0
        new = ~bad

        n_region = int(region.sum())
        n_clipped = int((bad & region).sum())
        if n_region and n_clipped > max_fraction * n_region:
            logger.warning(
                f"Sigma clipping would remove {n_clipped} of {n_region} fitted "
                f"pixels (more than {max_fraction:.0%}); keeping the previous "
                "clip mask."
            )
            return False

        old = getattr(self, "clip_mask", None)
        changed = old is None or not np.array_equal(old & region, new & region)
        self.clip_mask = new
        if changed:
            logger.info(f"Sigma clipping: {n_clipped} of {n_region} pixels clipped")
        return changed

    def reset_clip_mask(self):
        """Undo the sigma clipping (all fit-region pixels are fitted again)."""

        self.clip_mask = None

    def reset_fit_mask(self):
        """Reset the fit mask based on the supplied astronomical spectrum.

        :return: None
        """

        self._require_spectrum("reset the fit mask")
        self.gpm_fit = np.zeros_like(self.spec.dispersion, dtype="bool")

    def _range_slice(self, disp_x1, disp_x2):
        """Pixels from the one nearest the lower to the one nearest the upper
        dispersion value, both included."""

        mask_between = np.sort(np.array([disp_x1, disp_x2]))
        lo_index = np.argmin(np.abs(self.spec.dispersion - mask_between[0]))
        up_index = np.argmin(np.abs(self.spec.dispersion - mask_between[1]))
        return slice(lo_index, up_index + 1)

    def add_wavelength_range_to_fit_mask(self, disp_x1, disp_x2):
        """Add a dispersion range to the fit mask.

        The pixels from the one nearest `disp_x1` to the one nearest `disp_x2`
        (both included; the order of the two values does not matter) are
        added to the fit regions.

        :param disp_x1: One end of the range, in the spectrum's dispersion unit
        :type disp_x1: float
        :param disp_x2: The other end of the range
        :type disp_x2: float
        """

        logger.info(f"Manual mask range: {disp_x1} - {disp_x2}")

        self._require_spectrum("add a range to the fit mask")
        if self.gpm_fit is None:
            self.reset_fit_mask()

        self.gpm_fit[self._range_slice(disp_x1, disp_x2)] = True

    def remove_wavelength_range_from_fit_mask(self, disp_x1, disp_x2):
        """Remove a dispersion range from the fit mask.

        The pixels from the one nearest `disp_x1` to the one nearest `disp_x2`
        (both included) are removed from the fit regions.

        :param disp_x1: One end of the range, in the spectrum's dispersion unit
        :type disp_x1: float
        :param disp_x2: The other end of the range
        :type disp_x2: float
        """

        logger.info(f"Manual mask range: {disp_x1} - {disp_x2}")

        self._require_spectrum("remove a range from the fit mask")
        if self.gpm_fit is None:
            self.reset_fit_mask()

        self.gpm_fit[self._range_slice(disp_x1, disp_x2)] = False

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

        Labeled "MAP sample" in the plots: the maximum of the posterior among
        the samples (the log-probability includes the priors), not a maximum
        likelihood fit.

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
        low_model, upp_model = np.percentile(
            model_fluxden, scut.SIGMA_PERCENTILES, axis=0
        )

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
        save_name="fit_result.png",
        save_data=False,
        pool=None,
    ):
        """Plot the spectrum with the posterior model and the residuals.

        Shows the median model with its 1-sigma band, optionally the
        maximum-posterior (MAP) sample, the per-component medians and the fit mask,
        and the reduced chi-square in the legend.

        :param discard: Number of burn-in steps to discard
        :type discard: int
        :param show_fit_mask: Shade the fitted regions
        :type show_fit_mask: bool
        :param show_ml_model: Also plot the maximum-posterior sample and its
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
        :param save_name: File name of the figure; the extension sets the
            format (PNG by default, e.g. ".pdf" for a PDF)
        :type save_name: str
        :param save_data: Also write the plotted models as CSV files
        :type save_data: bool
        :param pool: Pool for parallel evaluation of the posterior models
        :return: Residuals of the median model and the matching dispersion
            (both within the spectrum's mask)
        :rtype: tuple(np.ndarray, np.ndarray)
        """

        self._require_spectrum("plot the MCMC result")

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

        component_fluxes = []
        if show_components:
            logger.info("Evaluating the components in the MCMC chain.")
            for component in self.components:
                comp = component.name
                med_comp, _, _ = self._posterior_model_bands(
                    discard=discard, components=[component], pool=pool
                )
                component_fluxes.append((comp, med_comp))

                if save_data:
                    filename = os.path.join(save_dir, f"mcmc_model_results_{comp}.csv")
                    np.savetxt(
                        filename,
                        np.vstack([self.spec.dispersion, med_comp]).T,
                        delimiter=",",
                        header="dispersion, med_comp",
                    )

        extra = None
        if show_ml_model:
            self.model_fluxden = self.get_mode_model(discard=discard)
            extra = (self.model_fluxden, "MAP sample")

            if save_data:
                filename = os.path.join(save_dir, "mcmc_model_results_ml.csv")
                np.savetxt(
                    filename,
                    np.vstack([self.spec.dispersion, self.model_fluxden]).T,
                    delimiter=",",
                    header="dispersion, model_flux",
                )

        _chi_sq, _dof, reduced_chi_sq = self.calculate_chi_sq()

        return self._plot_fit(
            med_model,
            "Median",
            band=(low_model, upp_model),
            component_fluxes=component_fluxes,
            extra=extra,
            reduced_chi_sq=reduced_chi_sq,
            show_fit_mask=show_fit_mask,
            ylim=ylim,
            xlim=xlim,
            resid_ylim=resid_ylim,
            save=save,
            save_dir=save_dir,
            save_name=save_name,
        )

    def plot_quick_fit_result(
        self,
        show_fit_mask=True,
        ylim=None,
        xlim=None,
        resid_ylim=None,
        save=False,
        save_dir=".",
        show_components=False,
        save_name="quick_fit_result.png",
        save_data=False,
    ):
        """Plot the spectrum with the quick-fit model and the residuals.

        The same figure as `plot_mcmc_result`, for the best fit of the last
        `run_quick_fit` (no uncertainty band).

        :param show_fit_mask: Shade the fitted regions
        :type show_fit_mask: bool
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
        :param show_components: Plot every component at the best fit
        :type show_components: bool
        :param save_name: File name of the figure; the extension sets the
            format (PNG by default, e.g. ".pdf" for a PDF)
        :type save_name: str
        :param save_data: Also write the plotted models as CSV files
        :type save_data: bool
        :return: Residuals of the best-fit model and the matching dispersion
            (both within the spectrum's mask)
        :rtype: tuple(np.ndarray, np.ndarray)
        :raises ValueError: if no spectrum is set or there is no quick-fit
            result
        """

        self._require_spectrum("plot the quick-fit result")
        result = getattr(self, "quick_fit_result", None)
        if result is None:
            raise ValueError("No quick-fit result. Run run_quick_fit() first.")

        dispersion = self.spec.dispersion
        self.model_fluxden = self.eval(dispersion, result.values)

        if save_data:
            np.savetxt(
                os.path.join(save_dir, "quick_fit_model.csv"),
                np.vstack([dispersion, self.model_fluxden]).T,
                delimiter=",",
                header="dispersion, model_flux",
            )

        component_fluxes = []
        if show_components:
            for component in self.components:
                comp_flux = self.eval(dispersion, result.values, components=[component])
                component_fluxes.append((component.name, comp_flux))
                if save_data:
                    np.savetxt(
                        os.path.join(save_dir, f"quick_fit_model_{component.name}.csv"),
                        np.vstack([dispersion, comp_flux]).T,
                        delimiter=",",
                        header="dispersion, model_flux",
                    )

        return self._plot_fit(
            self.model_fluxden,
            "Best-fit",
            component_fluxes=component_fluxes,
            reduced_chi_sq=result.red_chisq,
            show_fit_mask=show_fit_mask,
            ylim=ylim,
            xlim=xlim,
            resid_ylim=resid_ylim,
            save=save,
            save_dir=save_dir,
            save_name=save_name,
        )

    def _plot_fit(
        self,
        model_flux,
        label,
        band=None,
        component_fluxes=(),
        extra=None,
        reduced_chi_sq=None,
        show_fit_mask=True,
        ylim=None,
        xlim=None,
        resid_ylim=None,
        save=False,
        save_dir=".",
        save_name="fit_result.png",
    ):
        """Spectrum, model and residual figure shared by `plot_mcmc_result`
        and `plot_quick_fit_result`.

        :param model_flux: Main model ("<label> model", with residuals)
        :param label: Name of the main model in the legend
        :param band: (lower, upper) model band to shade
        :param component_fluxes: (name, flux) per component to plot
        :param extra: (flux, label) of a second model, dashed, with residuals
        :param reduced_chi_sq: Shown in the legend
        :return: Residuals of the main model and the matching dispersion
        """

        scp.set_presentation_defaults()

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
        # Mark the sigma-clipped pixels
        clipped = self._fit_region_mask() & ~self.get_fit_mask()
        if np.any(clipped):
            ax_main.plot(
                self.spec.dispersion[clipped],
                self.spec.fluxden[clipped],
                "x",
                color=scp.vermillion,
                ms=5,
                label="Clipped",
                zorder=10,
            )

        # Plot the spectrum flux density
        ax_main.step(
            self.spec.dispersion[self.spec.mask],
            self.spec.fluxden[self.spec.mask],
            "k",
            where="mid",
            lw=1.5,
        )

        # Plot the main model
        ax_main.step(
            self.spec.dispersion,
            model_flux,
            color=scp.dblue,
            where="mid",
            label=f"{label} model",
            lw=2,
        )
        # Plot the 1-sigma model region
        if band is not None:
            ax_main.fill_between(
                self.spec.dispersion,
                band[0],
                band[1],
                color=scp.dblue,
                alpha=0.3,
                step="mid",
            )

        n_comp = len(self.components)
        for ldx, (comp, comp_flux) in enumerate(component_fluxes):
            ax_main.step(
                self.spec.dispersion,
                comp_flux,
                where="mid",
                color=cmap(ldx / n_comp),
                lw=1.5,
                label=comp,
            )

        # Plot the second model (e.g. the MAP sample)
        if extra is not None:
            ax_main.step(
                self.spec.dispersion,
                extra[0],
                color=scp.vermillion,
                where="mid",
                label=extra[1],
                ls="--",
                lw=2,
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
        residuals = self.spec.fluxden[self.spec.mask] - model_flux[self.spec.mask]
        ax_resid.step(
            self.spec.dispersion[self.spec.mask],
            residuals,
            color=scp.dblue,
            where="mid",
            lw=1.5,
            label=f"{label} residuals",
            zorder=-100,
        )
        if extra is not None:
            residuals_extra = (
                self.spec.fluxden[self.spec.mask] - extra[0][self.spec.mask]
            )
            ax_resid.step(
                self.spec.dispersion[self.spec.mask],
                residuals_extra,
                color=scp.vermillion,
                where="mid",
                lw=1.5,
                label=extra[1].replace("sample", "residuals"),
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
        handles, labels = ax_main.get_legend_handles_labels()
        if reduced_chi_sq is not None:
            stats_handle = Line2D(
                [0], [0], color="none", label=f"reduced chi-sq = {reduced_chi_sq:.2f}"
            )
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
            plt.savefig(filename, bbox_inches="tight", dpi=150)
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
        :param save_name: File name of the figure; the extension sets the
            format (PNG by default, e.g. ".pdf" for a PDF)
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
        self, discard=2000, save=False, save_dir=".", save_name_prefix="posterior_hist"
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
            lower, upper = np.percentile(samples, scut.SIGMA_PERCENTILES)

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
