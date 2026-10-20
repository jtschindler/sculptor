"""Quick fit: the maximum of the posterior (MAP) found with an optimizer.

The quick fit maximizes the same log-probability the MCMC samples
(`sculptor.model.log_probability`: the priors plus the Gaussian likelihood
over the fit mask), so a quick fit and an MCMC run of the same model differ
only in how the posterior is explored. Use it for a fast best fit, or as the
starting point of an MCMC run (`FitModel.positions_around_quick_fit`).

`FitModel.run_quick_fit` is the entry point; this module holds the
optimizer and the result.
"""

import logging
from dataclasses import dataclass, field

import numpy as np
from scipy import optimize

from sculptor.prior import GaussianPrior, UniformPrior

logger = logging.getLogger(__name__)

# Relative step (in units of the parameter scale) of the finite differences
# for the Jacobian.
_JACOBIAN_STEP = 1e-6
# A value closer than this (in units of the parameter scale) to a uniform
# prior's bound counts as "at the bound".
_AT_BOUND_TOL = 1e-6


@dataclass
class QuickFitResult:
    """Result of a quick fit (`FitModel.run_quick_fit`).

    The uncertainties come from the curvature of the log-posterior at the
    best fit (Gauss-Newton: the model Jacobian over the fit mask plus the
    curvature of the Gaussian priors). They are meaningful for a roughly
    Gaussian posterior away from the prior bounds; parameters at a uniform
    prior's bound are listed in `at_bound`. Use MCMC for the full posterior.

    :param param_names: Names of the variable parameters, in the order of
        `FitModel.params_variable`
    :type param_names: list(str)
    :param values: Best-fit values
    :type values: np.ndarray
    :param errors: 1-sigma uncertainties (NaN if the curvature is singular)
    :type errors: np.ndarray
    :param covariance: Covariance matrix
    :type covariance: np.ndarray
    :param log_prob: Log-probability (log prior + log-likelihood) at the
        best fit
    :type log_prob: float
    :param log_likelihood: Log-likelihood at the best fit
    :type log_likelihood: float
    :param chi_sq: Chi-square over the fit mask
    :type chi_sq: float
    :param dof: Degrees of freedom (fitted pixels - variable parameters)
    :type dof: int
    :param bic: Bayesian information criterion, k ln(n) - 2 ln(L) at the
        best fit (see also `aic`, `aicc`). The best fit maximizes the
        posterior, so with Gaussian priors L is close to, not exactly, the
        maximum likelihood.
    :type bic: float
    :param success: Whether the optimizer reports convergence
    :type success: bool
    :param message: The optimizer's message
    :type message: str
    :param method: Optimizer (`scipy.optimize.minimize` method)
    :type method: str
    :param nfev: Number of log-probability evaluations
    :type nfev: int
    :param at_bound: Parameters at a uniform prior's bound
    :type at_bound: list(str)
    """

    param_names: list
    values: np.ndarray
    errors: np.ndarray
    covariance: np.ndarray
    log_prob: float
    log_likelihood: float
    chi_sq: float
    dof: int
    bic: float
    success: bool
    message: str
    method: str
    nfev: int
    at_bound: list = field(default_factory=list)

    @property
    def red_chisq(self):
        """Reduced chi-square, chi_sq / dof.

        :rtype: float
        """
        return self.chi_sq / self.dof if self.dof > 0 else np.nan

    @property
    def n_data(self):
        """Number of fitted pixels.

        :rtype: int
        """
        return self.dof + len(self.param_names)

    @property
    def aic(self):
        """Akaike information criterion, 2k - 2 ln(L) at the best fit.

        :rtype: float
        """
        return 2 * len(self.param_names) - 2 * self.log_likelihood

    @property
    def aicc(self):
        """AIC corrected for a finite number of pixels,
        AIC + 2k(k + 1) / (n - k - 1).

        :rtype: float
        """
        k, n = len(self.param_names), self.n_data
        if n - k - 1 <= 0:
            return np.inf
        return self.aic + 2 * k * (k + 1) / (n - k - 1)

    def params_dict(self):
        """Best-fit values and uncertainties by parameter name.

        :return: name -> (value, error)
        :rtype: dict
        """
        return {
            name: (float(value), float(error))
            for name, value, error in zip(
                self.param_names, self.values, self.errors, strict=True
            )
        }

    def __str__(self):
        status = "converged" if self.success else "NOT converged"
        header = (
            f"Quick fit ({self.method}): {status}, "
            f"reduced chi-sq = {self.red_chisq:.3f}, "
            f"log-probability = {self.log_prob:.3f}"
        )
        lines = [header]
        for name, (value, error) in self.params_dict().items():
            flag = "  (at prior bound)" if name in self.at_bound else ""
            lines.append(f"  {name} = {value:.6g} +- {error:.3g}{flag}")
        return "\n".join(lines)


def _bounds_and_scales(params_variable):
    """Box bounds (from uniform priors) and a typical scale per parameter.

    The scale (prior width, Gaussian sigma, else the value's magnitude) is
    used to bring all parameters to order unity for the optimizer and the
    finite differences.
    """

    bounds, scales = [], []
    for param in params_variable.values():
        prior = param.prior
        if isinstance(prior, UniformPrior):
            bounds.append((prior.low, prior.upp))
            scale = prior.upp - prior.low
        elif isinstance(prior, GaussianPrior):
            bounds.append((-np.inf, np.inf))
            scale = prior.sigma
        else:
            bounds.append((-np.inf, np.inf))
            scale = abs(param.value) if param.value else 1.0
        scales.append(scale if np.isfinite(scale) and scale > 0 else 1.0)
    return np.array(bounds, dtype=float), np.array(scales, dtype=float)


def _start_values(params_variable, bounds):
    """Current parameter values, moved inside the uniform-prior bounds."""

    missing = [name for name, p in params_variable.items() if p.value is None]
    if missing:
        raise ValueError(
            f"Variable parameters without a starting value: {missing}. Set "
            "their `value` before the quick fit."
        )
    theta0 = np.array([p.value for p in params_variable.values()], dtype=float)
    clipped = np.clip(theta0, bounds[:, 0], bounds[:, 1])
    for name, old, new in zip(params_variable, theta0, clipped, strict=True):
        if old != new:
            logger.warning(
                f"Starting value of {name} ({old}) is outside its prior; "
                f"starting at {new} instead."
            )
    return clipped


def _covariance(model, theta, scales, x, yerr):
    """Gauss-Newton covariance at `theta`: inverse of J^T J plus the
    curvature of the Gaussian priors (uniform priors add none)."""

    fit_mask = model.get_fit_mask()
    sigma = yerr[fit_mask]

    jac = np.empty((int(fit_mask.sum()), theta.size))
    for idx in range(theta.size):
        step = _JACOBIAN_STEP * scales[idx]
        up, down = theta.copy(), theta.copy()
        up[idx] += step
        down[idx] -= step
        diff = model.eval(x, up)[fit_mask] - model.eval(x, down)[fit_mask]
        jac[:, idx] = diff / (2 * step) / sigma

    curvature = jac.T @ jac
    for idx, param in enumerate(model.params_variable.values()):
        if isinstance(param.prior, GaussianPrior):
            curvature[idx, idx] += 1 / param.prior.sigma**2

    try:
        covariance = np.linalg.inv(curvature)
    except np.linalg.LinAlgError:
        logger.warning(
            "The curvature matrix is singular (a parameter the data do not "
            "constrain?); the quick-fit uncertainties are NaN."
        )
        return np.full_like(curvature, np.nan)
    return covariance


def quick_fit(model, log_probability, x, y, yerr, method="L-BFGS-B", options=None):
    """Maximize `log_probability` over the model's variable parameters.

    Called by `FitModel.run_quick_fit`; see there. The parameters are
    rescaled to order unity (by their prior width or sigma) for the
    optimizer; uniform priors become box bounds.

    :param model: The model; starts from its parameters' current values
    :type model: sculptor.model.FitModel
    :param log_probability: Function to maximize, with the signature of
        `sculptor.model.log_probability`
    :type log_probability: callable
    :param x: Dispersion axis
    :type x: np.ndarray
    :param y: Observed flux density
    :type y: np.ndarray
    :param yerr: Flux density uncertainty
    :type yerr: np.ndarray
    :param method: `scipy.optimize.minimize` method; one that supports
        bounds ("L-BFGS-B", "Powell", "Nelder-Mead", "TNC", "SLSQP", ...)
    :type method: str
    :param options: Passed on to `scipy.optimize.minimize`
    :type options: dict
    :return: The fit result
    :rtype: QuickFitResult
    """

    model.get_params_to_sample()
    params_variable = model.params_variable
    if not params_variable:
        raise ValueError("The model has no variable parameters to fit.")

    bounds, scales = _bounds_and_scales(params_variable)
    theta0 = _start_values(params_variable, bounds)

    def _to_theta(u):
        return theta0 + u * scales

    def _negative_log_prob(u):
        lp = log_probability(_to_theta(u), x, model, y, yerr)
        # A large finite value instead of inf keeps the line searches working.
        return -lp if np.isfinite(lp) else 1e300

    u_bounds = [
        ((low - t0) / s, (upp - t0) / s)
        for (low, upp), t0, s in zip(bounds, theta0, scales, strict=True)
    ]
    u_bounds = [
        (None if not np.isfinite(lo) else lo, None if not np.isfinite(hi) else hi)
        for lo, hi in u_bounds
    ]

    res = optimize.minimize(
        _negative_log_prob,
        np.zeros(theta0.size),
        method=method,
        bounds=u_bounds,
        options=options,
    )
    theta = _to_theta(res.x)
    if not res.success:
        logger.warning(f"Quick fit did not converge: {res.message}")

    fit_mask = model.get_fit_mask()
    n_data = int(fit_mask.sum())
    residuals = (y[fit_mask] - model.eval(x, theta)[fit_mask]) / yerr[fit_mask]
    # Same as `sculptor.model.log_likelihood_chi_square`
    log_like = -0.5 * np.sum(residuals**2 + np.log(yerr[fit_mask] ** 2))

    covariance = _covariance(model, theta, scales, x, yerr)
    errors = np.sqrt(np.clip(np.diag(covariance), 0, None))

    tol = _AT_BOUND_TOL * scales
    at_bound = [
        name
        for name, t, (low, upp), tl in zip(
            params_variable, theta, bounds, tol, strict=True
        )
        if t - low < tl or upp - t < tl
    ]
    if at_bound:
        logger.warning(
            f"Quick fit: {at_bound} at a prior bound; their uncertainties "
            "ignore the bound."
        )

    return QuickFitResult(
        param_names=list(params_variable),
        values=theta,
        errors=errors,
        covariance=covariance,
        log_prob=float(-res.fun),
        log_likelihood=float(log_like),
        chi_sq=float(np.sum(residuals**2)),
        dof=n_data - theta.size,
        bic=float(theta.size * np.log(n_data) - 2 * log_like),
        success=bool(res.success),
        message=str(res.message),
        method=method,
        nfev=int(res.nfev),
        at_bound=at_bound,
    )
