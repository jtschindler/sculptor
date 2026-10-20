"""Quick fit: the maximum of the posterior (MAP) found with an optimizer.

The quick fit maximizes the same log-probability the MCMC samples
(`sculptor.model.log_probability`: the priors plus the Gaussian likelihood
over the fit mask), so a quick fit and an MCMC run of the same model differ
only in how the posterior is explored. Use it for a fast best fit, or as the
starting point of an MCMC run (`FitModel.positions_around_quick_fit`).

`FitModel.run_quick_fit` is the entry point; this module holds the
optimizer and the result.
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from astropy.table import Table
from scipy import optimize

from sculptor import utils as scut
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
    :param fixed_params: Values of the fixed parameters during the fit
    :type fixed_params: dict
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
    fixed_params: dict = field(default_factory=dict)

    # Scalars written to / read from the export files, besides the table
    _META_KEYS = (
        "log_prob",
        "log_likelihood",
        "chi_sq",
        "dof",
        "bic",
        "success",
        "message",
        "method",
        "nfev",
        "at_bound",
        "fixed_params",
    )

    def to_table(self):
        """The result as an astropy Table: one row per variable parameter
        (`name`, `value`, `error`, `at_bound`); the fit statistics, the
        covariance and the fixed parameters in `meta`.

        :rtype: astropy.table.Table
        """
        table = Table(
            {
                "name": list(self.param_names),
                "value": np.asarray(self.values, dtype=float),
                "error": np.asarray(self.errors, dtype=float),
                "at_bound": [name in self.at_bound for name in self.param_names],
            }
        )
        table.meta.update(self._meta())
        return table

    def _meta(self):
        """Plain-Python fit statistics, covariance and derived criteria."""
        meta = {key: getattr(self, key) for key in self._META_KEYS}
        for key in ("log_prob", "log_likelihood", "chi_sq", "bic"):
            meta[key] = float(meta[key])
        meta["dof"], meta["nfev"] = int(self.dof), int(self.nfev)
        meta["success"] = bool(self.success)
        meta["at_bound"] = list(self.at_bound)
        meta["fixed_params"] = {
            k: (float(v) if v is not None else None)
            for k, v in self.fixed_params.items()
        }
        meta["covariance"] = np.asarray(self.covariance, dtype=float).tolist()
        # Derived, written for convenience (recomputed on read)
        meta["red_chisq"] = float(self.red_chisq)
        meta["aic"] = float(self.aic)
        meta["aicc"] = float(self.aicc)
        return meta

    def write(self, path, overwrite=False):
        """Write the result to a light, human-readable file.

        The format follows the extension: ".ecsv" (astropy ECSV: a table of
        name, value, error, at_bound with the statistics in the YAML header)
        or ".json". Read it back with `QuickFitResult.read`. The model
        itself (components, priors, spectrum) is not included; save it with
        `FitModel.save` for that.

        :param path: Output file (.ecsv or .json)
        :type path: str or pathlib.Path
        :param overwrite: Replace an existing file
        :type overwrite: bool
        :raises FileExistsError: if the file exists and `overwrite` is False
        :raises ValueError: for an unknown extension
        """
        path = Path(path)
        if path.exists() and not overwrite:
            raise FileExistsError(f"{path} exists; pass overwrite=True.")
        suffix = path.suffix.lower()
        if suffix == ".ecsv":
            self.to_table().write(path, format="ascii.ecsv", overwrite=True)
        elif suffix == ".json":
            data = {
                "params": {
                    name: {"value": float(v), "error": float(e)}
                    for name, v, e in zip(
                        self.param_names, self.values, self.errors, strict=True
                    )
                },
                **self._meta(),
            }
            # JSON has no NaN / inf; write them as strings
            path.write_text(json.dumps(_json_safe(data), indent=2) + "\n")
        else:
            raise ValueError(f"Unknown extension {suffix!r}; use .ecsv or .json.")

    @classmethod
    def read(cls, path):
        """Read a result written by `write`.

        :param path: An .ecsv or .json file
        :type path: str or pathlib.Path
        :rtype: QuickFitResult
        :raises ValueError: for an unknown extension
        """
        path = Path(path)
        suffix = path.suffix.lower()
        if suffix == ".ecsv":
            table = Table.read(path, format="ascii.ecsv")
            meta = dict(table.meta)
            names = [str(n) for n in table["name"]]
            values = np.asarray(table["value"], dtype=float)
            errors = np.asarray(table["error"], dtype=float)
        elif suffix == ".json":
            meta = _json_restore(json.loads(path.read_text()))
            params = meta.pop("params")
            names = list(params)
            values = np.array([params[n]["value"] for n in names], dtype=float)
            errors = np.array([params[n]["error"] for n in names], dtype=float)
        else:
            raise ValueError(f"Unknown extension {suffix!r}; use .ecsv or .json.")

        return cls(
            param_names=names,
            values=values,
            errors=errors,
            covariance=np.array(meta["covariance"], dtype=float).reshape(
                len(names), len(names)
            ),
            **{key: meta[key] for key in cls._META_KEYS},
        )

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


def _json_safe(obj):
    """Replace NaN and inf (not valid JSON) by strings, recursively."""
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, float) and not np.isfinite(obj):
        return str(obj)
    return obj


def _json_restore(obj):
    """Undo `_json_safe`."""
    if isinstance(obj, dict):
        return {k: _json_restore(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_restore(v) for v in obj]
    if obj in ("nan", "inf", "-inf"):
        return float(obj)
    return obj


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
    # Strictly inside: least-squares trust-region methods do not move a
    # parameter that starts on its bound
    width = bounds[:, 1] - bounds[:, 0]
    inset = np.where(np.isfinite(width), 1e-3 * width, 0.0)
    clipped = np.clip(clipped, bounds[:, 0] + inset, bounds[:, 1] - inset)
    outside = (theta0 < bounds[:, 0]) | (theta0 > bounds[:, 1])
    for name, old, new, out in zip(
        params_variable, theta0, clipped, outside, strict=True
    ):
        if out:
            logger.warning(
                f"Starting value of {name} ({old}) is outside its prior; "
                f"starting at {new} instead."
            )
    return clipped


class _ComponentEvaluator:
    """Evaluates a model component by component, so that the derivative
    with respect to a parameter only re-evaluates the components that use
    it. The model is linear in its components (the resolution broadening
    too), so the sum of the (broadened) components is `FitModel.eval`.
    """

    def __init__(self, model, x):
        self.model = model
        self.x = x
        self.names = list(model.params_variable)
        self.index = {name: idx for idx, name in enumerate(self.names)}
        self.constants = dict(
            zip(model.params_constant, model.params_constant_val, strict=True)
        )
        self.keys = [comp.get_arg_keys() for comp in model.components]
        # Components that use each variable parameter
        self.users = [[] for _ in self.names]
        for c_idx, keys in enumerate(self.keys):
            for key in set(keys):
                if key in self.index:
                    self.users[self.index[key]].append(c_idx)

        resolution = model.resolution
        if isinstance(resolution, (int, float)):
            resolution = np.full_like(x, resolution, dtype=np.float64)
        self.resolution = resolution

    def component(self, c_idx, theta):
        """Flux of component `c_idx` at `theta` (broadened if the model
        has a resolution)."""
        values = [
            theta[self.index[k]] if k in self.index else self.constants[k]
            for k in self.keys[c_idx]
        ]
        flux = np.asarray(
            self.model.components[c_idx].eval(self.x, values), dtype=float
        )
        if self.resolution is not None:
            flux = scut.broaden_spectrum(self.x, flux, self.resolution, fwhm_lim=5.0)
        return flux

    def all(self, theta):
        """Fluxes of all components, shape (n_components, n_pixels)."""
        return np.array([self.component(c, theta) for c in range(len(self.keys))])

    def derivatives(self, theta, comps, steps, mask, central=False):
        """d model / d theta over `mask` by finite differences of the
        components that use each parameter, shape (n_masked, n_params)."""
        jac = np.zeros((int(mask.sum()), theta.size))
        for j, users in enumerate(self.users):
            if not users:
                continue
            up = theta.copy()
            up[j] += steps[j]
            diff = sum(self.component(c, up)[mask] for c in users)
            if central:
                down = theta.copy()
                down[j] -= steps[j]
                diff = diff - sum(self.component(c, down)[mask] for c in users)
                jac[:, j] = diff / (2 * steps[j])
            else:
                diff = diff - sum(comps[c][mask] for c in users)
                jac[:, j] = diff / steps[j]
        return jac


def _log_prior_and_gradient(params_variable, theta):
    """Sum of the log priors and its gradient (uniform: 0 inside; Gaussian:
    analytic; other priors: finite differences)."""
    total, grad = 0.0, np.zeros(theta.size)
    for j, param in enumerate(params_variable.values()):
        prior = param.prior
        if prior is None:
            continue
        value = prior.logprior(theta[j])
        total += float(value)
        if not np.isfinite(total):
            return -np.inf, grad
        if isinstance(prior, UniformPrior):
            continue
        if isinstance(prior, GaussianPrior):
            grad[j] = -(theta[j] - prior.mean) / prior.sigma**2
        else:
            step = 1e-6 * max(abs(theta[j]), 1.0)
            grad[j] = (prior.logprior(theta[j] + step) - value) / step
    return total, grad


def _least_squares(
    evaluator,
    params_variable,
    to_theta,
    scales,
    y_fit,
    err_fit,
    mask,
    steps,
    u_bounds,
    method,
    options,
):
    """MAP by `scipy.optimize.least_squares`: residuals (y - m) / err over
    the fit mask plus (theta - mean) / sigma per Gaussian prior; uniform
    priors are the bounds. -1/2 sum(residuals^2) is the log-probability up
    to a constant."""

    gauss = [
        (j, p.prior)
        for j, p in enumerate(params_variable.values())
        if isinstance(p.prior, GaussianPrior)
    ]
    cache = {}

    def _state(u):
        key = u.tobytes()
        if key not in cache:
            cache.clear()
            theta = to_theta(u)
            comps = evaluator.all(theta)
            cache[key] = (theta, comps)
        return cache[key]

    def _fun(u):
        theta, comps = _state(u)
        resid = (y_fit - comps.sum(axis=0)[mask]) / err_fit
        prior = [(theta[j] - pr.mean) / pr.sigma for j, pr in gauss]
        return np.concatenate([resid, prior])

    def _jac(u):
        theta, comps = _state(u)
        jac = -evaluator.derivatives(theta, comps, steps, mask) / err_fit[:, None]
        rows = np.zeros((len(gauss), theta.size))
        for row, (j, pr) in enumerate(gauss):
            rows[row, j] = 1 / pr.sigma
        return np.vstack([jac, rows]) * scales

    lo = np.array([-np.inf if b[0] is None else b[0] for b in u_bounds])
    hi = np.array([np.inf if b[1] is None else b[1] for b in u_bounds])
    res = optimize.least_squares(
        _fun,
        np.zeros(scales.size),
        jac=_jac,
        bounds=(lo, hi),
        method=method,
        x_scale="jac",
        **(options or {}),
    )
    res.nfev = int(res.nfev) + int(res.njev or 0)
    res.success = bool(res.status > 0)
    return res


def _covariance(model, theta, scales, x, yerr):
    """Gauss-Newton covariance at `theta`: inverse of J^T J plus the
    curvature of the Gaussian priors (uniform priors add none)."""

    fit_mask = model.get_fit_mask()
    sigma = yerr[fit_mask]

    evaluator = _ComponentEvaluator(model, x)
    jac = evaluator.derivatives(
        theta, None, _JACOBIAN_STEP * scales, fit_mask, central=True
    )
    jac /= sigma[:, None]

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


def quick_fit(
    model,
    log_probability,
    x,
    y,
    yerr,
    method="auto",
    options=None,
    gradient=False,
):
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
    :param method: "trf" or "dogbox" (`scipy.optimize.least_squares` on the
        residuals plus one residual per Gaussian prior: the same maximum for
        uniform and Gaussian priors, and much faster; needs `gradient`), a
        `scipy.optimize.minimize` method that supports bounds ("L-BFGS-B",
        "Powell", "Nelder-Mead", "TNC", "SLSQP", ...), or "auto": "trf" if
        possible (`gradient` and only uniform / Gaussian priors), else
        "L-BFGS-B"
    :type method: str
    :param options: Passed on to `scipy.optimize.minimize` (`options`) or
        `scipy.optimize.least_squares` (keyword arguments, e.g. `max_nfev`)
    :type options: dict
    :param gradient: `log_probability` is `sculptor.model.log_probability`:
        compute the log-probability and its gradient here, component by
        component (much faster for models with many components, and more
        accurate than the optimizer's own finite differences)
    :type gradient: bool
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
        # Clipped: at a bound, u * scales can miss it by round-off and put
        # the value just outside a uniform prior (log prior -inf)
        return np.clip(theta0 + u * scales, bounds[:, 0], bounds[:, 1])

    def _negative_log_prob(u):
        lp = log_probability(_to_theta(u), x, model, y, yerr)
        # A large finite value instead of inf keeps the line searches working.
        return -lp if np.isfinite(lp) else 1e300

    fit_mask = model.get_fit_mask()
    evaluator = _ComponentEvaluator(model, x)
    y_fit, inv_var = y[fit_mask], 1 / yerr[fit_mask] ** 2
    log_norm = np.sum(np.log(yerr[fit_mask] ** 2))
    steps = _JACOBIAN_STEP * scales

    def _value_and_gradient(u):
        theta = _to_theta(u)
        lp, lp_grad = _log_prior_and_gradient(params_variable, theta)
        if not np.isfinite(lp):
            return 1e300, np.zeros_like(u)
        comps = evaluator.all(theta)
        resid = y_fit - comps.sum(axis=0)[fit_mask]
        log_like = -0.5 * (np.sum(resid**2 * inv_var) + log_norm)
        jac = evaluator.derivatives(theta, comps, steps, fit_mask)
        grad = jac.T @ (resid * inv_var) + lp_grad
        value = lp + log_like
        if not np.isfinite(value):
            return 1e300, np.zeros_like(u)
        return -value, -grad * scales

    u_bounds = [
        ((low - t0) / s, (upp - t0) / s)
        for (low, upp), t0, s in zip(bounds, theta0, scales, strict=True)
    ]
    u_bounds = [
        (None if not np.isfinite(lo) else lo, None if not np.isfinite(hi) else hi)
        for lo, hi in u_bounds
    ]

    simple_priors = all(
        p.prior is None or isinstance(p.prior, (UniformPrior, GaussianPrior))
        for p in params_variable.values()
    )
    if method == "auto":
        method = "trf" if gradient and simple_priors else "L-BFGS-B"
    if method in ("trf", "dogbox"):
        if not (gradient and simple_priors):
            raise ValueError(
                f"method={method!r} needs the default log-probability and only "
                "uniform or Gaussian priors; use a scipy.optimize.minimize method."
            )
        res = _least_squares(
            evaluator,
            params_variable,
            _to_theta,
            scales,
            y_fit,
            yerr[fit_mask],
            fit_mask,
            steps,
            u_bounds,
            method,
            options,
        )
    elif gradient:
        res = optimize.minimize(
            _value_and_gradient,
            np.zeros(theta0.size),
            jac=True,
            method=method,
            bounds=u_bounds,
            options=options,
        )
    else:
        res = optimize.minimize(
            _negative_log_prob,
            np.zeros(theta0.size),
            method=method,
            bounds=u_bounds,
            options=options,
        )
    theta = _to_theta(res.x)
    log_prob = float(log_probability(theta, x, model, y, yerr))
    if not res.success:
        logger.warning(f"Quick fit did not converge: {res.message}")

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
        log_prob=log_prob,
        log_likelihood=float(log_like),
        chi_sq=float(np.sum(residuals**2)),
        dof=n_data - theta.size,
        bic=float(theta.size * np.log(n_data) - 2 * log_like),
        success=bool(res.success),
        message=str(res.message),
        method=method,
        nfev=int(res.nfev),
        at_bound=at_bound,
        fixed_params=dict(
            zip(model.params_constant, model.params_constant_val, strict=True)
        ),
    )
