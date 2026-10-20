"""Automatic model building: greedy forward / backward selection of
components on the quick fit, scored with an information criterion.

Similar in purpose to AI-VPFIT, without a genetic algorithm. Starting from
a base model (e.g. a power-law continuum), each iteration

1. asks every rule for candidate components (e.g. a Gaussian where the
   residuals show the most significant unmodeled emission),
2. quick-fits the model with each candidate added and scores it with the
   criterion (BIC by default, or AICc, or any callable),
3. keeps the best candidate if it lowers the criterion by more than
   `threshold`, then tries removing each removable component and drops it
   if it no longer earns `threshold`,

and stops when no candidate improves the model. Run an MCMC on the result
(`model.fit("quick+mcmc", ...)`) for the posterior.

Example::

    from sculptor.autofit import AutoFit, GaussianLineRule

    rule = GaussianLineRule("line", redshift=3.227, fwhm_range=(300, 10000))
    result = AutoFit(base_model, [rule]).run()
    print(result)
    result.model.plot_quick_fit_result()

Pixels of resampled spectra are correlated, which makes every criterion
favor too many components; calibrate `threshold` on simulated spectra with
a known number of components.
"""

import copy
import logging
from dataclasses import dataclass, field

import numpy as np

from sculptor import analysis as ana
from sculptor import utils as scut
from sculptor.component import FitComponent
from sculptor.prior import UniformPrior

logger = logging.getLogger(__name__)

_FWHM_TO_SIGMA = 1 / np.sqrt(8 * np.log(2))

CRITERIA = {
    "bic": lambda result: result.bic,
    "aicc": lambda result: result.aicc,
    "aic": lambda result: result.aic,
}


@dataclass
class Candidate:
    """A proposed change to the model: components and parameters to add.

    :param description: Short description for the history / log
    :type description: str
    :param components: Components to add
    :type components: list(sculptor.component.FitComponent)
    :param parameters: Their parameters, name -> FitParameter
    :type parameters: dict
    :param significance: Detection significance used to rank candidates
    :type significance: float
    """

    description: str
    components: list
    parameters: dict
    significance: float = np.nan


class GaussianLineRule:
    """Rule: Gaussian emission lines (`analysis.line_model_gaussian`).

    Candidates are placed where a matched filter on the residuals of the
    current fit finds the most significant unmodeled emission, for a few
    trial widths within `fwhm_range`. With `lines` (rest-frame centers),
    only those lines are considered: the center is fixed and the redshift
    may vary by +-`velocity_range`; without, the center is free (+-1 FWHM
    around the detection) and the redshift fixed at `redshift`.

    :param prefix: Prefix of the component names (`<prefix>_<n>`)
    :type prefix: str
    :param redshift: Systemic redshift
    :type redshift: float
    :param fwhm_range: Allowed FWHM in km/s (uniform prior)
    :type fwhm_range: (float, float)
    :param lines: Rest-frame centers of allowed lines, or None for lines
        anywhere
    :type lines: list(float)
    :param velocity_range: With `lines`: allowed velocity offset, km/s
    :type velocity_range: float
    :param window: Observed-frame range to search in (default: the whole
        fit mask)
    :type window: (float, float)
    :param max_components: Maximum number of components this rule adds
    :type max_components: int
    :param n_trial_fwhm: Number of trial widths (log-spaced in
        `fwhm_range`) for the matched filter
    :type n_trial_fwhm: int
    :param min_significance: Minimum matched-filter significance (sigma)
        of a candidate
    :type min_significance: float
    :param n_candidates: Candidates proposed per iteration
    :type n_candidates: int
    """

    def __init__(
        self,
        prefix,
        redshift,
        fwhm_range=(300.0, 10000.0),
        lines=None,
        velocity_range=3000.0,
        window=None,
        max_components=10,
        n_trial_fwhm=4,
        min_significance=3.0,
        n_candidates=4,
    ):
        self.prefix = prefix
        self.redshift = redshift
        self.fwhm_range = tuple(fwhm_range)
        self.lines = None if lines is None else list(lines)
        self.velocity_range = velocity_range
        self.window = window
        self.max_components = max_components
        self.trial_fwhm = np.geomspace(*self.fwhm_range, n_trial_fwhm + 2)[1:-1]
        self.min_significance = min_significance
        self.n_candidates = n_candidates

    def _own_components(self, model):
        """Names of the components this rule added."""
        return [
            c.name for c in model.components if c.name.startswith(self.prefix + "_")
        ]

    def _new_name(self, model):
        """First free `<prefix>_<n>`."""
        taken = {c.name for c in model.components}
        idx = 0
        while f"{self.prefix}_{idx}" in taken:
            idx += 1
        return f"{self.prefix}_{idx}"

    def significance_map(self, model, result, fwhm_km_s):
        """Matched-filter search for one Gaussian width on the residuals.

        For each pixel j: amplitude A_j = sum(w g (y - m)) / sum(w g^2)
        and significance A_j / sigma_A, with g a Gaussian of `fwhm_km_s`
        centered on j and w = 1/err^2 on the searched pixels (0 elsewhere).

        :param model: The current model
        :type model: sculptor.model.FitModel
        :param result: Its quick fit
        :type result: sculptor.quick_fit.QuickFitResult
        :param fwhm_km_s: Trial FWHM
        :type fwhm_km_s: float
        :return: significance, peak flux density A (both per pixel)
        :rtype: (np.ndarray, np.ndarray)
        """
        spec = model.spec
        x = spec.dispersion
        resid = spec.fluxden - model.eval(x, result.values)
        weight = np.where(self._search_mask(model), 1 / spec.fluxden_err**2, 0.0)

        # Kernel width in pixels, from the median pixel scale (constant for
        # log-linear spectra like SDSS)
        mid = np.median(x)
        pix = np.median(np.diff(x))
        sigma_pix = fwhm_km_s / scut.c_km_s * mid * _FWHM_TO_SIGMA / pix
        half = int(np.ceil(4 * sigma_pix))
        offsets = np.arange(-half, half + 1)
        kernel = np.exp(-0.5 * (offsets / sigma_pix) ** 2)

        num = np.convolve(weight * resid, kernel, mode="same")
        den = np.convolve(weight, kernel**2, mode="same")
        with np.errstate(divide="ignore", invalid="ignore"):
            amp = np.where(den > 0, num / den, 0.0)
            signif = np.where(den > 0, num / np.sqrt(den), 0.0)
        return signif, amp

    def _search_mask(self, model):
        """Fitted pixels inside the window (and near the listed lines)."""
        x = model.spec.dispersion
        mask = model.get_fit_mask().copy()
        if self.window is not None:
            mask &= (x >= min(self.window)) & (x <= max(self.window))
        if self.lines is not None:
            near = np.zeros_like(mask)
            for line in self.lines:
                obs = line * (1 + self.redshift)
                dv = self.velocity_range / scut.c_km_s * obs
                near |= np.abs(x - obs) <= dv
            mask &= near
        return mask

    def propose(self, model, result):
        """Candidate components for the current model.

        :param model: The current model
        :type model: sculptor.model.FitModel
        :param result: Its quick fit
        :type result: sculptor.quick_fit.QuickFitResult
        :return: Up to `n_candidates` candidates, most significant first
        :rtype: list(Candidate)
        """
        if len(self._own_components(model)) >= self.max_components:
            return []

        x = model.spec.dispersion
        search = self._search_mask(model)
        detections = []
        for fwhm in self.trial_fwhm:
            signif, amp = self.significance_map(model, result, fwhm)
            signif = np.where(search, signif, 0.0)
            for idx in _local_maxima(signif, self.min_significance):
                detections.append((signif[idx], x[idx], fwhm, amp[idx]))

        # Most significant first; one candidate per position (within 1 FWHM)
        detections.sort(key=lambda d: -d[0])
        chosen = []
        for det in detections:
            _, obs, fwhm, _ = det
            if all(abs(obs - other[1]) > fwhm / scut.c_km_s * obs for other in chosen):
                chosen.append(det)
            if len(chosen) == self.n_candidates:
                break

        return [self._candidate(model, *det) for det in chosen]

    def _candidate(self, model, signif, obs, fwhm, amp):
        """A Gaussian at observed wavelength `obs` with width `fwhm`."""
        name = self._new_name(model)
        comp = FitComponent(name, ana.line_model_gaussian)
        pars = comp.create_params()
        sigma_obs = fwhm / scut.c_km_s * obs * _FWHM_TO_SIGMA
        flux = max(amp, 0.0) * np.sqrt(2 * np.pi) * sigma_obs

        if self.lines is None:
            rest = obs / (1 + self.redshift)
            half = fwhm / scut.c_km_s * rest
            pars[f"{name}_redsh"].value = self.redshift
            pars[f"{name}_redsh"].vary = False
            pars[f"{name}_cen"].value = rest
            pars[f"{name}_cen"].prior = UniformPrior(
                f"{name}_cen", rest - half, rest + half
            )
            where = f"{rest:.1f} A rest"
        else:
            rest = min(
                self.lines, key=lambda line: abs(line * (1 + self.redshift) - obs)
            )
            dz = self.velocity_range / scut.c_km_s * (1 + self.redshift)
            pars[f"{name}_cen"].value = rest
            pars[f"{name}_cen"].vary = False
            pars[f"{name}_redsh"].value = obs / rest - 1
            pars[f"{name}_redsh"].prior = UniformPrior(
                f"{name}_redsh", self.redshift - dz, self.redshift + dz
            )
            where = f"line {rest:.2f} A"

        pars[f"{name}_flux"].value = flux
        pars[f"{name}_flux"].prior = UniformPrior(f"{name}_flux", 0, 100 * flux)
        pars[f"{name}_fwhm_km_s"].value = fwhm
        pars[f"{name}_fwhm_km_s"].prior = UniformPrior(
            f"{name}_fwhm_km_s", *self.fwhm_range
        )
        return Candidate(
            description=f"add {name} at {where}, FWHM {fwhm:.0f} km/s "
            f"({signif:.1f} sigma)",
            components=[comp],
            parameters=pars,
            significance=float(signif),
        )


def _local_maxima(values, minimum):
    """Indices of local maxima of `values` above `minimum`."""
    inner = (values[1:-1] > values[:-2]) & (values[1:-1] >= values[2:])
    idx = np.flatnonzero(inner) + 1
    return idx[values[idx] > minimum]


@dataclass
class AutoFitStep:
    """One accepted change of the model.

    :param action: "start", "add" or "remove"
    :type action: str
    :param description: What changed
    :type description: str
    :param score: Criterion after the change
    :type score: float
    :param delta: Change of the criterion (negative = better)
    :type delta: float
    :param components: Component names after the change
    :type components: list(str)
    """

    action: str
    description: str
    score: float
    delta: float
    components: list


@dataclass
class AutoFitResult:
    """Result of `AutoFit.run`.

    :param model: The selected model, with its quick fit in
        `model.quick_fit_result` and the best-fit values in its parameters
    :type model: sculptor.model.FitModel
    :param result: Its quick fit
    :type result: sculptor.quick_fit.QuickFitResult
    :param criterion: Name of the criterion
    :type criterion: str
    :param history: Accepted changes, in order
    :type history: list(AutoFitStep)
    :param n_fits: Number of quick fits run
    :type n_fits: int
    """

    model: object
    result: object
    criterion: str
    history: list = field(default_factory=list)
    n_fits: int = 0

    def __str__(self):
        header = (
            f"AutoFit ({self.criterion}): {len(self.model.components)} components "
            f"after {len(self.history) - 1} changes and {self.n_fits} quick fits"
        )
        lines = [header]
        for step in self.history:
            lines.append(
                f"  {step.action:6s} {self.criterion} = {step.score:.2f} "
                f"(delta {step.delta:+.2f}): {step.description}"
            )
        return "\n".join(lines)


class AutoFit:
    """Greedy model building on the quick fit (see the module docstring).

    The base model is copied, not changed.

    :param model: Base model: spectrum, fit mask and the components to start
        with (e.g. the continuum), with starting values and priors
    :type model: sculptor.model.FitModel
    :param rules: Rules that propose components (e.g. `GaussianLineRule`)
    :type rules: list
    :param criterion: "bic" (default), "aicc", "aic", or a callable taking
        a `QuickFitResult` and returning the score (lower is better)
    :type criterion: str or callable
    :param threshold: Minimum improvement of the criterion to add a
        component (and to keep one)
    :type threshold: float
    :param removable: Names of base-model components that may be removed;
        components added by the rules always may
    :type removable: list(str)
    :param quick_fit_kwargs: Passed on to `FitModel.run_quick_fit`
    :type quick_fit_kwargs: dict
    """

    def __init__(
        self,
        model,
        rules,
        criterion="bic",
        threshold=10.0,
        removable=(),
        quick_fit_kwargs=None,
    ):
        if callable(criterion):
            self.criterion_name = getattr(criterion, "__name__", "criterion")
            self.criterion = criterion
        else:
            if criterion not in CRITERIA:
                raise ValueError(
                    f"Unknown criterion {criterion!r}; use one of {list(CRITERIA)} "
                    "or a callable."
                )
            self.criterion_name = criterion
            self.criterion = CRITERIA[criterion]
        self.base = model
        self.rules = list(rules)
        self.threshold = threshold
        self.removable = set(removable)
        self.quick_fit_kwargs = dict(quick_fit_kwargs or {})
        self.n_fits = 0

    def _fit(self, model):
        """Quick fit (best-fit values written into the parameters) and score;
        None if the fit fails."""
        self.n_fits += 1
        try:
            result = model.run_quick_fit(update_values=True, **self.quick_fit_kwargs)
        except (ValueError, np.linalg.LinAlgError) as err:
            logger.debug(f"AutoFit: quick fit failed: {err}")
            return None, np.inf
        score = self.criterion(result)
        return result, score if np.isfinite(score) else np.inf

    def _removable(self, model):
        """Components that may be removed."""
        added = set()
        for rule in self.rules:
            added.update(rule._own_components(model))
        return [c.name for c in model.components if c.name in added | self.removable]

    @staticmethod
    def _with(model, candidate):
        """Copy of `model` with the candidate's components added."""
        new = copy.deepcopy(model)
        new.components.extend(candidate.components)
        new.add_parameters(candidate.parameters)
        return new

    @staticmethod
    def _without(model, name):
        """Copy of `model` without component `name` and the parameters only
        it used."""
        new = copy.deepcopy(model)
        new.components = [c for c in new.components if c.name != name]
        used = {key for c in new.components for key in c.get_arg_keys()}
        new.parameters = {k: p for k, p in new.parameters.items() if k in used}
        return new

    def run(self, max_iterations=20):
        """Build the model.

        :param max_iterations: Maximum number of accepted additions
        :type max_iterations: int
        :return: The selected model, its quick fit and the history
        :rtype: AutoFitResult
        :raises ValueError: if the base model cannot be fitted
        """
        self.n_fits = 0
        logger.info(
            f"AutoFit: {self.criterion_name}, threshold {self.threshold}, "
            f"{len(self.rules)} rule(s)"
        )
        model = copy.deepcopy(self.base)
        result, score = self._fit(model)
        if result is None:
            raise ValueError("AutoFit: the base model cannot be fitted.")
        history = [
            AutoFitStep(
                "start",
                "base model",
                score,
                0.0,
                [c.name for c in model.components],
            )
        ]
        seen = {_signature(model)}

        for _ in range(max_iterations):
            best = None
            for rule in self.rules:
                for cand in rule.propose(model, result):
                    trial = self._with(model, cand)
                    if _signature(trial) in seen:
                        continue
                    trial_result, trial_score = self._fit(trial)
                    logger.debug(
                        f"AutoFit: {cand.description}: "
                        f"{self.criterion_name} {trial_score - score:+.2f}"
                    )
                    if trial_result is not None and (
                        best is None or trial_score < best[2]
                    ):
                        best = (trial, trial_result, trial_score, cand)

            if best is None or score - best[2] <= self.threshold:
                break

            model, result, new_score, cand = best
            history.append(
                AutoFitStep(
                    "add",
                    cand.description,
                    new_score,
                    new_score - score,
                    [c.name for c in model.components],
                )
            )
            logger.info(f"AutoFit: {cand.description}: {new_score - score:+.2f}")
            score = new_score
            seen.add(_signature(model))

            model, result, score = self._backward(model, result, score, history, seen)

        model.quick_fit_result = result
        logger.info(
            f"AutoFit done: {[c.name for c in model.components]}, "
            f"{self.criterion_name} = {score:.2f}"
        )
        return AutoFitResult(model, result, self.criterion_name, history, self.n_fits)

    def _backward(self, model, result, score, history, seen):
        """Remove components that no longer earn `threshold`, one at a time
        (the least useful first)."""
        while True:
            worst = None
            for name in self._removable(model):
                trial = self._without(model, name)
                if not trial.components:
                    continue
                trial_result, trial_score = self._fit(trial)
                if trial_result is None:
                    continue
                # Keeping `name` is worth trial_score - score
                if trial_score - score < self.threshold and (
                    worst is None or trial_score < worst[2]
                ):
                    worst = (trial, trial_result, trial_score, name)
            if worst is None:
                return model, result, score

            model, result, new_score, name = worst
            history.append(
                AutoFitStep(
                    "remove",
                    f"remove {name}",
                    new_score,
                    new_score - score,
                    [c.name for c in model.components],
                )
            )
            logger.info(f"AutoFit: remove {name}: {new_score - score:+.2f}")
            score = new_score
            seen.add(_signature(model))


def _signature(model):
    """Hashable description of a model's structure (component functions and
    rounded fixed values / prior ranges), to avoid revisiting a model."""
    parts = []
    for comp in model.components:
        func = getattr(comp.function, "__name__", str(comp.function))
        keys = []
        for key in comp.get_arg_keys():
            p = model.parameters[key]
            if not p.vary:
                keys.append(("fixed", round(float(p.value), 3)))
            elif isinstance(p.prior, UniformPrior):
                keys.append(("free", round(p.prior.low, 1), round(p.prior.upp, 1)))
            else:
                keys.append(("free",))
        parts.append((func, tuple(keys[:2]), tuple(keys[2:])))
    return tuple(sorted(parts, key=repr))


def correlated_noise(model, result, rng):
    """Noise with the correlations of the quick-fit residuals.

    The residuals over the fitted pixels, divided by the errors, are taken as
    one series; a series with the same power spectrum and random phases is
    drawn, scaled to unit variance and multiplied by the errors. Pixels
    outside the fit mask get white noise.

    :param model: The model
    :type model: sculptor.model.FitModel
    :param result: Its quick fit
    :type result: sculptor.quick_fit.QuickFitResult
    :param rng: Random generator
    :type rng: np.random.Generator
    :return: Noise, one value per pixel of `model.spec`
    :rtype: np.ndarray
    """
    spec = model.spec
    mask = model.get_fit_mask()
    err = spec.fluxden_err
    resid = (spec.fluxden - model.eval(spec.dispersion, result.values))[mask]
    norm = resid / err[mask]
    norm = norm - norm.mean()

    amplitude = np.abs(np.fft.rfft(norm))
    phases = rng.uniform(0, 2 * np.pi, amplitude.size)
    phases[0] = 0.0
    series = np.fft.irfft(amplitude * np.exp(1j * phases), n=norm.size)
    series /= series.std()

    noise = rng.standard_normal(spec.dispersion.size)
    noise[mask] = series
    return noise * err


def calibrate_threshold(
    model,
    rules,
    criterion="bic",
    n_sims=30,
    quantile=0.95,
    noise="residuals",
    seed=0,
    quick_fit_kwargs=None,
):
    """Threshold for `AutoFit` from simulations without extra components.

    Each simulation replaces the flux of a copy of `model` by its best fit
    plus noise ("residuals": with the correlations of the residuals, see
    `correlated_noise`; "white": independent Gaussian noise from the
    errors), quick-fits it, and runs one forward step of the search. The
    largest improvement of the criterion that any candidate achieves is
    noise only; the `quantile` of these improvements is the threshold
    above which an added component is unlikely to fit noise (a false
    positive rate of about 1 - `quantile` per step).

    Use the base model, or the model `AutoFit` selected (residuals that
    still contain real features make the threshold conservative).

    :param model: The null model (spectrum, fit mask, starting values)
    :type model: sculptor.model.FitModel
    :param rules: Rules, as for `AutoFit`
    :type rules: list
    :param criterion: As for `AutoFit`
    :type criterion: str or callable
    :param n_sims: Number of simulations
    :type n_sims: int
    :param quantile: Quantile of the null improvements to return
    :type quantile: float
    :param noise: "residuals" or "white"
    :type noise: str
    :param seed: Seed of the random generator
    :type seed: int
    :param quick_fit_kwargs: Passed on to `FitModel.run_quick_fit`
    :type quick_fit_kwargs: dict
    :return: threshold, and the improvement of every simulation
    :rtype: (float, np.ndarray)
    :raises ValueError: for an unknown `noise`
    """
    if noise not in ("residuals", "white"):
        raise ValueError(f"Unknown noise {noise!r}; use 'residuals' or 'white'.")
    rng = np.random.default_rng(seed)
    search = AutoFit(
        model, rules, criterion=criterion, quick_fit_kwargs=quick_fit_kwargs
    )

    null = copy.deepcopy(model)
    null_result, _ = search._fit(null)
    if null_result is None:
        raise ValueError("calibrate_threshold: the model cannot be fitted.")
    best_flux = null.eval(null.spec.dispersion, null_result.values)

    improvements = np.zeros(n_sims)
    for idx in range(n_sims):
        sim = copy.deepcopy(null)
        if noise == "residuals":
            sim.spec.fluxden = best_flux + correlated_noise(null, null_result, rng)
        else:
            sim.spec.fluxden = best_flux + rng.normal(0, sim.spec.fluxden_err)
        sim_result, sim_score = search._fit(sim)
        if sim_result is None:
            continue
        best = sim_score
        for rule in search.rules:
            for cand in rule.propose(sim, sim_result):
                _, score = search._fit(search._with(sim, cand))
                best = min(best, score)
        improvements[idx] = sim_score - best

    threshold = float(np.quantile(improvements, quantile))
    logger.info(
        f"calibrate_threshold: {search.criterion_name} improvements from noise: "
        f"median {np.median(improvements):.1f}, "
        f"{quantile:.0%} quantile {threshold:.1f} ({n_sims} simulations)"
    )
    return threshold, improvements
