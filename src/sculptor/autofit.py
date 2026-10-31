"""Automatic model building: greedy forward / backward selection of
components on the quick fit, scored with an information criterion.

Similar in purpose to AI-VPFIT, without a genetic algorithm. Starting from
a base model (e.g. a power-law continuum), each iteration

1. asks every rule for candidate components (e.g. a Gaussian where the
   residuals show the most significant unmodeled emission),
2. quick-fits the model with each candidate added and scores it with the
   criterion (BIC by default, or AICc, or any callable),
3. keeps the best candidate if it lowers the criterion by at least
   `threshold`, then tries removing each removable component (or group of
   components that share guarded parameters, see below) and drops it if
   it no longer earns `threshold`,

and stops when no candidate improves the model. Every accepted change (an
addition, or at convergence a removal at a bound or a tie, see below) is
followed by the backward steps and, with `clip`, the clipping;
`run(max_iterations=...)` caps the number of changes, removals included.

A candidate can carry alternative starting values (`Candidate.starts`,
e.g. several widths of a new Gaussian, see `GaussianLineRule`): it is
fitted from each and the best fit counts. This matters for degenerate
components, e.g. a second broad Gaussian of a line, whose fit depends on
where it starts.

Several weak lines with the same kinematics, e.g. the narrow cores of
Hbeta and [OIII], may earn the threshold together but not one at a time:
`LineGroupRule` proposes them as one candidate with shared parameters,
starting at the systemic redshift. Once the group is in the model,
`CompanionLineRule` proposes further lines on its kinematics one at a
time (e.g. a narrow Hgamma), each with its flux as the only new
parameter.

Ties (`Tie`, passed among the rules) are hypotheses that components share
parameters, e.g. the narrow cores of Hbeta and [OIII] their redshift and
width. When the search converges, each applicable tie is tested like a
removal: it is kept unless the untied model earns the tie's threshold
(its own calibrated one with `per_rule`, see `calibrate_threshold`, else
`threshold`), and the search resumes from the tied model. `AutoFitResult.ties` reports each
outcome, e.g. a tie the data reject.

A rule may also name parameters that must not end at a prior bound
(`Rule.guarded`, e.g. the shared redshift and width of a `LineGroupRule`,
or a line's width and offset with `GaussianLineRule(guard=...)`; either
bound, or one only, e.g. a broad line's largest width but not its
smallest): a candidate is rejected if one does, and when the search converges the
components of a guarded parameter that has since drifted to a bound are
removed and the search resumes. Components that share guarded parameters
are removed together: a group earns the threshold jointly, so backward
steps test it as a whole, not one member at a time.

The threshold is calibrated by default (`threshold="calibrate"`): a
preliminary search with a fixed threshold gives a model of the real
features; `calibrate_threshold` simulates it with noise like its residuals
and takes quantiles of the criterion improvements that noise alone
achieves (95% and 99% by default); the search is then rerun for each
quantile. One value, `min_threshold` (default `MIN_THRESHOLD` = 10), is
the preliminary search's threshold and the floor of every calibrated
threshold (global and per rule). A change earns a threshold when it
improves the criterion by at least that much (additions, backward steps
and ties alike). With `recalibrate` the calibration is repeated with the model the
first quantile selects as the null (the noise still from the preliminary
model's residuals), and the searches rerun, until that model no longer
changes; if the models cycle instead (A selects B, B selects A), the
calibration of the cycle with the highest threshold is kept. `AutoFitResult` holds the model of the first quantile, the others
in `by_quantile`, and the calibrations.

With `clip`, the fit mask is sigma-clipped (`FitModel.sigma_clip`) after
every accepted change once `clip_after` components were added, and the
current model is refitted and re-scored on the new mask; once more at the
end of the search, so that the mask matches the final model (after late
removals and ties). Candidates are
always compared on the same pixels: criteria from different masks are not
comparable. Run an MCMC on the result
(`model.fit("quick+mcmc", ...)`) for the posterior.

Example::

    from sculptor.autofit import AutoFit, GaussianLineRule

    rule = GaussianLineRule("line", redshift=3.227, fwhm_range=(300, 10000))
    result = AutoFit(base_model, [rule]).run()
    print(result)
    result.model.plot_quick_fit_result()

Pixels of resampled spectra are correlated and models of real spectra are
imperfect, which makes every criterion favor too many components; this is
what the calibration accounts for.
"""

import contextlib
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

# Threshold of the preliminary search and floor of the calibrated
# thresholds: a BIC difference above 10 is "very strong" evidence (Kass &
# Raftery 1995). Where noise rarely improves the criterion (few searched
# pixels, near-white residuals) the calibrated quantile falls to about 0
# and would accept any improvement (round 28, J1504 optical piece: 0.2, a
# narrow Hgamma added at a BIC gain of 1.6)
MIN_THRESHOLD = 10.0

# Loggers of the quick fits run during the search; their per-fit messages
# are silenced (warnings included: candidates at prior bounds are expected)
_FIT_LOGGERS = ("sculptor.quick_fit", "sculptor.model")


@contextlib.contextmanager
def _quiet_fits():
    """Only errors from the per-fit loggers inside the block."""
    loggers = [logging.getLogger(name) for name in _FIT_LOGGERS]
    levels = [lg.level for lg in loggers]
    for lg in loggers:
        lg.setLevel(logging.ERROR)
    try:
        yield
    finally:
        for lg, level in zip(loggers, levels, strict=True):
            lg.setLevel(level)


def spic(result, model, f=0.5):
    """Spectral information criterion (Webb, Lee, Carswell & Milakovic
    2021, MNRAS 501, 2268, eq. 3), adapted to additive components.

    SpIC = chi^2 + sum_a [2 f k_a R_a / (R_a - k_a - 1)
    + (1 - f) k_a ln(R_a)], summed over the components a, where k_a is the
    number of variable parameters of component a (a parameter shared by n
    components counts 1/n for each) and R_a = sum_i |m_a,i| / err_i over
    the fitted pixels: the component's flux in units of the noise, the
    analogue of the paper's normalized absorption depth (eq. 5, 6). R_a is
    bounded below by k_a + 2. Unlike AICc and BIC, a component is penalized
    by the pixels it affects and its strength, not by the size of the fit
    window. f = 1 is AICc-like, f = 0 BIC-like; the paper favors
    f = 1/2 (SpIC_H).

    :param result: Quick fit of `model`
    :type result: sculptor.quick_fit.QuickFitResult
    :param model: The fitted model
    :type model: sculptor.model.FitModel
    :param f: Weight of the AICc-like term
    :type f: float
    :return: SpIC
    :rtype: float
    """
    x = model.spec.dispersion
    mask = model.get_fit_mask()
    err = model.spec.fluxden_err[mask]
    variable = set(result.param_names)

    users = {}
    for comp in model.components:
        for key in set(comp.get_arg_keys()) & variable:
            users[key] = users.get(key, 0) + 1

    penalty = 0.0
    for comp in model.components:
        k = sum(1 / users[key] for key in set(comp.get_arg_keys()) & variable)
        if k == 0:
            continue
        flux = model.eval(x, result.values, components=[comp])[mask]
        r = max(np.sum(np.abs(flux) / err), k + 2)
        penalty += 2 * f * k * r / (r - k - 1) + (1 - f) * k * np.log(r)
    return result.chi_sq + penalty


CRITERIA = {
    "bic": lambda result, model: result.bic,
    "aicc": lambda result, model: result.aicc,
    "aic": lambda result, model: result.aic,
    "spic": spic,
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
    :param starts: Alternative starting values, each a dict parameter name
        -> value applied on top of `parameters`; the candidate is fitted
        from `parameters` and from each of these, and the best fit counts
    :type starts: list(dict)
    :param reject_at_bound: Parameters that must not end at a prior bound:
        the candidate is rejected if one does. Each a name (either bound)
        or (name, side) for one bound only, side "lower" or "upper"
    :type reject_at_bound: list(str or tuple(str, str))
    """

    description: str
    components: list
    parameters: dict
    significance: float = np.nan
    starts: list = field(default_factory=list)
    reject_at_bound: list = field(default_factory=list)


BOUND_SIDES = ("lower", "upper")


def _guard_entries(guarded):
    """(name, side) of each guarded entry: a name (side None, either
    bound) or (name, side) with side in `BOUND_SIDES`."""
    return [
        (entry, None) if isinstance(entry, str) else tuple(entry) for entry in guarded
    ]


def _at_guarded_bound(guarded, result, parameters):
    """Names of the `guarded` entries (see `_guard_entries`) at their
    guarded prior bound in `result`; `parameters` (name -> parameter) give
    the priors of the entries with a side."""
    at_bound = set(result.at_bound)
    hits = set()
    for name, side in _guard_entries(guarded):
        if name not in at_bound:
            continue
        if side is not None:
            prior = parameters[name].prior
            value = result.values[result.param_names.index(name)]
            upper = abs(prior.upp - value) < abs(value - prior.low)
            if side != BOUND_SIDES[int(upper)]:
                continue
        hits.add(name)
    return hits


def _at_bound(candidate, result):
    """The candidate's `reject_at_bound` parameters at their guarded prior
    bound in `result`."""
    return _at_guarded_bound(candidate.reject_at_bound, result, candidate.parameters)


def matched_filter(x, resid, weight, fwhm_km_s):
    """Matched filter of a Gaussian of FWHM `fwhm_km_s` on residuals:
    significance and peak amplitude of the best-fitting Gaussian centered
    on each pixel (weighted least squares, the weights 1 / error^2 and 0
    for pixels to ignore).

    The kernel width in pixels follows the local pixel scale in velocity
    (and `fwhm_km_s`, if it is given per pixel): pixels whose kernel widths
    agree within ~5% share one kernel, e.g. the sections of a coadd of
    several gratings.

    :param x: Dispersion, increasing
    :type x: np.ndarray
    :param resid: Residuals (flip the sign to look for absorption)
    :type resid: np.ndarray
    :param weight: Weights
    :type weight: np.ndarray
    :param fwhm_km_s: FWHM, one value or one per pixel
    :type fwhm_km_s: float or np.ndarray
    :return: significance, peak amplitude (both per pixel)
    :rtype: (np.ndarray, np.ndarray)
    """
    fwhm_pix = (
        np.broadcast_to(fwhm_km_s, x.shape) / scut.c_km_s / np.gradient(np.log(x))
    )
    group = np.round(np.log10(fwhm_pix) / 0.02)
    # Ignored pixels may hold NaN (masked bad pixels): 0 * NaN would spread
    # NaN over the kernel
    with np.errstate(invalid="ignore"):
        weighted = np.where(weight > 0, weight * resid, 0.0)
    num = np.zeros_like(resid, dtype=float)
    den = np.zeros_like(resid, dtype=float)
    for value in np.unique(group):
        sel = group == value
        sigma_pix = np.median(fwhm_pix[sel]) * _FWHM_TO_SIGMA
        half = int(np.ceil(4 * sigma_pix))
        offsets = np.arange(-half, half + 1)
        kernel = np.exp(-0.5 * (offsets / sigma_pix) ** 2)
        # The central part of the full convolution ("same" returns the
        # kernel's length when the kernel is the longer)
        centre = slice(half, half + x.size)
        num[sel] = np.convolve(weighted, kernel)[centre][sel]
        den[sel] = np.convolve(weight, kernel**2)[centre][sel]
    with np.errstate(divide="ignore", invalid="ignore"):
        amp = np.where(den > 0, num / den, 0.0)
        signif = np.where(den > 0, num / np.sqrt(den), 0.0)
    return signif, amp


class Rule:
    """Base class of the rules: something that proposes changes.

    Subclasses implement `propose` and `owned`.
    """

    def propose(self, model, result):
        """Candidate changes for the current model and its quick fit.

        :param model: The current model
        :type model: sculptor.model.FitModel
        :param result: Its quick fit
        :type result: sculptor.quick_fit.QuickFitResult
        :return: Candidates
        :rtype: list(Candidate)
        """
        raise NotImplementedError

    def owned(self, model):
        """Names of the components of `model` this rule added (AutoFit may
        remove them again).

        :param model: The model
        :type model: sculptor.model.FitModel
        :rtype: list(str)
        """
        raise NotImplementedError

    def guarded(self, model):
        """Parameters of `model` that must not end at a prior bound, with
        the components that use them: AutoFit removes those components
        when, at convergence, one of the parameters is at a bound, and
        backward steps remove them together, never one alone (except those
        some rule names in `removable_alone`).

        :param model: The model
        :type model: sculptor.model.FitModel
        :return: (parameters, component names) pairs; a parameter is a
            name (either bound) or (name, side) for one bound only, side
            "lower" or "upper"
        :rtype: list(tuple(list, list(str)))
        """
        return []

    def removable_alone(self, model):
        """Components of `model` that backward steps may also remove on
        their own, against this rule's threshold, although they share
        guarded parameters (e.g. `CompanionLineRule`'s lines on a group's
        kinematics: they go with the group, but the group does not need
        them).

        :param model: The model
        :type model: sculptor.model.FitModel
        :rtype: list(str)
        """
        return []

    def null_removals(self, model):
        """Components of `model` that represent this rule's hypothesis:
        `calibrate_threshold` calibrates the rule on simulations of `model`
        without them (else it measures adding the hypothesis twice). Default:
        the guarded components.

        :param model: The model
        :type model: sculptor.model.FitModel
        :rtype: list(str)
        """
        return [name for _, comps in self.guarded(model) for name in comps]


class GaussianLineRule(Rule):
    """Rule: Gaussian emission lines (`analysis.line_model_gaussian`).
    Absorbers are handled by `sculptor.absorbers` (masked, not modeled).

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
    :param function: Line profile (default `analysis.line_model_gaussian`)
        with the arguments `redsh`, `flux`, `fwhm_km_s` and optionally `cen`
        (fixed at the line in `lines`; a function without `cen`, e.g. a
        doublet, needs `lines`) plus any in `fixed`; linear in `flux`
    :type function: callable
    :param fixed: Fixed values of further arguments, e.g.
        `{"fluxratio": 2.98}` for the [OIII] doublet
    :type fixed: dict
    :param start_fwhm_factors: Extra starting widths of a candidate,
        relative to the detection width (clipped to `fwhm_range`), with the
        flux re-estimated for each; the best fit counts. () for one start.
    :type start_fwhm_factors: tuple(float)
    :param guard: Arguments (e.g. `("redsh", "fwhm_km_s")`) whose varying
        parameters must not end at a prior bound (`Rule.guarded`): a
        candidate is rejected if one does, and a component whose parameter
        drifts to a bound later is removed. An argument guards both bounds,
        (argument, side) one only, side "lower" or "upper" (e.g.
        `("fwhm_km_s", "upper")`). A line at its widest width or largest
        offset is often a filler of continuum misfit, not the line.
        () (default) for none
    :type guard: tuple(str or tuple(str, str))
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
        function=None,
        fixed=None,
        start_fwhm_factors=(1 / 3, 3.0),
        guard=(),
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
        self.function = ana.line_model_gaussian if function is None else function
        self.fixed = dict(fixed or {})
        self.start_fwhm_factors = tuple(start_fwhm_factors)
        self.guard = tuple(guard)
        args = FitComponent("check", self.function).function_args
        entries = _guard_entries(self.guard)
        unknown = {arg for arg, _ in entries} - set(args)
        if unknown:
            raise ValueError(f"`guard` names unknown arguments {sorted(unknown)}")
        sides = {side for _, side in entries} - {None, *BOUND_SIDES}
        if sides:
            raise ValueError(
                f"`guard` sides must be in {BOUND_SIDES}, not {sorted(sides)}"
            )
        missing = {"redsh", "flux", "fwhm_km_s"} - set(args)
        if missing:
            raise ValueError(f"`function` lacks the arguments {sorted(missing)}")
        if "cen" not in args and self.lines is None:
            raise ValueError("A `function` without `cen` needs `lines`.")

    def owned(self, model):
        """Names of the components this rule added (`<prefix>_<n>`).

        :param model: The model
        :type model: sculptor.model.FitModel
        :rtype: list(str)
        """
        head = self.prefix + "_"
        return [
            c.name
            for c in model.components
            if c.name.startswith(head) and c.name[len(head) :].isdigit()
        ]

    def guarded(self, model):
        """Each of the rule's components with its `guard` parameters that
        vary.

        :param model: The model
        :type model: sculptor.model.FitModel
        :rtype: list(tuple(list(str), list(str)))
        """
        if not self.guard:
            return []
        names = set(self.owned(model))
        pairs = []
        for comp in model.components:
            if comp.name not in names:
                continue
            params = self._guarded_params(comp, model.parameters)
            if params:
                pairs.append((params, [comp.name]))
        return pairs

    def _guarded_params(self, comp, parameters):
        """The varying parameters of `comp` for the `guard` arguments (with
        their side, see `Rule.guarded`), its own only (shared ones, e.g. a
        narrow group's, are guarded by the rule they belong to)."""
        params = []
        for arg, side in _guard_entries(self.guard):
            key = comp.param_mapping[arg]
            if (
                key.startswith(f"{comp.name}_")
                and key in parameters
                and parameters[key].vary
            ):
                params.append(key if side is None else (key, side))
        return params

    def null_removals(self, model):
        """None: the rule is calibrated by adding one more line to the null
        (its guarded components are single lines, not one hypothesis).

        :rtype: list(str)
        """
        return []

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
        return matched_filter(x, resid, weight, fwhm_km_s)

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
        if len(self.owned(model)) >= self.max_components:
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

        # With `lines`, the detections near one line are starts of one
        # candidate, not separate candidates
        groups = {}
        for det in chosen:
            key = det[1] if self.lines is None else self._nearest_line(det[1])
            groups.setdefault(key, []).append(det)
        candidates = [
            self._candidate(model, result, dets[0][0], dets[0][1], dets[0][2], dets[1:])
            for dets in groups.values()
        ]
        return [cand for cand in candidates if cand is not None]

    def _nearest_line(self, obs):
        """The line in `lines` closest to observed wavelength `obs`."""
        return min(self.lines, key=lambda line: abs(line * (1 + self.redshift) - obs))

    def _candidate(self, model, result, signif, obs, fwhm, others=()):
        """A line at observed wavelength `obs` with width `fwhm` (and
        further detections `others` of the same line as alternative
        starts), or None if the residuals do not ask for it (least-squares
        flux of the wrong sign)."""
        name = self._new_name(model)
        comp = FitComponent(name, self.function)
        pars = comp.create_params()
        args = comp.function_args

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
            rest = self._nearest_line(obs)
            dz = self.velocity_range / scut.c_km_s * (1 + self.redshift)
            if "cen" in args:
                pars[f"{name}_cen"].value = rest
                pars[f"{name}_cen"].vary = False
            pars[f"{name}_redsh"].value = obs / rest - 1
            pars[f"{name}_redsh"].prior = UniformPrior(
                f"{name}_redsh", self.redshift - dz, self.redshift + dz
            )
            where = f"line {rest:.2f} A"

        pars[f"{name}_fwhm_km_s"].value = fwhm
        pars[f"{name}_fwhm_km_s"].prior = UniformPrior(
            f"{name}_fwhm_km_s", *self.fwhm_range
        )
        for arg, value in self.fixed.items():
            pars[f"{name}_{arg}"].value = value
            pars[f"{name}_{arg}"].vary = False

        flux_key = f"{name}_flux"
        pars[flux_key].value = 1.0
        amp = _linear_amplitude(model, result, comp, pars, flux_key)
        if amp is None or amp <= 0:
            return None
        pars[flux_key].value = amp
        pars[flux_key].prior = UniformPrior(flux_key, 0.0, 100 * amp)
        return Candidate(
            description=f"add emission {name} at {where}, FWHM {fwhm:.0f} km/s "
            f"({signif:.1f} sigma)",
            components=[comp],
            parameters=pars,
            significance=float(signif),
            starts=self._starts(model, result, comp, pars, rest, obs, fwhm, others),
            reject_at_bound=self._guarded_params(comp, pars),
        )

    def _starts(self, model, result, comp, pars, rest, obs, fwhm, others):
        """Alternative starts: the other detections (`others`, with
        `lines`: their redshift and width) and `start_fwhm_factors` times
        the width, each with its least-squares flux; near-duplicate starts
        are dropped."""
        keys = {arg: comp.param_mapping[arg] for arg in ("redsh", "fwhm_km_s", "flux")}
        prior = pars[keys["flux"]].prior
        variants = [(o, w) for _, o, w, _ in others]
        variants += [(obs, factor * fwhm) for factor in self.start_fwhm_factors]

        starts, tried = [], [(obs, fwhm)]
        for at, width in variants:
            width = float(np.clip(width, *self.fwhm_range))
            start = {keys["fwhm_km_s"]: width}
            if self.lines is not None:
                start[keys["redsh"]] = float(at / rest - 1)
            if any(
                abs(np.log(width / w)) < 0.1
                and abs(at - o) / o * scut.c_km_s < width / 4
                for o, w in tried
            ):
                continue
            trial = copy.deepcopy(pars)
            for key, value in start.items():
                trial[key].value = value
            trial[keys["flux"]].value = 1.0
            amp = _linear_amplitude(model, result, comp, trial, keys["flux"])
            if amp is None or amp <= 0:
                continue
            tried.append((at, width))
            start[keys["flux"]] = float(np.clip(amp, prior.low, prior.upp))
            starts.append(start)
        return starts


def _linear_amplitude(model, result, comp, params, key):
    """Least-squares amplitude of a new component on the residuals of the
    current fit, for a component linear in parameter `key` (evaluated with
    `key` = 1); None if it has no flux in the fit."""
    amps = _linear_amplitudes(model, result, [comp], params, [key])
    return None if amps is None else float(amps[0])


def _linear_amplitudes(model, result, comps, params, keys):
    """Joint least-squares amplitudes of new components on the residuals of
    the current fit, for components linear (and additive) in the
    parameters `keys`, each evaluated with its key = 1 and the others 0;
    None if one has no flux in the fit."""
    mask = model.get_fit_mask()
    x = model.spec.dispersion
    lookup = {name: p.value for name, p in model.parameters.items()}
    lookup.update(dict(zip(result.param_names, result.values, strict=True)))
    lookup.update({name: p.value for name, p in params.items()})
    # Broadened by the resolution, like the model the residuals come from
    matrix, x_eval = None, x
    if model.resolution is not None:
        matrix, x_eval = model.resolution_matrix(x)
    units = []
    for key in keys:
        trial = {**lookup, **dict.fromkeys(keys, 0.0), key: 1.0}
        unit = sum(
            comp.eval(x_eval, [trial[k] for k in comp.get_arg_keys()]) for comp in comps
        )
        if matrix is not None:
            unit = matrix @ unit
        units.append(unit[mask])
    design = np.array(units).T
    resid = (model.spec.fluxden - model.eval(x, result.values))[mask]
    weight = 1 / model.spec.fluxden_err[mask] ** 2
    gram = design.T @ (weight[:, None] * design)
    if not np.all(np.diag(gram) > 0):
        return None
    try:
        return np.linalg.solve(gram, design.T @ (weight * resid))
    except np.linalg.LinAlgError:
        return None


def _scaled_fluxes(amps, signs, prior_factor):
    """Starting values and uniform prior bounds of the linear parameters of
    a new group of components from their least-squares amplitudes (`signs`:
    +1 emission, -1 absorption): those of the wrong sign start at 1% of the
    strongest. None if none has the right sign."""
    signed = np.asarray(amps) * np.asarray(signs)
    top = np.max(signed)
    if not top > 0:
        return None
    values = np.maximum(signed, 0.01 * top) * signs
    bounds = [tuple(sorted([0.0, prior_factor * top * sign])) for sign in signs]
    return [float(v) for v in values], bounds


class ComponentRule(Rule):
    """Rule: add a predefined component (or several), e.g. an iron
    template, a Balmer continuum or a line complex.

    `factory(model)` builds the components and parameters for the current
    model and returns `(component, parameters)`, `(components,
    parameters)`, or None when there is nothing to propose. The rule
    proposes the result while none of its components is in the model and it
    was used fewer than `max_uses` times, so a factory that names its
    components `<line>_broad_a`, `<line>_broad_b`, ... can add one more each
    time.

    With `linear_param` (e.g. "amp" or "flux"), the starting value of that
    parameter of the (single) new component is the least-squares amplitude
    on the current residuals, and the uniform prior's upper bound is
    `prior_factor` times it; a non-positive amplitude means the data do not
    ask for the component, and nothing is proposed. A list (e.g. the fluxes
    of a line complex) is estimated jointly; the component is proposed if
    any is positive, and the others start at 1% of the largest.

    :param name: Name of the rule (for the log)
    :type name: str
    :param factory: Builds the components, see above
    :type factory: callable
    :param max_uses: How many components the rule may add (one per use)
    :type max_uses: int
    :param linear_param: Parameter (without the component prefix) that
        scales the new component linearly, or several
    :type linear_param: str or list(str)
    :param prior_factor: Upper prior bound in units of the estimated
        amplitude, with `linear_param`
    :type prior_factor: float
    :param starts: Alternative starting values, each a dict parameter name
        -> value (see `Candidate.starts`); with `linear_param`, its value is
        re-estimated for each
    :type starts: list(dict)
    :param reject_at_bound: Parameter names (see
        `Candidate.reject_at_bound`): the candidate is rejected if one ends
        at a prior bound
    :type reject_at_bound: list(str)
    """

    def __init__(
        self,
        name,
        factory,
        max_uses=1,
        linear_param=None,
        prior_factor=100.0,
        starts=(),
        reject_at_bound=(),
    ):
        self.name = name
        self.factory = factory
        self.max_uses = max_uses
        self.linear_params = (
            []
            if linear_param is None
            else [linear_param]
            if isinstance(linear_param, str)
            else list(linear_param)
        )
        self.prior_factor = prior_factor
        self.starts = [dict(start) for start in starts]
        self.reject_at_bound = list(reject_at_bound)
        self._produced = set()

    def owned(self, model):
        """Names of the components of `model` this rule added.

        :param model: The model
        :type model: sculptor.model.FitModel
        :rtype: list(str)
        """
        return [c.name for c in model.components if c.name in self._produced]

    def null_removals(self, model):
        """The rule's components in `model` (its hypothesis: while they are
        there it proposes nothing, so it can only be calibrated without
        them), and the components that use their parameters (e.g. a second
        CIII] Gaussian on the complex's redshift).

        :param model: The model
        :type model: sculptor.model.FitModel
        :rtype: list(str)
        """
        names = self.owned(model)
        own = {
            key
            for c in model.components
            if c.name in names
            for key in c.get_arg_keys()
            if key.startswith(f"{c.name}_")
        }
        return names + [
            c.name
            for c in model.components
            if c.name not in names and own & set(c.get_arg_keys())
        ]

    def propose(self, model, result):
        """The factory's components as one candidate (see the class).

        :param model: The current model
        :type model: sculptor.model.FitModel
        :param result: Its quick fit
        :type result: sculptor.quick_fit.QuickFitResult
        :return: Zero or one candidate
        :rtype: list(Candidate)
        """
        if len(self.owned(model)) >= self.max_uses:
            return []
        built = self.factory(model)
        if built is None:
            return []
        comps, params = built
        if not isinstance(comps, (list, tuple)):
            comps = [comps]
        names = {c.name for c in model.components}
        if any(c.name in names for c in comps):
            return []
        self._produced.update(c.name for c in comps)

        description = f"{self.name}: add {', '.join(c.name for c in comps)}"
        if self.linear_params and not self._scale(model, result, comps, params):
            return []
        return [
            Candidate(
                description=description,
                components=list(comps),
                parameters=dict(params),
                starts=self._starts(model, result, comps, params),
                reject_at_bound=list(self.reject_at_bound),
            )
        ]

    def _starts(self, model, result, comps, params):
        """`starts`, with `linear_param` re-estimated for each; starts the
        residuals do not ask for are dropped."""
        if not self.linear_params:
            return [dict(start) for start in self.starts]
        keys = [comps[0].param_mapping[p] for p in self.linear_params]
        starts = []
        for start in self.starts:
            trial = copy.deepcopy(params)
            for name, value in start.items():
                trial[name].value = value
            amps = _linear_amplitudes(model, result, comps[:1], trial, keys)
            scaled = (
                None if amps is None else _scaled_fluxes(amps, [1.0] * len(keys), 1)
            )
            if scaled is None:
                continue
            start = dict(start)
            for key, value in zip(keys, scaled[0], strict=True):
                prior = params[key].prior
                start[key] = float(np.clip(value, prior.low, prior.upp))
            starts.append(start)
        return starts

    def _scale(self, model, result, comps, params):
        """Least-squares starting values for `linear_param`; False if the
        residuals do not ask for the component."""
        if len(comps) != 1:
            raise ValueError("linear_param needs a factory with one component")
        keys = [comps[0].param_mapping[p] for p in self.linear_params]
        amps = _linear_amplitudes(model, result, comps, params, keys)
        scaled = (
            None
            if amps is None
            else _scaled_fluxes(amps, [1.0] * len(keys), self.prior_factor)
        )
        if scaled is None:
            return False
        for key, value, bounds in zip(keys, *scaled, strict=True):
            params[key].value = value
            params[key].prior = UniformPrior(key, *bounds)
        return True


class LineGroupRule(Rule):
    """Rule: one line component of each of several `GaussianLineRule`s at
    once, sharing parameters, e.g. the narrow cores of Hbeta and [OIII]
    with one redshift and width.

    Lines too weak to earn the threshold one at a time may earn it
    together, and the shared kinematics keep e.g. a narrow Hbeta from
    fitting the peak of a broad profile. The candidate starts at
    `redshift`, trusted unless the data say otherwise: its prior allows
    +-`velocity_range`, and the strongest narrow residual peak near the
    members' lines is a further start. It is fitted from each width in
    `start_fwhm`, with the fluxes from a joint least-squares fit to the
    residuals.

    With `after_members` (default), the group is proposed only once every
    member rule has a component (e.g. the broad Hbeta and the [OIII]
    doublet): it then tests narrow cores on top of the lines' fits instead
    of standing in for the lines while they are unmodeled. Members whose
    line the fit does not cover are left out (at least two must remain). A
    list of member
    prefixes waits for those members only (e.g. the broad Hbeta, not an
    [OIII] that is narrow only and may be too weak to be added alone). It is proposed
    while no component uses its shared parameters (`<name>_<arg>`) and
    every member rule has room. The shared parameters are guarded
    (`guarded`): the group is rejected if one ends at a prior bound (a
    width at a bound of `fwhm_range` is not narrow, a redshift at
    +-`velocity_range` not systemic), and removed if one drifts there
    later. The components are named and owned by the member rules, so
    backward steps may remove them one at a time. Give a `Tie` of the same
    lines the same name (e.g. "narrow"): it then counts the group as tied.

    :param name: Name of the group and prefix of the shared parameters
    :type name: str
    :param members: Rules of the lines, each with one line in `lines`
    :type members: list(GaussianLineRule)
    :param redshift: Starting redshift (e.g. the systemic one)
    :type redshift: float
    :param fwhm_range: FWHM prior range, km/s
    :type fwhm_range: (float, float)
    :param velocity_range: Allowed velocity offset from `redshift`, km/s
    :type velocity_range: float
    :param start_fwhm: Starting widths, km/s (default: three log-spaced in
        `fwhm_range`)
    :type start_fwhm: list(float)
    :param args: Shared arguments
    :type args: list(str)
    :param prior_factor: Upper flux prior bound in units of the largest
        estimated flux
    :type prior_factor: float
    :param after_members: Propose the group only once every member rule
        has a component (True), the members with these prefixes (a list),
        or at once (False)
    :type after_members: bool or list(str)
    """

    def __init__(
        self,
        name,
        members,
        redshift,
        fwhm_range=(100.0, 1200.0),
        velocity_range=1000.0,
        start_fwhm=None,
        args=("redsh", "fwhm_km_s"),
        prior_factor=100.0,
        after_members=True,
    ):
        for member in members:
            if member.lines is None or len(member.lines) != 1:
                raise ValueError(f"Member {member.prefix!r} needs exactly one line.")
        if len({member.prefix for member in members}) != len(members):
            raise ValueError("Members need distinct prefixes.")
        self.name = name
        self.members = list(members)
        self.redshift = redshift
        self.fwhm_range = tuple(fwhm_range)
        self.velocity_range = velocity_range
        self.start_fwhm = (
            list(np.geomspace(*self.fwhm_range, 5)[1:-1])
            if start_fwhm is None
            else list(start_fwhm)
        )
        self.args = list(args)
        self.prior_factor = prior_factor
        self.after_members = after_members

    def owned(self, model):
        """The member rules own the components.

        :rtype: list
        """
        return []

    def shared_name(self, arg):
        """Name of the shared parameter of `arg`."""
        return f"{self.name}_{arg}"

    def guarded(self, model):
        """The shared parameters, with the components that use them.

        :param model: The model
        :type model: sculptor.model.FitModel
        :rtype: list(tuple(list(str), list(str)))
        """
        shared = [self.shared_name(arg) for arg in self.args]
        comps = [
            c.name
            for c in model.components
            if set(c.param_mapping.values()) & set(shared)
        ]
        return [(shared, comps)] if comps else []

    def null_removals(self, model):
        """The group's components and the members' components narrower
        than the group's largest width (e.g. a narrow Hbeta added on its own
        by the Hbeta rule): the same hypothesis. A component at that width
        is not counted: a broad-only member's range may start there.

        :param model: The model
        :type model: sculptor.model.FitModel
        :rtype: list(str)
        """
        names = super().null_removals(model)
        for member in self.members:
            for name in member.owned(model):
                comp = next(c for c in model.components if c.name == name)
                key = comp.param_mapping.get("fwhm_km_s", f"{name}_fwhm_km_s")
                par = model.parameters.get(key)
                if (
                    par is not None
                    and par.value < self.fwhm_range[1]
                    and name not in names
                ):
                    names.append(name)
        return names

    def propose(self, model, result):
        """The group as one candidate (see the class).

        :param model: The current model
        :type model: sculptor.model.FitModel
        :param result: Its quick fit
        :type result: sculptor.quick_fit.QuickFitResult
        :return: Zero or one candidate
        :rtype: list(Candidate)
        """
        if any(self.shared_name(arg) in model.parameters for arg in self.args):
            return []
        # Members whose line the fit covers (e.g. not Halpha at z ~ 7)
        members = [m for m in self.members if m._search_mask(model).any()]
        if len(members) < 2:
            return []
        owned = [len(m.owned(model)) for m in members]
        if any(n >= m.max_components for n, m in zip(owned, members, strict=True)):
            return []
        if self.after_members is True:
            waits = owned
        else:
            waits = [
                n
                for n, m in zip(owned, members, strict=True)
                if m.prefix in (self.after_members or ())
            ]
        if not all(waits):
            return []

        dz = self.velocity_range / scut.c_km_s * (1 + self.redshift)
        start = {
            "redsh": (self.redshift, (self.redshift - dz, self.redshift + dz)),
            "fwhm_km_s": (self.start_fwhm[0], self.fwhm_range),
        }
        comps, pars = [], {}
        for member in members:
            cname = member._new_name(model)
            comp = FitComponent(cname, member.function)
            own = comp.create_params()
            if "cen" in comp.function_args:
                own[f"{cname}_cen"].value = member.lines[0]
                own[f"{cname}_cen"].vary = False
            for arg, value in member.fixed.items():
                own[f"{cname}_{arg}"].value = value
                own[f"{cname}_{arg}"].vary = False
            for arg in self.args:
                param = own.pop(f"{cname}_{arg}")
                shared = self.shared_name(arg)
                comp.param_mapping[arg] = shared
                if shared not in pars:
                    param.name = shared
                    if arg in start:
                        param.value = start[arg][0]
                        param.prior = UniformPrior(shared, *start[arg][1])
                    elif param.prior is not None:
                        param.prior.name = shared
                    pars[shared] = param
            comps.append(comp)
            pars.update(own)

        keys = [comp.param_mapping["flux"] for comp in comps]
        signs = [1.0] * len(members)
        valid = []
        for variant in self._variants(model, result, pars):
            trial = copy.deepcopy(pars)
            for key, value in variant.items():
                trial[key].value = value
            amps = _linear_amplitudes(model, result, comps, trial, keys)
            scaled = None if amps is None else _scaled_fluxes(amps, signs, 1.0)
            if scaled is not None:
                valid.append((variant, scaled[0]))
        if not valid:
            return []
        top = max(abs(v) for _, values in valid for v in values)
        for key, sign in zip(keys, signs, strict=True):
            bounds = sorted([0.0, sign * self.prior_factor * top])
            pars[key].prior = UniformPrior(key, *bounds)
        starts = []
        for variant, values in valid:
            start = dict(variant)
            for key, value in zip(keys, values, strict=True):
                prior = pars[key].prior
                start[key] = float(np.clip(value, prior.low, prior.upp))
            starts.append(start)
        for key, value in starts[0].items():
            pars[key].value = value

        names = ", ".join(comp.name for comp in comps)
        return [
            Candidate(
                description=f"add {self.name} group {names} at z "
                f"{pars[self.shared_name('redsh')].value:.4f}"
                if "redsh" in self.args
                else f"add {self.name} group {names}",
                components=comps,
                parameters=pars,
                starts=starts[1:],
                reject_at_bound=[self.shared_name(arg) for arg in self.args],
            )
        ]

    def _peak_redshift(self, model, result, fwhm):
        """Redshift of the most significant residual peak of width `fwhm`
        near the members' lines, or None."""
        x = model.spec.dispersion
        best = (0.0, None)
        for member in self.members:
            line = member.lines[0]
            obs = line * (1 + self.redshift)
            near = np.abs(x - obs) <= self.velocity_range / scut.c_km_s * obs
            near &= model.get_fit_mask()
            if not np.any(near):
                continue
            signif, _ = member.significance_map(model, result, fwhm)
            signif = np.where(near, signif, 0.0)
            idx = int(np.argmax(signif))
            if signif[idx] > max(best[0], member.min_significance):
                best = (signif[idx], x[idx] / line - 1)
        return best[1]

    def _variants(self, model, result, pars):
        """Starting values: each width of `start_fwhm` at `redshift`, then
        the strongest residual peak's redshift (if offset by more than a
        quarter of the width)."""
        z_key = self.shared_name("redsh") if "redsh" in self.args else None
        w_key = self.shared_name("fwhm_km_s") if "fwhm_km_s" in self.args else None
        widths = self.start_fwhm if w_key is not None else [self.start_fwhm[0]]
        variants = []
        for width in widths:
            variant = {} if w_key is None else {w_key: float(width)}
            if z_key is not None:
                variant[z_key] = float(self.redshift)
            variants.append(variant)
        if z_key is not None:
            width = self.start_fwhm[0]
            z_peak = self._peak_redshift(model, result, width)
            dz = width / 4 / scut.c_km_s * (1 + self.redshift)
            prior = pars[z_key].prior
            if (
                z_peak is not None
                and abs(z_peak - self.redshift) > dz
                and prior.low < z_peak < prior.upp
            ):
                variant = {z_key: float(z_peak)}
                if w_key is not None:
                    variant[w_key] = float(width)
                variants.append(variant)
        return variants


class CompanionLineRule(Rule):
    """Rule: a further line on the kinematics of a `LineGroupRule`, e.g. a
    narrow Hgamma at the redshift and width of the narrow Hbeta and [OIII].

    Proposed once the group is in the model (its shared parameters are
    there): one component of `member`'s line using the group's shared
    parameters, so that only its flux is new and a weak line may earn the
    threshold. The component is named and owned by `member`; backward
    steps may remove it alone, against this rule's threshold
    (`removable_alone`), and it goes with the group when the group is
    removed. `calibrate_threshold` calibrates the rule on simulations
    without its components (`null_removals`).

    :param group: The group whose shared parameters the line uses
    :type group: LineGroupRule
    :param member: Rule of the line, with one line in `lines`
    :type member: GaussianLineRule
    :param name: Name of the rule (default "<group> <member>")
    :type name: str
    """

    def __init__(self, group, member, name=None):
        if member.lines is None or len(member.lines) != 1:
            raise ValueError(f"Member {member.prefix!r} needs exactly one line.")
        self.group = group
        self.member = member
        self.name = f"{group.name} {member.prefix}" if name is None else name

    def _shared(self):
        return {self.group.shared_name(arg) for arg in self.group.args}

    def owned(self, model):
        """The member rule owns the components.

        :rtype: list
        """
        return []

    def companions(self, model):
        """The member's components on the group's shared parameters.

        :param model: The model
        :type model: sculptor.model.FitModel
        :rtype: list(str)
        """
        owned = set(self.member.owned(model))
        shared = self._shared()
        return [
            c.name
            for c in model.components
            if c.name in owned and set(c.param_mapping.values()) & shared
        ]

    def removable_alone(self, model):
        """The companions (see `Rule.removable_alone`).

        :rtype: list(str)
        """
        return self.companions(model)

    def null_removals(self, model):
        """The companions (the rule's hypothesis).

        :rtype: list(str)
        """
        return self.companions(model)

    def propose(self, model, result):
        """The line on the group's kinematics (see the class).

        :param model: The current model
        :type model: sculptor.model.FitModel
        :param result: Its quick fit
        :type result: sculptor.quick_fit.QuickFitResult
        :return: Zero or one candidate
        :rtype: list(Candidate)
        """
        member = self.member
        if not self._shared() <= set(model.parameters):
            return []
        if self.companions(model) or not member._search_mask(model).any():
            return []
        if len(member.owned(model)) >= member.max_components:
            return []
        cname = member._new_name(model)
        comp = FitComponent(cname, member.function)
        pars = comp.create_params()
        if "cen" in comp.function_args:
            pars[f"{cname}_cen"].value = member.lines[0]
            pars[f"{cname}_cen"].vary = False
        for arg, value in member.fixed.items():
            pars[f"{cname}_{arg}"].value = value
            pars[f"{cname}_{arg}"].vary = False
        for arg in self.group.args:
            del pars[f"{cname}_{arg}"]
            comp.param_mapping[arg] = self.group.shared_name(arg)
        flux_key = comp.param_mapping["flux"]
        pars[flux_key].value = 1.0
        amp = _linear_amplitude(model, result, comp, pars, flux_key)
        if amp is None or amp <= 0:
            return []
        pars[flux_key].value = amp
        pars[flux_key].prior = UniformPrior(flux_key, 0.0, 100 * amp)
        return [
            Candidate(
                description=f"add {cname} at line {member.lines[0]:.2f} A on the "
                f"{self.group.name} group's kinematics",
                components=[comp],
                parameters=pars,
            )
        ]


@dataclass
class TieTest:
    """Outcome of the last test of a `Tie`.

    :param status: "tied" (accepted now or earlier), "rejected" (the untied
        model earns the threshold) or "not applicable" (see `reason`)
    :type status: str
    :param components: Names of the components it applies to
    :type components: list(str)
    :param delta: Criterion of the tied minus the untied model (NaN if not
        tested)
    :type delta: float
    :param reason: Why it was not applicable
    :type reason: str
    """

    status: str
    components: list = field(default_factory=list)
    delta: float = np.nan
    reason: str = ""

    def __str__(self):
        if self.status == "not applicable":
            return f"not applicable: {self.reason}"
        text = f"{self.status} ({', '.join(self.components)}"
        if np.isfinite(self.delta):
            text += f", tied - untied {self.delta:+.2f}"
        return text + ")"


def _shared_range(model, names, arg):
    """Intersection (low, upp) of the uniform priors of `arg` of the
    components `names`; False if they do not overlap, None if not all the
    priors are uniform."""
    comps = [c for c in model.components if c.name in names]
    priors = [model.parameters[c.param_mapping[arg]].prior for c in comps]
    if not all(isinstance(p, UniformPrior) for p in priors):
        return None
    low, upp = max(p.low for p in priors), min(p.upp for p in priors)
    return (low, upp) if low < upp else False


class Tie(Rule):
    """Hypothesis that components of several rules share parameters, e.g.
    the narrow cores of Hbeta and [OIII] share redshift and width.

    It applies to the narrowest component of each rule `<prefix>_<n>` in
    `prefixes` whose `width_arg` is below `max_width`, if at least
    `min_components` rules have one. `AutoFit` tests it when its search
    converges, like a removal: the tied model is kept unless the untied one
    earns the tie's threshold (see the module docstring). The shared parameters
    are `<name>_<arg>`, with the intersection of the components' uniform
    priors, fitted from each component's values (multi-start). A tie is
    not undone later.

    Pass ties among the rules of `AutoFit`; they propose no components.

    :param name: Name of the tie and prefix of the shared parameters
    :type name: str
    :param prefixes: Prefixes of the rules whose components are tied (e.g.
        `GaussianLineRule` prefixes)
    :type prefixes: list(str)
    :param args: Arguments to share
    :type args: list(str)
    :param max_width: Limit of `width_arg` of a tied component (e.g. the
        narrow-line FWHM limit; a component at it is not tied: a broad-only
        rule's range may start there), or None for any
    :type max_width: float
    :param width_arg: Argument that measures the width
    :type width_arg: str
    :param min_components: Smallest number of components to tie
    :type min_components: int
    """

    def __init__(
        self,
        name,
        prefixes,
        args=("redsh", "fwhm_km_s"),
        max_width=None,
        width_arg="fwhm_km_s",
        min_components=2,
    ):
        self.name = name
        self.prefixes = list(prefixes)
        self.args = list(args)
        self.max_width = max_width
        self.width_arg = width_arg
        self.min_components = min_components

    def propose(self, model, result):
        """Ties propose no components.

        :rtype: list
        """
        return []

    def owned(self, model):
        """Ties own no components.

        :rtype: list
        """
        return []

    def shared_name(self, arg):
        """Name of the shared parameter of `arg`."""
        return f"{self.name}_{arg}"

    def select(self, model):
        """The components to tie, one per prefix.

        :param model: The model (with its best-fit values)
        :type model: sculptor.model.FitModel
        :return: component names, or None and the reason
        :rtype: (list(str) or None, str)
        """
        chosen, missing = [], []
        for prefix in self.prefixes:
            head = prefix + "_"
            widths = {}
            for comp in model.components:
                if not (
                    comp.name.startswith(head) and comp.name[len(head) :].isdigit()
                ):
                    continue
                key = comp.param_mapping.get(self.width_arg)
                if key is None:
                    continue
                width = model.parameters[key].value
                if self.max_width is None or width < self.max_width:
                    widths[comp.name] = width
            # A component already sharing the tie's parameters (e.g. added
            # by a `LineGroupRule` of the same name) comes first
            sharing = [
                c.name
                for c in model.components
                if c.name in widths
                and c.param_mapping.get(self.width_arg)
                == self.shared_name(self.width_arg)
            ]
            if sharing:
                chosen.append(sharing[0])
            elif widths:
                chosen.append(min(widths, key=widths.get))
            else:
                missing.append(prefix)
        if len(chosen) < self.min_components:
            limit = "" if self.max_width is None else f" < {self.max_width:g}"
            return None, (
                f"no component with {self.width_arg}{limit} for {', '.join(missing)}"
            )
        for name in chosen:
            comp = next(c for c in model.components if c.name == name)
            for arg in self.args:
                key = comp.param_mapping.get(arg)
                if key is None or not model.parameters[key].vary:
                    return None, f"{name} has no variable {arg}"
        for arg in self.args:
            if _shared_range(model, chosen, arg) is False:
                return None, f"the {arg} priors of {', '.join(chosen)} do not overlap"
        return chosen, ""

    def is_tied(self, model, names):
        """Whether the components already share all the parameters."""
        comps = [c for c in model.components if c.name in names]
        return all(len({c.param_mapping[arg] for c in comps}) == 1 for arg in self.args)

    def apply(self, model, names):
        """Copy of `model` with the components `names` tied.

        :return: the tied model and alternative starts (the values of each
            further component)
        :rtype: (sculptor.model.FitModel, list(dict))
        """
        new = copy.deepcopy(model)
        comps = [c for c in new.components if c.name in names]
        starts = [{} for _ in comps[1:]]
        for arg in self.args:
            keys = [c.param_mapping[arg] for c in comps]
            shared = self.shared_name(arg)
            param = copy.deepcopy(new.parameters[keys[0]])
            param.name = shared
            shared_range = _shared_range(new, names, arg)
            if shared_range is False:
                raise ValueError(f"Tie {self.name}: the {arg} priors do not overlap")
            if shared_range is not None:
                param.prior = UniformPrior(shared, *shared_range)
            elif param.prior is not None:
                param.prior.name = shared
            values = [new.parameters[k].value for k in keys]
            if isinstance(param.prior, UniformPrior):
                values = [
                    float(np.clip(v, param.prior.low, param.prior.upp)) for v in values
                ]
            param.value = values[0]
            for start, value in zip(starts, values[1:], strict=True):
                start[shared] = value
            for comp in comps:
                comp.param_mapping[arg] = shared
            new.parameters[shared] = param
        used = {key for c in new.components for key in c.get_arg_keys()}
        new.parameters = {k: p for k, p in new.parameters.items() if k in used}
        return new, starts

    def untie(self, model, names):
        """Copy of `model` with the components `names` untied: each gets its
        own copy (`<component>_<arg>`, same value and prior) of the shared
        parameters.

        :rtype: sculptor.model.FitModel
        """
        new = copy.deepcopy(model)
        for comp in new.components:
            if comp.name not in names:
                continue
            for arg in self.args:
                own = f"{comp.name}_{arg}"
                param = copy.deepcopy(new.parameters[comp.param_mapping[arg]])
                param.name = own
                if param.prior is not None:
                    param.prior.name = own
                new.parameters[own] = param
                comp.param_mapping[arg] = own
        used = {key for c in new.components for key in c.get_arg_keys()}
        new.parameters = {k: p for k, p in new.parameters.items() if k in used}
        return new


def _local_maxima(values, minimum):
    """Indices of local maxima of `values` above `minimum`."""
    inner = (values[1:-1] > values[:-2]) & (values[1:-1] >= values[2:])
    idx = np.flatnonzero(inner) + 1
    return idx[values[idx] > minimum]


@dataclass
class AutoFitStep:
    """One accepted change of the model.

    :param action: "start", "add", "remove", "tie" or "clip" (the fit mask
        changed; later scores are on the new mask, `delta` is not an
        improvement)
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
    :param n_fits: Number of quick fits run (of this search; of all the
        searches and the calibration for the result `AutoFit.run` returns)
    :type n_fits: int
    :param threshold: Threshold of this search
    :type threshold: float
    :param quantile: With a calibrated threshold: its quantile
    :type quantile: float
    :param calibration: The calibration, if the threshold was calibrated
    :type calibration: Calibration
    :param by_quantile: With a calibrated threshold: quantile -> result of
        the search with that quantile's threshold (this one included)
    :type by_quantile: dict
    :param preliminary: With a calibrated threshold: the preliminary search
        the first calibration simulated
    :type preliminary: AutoFitResult
    :param calibrations: With a calibrated threshold: every calibration, in
        order (the first on the preliminary model, each later one on the
        model selected with the one before; the last is `calibration`)
    :type calibrations: list(Calibration)
    :param absorbers: The absorber inventory, if `AutoFit(absorbers=...)`
        searched for them (`sculptor.absorbers.AbsorberInventory`)
    :type absorbers: sculptor.absorbers.AbsorberInventory
    :param ties: Tie name -> outcome of its last test
    :param rule_thresholds: Rule key -> its own threshold in this search
        (per-rule calibration); the others used `threshold`
    :type rule_thresholds: dict
    :type ties: dict(str, TieTest)
    :param converged: False if the search stopped at `max_iterations`
        (the model is the last state, not a converged one)
    :type converged: bool
    """

    model: object
    result: object
    criterion: str
    history: list = field(default_factory=list)
    n_fits: int = 0
    threshold: float = np.nan
    quantile: float = np.nan
    calibration: object = None
    by_quantile: dict = field(default_factory=dict)
    preliminary: object = None
    calibrations: list = field(default_factory=list)
    ties: dict = field(default_factory=dict)
    absorbers: object = None
    rule_thresholds: dict = field(default_factory=dict)
    converged: bool = True

    @property
    def at_bound(self):
        """Parameters of the selected model at a uniform prior's bound (the
        data may want a value the prior excludes).

        :rtype: list(str)
        """
        return list(self.result.at_bound)

    def components(self):
        """Names of the selected components.

        :rtype: list(str)
        """
        return [c.name for c in self.model.components]

    def __str__(self):
        where = f"threshold {self.threshold:.1f}"
        if self.calibration is not None:
            where += f" ({self.quantile:.0%} quantile)"
        header = (
            f"AutoFit ({self.criterion}, {where}): "
            f"{len(self.model.components)} components after "
            f"{len(self.history) - 1} changes and {self.n_fits} quick fits"
        )
        lines = [header]
        for step in self.history:
            lines.append(
                f"  {step.action:6s} {self.criterion} = {step.score:.2f} "
                f"(delta {step.delta:+.2f}): {step.description}"
            )
        for name, test in self.ties.items():
            lines.append(f"  tie {name}: {test}")
        if self.at_bound:
            lines.append(f"  at a prior bound: {', '.join(self.at_bound)}")
        if not self.converged:
            lines.append("  not converged: stopped at max_iterations")
        if self.absorbers is not None:
            lines += ["", str(self.absorbers)]
        if self.calibration is not None:
            lines += ["", str(self.calibration)]
            if len(self.calibrations) > 1:
                first = self.calibrations[0]
                lines.append(
                    f"  recalibrated {len(self.calibrations) - 1} time(s) on the "
                    f"selected model; on the preliminary model: "
                    + ", ".join(
                        f"{q:.1%}: {t:.1f}" for q, t in first.thresholds.items()
                    )
                )
                if self.calibration is not self.calibrations[-1]:
                    lines.append(
                        "  the selected models cycle: kept calibration "
                        f"{self.calibrations.index(self.calibration)}, the "
                        "cycle's highest threshold"
                    )
            lines += ["", "Models by quantile:"]
            mine = set(self.components())
            for quantile, other in self.by_quantile.items():
                names = other.components()
                text = (
                    f"  {quantile:.1%}: threshold {other.threshold:.1f}, "
                    f"{len(names)} components, reduced chi2 "
                    f"{other.result.red_chisq:.3f}"
                )
                extra = sorted(set(names) - mine)
                missing = sorted(mine - set(names))
                if extra:
                    text += f"; also {', '.join(extra)}"
                if missing:
                    text += f"; without {', '.join(missing)}"
                if not other.converged:
                    text += "; not converged"
                lines.append(text)
        return "\n".join(lines)


def _component_names(model):
    """The model's component names, sorted (to compare models)."""
    return tuple(sorted(c.name for c in model.components))


class AutoFit:
    """Greedy model building on the quick fit (see the module docstring).

    The base model is copied, not changed.

    :param model: Base model: spectrum, fit mask and the components to start
        with (e.g. the continuum), with starting values and priors
    :type model: sculptor.model.FitModel
    :param rules: Rules that propose components (e.g. `GaussianLineRule`)
    :type rules: list
    :param criterion: "bic" (default), "aicc", "aic", "spic" (see `spic`),
        or a callable `criterion(result, model)` taking the
        `QuickFitResult` and the fitted model and returning the score
        (lower is better)
    :type criterion: str or callable
    :param threshold: Minimum improvement of the criterion to add a
        component (and to keep one), or "calibrate" (default): calibrated
        with `calibrate_threshold` on the model of a preliminary search
        (see the module docstring)
    :type threshold: float or str
    :param removable: Names of base-model components that may be removed;
        components added by the rules always may
    :type removable: list(str)
    :param quick_fit_kwargs: Passed on to `FitModel.run_quick_fit`
    :type quick_fit_kwargs: dict
    :param clip: Sigma clipping between steps: keyword arguments of
        `FitModel.sigma_clip` plus `max_iterations` (default 10: clip and
        refit until the mask no longer changes; a warning if it still
        changes after that many passes or returns to an earlier mask), e.g.
        `{"low": 3, "grow": 2}`; None for no clipping
    :type clip: dict
    :param clip_after: Start clipping once this many components were added
        (before, the model is too far off and would clip real features)
    :type clip_after: int
    :param multistart: Fit candidates from their alternative starting values
        too (`Candidate.starts`); False for one start (faster)
    :type multistart: bool
    :param calibration: With `threshold="calibrate"`: keyword arguments of
        `calibrate_threshold`, e.g. `{"quantiles": (0.9, 0.95, 0.99),
        "n_sims": 500}`; the first quantile selects `AutoFitResult.model`
        (the floor is `min_threshold`, not set here)
    :type calibration: dict
    :param min_threshold: With `threshold="calibrate"`: threshold of the
        preliminary search and floor of the calibrated thresholds, global
        and per rule (see `MIN_THRESHOLD`)
    :type min_threshold: float
    :param absorbers: Keyword arguments of
        `sculptor.absorbers.find_absorbers` (e.g. `{"redshift": 7.18,
        "name": "J0412"}`) to search the preliminary model's residuals (a
        first search with a fixed threshold) for narrow absorbers, report
        them (`AutoFitResult.absorbers`) and mask them (`absorber_mask`) for
        the calibration and the searches; None (default) for no search
    :type absorbers: dict
    :param per_rule: With `threshold="calibrate"`: each rule's candidates
        must also earn the rule's own calibrated threshold, where it has one
        (`calibrate_threshold(per_rule=True)`), instead of the global one
    :type per_rule: bool
    :param recalibrate: With `threshold="calibrate"`: at most this many
        further calibrations, each on the model the first quantile selected
        with the previous one (then the searches are rerun); stops early
        when the selected model has the components of the model calibrated
        on, or of an earlier one: a cycle, of whose calibrations the one
        with the highest (first-quantile) threshold and its searches are
        kept. The noise of every calibration comes
        from the preliminary model's residuals. 0 (default) calibrates on
        the preliminary model only
    :type recalibrate: int
    """

    def __init__(
        self,
        model,
        rules,
        criterion="bic",
        threshold="calibrate",
        removable=(),
        quick_fit_kwargs=None,
        clip=None,
        clip_after=2,
        multistart=True,
        calibration=None,
        min_threshold=MIN_THRESHOLD,
        absorbers=None,
        per_rule=True,
        recalibrate=0,
    ):
        if isinstance(threshold, str) and threshold != "calibrate":
            raise ValueError(
                f"threshold must be a number or 'calibrate', not {threshold!r}"
            )
        if "min_threshold" in (calibration or {}):
            raise ValueError(
                "Set the floor with AutoFit(min_threshold=...), not in `calibration`."
            )
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
        self.criterion_arg = criterion
        self.base = model
        self.rules = list(rules)
        self.threshold = threshold
        self.removable = set(removable)
        self.quick_fit_kwargs = dict(quick_fit_kwargs or {})
        self.clip = None if clip is None else dict(clip)
        self.clip_after = clip_after
        self.multistart = multistart
        self.calibration = dict(calibration or {})
        self.min_threshold = min_threshold
        self.absorbers = None if absorbers is None else dict(absorbers)
        self.per_rule = per_rule
        self.recalibrate = int(recalibrate)
        self.rule_keys = rule_keys(self.rules)
        # Rule key -> threshold of the running search (per-rule thresholds)
        self._rule_thresholds = {}
        self.n_fits = 0

    def _fit(self, model):
        """Quick fit (best-fit values written into the parameters) and score;
        None if the fit fails."""
        self.n_fits += 1
        try:
            with _quiet_fits():
                result = model.run_quick_fit(
                    update_values=True, **self.quick_fit_kwargs
                )
        except (ValueError, np.linalg.LinAlgError) as err:
            logger.debug(f"AutoFit: quick fit failed: {err}")
            return None, np.inf
        score = self.criterion(result, model)
        return result, score if np.isfinite(score) else np.inf

    def _fit_candidate(self, model, candidate):
        """Fit `model` with the candidate from each of its starts.

        The best start is the best-scoring one among those whose fit is not
        rejected at a prior bound (`_rejected_at_bound`), or among all of
        them if every fit is: a start that drifts to a bound does not hide
        one that does not.

        :return: trial model, its quick fit and score, of the best start
            (quick fit None and score inf if all fits failed)
        """
        starts = [{}] + (candidate.starts if self.multistart else [])
        best, best_key = (None, None, np.inf), None
        for idx, start in enumerate(starts):
            trial = self._with(model, candidate)
            for name, value in start.items():
                trial.parameters[name].value = value
            trial_result, trial_score = self._fit(trial)
            if trial_result is None:
                key = (2, np.inf)
            else:
                key = (int(bool(_at_bound(candidate, trial_result))), trial_score)
            if best_key is None or key < best_key:
                if best_key is not None:
                    logger.debug(
                        f"AutoFit: {candidate.description}: start {idx} {start} "
                        f"is better ({trial_score:.2f})"
                    )
                best, best_key = (trial, trial_result, trial_score), key
        return best

    def _rejected_at_bound(self, candidate, result):
        """Whether the candidate's fit is rejected for parameters at a prior
        bound (its `reject_at_bound` ones)."""
        own = _at_bound(candidate, result)
        if own:
            logger.debug(
                f"AutoFit: {candidate.description}: rejected, "
                f"{', '.join(sorted(own))} at a prior bound"
            )
        return bool(own)

    def _removable(self, model):
        """Components that may be removed."""
        added = set()
        for rule in self.rules:
            added.update(rule.owned(model))
        return [c.name for c in model.components if c.name in added | self.removable]

    def _removal_units(self, model):
        """What a backward step may remove: each removable component on its
        own, except that components sharing guarded parameters (see
        `Rule.guarded`) go together.

        :rtype: list(list(str))
        """
        return [names for names, _ in self._keyed_removal_units(model)]

    def _keyed_removal_units(self, model):
        """`_removal_units` with the key of the rule that owns each unit
        (the guarding rule for a group, the `removable_alone` rule for a
        component it names; None if no rule owns it).

        :rtype: list(tuple(list(str), str))
        """
        removable = self._removable(model)
        units, grouped, alone = [], set(), {}
        for key, rule in zip(self.rule_keys, self.rules, strict=True):
            for _, comps in rule.guarded(model):
                if all(name in removable for name in comps):
                    units.append((list(comps), key))
                grouped.update(comps)
            for name in rule.removable_alone(model):
                alone.setdefault(name, key)
        owner = {}
        for key, rule in zip(self.rule_keys, self.rules, strict=True):
            for name in rule.owned(model):
                owner.setdefault(name, key)
        units.extend(
            ([name], alone.get(name, owner.get(name)))
            for name in removable
            if name not in grouped or name in alone
        )
        return units

    @staticmethod
    def _with(model, candidate):
        """Copy of `model` with the candidate's components added."""
        new = copy.deepcopy(model)
        new.components.extend(copy.deepcopy(candidate.components))
        new.add_parameters(copy.deepcopy(candidate.parameters))
        return new

    @staticmethod
    def _without(model, name, copy_model=True):
        """Copy of `model` (or `model` itself) without component `name` and
        the parameters only it used."""
        new = copy.deepcopy(model) if copy_model else model
        new.components = [c for c in new.components if c.name != name]
        used = {key for c in new.components for key in c.get_arg_keys()}
        new.parameters = {k: p for k, p in new.parameters.items() if k in used}
        return new

    def run(self, max_iterations=40):
        """Build the model: one search with a fixed threshold, or with
        `threshold="calibrate"` a preliminary search, the calibration and a
        search per quantile.

        :param max_iterations: Maximum number of changes of the model per
            search (additions, removals and ties; the search stops, not
            converged, when a further one is due)
        :type max_iterations: int
        :return: The selected model, its quick fit and the history (and the
            calibration and the models of the other quantiles)
        :rtype: AutoFitResult
        :raises ValueError: if the base model cannot be fitted
        """
        base = self.base
        try:
            return self._run(max_iterations)
        finally:
            # The absorber mask applies to this run only
            self.base = base

    def _run(self, max_iterations):
        """`run` (it restores the base model)."""
        self.n_fits = 0
        inventory = None
        if self.threshold != "calibrate":
            if self.absorbers is None:
                return self._search(self.threshold, max_iterations)
            first = self._search(self.threshold, max_iterations)
            inventory = self._mask_absorbers(first.model)
            if not len(inventory.features):
                first.absorbers = inventory
                return first
            found = self._search(self.threshold, max_iterations)
            found.absorbers = inventory
            found.n_fits = self.n_fits
            return found

        preliminary = self._search(self.min_threshold, max_iterations)
        if self.absorbers is not None:
            inventory = self._mask_absorbers(preliminary.model)
            if len(inventory.features):
                # The null model of the calibration without the absorbers
                preliminary.model.absorber_mask = self.base.absorber_mask
                refit, _ = self._fit(preliminary.model)
                if refit is not None:
                    preliminary.model.quick_fit_result = refit
        # The noise always comes from the preliminary model's residuals, so
        # that it does not depend on the null. Recalibrating still changes
        # the thresholds: they depend on the null's components (J1504: 44 on
        # the preliminary model, 87 on the selected one, about the same as
        # with the selected model's residuals as the noise)
        kwargs = {
            "per_rule": self.per_rule,
            "noise_model": preliminary.model,
            "min_threshold": self.min_threshold,
            **self.calibration,
        }
        null, calibrations, nulls, rounds = preliminary.model, [], [], []
        for round_ in range(self.recalibrate + 1):
            nulls.append(_component_names(null))
            calibration = calibrate_threshold(
                null,
                self.rules,
                criterion=self.criterion_arg,
                quick_fit_kwargs=self.quick_fit_kwargs,
                multistart=self.multistart,
                **kwargs,
            )
            self.n_fits += calibration.n_fits
            calibrations.append(calibration)
            by_quantile = self._quantile_searches(calibration, max_iterations)
            rounds.append((calibration, by_quantile))
            names = _component_names(next(iter(by_quantile.values())).model)
            if names in nulls:
                # A model calibrated on before: the same null again (a fixed
                # point), or a cycle through the nulls since then, whose
                # highest threshold is kept (the result does not depend on
                # where `recalibrate` stops)
                cycle = rounds[nulls.index(names) :]
                calibration, by_quantile = max(
                    cycle, key=lambda item: next(iter(item[0].thresholds.values()))
                )
                if len(cycle) > 1:
                    logger.info(
                        f"AutoFit: recalibration cycles through {len(cycle)} "
                        "models; the highest threshold is kept"
                    )
                break
            if round_ == self.recalibrate:
                break
            selected = next(iter(by_quantile.values()))
            logger.info(
                "AutoFit: recalibrating on the selected model "
                f"({', '.join(selected.components())})"
            )
            null = copy.deepcopy(selected.model)
        selected = next(iter(by_quantile.values()))
        for found in by_quantile.values():
            found.calibration = calibration
            found.calibrations = calibrations
            found.preliminary = preliminary
            found.absorbers = inventory
        selected.n_fits = self.n_fits
        return selected

    def _quantile_searches(self, calibration, max_iterations):
        """A search per quantile of `calibration` (shared where the
        thresholds are the same).

        :return: Quantile -> its search, in the calibration's order
        :rtype: dict
        """
        by_quantile, by_threshold = {}, {}
        for quantile, threshold in calibration.thresholds.items():
            rule_thresholds = calibration.rule_thresholds(quantile)
            cache = (threshold, tuple(sorted(rule_thresholds.items())))
            if cache not in by_threshold:
                by_threshold[cache] = self._search(
                    threshold, max_iterations, rule_thresholds
                )
            found = copy.copy(by_threshold[cache])
            found.quantile = quantile
            found.by_quantile = by_quantile
            by_quantile[quantile] = found
        return by_quantile

    def _mask_absorbers(self, model):
        """Absorber inventory from `model`'s residuals; the absorbers are
        masked in the base model of the following searches, on top of an
        absorber mask the base model already has.

        :rtype: sculptor.absorbers.AbsorberInventory
        """
        from sculptor import absorbers as scabs

        inventory = scabs.find_absorbers(model, **self.absorbers)
        logger.info(f"AutoFit: {inventory}")
        given = getattr(self.base, "absorber_mask", None)
        self.base = copy.deepcopy(self.base)
        self.base.absorber_mask = (
            inventory.mask if given is None else given & inventory.mask
        )
        return inventory

    def _search(self, threshold, max_iterations, rule_thresholds=None):
        """One greedy search with a fixed threshold (and per-rule ones)."""
        self._rule_thresholds = dict(rule_thresholds or {})
        try:
            return self._greedy(threshold, max_iterations)
        finally:
            self._rule_thresholds = {}

    def _greedy(self, threshold, max_iterations):
        """`_search` (with `_rule_thresholds` set)."""
        n_fits_before = self.n_fits
        logger.info(
            f"AutoFit: {self.criterion_name}, threshold {threshold:.2f}, "
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
        ties = {}

        # Every accepted change (an addition, a removal at a bound or a tie)
        # is followed by the backward steps and, once `clip_after`
        # components were added, by the clipping
        converged = False
        while True:
            change = self._next_change(model, result, score, seen, threshold, ties)
            if change is None:
                converged = True
                break
            if _n_changes(history) >= max_iterations:
                break
            action, trial, trial_result, trial_score, description, tie_test = change
            history.append(
                AutoFitStep(
                    action,
                    description,
                    trial_score,
                    trial_score - score,
                    [c.name for c in trial.components],
                )
            )
            logger.info(f"AutoFit: {description}: {trial_score - score:+.2f}")
            if tie_test is not None:
                ties[tie_test[0]] = tie_test[1]
            model, result, score = trial, trial_result, trial_score
            seen.add(_signature(model))

            model, result, score, stopped = self._backward(
                model, result, score, history, seen, threshold, max_iterations
            )
            n_added = sum(step.action == "add" for step in history)
            if self.clip is not None and n_added >= self.clip_after:
                model, result, score = self._clip(model, result, score, history)
            if stopped:
                break

        if not converged:
            logger.warning(
                f"AutoFit: not converged after {max_iterations} changes; "
                "the model is the last state of the search"
            )
        if self.clip is not None:
            model, result, score = self._clip(model, result, score, history, final=True)
        model.quick_fit_result = result
        logger.info(
            f"AutoFit done: {[c.name for c in model.components]}, "
            f"{self.criterion_name} = {score:.2f}"
        )
        return AutoFitResult(
            model,
            result,
            self.criterion_name,
            history,
            self.n_fits - n_fits_before,
            threshold=threshold,
            ties=ties,
            rule_thresholds=dict(self._rule_thresholds),
            converged=converged,
        )

    def _next_change(self, model, result, score, seen, threshold, ties):
        """The next change of the search: the best addition; once none earns
        its threshold (convergence), the removal of components whose
        guarded parameters drifted to a bound, else the first tie the data
        accept.

        :return: (action, model, result, score, description, tie outcome
            (name, TieTest) or None), or None if nothing changes
        """
        best = self._best_addition(model, result, score, seen, threshold)
        if best is not None:
            trial, trial_result, trial_score, cand = best
            return "add", trial, trial_result, trial_score, cand.description, None
        change = self._removal_at_bound(model, result)
        if change is not None:
            return change
        return self._accepted_tie(model, score, threshold, ties)

    def _best_addition(self, model, result, score, seen, threshold):
        """The candidate that improves the criterion most, among those that
        earn their rule's threshold.

        :return: (model, result, score, candidate), or None
        """
        best = None
        for key, rule in zip(self.rule_keys, self.rules, strict=True):
            rule_threshold = self._rule_thresholds.get(key, threshold)
            for cand in rule.propose(model, result):
                if _signature(self._with(model, cand)) in seen:
                    continue
                trial, trial_result, trial_score = self._fit_candidate(model, cand)
                logger.debug(
                    f"AutoFit: {cand.description}: "
                    f"{self.criterion_name} {trial_score - score:+.2f}"
                )
                if trial_result is None or self._rejected_at_bound(cand, trial_result):
                    continue
                if score - trial_score < rule_threshold:
                    continue
                if best is None or trial_score < best[2]:
                    best = (trial, trial_result, trial_score, cand)
        return best

    def _removal_at_bound(self, model, result):
        """The removal of the components of the first guarded parameters
        (see `Rule.guarded`) one of which is at a prior bound.

        :return: a change as from `_next_change`, or None
        """
        for rule in self.rules:
            for params, comps in rule.guarded(model):
                bad = sorted(_at_guarded_bound(params, result, model.parameters))
                if not bad:
                    continue
                trial = model
                for name in comps:
                    trial = self._without(trial, name)
                trial_result, trial_score = self._fit(trial)
                if trial_result is None:
                    continue
                description = (
                    f"remove {', '.join(comps)} ({', '.join(bad)} at a prior bound)"
                )
                return "remove", trial, trial_result, trial_score, description, None
        return None

    def _accepted_tie(self, model, score, threshold, ties):
        """Test the applicable ties (see `Tie`) until the data accept one.
        The outcomes of the others are written into `ties`; the accepted
        one's comes with the change.

        :return: a change as from `_next_change`, or None
        """
        for key, tie in zip(self.rule_keys, self.rules, strict=True):
            if not isinstance(tie, Tie):
                continue
            tie_threshold = self._rule_thresholds.get(key, threshold)
            names, reason = tie.select(model)
            if names is None:
                ties[tie.name] = TieTest("not applicable", reason=reason)
                continue
            if tie.is_tied(model, names):
                if ties.get(tie.name) is None or ties[tie.name].status != "tied":
                    ties[tie.name] = TieTest("tied", names)
                continue
            tied, starts = tie.apply(model, names)
            best = (None, None, np.inf)
            for start in [{}, *(starts if self.multistart else [])]:
                trial = copy.deepcopy(tied)
                for key, value in start.items():
                    trial.parameters[key].value = value
                trial_result, trial_score = self._fit(trial)
                if trial_result is not None and trial_score < best[2]:
                    best = (trial, trial_result, trial_score)
            trial, trial_result, trial_score = best
            delta = trial_score - score
            if trial_result is None or delta >= tie_threshold:
                ties[tie.name] = TieTest("rejected", names, delta)
                logger.info(f"AutoFit: tie {tie.name} rejected: {delta:+.2f}")
                continue
            description = f"tie {', '.join(names)} ({', '.join(tie.args)})"
            outcome = (tie.name, TieTest("tied", names, delta))
            return "tie", trial, trial_result, trial_score, description, outcome
        return None

    def _clip(self, model, result, score, history, final=False):
        """Sigma-clip with the current model and refit, until the mask no
        longer changes; record a "clip" step if it changed. The `final` clip
        (end of the search) is logged even if it changed nothing."""
        label = "final sigma clipping" if final else "sigma clipping"
        clip = dict(self.clip)
        max_iterations = clip.pop("max_iterations", 10)
        changed = False
        before = score
        seen = [] if model.clip_mask is None else [model.clip_mask.tobytes()]
        for _ in range(max_iterations):
            previous = model.clip_mask
            with _quiet_fits():
                if not model.sigma_clip(values=result.values, **clip):
                    break
            new_result, new_score = self._fit(model)
            if new_result is None:
                # Keep result and score consistent with the mask
                model.clip_mask = previous
                break
            changed = True
            result, score = new_result, new_score
            key = model.clip_mask.tobytes()
            if key in seen:
                logger.warning(
                    f"AutoFit: {label} returned to an earlier mask (it "
                    "oscillates); keeping the current one."
                )
                break
            seen.append(key)
        else:
            logger.warning(
                f"AutoFit: {label} did not converge in {max_iterations} passes; "
                "keeping the last mask."
            )
        n_clipped = int((model._fit_region_mask() & ~model.get_fit_mask()).sum())
        if final and not changed:
            logger.info(
                f"AutoFit: {label}: mask unchanged ({n_clipped} pixels clipped)"
            )
        if changed:
            history.append(
                AutoFitStep(
                    "clip",
                    f"{label}: {n_clipped} pixels clipped",
                    score,
                    score - before,
                    [c.name for c in model.components],
                )
            )
            logger.info(f"AutoFit: {label}: {n_clipped} pixels clipped, re-scored")
        return model, result, score

    def _guarded_at_bound(self, model, result):
        """Guarded parameters of `model` (see `Rule.guarded`) at their
        guarded prior bound in `result`."""
        return {
            name
            for rule in self.rules
            for params, _ in rule.guarded(model)
            for name in _at_guarded_bound(params, result, model.parameters)
        }

    def _backward(
        self, model, result, score, history, seen, threshold, max_changes=np.inf
    ):
        """Remove components that no longer earn `threshold`, one at a time
        (the least useful first). As for an addition, a removal whose fit
        leaves further guarded parameters at a prior bound is not made: the
        remaining components would take over the removed one's flux.

        :return: model, result, score, and whether a removal is left undone
            because `history` reached `max_changes` changes
        """
        while True:
            worst = None
            before = self._guarded_at_bound(model, result)
            for names, key in self._keyed_removal_units(model):
                trial = model
                for name in names:
                    trial = self._without(trial, name)
                if not trial.components:
                    continue
                trial_result, trial_score = self._fit(trial)
                if trial_result is None:
                    continue
                bad = self._guarded_at_bound(trial, trial_result) - before
                if bad:
                    logger.debug(
                        f"AutoFit: remove {', '.join(names)}: rejected, "
                        f"{', '.join(sorted(bad))} at a prior bound"
                    )
                    continue
                # Keeping `names` is worth trial_score - score
                rule_threshold = self._rule_thresholds.get(key, threshold)
                if trial_score - score < rule_threshold and (
                    worst is None or trial_score < worst[2]
                ):
                    worst = (trial, trial_result, trial_score, ", ".join(names))
            if worst is None:
                return model, result, score, False
            if _n_changes(history) >= max_changes:
                return model, result, score, True

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


def _n_changes(history):
    """Number of changes of the model in `history` (additions, removals and
    ties; not the start or the clipping)."""
    return sum(step.action in ("add", "remove", "tie") for step in history)


def _signature(model):
    """Hashable description of a model's structure (component functions and
    rounded fixed values / prior ranges), to avoid revisiting a model.

    The prior ranges of free amplitudes (arguments named "flux..." or
    "amp...") are left out: rules set them from the residuals, so the same
    component proposed again after a different history would look new."""
    parts = []
    for comp in model.components:
        func = getattr(comp.function, "__name__", str(comp.function))
        keys = []
        for arg, key in zip(comp.function_args, comp.get_arg_keys(), strict=True):
            p = model.parameters[key]
            if not p.vary:
                keys.append(("fixed", round(float(p.value), 3)))
            elif arg.startswith(("flux", "amp")):
                keys.append(("free",))
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
    drawn and multiplied by the errors. It keeps the residuals' amplitude, so
    errors that underestimate the scatter are accounted for too. Pixels
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
    series *= norm.std() / series.std()

    noise = rng.standard_normal(spec.dispersion.size)
    noise[mask] = series
    return noise * err


@dataclass
class Calibration:
    """Result of `calibrate_threshold`: criterion improvements that noise
    alone achieves, and thresholds at their quantiles.

    :param criterion: Name of the criterion
    :type criterion: str
    :param improvements: Largest improvement of each simulation (failed
        simulations left out)
    :type improvements: np.ndarray
    :param thresholds: Quantile -> threshold (at least `min_threshold`)
    :type thresholds: dict
    :param intervals: Quantile -> (low, high): 90% bootstrap interval of the
        quantile of the improvements, its uncertainty from the finite number
        of simulations (before the floor)
    :type intervals: dict
    :param noise: Noise model of the simulations
    :type noise: str
    :param n_fits: Number of quick fits run
    :type n_fits: int
    :param rule_improvements: Rule key (`rule_keys`) -> largest improvement
        of the rule's candidates in each simulation (NaN where it proposed
        none)
    :type rule_improvements: dict
    :param by_rule: Rule key -> {quantile: threshold}: thresholds of the
        rules on their own, for the quantile corrected for the number of
        calibrated rules (see `calibrate_threshold`)
    :type by_rule: dict
    :param min_threshold: Floor of the thresholds
    :type min_threshold: float
    """

    criterion: str
    improvements: np.ndarray
    thresholds: dict
    intervals: dict
    noise: str = "residuals"
    n_fits: int = 0
    rule_improvements: dict = field(default_factory=dict)
    by_rule: dict = field(default_factory=dict)
    min_threshold: float = 0.0

    def rule_thresholds(self, quantile):
        """Rule key -> threshold at `quantile`, for the calibrated rules.

        :param quantile: Quantile, one of `thresholds`
        :type quantile: float
        :rtype: dict
        """
        return {key: values[quantile] for key, values in self.by_rule.items()}

    def threshold(self, quantile):
        """The threshold at `quantile` (computed if it was not requested).

        :param quantile: Quantile, e.g. 0.95
        :type quantile: float
        :rtype: float
        """
        if quantile in self.thresholds:
            return self.thresholds[quantile]
        return max(float(np.quantile(self.improvements, quantile)), self.min_threshold)

    def __str__(self):
        imp = self.improvements
        header = (
            f"Calibration ({self.criterion}, {imp.size} simulations, "
            f"{self.noise} noise): improvements from noise: median "
            f"{np.median(imp):.1f}, zero in {np.mean(imp == 0):.0%}"
        )
        lines = [header]
        for quantile, value in self.thresholds.items():
            low, high = self.intervals[quantile]
            raw = float(np.quantile(imp, quantile))
            floor = f"the floor; quantile {raw:.1f}, " if raw < value else ""
            lines.append(
                f"  {quantile:.1%} quantile: threshold {value:.1f} "
                f"({floor}90% bootstrap interval {low:.1f} - {high:.1f})"
            )
        if self.by_rule:
            lines.append(
                f"  per rule ({len(self.by_rule)} rules; quantile corrected for "
                "their number; others use the threshold above):"
            )
            for key, values in self.by_rule.items():
                text = ", ".join(f"{q:.1%}: {v:.1f}" for q, v in values.items())
                lines.append(f"    {key}: {text}")
        return "\n".join(lines)


def rule_keys(rules):
    """A name per rule (its `name` or `prefix`, else its class; "tie
    <name>" for a `Tie`, which shares its name with the group it tests),
    made unique with the rule's index.

    :param rules: Rules
    :type rules: list
    :rtype: list(str)
    """
    names = [
        f"tie {rule.name}"
        if isinstance(rule, Tie)
        else getattr(rule, "name", None)
        or getattr(rule, "prefix", None)
        or type(rule).__name__
        for rule in rules
    ]
    return [
        name if names.count(name) == 1 else f"{name}#{idx}"
        for idx, name in enumerate(names)
    ]


def tail_threshold(values, quantile, tail_fraction=0.2):
    """Quantile of `values`; above 1 - `tail_fraction`, from an
    exponential fit to the samples above that quantile (peaks over a
    threshold), which also reaches quantiles beyond the sample.

    The tail holds the samples strictly above the base quantile: with ties
    at the base (e.g. many improvements of 0) that is fewer than
    `tail_fraction` of them, and the extrapolation uses their actual
    fraction. A quantile within the ties is the base itself.

    :param values: Samples (e.g. improvements of noise-only simulations)
    :type values: np.ndarray
    :param quantile: Quantile
    :type quantile: float
    :param tail_fraction: Fraction of the samples in the fitted tail, at
        most
    :type tail_fraction: float
    :rtype: float
    """
    values = np.asarray(values, dtype=float)
    if quantile <= 1 - tail_fraction:
        return float(np.quantile(values, quantile))
    base = float(np.quantile(values, 1 - tail_fraction))
    excess = values[values > base] - base
    if excess.size == 0:
        return base
    scale = float(np.mean(excess))
    above = excess.size / values.size  # P(value > base)
    return base + scale * max(0.0, float(np.log(above / (1 - quantile))))


def calibrate_threshold(
    model,
    rules,
    criterion="bic",
    n_sims=200,
    quantiles=(0.95, 0.99),
    noise="residuals",
    seed=0,
    quick_fit_kwargs=None,
    multistart=True,
    n_bootstrap=1000,
    per_rule=True,
    min_rule_sims=20,
    noise_model=None,
    min_threshold=MIN_THRESHOLD,
):
    """Thresholds for `AutoFit` from simulations without extra components.

    Each simulation replaces the flux of a copy of `model` by its best fit
    plus noise ("residuals": with the correlations of the residuals, see
    `correlated_noise`; "white": independent Gaussian noise from the
    errors), quick-fits it, and runs one forward step of the search. The
    largest improvement of the criterion that any candidate achieves is
    noise only; a quantile q of these improvements is the threshold above
    which an added component is unlikely to fit noise (a false positive
    rate of about 1 - q per step). Each threshold comes with a bootstrap
    interval; high quantiles need many simulations (n_sims (1 - q) of them
    lie above the threshold). The thresholds, global and per rule, are at
    least `min_threshold` (see `MIN_THRESHOLD`).

    Use a model of the real features, e.g. from a search with a fixed
    threshold (as `AutoFit` does with `threshold="calibrate"`): residuals
    that still contain real features make the thresholds conservative.

    With `per_rule`, each rule also gets its own threshold from the
    improvements of its own candidates: a constrained hypothesis (e.g. the
    narrow lines at the systemic redshift, `LineGroupRule`) fits noise far
    less than a Gaussian free to go anywhere, and is held to its own noise.
    The quantile is corrected for the number k of calibrated rules, 1 -
    (1 - q) / k, so that the chance that any rule fits noise stays about 1
    - q; such quantiles lie in the tail of a few hundred simulations and are
    extrapolated with an exponential fit to it (`tail_threshold`). A rule
    whose hypothesis is in `model` (`Rule.null_removals`: e.g. the narrow
    group found by the preliminary search, a narrow Hbeta its member rule
    added alone, or the iron template of a `ComponentRule`) is calibrated
    on simulations of `model` without it. Rules that propose nothing in at
    least `min_rule_sims` simulations keep the global threshold. A `Tie`
    that applies to `model` is calibrated on simulations of the tied model
    (`Tie.apply`, or `model` itself if already tied), by the improvement
    that untying (`Tie.untie`) achieves; one that does not apply keeps the
    global threshold. The floor
    matters most here: a hypothesis that never improves the criterion on
    noise gives a threshold of 0, which says only that the simulations do
    not reach its tail.

    Candidates are judged as in the search: a fit rejected for parameters
    at a prior bound (`Candidate.reject_at_bound`) improves
    nothing (the search would not add it), and neither does a failed fit.

    :param model: The null model (spectrum, fit mask, starting values)
    :type model: sculptor.model.FitModel
    :param rules: Rules, as for `AutoFit`
    :type rules: list
    :param criterion: As for `AutoFit`
    :type criterion: str or callable
    :param n_sims: Number of simulations
    :type n_sims: int
    :param quantiles: Quantiles of the null improvements to report
    :type quantiles: tuple(float)
    :param noise: "residuals" or "white"
    :type noise: str
    :param seed: Seed of the random generator
    :type seed: int
    :param quick_fit_kwargs: Passed on to `FitModel.run_quick_fit`
    :type quick_fit_kwargs: dict
    :param multistart: As for `AutoFit` (use the same setting as the
        search)
    :type multistart: bool
    :param n_bootstrap: Bootstrap resamples for the intervals
    :type n_bootstrap: int
    :param per_rule: Also calibrate a threshold per rule
    :type per_rule: bool
    :param min_rule_sims: Simulations a rule needs candidates in to get its
        own threshold
    :type min_rule_sims: int
    :param min_threshold: Floor of the thresholds, global and per rule
        (default `MIN_THRESHOLD`)
    :type min_threshold: float
    :param noise_model: With `noise="residuals"`: the model whose residuals
        give the noise (same spectrum as `model`), None (default) for
        `model`. A null model without some real features counts their
        misfit as noise; the residuals of a richer model (e.g. `AutoFit`'s
        preliminary one) do not
    :type noise_model: sculptor.model.FitModel
    :return: The improvements and the thresholds
    :rtype: Calibration
    :raises ValueError: for an unknown `noise`, quantiles outside (0, 1), or
        a model (or noise model) that cannot be fitted
    """
    if noise not in ("residuals", "white"):
        raise ValueError(f"Unknown noise {noise!r}; use 'residuals' or 'white'.")
    quantiles = tuple(float(q) for q in np.atleast_1d(quantiles))
    if not quantiles or not all(0 < q < 1 for q in quantiles):
        raise ValueError(f"quantiles must lie in (0, 1), not {quantiles}")
    rng = np.random.default_rng(seed)
    search = AutoFit(
        model,
        rules,
        criterion=criterion,
        threshold=0.0,
        quick_fit_kwargs=quick_fit_kwargs,
        multistart=multistart,
    )

    def best_score(sim, sim_result, sim_score, rule):
        """Best score of the rule's candidates on `sim`; `sim_score` if all
        are rejected (the search would add none), None if it proposes none.
        """
        best = None
        for cand in rule.propose(sim, sim_result):
            _, result, score = search._fit_candidate(sim, cand)
            if result is None or search._rejected_at_bound(cand, result):
                score = sim_score
            best = score if best is None else min(best, score)
        return best

    null = copy.deepcopy(model)
    null_result, _ = search._fit(null)
    if null_result is None:
        raise ValueError("calibrate_threshold: the model cannot be fitted.")
    best_flux = null.eval(null.spec.dispersion, null_result.values)
    source, source_result = null, null_result
    if noise == "residuals" and noise_model is not None and noise_model is not model:
        source = copy.deepcopy(noise_model)
        source_result, _ = search._fit(source)
        if source_result is None:
            raise ValueError("calibrate_threshold: the noise model cannot be fitted.")
    keys = rule_keys(search.rules)

    # Rules whose components are in the null model: their own null model
    # without them
    without = {}
    if per_rule:
        for key, rule in zip(keys, search.rules, strict=True):
            names = rule.null_removals(null)
            if not names:
                continue
            reduced = null
            for name in names:
                reduced = search._without(reduced, name)
            reduced_result, _ = search._fit(reduced)
            if reduced_result is not None:
                flux = reduced.eval(reduced.spec.dispersion, reduced_result.values)
                without[key] = (reduced, flux)

    # Ties that apply to the null model: calibrated on simulations of the
    # tied model, by what untying gains
    tied_nulls = {}
    if per_rule:
        for key, tie in zip(keys, search.rules, strict=True):
            if not isinstance(tie, Tie):
                continue
            names, _ = tie.select(null)
            if names is None:
                continue
            if tie.is_tied(null, names):
                tied = copy.deepcopy(null)
            else:
                tied = tie.apply(null, names)[0]
            tied_result, _ = search._fit(tied)
            if tied_result is not None:
                flux = tied.eval(tied.spec.dispersion, tied_result.values)
                tied_nulls[key] = (tie, names, tied, flux)

    improvements = np.full(n_sims, np.nan)
    by_key = {key: np.full(n_sims, np.nan) for key in keys}
    for idx in range(n_sims):
        if noise == "residuals":
            realization = correlated_noise(source, source_result, rng)
        else:
            realization = rng.normal(0, null.spec.fluxden_err)
        sim = copy.deepcopy(null)
        sim.spec.fluxden = best_flux + realization
        sim_result, sim_score = search._fit(sim)
        if sim_result is None:
            continue
        best = sim_score
        for key, rule in zip(keys, search.rules, strict=True):
            rule_best = best_score(sim, sim_result, sim_score, rule)
            if rule_best is not None:
                best = min(best, rule_best)
                by_key[key][idx] = max(sim_score - rule_best, 0.0)
        improvements[idx] = sim_score - best

        for key, (reduced, flux) in without.items():
            rule = search.rules[keys.index(key)]
            red = copy.deepcopy(reduced)
            red.spec.fluxden = flux + realization
            red_result, red_score = search._fit(red)
            if red_result is None:
                continue
            rule_best = best_score(red, red_result, red_score, rule)
            if rule_best is not None:
                by_key[key][idx] = max(red_score - rule_best, 0.0)

        for key, (tie, names, tied, flux) in tied_nulls.items():
            sim_tied = copy.deepcopy(tied)
            sim_tied.spec.fluxden = flux + realization
            tied_result, tied_score = search._fit(sim_tied)
            if tied_result is None:
                continue
            # Untied from the tied best fit (a failed fit gains nothing)
            _, untied_score = search._fit(tie.untie(sim_tied, names))
            by_key[key][idx] = max(tied_score - untied_score, 0.0)
    improvements = improvements[np.isfinite(improvements)]
    if improvements.size == 0:
        raise ValueError("calibrate_threshold: all simulations failed.")
    if improvements.size < n_sims:
        logger.warning(
            f"calibrate_threshold: {n_sims - improvements.size} of {n_sims} "
            "simulations failed"
        )

    resampled = rng.choice(improvements, (n_bootstrap, improvements.size))
    thresholds, intervals = {}, {}
    for quantile in quantiles:
        thresholds[quantile] = float(np.quantile(improvements, quantile))
        if thresholds[quantile] < min_threshold:
            logger.info(
                f"calibrate_threshold: {quantile:.1%} threshold "
                f"{thresholds[quantile]:.1f} raised to the floor {min_threshold:.1f}"
            )
            thresholds[quantile] = float(min_threshold)
        boot = np.quantile(resampled, quantile, axis=1)
        intervals[quantile] = tuple(float(v) for v in np.quantile(boot, [0.05, 0.95]))
        if improvements.size * (1 - quantile) < 3:
            logger.warning(
                f"calibrate_threshold: only {improvements.size * (1 - quantile):.1f}"
                f" simulations above the {quantile:.1%} quantile; use more "
                "simulations for a stable threshold"
            )

    rule_improvements = {
        key: values[np.isfinite(values)]
        for key, values in by_key.items()
        if np.isfinite(values).sum() >= min_rule_sims
    }
    by_rule = {}
    if per_rule and rule_improvements:
        n_rules = len(rule_improvements)
        for key, values in rule_improvements.items():
            by_rule[key] = {}
            for q in quantiles:
                value = tail_threshold(values, 1 - (1 - q) / n_rules)
                if value < min_threshold:
                    logger.info(
                        f"calibrate_threshold: {key}: {q:.1%} threshold "
                        f"{value:.1f} raised to the floor {min_threshold:.1f}"
                    )
                    value = float(min_threshold)
                by_rule[key][q] = value

    calibration = Calibration(
        search.criterion_name,
        improvements,
        thresholds,
        intervals,
        noise=noise,
        n_fits=search.n_fits,
        rule_improvements=rule_improvements,
        by_rule=by_rule,
        min_threshold=float(min_threshold),
    )
    logger.info(str(calibration))
    return calibration
