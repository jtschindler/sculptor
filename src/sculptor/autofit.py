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

and stops when no candidate improves the model.

A candidate can carry alternative starting values (`Candidate.starts`,
e.g. several widths of a new Gaussian, see `GaussianLineRule`): it is
fitted from each and the best fit counts. This matters for degenerate
components, e.g. a second broad Gaussian of a line, whose fit depends on
where it starts.

Ties (`Tie`, passed among the rules) are hypotheses that components share
parameters, e.g. the narrow cores of Hbeta and [OIII] their redshift and
width. When the search converges, each applicable tie is tested like a
removal: it is kept unless the untied model earns `threshold`, and the
search resumes from the tied model. `AutoFitResult.ties` reports each
outcome, e.g. a tie the data reject.

The threshold is calibrated by default (`threshold="calibrate"`): a
preliminary search with a fixed threshold gives a model of the real
features; `calibrate_threshold` simulates it with noise like its residuals
and takes quantiles of the criterion improvements that noise alone
achieves (95% and 99% by default); the search is then rerun for each
quantile. `AutoFitResult` holds the model of the first quantile, the
others in `by_quantile`, and the calibration.

With `clip`, the fit mask is sigma-clipped (`FitModel.sigma_clip`) after
every accepted change once `clip_after` components were added, and the
current model is refitted and re-scored on the new mask. Candidates are
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
    :param remove: Names of components to remove first (e.g. a continuum
        model the candidate replaces)
    :type remove: list(str)
    :param starts: Alternative starting values, each a dict parameter name
        -> value applied on top of `parameters`; the candidate is fitted
        from `parameters` and from each of these, and the best fit counts
    :type starts: list(dict)
    """

    description: str
    components: list
    parameters: dict
    significance: float = np.nan
    remove: list = field(default_factory=list)
    starts: list = field(default_factory=list)


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


class GaussianLineRule(Rule):
    """Rule: Gaussian emission (or absorption) lines
    (`analysis.line_model_gaussian`).

    Candidates are placed where a matched filter on the residuals of the
    current fit finds the most significant unmodeled emission (absorption
    with `kind="absorption"`: a Gaussian with negative flux, an additive
    approximation valid for weak, resolved absorbers), for a few
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
    :param kind: "emission" or "absorption". Do not combine absorption
        lines with sigma clipping of negative outliers (`clip` with `low`),
        which removes the absorbers before they can be modeled.
    :type kind: str
    :param function: Line profile (default `analysis.line_model_gaussian`)
        with the arguments `redsh`, `flux`, `fwhm_km_s` and optionally `cen`
        (fixed at the line in `lines`; a function without `cen`, e.g. a
        doublet, needs `lines`) plus any in `fixed`; linear in `flux`
    :type function: callable
    :param fixed: Fixed values of further arguments, e.g.
        `{"fluxratio": 2.98}` for the [OIII] doublet
    :type fixed: dict
    :param shared: Argument -> name of a parameter shared by the
        components this rule adds (and by other rules naming the same
        parameter), e.g. `{"redsh": "redsh_narrow"}`; created with the first
        component that uses it (its starting value and prior), dropped with
        the last
    :type shared: dict
    :param n_shared: How many of the rule's components use the `shared`
        parameters (None: all); further ones get their own, e.g. 1 to tie
        only a line's core and leave an outflow component free
    :type n_shared: int
    :param start_fwhm_factors: Extra starting widths of a candidate,
        relative to the detection width (clipped to `fwhm_range`), with the
        flux re-estimated for each; the best fit counts. () for one start.
    :type start_fwhm_factors: tuple(float)
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
        kind="emission",
        function=None,
        fixed=None,
        shared=None,
        n_shared=None,
        start_fwhm_factors=(1 / 3, 3.0),
    ):
        if kind not in ("emission", "absorption"):
            raise ValueError(f"kind must be 'emission' or 'absorption', not {kind!r}")
        self.kind = kind
        self.sign = 1.0 if kind == "emission" else -1.0
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
        self.shared = dict(shared or {})
        self.n_shared = n_shared
        self.start_fwhm_factors = tuple(start_fwhm_factors)
        args = FitComponent("check", self.function).function_args
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
        resid = self.sign * (spec.fluxden - model.eval(x, result.values))
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

        # Shared parameters: map the argument onto one parameter for the
        # first `n_shared` of the rule's components, created with the first
        # component (of any rule) that uses it
        owned = set(self.owned(model))
        for arg, shared in self.shared.items():
            users = sum(
                c.name in owned and c.param_mapping.get(arg) == shared
                for c in model.components
            )
            if self.n_shared is not None and users >= self.n_shared:
                continue
            own = pars.pop(f"{name}_{arg}")
            comp.param_mapping[arg] = shared
            if shared not in model.parameters:
                own.name = shared
                if own.prior is not None:
                    own.prior.name = shared
                pars[shared] = own

        flux_key = f"{name}_flux"
        pars[flux_key].value = self.sign
        amp = _linear_amplitude(model, result, comp, pars, flux_key)
        if amp is None or self.sign * amp <= 0:
            return None
        pars[flux_key].value = amp
        bounds = sorted([0.0, 100 * amp])
        pars[flux_key].prior = UniformPrior(flux_key, *bounds)
        return Candidate(
            description=f"add {self.kind} {name} at {where}, FWHM {fwhm:.0f} km/s "
            f"({signif:.1f} sigma)",
            components=[comp],
            parameters=pars,
            significance=float(signif),
            starts=self._starts(model, result, comp, pars, rest, obs, fwhm, others),
        )

    def _starts(self, model, result, comp, pars, rest, obs, fwhm, others):
        """Alternative starts: the other detections (`others`, with
        `lines`: their redshift and width) and `start_fwhm_factors` times
        the width, each with its least-squares flux. Parameters shared with
        components already in the model are not the candidate's to choose
        and keep their values; near-duplicate starts are dropped."""
        keys = {arg: comp.param_mapping[arg] for arg in ("redsh", "fwhm_km_s", "flux")}
        free = {arg: key in pars and pars[key].vary for arg, key in keys.items()}
        prior = pars[keys["flux"]].prior
        variants = [(o, w) for _, o, w, _ in others]
        variants += [(obs, factor * fwhm) for factor in self.start_fwhm_factors]

        starts, tried = [], [(obs, fwhm)]
        for at, width in variants:
            start = {}
            if free["fwhm_km_s"]:
                width = float(np.clip(width, *self.fwhm_range))
                start[keys["fwhm_km_s"]] = width
            if free["redsh"] and self.lines is not None:
                start[keys["redsh"]] = float(at / rest - 1)
            if not start or any(
                abs(np.log(width / w)) < 0.1
                and abs(at - o) / o * scut.c_km_s < width / 4
                for o, w in tried
            ):
                continue
            trial = copy.deepcopy(pars)
            for key, value in start.items():
                trial[key].value = value
            trial[keys["flux"]].value = self.sign
            amp = _linear_amplitude(model, result, comp, trial, keys["flux"])
            if amp is None or self.sign * amp <= 0:
                continue
            tried.append((at, width))
            start[keys["flux"]] = float(np.clip(amp, prior.low, prior.upp))
            starts.append(start)
        return starts


def _linear_amplitude(model, result, comp, params, key):
    """Least-squares amplitude of a new component on the residuals of the
    current fit, for a component linear in parameter `key` (evaluated with
    `key` = 1); None if it has no flux in the fit."""
    mask = model.get_fit_mask()
    x = model.spec.dispersion
    lookup = {name: p.value for name, p in model.parameters.items()}
    lookup.update(dict(zip(result.param_names, result.values, strict=True)))
    lookup.update({name: p.value for name, p in params.items()})
    lookup[key] = 1.0
    unit = comp.eval(x, [lookup[k] for k in comp.get_arg_keys()])[mask]
    resid = (model.spec.fluxden - model.eval(x, result.values))[mask]
    weight = 1 / model.spec.fluxden_err[mask] ** 2
    denom = np.sum(weight * unit**2)
    if not denom > 0:
        return None
    return float(np.sum(weight * unit * resid) / denom)


class ComponentRule(Rule):
    """Rule: add a predefined component (or several), or swap one for
    another, e.g. an iron template, a Balmer continuum, the next Gaussian of
    an emission line, or a different continuum model.

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
    ask for the component, and nothing is proposed.

    :param name: Name of the rule (for the log)
    :type name: str
    :param factory: Builds the components, see above
    :type factory: callable
    :param replaces: Names of components the new ones replace (removed in
        the same step, e.g. the old continuum)
    :type replaces: list(str)
    :param max_uses: How many components the rule may add (one per use)
    :type max_uses: int
    :param linear_param: Parameter (without the component prefix) that
        scales the new component linearly
    :type linear_param: str
    :param prior_factor: Upper prior bound in units of the estimated
        amplitude, with `linear_param`
    :type prior_factor: float
    :param starts: Alternative starting values, each a dict parameter name
        -> value (see `Candidate.starts`); with `linear_param`, its value is
        re-estimated for each
    :type starts: list(dict)
    """

    def __init__(
        self,
        name,
        factory,
        replaces=(),
        max_uses=1,
        linear_param=None,
        prior_factor=100.0,
        starts=(),
    ):
        self.name = name
        self.factory = factory
        self.replaces = list(replaces)
        self.max_uses = max_uses
        self.linear_param = linear_param
        self.prior_factor = prior_factor
        self.starts = [dict(start) for start in starts]
        self._produced = set()

    def owned(self, model):
        """Names of the components of `model` this rule added.

        :param model: The model
        :type model: sculptor.model.FitModel
        :rtype: list(str)
        """
        return [c.name for c in model.components if c.name in self._produced]

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
        if self.replaces and not any(c.name in self.replaces for c in model.components):
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
        if self.replaces:
            description += f" (replacing {', '.join(self.replaces)})"
        if self.linear_param is not None and not self._scale(
            model, result, comps, params
        ):
            return []
        return [
            Candidate(
                description=description,
                components=list(comps),
                parameters=dict(params),
                remove=[n for n in self.replaces if n in names],
                starts=self._starts(model, result, comps, params),
            )
        ]

    def _starts(self, model, result, comps, params):
        """`starts`, with `linear_param` re-estimated for each; starts the
        residuals do not ask for are dropped."""
        if self.linear_param is None:
            return [dict(start) for start in self.starts]
        key = comps[0].param_mapping[self.linear_param]
        prior = params[key].prior
        starts = []
        for start in self.starts:
            trial = copy.deepcopy(params)
            for name, value in start.items():
                trial[name].value = value
            trial[key].value = 1.0
            amp = _linear_amplitude(model, result, comps[0], trial, key)
            if amp is None or not amp > 0:
                continue
            starts.append({**start, key: float(np.clip(amp, prior.low, prior.upp))})
        return starts

    def _scale(self, model, result, comps, params):
        """Least-squares starting value for `linear_param`; False if the
        residuals do not ask for the component."""
        if len(comps) != 1:
            raise ValueError("linear_param needs a factory with one component")
        comp = comps[0]
        key = comp.param_mapping[self.linear_param]
        amp = _linear_amplitude(model, result, comp, params, key)
        if amp is None or not amp > 0:
            return False
        params[key].value = amp
        params[key].prior = UniformPrior(key, 0.0, self.prior_factor * amp)
        return True


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


class Tie(Rule):
    """Hypothesis that components of several rules share parameters, e.g.
    the narrow cores of Hbeta and [OIII] share redshift and width.

    It applies to the narrowest component of each rule `<prefix>_<n>` in
    `prefixes` whose `width_arg` is at most `max_width`, if at least
    `min_components` rules have one. `AutoFit` tests it when its search
    converges, like a removal: the tied model is kept unless the untied one
    earns the threshold (see the module docstring). The shared parameters
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
    :param max_width: Largest `width_arg` of a tied component (e.g. the
        narrow-line FWHM limit), or None for any
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
                if self.max_width is None or width <= self.max_width:
                    widths[comp.name] = width
            if widths:
                chosen.append(min(widths, key=widths.get))
            else:
                missing.append(prefix)
        if len(chosen) < self.min_components:
            limit = "" if self.max_width is None else f" <= {self.max_width:g}"
            return None, (
                f"no component with {self.width_arg}{limit} for {', '.join(missing)}"
            )
        for name in chosen:
            comp = next(c for c in model.components if c.name == name)
            for arg in self.args:
                key = comp.param_mapping.get(arg)
                if key is None or not model.parameters[key].vary:
                    return None, f"{name} has no variable {arg}"
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
            priors = [new.parameters[k].prior for k in keys]
            if all(isinstance(p, UniformPrior) for p in priors):
                low = max(p.low for p in priors)
                upp = min(p.upp for p in priors)
                if low < upp:
                    param.prior = UniformPrior(shared, low, upp)
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
        the calibration simulated
    :type preliminary: AutoFitResult
    :param ties: Tie name -> outcome of its last test
    :type ties: dict(str, TieTest)
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
    ties: dict = field(default_factory=dict)

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
        if self.calibration is not None:
            lines += ["", str(self.calibration), "", "Models by quantile:"]
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
                lines.append(text)
        return "\n".join(lines)


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
        `FitModel.sigma_clip` plus `max_iterations` (default 5), e.g.
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
    :type calibration: dict
    :param preliminary_threshold: With `threshold="calibrate"`: fixed
        threshold of the preliminary search
    :type preliminary_threshold: float
    :param at_bound: "flag" (default): parameters at a prior bound are
        listed in `AutoFitResult.at_bound`; "reject": in addition, a
        candidate whose own parameters end at a bound is not added
    :type at_bound: str
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
        preliminary_threshold=10.0,
        at_bound="flag",
    ):
        if isinstance(threshold, str) and threshold != "calibrate":
            raise ValueError(
                f"threshold must be a number or 'calibrate', not {threshold!r}"
            )
        if at_bound not in ("flag", "reject"):
            raise ValueError(f"at_bound must be 'flag' or 'reject', not {at_bound!r}")
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
        self.ties = [rule for rule in self.rules if isinstance(rule, Tie)]
        self.threshold = threshold
        self.removable = set(removable)
        self.quick_fit_kwargs = dict(quick_fit_kwargs or {})
        self.clip = None if clip is None else dict(clip)
        self.clip_after = clip_after
        self.multistart = multistart
        self.calibration = dict(calibration or {})
        self.preliminary_threshold = preliminary_threshold
        self.at_bound = at_bound
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

        :return: trial model, its quick fit and score, of the best start
            (quick fit None and score inf if all fits failed)
        """
        starts = [{}] + (candidate.starts if self.multistart else [])
        best = (None, None, np.inf)
        for idx, start in enumerate(starts):
            trial = self._with(model, candidate)
            for name, value in start.items():
                trial.parameters[name].value = value
            trial_result, trial_score = self._fit(trial)
            if best[0] is None:
                best = (trial, trial_result, trial_score)
            elif trial_result is not None and (
                best[1] is None or trial_score < best[2]
            ):
                best = (trial, trial_result, trial_score)
                logger.debug(
                    f"AutoFit: {candidate.description}: start {idx} {start} "
                    f"is better ({trial_score:.2f})"
                )
        return best

    def _removable(self, model):
        """Components that may be removed."""
        added = set()
        for rule in self.rules:
            added.update(rule.owned(model))
        return [c.name for c in model.components if c.name in added | self.removable]

    @staticmethod
    def _with(model, candidate):
        """Copy of `model` with the candidate's removals and additions."""
        new = copy.deepcopy(model)
        for name in candidate.remove:
            new = AutoFit._without(new, name, copy_model=False)
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

    def run(self, max_iterations=20):
        """Build the model: one search with a fixed threshold, or with
        `threshold="calibrate"` a preliminary search, the calibration and a
        search per quantile.

        :param max_iterations: Maximum number of accepted additions per
            search
        :type max_iterations: int
        :return: The selected model, its quick fit and the history (and the
            calibration and the models of the other quantiles)
        :rtype: AutoFitResult
        :raises ValueError: if the base model cannot be fitted
        """
        self.n_fits = 0
        if self.threshold != "calibrate":
            return self._search(self.threshold, max_iterations)

        preliminary = self._search(self.preliminary_threshold, max_iterations)
        calibration = calibrate_threshold(
            preliminary.model,
            self.rules,
            criterion=self.criterion_arg,
            quick_fit_kwargs=self.quick_fit_kwargs,
            multistart=self.multistart,
            **self.calibration,
        )
        self.n_fits += calibration.n_fits
        by_quantile, by_threshold = {}, {}
        for quantile, threshold in calibration.thresholds.items():
            if threshold not in by_threshold:
                by_threshold[threshold] = self._search(threshold, max_iterations)
            found = copy.copy(by_threshold[threshold])
            found.quantile = quantile
            found.calibration = calibration
            found.preliminary = preliminary
            found.by_quantile = by_quantile
            by_quantile[quantile] = found
        selected = next(iter(by_quantile.values()))
        selected.n_fits = self.n_fits
        return selected

    def _search(self, threshold, max_iterations):
        """One greedy search with a fixed threshold."""
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

        for _ in range(max_iterations):
            best = None
            for rule in self.rules:
                for cand in rule.propose(model, result):
                    if _signature(self._with(model, cand)) in seen:
                        continue
                    trial, trial_result, trial_score = self._fit_candidate(model, cand)
                    logger.debug(
                        f"AutoFit: {cand.description}: "
                        f"{self.criterion_name} {trial_score - score:+.2f}"
                    )
                    if trial_result is None:
                        continue
                    if self.at_bound == "reject":
                        own = set(cand.parameters) & set(trial_result.at_bound)
                        if own:
                            logger.debug(
                                f"AutoFit: {cand.description}: rejected, "
                                f"{', '.join(sorted(own))} at a prior bound"
                            )
                            continue
                    if best is None or trial_score < best[2]:
                        best = (trial, trial_result, trial_score, cand)

            if best is None or score - best[2] <= threshold:
                # Converged: a tie the data accept simplifies the model, and
                # the search goes on from there
                model, result, score, tied = self._test_ties(
                    model, result, score, history, seen, threshold, ties
                )
                if tied:
                    continue
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

            model, result, score = self._backward(
                model, result, score, history, seen, threshold
            )

            n_added = sum(step.action == "add" for step in history)
            if self.clip is not None and n_added >= self.clip_after:
                model, result, score = self._clip(model, result, score, history)

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
        )

    def _test_ties(self, model, result, score, history, seen, threshold, ties):
        """Test the applicable ties (see `Tie`); apply the first one the data
        accept. Outcomes are written into `ties`.

        :return: model, result, score, and whether a tie was applied
        """
        for tie in self.ties:
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
            if trial_result is None or delta >= threshold:
                ties[tie.name] = TieTest("rejected", names, delta)
                logger.info(f"AutoFit: tie {tie.name} rejected: {delta:+.2f}")
                continue
            ties[tie.name] = TieTest("tied", names, delta)
            history.append(
                AutoFitStep(
                    "tie",
                    f"tie {', '.join(names)} ({', '.join(tie.args)})",
                    trial_score,
                    delta,
                    [c.name for c in trial.components],
                )
            )
            logger.info(f"AutoFit: tie {tie.name} ({', '.join(names)}): {delta:+.2f}")
            seen.add(_signature(trial))
            return trial, trial_result, trial_score, True
        return model, result, score, False

    def _clip(self, model, result, score, history):
        """Sigma-clip with the current model and refit, until the mask no
        longer changes; record a "clip" step if it changed."""
        clip = dict(self.clip)
        max_iterations = clip.pop("max_iterations", 5)
        changed = False
        before = score
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
        if changed:
            n_clipped = int((model._fit_region_mask() & ~model.get_fit_mask()).sum())
            history.append(
                AutoFitStep(
                    "clip",
                    f"sigma clipping: {n_clipped} pixels clipped",
                    score,
                    score - before,
                    [c.name for c in model.components],
                )
            )
            logger.info(f"AutoFit: {n_clipped} pixels clipped, re-scored")
        return model, result, score

    def _backward(self, model, result, score, history, seen, threshold):
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
                if trial_score - score < threshold and (
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
    :param thresholds: Quantile -> threshold
    :type thresholds: dict
    :param intervals: Quantile -> (low, high): 90% bootstrap interval of the
        threshold, its uncertainty from the finite number of simulations
    :type intervals: dict
    :param noise: Noise model of the simulations
    :type noise: str
    :param n_fits: Number of quick fits run
    :type n_fits: int
    """

    criterion: str
    improvements: np.ndarray
    thresholds: dict
    intervals: dict
    noise: str = "residuals"
    n_fits: int = 0

    def threshold(self, quantile):
        """The threshold at `quantile` (computed if it was not requested).

        :param quantile: Quantile, e.g. 0.95
        :type quantile: float
        :rtype: float
        """
        if quantile in self.thresholds:
            return self.thresholds[quantile]
        return float(np.quantile(self.improvements, quantile))

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
            lines.append(
                f"  {quantile:.1%} quantile: threshold {value:.1f} "
                f"(90% bootstrap interval {low:.1f} - {high:.1f})"
            )
        return "\n".join(lines)


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
    lie above the threshold).

    Use a model of the real features, e.g. from a search with a fixed
    threshold (as `AutoFit` does with `threshold="calibrate"`): residuals
    that still contain real features make the thresholds conservative.

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
    :return: The improvements and the thresholds
    :rtype: Calibration
    :raises ValueError: for an unknown `noise`, quantiles outside (0, 1), or
        a model that cannot be fitted
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

    null = copy.deepcopy(model)
    null_result, _ = search._fit(null)
    if null_result is None:
        raise ValueError("calibrate_threshold: the model cannot be fitted.")
    best_flux = null.eval(null.spec.dispersion, null_result.values)

    improvements = np.full(n_sims, np.nan)
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
                _, _, score = search._fit_candidate(sim, cand)
                best = min(best, score)
        improvements[idx] = sim_score - best
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
        boot = np.quantile(resampled, quantile, axis=1)
        intervals[quantile] = tuple(float(v) for v in np.quantile(boot, [0.05, 0.95]))
        if improvements.size * (1 - quantile) < 3:
            logger.warning(
                f"calibrate_threshold: only {improvements.size * (1 - quantile):.1f}"
                f" simulations above the {quantile:.1%} quantile; use more "
                "simulations for a stable threshold"
            )

    calibration = Calibration(
        search.criterion_name,
        improvements,
        thresholds,
        intervals,
        noise=noise,
        n_fits=search.n_fits,
    )
    logger.info(str(calibration))
    return calibration
