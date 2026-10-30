"""Tests for the automatic model building (`sculptor.autofit`)."""

import copy
import logging
from types import SimpleNamespace

import numpy as np
import pytest
from astropy import units as u
from speconed import speconed as sod

from sculptor import analysis as ana
from sculptor import autofit
from sculptor import component as sccomp
from sculptor import model as scmod
from sculptor import prior as scpri
from sculptor.autofit import (
    ComponentRule,
    GaussianLineRule,
    calibrate_threshold,
    correlated_noise,
    spic,
)

Z = 2.0
CIV, CIII, HEII = 1549.06, 1908.73, 1640.42
# (flux, rest-frame center, FWHM km/s): broad + narrow CIV, CIII]
LINES = [(60.0, CIV, 5000.0), (20.0, CIV, 1200.0), (15.0, CIII, 3000.0)]
ERR = 0.05


class AutoFit(autofit.AutoFit):
    """AutoFit with the fixed threshold 10: these tests check the search,
    not the calibration (see the calibration tests)."""

    def __init__(self, *args, threshold=10.0, **kwargs):
        super().__init__(*args, threshold=threshold, **kwargs)


def _continuum(x):
    return ana.power_law_at2500(x, Z, 1.0, -1.5)


def _model(lines=LINES, seed=0, n=800, noise=None):
    """Power-law base model on a synthetic spectrum with `lines`."""

    x = np.geomspace(4200, 6200, n)
    rng = np.random.default_rng(seed)
    flux = _continuum(x)
    for line in lines:
        flux = flux + ana.line_model_gaussian(x, Z, *line)
    flux = flux + (rng.normal(0, ERR, n) if noise is None else noise)
    spec = sod.SpecOneD(
        dispersion=x,
        fluxden=flux,
        fluxden_err=np.full_like(x, ERR),
        dispersion_unit=u.AA,
        fluxden_unit=u.erg / u.s / u.cm**2 / u.AA,
    )
    comp = sccomp.FitComponent("PL", ana.power_law_at2500)
    pars = comp.create_params()
    pars["PL_redsh"].value, pars["PL_redsh"].vary = Z, False
    pars["PL_amp"].value = 0.8
    pars["PL_amp"].prior = scpri.UniformPrior("PL_amp", 0, 10)
    pars["PL_slope"].value = -1.0
    pars["PL_slope"].prior = scpri.UniformPrior("PL_slope", -3, 1)
    model = scmod.FitModel(components=[comp], parameters=pars, spectrum=spec)
    model.add_wavelength_range_to_fit_mask(4250, 6150)
    return model


def _rule(**kwargs):
    return GaussianLineRule("line", redshift=Z, fwhm_range=(300, 15000), **kwargs)


def _centers(model):
    """Rest-frame centers of the added Gaussians, sorted."""
    return sorted(
        model.parameters[f"{c.name}_cen"].value
        for c in model.components
        if c.name.startswith("line_")
    )


def test_matched_filter_uses_the_local_pixel_scale():
    """A grid with 80 km/s pixels in the blue and 180 km/s in the red (a
    coadd of two gratings): a line in the red gets the right amplitude."""
    blue = 4200 * np.exp(np.arange(2000) * 80 / 2.998e5)
    red = blue[-1] * np.exp(np.arange(1, 400) * 180 / 2.998e5)
    x = np.concatenate([blue, red])
    model = _model(lines=[], n=x.size)
    cen = 0.5 * (red[100] + red[300]) / (1 + Z)
    line = ana.line_model_gaussian(x, Z, 50.0, cen, 3000.0)
    model.spec = sod.SpecOneD(
        dispersion=x,
        fluxden=_continuum(x) + line,
        fluxden_err=np.full_like(x, ERR),
        dispersion_unit=u.AA,
        fluxden_unit=u.erg / u.s / u.cm**2 / u.AA,
    )
    model.reset_fit_mask()
    model.add_wavelength_range_to_fit_mask(x[0], x[-1])
    model.parameters["PL_amp"].value, model.parameters["PL_slope"].value = 1.0, -1.5
    result = model.run_quick_fit()
    signif, amp = _rule().significance_map(model, result, 3000.0)
    peak = np.argmax(signif)
    assert x[peak] == pytest.approx(cen * (1 + Z), rel=1e-3)
    assert amp[peak] == pytest.approx(line.max(), rel=0.1)


def test_matched_filter_kernel_longer_than_the_data():
    """A kernel wider than the spectrum gives one value per pixel, the
    same as the direct sum (np.convolve "same" would return the kernel's
    length)."""
    x = np.geomspace(5000.0, 5100.0, 50)
    resid = np.zeros(x.size)
    resid[25] = 1.0
    weight = np.ones(x.size)
    signif, amp = autofit.matched_filter(x, resid, weight, 20000.0)
    assert signif.shape == amp.shape == x.shape
    # Direct sum at the spike: amp = K(0) / sum K^2 over the pixels
    fwhm_pix = 20000.0 / 2.998e5 / np.log(x[1] / x[0])
    sigma = fwhm_pix * autofit._FWHM_TO_SIGMA
    kernel = np.exp(-0.5 * ((np.arange(x.size) - 25) / sigma) ** 2)
    assert amp[25] == pytest.approx(1.0 / np.sum(kernel**2), rel=1e-3)


def test_masked_nan_pixel_next_to_a_line():
    """A NaN pixel masked in the spectrum does not hide a line beside it
    (0 weight x NaN residual spread NaN over the filter kernel)."""
    base = _model(lines=[(15.0, CIII, 3000.0)])
    x = base.spec.dispersion
    near = np.argmin(np.abs(x - CIII * (1 + Z))) + 15
    base.spec.fluxden[near] = np.nan
    base.spec.mask[near] = False

    signif, _ = _rule().significance_map(base, base.run_quick_fit(), 3000.0)
    assert np.all(np.isfinite(signif))
    result = AutoFit(base, [_rule()]).run()
    assert _centers(result.model) == pytest.approx([CIII], abs=2)


def test_finds_the_input_lines():
    base = _model()
    result = AutoFit(base, [_rule()]).run()

    assert result.model.components[0].name == "PL"
    assert len(result.model.components) == 4
    np.testing.assert_allclose(_centers(result.model), [CIV, CIV, CIII], atol=2)
    fwhms = sorted(
        result.model.parameters[f"{c.name}_fwhm_km_s"].value
        for c in result.model.components[1:]
    )
    np.testing.assert_allclose(fwhms, [1200, 3000, 5000], rtol=0.15)
    assert 0.85 < result.result.red_chisq < 1.15
    assert result.converged
    assert result.model.quick_fit_result is result.result
    assert [s.action for s in result.history] == ["start", "add", "add", "add"]
    assert all(s.delta < -10 for s in result.history[1:])
    assert "line_0" in str(result)

    # The base model is not changed
    assert [c.name for c in base.components] == ["PL"]
    assert base.parameters["PL_amp"].value == 0.8


def test_adds_nothing_to_a_line_free_spectrum():
    for seed in range(3):
        result = AutoFit(_model(lines=[], seed=seed), [_rule()]).run()
        assert [c.name for c in result.model.components] == ["PL"]
        assert len(result.history) == 1


def test_known_lines_mode():
    """With a line list only those lines are tried, at fixed centers."""

    rule = _rule(lines=[CIV, HEII, CIII], velocity_range=2000)
    result = AutoFit(_model(), [rule]).run()

    cens = [
        result.model.parameters[f"{c.name}_cen"] for c in result.model.components[1:]
    ]
    assert all(not p.vary for p in cens)
    assert sorted(p.value for p in cens) == [CIV, CIV, CIII]  # no HeII
    for comp in result.model.components[1:]:
        assert result.model.parameters[f"{comp.name}_redsh"].value == pytest.approx(
            Z, abs=0.01
        )


def test_window_and_max_components():
    result = AutoFit(_model(), [_rule(window=(5500, 6000))]).run()
    assert _centers(result.model) == pytest.approx([CIII], abs=2)

    result = AutoFit(_model(), [_rule(max_components=1)]).run()
    assert len(result.model.components) == 2


def test_removes_a_component_that_does_not_earn_the_threshold():
    base = _model(lines=[(15.0, CIII, 3000.0)])
    spurious = sccomp.FitComponent("fake", ana.line_model_gaussian)
    pars = spurious.create_params()
    pars["fake_redsh"].value, pars["fake_redsh"].vary = Z, False
    pars["fake_cen"].value, pars["fake_cen"].vary = 1700.0, False
    pars["fake_flux"].value = 1.0
    pars["fake_flux"].prior = scpri.UniformPrior("fake_flux", 0, 100)
    pars["fake_fwhm_km_s"].value = 2000
    pars["fake_fwhm_km_s"].prior = scpri.UniformPrior("fake_fwhm_km_s", 300, 15000)
    base.add_component(spurious)
    base.add_parameters(pars)

    kept = AutoFit(base, [_rule()]).run()
    assert "fake" in [c.name for c in kept.model.components]

    result = AutoFit(base, [_rule()], removable=["fake"]).run()
    names = [c.name for c in result.model.components]
    assert "fake" not in names
    assert not any(k.startswith("fake_") for k in result.model.parameters)
    assert "remove" in [s.action for s in result.history]
    assert _centers(result.model) == pytest.approx([CIII], abs=2)

    # max_iterations counts every change, the removal too
    n = autofit._n_changes(result.history)
    assert [s.action for s in result.history] == ["start", "add", "remove"]
    search = AutoFit(base, [_rule()], removable=["fake"])
    assert search.run(max_iterations=n).converged
    short = search.run(max_iterations=n - 1)
    assert not short.converged
    assert autofit._n_changes(short.history) == n - 1
    assert "fake" in [c.name for c in short.model.components]


def test_criteria():
    for criterion in ("aicc", "aic", lambda r, m: r.bic):
        result = AutoFit(_model(), [_rule()], criterion=criterion).run()
        assert len(result.model.components) >= 4

    with pytest.raises(ValueError, match="Unknown criterion"):
        AutoFit(_model(), [_rule()], criterion="nope")


def test_correlated_noise_keeps_the_residual_correlations_and_amplitude():
    width = 5
    rng = np.random.default_rng(1)
    raw = np.convolve(rng.normal(0, 1, 820), np.ones(width), "same")[10:-10]
    model = _model(lines=[], noise=raw * ERR / raw.std())
    result = model.run_quick_fit()

    noise = correlated_noise(model, result, np.random.default_rng(2))
    mask = model.get_fit_mask()
    norm = noise[mask] / ERR

    def lag1(series):
        series = series - series.mean()
        return np.sum(series[1:] * series[:-1]) / np.sum(series**2)

    assert np.std(norm) == pytest.approx(1, rel=0.05)
    resid = (model.spec.fluxden - model.eval(model.spec.dispersion, result.values))[
        mask
    ]
    assert lag1(norm) == pytest.approx(lag1(resid), abs=0.1)
    assert lag1(norm) > 0.5

    # White noise 4x the errors: the simulated noise is 4x too
    model = _model(lines=[], noise=np.random.default_rng(4).normal(0, 4 * ERR, 800))
    result = model.run_quick_fit()
    noise = correlated_noise(model, result, np.random.default_rng(5))
    mask = model.get_fit_mask()
    assert np.std(noise[mask]) == pytest.approx(4 * ERR, rel=0.1)


def test_calibrate_threshold_grows_with_correlated_noise():
    white = calibrate_threshold(_model(lines=[]), [_rule()], n_sims=5)
    assert white.improvements.shape == (5,)
    assert np.all(white.improvements >= 0)
    assert list(white.thresholds) == [0.95, 0.99]

    rng = np.random.default_rng(1)
    raw = np.convolve(rng.normal(0, 1, 820), np.ones(5), "same")[10:-10]
    corr_model = _model(lines=[], noise=raw * ERR / raw.std())
    corr = calibrate_threshold(corr_model, [_rule()], n_sims=5, quantiles=[0.9])
    assert list(corr.thresholds) == [0.9]
    assert corr.threshold(0.9) > white.threshold(0.9)
    assert corr.threshold(0.9) > 10

    with pytest.raises(ValueError, match="Unknown noise"):
        calibrate_threshold(corr_model, [_rule()], noise="pink")


ABSORBERS = [(4600, 0.9), (4660, 0.6), (5200, 0.8)]  # observed center, depth


def _with_absorbers(model):
    x = model.spec.dispersion
    for cen, depth in ABSORBERS:
        model.spec.fluxden = model.spec.fluxden * (
            1 - depth * np.exp(-0.5 * ((x - cen) / 1.5) ** 2)
        )
    return model


CLIP = {"low": 3, "grow": 2}


def _absorbers_masked(model):
    x = model.spec.dispersion
    fitted = model.get_fit_mask()
    return not any(fitted[np.argmin(np.abs(x - cen))] for cen, _ in ABSORBERS)


def test_clipping_keeps_absorbers_from_becoming_components(caplog):
    base = _with_absorbers(_model())

    plain = AutoFit(base, [_rule()]).run()
    assert len(plain.model.components) > 4  # absorbers modeled as lines
    assert all(s.action != "clip" for s in plain.history)

    fit = AutoFit(base, [_rule()], clip=CLIP)
    clipped = fit.run()
    assert len(clipped.model.components) == 4
    np.testing.assert_allclose(_centers(clipped.model), [CIV, CIV, CIII], atol=2)
    assert 0.85 < clipped.result.red_chisq < 1.15
    actions = [s.action for s in clipped.history]
    assert actions.index("clip") == 3  # after clip_after=2 additions
    assert _absorbers_masked(clipped.model)
    assert base.clip_mask is None  # the base model is not changed

    # The final clip against the final model: here it changes nothing, and
    # says so
    history = []
    with caplog.at_level(logging.INFO, logger="sculptor.autofit"):
        fit._clip(clipped.model, clipped.result, 0.0, history, final=True)
    assert history == []
    assert "final sigma clipping: mask unchanged" in caplog.text

    # clip_after delays the clipping
    later = AutoFit(base, [_rule()], clip=CLIP, clip_after=3).run()
    assert [s.action for s in later.history].index("clip") > 3

    # The search ends with a clip against the final model, even when the
    # clipping during the search never started
    never = AutoFit(base, [_rule()], clip=CLIP, clip_after=99).run()
    last = never.history[-1]
    assert last.action == "clip"
    assert last.description.startswith("final sigma clipping")
    assert [s.action for s in never.history].count("clip") == 1
    assert _absorbers_masked(never.model)


def _with_mgii_absorber(model, z_abs=0.9):
    """A MgII doublet absorber (unresolved, 60% / 40% deep) in the
    continuum."""
    x = model.spec.dispersion
    for rest, depth in ((2796.354, 0.6), (2803.531, 0.4)):
        cen = rest * (1 + z_abs)
        sigma = 150.0 / 2.998e5 * cen / 2.3548
        model.spec.fluxden = model.spec.fluxden * (
            1 - depth * np.exp(-0.5 * ((x - cen) / sigma) ** 2)
        )
    return model


@pytest.mark.parametrize("threshold", [10.0, "calibrate"])
def test_absorbers_are_reported_and_masked(threshold):
    base = _with_mgii_absorber(_model(n=2500))
    fit = autofit.AutoFit(
        base,
        [_rule()],
        threshold=threshold,
        calibration={"n_sims": 5, "quantiles": [0.9]},
        absorbers={"redshift": Z, "name": "QSO", "n_sims": 30},
    )
    result = fit.run()
    inventory = result.absorbers
    assert list(inventory.systems["species"]) == ["MgII"]
    assert inventory.systems["z_abs"][0] == pytest.approx(0.9, abs=3e-4)
    assert not result.model.get_fit_mask()[~inventory.mask].any()
    np.testing.assert_allclose(_centers(result.model), [CIV, CIV, CIII], atol=2)
    assert "MgII" in str(result)
    assert fit.base is base and base.absorber_mask is None


def test_tail_threshold_and_rule_keys():
    rng = np.random.default_rng(3)
    values = rng.exponential(10.0, 400)
    # Beyond the sample: exponential extrapolation, P(> t) = exp(-t / 10)
    assert autofit.tail_threshold(values, 0.999) == pytest.approx(69.1, rel=0.2)
    assert autofit.tail_threshold(values, 0.5) == pytest.approx(np.median(values))
    # 95% zeros (a rule that improves nothing in most simulations), 5%
    # exponential with mean 5: P(> t) = 0.05 exp(-t / 5)
    tied = np.concatenate([np.zeros(3800), rng.exponential(5.0, 200)])
    assert autofit.tail_threshold(tied, 0.99) == pytest.approx(5 * np.log(5), rel=0.2)
    assert autofit.tail_threshold(tied, 0.9) == 0.0  # within the zeros
    rules = [_rule(), _rule(), GaussianLineRule("other", redshift=Z)]
    assert autofit.rule_keys(rules) == ["line#0", "line#1", "other"]
    # A tie shares its name with the group it tests; it gets its own key
    tie = autofit.Tie("other", ["line"])
    assert autofit.rule_keys([rules[2], tie]) == ["other", "tie other"]


def test_search_uses_the_rule_thresholds():
    """A huge global threshold, but the rule's own lets its lines in (and
    keeps them in the backward steps)."""
    fit = AutoFit(_model(), [_rule()])
    found = fit._search(1e9, 20, rule_thresholds={"line": 10.0})
    assert len(found.model.components) == 4
    assert found.rule_thresholds == {"line": 10.0}
    assert len(fit._search(1e9, 20).model.components) == 1


def test_calibration_per_rule():
    # min_significance 0: the rules propose candidates on noise too
    narrow = GaussianLineRule(
        "narrow",
        redshift=Z,
        lines=[CIV],
        fwhm_range=(300, 1500),
        max_components=4,
        min_significance=0.0,
    )
    free = _rule(max_components=5, min_significance=0.0)
    found = AutoFit(_model(), [_rule()]).run()
    # Chi-square: no penalty, so every candidate's improvement shows
    cal = autofit.calibrate_threshold(
        found.model,
        [free, narrow],
        criterion=lambda result, model: result.chi_sq,
        n_sims=25,
        quantiles=[0.9],
        min_rule_sims=10,
    )
    assert set(cal.by_rule) == {"line", "narrow"}
    # The free Gaussian fits noise better than the one at a fixed line
    assert cal.by_rule["narrow"][0.9] < cal.by_rule["line"][0.9]
    for values in cal.by_rule.values():
        assert np.isfinite(values[0.9]) and values[0.9] >= 0
    assert "per rule" in str(cal)
    assert cal.rule_thresholds(0.9) == {k: v[0.9] for k, v in cal.by_rule.items()}
    plain = autofit.calibrate_threshold(found.model, [free], n_sims=5, per_rule=False)
    assert plain.by_rule == {}


def test_calibration_rule_floor_and_bound_rejection(monkeypatch):
    free = _rule(max_components=2, min_significance=0.0)
    narrow = GaussianLineRule(
        "narrow", redshift=Z, lines=[CIV], fwhm_range=(300, 1500), min_significance=0.0
    )

    def calibrate(**kwargs):
        # Chi-square: no penalty, every accepted candidate improves
        return autofit.calibrate_threshold(
            _model(lines=[]),
            [free, narrow],
            criterion=lambda result, model: result.chi_sq,
            n_sims=4,
            quantiles=[0.9],
            min_rule_sims=2,
            **kwargs,
        )

    assert calibrate().rule_improvements["narrow"].min() > 0
    # The search would reject every narrow candidate: they improve nothing,
    # but the simulations still count
    original = autofit.AutoFit._rejected_at_bound
    monkeypatch.setattr(
        autofit.AutoFit,
        "_rejected_at_bound",
        lambda self, cand, result: (
            "narrow" in cand.description or original(self, cand, result)
        ),
    )
    cal = calibrate()
    assert np.all(cal.rule_improvements["narrow"] == 0)
    assert cal.rule_improvements["narrow"].size == 4
    # A threshold of 0, raised to the floor (the global one's too)
    assert cal.thresholds[0.9] > autofit.MIN_THRESHOLD
    assert cal.by_rule["narrow"][0.9] == autofit.MIN_THRESHOLD
    assert calibrate(min_threshold=0.0).by_rule["narrow"][0.9] == 0.0


def _clip_with(monkeypatch, masks):
    """A clip whose sigma_clip sets the given masks in turn (always
    "changed"); returns the warnings and the number of calls."""
    fit = AutoFit(_model(), [_rule()], clip={"low": 3})
    model = copy.deepcopy(fit.base)
    result, score = fit._fit(model)
    calls = []

    def fake(values=None, **kwargs):
        model.clip_mask = masks(len(calls), model.spec.dispersion.size).copy()
        calls.append(1)
        return True

    monkeypatch.setattr(model, "sigma_clip", fake)
    return fit, model, result, score, calls


def test_clip_warns_when_the_mask_oscillates(monkeypatch, caplog):
    def alternate(i, n):
        mask = np.ones(n, bool)
        mask[10 if i % 2 == 0 else 20] = False
        return mask

    fit, model, result, score, calls = _clip_with(monkeypatch, alternate)
    with caplog.at_level(logging.WARNING, logger="sculptor.autofit"):
        fit._clip(model, result, score, [])
    assert "oscillates" in caplog.text
    assert len(calls) == 3  # a, b, a again


def test_clip_warns_when_it_does_not_converge(monkeypatch, caplog):
    def fresh(i, n):
        mask = np.ones(n, bool)
        mask[i] = False
        return mask

    fit, model, result, score, calls = _clip_with(monkeypatch, fresh)
    with caplog.at_level(logging.WARNING, logger="sculptor.autofit"):
        fit._clip(model, result, score, [])
    assert "did not converge in 10 passes" in caplog.text
    assert len(calls) == 10


# -- component rules; SpIC -----------------------------------------------------


def _line_factory(name, cen, fwhm=3000.0):
    """Factory for a fixed-center Gaussian named `name`."""

    def factory(model):
        comp = sccomp.FitComponent(name, ana.line_model_gaussian)
        pars = comp.create_params()
        pars[f"{name}_redsh"].value, pars[f"{name}_redsh"].vary = Z, False
        pars[f"{name}_cen"].value, pars[f"{name}_cen"].vary = cen, False
        pars[f"{name}_flux"].value = 1.0
        pars[f"{name}_fwhm_km_s"].value = fwhm
        pars[f"{name}_fwhm_km_s"].prior = scpri.UniformPrior(
            f"{name}_fwhm_km_s", 300, 15000
        )
        return comp, pars

    return factory


def test_component_rule_scales_and_adds_once():
    rule = ComponentRule("CIII]", _line_factory("CIII_x", CIII), linear_param="flux")
    base = _model(lines=[(15.0, CIII, 3000.0)])
    result = base.run_quick_fit()

    (cand,) = rule.propose(base, result)
    flux = cand.parameters["CIII_x_flux"]
    assert flux.value == pytest.approx(15.0, rel=0.1)  # least squares
    assert flux.prior.upp == pytest.approx(100 * flux.value)

    final = AutoFit(base, [rule]).run()
    assert [c.name for c in final.model.components] == ["PL", "CIII_x"]
    assert rule.owned(final.model) == ["CIII_x"]
    assert rule.propose(final.model, final.result) == []  # already there


def test_linear_amplitude_with_the_resolution():
    """With a resolution the least-squares flux is fitted with the broadened
    template: a 300 km/s line seen at R = 1000 (300 km/s), i.e. 424 km/s
    wide. The unbroadened template gave ~18% too little flux."""
    base = _model(lines=[(15.0, CIII, np.hypot(300.0, 2.998e5 / 1000))])
    base.resolution = 1000.0
    result = base.run_quick_fit()
    rule = ComponentRule(
        "CIII]", _line_factory("CIII_x", CIII, fwhm=300.0), linear_param="flux"
    )
    (cand,) = rule.propose(base, result)
    assert cand.parameters["CIII_x_flux"].value == pytest.approx(15.0, rel=0.05)


def test_component_rule_calibrated_without_its_component():
    """A component the preliminary model already has: its rule proposes
    nothing on that null, so it is calibrated on simulations without it
    and gets its own threshold."""
    rule = ComponentRule("CIII]", _line_factory("CIII_x", CIII), linear_param="flux")
    found = AutoFit(_model(lines=[(15.0, CIII, 3000.0)]), [rule]).run().model
    assert rule.null_removals(found) == ["CIII_x"]
    kwargs = {"n_sims": 8, "quantiles": [0.9], "min_rule_sims": 4}
    cal = calibrate_threshold(found, [rule], **kwargs)
    # Proposed in the simulations where the noise has positive flux there
    assert cal.rule_improvements["CIII]"].size >= 4
    assert "CIII]" in cal.by_rule
    assert calibrate_threshold(found, [rule], per_rule=False, **kwargs).by_rule == {}


def test_component_rule_skips_components_the_data_do_not_ask_for():
    rule = ComponentRule("HeII", _line_factory("HeII_x", HEII), linear_param="flux")
    base = _model(lines=[])
    result = AutoFit(base, [rule]).run()
    assert [c.name for c in result.model.components] == ["PL"]


def test_component_rule_max_uses_and_none():
    names = iter([("a", 5000.0), ("b", 1200.0), ("c", 3000.0)])

    def factory(model):
        nxt = next(names, None)
        if nxt is None:
            return None
        return _line_factory(f"CIV_{nxt[0]}", CIV, fwhm=nxt[1])(model)

    rule = ComponentRule("CIV", factory, max_uses=2, linear_param="flux")
    result = AutoFit(_model(), [rule]).run()
    assert [c.name for c in result.model.components] == ["PL", "CIV_a", "CIV_b"]


def test_spic():
    model = _model()
    result = AutoFit(model, [_rule()], criterion="spic").run()
    assert len(result.model.components) == 4
    assert result.criterion == "spic"

    fit = result.result
    value = spic(fit, result.model)
    assert value == pytest.approx(result.history[-1].score)
    assert value > fit.chi_sq
    # f = 1 (AICc-like) penalizes more than f = 0 (BIC-like) for strong lines
    assert spic(fit, result.model, f=1.0) != spic(fit, result.model, f=0.0)
    # No lines: SpIC of the base model is chi^2 plus the continuum's penalty
    base = _model(lines=[])
    base_fit = base.run_quick_fit()
    assert spic(base_fit, base) > base_fit.chi_sq


# -- line profiles, fixed and shared parameters --------------------------------


def doublet(x, redsh, flux, fwhm_km_s, fluxratio):
    """Two Gaussians at 1800 and 1850 A, the second `fluxratio` x brighter."""
    return ana.line_model_gaussian(
        x, redsh, flux, 1800.0, fwhm_km_s
    ) + ana.line_model_gaussian(x, redsh, fluxratio * flux, 1850.0, fwhm_km_s)


def _no_flux(x, redsh, amp, fwhm_km_s):
    """A profile without a `flux` argument."""
    return amp * np.ones_like(x)


def test_doublet_function_with_fixed_ratio():
    base = _model(lines=[(4.0, 1800.0, 1500.0), (12.0, 1850.0, 1500.0)])
    rule = GaussianLineRule(
        "D",
        redshift=Z,
        lines=[1850.0],
        fwhm_range=(300, 5000),
        function=doublet,
        fixed={"fluxratio": 3.0},
    )
    result = AutoFit(base, [rule]).run()

    assert [c.name for c in result.model.components] == ["PL", "D_0"]
    par = result.model.parameters
    assert not par["D_0_fluxratio"].vary
    assert par["D_0_flux"].value == pytest.approx(4.0, rel=0.1)
    assert par["D_0_fwhm_km_s"].value == pytest.approx(1500, rel=0.1)

    with pytest.raises(ValueError, match="needs `lines`"):
        GaussianLineRule("D", redshift=Z, function=doublet)
    with pytest.raises(ValueError, match="lacks the arguments"):
        GaussianLineRule("D", redshift=Z, lines=[1850.0], function=_no_flux)


def test_rule_owns_only_its_numbered_components():
    model = _model(lines=[])
    for name in ("Hbeta_0", "Hbeta_narrow_0", "Hbeta_x"):
        model.components.append(sccomp.FitComponent(name, ana.line_model_gaussian))
    assert GaussianLineRule("Hbeta", redshift=Z).owned(model) == ["Hbeta_0"]


# -- multi-start ---------------------------------------------------------------


def test_line_candidates_carry_alternative_widths():
    base = _model(lines=[(20.0, CIV, 3000.0)])
    result = base.run_quick_fit()
    rule = GaussianLineRule("line", redshift=Z, lines=[CIV], fwhm_range=(500, 15000))
    (cand,) = rule.propose(base, result)  # one candidate per line

    fwhm0 = cand.parameters["line_0_fwhm_km_s"].value
    widths = [start["line_0_fwhm_km_s"] for start in cand.starts]
    assert widths  # other detections and fwhm0 / 3, 3 fwhm0
    for start in cand.starts:
        assert set(start) == {"line_0_fwhm_km_s", "line_0_flux", "line_0_redsh"}
        assert 500 <= start["line_0_fwhm_km_s"] <= 15000
        assert start["line_0_flux"] > 0
    for factor in (1 / 3, 3):
        width = np.clip(factor * fwhm0, 500, 15000)
        assert any(w == pytest.approx(width) for w in widths)

    single = GaussianLineRule(
        "line", redshift=Z, lines=[CIV], fwhm_range=(500, 15000), start_fwhm_factors=()
    )
    (cand,) = single.propose(base, result)
    assert all(abs(np.log(s["line_0_fwhm_km_s"] / fwhm0)) > 0.1 for s in cand.starts)


def test_fit_candidate_keeps_the_best_start():
    base = _model(lines=[(20.0, CIV, 3000.0)])
    search = AutoFit(base, [_rule()])
    result = base.run_quick_fit()
    (cand, *_) = _rule(lines=[CIV]).propose(base, result)
    assert cand.starts

    scores = []
    for start in [{}, *cand.starts]:
        trial = AutoFit._with(base, cand)
        for name, value in start.items():
            trial.parameters[name].value = value
        scores.append(search._fit(trial)[1])

    trial, trial_result, score = search._fit_candidate(base, cand)
    assert score == pytest.approx(min(scores))
    assert trial_result is not None
    assert (
        trial.parameters["line_0_flux"].value
        == trial_result.values[trial_result.param_names.index("line_0_flux")]
    )

    one = AutoFit(base, [_rule()], multistart=False)
    assert one._fit_candidate(base, cand)[2] == pytest.approx(scores[0])


def test_fit_candidate_prefers_a_start_inside_the_bounds(monkeypatch):
    """The best-scoring start ends at a `reject_at_bound` bound, a worse
    one does not: the candidate is the worse one, not rejected."""
    base = _model(lines=[(20.0, CIV, 3000.0)])
    (cand, *_) = _rule(lines=[CIV]).propose(base, base.run_quick_fit())
    cand.reject_at_bound = ["line_0_fwhm_km_s"]
    assert len(cand.starts) >= 2
    search = AutoFit(base, [_rule()])
    # (at_bound, score) of the fits from the default start and starts 1, 2
    fits = iter([(["line_0_fwhm_km_s"], 1.0), ([], 5.0), ([], 3.0)])

    def fake_fit(model):
        at_bound, score = next(fits, (["line_0_fwhm_km_s"], 0.0))
        return SimpleNamespace(at_bound=at_bound), score

    monkeypatch.setattr(search, "_fit", fake_fit)
    _, result, score = search._fit_candidate(base, cand)
    assert score == 3.0
    assert not search._rejected_at_bound(cand, result)

    # Every start at the bound: the best of them, rejected by the caller
    fits = iter([])
    _, result, score = search._fit_candidate(base, cand)
    assert score == 0.0 and search._rejected_at_bound(cand, result)


def test_component_rule_starts_rescale_the_amplitude():
    base = _model(lines=[(20.0, CIII, 3000.0)])
    result = base.run_quick_fit()

    def factory(model):
        comp = sccomp.FitComponent("G", ana.line_model_gaussian)
        pars = comp.create_params()
        pars["G_redsh"].value, pars["G_redsh"].vary = Z, False
        pars["G_cen"].value, pars["G_cen"].vary = CIII, False
        pars["G_fwhm_km_s"].value = 1000.0
        pars["G_fwhm_km_s"].prior = scpri.UniformPrior("G_fwhm_km_s", 300, 9000)
        return comp, pars

    rule = ComponentRule(
        "G",
        factory,
        linear_param="flux",
        starts=[{"G_fwhm_km_s": 3000.0}, {"G_fwhm_km_s": 9000.0}],
    )
    (cand,) = rule.propose(base, result)
    assert [s["G_fwhm_km_s"] for s in cand.starts] == [3000.0, 9000.0]
    assert cand.starts[0]["G_flux"] == pytest.approx(20.0, rel=0.1)

    trial, _, _ = AutoFit(base, [rule])._fit_candidate(base, cand)
    assert trial.parameters["G_fwhm_km_s"].value == pytest.approx(3000, rel=0.1)


def test_signature_ignores_the_amplitude_priors():
    """The same line proposed on different residuals gets a different flux
    prior, but is the same model (J1340 round 22a revisited models)."""
    rule = _rule(lines=[CIV, CIII], max_components=1, start_fwhm_factors=())
    signatures = []
    for flux in (60.0, 20.0):
        base = _model(lines=[(flux, CIV, 3000.0)])
        cand = next(
            c
            for c in rule.propose(base, base.run_quick_fit())
            if "1549" in c.description
        )
        signatures.append(
            (
                cand.parameters["line_0_flux"].prior.upp,
                autofit._signature(AutoFit._with(base, cand)),
            )
        )
    (upp_a, sig_a), (upp_b, sig_b) = signatures
    assert upp_a != pytest.approx(upp_b, rel=0.1)
    assert sig_a == sig_b

    base = _model(lines=[(20.0, CIII, 3000.0)])
    ciii = next(
        c for c in rule.propose(base, base.run_quick_fit()) if "1908" in c.description
    )
    assert autofit._signature(AutoFit._with(base, ciii)) != sig_a


def test_search_reports_when_it_does_not_converge(caplog):
    with caplog.at_level(logging.WARNING, logger=autofit.logger.name):
        stopped = AutoFit(_model(), [_rule()]).run(max_iterations=1)
    assert not stopped.converged
    assert "not converged" in str(stopped) and "not converged" in caplog.text


def test_calibration_reports_every_quantile():
    calibration = calibrate_threshold(
        _model(lines=[]),
        [_rule()],
        n_sims=20,
        quantiles=(0.5, 0.8, 0.95),
        min_threshold=0.0,
    )
    imp = calibration.improvements
    assert list(calibration.thresholds) == [0.5, 0.8, 0.95]
    for quantile, value in calibration.thresholds.items():
        assert value == pytest.approx(np.quantile(imp, quantile))
        low, high = calibration.intervals[quantile]
        assert low <= high
    assert calibration.threshold(0.7) == pytest.approx(np.quantile(imp, 0.7))
    assert calibration.n_fits > 20
    text = str(calibration)
    assert "20 simulations" in text and "80.0% quantile" in text

    with pytest.raises(ValueError, match="quantiles"):
        calibrate_threshold(_model(lines=[]), [_rule()], quantiles=(95,))


def test_calibration_threshold_floor():
    # A rule that proposes nothing: noise improves nothing, quantiles of 0
    silent = _rule(min_significance=1e9)
    floored = calibrate_threshold(_model(lines=[]), [silent], n_sims=5)
    assert np.all(floored.improvements == 0)
    assert floored.thresholds == {0.95: 10.0, 0.99: 10.0}
    assert floored.threshold(0.5) == 10.0
    assert "the floor; quantile 0.0" in str(floored)

    raw = calibrate_threshold(_model(lines=[]), [silent], n_sims=5, min_threshold=0.0)
    assert raw.thresholds == {0.95: 0.0, 0.99: 0.0}
    assert "the floor" not in str(raw)


def test_calibrated_threshold_is_the_default():
    base = _model()
    search = autofit.AutoFit(
        base, [_rule()], calibration={"n_sims": 10, "quantiles": (0.6, 0.9)}
    )
    assert search.threshold == "calibrate"
    result = search.run()

    cal = result.calibration
    assert list(result.by_quantile) == [0.6, 0.9]
    assert result.by_quantile[0.6] is result
    assert result.quantile == 0.6
    assert result.threshold == cal.thresholds[0.6]
    assert result.by_quantile[0.9].threshold == cal.thresholds[0.9]
    assert result.preliminary.threshold == 10.0
    assert result.n_fits > cal.n_fits
    assert len(result.model.components) == 4  # the three input lines
    assert cal.criterion == "bic"
    text = str(result)
    assert "Models by quantile" in text and "60.0%" in text and "90.0%" in text

    with pytest.raises(ValueError, match="threshold must be"):
        autofit.AutoFit(base, [_rule()], threshold="auto")


def test_min_threshold_is_the_preliminary_threshold_and_the_floor():
    calibration = {"n_sims": 6, "quantiles": (0.9,), "min_rule_sims": 3}
    result = autofit.AutoFit(
        _model(), [_rule()], calibration=calibration, min_threshold=40.0
    ).run()
    assert result.preliminary.threshold == 40.0
    assert result.calibration.min_threshold == 40.0
    assert result.threshold >= 40.0
    assert all(v[0.9] >= 40.0 for v in result.calibration.by_rule.values())
    with pytest.raises(ValueError, match="min_threshold"):
        autofit.AutoFit(_model(), [_rule()], calibration={"min_threshold": 5.0})


def test_an_improvement_equal_to_the_threshold_earns_it():
    """A component worth exactly the threshold is added, and the backward
    step keeps it."""
    fit = AutoFit(_model(lines=[(15.0, CIII, 3000.0)]), [_rule(max_components=1)])
    gain = -fit.run().history[1].delta
    found = fit._search(gain, 20)
    assert [step.action for step in found.history] == ["start", "add"]
    assert fit._search(np.nextafter(gain, np.inf), 20).history[-1].action == "start"


def _record_nulls(monkeypatch, noise_models=None):
    """Component names of every null model `calibrate_threshold` gets (and
    of its noise models, into `noise_models`)."""
    nulls, original = [], autofit.calibrate_threshold

    def recording(model, *args, **kwargs):
        nulls.append(autofit._component_names(model))
        if noise_models is not None:
            noise_models.append(autofit._component_names(kwargs["noise_model"]))
        return original(model, *args, **kwargs)

    monkeypatch.setattr(autofit, "calibrate_threshold", recording)
    return nulls


def test_recalibration_on_the_selected_model(monkeypatch, caplog):
    """A preliminary search that takes everything (threshold -inf) overfits;
    the calibration is repeated with the model the first quantile selects
    as the null, and the noise of the preliminary model."""
    noise_models = []
    nulls = _record_nulls(monkeypatch, noise_models)
    calibration = {"n_sims": 10, "quantiles": (0.9,)}
    with caplog.at_level(logging.INFO, logger=autofit.logger.name):
        result = autofit.AutoFit(
            _model(),
            [_rule(max_components=6, min_significance=0.0)],
            calibration=calibration,
            min_threshold=-np.inf,
            recalibrate=1,
        ).run()
    preliminary = autofit._component_names(result.preliminary.model)
    assert len(preliminary) > len(result.model.components)
    assert len(result.calibrations) == 2 == len(nulls)
    assert nulls[0] == preliminary != nulls[1]
    assert noise_models == [preliminary, preliminary]
    assert result.calibration is result.calibrations[-1]
    assert "recalibrated 1 time(s)" in str(result)
    # Announced once: no round after the last allowed one
    assert caplog.text.count("recalibrating on") == 1

    # recalibrate=0 (default): the preliminary model only
    nulls.clear()
    once = autofit.AutoFit(
        _model(),
        [_rule(max_components=6, min_significance=0.0)],
        calibration=calibration,
        min_threshold=-np.inf,
    ).run()
    assert len(once.calibrations) == 1 == len(nulls)
    assert "recalibrated" not in str(once)


def test_calibration_noise_from_another_model():
    """A null without the lines counts their misfit as noise; the residuals
    of the model with the lines give the noise of the data."""
    data = _model(lines=[(15.0, CIII, 3000.0)])
    rich = AutoFit(data, [_rule(lines=[CIII])]).run().model
    assert len(rich.components) == 2
    kwargs = {"n_sims": 20, "quantiles": (0.9,), "per_rule": False}

    own = calibrate_threshold(data, [_rule()], **kwargs)
    cleaner = calibrate_threshold(data, [_rule()], noise_model=rich, **kwargs)
    assert cleaner.thresholds[0.9] < own.thresholds[0.9]
    same = calibrate_threshold(data, [_rule()], noise_model=data, **kwargs)
    assert same.thresholds == own.thresholds


def test_recalibration_stops_when_the_model_does_not_change(monkeypatch):
    """The preliminary search already finds the selected model: no second
    calibration, however many are allowed."""
    nulls = _record_nulls(monkeypatch)
    result = autofit.AutoFit(
        _model(),
        [_rule()],
        calibration={"n_sims": 10, "quantiles": (0.9,)},
        recalibrate=3,
    ).run()
    assert nulls == [autofit._component_names(result.model)]
    assert len(result.calibrations) == 1


# -- prior bounds --------------------------------------------------------------


def test_parameters_at_a_prior_bound_are_flagged():
    """A 4000 km/s line with widths limited to 1500 km/s."""
    base = _model(lines=[(20.0, CIII, 4000.0)])
    narrow = GaussianLineRule(
        "line", redshift=Z, lines=[CIII], fwhm_range=(300, 1500), max_components=1
    )
    flagged = AutoFit(base, [narrow]).run()
    assert "line_0_fwhm_km_s" in flagged.at_bound
    assert "at a prior bound: " in str(flagged)
