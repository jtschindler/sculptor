"""Tests for the automatic model building (`sculptor.autofit`)."""

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


def test_criteria():
    for criterion in ("aicc", "aic", lambda r, m: r.bic):
        result = AutoFit(_model(), [_rule()], criterion=criterion).run()
        assert len(result.model.components) >= 4

    with pytest.raises(ValueError, match="Unknown criterion"):
        AutoFit(_model(), [_rule()], criterion="nope")


def test_a_high_threshold_keeps_the_base_model():
    result = AutoFit(_model(), [_rule()], threshold=1e9).run()
    assert len(result.model.components) == 1


def test_correlated_noise_keeps_the_residual_correlations():
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


def test_clipping_keeps_absorbers_from_becoming_components():
    base = _with_absorbers(_model())

    plain = AutoFit(base, [_rule()]).run()
    clipped = AutoFit(base, [_rule()], clip={"low": 3, "grow": 2}).run()

    assert len(plain.model.components) > 4  # absorbers modeled as lines
    assert len(clipped.model.components) == 4
    np.testing.assert_allclose(_centers(clipped.model), [CIV, CIV, CIII], atol=2)
    assert 0.85 < clipped.result.red_chisq < 1.15
    actions = [s.action for s in clipped.history]
    assert actions.index("clip") == 3  # after clip_after=2 additions

    x = clipped.model.spec.dispersion
    fitted = clipped.model.get_fit_mask()
    for cen, _ in ABSORBERS:
        assert not fitted[np.argmin(np.abs(x - cen))]
    assert base.clip_mask is None  # the base model is not changed


def test_clip_after_delays_clipping():
    base = _with_absorbers(_model())
    result = AutoFit(base, [_rule()], clip={"low": 3, "grow": 2}, clip_after=3).run()
    actions = [s.action for s in result.history]
    assert actions.index("clip") > 3


# -- rules: components, swaps, absorption; SpIC -------------------------------


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


def curved_power_law(x, redsh, amp, slope, curv):
    """Power law with a log-parabolic curvature."""
    logx = np.log(x / 5000.0)
    return amp * np.exp((slope + curv * logx) * logx)


def test_component_rule_swaps_the_continuum():
    """A curved continuum: swapping the power law for a curved one is
    accepted, and the old power law and its parameters are gone."""

    model = _model(lines=[], noise=np.zeros(800))
    x = model.spec.dispersion
    model.spec.fluxden = curved_power_law(x, Z, 2.0, -1.5, -3.0) + (
        np.random.default_rng(3).normal(0, ERR, x.size)
    )

    def factory(model):
        comp = sccomp.FitComponent("curvedPL", curved_power_law)
        pars = comp.create_params()
        pars["curvedPL_redsh"].value, pars["curvedPL_redsh"].vary = Z, False
        pars["curvedPL_amp"].value = 1.0
        pars["curvedPL_amp"].prior = scpri.UniformPrior("curvedPL_amp", 0, 10)
        pars["curvedPL_slope"].value = -1.0
        pars["curvedPL_slope"].prior = scpri.UniformPrior("curvedPL_slope", -4, 2)
        pars["curvedPL_curv"].value = 0.0
        pars["curvedPL_curv"].prior = scpri.UniformPrior("curvedPL_curv", -10, 10)
        return comp, pars

    rule = ComponentRule("curvature", factory, replaces=["PL"])
    result = AutoFit(model, [rule]).run()

    assert [c.name for c in result.model.components] == ["curvedPL"]
    assert not any(k.startswith("PL_") for k in result.model.parameters)
    assert "replacing PL" in result.history[1].description
    values = dict(zip(result.result.param_names, result.result.values, strict=True))
    assert values["curvedPL_curv"] == pytest.approx(-3.0, abs=0.3)

    # Nothing left to replace: no proposal
    assert rule.propose(result.model, result.result) == []


def test_absorption_rule():
    base = _model(lines=[(-8.0, 1800.0, 400.0)])
    rule = GaussianLineRule(
        "abs", redshift=Z, fwhm_range=(100, 2000), kind="absorption"
    )
    result = AutoFit(base, [rule]).run()

    assert len(result.model.components) == 2
    par = result.model.parameters
    assert par["abs_0_flux"].value == pytest.approx(-8.0, rel=0.15)
    assert par["abs_0_flux"].prior.upp == 0
    assert par["abs_0_cen"].value == pytest.approx(1800.0, abs=1)

    with pytest.raises(ValueError, match="kind"):
        GaussianLineRule("x", redshift=Z, kind="both")


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


def test_correlated_noise_keeps_the_residual_amplitude():
    model = _model(lines=[], noise=np.random.default_rng(4).normal(0, 4 * ERR, 800))
    result = model.run_quick_fit()
    noise = correlated_noise(model, result, np.random.default_rng(5))
    mask = model.get_fit_mask()
    assert np.std(noise[mask]) == pytest.approx(4 * ERR, rel=0.1)


# -- line profiles, fixed and shared parameters ---------------------------------


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


def test_shared_redshift():
    """Two lines at the same velocity offset: one shared redshift."""

    z_true = Z * 1.0005
    lines = [(15.0, CIII, 2000.0), (20.0, CIV, 2000.0)]
    base = _model(lines=[])
    x = base.spec.dispersion
    for flux, cen, fwhm in lines:
        base.spec.fluxden = base.spec.fluxden + ana.line_model_gaussian(
            x, z_true, flux, cen, fwhm
        )
    rules = [
        GaussianLineRule(
            name,
            redshift=Z,
            lines=[cen],
            fwhm_range=(300, 5000),
            max_components=1,
            shared={"redsh": "z_lines"},
        )
        for name, cen in (("CIII", CIII), ("CIV", CIV))
    ]
    result = AutoFit(base, rules).run()

    names = sorted(c.name for c in result.model.components)
    assert names == ["CIII_0", "CIV_0", "PL"]
    par = result.model.parameters
    assert "CIII_0_redsh" not in par and "CIV_0_redsh" not in par
    assert par["z_lines"].vary
    assert par["z_lines"].value == pytest.approx(z_true, abs=2e-4)
    for comp in result.model.components[1:]:
        assert comp.param_mapping["redsh"] == "z_lines"

    # Removing one user keeps the shared parameter; removing both drops it
    one = AutoFit._without(result.model, "CIV_0")
    assert "z_lines" in one.parameters
    none = AutoFit._without(one, "CIII_0")
    assert "z_lines" not in none.parameters


def test_rule_owns_only_its_numbered_components():
    model = _model(lines=[])
    for name in ("Hbeta_0", "Hbeta_narrow_0", "Hbeta_x"):
        model.components.append(sccomp.FitComponent(name, ana.line_model_gaussian))
    assert GaussianLineRule("Hbeta", redshift=Z).owned(model) == ["Hbeta_0"]


def test_n_shared_ties_only_the_first_components():
    """A core and a shifted wing of one line: with n_shared=1 only the
    core uses the shared redshift."""

    z_wing = Z - 3000 / 3e5 * (1 + Z)
    base = _model(lines=[])
    x = base.spec.dispersion
    base.spec.fluxden = (
        base.spec.fluxden
        + ana.line_model_gaussian(x, Z, 20.0, CIII, 1500.0)
        + ana.line_model_gaussian(x, z_wing, 10.0, CIII, 4000.0)
    )
    rule = GaussianLineRule(
        "CIII",
        redshift=Z,
        lines=[CIII],
        fwhm_range=(300, 8000),
        velocity_range=5000,
        shared={"redsh": "z_core"},
        n_shared=1,
    )
    result = AutoFit(base, [rule]).run()

    par = result.model.parameters
    comps = {c.name: c for c in result.model.components}
    assert sorted(comps) == ["CIII_0", "CIII_1", "PL"]
    shared = [n for n, c in comps.items() if c.param_mapping.get("redsh") == "z_core"]
    assert len(shared) == 1
    (own,) = {"CIII_0", "CIII_1"} - set(shared)
    assert par["z_core"].value == pytest.approx(Z, abs=3e-4)
    assert par[f"{own}_redsh"].value == pytest.approx(z_wing, abs=3e-3)
    assert par[f"{own}_redsh"].value < par["z_core"].value - 3e-3


# -- multi-start ----------------------------------------------------------------


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


def test_calibration_reports_every_quantile():
    calibration = calibrate_threshold(
        _model(lines=[]), [_rule()], n_sims=20, quantiles=(0.5, 0.8, 0.95)
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
