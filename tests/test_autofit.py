"""Tests for the automatic model building (`sculptor.autofit`)."""

import numpy as np
import pytest
from astropy import units as u
from speconed import speconed as sod

from sculptor import analysis as ana
from sculptor import component as sccomp
from sculptor import model as scmod
from sculptor import prior as scpri
from sculptor.autofit import (
    AutoFit,
    GaussianLineRule,
    calibrate_threshold,
    correlated_noise,
)

Z = 2.0
CIV, CIII, HEII = 1549.06, 1908.73, 1640.42
# (flux, rest-frame center, FWHM km/s): broad + narrow CIV, CIII]
LINES = [(60.0, CIV, 5000.0), (20.0, CIV, 1200.0), (15.0, CIII, 3000.0)]
ERR = 0.05


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
    for criterion in ("aicc", "aic", lambda r: r.bic):
        result = AutoFit(_model(), [_rule()], criterion=criterion).run()
        assert len(result.model.components) >= 4

    with pytest.raises(ValueError, match="Unknown criterion"):
        AutoFit(_model(), [_rule()], criterion="spic")


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
    white, white_imp = calibrate_threshold(_model(lines=[]), [_rule()], n_sims=5)
    assert white_imp.shape == (5,)
    assert np.all(white_imp >= 0)

    rng = np.random.default_rng(1)
    raw = np.convolve(rng.normal(0, 1, 820), np.ones(5), "same")[10:-10]
    corr_model = _model(lines=[], noise=raw * ERR / raw.std())
    corr, _ = calibrate_threshold(corr_model, [_rule()], n_sims=5)
    assert corr > white
    assert corr > 10

    with pytest.raises(ValueError, match="Unknown noise"):
        calibrate_threshold(corr_model, [_rule()], noise="pink")
