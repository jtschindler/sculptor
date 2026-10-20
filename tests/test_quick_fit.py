"""Tests for the quick fit (`FitModel.run_quick_fit`, `sculptor.quick_fit`)
and the `FitModel.fit` dispatcher."""

import logging

import numpy as np
import pytest
from astropy import units as u
from speconed import speconed as sod

from sculptor import analysis as ana
from sculptor import component as sccomp
from sculptor import model as scmod
from sculptor import prior as scpri
from sculptor.quick_fit import QuickFitResult

TRUTH = {
    "pl_amp": 0.02,
    "pl_slope": -1.5,
    "SiIV_a_flux": 6.0,
    "SiIV_a_fwhm_km_s": 1200,
}


def _perturb(model):
    """Start away from the truth the synthetic spectrum was made with."""
    model.parameters["pl_amp"].value = 0.05
    model.parameters["SiIV_a_flux"].value = 3.0
    model.parameters["SiIV_a_fwhm_km_s"].value = 3000


def linear(x, a, b):
    """Straight line a * x + b."""
    return a * x + b


def _linear_model(prior_a=None, prior_b=None, seed=1):
    x = np.linspace(0, 10, 50)
    rng = np.random.default_rng(seed)
    err = np.full_like(x, 0.5)
    spec = sod.SpecOneD(
        dispersion=x,
        fluxden=linear(x, 2.0, 1.0) + rng.normal(0, 0.5, x.size),
        fluxden_err=err,
        dispersion_unit=u.AA,
        fluxden_unit=u.erg / u.s / u.cm**2 / u.AA,
    )
    comp = sccomp.FitComponent("lin", linear)
    pars = comp.create_params()
    pars["lin_a"].value, pars["lin_a"].prior = 1.0, prior_a
    pars["lin_b"].value, pars["lin_b"].prior = 0.0, prior_b
    model = scmod.FitModel(components=[comp], parameters=pars, spectrum=spec)
    model.add_wavelength_range_to_fit_mask(0, 10)
    return model, x, err


def test_recovers_the_input_parameters(fit_model):
    _perturb(fit_model)
    result = fit_model.run_quick_fit()

    assert isinstance(result, QuickFitResult)
    assert fit_model.quick_fit_result is result
    assert result.success
    assert result.param_names == list(TRUTH)
    assert result.at_bound == []
    for name, (value, error) in result.params_dict().items():
        assert error > 0
        assert abs(value - TRUTH[name]) < 4 * error, name
    assert 0.8 < result.red_chisq < 1.2


def test_linear_model_matches_weighted_least_squares():
    """For a linear model without priors the MAP is the weighted
    least-squares solution and the Gauss-Newton covariance is exact."""

    model, x, err = _linear_model()
    result = model.run_quick_fit()

    design = np.vstack([x, np.ones_like(x)]).T / err[:, None]
    coef, *_ = np.linalg.lstsq(design, model.spec.fluxden / err, rcond=None)
    cov = np.linalg.inv(design.T @ design)

    np.testing.assert_allclose(result.values, coef, rtol=1e-5)
    np.testing.assert_allclose(result.covariance, cov, rtol=1e-4)
    assert result.dof == x.size - 2
    assert result.n_data == x.size


def test_information_criteria():
    model, x, _ = _linear_model()
    result = model.run_quick_fit()

    k, n = 2, x.size
    assert result.bic == pytest.approx(k * np.log(n) - 2 * result.log_likelihood)
    assert result.aic == pytest.approx(2 * k - 2 * result.log_likelihood)
    assert result.aicc == pytest.approx(result.aic + 2 * k * (k + 1) / (n - k - 1))


def test_uses_the_mcmc_log_probability(fit_model):
    """The stored log-probability is the one emcee samples."""

    result = fit_model.run_quick_fit()
    spec = fit_model.spec
    expected = scmod.log_probability(
        result.values, spec.dispersion, fit_model, spec.fluxden, spec.fluxden_err
    )
    assert result.log_prob == pytest.approx(expected)
    assert result.log_likelihood == pytest.approx(
        scmod.log_likelihood_chi_square(
            result.values, fit_model, spec.dispersion, spec.fluxden, spec.fluxden_err
        )
    )


def test_gaussian_prior_enters_value_and_uncertainty():
    """A strong Gaussian prior pulls the fit and shrinks the uncertainty."""

    free, *_ = _linear_model()
    free_result = free.run_quick_fit()

    prior = scpri.GaussianPrior("lin_b", 5.0, 0.01)
    tied, *_ = _linear_model(prior_b=prior)
    tied_result = tied.run_quick_fit()

    assert tied_result.values[1] == pytest.approx(5.0, abs=0.05)
    assert tied_result.errors[1] < 0.01
    assert tied_result.errors[1] < free_result.errors[1]


def test_uniform_prior_bound_is_respected_and_flagged(caplog):
    model, *_ = _linear_model(prior_a=scpri.UniformPrior("lin_a", 0, 1.5))
    with caplog.at_level(logging.WARNING):
        result = model.run_quick_fit()

    assert result.values[0] <= 1.5
    assert result.values[0] == pytest.approx(1.5)
    assert result.at_bound == ["lin_a"]
    assert "prior bound" in caplog.text


def test_start_outside_the_prior_is_moved_inside(caplog):
    model, *_ = _linear_model(prior_a=scpri.UniformPrior("lin_a", 1.5, 3))
    model.parameters["lin_a"].value = 10.0
    with caplog.at_level(logging.WARNING):
        result = model.run_quick_fit()

    assert "outside its prior" in caplog.text
    assert result.success
    assert result.values[0] == pytest.approx(2.0, abs=0.2)


def test_fixed_parameters_stay_fixed(fit_model):
    fit_model.parameters["pl_slope"].vary = False
    fit_model.parameters["pl_slope"].value = -1.5
    result = fit_model.run_quick_fit()

    assert "pl_slope" not in result.param_names
    assert fit_model.parameters["pl_slope"].value == -1.5


def test_update_values(fit_model):
    _perturb(fit_model)
    fit_model.run_quick_fit()
    assert fit_model.parameters["SiIV_a_flux"].value == 3.0

    result = fit_model.run_quick_fit(update_values=True)
    for name, value in zip(result.param_names, result.values, strict=True):
        assert fit_model.parameters[name].value == value


def test_requires_values_and_variable_parameters():
    model, *_ = _linear_model()
    model.parameters["lin_a"].value = None
    with pytest.raises(ValueError, match="without a starting value"):
        model.run_quick_fit()

    model.parameters["lin_a"].vary = False
    model.parameters["lin_b"].vary = False
    with pytest.raises(ValueError, match="no variable parameters"):
        model.run_quick_fit()


def test_spec_with_other_pixels_raises(fit_model):
    other = fit_model.spec.copy()
    other.dispersion = other.dispersion[:-1]
    other.fluxden = other.fluxden[:-1]
    other.fluxden_err = other.fluxden_err[:-1]
    with pytest.raises(ValueError, match="pixels"):
        fit_model.run_quick_fit(spec=other)


def test_positions_around_quick_fit(fit_model):
    with pytest.raises(ValueError, match="run_quick_fit"):
        fit_model.positions_around_quick_fit(8)

    result = fit_model.run_quick_fit()
    pos = fit_model.positions_around_quick_fit(32, spread=1.0)

    assert pos.shape == (32, len(result.param_names))
    np.testing.assert_allclose(pos.mean(axis=0), result.values, atol=0, rtol=0.1)
    for idx, name in enumerate(result.param_names):
        prior = fit_model.parameters[name].prior
        if isinstance(prior, scpri.UniformPrior):
            assert np.all((pos[:, idx] >= prior.low) & (pos[:, idx] <= prior.upp))

    # A changed set of variable parameters invalidates the result.
    fit_model.parameters["pl_slope"].vary = False
    with pytest.raises(ValueError, match="current variable parameters"):
        fit_model.positions_around_quick_fit(8)


def test_positions_stay_inside_a_bound():
    model, *_ = _linear_model(prior_a=scpri.UniformPrior("lin_a", 0, 1.5))
    model.run_quick_fit()
    pos = model.positions_around_quick_fit(200, spread=5.0)
    assert np.all(pos[:, 0] <= 1.5)
    assert np.all(pos[:, 0] >= 0)


def test_fit_dispatcher(fit_model):
    result = fit_model.fit("quick")
    assert fit_model.quick_fit_result is result

    assert fit_model.fit("mcmc", nsteps=5, nwalkers=10, progress=False) is None
    assert fit_model.sampler.iteration == 5

    with pytest.raises(ValueError, match="Unknown fit method"):
        fit_model.fit("lmfit")


def test_quick_then_mcmc_starts_at_the_best_fit(fit_model):
    _perturb(fit_model)
    fit_model.fit("quick+mcmc", nsteps=3, nwalkers=10, progress=False)

    best = fit_model.quick_fit_result
    first = fit_model.sampler.get_chain()[0]
    spread = np.abs(first - best.values) / best.errors
    assert np.all(spread < 2)

    with pytest.raises(ValueError, match="sets `pos`"):
        fit_model.fit("quick+mcmc", nsteps=3, nwalkers=10, pos=first, progress=False)


def test_quick_fit_result_survives_save_and_load(fit_model, tmp_path):
    result = fit_model.run_quick_fit()
    fit_model.save("fit", tmp_path)

    loaded = scmod.FitModel()
    loaded.load("fit", tmp_path)
    np.testing.assert_array_equal(loaded.quick_fit_result.values, result.values)
    assert loaded.quick_fit_result.param_names == result.param_names


def test_analysis_runs_on_the_best_fit(fit_model):
    """The analysis functions take the best-fit vector like a sample."""

    result = fit_model.run_quick_fit()
    comp_pl, comp_line = fit_model.components
    table = ana.analyze_mcmc_results(
        fit_model,
        result.values[None, :],
        {"components": [comp_pl], "rest_frame_wavelengths": [1450.0]},
        [
            {
                "feature_name": "SiIV",
                "components": [comp_line],
                "rest_frame_wavelength": 1399.8,
            }
        ],
        fit_model.redshift,
        None,
    )
    assert len(table) == 1
    assert table["SiIV_FWHM"][0].to_value(u.km / u.s) == pytest.approx(
        result.params_dict()["SiIV_a_fwhm_km_s"][0], rel=0.05
    )


def test_str_lists_parameters(fit_model):
    text = str(fit_model.run_quick_fit())
    assert "converged" in text
    for name in TRUTH:
        assert name in text


def test_plot_quick_fit_result(fit_model, tmp_path):
    import matplotlib

    matplotlib.use("Agg")
    with pytest.raises(ValueError, match="run_quick_fit"):
        fit_model.plot_quick_fit_result(save=True, save_dir=tmp_path)

    result = fit_model.run_quick_fit()
    residuals, dispersion = fit_model.plot_quick_fit_result(
        save=True, save_dir=tmp_path, show_components=True, save_data=True
    )

    assert (tmp_path / "quick_fit_result.png").exists()
    assert (tmp_path / "quick_fit_model.csv").exists()
    assert (tmp_path / "quick_fit_model_pl.csv").exists()
    best = fit_model.eval(fit_model.spec.dispersion, result.values)
    np.testing.assert_allclose(residuals, fit_model.spec.fluxden - best)
    assert dispersion.size == residuals.size


@pytest.mark.parametrize("suffix", [".ecsv", ".json"])
def test_export_round_trip(fit_model, tmp_path, suffix):
    fit_model.parameters["pl_slope"].vary = False
    result = fit_model.run_quick_fit()
    path = tmp_path / f"fit{suffix}"
    result.write(path)

    back = QuickFitResult.read(path)
    assert back.param_names == result.param_names
    np.testing.assert_allclose(back.values, result.values, rtol=1e-12)
    np.testing.assert_allclose(back.errors, result.errors, rtol=1e-12)
    np.testing.assert_allclose(back.covariance, result.covariance, rtol=1e-12)
    for key in ("log_prob", "chi_sq", "bic", "aicc", "red_chisq"):
        assert getattr(back, key) == pytest.approx(getattr(result, key))
    assert back.dof == result.dof
    assert back.success == result.success
    assert back.fixed_params == {
        "pl_redsh": pytest.approx(3.227),
        "pl_slope": -1.5,
        "SiIV_a_redsh": pytest.approx(3.227),
        "SiIV_a_cen": pytest.approx(1399.8),
    }

    with pytest.raises(FileExistsError):
        result.write(path)
    result.write(path, overwrite=True)


def test_ecsv_export_is_a_readable_table(fit_model, tmp_path):
    from astropy.table import Table

    result = fit_model.run_quick_fit()
    result.write(tmp_path / "fit.ecsv")
    table = Table.read(tmp_path / "fit.ecsv")
    assert list(table["name"]) == result.param_names
    assert table.meta["aicc"] == pytest.approx(result.aicc)


def test_json_export_handles_nan(tmp_path):
    import json

    model, *_ = _linear_model()
    result = model.run_quick_fit()
    result.errors[:] = np.nan
    result.write(tmp_path / "fit.json")
    json.loads((tmp_path / "fit.json").read_text())  # valid JSON
    assert np.all(np.isnan(QuickFitResult.read(tmp_path / "fit.json").errors))


def test_export_unknown_extension(fit_model, tmp_path):
    result = fit_model.run_quick_fit()
    with pytest.raises(ValueError, match="extension"):
        result.write(tmp_path / "fit.txt")


def _absorbed(fit_model):
    """The fixture's spectrum with a narrow absorber on the continuum."""
    x = fit_model.spec.dispersion
    dip = np.abs(x - 6000) < 6
    fit_model.spec.fluxden = np.where(
        dip, fit_model.spec.fluxden - 0.03, fit_model.spec.fluxden
    )
    return dip


def test_sigma_clip_masks_and_restores(fit_model):
    dip = _absorbed(fit_model)
    result = fit_model.run_quick_fit()
    with pytest.raises(ValueError, match="run_quick_fit"):
        scmod.FitModel(spectrum=fit_model.spec).sigma_clip()

    assert fit_model.sigma_clip(low=3, grow=1)
    mask = fit_model.get_fit_mask()
    assert not np.any(mask[dip])
    assert mask.sum() < fit_model._fit_region_mask().sum()
    # Same input, same mask: no change
    assert not fit_model.sigma_clip(low=3, grow=1, values=result.values)

    # Recomputed from scratch: once the data are fixed, the pixels come back
    fit_model.spec.fluxden[dip] += 0.03
    fit_model.sigma_clip(low=3, grow=1)
    assert fit_model.get_fit_mask()[dip].all()

    fit_model.reset_clip_mask()
    assert fit_model.clip_mask is None


def test_sigma_clip_upper_and_max_fraction(fit_model, caplog):
    fit_model.spec.fluxden[200] += 0.1
    result = fit_model.run_quick_fit()
    fit_model.sigma_clip(low=None, high=None)
    assert fit_model.get_fit_mask()[200]
    fit_model.sigma_clip(low=None, high=5)
    assert not fit_model.get_fit_mask()[200]

    before = fit_model.clip_mask.copy()
    with caplog.at_level(logging.WARNING):
        assert not fit_model.sigma_clip(low=0, values=result.values)
    assert "keeping the previous" in caplog.text
    np.testing.assert_array_equal(fit_model.clip_mask, before)


def test_quick_fit_with_clipping(fit_model):
    dip = _absorbed(fit_model)
    plain = fit_model.run_quick_fit()
    clipped = fit_model.run_quick_fit(clip={"low": 3, "grow": 1})

    assert not np.any(fit_model.get_fit_mask()[dip])
    assert clipped.n_data < plain.n_data
    assert clipped.red_chisq < plain.red_chisq
    with pytest.raises(ValueError, match="clip"):
        fit_model.run_quick_fit(spec=fit_model.spec, clip={"low": 3})


def test_sigma_clip_scale_follows_underestimated_errors(fit_model):
    """Errors 4x too small: with scale="errors" ordinary noise is clipped,
    with the robust scales only the absorber."""

    dip = _absorbed(fit_model)
    fit_model.spec.fluxden_err = fit_model.spec.fluxden_err / 4
    values = fit_model.run_quick_fit().values
    region = fit_model._fit_region_mask().sum()

    fit_model.sigma_clip(low=3, scale="errors", values=values, max_fraction=1)
    n_errors = region - fit_model.get_fit_mask().sum()
    assert n_errors > 0.1 * region

    for scale in ("robust", "local"):
        fit_model.sigma_clip(low=3, scale=scale, values=values)
        clipped = fit_model._fit_region_mask() & ~fit_model.get_fit_mask()
        assert clipped[dip & fit_model._fit_region_mask()].all()
        assert clipped.sum() < dip.sum() + 10, scale

    with pytest.raises(ValueError, match="scale"):
        fit_model.sigma_clip(scale="mad", values=values)


def test_sigma_clip_high_max_width_spares_unmodeled_lines(fit_model):
    """A one-pixel spike is clipped; a resolved line the model lacks is not."""

    x = fit_model.spec.dispersion
    fit_model.spec.fluxden[150] += 0.1
    line = ana.line_model_gaussian(x, 3.227, 3.0, 1430.0, 1500)
    fit_model.spec.fluxden = fit_model.spec.fluxden + line
    values = fit_model.run_quick_fit().values
    core = line > 0.5 * line.max()

    fit_model.sigma_clip(low=None, high=5, values=values, max_fraction=1)
    assert not fit_model.get_fit_mask()[core].any()

    fit_model.sigma_clip(low=None, high=5, high_max_width=3, values=values)
    mask = fit_model.get_fit_mask()
    assert not mask[150]
    assert mask[core].all()


def test_least_squares_and_minimize_agree(fit_model):
    """The default least-squares path ("trf") finds the same MAP as
    L-BFGS-B on the log-probability, with the same log-probability."""

    _perturb(fit_model)
    fast = fit_model.run_quick_fit()
    slow = fit_model.run_quick_fit(method="L-BFGS-B")
    assert fast.method == "trf"
    np.testing.assert_allclose(fast.values, slow.values, rtol=1e-3)
    assert fast.log_prob == pytest.approx(slow.log_prob, abs=1e-3)
    assert fast.log_prob >= slow.log_prob - 1e-6


def test_trf_needs_the_default_log_probability(fit_model):
    def custom(theta, x, model, y, yerr):
        return scmod.log_probability(theta, x, model, y, yerr)

    result = fit_model.run_quick_fit(log_probability=custom)
    assert result.method == "L-BFGS-B"
    with pytest.raises(ValueError, match="default log-probability"):
        fit_model.run_quick_fit(log_probability=custom, method="trf")


def test_start_on_a_bound_is_not_reported_outside(caplog):
    model, *_ = _linear_model(prior_a=scpri.UniformPrior("lin_a", 2.0, 3))
    model.parameters["lin_a"].value = 2.0
    with caplog.at_level(logging.WARNING):
        result = model.run_quick_fit()
    assert "outside its prior" not in caplog.text
    assert result.success


def test_component_evaluator_matches_eval(fit_model):
    from sculptor import quick_fit as qf

    fit_model.resolution = 2000.0
    fit_model.get_params_to_sample()
    x = fit_model.spec.dispersion
    theta = np.array([p.value for p in fit_model.params_variable.values()])
    evaluator = qf._ComponentEvaluator(fit_model, x)
    np.testing.assert_allclose(
        evaluator.all(theta).sum(axis=0), fit_model.eval(x, theta), rtol=1e-10
    )
