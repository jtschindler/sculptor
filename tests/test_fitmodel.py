"""Tests for the `FitModel` clean-up (review §2.2, 2.4, 2.5, 2.6, §5.3, 5.4).

- One chain accessor (`get_flat_chain`) used by every method.
- A model with only a `flat_chain` (no sampler, `nsteps`/`nwalkers` None)
  can be analysed and plotted.
- Clear errors when no spectrum is set (used to be `AttributeError` or a
  silent no-op).
- Per-component posterior curves go through the same helper (and pool) as
  the full model.
- `get_components_by_prefix` matches prefixes, not exact names.
"""

from multiprocessing.pool import ThreadPool

import matplotlib
import numpy as np
import pytest

from sculptor import model as scmod

matplotlib.use("Agg")

DISCARD = 10


def _chain_only_copy(model):
    """The same model, but only with a flat chain (no sampler, no step
    counts), as after assigning a chain from elsewhere."""

    other = scmod.FitModel(
        components=model.components,
        parameters=model.parameters,
        spectrum=model.spec,
        redshift=model.redshift,
    )
    other.gpm_fit = model.gpm_fit.copy()
    other.flat_chain = model.get_flat_chain(discard=DISCARD).copy()
    other.flat_chain_discard = DISCARD
    return other


# -- chain access (§2.5) -----------------------------------------------------


def test_get_flat_chain_reads_the_sampler(fit_model):
    chain = fit_model.get_flat_chain(discard=DISCARD)

    assert np.array_equal(
        chain, fit_model.sampler.get_chain(discard=DISCARD, flat=True)
    )
    assert np.array_equal(
        fit_model.flat_log_prob,
        fit_model.sampler.get_log_prob(discard=DISCARD, flat=True),
    )
    assert fit_model.flat_chain_discard == DISCARD
    assert fit_model.get_flat_chain(discard=5).shape[0] > chain.shape[0]


def test_rerunning_emcee_invalidates_cached_chain(fit_model):
    first = fit_model.get_flat_chain(discard=DISCARD).copy()
    fit_model.run_emcee(nsteps=20, nwalkers=10, progress=False)

    assert fit_model.mcmc_model_posterior is None
    assert not np.array_equal(fit_model.get_flat_chain(discard=DISCARD), first)


def test_chain_only_model_rejects_other_discard(fit_model):
    other = _chain_only_copy(fit_model)

    assert np.array_equal(other.get_flat_chain(discard=DISCARD), fit_model.flat_chain)
    with pytest.raises(ValueError, match="built with discard=10"):
        other.get_flat_chain(discard=5)


def test_no_chain_is_an_error():
    with pytest.raises(ValueError, match="No MCMC chain"):
        scmod.FitModel().get_flat_chain()


def test_log_prob_is_recomputed_without_sampler(fit_model):
    expected = fit_model.sampler.get_log_prob(discard=DISCARD, flat=True)
    other = _chain_only_copy(fit_model)

    np.testing.assert_allclose(
        other.get_flat_log_prob(discard=DISCARD), expected, rtol=1e-12
    )
    assert np.array_equal(
        other.get_mode_model(discard=DISCARD),
        fit_model.get_mode_model(discard=DISCARD),
    )


def test_log_probability_includes_prior_on_a_fresh_model(fit_model):
    """Regression: on a model whose `params_variable` was not populated yet,
    the first `log_probability` call skipped the prior (it was read before
    `eval()` filled the dict). This made `get_flat_log_prob` on a
    chain-only model wrong for its first sample."""

    theta = fit_model.get_flat_chain(discard=DISCARD)[0]
    spec = fit_model.spec
    args = (spec.dispersion, spec.fluxden, spec.fluxden_err)
    expected = scmod.log_prior(
        theta, fit_model.params_variable
    ) + scmod.log_likelihood_chi_square(theta, fit_model, *args)

    fresh = _chain_only_copy(fit_model)
    assert fresh.params_variable == {}
    got = scmod.log_probability(theta, args[0], fresh, args[1], args[2])

    assert scmod.log_prior(theta, fit_model.params_variable) != 0.0
    assert got == expected


# -- plotting a chain-only model (§5.4) ---------------------------------------


def test_plot_mcmc_result_without_sampler(fit_model, tmp_path):
    other = _chain_only_copy(fit_model)
    assert other.nsteps is None and other.nwalkers is None

    other.plot_mcmc_result(
        discard=DISCARD, save=True, save_dir=tmp_path, show_components=True
    )

    assert (tmp_path / "fit_result.pdf").exists()
    assert other.posterior_discard == DISCARD


def test_posterior_is_recomputed_when_discard_changes(fit_model, tmp_path):
    fit_model.get_mcmc_model_results(discard=DISCARD)
    fit_model.plot_mcmc_result(discard=5, save=True, save_dir=tmp_path)

    assert fit_model.posterior_discard == 5


# -- posterior bands and components (§2.4) ------------------------------------


def test_posterior_bands_have_no_side_effects(fit_model):
    bands = fit_model._posterior_model_bands(discard=DISCARD)

    assert fit_model.mcmc_model_posterior is None
    med, low, upp = bands
    assert np.all(low <= med) and np.all(med <= upp)


def test_component_bands_use_the_pool(fit_model):
    line = fit_model.components[1]
    serial = fit_model._posterior_model_bands(discard=DISCARD, components=[line])

    with ThreadPool(2) as pool:
        pooled = fit_model._posterior_model_bands(
            discard=DISCARD, components=[line], pool=pool
        )

    for a, b in zip(serial, pooled, strict=True):
        assert a == pytest.approx(b)

    # Components add up to the full model sample by sample, so the medians
    # of a single-component model are its own medians, not the full one's.
    full = fit_model._posterior_model_bands(discard=DISCARD)
    assert not np.allclose(serial[0], full[0])


# -- no spectrum (§5.3) --------------------------------------------------------


@pytest.mark.parametrize(
    "call",
    [
        lambda m: m.initialize_emcee(10),
        lambda m: m.reset_fit_mask(),
        lambda m: m.add_wavelength_range_to_fit_mask(1, 2),
        lambda m: m.remove_wavelength_range_from_fit_mask(1, 2),
        lambda m: m.plot_mcmc_result(),
        lambda m: m.calculate_chi_sq(),
    ],
    ids=["emcee", "reset", "add", "remove", "plot", "chi_sq"],
)
def test_no_spectrum_raises_value_error(call):
    with pytest.raises(ValueError, match="No spectrum set"):
        call(scmod.FitModel())


def test_spectrum_must_be_speconed():
    with pytest.raises(TypeError, match="SpecOneD"):
        scmod.FitModel(spectrum=np.ones(10))


def test_spectrum_assigned_later_gets_a_fit_mask(fit_model):
    model = scmod.FitModel(components=fit_model.components)
    model.spec = fit_model.spec

    model.add_wavelength_range_to_fit_mask(5600, 5700)

    assert model.gpm_fit.dtype == bool and model.gpm_fit.any()


# -- prefixes (§2.6) and removed stub (§2.2) -----------------------------------


def test_get_components_by_prefix(fit_model):
    names = [c.name for c in fit_model.get_components_by_prefix(["SiIV"])]
    assert names == ["SiIV_a"]

    names = [c.name for c in fit_model.get_components_by_prefix(["p", "SiIV"])]
    assert names == ["pl", "SiIV_a"]

    assert [c.name for c in fit_model.get_components_by_prefix("pl")] == ["pl"]
    assert fit_model.get_all_prefix_components("SiIV")[0].name == "SiIV_a"
    assert fit_model.get_components_by_prefix(["CIV"]) == []


def test_removed_methods():
    # Empty stub (§2.2) and the broken pairwise corner plots (§2.3, moved to
    # deprecated/src/sculptor/model_evaluate_parameters.py).
    assert not hasattr(scmod.FitModel, "plot_posterior_histograms")
    assert not hasattr(scmod.FitModel, "evaluate_parameters")


# -- spectrum mask (adopted from upstream `kat` 4ee90b4) -----------------------


def _bad_pixel(model):
    """Turn one pixel inside the fit regions into a NaN, masked in the
    spectrum."""

    idx = int(np.flatnonzero(model.gpm_fit)[10])
    model.spec.fluxden[idx] = np.nan
    model.spec.mask[idx] = False
    return idx


def test_fit_mask_combines_fit_regions_and_spectrum_mask(fit_model):
    idx = _bad_pixel(fit_model)

    fit_mask = fit_model.get_fit_mask()
    assert fit_model.gpm_fit[idx] and not fit_mask[idx]
    assert fit_mask.sum() == fit_model.gpm_fit.sum() - 1


def test_masked_pixels_are_left_out_of_the_likelihood(fit_model):
    theta = fit_model.get_flat_chain(discard=DISCARD)[-1]
    spec = fit_model.spec
    args = (spec.dispersion, spec.fluxden, spec.fluxden_err)
    before = scmod.log_likelihood_chi_square(theta, fit_model, *args)

    idx = int(np.flatnonzero(fit_model.gpm_fit)[10])
    model_flux = fit_model.eval(spec.dispersion, theta)[idx]
    sigma2 = spec.fluxden_err[idx] ** 2
    term = -0.5 * ((spec.fluxden[idx] - model_flux) ** 2 / sigma2 + np.log(sigma2))

    assert _bad_pixel(fit_model) == idx
    after = scmod.log_likelihood_chi_square(theta, fit_model, *args)

    # The NaN pixel would make the likelihood NaN if it were still used;
    # masking it removes exactly its term.
    assert after == pytest.approx(before - term, rel=1e-12)


def test_chi_sq_and_bic_count_only_fitted_pixels(fit_model):
    fit_model.get_mcmc_model_results(discard=DISCARD)
    _, dof_before, _ = fit_model.calculate_chi_sq()

    _bad_pixel(fit_model)
    chi_sq, dof_after, _ = fit_model.calculate_chi_sq()

    assert dof_after == dof_before - 1
    assert np.isfinite(chi_sq)
    assert np.isfinite(fit_model.compute_bic(discard=DISCARD))


def test_fit_mask_without_spectrum_mask(fit_model):
    fit_model.spec.mask = None
    assert np.array_equal(fit_model.get_fit_mask(), fit_model.gpm_fit)
