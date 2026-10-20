"""Fit helpers and plots that had no tests: the extincted / scattered power
laws, the `Qso_autofit` chain helpers and line priors, and the posterior
plots of `FitModel`."""

import matplotlib
import numpy as np
import pandas as pd
import pytest
from astropy import units as u
from speconed import speconed as sod

from sculptor import analysis as ana
from sculptor.prior import UniformPrior
from sculptor_extensions import Qso_autofit as qa

matplotlib.use("Agg")

DISCARD = 10


def test_ext_and_scattered_power_laws():
    x = np.linspace(3000, 9000, 200)
    plain = ana.power_law_at2500(x, 1.0, 2.0, -1.5)
    np.testing.assert_allclose(ana.ext_power_law_at2500(x, 1.0, 2.0, -1.5, 0.0), plain)
    reddened = ana.ext_power_law_at2500(x, 1.0, 2.0, -1.5, 0.5)
    assert np.all(reddened < plain)
    # Extinction is stronger in the blue.
    assert reddened[0] / plain[0] < reddened[-1] / plain[-1]
    np.testing.assert_allclose(
        ana.scattered_power_law_at2500(x, 1.0, 2.0, -1.5, -1.0), 0.1 * plain
    )


def test_create_emission_line_dict_sets_values_and_prior_bounds():
    """The upper prior bounds are set from the measured line (`prior_upp_factor`
    times the value); they used to be written to a `.upper` attribute that
    `UniformPrior` does not use (its bound is `.upp`)."""
    redsh, flux, fwhm = 1.0, 50.0, 3000.0
    disp = np.linspace(5400, 5800, 4001)
    fluxden = ana.line_model_gaussian(disp, redsh, flux, 2798.75, fwhm)
    spec = sod.SpecOneD(
        dispersion=disp,
        fluxden=fluxden,
        dispersion_unit=u.AA,
        fluxden_unit=1e-17 * u.erg / u.s / u.cm**2 / u.AA,
    )
    params = {
        "flux": {
            "value": 0,
            "vary": True,
            "prior": UniformPrior("f", 0, 1),
            "prior_upp_factor": 3,
        },
        "fwhm_km_s": {
            "value": 0,
            "vary": True,
            "prior": UniformPrior("w", 0, 1),
            "prior_upp_factor": 2,
        },
    }
    line_dict = {"MgII": {"broad": {"params": params}}}

    qa.create_emission_line_dict(spec, redsh, line_dict)

    assert params["flux"]["value"] == pytest.approx(flux, rel=1e-3)
    assert params["flux"]["prior"].upp == pytest.approx(3 * flux, rel=1e-3)
    assert params["fwhm_km_s"]["value"] == pytest.approx(fwhm, rel=1e-3)
    assert params["fwhm_km_s"]["prior"].upp == pytest.approx(
        2 * params["fwhm_km_s"]["value"]
    )


def test_print_variable_parameters(fit_model):
    summary = qa.print_variable_parameters(fit_model)
    for name in fit_model.params_variable:
        assert f"- {name}" in summary
    assert "Lower bound" in summary


def test_get_sampled_values(fit_model):
    name = list(fit_model.params_variable)[1]
    np.testing.assert_array_equal(
        qa.get_sampled_values(fit_model, name, discard=DISCARD),
        fit_model.get_flat_chain(discard=DISCARD)[:, 1],
    )
    with pytest.raises(ValueError, match="not found"):
        qa.get_sampled_values(fit_model, "nope", discard=DISCARD)


def test_plot_burnin(fit_model, tmp_path):
    qa.plot_burnin(fit_model, save_dir=tmp_path, plot_name="burnin.png")
    assert (tmp_path / "burnin.png").exists()

    fit_model.flat_chain = fit_model.get_flat_chain(discard=DISCARD)
    fit_model.sampler = None
    with pytest.raises(ValueError, match="sampler"):
        qa.plot_burnin(fit_model, save_dir=tmp_path)


def test_add_line_snr_stats(fit_model, tmp_path):
    chain = fit_model.get_flat_chain(discard=DISCARD)
    df = pd.DataFrame({"source": ["J0001", "J0001"]})
    df = qa.add_line_snr_stats(
        df,
        fit_model,
        chain,
        fit_model.redshift,
        np.full(len(chain), 1200.0),
        fit_model.components[0],
        "J0001",
        tmp_path,
        "SiIV",
        1399.8,
    )
    assert (tmp_path / "J0001_SiIV_line_snr.csv").exists()
    assert (tmp_path / "J0001_SiIV_peak_snr.csv").exists()
    # The short test chain is not converged, so only check the structure.
    for kind in ("line", "peak"):
        low, med, upp = (df[f"SiIV_{kind}_snr_{k}"] for k in ("low", "median", "upp"))
        assert np.all(np.isfinite(med)) and np.all(low <= med) and np.all(med <= upp)


def test_posterior_plots(fit_model, tmp_path):
    summary = fit_model.plot_posterior_corner(
        discard=DISCARD, save=True, save_dir=tmp_path, save_name="corner.png"
    )
    assert (tmp_path / "corner.png").exists()
    assert summary

    fit_model.plot_posterior_param_histograms(
        discard=DISCARD, save=True, save_dir=tmp_path
    )
    for name in fit_model.params_variable:
        assert (tmp_path / f"posterior_hist_{name}.pdf").exists()
