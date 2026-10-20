"""Regression tests for bug fixes and a performance optimization applied to
`sculptor.model.FitModel` and `sculptor.component.FitComponent`.

1. `FitModel.get_params_to_sample()` used to append to `params_constant_val`
   without ever clearing it, so calling it more than once (which happens in
   normal usage: `eval()` calls it once, then `initialize_emcee()` /
   `initialize_positions()` call it again unconditionally) silently
   duplicated constant-parameter values.
2. `FitModel.calculate_chi_sq()` computed degrees of freedom using
   `len(self.parameters)` (fixed + variable) instead of only the variable
   (actually fitted) parameters.
3. `FitModel.eval()` used to recompute, on every single call, the mapping
   from each component's function arguments to their parameter-dict keys --
   static data that never changes after `FitComponent.create_params()` is
   called. This is now cached on the component via `get_arg_keys()`.
"""

from types import SimpleNamespace

import numpy as np

from sculptor import component, model, parameter


def _gaussian(x, amp, cen, sigma):
    return amp * np.exp(-0.5 * ((x - cen) / sigma) ** 2)


def _linear(x, slope, intercept):
    return slope * x + intercept


def _build_model_with_fixed_and_variable_params():
    """A FitModel with two components, each contributing one fixed and one
    variable parameter, so both `params_variable` and `params_constant` are
    non-empty.
    """

    line = component.FitComponent("line", _gaussian)
    line_params = line.create_params()
    line_params["line_amp"] = parameter.FitParameter("line_amp", 10.0, vary=True)
    line_params["line_cen"] = parameter.FitParameter("line_cen", 5000.0, vary=True)
    line_params["line_sigma"] = parameter.FitParameter("line_sigma", 5.0, vary=False)

    cont = component.FitComponent("cont", _linear)
    cont_params = cont.create_params()
    cont_params["cont_slope"] = parameter.FitParameter("cont_slope", 0.01, vary=False)
    cont_params["cont_intercept"] = parameter.FitParameter(
        "cont_intercept", 1.0, vary=True
    )

    fit_model = model.FitModel()
    fit_model.add_component(line)
    fit_model.add_component(cont)
    fit_model.add_parameters(line_params)
    fit_model.add_parameters(cont_params)

    return fit_model


# --------------------------------------------------------------------------
# Bug 1: get_params_to_sample() must be idempotent.
# --------------------------------------------------------------------------


def test_get_params_to_sample_idempotent():
    fit_model = _build_model_with_fixed_and_variable_params()

    fit_model.get_params_to_sample()
    assert len(fit_model.params_constant_val) == len(fit_model.params_constant)
    assert len(fit_model.params_constant_val) == 2  # line_sigma, cont_slope

    # Call it again, mirroring eval() -> initialize_emcee()/initialize_positions().
    fit_model.get_params_to_sample()
    fit_model.get_params_to_sample()

    assert len(fit_model.params_constant_val) == len(fit_model.params_constant)
    assert len(fit_model.params_constant_val) == 2
    assert fit_model.vardim == len(fit_model.params_variable)
    assert fit_model.vardim == 3  # line_amp, line_cen, cont_intercept


def test_eval_unaffected_by_repeated_get_params_to_sample_calls():
    fit_model = _build_model_with_fixed_and_variable_params()

    x = np.linspace(4900, 5100, 200)
    values = [10.0, 5000.0, 1.0]  # line_amp, line_cen, cont_intercept (in order)

    y_before = fit_model.eval(x, values, broaden_by_resolution=False)

    # Simulate repeated setup calls, as initialize_emcee/initialize_positions do.
    fit_model.get_params_to_sample()
    fit_model.get_params_to_sample()

    y_after = fit_model.eval(x, values, broaden_by_resolution=False)

    np.testing.assert_allclose(y_before, y_after)


# --------------------------------------------------------------------------
# Bug 2: calculate_chi_sq() degrees of freedom.
# --------------------------------------------------------------------------


def test_calculate_chi_sq_dof_counts_only_variable_params():
    fit_model = _build_model_with_fixed_and_variable_params()
    fit_model.get_params_to_sample()

    n_data = 50
    flux = np.ones(n_data)
    flux_err = np.full(n_data, 0.1)
    flux_model = np.ones(n_data) * 0.9

    fit_model.spec = SimpleNamespace(fluxden=flux, fluxden_err=flux_err)
    fit_model.gpm_fit = np.ones(n_data, dtype=bool)
    fit_model.mcmc_model_posterior = [flux_model, flux_model, flux_model]

    chi_sq, dof, reduced_chi_sq = fit_model.calculate_chi_sq()

    expected_dof = n_data - len(fit_model.params_variable)
    assert dof == expected_dof
    assert dof != n_data - len(fit_model.parameters)  # the old, wrong formula
    np.testing.assert_allclose(reduced_chi_sq, chi_sq / expected_dof)


# --------------------------------------------------------------------------
# Optimization: cached arg-key mapping in FitComponent, used by FitModel.eval.
# --------------------------------------------------------------------------


def test_get_arg_keys_matches_param_mapping_and_is_cached():
    comp = component.FitComponent("line", _gaussian)
    comp.create_params()

    expected = [comp.param_mapping[arg] for arg in comp.function_args]
    arg_keys = comp.get_arg_keys()

    assert arg_keys == expected
    # Cached: same object returned on subsequent calls.
    assert comp.get_arg_keys() is arg_keys


def test_eval_output_unchanged_across_repeated_calls_with_caching():
    fit_model = _build_model_with_fixed_and_variable_params()

    x = np.linspace(4900, 5100, 200)
    values = [10.0, 5000.0, 1.0]

    y_first = fit_model.eval(x, values, broaden_by_resolution=False)
    y_second = fit_model.eval(x, values, broaden_by_resolution=False)

    np.testing.assert_allclose(y_first, y_second)

    expected = _gaussian(x, 10.0, 5000.0, 5.0) + _linear(x, 0.01, 1.0)
    np.testing.assert_allclose(y_first, expected)


# --------------------------------------------------------------------------
# Bug fix: big-endian dispersion no longer crashes FitModel.eval() when
# resolution broadening is active (FITS-loaded spectra are commonly
# big-endian; see src/sculptor/utils.py::broaden_spectrum).
# --------------------------------------------------------------------------


def test_eval_with_resolution_handles_bigendian_dispersion():
    comp = component.FitComponent("line", _gaussian)
    params = comp.create_params()
    params["line_amp"] = parameter.FitParameter("line_amp", 10.0)
    params["line_cen"] = parameter.FitParameter("line_cen", 5000.0)
    params["line_sigma"] = parameter.FitParameter("line_sigma", 5.0)

    fit_model = model.FitModel()
    fit_model.add_component(comp)
    fit_model.add_parameters(params)
    fit_model.resolution = 20.0
    fit_model.get_params_to_sample()

    x_native = np.linspace(4900, 5100, 200).astype(np.float64)
    x_bigendian = x_native.astype(">f8")
    values = [10.0, 5000.0, 5.0]

    y_native = fit_model.eval(x_native, values)  # must not raise
    y_bigendian = fit_model.eval(x_bigendian, values)  # must not raise either

    assert np.all(np.isfinite(y_bigendian))
    np.testing.assert_array_equal(y_native, y_bigendian)


# --------------------------------------------------------------------------
# Fix: FitComponent.param_mapping mutation always invalidates the cached
# get_arg_keys() result, even after the component has already been
# evaluated once (the coupled-parameter idiom, e.g.
# `comp.param_mapping.update({'redsh': 'redsh_nl'})`).
# --------------------------------------------------------------------------


def test_get_arg_keys_reflects_in_place_param_mapping_update_after_caching():
    comp = component.FitComponent("line", _gaussian)
    comp.create_params()

    # Warm the cache (as a first FitModel.eval() call would).
    first = comp.get_arg_keys()
    assert first == ["line_amp", "line_cen", "line_sigma"]

    # Re-couple 'cen' onto a shared/global parameter, in place -- exactly
    # the idiom agn_fit_new.py uses to couple redshift/FWHM across
    # components.
    comp.param_mapping.update({"cen": "redsh_nl"})

    updated = comp.get_arg_keys()
    assert updated == ["line_amp", "redsh_nl", "line_sigma"]


def test_get_arg_keys_reflects_param_mapping_reassignment_after_caching():
    comp = component.FitComponent("line", _gaussian)
    comp.create_params()

    comp.get_arg_keys()  # warm the cache

    comp.param_mapping = {"amp": "shared_amp", "cen": "line_cen", "sigma": "line_sigma"}

    assert comp.get_arg_keys() == ["shared_amp", "line_cen", "line_sigma"]


# --------------------------------------------------------------------------
# `FitModel.eval()` zips `params_variable`/`params_constant` keys against the
# caller-supplied values with `strict=True`, so a length mismatch (e.g. an
# `emcee` walker position with the wrong dimensionality) raises immediately
# instead of silently zipping only up to the shorter length and evaluating
# with a wrong/incomplete parameter set.
# --------------------------------------------------------------------------


def test_eval_raises_on_params_values_length_mismatch():
    fit_model = _build_model_with_fixed_and_variable_params()
    fit_model.get_params_to_sample()

    x = np.linspace(4900, 5100, 50)
    too_few = np.ones(len(fit_model.params_variable) - 1)

    try:
        fit_model.eval(x, too_few)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for mismatched params_values length")
