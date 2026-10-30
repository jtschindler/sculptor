"""`FitModel.eval` and what it relies on: the split into variable and fixed
parameters (`get_params_to_sample`), the cached argument keys of
`FitComponent` (`get_arg_keys`, kept up to date when `param_mapping`
changes), the chi-square degrees of freedom and big-endian input.
"""

from types import SimpleNamespace

import numpy as np
import pytest

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


def test_get_params_to_sample_is_idempotent():
    """Repeated calls (as `eval`, `initialize_emcee` and
    `initialize_positions` make) give the same split and the same model."""
    fit_model = _build_model_with_fixed_and_variable_params()
    x = np.linspace(4900, 5100, 200)
    values = [10.0, 5000.0, 1.0]  # line_amp, line_cen, cont_intercept (in order)
    y_first = fit_model.eval(x, values, broaden_by_resolution=False)

    for _ in range(3):
        fit_model.get_params_to_sample()
    assert len(fit_model.params_constant_val) == len(fit_model.params_constant) == 2
    assert fit_model.vardim == len(fit_model.params_variable) == 3

    y_again = fit_model.eval(x, values, broaden_by_resolution=False)
    np.testing.assert_allclose(y_again, y_first)
    expected = _gaussian(x, 10.0, 5000.0, 5.0) + _linear(x, 0.01, 1.0)
    np.testing.assert_allclose(y_first, expected)


def test_eval_picks_up_parameter_changes(fit_model):
    """`eval` used to cache the variable/fixed split and the fixed values on
    its first call, ignoring later edits until `get_params_to_sample()`."""
    x = fit_model.spec.dispersion
    theta = [p.value for p in fit_model.parameters.values() if p.vary]
    before = fit_model.eval(x, theta)

    fixed = next(p for p in fit_model.parameters.values() if not p.vary)
    fixed.value = fixed.value * 1.01 + 0.01
    assert not np.allclose(fit_model.eval(x, theta), before)

    values = dict(zip(fit_model.params_variable, theta, strict=True))
    fixed.vary = True
    values[fixed.name] = fixed.value
    theta_more = [values[n] for n, p in fit_model.parameters.items() if p.vary]
    assert len(theta_more) == len(theta) + 1
    fit_model.eval(x, theta_more)  # no length mismatch
    assert fixed.name in fit_model.params_variable


def test_eval_raises_on_params_values_length_mismatch():
    """A wrong number of values (e.g. an emcee walker of the wrong
    dimension) raises instead of evaluating an incomplete parameter set."""
    fit_model = _build_model_with_fixed_and_variable_params()
    fit_model.get_params_to_sample()
    too_few = np.ones(len(fit_model.params_variable) - 1)
    with pytest.raises(ValueError):
        fit_model.eval(np.linspace(4900, 5100, 50), too_few)


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


def test_get_arg_keys_is_cached_and_follows_param_mapping():
    """Cached, and invalidated by any change of `param_mapping` after the
    first call: in place (the coupled-parameter idiom,
    `comp.param_mapping.update({"redsh": "redsh_nl"})`) or reassigned."""
    comp = component.FitComponent("line", _gaussian)
    comp.create_params()

    arg_keys = comp.get_arg_keys()
    assert arg_keys == [comp.param_mapping[arg] for arg in comp.function_args]
    assert arg_keys == ["line_amp", "line_cen", "line_sigma"]
    assert comp.get_arg_keys() is arg_keys

    comp.param_mapping.update({"cen": "redsh_nl"})
    assert comp.get_arg_keys() == ["line_amp", "redsh_nl", "line_sigma"]

    comp.param_mapping = {"amp": "shared_amp", "cen": "line_cen", "sigma": "line_sigma"}
    assert comp.get_arg_keys() == ["shared_amp", "line_cen", "line_sigma"]


def test_eval_with_resolution_handles_bigendian_dispersion():
    """FITS-loaded spectra are often big-endian; the resolution broadening
    used to crash on them."""
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

    y_native = fit_model.eval(x_native, values)
    y_bigendian = fit_model.eval(x_bigendian, values)

    assert np.all(np.isfinite(y_bigendian))
    np.testing.assert_array_equal(y_native, y_bigendian)
