"""Build a fit component and its parameters from a parameter dictionary.

Every `initialize_*` function in `qso_autofit` and `qso_advanced_models`
is `initialize_component` with a different model function.
"""

from sculptor import component as sccomp


def initialize_component(
    prefix, function, param_dict, fixed=None, dict_name="param_dict"
):
    """Create a `FitComponent` for `function` and set its parameters from
    `param_dict`.

    `param_dict` has one entry per parameter of `function` (its arguments
    after `x`), keyed by the parameter name without the prefix. Each entry
    is either

    - ``{"value": ..., "vary": ..., "prior": ...}``: initial value, whether
      the parameter varies and (if it does) its prior, or
    - ``{"parameter": "<name>"}``: map it onto an existing parameter of the
      model (e.g. a shared redshift); the component's own parameter is
      dropped.

    :param prefix: component prefix
    :type prefix: str
    :param function: model function, or a callable with a compatible
        `__code__` (see `sculptor.component.FakeCode`)
    :type function: callable
    :param param_dict: parameter settings, see above
    :type param_dict: dict
    :param fixed: values for parameters that are constant and not read from
        `param_dict` (e.g. a template's arrays), keyed by parameter name
    :type fixed: dict
    :param dict_name: name of `param_dict` in the calling function, for the
        error message
    :type dict_name: str
    :return: the component and its parameters
    :rtype: sculptor.component.FitComponent, dict
    :raises ValueError: if `param_dict` is empty or None
    """
    if not param_dict:
        raise ValueError(f"Provide `{dict_name}`")

    fixed = fixed or {}
    comp = sccomp.FitComponent(prefix, function)
    params = comp.create_params()

    # list(): parameters mapped onto others are popped while looping
    for key in list(params):
        par = key.removeprefix(f"{prefix}_")
        if par in fixed:
            params[key].value = fixed[par]
            params[key].vary = False
            continue

        settings = param_dict[par]
        if "parameter" in settings:
            params.pop(key)
            comp.param_mapping.update({par: settings["parameter"]})
        else:
            params[key].value = settings["value"]
            params[key].vary = settings["vary"]
            if settings["vary"]:
                params[key].prior = settings["prior"]

    return comp, params
