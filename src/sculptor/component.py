from sculptor.parameter import FitParameter
from sculptor.persistence import PickleMixin


class _ObservableParamMapping(dict):
    """A `dict` that notifies its owning `FitComponent` of any in-place
    mutation, so `FitComponent._arg_keys` (the cache behind `get_arg_keys()`)
    is never left stale.

    Coupling parameters across components by mutating `param_mapping` in
    place (e.g. `comp.param_mapping.update({'redsh': 'redsh_nl'})`, to alias
    a component's local argument onto a shared/global `FitParameter`) is a
    normal, supported usage pattern -- not just reassigning the whole dict.
    Without this, `get_arg_keys()`'s cache would only be correct if such a
    mutation always happened before the component was first evaluated;
    mutating it again afterwards (e.g. re-coupling parameters on an
    already-fit model) would silently keep using the stale mapping.
    """

    def __init__(self, owner, *args, **kwargs):
        self._owner = owner
        super().__init__(*args, **kwargs)

    def _invalidate(self):
        # Unpickling (and `copy`) a dict subclass restores its items before
        # its attributes, so `__setitem__` runs while `_owner` is not set
        # yet. There is no cache to invalidate at that point.
        owner = getattr(self, "_owner", None)
        if owner is not None:
            owner._arg_keys = None

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        self._invalidate()

    def __delitem__(self, key):
        super().__delitem__(key)
        self._invalidate()

    def update(self, *args, **kwargs):
        """Like `dict.update`; then invalidate the owner's argument-key cache."""

        super().update(*args, **kwargs)
        self._invalidate()

    def pop(self, *args, **kwargs):
        """Like `dict.pop`; then invalidate the owner's argument-key cache."""

        result = super().pop(*args, **kwargs)
        self._invalidate()
        return result

    def clear(self):
        """Like `dict.clear`; then invalidate the owner's argument-key cache."""

        super().clear()
        self._invalidate()


class FakeCode:
    """Stand-in for a function's `__code__`, for callable objects used as a
    `FitComponent` function.

    `FitComponent` reads the model's argument names from
    `function.__code__.co_argcount` / `co_varnames`; a callable object (e.g.
    `sculptor.analysis.TemplateModel`) sets `self.__code__ = FakeCode(...)`
    with the argument names of its `__call__` (without `self`).

    :param co_argcount: Number of arguments, including `x`
    :type co_argcount: int
    :param co_varnames: Argument names, starting with "x"
    :type co_varnames: tuple(str)
    """

    def __init__(self, co_argcount, co_varnames):
        self.co_argcount = co_argcount
        self.co_varnames = co_varnames


class FitComponent(PickleMixin):
    """One additive model component: a model function and the mapping of its
    arguments onto the model's parameters.

    The function's first argument is the dispersion axis `x`; every further
    argument is a fit parameter (`function_args`). By default
    `create_params()` names them `<name>_<argument>`; `param_mapping` can
    point an argument at any other parameter, e.g. a shared redshift.

    :param name: Component name, also the prefix of its parameter names
    :type name: str
    :param function: Model function `f(x, *params)`, or a callable object
        exposing a compatible `__code__` (see
        `sculptor_extensions.qso_advanced_models.BalmerContinuumWithHOL`)
    :type function: callable
    :param parameters: Unused; kept for backwards compatibility.
    :param param_names: Unused; kept for backwards compatibility.
    :param param_mapping: Function argument name -> parameter name. Usually
        set by `create_params()`.
    :type param_mapping: dict
    """

    def __init__(
        self, name, function, parameters=None, param_names=None, param_mapping=None
    ):

        self.name = name
        self.function = function
        self.function_n_args = self.function.__code__.co_argcount
        self.function_args = list(
            self.function.__code__.co_varnames[1 : self.function_n_args]
        )

        self._arg_keys = None
        self.param_mapping = param_mapping

    def __str__(self):
        return f"{self.name}: {self.function}"

    def info(self):
        """Print the component's name, function name and function arguments."""

        print(f"[INFO] Component name: {self.name}")
        print(f"[INFO] Function name: {self.function.__name__}")
        print(f"[INFO] Function arguments: {self.function_args}")

    @property
    def param_mapping(self):
        """Function argument name -> parameter name (a key of the model's
        parameter dict). Mutating it in place, or assigning a new dict, keeps
        `get_arg_keys()` up to date.
        """

        return self._param_mapping

    @param_mapping.setter
    def param_mapping(self, value):
        """Set the mapping (wrapped so that in-place edits invalidate the cache)."""
        self._arg_keys = None
        if value is None:
            self._param_mapping = None
        else:
            self._param_mapping = _ObservableParamMapping(self, value)

    def eval(self, x, params):
        """Evaluate the component.

        :param x: Dispersion axis
        :type x: np.ndarray
        :param params: Parameter values in the order of `function_args`
        :type params: iterable
        :return: Model flux density
        :rtype: np.ndarray
        """

        return self.function(x, *params)

    def get_arg_keys(self):
        """Return the parameter-dict keys for `function_args`, in order.

        Cached, since `param_mapping`/`function_args` are typically static
        once the component's parameters have been created -- but the cache
        is transparently invalidated whenever `param_mapping` is mutated
        (in place or by reassignment), so it stays correct even if a caller
        re-couples parameters after the component has already been
        evaluated once.
        """

        if self._arg_keys is None:
            self._arg_keys = [self.param_mapping[arg] for arg in self.function_args]

        return self._arg_keys

    def create_params(self):
        """Create one `FitParameter` per function argument and map the arguments
        onto them.

        The parameters are named `<component name>_<argument>`, start at value
        0 and vary by default; set values, `vary` and priors on the returned
        objects.

        :return: Parameter name -> FitParameter
        :rtype: dict
        """

        parameters = {}
        mapping = {}
        for name in self.function_args:
            parameters.update(
                {self.name + "_" + name: FitParameter(self.name + "_" + name, 0)}
            )
            mapping.update({name: self.name + "_" + name})
        self.param_mapping = mapping

        return parameters
