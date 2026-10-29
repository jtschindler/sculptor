"""Save/load round trips for `FitParameter`, `FitComponent` and `FitModel`.

Replaces the interactive `examples/test_saving.py` scratch script (now in
`deprecated/examples/`).

Also a regression test: `FitComponent.param_mapping` is a dict subclass
that notifies its owner on mutation. Unpickling a dict subclass restores
its items before its attributes, so the notification ran before `_owner`
existed and raised `AttributeError`. Every component with a mapping (i.e.
every component after `create_params()`) failed to unpickle, which broke
all `save`/`load` methods and the `pool=` options that send the model to
worker processes.
"""

import gc
import multiprocessing
import pickle

import numpy as np
import pytest

from sculptor import analysis as ana
from sculptor import component as sccomp
from sculptor import model as scmod
from sculptor import parameter as scpar
from sculptor import prior as scpri

REDSHIFT = 3.227


def test_parameter_round_trip(tmp_path):
    par = scpar.FitParameter("redsh_nl", value=REDSHIFT, vary=True)
    par.prior = scpri.UniformPrior("redsh_nl", 3.2, 3.3)

    par.save("redsh_nl", tmp_path, prefix="run1")
    loaded = scpar.FitParameter()
    loaded.load("redsh_nl", tmp_path, prefix="run1")

    assert (tmp_path / "run1_redsh_nl.pkl").exists()
    assert (loaded.name, loaded.value, loaded.vary) == ("redsh_nl", REDSHIFT, True)
    assert isinstance(loaded.prior, scpri.UniformPrior)
    assert (loaded.prior.low, loaded.prior.upp) == (3.2, 3.3)


def test_component_with_param_mapping_pickles(line_component):
    comp, _ = line_component()

    restored = pickle.loads(pickle.dumps(comp))

    assert dict(restored.param_mapping) == dict(comp.param_mapping)
    assert isinstance(restored.param_mapping, sccomp._ObservableParamMapping)
    # The restored mapping must notify the restored component, not the
    # original one.
    assert restored.param_mapping._owner is restored


def test_component_round_trip(tmp_path, line_component):
    comp, _ = line_component()
    comp.param_mapping.update({"redsh": "redsh_nl"})
    comp.save("SiIV_a", tmp_path)

    loaded = sccomp.FitComponent("other", ana.power_law_at2500)
    loaded.load("SiIV_a", tmp_path)
    # `load` used to share the unpickled object's __dict__: collecting that
    # object emptied the loaded state (Python 3.13)
    gc.collect()

    assert loaded.name == "SiIV_a"
    assert loaded.param_mapping._owner is loaded
    assert loaded.function is ana.line_model_gaussian
    assert loaded.get_arg_keys() == comp.get_arg_keys()

    x = np.linspace(5800, 6000, 50)
    params = [6.0, 1399.8, 1200]
    assert np.array_equal(
        loaded.eval(x, [REDSHIFT, *params]), comp.eval(x, [REDSHIFT, *params])
    )

    # The cache must still be invalidated by in-place updates after loading.
    loaded.param_mapping.update({"redsh": "SiIV_a_redsh"})
    assert loaded.get_arg_keys()[0] == "SiIV_a_redsh"


def test_model_round_trip(tmp_path, fit_model):
    model = fit_model
    model.save("fit", tmp_path)

    loaded = scmod.FitModel()
    loaded.load("fit", tmp_path)
    gc.collect()

    assert (tmp_path / "fit.pkl").exists()
    assert [c.name for c in loaded.components] == ["pl", "SiIV_a"]
    assert list(loaded.parameters) == list(model.parameters)
    assert np.array_equal(loaded.gpm_fit, model.gpm_fit)
    assert np.array_equal(loaded.spec.fluxden, model.spec.fluxden)

    chain = model.sampler.get_chain(flat=True)
    assert np.array_equal(loaded.sampler.get_chain(flat=True), chain)

    x = model.spec.dispersion
    assert np.array_equal(loaded.eval(x, chain[-1]), model.eval(x, chain[-1]))


def test_model_evaluates_in_spawned_worker(fit_model):
    """The `pool=` options send the model to worker processes.

    With the unpicklable mapping, a worker could not unpickle its task, died,
    and the pool waited forever, so `get_mcmc_model_results(pool=...)` hung.
    This sends the model the same way (`_evaluate_single_sample`), but with a
    timeout, so a regression fails instead of hanging the test suite.
    """

    model = fit_model
    x = model.spec.dispersion
    theta = model.sampler.get_chain(flat=True)[-1]
    expected = model.eval(x, theta)

    ctx = multiprocessing.get_context("spawn")
    with ctx.Pool(1) as pool:
        result = pool.apply_async(
            scmod._evaluate_single_sample, ((model, x, theta, None),)
        ).get(timeout=60)

    assert result == pytest.approx(expected)


def test_load_rejects_a_different_class(tmp_path, line_component):
    comp, _ = line_component()
    comp.save("SiIV_a", tmp_path)

    with pytest.raises(TypeError, match="holds a FitComponent, not a FitParameter"):
        scpar.FitParameter().load("SiIV_a", tmp_path)


def test_save_load_signatures_are_identical():
    import inspect

    for method in ("save", "load"):
        signatures = {
            str(inspect.signature(getattr(cls, method)))
            for cls in (scpar.FitParameter, sccomp.FitComponent, scmod.FitModel)
        }
        assert len(signatures) == 1, signatures
