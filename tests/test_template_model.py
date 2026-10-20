"""Tests for `sculptor.analysis.TemplateModel` and `initialise_Fe_fast`
(ported from upstream `qso_auto_fit`, normalisation matched to
`template_model`)."""

import pickle
from pathlib import Path

import numpy as np
import pytest

from sculptor import analysis as ana
from sculptor.component import FitComponent
from sculptor.model import FitModel
from sculptor_extensions import qso_presets

IRON_DIR = (
    Path(__file__).resolve().parents[1] / "src" / "sculptor" / "data" / "iron_templates"
)
REDSH = 1.5


def _template(name):
    t = np.loadtxt(IRON_DIR / name)
    return t[:, 0], t[:, 1] * 1e17


def _x(templ_disp):
    return np.linspace(
        templ_disp.min() * (1 + REDSH) * 1.03,
        templ_disp.max() * (1 + REDSH) * 0.97,
        3000,
    )


@pytest.mark.parametrize(
    ("name", "tolerance"),
    [
        ("Fe_UVtemplt_A.asc", 1e-10),  # odd number of pixels: identical
        ("Park_2022.asc", 1e-10),
        ("Fe_OPT_BR92_linear.txt", 3e-3),  # even: speconed's 1/2-pixel shift
        ("Tsuzuki06.txt", 3e-3),
    ],
)
@pytest.mark.parametrize("fwhm", [1500.0, 3000.0, 6000.0])
def test_matches_template_model(name, tolerance, fwhm):
    templ_disp, templ_fluxden = _template(name)
    x = _x(templ_disp)

    expected = ana.template_model(
        x, REDSH, 2.0, fwhm, 900.0, templ_disp=templ_disp, templ_fluxden=templ_fluxden
    )
    result = ana.TemplateModel(templ_disp, templ_fluxden)(x, REDSH, 2.0, fwhm, 900.0)

    assert np.max(np.abs(result - expected)) <= tolerance * expected.max()


def test_component_parameters_and_pickling():
    model = ana.TemplateModel(*_template("Fe_UVtemplt_A.asc"))

    assert FitComponent("Fe", model).function_args == [
        "redsh",
        "amp",
        "fwhm",
        "intr_fwhm",
    ]
    restored = pickle.loads(pickle.dumps(model))
    x = _x(model.templ_disp)
    assert np.array_equal(
        restored(x, REDSH, 1.0, 3000.0, 900.0), model(x, REDSH, 1.0, 3000.0, 900.0)
    )


def _fe_dict(slow):
    values = {"redsh": REDSH, "amp": 2.0, "fwhm": 3000.0, "intr_fwhm": 900.0}
    fe_dict = {k: {"value": v, "vary": False} for k, v in values.items()}
    if slow:  # template_model also has the template-unit arguments
        fe_dict |= {
            "templ_disp_unit_str": {"value": None, "vary": False},
            "templ_fluxden_unit_str": {"value": None, "vary": False},
        }
    return fe_dict


def test_fast_iron_setup_gives_the_same_model():
    slow_comp, slow_params = qso_presets.setup_iron_template_VW01_UV(
        "Fe", fe_dict=_fe_dict(True), fast=False
    )
    fast_comp, fast_params = qso_presets.setup_iron_template_VW01_UV(
        "Fe", fe_dict=_fe_dict(False), fast=True
    )

    assert slow_comp.function is ana.template_model
    assert isinstance(fast_comp.function, ana.TemplateModel)
    assert sorted(fast_params) == ["Fe_amp", "Fe_fwhm", "Fe_intr_fwhm", "Fe_redsh"]

    x = _x(fast_comp.function.templ_disp)
    slow = FitModel(components=[slow_comp], parameters=slow_params).eval(x, [])
    fast = FitModel(components=[fast_comp], parameters=fast_params).eval(x, [])
    np.testing.assert_allclose(fast, slow, rtol=1e-9, atol=1e-12 * slow.max())


def test_all_iron_setups_accept_fast():
    import inspect

    setups = [n for n in dir(qso_presets) if n.startswith("setup_iron_template_")]
    assert len(setups) == 10
    for name in setups:
        assert (
            inspect.signature(getattr(qso_presets, name)).parameters["fast"].default
            is True
        )
