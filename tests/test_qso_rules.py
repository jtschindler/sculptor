"""Tests for the quasar rules of the automatic model building
(`sculptor_extensions.qso_rules`)."""

import copy

import numpy as np
import pytest
from astropy import units as u
from speconed import speconed as sod

from sculptor import analysis as ana
from sculptor import component as sccomp
from sculptor import model as scmod
from sculptor import prior as scpri
from sculptor.autofit import AutoFit, ComponentRule, GaussianLineRule
from sculptor_extensions import qso_presets
from sculptor_extensions import qso_rules as qr

Z = 2.0
ERR = 0.05


def _model(extra=None, x=None):
    """Power-law base model on a synthetic spectrum (plus `extra(x)`)."""
    x = np.geomspace(4200, 6200, 800) if x is None else x
    flux = ana.power_law_at2500(x, Z, 1.0, -1.5)
    if extra is not None:
        flux = flux + extra(x)
    flux = flux + np.random.default_rng(0).normal(0, ERR, x.size)
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
    model.add_wavelength_range_to_fit_mask(x[0], x[-1])
    return model


def _fixed(value):
    return {"value": value, "vary": False}


def _iron(x, amp):
    """The VW01_UV template at z=2, FWHM 2500 km/s, scaled by `amp`."""
    fe_dict = {
        "redsh": _fixed(Z),
        "amp": _fixed(amp),
        "fwhm": _fixed(2500.0),
        "intr_fwhm": _fixed(900.0),
    }
    comp, pars = qso_presets.setup_iron_template_VW01_UV("Fe", fe_dict=fe_dict)
    return comp.eval(x, [pars[k].value for k in comp.get_arg_keys()])


def test_line_rules():
    rules = qr.line_rules(Z, ["CIV", "OIII_5008"], max_components=2)
    assert [r.prefix for r in rules] == ["CIV", "OIII_5008"]
    assert all(isinstance(r, GaussianLineRule) for r in rules)
    assert rules[0].lines == [qr.QSO_LINES["CIV"]]
    assert rules[1].fwhm_range == (100.0, 1500.0)  # narrow
    assert rules[0].max_components == 2

    custom = qr.line_rules(Z, {"X": 1700.0})
    assert custom[0].lines == [1700.0]


def test_iron_template_rule_finds_the_template():
    scale = 0.5 / np.max(_iron(np.geomspace(4200, 6200, 800), 1.0))
    model = _model(extra=lambda x: _iron(x, scale))
    rule = qr.iron_template_rule(Z, vary_fwhm=False)
    assert isinstance(rule, ComponentRule)

    result = AutoFit(model, [rule]).run()
    assert [c.name for c in result.model.components] == ["PL", "FeII"]
    assert result.model.parameters["FeII_amp"].value == pytest.approx(scale, rel=0.1)

    # No iron in the data: not added
    plain = AutoFit(_model(), [qr.iron_template_rule(Z, vary_fwhm=False)]).run()
    assert [c.name for c in plain.model.components] == ["PL"]

    with pytest.raises(ValueError, match="preset"):
        qr.iron_template_rule(Z, preset="nope")


def test_balmer_rule_only_below_the_edge():
    rule = qr.balmer_continuum_rule(Z)
    model = _model()  # rest frame 1400 - 2067 A, below the 3646 A edge
    assert len(rule.propose(model, model.run_quick_fit())) <= 1

    red = _model(x=np.geomspace(3700 * (1 + Z), 4500 * (1 + Z), 400))
    assert rule.propose(red, red.run_quick_fit()) == []


def test_line_dict_rule_adds_lettered_components():
    line_dict = {
        "CIII": {
            "broad": {
                "params": {
                    "redsh": {"value": Z, "vary": False},
                    "flux": {
                        "value": 1.0,
                        "vary": True,
                        "prior": scpri.UniformPrior("flux", 0, 1000),
                    },
                    "cen": {"value": qr.QSO_LINES["CIII"], "vary": False},
                    "fwhm_km_s": {
                        "value": 3000.0,
                        "vary": True,
                        "prior": scpri.UniformPrior("fwhm", 500, 15000),
                    },
                }
            }
        }
    }
    original = copy.deepcopy(line_dict)
    model = _model(extra=lambda x: ana.line_model_gaussian(x, Z, 15.0, 1908.73, 3000.0))
    rule = qr.line_dict_rule(line_dict, "CIII", "broad", max_components=2)
    result = AutoFit(model, [rule]).run()

    names = [c.name for c in result.model.components]
    assert names[:2] == ["PL", "CIII_broad_a"]
    assert result.model.parameters["CIII_broad_a_flux"].value == pytest.approx(
        15.0, rel=0.1
    )
    assert line_dict["CIII"]["broad"]["params"]["flux"]["value"] == 1.0
    assert line_dict.keys() == original.keys()


def test_qso_rules_bundle():
    rules = qr.qso_rules(Z)
    kinds = [type(r).__name__ for r in rules]
    assert kinds.count("GaussianLineRule") == len(qr.DEFAULT_LINES)
    assert kinds.count("ComponentRule") == 2
    assert len(qr.qso_rules(Z, iron=None, balmer=False)) == len(qr.DEFAULT_LINES)
