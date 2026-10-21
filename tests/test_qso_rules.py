"""Tests for the quasar rules of the automatic model building
(`sculptor_extensions.qso_rules`)."""

import copy

import numpy as np
import pytest
from astropy import units as u
from speconed import speconed as sod

from sculptor import analysis as ana
from sculptor import autofit
from sculptor import component as sccomp
from sculptor import model as scmod
from sculptor import prior as scpri
from sculptor.autofit import ComponentRule, GaussianLineRule, LineGroupRule
from sculptor_extensions import qso_presets
from sculptor_extensions import qso_rules as qr

Z = 2.0
ERR = 0.05


class AutoFit(autofit.AutoFit):
    """AutoFit with the fixed threshold 10: these tests check the search,
    not the calibration (see the calibration tests)."""

    def __init__(self, *args, threshold=10.0, **kwargs):
        super().__init__(*args, threshold=threshold, **kwargs)


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
    assert rules[1].fwhm_range == (100.0, 2500.0)  # narrow
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
    assert kinds.count("Tie") == 1
    assert kinds.count("LineGroupRule") == 1
    assert rules[-1].prefixes == ["Hbeta", "Halpha", "OIII"]
    group = next(r for r in rules if isinstance(r, LineGroupRule))
    assert [m.prefix for m in group.members] == ["Hbeta", "Halpha", "OIII"]
    assert group.fwhm_range == (100.0, qr.NARROW_FWHM_MAX)
    assert group.velocity_range == qr.NARROW_VELOCITY_RANGE
    bare = qr.qso_rules(
        Z, iron=None, balmer=False, narrow_tie=False, narrow_group=False
    )
    assert len(bare) == len(qr.DEFAULT_LINES)
    assert len(qr.qso_rules(Z, lines=["CIV", "Hbeta"])) == 4  # nothing to tie


def test_oiii_doublet_rule():
    (rule,) = qr.line_rules(Z, ["OIII"], shared={"OIII": {"redsh": "redsh_nl"}})
    assert rule.lines == [5008.24]
    assert rule.fixed == {"fluxratio": 2.98}
    assert rule.shared == {"redsh": "redsh_nl"}
    assert rule.n_shared == 1
    assert rule.fwhm_range == (100.0, 2500.0)
    (rule,) = qr.line_rules(Z, ["OIII"], shared={})
    assert rule.shared == {}

    # Found in a spectrum at the right rest frame
    z = 0.2
    x = np.geomspace(4800 * (1 + z), 5100 * (1 + z), 600)
    model = _model(x=x)
    model.parameters["PL_redsh"].value = z
    oiii = qso_presets.qso.line_model_gaussian_oiii_doublet(x, z, 5.0, 600.0, 2.98)
    model.spec.fluxden = model.spec.fluxden + oiii
    (rule,) = qr.line_rules(z, ["OIII"])
    result = AutoFit(model, [rule]).run()
    names = [c.name for c in result.model.components]
    assert names[:2] == ["PL", "OIII_0"]
    assert result.model.parameters["OIII_0_flux"].value == pytest.approx(5.0, rel=0.1)


def _narrow_lines_model(hbeta_dv=0.0, oiii_fwhm=500.0, hbeta_fwhm=500.0):
    """Narrow Hbeta (offset by `hbeta_dv` km/s) and the [OIII] doublet at
    z = 0.2."""
    z = 0.2
    x = np.geomspace(4750 * (1 + z), 5100 * (1 + z), 900)
    model = _model(x=x)
    model.parameters["PL_redsh"].value = z
    z_hb = z + hbeta_dv / 2.998e5 * (1 + z)
    model.spec.fluxden = (
        model.spec.fluxden
        + qso_presets.qso.line_model_gaussian_oiii_doublet(x, z, 5.0, oiii_fwhm, 2.98)
        + ana.line_model_gaussian(x, z_hb, 4.0, 4862.68, hbeta_fwhm)
    )
    rules = [
        *qr.line_rules(z, ["Hbeta", "OIII"]),
        qr.narrow_line_tie(["Hbeta", "OIII"]),
    ]
    return model, rules, z


def test_narrow_line_tie_accepted():
    model, rules, z = _narrow_lines_model()
    result = AutoFit(model, rules).run()

    comps = {c.name: c for c in result.model.components}
    assert sorted(comps) == ["Hbeta_0", "OIII_0", "PL"]
    for name in ("Hbeta_0", "OIII_0"):
        assert comps[name].param_mapping["redsh"] == "narrow_redsh"
        assert comps[name].param_mapping["fwhm_km_s"] == "narrow_fwhm_km_s"
    par = result.model.parameters
    assert "Hbeta_0_redsh" not in par and "OIII_0_fwhm_km_s" not in par
    assert par["narrow_redsh"].value == pytest.approx(z, abs=1e-4)
    assert par["narrow_fwhm_km_s"].value == pytest.approx(500, rel=0.05)
    assert par["Hbeta_0_flux"].value == pytest.approx(4.0, rel=0.05)
    assert "tie" in [step.action for step in result.history]
    test = result.ties["narrow"]
    assert test.status == "tied"
    assert sorted(test.components) == ["Hbeta_0", "OIII_0"]
    assert test.delta < 0  # two parameters fewer, same fit
    assert "tie narrow: tied" in str(result)


def test_narrow_line_tie_rejected_for_an_offset():
    model, rules, _ = _narrow_lines_model(hbeta_dv=600.0)
    result = AutoFit(model, rules).run()
    test = result.ties["narrow"]
    assert test.status == "rejected"
    assert test.delta > 10
    assert "narrow_redsh" not in result.model.parameters


def test_narrow_line_tie_needs_narrow_components():
    model, rules, _ = _narrow_lines_model(oiii_fwhm=2000.0)
    result = AutoFit(model, rules).run()
    test = result.ties["narrow"]
    assert test.status == "not applicable"
    assert "OIII" in test.reason and "1200" in test.reason


def test_narrow_fwhm_limit_is_a_module_setting(monkeypatch):
    assert qr.narrow_line_tie().max_width == 1200.0
    monkeypatch.setattr(qr, "NARROW_FWHM_MAX", 300.0)
    assert qr.narrow_line_tie().max_width == 300.0
    assert qr.narrow_line_tie(max_fwhm=800.0).max_width == 800.0
    rules = {r.prefix: r for r in qr.line_rules(Z, ["Hbeta", "CIV", "OIII"])}
    assert rules["Hbeta"].fwhm_range == (100.0, 15000.0)
    assert rules["CIV"].fwhm_range == (1000.0, 15000.0)
    assert rules["OIII"].fwhm_range == (100.0, 2500.0)


def _weak_narrow_model(narrow_flux, dv=0.0):
    """Broad Hbeta with weak narrow Hbeta and [OIII] cores (400 km/s, offset
    by `dv` km/s from the systemic redshift) at z = 0.2."""
    z = 0.2
    x = np.geomspace(4700 * (1 + z), 5100 * (1 + z), 900)
    model = _model(x=x)
    model.parameters["PL_redsh"].value = z
    z_nl = z + dv / 2.998e5 * (1 + z)
    model.spec.fluxden = (
        model.spec.fluxden
        + ana.line_model_gaussian(x, z, 40.0, 4862.68, 4000.0)
        + ana.line_model_gaussian(x, z_nl, narrow_flux, 4862.68, 400.0)
        + qso_presets.qso.line_model_gaussian_oiii_doublet(
            x, z_nl, narrow_flux, 400.0, 2.98
        )
    )
    lines = qr.line_rules(z, ["Hbeta", "OIII"])
    return model, lines, z


def test_narrow_line_group_found_at_systemic():
    model, lines, z = _weak_narrow_model(1.0)
    group = qr.narrow_line_group(lines, z)
    rules = [*lines, group, qr.narrow_line_tie(["Hbeta", "OIII"])]
    result = AutoFit(model, rules).run()

    comps = {c.name: c for c in result.model.components}
    tied = [
        name
        for name, comp in comps.items()
        if comp.param_mapping.get("redsh") == "narrow_redsh"
    ]
    assert len(tied) == 2
    assert {name.split("_")[0] for name in tied} == {"Hbeta", "OIII"}
    par = result.model.parameters
    assert par["narrow_redsh"].value == pytest.approx(z, abs=3e-4)
    assert par["narrow_fwhm_km_s"].value == pytest.approx(400, rel=0.2)
    assert "narrow group" in " ".join(step.description for step in result.history)
    assert result.ties["narrow"].status == "tied"  # the tie sees the group


def test_narrow_line_group_starts_at_a_residual_peak():
    model, lines, z = _weak_narrow_model(3.0, dv=500.0)
    group = qr.narrow_line_group(lines, z)
    result = AutoFit(model, [*lines, group]).run()
    par = result.model.parameters
    z_nl = z + 500.0 / 2.998e5 * (1 + z)
    assert par["narrow_redsh"].value == pytest.approx(z_nl, abs=3e-4)


def test_narrow_line_group_needs_two_lines():
    lines = qr.line_rules(Z, ["CIV", "Hbeta"])
    assert qr.narrow_line_group(lines, Z) is None
    with pytest.raises(ValueError, match="one line"):
        LineGroupRule("g", [GaussianLineRule("a", Z), lines[1]], Z)


def test_narrow_line_group_proposes_once():
    model, lines, z = _weak_narrow_model(1.0)
    group = qr.narrow_line_group(lines, z)
    fitted = AutoFit(model, [*lines, group]).run()
    assert group.propose(fitted.model, fitted.result) == []
    (cand,) = group.propose(model, model.run_quick_fit())
    widths = [s["narrow_fwhm_km_s"] for s in cand.starts if "narrow_fwhm_km_s" in s]
    assert len(widths) == 2  # the other start widths
    fluxes = [p for p in cand.parameters if p.endswith("_flux")]
    assert len(fluxes) == 2
    assert all(cand.parameters[f].value > 0 for f in fluxes)


def test_ciii_complex_rule():
    def complex_(x):
        return qso_presets.qso.CIII_complex_model_func(
            x,
            Z,
            1908.73,
            1857.40,
            1892.03,
            30.0,
            3000.0,
            0.0,
            6.0,
            2500.0,
            0.0,
            8.0,
            2500.0,
            0.0,
        )

    model = _model(complex_)
    rule = qr.ciii_complex_rule(Z)
    (cand,) = rule.propose(model, model.run_quick_fit())
    assert (
        cand.parameters["CIIIc_flux"].value
        > cand.parameters["CIIIc_flux_siIII"].value
        > 0
    )
    assert len(cand.starts) == 2
    assert all("CIIIc_flux_alIII" in start for start in cand.starts)

    result = AutoFit(model, [rule]).run()
    par = result.model.parameters
    assert [c.name for c in result.model.components] == ["PL", "CIIIc"]
    assert par["CIIIc_flux"].value == pytest.approx(30.0, rel=0.1)
    assert par["CIIIc_flux_siIII"].value == pytest.approx(8.0, rel=0.3)
    assert par["CIIIc_flux_alIII"].value == pytest.approx(6.0, rel=0.3)
    assert not par["CIIIc_shift_km_s"].vary


def test_blend_lines():
    rules = {r.prefix: r for r in qr.line_rules(Z, qr.BLEND_LINES)}
    assert rules["OIII1663"].lines == [1663.48]
    assert rules["SiIII"].fwhm_range == (1000.0, 15000.0)
    model = _model()
    assert rules["OIII1663"].owned(model) == []
