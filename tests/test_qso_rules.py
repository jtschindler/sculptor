"""Tests for the quasar rules of the automatic model building
(`sculptor_extensions.qso_rules`)."""

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
from sculptor_extensions import qso_advanced_models, qso_presets
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


def test_iron_template_schemes():
    def names(iron):
        rules = qr.qso_rules(Z, iron=iron)
        return [r.name for r in rules if getattr(r, "name", "").startswith("iron")]

    assert names(qr.IRON_TEMPLATES_VW01) == ["iron (VW01_UV)", "iron (BG92_OPT)"]
    assert names(qr.IRON_TEMPLATES_TS06) == [
        "iron (VW01_FUV)",
        "iron (Tsuz_UV)",
        "iron (BG92_OPT)",
    ]
    assert names(qr.IRON_TEMPLATES_GAP) == [
        "iron (VW01_UV)",
        "iron (Tsuz_UV 3089-3500)",
        "iron (BG92_OPT)",
    ]

    # The range restricts the template
    rule = qr.iron_template_rule(Z, preset="Tsuz_UV", disp_range=(3089.0, 3500.0))
    comp, pars = rule.factory(_model())
    rest = np.arange(2500.0, 4000.0, 1.0)
    flux = comp.eval(rest * (1 + Z), [pars[k].value for k in comp.get_arg_keys()])
    assert np.all(flux[(rest < 3000) | (rest > 3600)] == 0)
    assert np.any(flux[(rest > 3100) & (rest < 3490)] > 0)


def test_balmer_rule_only_below_the_edge():
    rule = qr.balmer_continuum_rule(Z)
    model = _model()  # rest frame 1400 - 2067 A, below the 3646 A edge
    assert len(rule.propose(model, model.run_quick_fit())) <= 1

    red = _model(x=np.geomspace(3700 * (1 + Z), 4500 * (1 + Z), 400))
    assert rule.propose(red, red.run_quick_fit()) == []


def _balmer_hol(x, fwhm, hol_ratio):
    """Balmer continuum + high-order lines at z=2 (amp_be 1)."""
    return qso_advanced_models.BalmerContinuumHOLRatio()(
        x, Z, 1.0, 15000.0, 1.0, 3646.0, fwhm, hol_ratio
    )


def test_hol_ratio_is_defined_at_the_edge():
    """`hol_ratio` = lines over continuum at the edge, on any grid."""
    model = qso_advanced_models.BalmerContinuumHOLRatio()
    args = (Z, 1.0, 15000.0, 1.0, 3646.0, 4000.0)
    edge = np.array([3646.0 * (1 + Z)])
    no_lines = model(edge * (1 - 1e-9), *args, 0.0)[0]  # continuum just below
    lines_only = model(edge, *args, 0.7)[0]  # the continuum is 0 from the edge
    assert lines_only / no_lines == pytest.approx(0.7, rel=1e-3)

    coarse = np.geomspace(3300 * (1 + Z), 4300 * (1 + Z), 300)
    fine = np.geomspace(3300 * (1 + Z), 4300 * (1 + Z), 2991)  # contains coarse
    on_fine = model(fine, *args, 0.7)[::10]
    np.testing.assert_allclose(model(coarse, *args, 0.7), on_fine, rtol=1e-9)
    assert qso_advanced_models.BalmerContinuumHOLRatio().__name__ == (
        "balmer_continuum_hol_ratio"
    )


def test_balmer_rule_with_high_order_lines():
    """Data with the high-order lines: the HOL rule recovers their width
    and strength, and fits better than the sharp edge."""
    x = np.geomspace(3300 * (1 + Z), 4300 * (1 + Z), 900)
    shape = _balmer_hol(x, 4000.0, 1.0)
    model = _model(extra=lambda x: 0.3 * shape / shape.max(), x=x)

    hol = qr.balmer_continuum_rule(Z, hol=True)
    assert hol.name == "Balmer continuum + HOL"
    with_hol = AutoFit(model, [hol]).run()
    assert [c.name for c in with_hol.model.components] == ["PL", "BalmerCont"]
    par = with_hol.model.parameters
    assert par["BalmerCont_fwhm"].value == pytest.approx(4000.0, rel=0.15)
    assert par["BalmerCont_hol_ratio"].value == pytest.approx(1.0, rel=0.15)

    sharp = AutoFit(model, [qr.balmer_continuum_rule(Z)]).run()
    assert with_hol.result.red_chisq < sharp.result.red_chisq

    names = [getattr(r, "name", None) for r in qr.qso_rules(Z, balmer="hol")]
    assert "Balmer continuum + HOL" in names and "Balmer continuum" not in names
    with pytest.raises(ValueError, match="balmer"):
        qr.qso_rules(Z, balmer="yes")


def test_qso_rules_bundle():
    rules = qr.qso_rules(Z)
    names = [getattr(r, "prefix", None) or r.name for r in rules]
    kinds = [type(r).__name__ for r in rules]
    # "CIII" is the complex (and its second component), not free Gaussians
    assert "CIII" not in names
    assert kinds.count("GaussianLineRule") == len(qr.DEFAULT_LINES) - 1
    assert "CIII] complex" in names and "CIII] second component" in names
    # VW01 below 2200 A, TS06 at 2200-3500 A, BG92
    assert [n for n in names if n.startswith("iron")] == [
        "iron (VW01_FUV)",
        "iron (Tsuz_UV)",
        "iron (BG92_OPT)",
    ]
    assert "Balmer continuum + HOL" in names
    assert "OI" in names  # OI 1304 / SiII 1306 blend
    oi = next(r for r in rules if getattr(r, "prefix", None) == "OI")
    model = _model()
    for name in ("OIII_0", "OI_0"):
        model.components.append(sccomp.FitComponent(name, ana.line_model_gaussian))
    assert oi.owned(model) == ["OI_0"]  # not [OIII]
    assert kinds.count("Tie") == 0  # optional: the group is the only route
    group = next(r for r in rules if isinstance(r, LineGroupRule))
    assert [m.prefix for m in group.members] == ["Hbeta", "Halpha", "OIII"]
    assert group.fwhm_range == (100.0, qr.NARROW_FWHM_MAX)
    assert group.velocity_range == qr.NARROW_VELOCITY_RANGE
    bare = qr.qso_rules(
        Z, iron=None, balmer=False, narrow_group=False, ciii_complex=False
    )
    assert len(bare) == len(qr.DEFAULT_LINES)
    tie = qr.qso_rules(Z, narrow_tie=True)[-1]
    assert tie.prefixes == ["Hbeta", "Halpha", "OIII"]
    assert len(qr.qso_rules(Z, lines=["CIV", "Hbeta"], iron=None)) == 3


def test_oiii_doublet_rule():
    (rule,) = qr.line_rules(Z, ["OIII"])
    assert rule.lines == [5008.24]
    assert rule.fixed == {"fluxratio": 2.98}
    assert rule.fwhm_range == (100.0, 2500.0)

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
        _narrow_capable_hbeta(z),
        *qr.line_rules(z, ["OIII"]),
        qr.narrow_line_tie(["Hbeta", "OIII"]),
    ]
    return model, rules, z


def _narrow_capable_hbeta(z):
    """An Hbeta rule that may add narrow Gaussians too (the default one is
    broad only): a custom rule set for `Tie` and `null_removals`."""
    return GaussianLineRule(
        "Hbeta",
        redshift=z,
        lines=[qr.QSO_LINES["Hbeta"]],
        fwhm_range=(100.0, 15000.0),
        velocity_range=5000.0,
        max_components=3,
    )


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
    # Broad only: above NARROW_FWHM_MAX, at least the broad minimum
    assert rules["Hbeta"].fwhm_range == (1000.0, 15000.0)
    assert rules["CIV"].fwhm_range == (1000.0, 15000.0)
    assert rules["OIII"].fwhm_range == (100.0, 2500.0)


def _weak_narrow_model(narrow_flux, dv=0.0):
    """Broad Hbeta and a blueshifted [OIII] outflow (1500 km/s at -800
    km/s), with weak narrow Hbeta and [OIII] cores (400 km/s, offset by
    `dv` km/s from the systemic redshift) at z = 0.2."""
    z = 0.2
    x = np.geomspace(4700 * (1 + z), 5100 * (1 + z), 900)
    model = _model(x=x)
    model.parameters["PL_redsh"].value = z
    z_nl = z + dv / 2.998e5 * (1 + z)
    z_out = z - 800.0 / 2.998e5 * (1 + z)
    doublet = qso_presets.qso.line_model_gaussian_oiii_doublet
    model.spec.fluxden = (
        model.spec.fluxden
        + ana.line_model_gaussian(x, z, 40.0, 4862.68, 4000.0)
        + doublet(x, z_out, 8.0, 1500.0, 2.98)
        + ana.line_model_gaussian(x, z_nl, narrow_flux, 4862.68, 400.0)
        + doublet(x, z_nl, narrow_flux, 400.0, 2.98)
    )
    lines = qr.line_rules(z, ["Hbeta", "OIII"])
    return model, lines, z


def _narrow_members(comps):
    return sorted(
        name
        for name, comp in comps.items()
        if comp.param_mapping.get("redsh") == "narrow_redsh"
    )


def test_narrow_line_group_found_at_systemic():
    model, lines, z = _weak_narrow_model(1.0)
    group = qr.narrow_line_group(lines, z)
    rules = [*lines, group, qr.narrow_line_tie(["Hbeta", "OIII"])]
    result = AutoFit(model, rules).run()

    comps = {c.name: c for c in result.model.components}
    tied = _narrow_members(comps)
    assert {name.split("_")[0] for name in tied} == {"Hbeta", "OIII"}
    par = result.model.parameters
    assert par["narrow_redsh"].value == pytest.approx(z, abs=3e-4)
    assert par["narrow_fwhm_km_s"].value == pytest.approx(400, rel=0.2)
    steps = [step.description for step in result.history]
    group_step = next(i for i, d in enumerate(steps) if "narrow group" in d)
    assert group_step > 2  # after the broad Hbeta and the [OIII] outflow
    assert result.ties["narrow"].status == "tied"  # the tie sees the group


def test_backward_step_keeps_a_line_the_group_cannot_replace():
    """Without the broad Hbeta the narrow group's width goes to its bound:
    the backward step does not remove the broad line, however high its
    rule's threshold (J1340 round 22a, q99)."""
    model, lines, z = _weak_narrow_model(1.0)
    rules = [*lines, qr.narrow_line_group(lines, z)]
    found = AutoFit(model, rules).run()
    names = [c.name for c in found.model.components]
    assert "Hbeta_0" in names and "narrow_fwhm_km_s" in found.model.parameters

    search = AutoFit(found.model, rules)
    without = search._without(found.model, "Hbeta_0")
    assert "narrow_fwhm_km_s" in search._fit(without)[0].at_bound

    hbeta = search.rule_keys[next(i for i, r in enumerate(rules) if r is lines[0])]
    assert lines[0].prefix == "Hbeta"
    search._rule_thresholds = {hbeta: 1e9}
    kept, _, _ = search._backward(
        found.model, found.result, search._fit(found.model)[1], [], set(), 10.0
    )
    assert "Hbeta_0" in [c.name for c in kept.components]


def test_narrow_line_group_starts_at_a_residual_peak():
    model, lines, z = _weak_narrow_model(1.5, dv=500.0)
    group = qr.narrow_line_group(lines, z, velocity_range=1000.0)
    result = AutoFit(model, [*lines, group]).run()
    par = result.model.parameters
    z_nl = z + 500.0 / 2.998e5 * (1 + z)
    assert par["narrow_redsh"].value == pytest.approx(z_nl, abs=3e-4)


def test_narrow_line_group_needs_two_lines():
    lines = qr.line_rules(Z, ["CIV", "Hbeta"])
    assert qr.narrow_line_group(lines, Z) is None
    with pytest.raises(ValueError, match="one line"):
        LineGroupRule("g", [GaussianLineRule("a", Z), lines[1]], Z)


def test_narrow_line_group_waits_for_the_members_and_proposes_once():
    model, lines, z = _weak_narrow_model(1.0)
    group = qr.narrow_line_group(lines, z)
    assert group.propose(model, model.run_quick_fit()) == []  # no members yet
    early = qr.narrow_line_group(lines, z, after_members=False)
    assert len(early.propose(model, model.run_quick_fit())) == 1

    fitted = AutoFit(model, lines).run()  # broad Hbeta and [OIII], no group
    (cand,) = group.propose(fitted.model, fitted.result)
    assert cand.reject_at_bound == ["narrow_redsh", "narrow_fwhm_km_s"]
    fluxes = [p for p in cand.parameters if p.endswith("_flux")]
    assert len(fluxes) == 2
    assert all(cand.parameters[f].value > 0 for f in fluxes)
    grouped = AutoFit(model, [*lines, group]).run()
    assert group.propose(grouped.model, grouped.result) == []


def test_narrow_line_group_waits_for_hbeta_not_oiii():
    """[OIII] is narrow only and may be too weak to be added alone: the
    group waits for the broad Hbeta only."""
    model, lines, z = _weak_narrow_model(1.0)
    group = qr.narrow_line_group(lines, z)
    assert group.after_members == ["Hbeta"]
    hbeta = next(r for r in lines if r.prefix == "Hbeta")
    only_hbeta = AutoFit(model, [hbeta]).run()  # broad Hbeta, no [OIII]
    assert not any(c.name.startswith("OIII") for c in only_hbeta.model.components)
    assert len(group.propose(only_hbeta.model, only_hbeta.result)) == 1
    both = qr.narrow_line_group(lines, z, after_members=True)
    assert both.propose(only_hbeta.model, only_hbeta.result) == []


def test_group_rejected_when_not_narrow():
    """A group whose width ends at the narrow limit is rejected: here the
    only narrow-ish emission is 1500 km/s wide."""
    model, lines, z = _weak_narrow_model(0.0)
    group = qr.narrow_line_group(lines, z)
    result = AutoFit(model, [*lines, group]).run()
    assert "narrow_redsh" not in result.model.parameters


def test_group_removed_as_a_whole():
    """Backward steps test the group as one unit: here its narrow Hbeta
    alone is worth less than the threshold (~36), the pair more (~235)."""
    model, lines, z = _weak_narrow_model(1.0)
    group = qr.narrow_line_group(lines, z)
    fit = AutoFit(model, [*lines, group], threshold=50.0)
    result = fit.run()
    tied = _narrow_members({c.name: c for c in result.model.components})
    assert {name.split("_")[0] for name in tied} == {"Hbeta", "OIII"}
    assert sorted(tied) in [sorted(u) for u in fit._removal_units(result.model)]
    assert [tied[0]] not in fit._removal_units(result.model)


def test_group_rule_calibrated_without_its_group():
    """The null model already has the narrow group: its rule is calibrated
    on simulations without it, and gets its own threshold."""
    model, lines, z = _weak_narrow_model(1.0)
    group = qr.narrow_line_group(lines, z)
    rules = [*lines, group]
    found = AutoFit(model, rules).run()
    assert "narrow_redsh" in found.model.parameters
    cal = autofit.calibrate_threshold(
        found.model, rules, n_sims=12, quantiles=[0.9], min_rule_sims=10
    )
    assert "narrow" in cal.by_rule
    assert cal.by_rule["narrow"][0.9] <= cal.thresholds[0.9]
    assert cal.rule_improvements["narrow"].size >= 10


def test_group_calibrated_without_narrow_lines_added_alone(monkeypatch):
    """A narrow Hbeta the Hbeta rule added on its own is the group's
    hypothesis too: the group is calibrated on simulations without it, not
    on adding a second narrow Hbeta."""
    model, lines, z = _weak_narrow_model(1.5)
    lines = [_narrow_capable_hbeta(z), *(r for r in lines if r.prefix != "Hbeta")]
    found = AutoFit(model, lines, threshold=10.0).run().model
    group = qr.narrow_line_group(lines, z)
    assert group.guarded(found) == []  # not the group's own components
    assert sorted(group.null_removals(found)) == ["Hbeta_1", "OIII_1"]

    removed = []
    without = autofit.AutoFit._without

    def spy(model, name, copy_model=True):
        removed.append(name)
        return without(model, name, copy_model)

    monkeypatch.setattr(autofit.AutoFit, "_without", staticmethod(spy))
    cal = autofit.calibrate_threshold(
        found, [*lines, group], n_sims=12, quantiles=[0.9], min_rule_sims=10
    )
    assert {"Hbeta_1", "OIII_1"} <= set(removed)
    assert "narrow" in cal.rule_improvements


def test_group_rejected_when_not_systemic():
    """Cores 600 km/s from the systemic redshift are outside the default
    window: the group would end at its redshift bound."""
    model, lines, z = _weak_narrow_model(1.5, dv=600.0)
    group = qr.narrow_line_group(lines, z)
    result = AutoFit(model, [*lines, group]).run()
    assert "narrow_redsh" not in result.model.parameters


def test_group_that_drifts_to_a_bound_is_removed_at_convergence():
    """A group already in the model whose redshift ends at its bound is
    removed when the search converges."""
    model, lines, z = _weak_narrow_model(1.5, dv=600.0)
    wide = qr.narrow_line_group(lines, z, velocity_range=1000.0)
    start = AutoFit(model, [*lines, wide]).run().model
    assert "narrow_redsh" in start.parameters
    dz = 300.0 / 2.998e5 * (1 + z)
    start.parameters["narrow_redsh"].prior = scpri.UniformPrior(
        "narrow_redsh", z - dz, z + dz
    )
    start.parameters["narrow_redsh"].value = z

    # Only the group's rule: no other change can take its place first
    result = AutoFit(start, [qr.narrow_line_group(lines, z)]).run()
    assert "narrow_redsh" not in result.model.parameters
    removal = next(s for s in result.history if "at a prior bound" in s.description)
    assert removal.action == "remove"
    assert "narrow_redsh" in removal.description


def test_narrow_line_windows():
    (window,) = qr.narrow_line_windows(7.0, lines=["OIII_5008"], velocity=300.0)
    centre = 5008.24 * 8.0
    assert sum(window) / 2 == pytest.approx(centre)
    assert (window[1] - window[0]) / centre == pytest.approx(600.0 / 2.998e5, rel=1e-3)


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
            3000.0,
            0.0,
            8.0,
            3000.0,
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
    (comp,) = cand.components
    assert comp.param_mapping["fwhm_km_s_siIII"] == "CIIIc_fwhm_km_s"
    assert "CIIIc_fwhm_km_s_alIII" not in cand.parameters
    assert len(cand.starts) == 2
    assert all("CIIIc_flux_alIII" in start for start in cand.starts)

    result = AutoFit(model, [rule]).run()
    par = result.model.parameters
    assert [c.name for c in result.model.components] == ["PL", "CIIIc"]
    assert par["CIIIc_flux"].value == pytest.approx(30.0, rel=0.1)
    assert par["CIIIc_flux_siIII"].value == pytest.approx(8.0, rel=0.3)
    assert par["CIIIc_flux_alIII"].value == pytest.approx(6.0, rel=0.3)
    assert par["CIIIc_fwhm_km_s"].value == pytest.approx(3000.0, rel=0.1)

    free = qr.ciii_complex_rule(Z, shared_width=False)
    (cand,) = free.propose(model, model.run_quick_fit())
    assert "CIIIc_fwhm_km_s_alIII" in cand.parameters


def _ciii_complex(x, fwhm=3000.0):
    return qso_presets.qso.CIII_complex_model_func(
        x, Z, 1908.73, 1857.40, 1892.03, 30.0, fwhm, 0.0, 6.0, fwhm, 0.0, 8.0,
        fwhm, 0.0,
    )  # fmt: skip


def test_ciii_second_component_rule():
    rules = [qr.ciii_complex_rule(Z), qr.ciii_second_component_rule()]
    second = rules[1]
    # Not without the complex
    plain = _model(_ciii_complex)
    assert second.propose(plain, plain.run_quick_fit()) == []

    # A sharper CIII] core on top of the complex: a second CIII] Gaussian at
    # the complex's redshift
    def cored(x):
        return _ciii_complex(x, fwhm=4000.0) + ana.line_model_gaussian(
            x, Z, 20.0, 1908.73, 1500.0
        )

    model = _model(cored)
    only = AutoFit(model, rules[:1], threshold=10.0).run().model
    (cand,) = second.propose(only, only.run_quick_fit())
    assert cand.reject_at_bound == ["CIIIc_b_fwhm_km_s", "CIIIc_fwhm_km_s"]
    assert "CIIIc_redsh" not in cand.parameters

    result = AutoFit(model, rules, threshold=10.0).run()
    names = [c.name for c in result.model.components]
    assert names == ["PL", "CIIIc", "CIIIc_b"]
    comp = result.model.components[2]
    assert comp.param_mapping["redsh"] == "CIIIc_redsh"
    assert "CIIIc_b_redsh" not in result.model.parameters
    widths = sorted(
        result.model.parameters[k].value
        for k in ("CIIIc_fwhm_km_s", "CIIIc_b_fwhm_km_s")
    )
    assert widths[0] == pytest.approx(1500.0, rel=0.2)
    assert result.model.parameters["CIIIc_redsh"].value == pytest.approx(Z, abs=1e-3)

    # One width is enough for a plain complex
    result = AutoFit(_model(_ciii_complex), rules, threshold=10.0).run()
    assert [c.name for c in result.model.components] == ["PL", "CIIIc"]


def test_qso_rules_ciii_complex_option():
    names = [getattr(r, "name", "") for r in qr.qso_rules(Z, ciii_second=False)]
    assert "CIII] complex" in names and "CIII] second component" not in names
    plain = qr.qso_rules(Z, ciii_complex=False)
    assert "CIII" in [getattr(r, "prefix", "") for r in plain]
    assert "CIII] complex" not in [getattr(r, "name", "") for r in plain]
