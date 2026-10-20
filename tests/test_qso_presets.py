"""Tests for the base functions in `Qso_autofit` / `qso` / `qso_advanced_models`
and their named variants in `sculptor_extensions.qso_presets`."""

import inspect

import numpy as np
import pytest
from astropy import units as u
from sculptor_extensions.component_setup import initialize_component

from sculptor import analysis as ana
from sculptor.model import FitModel
from sculptor.prior import UniformPrior
from sculptor_extensions import Qso_autofit as qa
from sculptor_extensions import qso, qso_presets
from sculptor_extensions import qso_advanced_models as qam

KM_S = u.km / u.s


def _fixed(value):
    return {"value": value, "vary": False}


def _fe_dict():
    # The template-unit entries are only used by the slow `template_model`.
    return {
        "redsh": _fixed(2.0),
        "amp": {"value": 1.0, "vary": True, "prior": UniformPrior("amp", 0, 10)},
        "fwhm": _fixed(2000.0),
        "intr_fwhm": _fixed(900.0),
        "templ_disp_unit_str": _fixed(None),
        "templ_fluxden_unit_str": _fixed(None),
    }


# --- initialize_component -------------------------------------------------


def test_initialize_component_maps_parameters_onto_others():
    """A "parameter" entry drops the component's own parameter and maps it.
    The old per-function loops popped from the dict they iterated over and
    raised `RuntimeError` here (all but the Gaussian line setup)."""
    settings = {
        "redsh": {"parameter": "global_redsh"},
        "amp": _fixed(1.0),
        "slope": {"value": -1.5, "vary": True, "prior": UniformPrior("slope", -3, 0)},
    }
    comp, params = qa.initialize_power_law("PL", 2.0, settings)

    assert sorted(params) == ["PL_amp", "PL_slope"]
    assert comp.param_mapping["redsh"] == "global_redsh"
    assert params["PL_slope"].vary
    assert not params["PL_amp"].vary


def test_initialize_component_sets_fixed_values_and_requires_a_dict():
    disp, flux = np.linspace(2000, 3000, 11), np.ones(11)
    _, params = qa.initialize_Fe("Fe", disp, flux, _fe_dict())
    np.testing.assert_array_equal(params["Fe_templ_disp"].value, disp)
    assert not params["Fe_templ_fluxden"].vary

    with pytest.raises(ValueError, match="Provide `balmer_dict`"):
        qa.initialize_balmer_without_hol("BC", None)
    with pytest.raises(ValueError, match="Provide `param_dict`"):
        initialize_component("G", ana.line_model_gaussian, {})


# --- iron templates ---------------------------------------------------------

IRON_PRESETS = [n for n in dir(qso_presets) if n.startswith("setup_iron_template_")]


def test_ten_iron_presets_with_fast_default():
    assert len(IRON_PRESETS) == 10
    for name in IRON_PRESETS:
        sig = inspect.signature(getattr(qso_presets, name))
        assert sig.parameters["fast"].default is True


@pytest.mark.parametrize("name", IRON_PRESETS)
def test_every_iron_preset_builds_a_component(name):
    """All presets load their data file; the Park et al. 2022 one used to
    point at a file that does not exist (`Fe_OPT_Hbeta_Park22.txt`)."""
    comp, params = getattr(qso_presets, name)("Fe", fe_dict=_fe_dict())
    assert isinstance(comp.function, ana.TemplateModel)
    assert sorted(params) == ["Fe_amp", "Fe_fwhm", "Fe_intr_fwhm", "Fe_redsh"]


def test_setup_iron_template_cuts_an_inclusive_range():
    comp, _ = qa.setup_iron_template(
        "Fe", "Fe_UVtemplt_A.asc", [2200, 3090], _fe_dict(), fast=False
    )
    assert comp.function is ana.template_model

    disp = np.loadtxt(qa.datadir + "iron_templates/Fe_UVtemplt_A.asc")[:, 0]
    expected = disp[(disp >= 2200) & (disp <= 3090)]
    fast, _ = qso_presets.setup_iron_template_VW01_UV_MgII("Fe", _fe_dict())
    np.testing.assert_array_equal(fast.function.templ_disp, expected)


def test_tsuzuki_optical_default_starts_above_3500():
    comp, _ = qso_presets.setup_iron_template_Tsuz_OPT("Fe", fe_dict=_fe_dict())
    assert comp.function.templ_disp.min() > 3500


# --- emission lines -----------------------------------------------------------


def _line_dict(*lines_and_kinds):
    params = {
        "redsh": _fixed(2.0),
        "flux": _fixed(10.0),
        "cen": _fixed(2798.75),
        "fwhm_km_s": _fixed(3000.0),
        "fluxratio": _fixed(3.0),
    }
    line_dict = {}
    for line, kind in lines_and_kinds:
        line_dict.setdefault(line, {})[kind] = {"params": params}
    return line_dict


def test_add_line_components_names():
    model = FitModel()
    line_dict = _line_dict(("MgII", "narrow"), ("MgII", "broad"))
    qa.add_line_components(model, line_dict, "MgII", "narrow")
    qa.add_line_components(model, line_dict, "MgII", "broad", 3)
    qa.add_line_components(model, line_dict, "MgII", "broad", 0)
    names = [c.name for c in model.components]
    assert names == ["MgII_narrow", "MgII_broad_a", "MgII_broad_b", "MgII_broad_c"]


def test_setup_hb_o3_lines_preset():
    model = FitModel()
    line_dict = _line_dict(
        ("Hbeta", "narrow"),
        ("OIII", "narrow"),
        ("Hbeta", "broad"),
        ("OIII", "wind"),
    )
    qso_presets.setup_Hbeta_OIII_lines(
        None, line_dict, Hbeta_broad=2, OIII_wind=1, model=model
    )
    names = [c.name for c in model.components]
    assert names == [
        "Hbeta_narrow",
        "OIII_narrow",
        "Hbeta_broad_a",
        "Hbeta_broad_b",
        "OIII_wind_a",
    ]
    assert model.components[1].function is qso.line_model_gaussian_oiii_doublet


# --- shared model code ----------------------------------------------------------


def test_doublets_are_two_gaussian_lines():
    x = np.linspace(19000, 21000, 2001)
    expected = ana.line_model_gaussian(x, 2.0, 5.0, 6549.85, 300.0)
    expected += ana.line_model_gaussian(x, 2.0, 5.0 * 2.96, 6585.28, 300.0)
    np.testing.assert_array_equal(
        qa.line_model_gaussian_NII_doublet(x, 2.0, 5.0, 300.0), expected
    )
    np.testing.assert_array_equal(
        qso.line_model_gaussian_nii_doublet(x, 2.0, 5.0, 5.0 * 2.96, 300.0, 300.0),
        expected,
    )


def test_virial_bh_mass():
    mass = qso.virial_bh_mass(
        3000 * KM_S, 1e45 * u.erg / u.s, zero_point=10**6.91, width_exp=2, lum_exp=0.5
    )
    assert mass.unit == u.Msun
    assert mass.value == pytest.approx(10**6.91 * 3**2 * 10**0.5)


def test_db20_matches_its_log_form():
    fwhm, lum, f = 4000 * KM_S, 3e42 * u.erg / u.s, 1.5
    mass, _ = qso.se_bhmass_db20(fwhm, lum, virial_factor=f)
    log_mass = (
        np.log10(f)
        + 7.015
        + 0.784 * (np.log10(3e42) - 42)
        + 1.387 * (np.log10(4000) - 3.5)
    )
    assert mass.to_value(u.Msun) == pytest.approx(10**log_mass, rel=1e-12)


def test_balmer_models_share_the_shape():
    x = np.linspace(2000, 4000, 401)
    redsh, Te, tau_be, lambda_be = 0.5, 15000.0, 1.0, 3646.0
    x_obs = x * (1 + redsh)
    shape = qam.balmer_continuum_shape(x, Te, tau_be, lambda_be)
    assert np.all(shape[x >= lambda_be] == 0)

    fast = qam.fast_balmer_continuum_model(x_obs, redsh, 2.0, Te, tau_be, lambda_be)
    np.testing.assert_allclose(fast, 2.0 * shape * 1e-20, rtol=1e-14)

    with_pl = qso.power_law_at_2500_plus_bc(
        x_obs, 1.0, -1.5, redsh, 3.0, Te, tau_be, lambda_be
    )
    edge = qam.balmer_continuum_at_edge(Te, tau_be, lambda_be)
    expected = ana.power_law_at2500(x_obs, redsh, 1.0, -1.5) + 3.0 * shape / edge
    np.testing.assert_allclose(with_pl, expected, rtol=1e-14)


def test_cached_hol_model_is_the_uncached_one():
    x = np.linspace(4000, 6000, 501)
    args = (x, 0.5, 1.0, 15000.0, 1.0, 3646.0, 2000.0)
    cached = qam.CachedBalmerContinuumWithHOL()
    assert isinstance(cached, qam.BalmerContinuumWithHOL)
    np.testing.assert_array_equal(cached(*args), qam.BalmerContinuumWithHOL()(*args))
    np.testing.assert_array_equal(cached(*args), qam.BalmerContinuumWithHOL()(*args))


# --- line properties -----------------------------------------------------------


# --- old names ------------------------------------------------------------------


def test_moved_names_still_work_with_a_deprecation_warning():
    with pytest.warns(DeprecationWarning, match="qso_presets.setup_MgII_lines"):
        func = qa.setup_MgII_lines
    assert func is qso_presets.setup_MgII_lines
    with pytest.raises(AttributeError):
        qa.no_such_function  # noqa: B018


@pytest.mark.parametrize("old,new", list(qso_presets.RENAMED.items()))
def test_renamed_presets_forward_with_a_deprecation_warning(old, new):
    with pytest.warns(DeprecationWarning, match=new):
        assert getattr(qso_presets, old) is getattr(qso_presets, new)
    with pytest.warns(DeprecationWarning, match=f"qso_presets.{new}"):
        assert getattr(qa, old) is getattr(qso_presets, new)


@pytest.mark.parametrize("old,new", list(qa.RENAMED.items()))
def test_british_spellings_forward_with_a_deprecation_warning(old, new):
    with pytest.warns(DeprecationWarning, match=new):
        assert getattr(qa, old) is getattr(qa, new)


def test_qam_british_spelling_forwards():
    with pytest.warns(DeprecationWarning, match="initialize_balmer_with_hol"):
        assert qam.initialise_balmer_with_hol is qam.initialize_balmer_with_hol


def test_broadshifted_components_and_old_spellings():
    model = FitModel()
    line_dict = _line_dict(("MgII", "broad"), ("MgII", "broadshifted"))
    qso_presets.setup_MgII_lines(None, line_dict, MgII_broadshifted=2, model=model)
    assert [c.name for c in model.components] == [
        "MgII_broad_a",
        "MgII_broadshifted_a",
        "MgII_broadshifted_b",
    ]

    # Old keyword and old line_dict key still work, with a warning.
    model = FitModel()
    old_dict = _line_dict(("MgII", "broad"), ("MgII", "broad_shifted"))
    with pytest.warns(DeprecationWarning) as record:
        qso_presets.setup_MgII_lines(None, old_dict, MgII_broad_shifted=1, model=model)
    messages = " ".join(str(w.message) for w in record)
    assert "MgII_broadshifted" in messages and '"broadshifted"' in messages
    assert [c.name for c in model.components][-1] == "MgII_broadshifted_a"
