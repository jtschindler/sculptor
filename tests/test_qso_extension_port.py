"""Consistency checks between the ported `sculptor_extensions` QSO modules
(`qso.py`, `Qso_autofit.py`, `qso_advanced_models.py`) and the original
source they were ported from (`tests/ground_truth/fgu/`).

For code that was carried over verbatim (or with only additive/default-
preserving changes), these tests assert the ported implementation produces
identical output to the original. For code that was intentionally *not*
ported unchanged (e.g. dropped `get_civ_line_prop`, kept current's
`initialise_gaussian_line_model` over out.py's regression), the tests
assert the intended, documented behavior instead.

The "original" modules are loaded directly from the test-only ground-truth
directory `tests/ground_truth/fgu/` via `importlib`, rather than duplicated
into the installed package, since they only exist to anchor these one-off
comparisons.
"""

import importlib.util
import inspect
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest
import sculptor_extensions.qso as new_qso
import sculptor_extensions.qso_advanced_models as new_qam
import sculptor_extensions.Qso_autofit as new_qso_autofit
from astropy import units as u

from sculptor import analysis as sculptor_analysis

REPO_ROOT = Path(__file__).resolve().parents[1]
FGU_DIR = REPO_ROOT / "tests" / "ground_truth" / "fgu"
SCULPTOR_DATA_DIR = REPO_ROOT / "src" / "sculptor" / "data"


def _redsh(kwargs):
    """The original reference copies call the redshift `z`; the maintained
    model functions call it `redsh` (review §4)."""
    return {("redsh" if k == "z" else k): v for k, v in kwargs.items()}


def _load_module_from_path(module_name, path):
    """Import a standalone .py file as a module, without installing it."""
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def hol_data_file():
    """The high-order Balmer line data, packaged as
    `sculptor/data/balmer_hol_storey_hummer1995.txt` (formerly
    `wave_2.txt`). The original `out.py` reference copy still loads
    `wave_2.txt` at import, so provide a temporary copy under that name and
    remove it afterwards (only if this fixture created it).
    """
    data_file = SCULPTOR_DATA_DIR / new_qam.HOL_DATA_FILE
    assert data_file.exists(), data_file
    legacy = SCULPTOR_DATA_DIR / "wave_2.txt"
    created = not legacy.exists()
    if created:
        shutil.copyfile(data_file, legacy)
    try:
        yield data_file
    finally:
        if created and legacy.exists():
            legacy.unlink()


@pytest.fixture(scope="module")
def original_qso(deprecated_sculptor):
    # Ensure the ported module (which explicitly does
    # `import importlib.resources`) has already run first, so the bare
    # `import importlib` in the original file also gets a working
    # `importlib.resources` attribute (submodule imports are cached
    # process-wide once anything has imported them).
    assert new_qso is not None
    return _load_module_from_path("_original_fgu_qso", FGU_DIR / "qso.py")


@pytest.fixture(scope="module")
def original_qso_autofit(deprecated_sculptor):
    assert new_qso_autofit is not None
    return _load_module_from_path(
        "_original_fgu_qso_autofit", FGU_DIR / "Qso_autofit.py"
    )


@pytest.fixture(scope="module")
def original_out(hol_data_file, deprecated_sculptor):
    assert new_qam is not None
    return _load_module_from_path("_original_fgu_out", FGU_DIR / "out.py")


class _AnyDict(dict):
    """Auto-vivifying dict: any missing key returns a nested _AnyDict."""

    def __missing__(self, key):
        value = _AnyDict()
        self[key] = value
        return value


class _RecordingModel:
    """Stand-in for `sculptor.model.FitModel` that just records the
    component/parameter values passed to `add_component`/`add_parameters`.
    """

    def __init__(self):
        self.added_components = []
        self.added_parameters = []

    def add_component(self, comp):
        self.added_components.append(comp)

    def add_parameters(self, pars):
        self.added_parameters.append(pars)


# --------------------------------------------------------------------------
# qso.py -- wholesale-replaced from the fgu source; should match exactly.
# The power law, Gaussian line and Balmer continuum duplicates were removed
# from qso.py (review §3); the maintained replacements must reproduce the
# original functions.
# --------------------------------------------------------------------------


def test_power_law_at2500_matches_original_qso_power_law(original_qso):
    x = np.linspace(1000, 5000, 200)
    orig = original_qso.power_law_at_2500(x, amp=1.5, slope=-1.7, z=2.0)
    new = sculptor_analysis.power_law_at2500(x, redsh=2.0, amp=1.5, slope=-1.7)
    np.testing.assert_allclose(orig, new, rtol=1e-12)


def test_fast_balmer_continuum_matches_original_qso_balmer(original_qso):
    x = np.linspace(2000, 14000, 2000)
    kwargs = {
        "z": 2.0,
        "amp_be": 1.0,
        "Te": 15000.0,
        "tau_be": 1.0,
        "lambda_be": 3646.0,
    }
    orig = original_qso.balmer_continuum_model(x, **kwargs)
    new = new_qam.fast_balmer_continuum_model(x, **_redsh(kwargs))
    assert np.count_nonzero(orig) > 100  # the range spans the Balmer edge
    np.testing.assert_allclose(orig, new, rtol=1e-12)


def test_line_model_gaussian_matches_original_qso_gaussian(original_qso):
    x = np.linspace(1500, 1600, 300)
    kwargs = {"flux": 10.0, "cen": 1549.06, "fwhm_km_s": 2000.0}
    orig = original_qso.line_model_gaussian(x, z=2.0, **kwargs)
    new = sculptor_analysis.line_model_gaussian(x, redsh=2.0, **kwargs)
    np.testing.assert_allclose(orig, new, rtol=1e-12)


def _line_dict(**values):
    return {par: {"value": v, "vary": False} for par, v in values.items()}


@pytest.mark.parametrize(
    ("prefix", "tied_func", "untied_func", "ratio"),
    [
        (
            "NII",
            "line_model_gaussian_NII_doublet",
            "line_model_gaussian_nii_doublet",
            2.96,
        ),
        (
            "SII",
            "line_model_gaussian_SII_doublet",
            "line_model_gaussian_sii_doublet",
            1.0,
        ),
    ],
)
def test_doublet_switch_tied_by_default(prefix, tied_func, untied_func, ratio):
    tied_dict = _line_dict(redsh=0.1, flux=5.0, fwhm_km_s=300.0, fluxratio=ratio)
    comp, params = new_qso_autofit.initialise_gaussian_line_model(prefix, tied_dict)
    assert comp.function is getattr(new_qso_autofit, tied_func)
    assert sorted(params) == sorted(f"{prefix}_{p}" for p in tied_dict)

    untied_dict = _line_dict(
        redsh=0.1, flux_a=5.0, flux_b=5.0 * ratio, fwhm_km_s_a=300.0, fwhm_km_s_b=300.0
    )
    comp_u, params_u = new_qso_autofit.initialise_gaussian_line_model(
        prefix, untied_dict, tied_doublet=False
    )
    assert comp_u.function is getattr(new_qso, untied_func)
    assert sorted(params_u) == sorted(f"{prefix}_{p}" for p in untied_dict)

    # With equal widths and flux_b = ratio * flux_a, both models agree.
    x = np.linspace(7000, 7600, 2000)
    tied = comp.function(x, 0.1, 5.0, 300.0, ratio)
    untied = comp_u.function(x, 0.1, 5.0, 5.0 * ratio, 300.0, 300.0)
    np.testing.assert_allclose(tied, untied, rtol=1e-12)


def test_qso_duplicates_were_removed():
    for name in (
        "power_law_at_2500",
        "line_model_gaussian",
        "template_model",
        "balmer_continuum_model",
        "power_law_at_2500_plus_fractional_bc",
    ):
        assert not hasattr(new_qso, name), name


def test_initialise_balmer_without_hol_uses_fast_model():
    balmer_dict = {
        par: {"value": value, "vary": False}
        for par, value in {
            "redsh": 2.0,
            "amp_be": 1.0,
            "Te": 15000.0,
            "tau_be": 1.0,
            "lambda_be": 3646.0,
        }.items()
    }
    comp, params = new_qso_autofit.initialise_balmer_without_hol("BC", balmer_dict)

    assert comp.function is new_qam.fast_balmer_continuum_model
    assert sorted(params) == sorted(f"BC_{p}" for p in balmer_dict)


def test_qso_calc_velocity_shifts_matches_original(original_qso):
    orig = original_qso.calc_velocity_shifts(2.005, z_sys=2.0, relativistic=True)
    new = new_qso.calc_velocity_shifts(2.005, z_sys=2.0, relativistic=True)
    np.testing.assert_allclose(orig.to("km/s").value, new.to("km/s").value)


def test_qso_eddington_luminosity_and_ratio_match_original(original_qso):
    bh_mass = 1e8 * u.Msun
    orig_edd = original_qso.calc_eddington_luminosity(bh_mass)
    new_edd = new_qso.calc_eddington_luminosity(bh_mass)
    np.testing.assert_allclose(orig_edd.cgs.value, new_edd.cgs.value)

    lbol = 1e45 * u.erg / u.s
    orig_ratio = original_qso.calc_eddington_ratio(lbol, bh_mass)
    new_ratio = new_qso.calc_eddington_ratio(lbol, bh_mass)
    np.testing.assert_allclose(
        orig_ratio.decompose().value, new_ratio.decompose().value
    )


def test_qso_bhmass_functions_match_original(original_qso):
    cont_lwav = 1e44 * u.erg / u.s / u.AA

    orig_mass, orig_ref = original_qso.se_bhmass_hbeta_vp06(
        2000 * u.km / u.s, cont_lwav, cont_wav=5100 * u.AA
    )
    new_mass, new_ref = new_qso.se_bhmass_hbeta_vp06(
        2000 * u.km / u.s, cont_lwav, cont_wav=5100 * u.AA
    )
    assert orig_ref == new_ref
    np.testing.assert_allclose(orig_mass.to(u.Msun).value, new_mass.to(u.Msun).value)

    orig_mass, orig_ref = original_qso.se_bhmass_mgii_s11_fwhm(
        3000 * u.km / u.s, cont_lwav, cont_wav=3000 * u.AA
    )
    new_mass, new_ref = new_qso.se_bhmass_mgii_s11_fwhm(
        3000 * u.km / u.s, cont_lwav, cont_wav=3000 * u.AA
    )
    assert orig_ref == new_ref
    np.testing.assert_allclose(orig_mass.to(u.Msun).value, new_mass.to(u.Msun).value)

    orig_mass, orig_ref = original_qso.se_bhmass_mgii_vo09_fwhm(
        3000 * u.km / u.s, cont_lwav, cont_wav=3000 * u.AA
    )
    new_mass, new_ref = new_qso.se_bhmass_mgii_vo09_fwhm(
        3000 * u.km / u.s, cont_lwav, cont_wav=3000 * u.AA
    )
    assert orig_ref == new_ref
    np.testing.assert_allclose(orig_mass.to(u.Msun).value, new_mass.to(u.Msun).value)

    orig_lbol, orig_ref = original_qso.calc_bolometric_luminosity(
        cont_lwav, cont_wav=1350 * u.AA
    )
    new_lbol, new_ref = new_qso.calc_bolometric_luminosity(
        cont_lwav, cont_wav=1350 * u.AA
    )
    assert orig_ref == new_ref
    np.testing.assert_allclose(orig_lbol.cgs.value, new_lbol.cgs.value)


@pytest.mark.parametrize(
    "func_name,coefficient,reference",
    [
        ("se_bhmass_civ_vp06_fwhm", 6.66, "CIV_VP06_fwhm"),
        ("se_bhmass_civ_vp06_sigma", 6.73, "CIV_VP06_sigma"),
        ("se_bhmass_civ_c17_fwhm", 6.71, "CIV_C17_fwhm"),
    ],
)
def test_qso_civ_bhmass_functions_units_bug_was_fixed(
    original_qso, func_name, coefficient, reference
):
    """The fgu source (and, until this fix, the wholesale-ported copy) used
    `u.kms` -- not a real astropy unit, should be `u.km / u.s` -- in all
    three CIV BH-mass estimators, so calling any of them raised
    `AttributeError`. The original (in the untracked scratch dir) still has
    the bug; the ported `qso.py` has been fixed. Verify the original still
    fails, and that the fixed function now returns a value matching an
    independently recomputed reference formula (not just "didn't crash").
    """
    cont_lwav = 1e44 * u.erg / u.s / u.AA
    cont_wav = 1350 * u.AA
    line_width = 5000 * u.km / u.s
    arg_name = "civ_sigma" if func_name == "se_bhmass_civ_vp06_sigma" else "civ_fwhm"
    kwargs = {"cont_lwav": cont_lwav, "cont_wav": cont_wav, arg_name: line_width}

    with pytest.raises(AttributeError, match="kms"):
        getattr(original_qso, func_name)(**kwargs)

    new_mass, new_ref = getattr(new_qso, func_name)(**kwargs)
    assert new_ref == reference

    expected_mass = (
        10**coefficient
        * (line_width / (1000.0 * u.km / u.s)) ** 2
        * (cont_wav * cont_lwav / (10**44 * u.erg / u.s)) ** 0.53
        * u.Msun
    )
    np.testing.assert_allclose(
        new_mass.to(u.Msun).value, expected_mass.to(u.Msun).value
    )


def test_qso_correct_civ_fwhm_for_blueshift_matches_original(original_qso):
    orig = original_qso.correct_CIV_fwhm_for_blueshift(
        5000 * u.km / u.s, 500 * u.km / u.s
    )
    new = new_qso.correct_CIV_fwhm_for_blueshift(5000 * u.km / u.s, 500 * u.km / u.s)
    np.testing.assert_allclose(orig.to("km/s").value, new.to("km/s").value)


# --------------------------------------------------------------------------
# Qso_autofit.py -- verbatim-ported functions should match the fgu source
# exactly; additive-parameter functions should match at default arguments.
# --------------------------------------------------------------------------


def test_qso_autofit_get_1sigma_range_matches_original(original_qso_autofit):
    rng = np.random.default_rng(0)
    data = rng.normal(size=5000)
    orig = original_qso_autofit.get_1sigma_range(data)
    new = new_qso_autofit.get_1sigma_range(data)
    np.testing.assert_allclose(orig, new)


def test_qso_autofit_extract_median_params_matches_original(original_qso_autofit):
    median_param_dict = {
        "CIV_broad_a_flux": 1.0,
        "CIV_broad_a_fwhm_km_s": 2000.0,
        "MgII_broad_a_flux": 3.0,
    }
    orig = original_qso_autofit.extract_median_params("CIV", median_param_dict)
    new = new_qso_autofit.extract_median_params("CIV", median_param_dict)
    assert orig == new


def test_qso_autofit_get_median_params_matches_original(original_qso_autofit):
    rng = np.random.default_rng(1)
    chain = rng.normal(loc=[1.0, 2.0], scale=[0.1, 0.5], size=(2000, 2))

    class _Sampler:
        def get_chain(self, discard=0, thin=1, flat=True):
            return chain[discard:]

    class _Model:
        def __init__(self):
            self.sampler = _Sampler()
            self.params_variable = {"amp": None, "slope": None}

    orig = original_qso_autofit.get_median_params(_Model(), discard=0)
    new = new_qso_autofit.get_median_params(_Model(), discard=0)
    assert orig.keys() == new.keys()
    for key in orig:
        np.testing.assert_allclose(orig[key], new[key])


# --------------------------------------------------------------------------
# The `median_params` alternate-input mechanism (dict-or-median_params) has
# been removed from the `initialise_*`/`setup_iron_template_*` functions,
# leaving only the `*_dict`-based input. `extract_median_params`,
# `get_median_params`, and `extract_median_params_from_summary` are
# unaffected -- they produce/extract a median-values dict, they don't
# consume one as an alternate input.
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "func_name",
    [
        "initialise_power_law",
        "initialise_Fe",
        "initialise_balmer_without_hol",
        "initialise_balmer",
        "setup_iron_template_VW01_UV",
    ],
)
def test_qso_autofit_median_params_no_longer_a_parameter(func_name):
    sig = inspect.signature(getattr(new_qso_autofit, func_name))
    assert "median_params" not in sig.parameters


def test_qam_median_params_no_longer_a_parameter():
    sig = inspect.signature(new_qam.initialise_balmer_with_hol)
    assert "median_params" not in sig.parameters


def test_qso_autofit_initialise_power_law_raises_without_dict():
    with pytest.raises(ValueError, match="powerlaw_dict"):
        new_qso_autofit.initialise_power_law("test", 0.0)


def test_qso_autofit_initialise_fe_raises_without_dict():
    with pytest.raises(ValueError, match="Fe_dict"):
        new_qso_autofit.initialise_Fe(
            "test", np.array([1.0, 2.0]), np.array([1.0, 2.0])
        )


def test_qso_autofit_initialise_balmer_without_hol_raises_without_dict():
    with pytest.raises(ValueError, match="balmer_dict"):
        new_qso_autofit.initialise_balmer_without_hol("test")


def test_qam_initialise_balmer_with_hol_raises_without_dict():
    with pytest.raises(ValueError, match="balmer_dict"):
        new_qam.initialise_balmer_with_hol("test")


def test_qso_autofit_doublet_line_models_match_original(original_qso_autofit):
    x = np.linspace(6500, 6620, 300)
    orig = original_qso_autofit.line_model_gaussian_NII_doublet(
        x, redsh=2.0, flux=10.0, fwhm_km_s=500.0, fluxratio=2.96
    )
    new = new_qso_autofit.line_model_gaussian_NII_doublet(
        x, redsh=2.0, flux=10.0, fwhm_km_s=500.0, fluxratio=2.96
    )
    np.testing.assert_allclose(orig, new)

    orig = original_qso_autofit.line_model_gaussian_SII_doublet(
        x, redsh=2.0, flux=10.0, fwhm_km_s=500.0, fluxratio=1.0
    )
    new = new_qso_autofit.line_model_gaussian_SII_doublet(
        x, redsh=2.0, flux=10.0, fwhm_km_s=500.0, fluxratio=1.0
    )
    np.testing.assert_allclose(orig, new)


_IRON_TEMPLATE_DATA_FILE = {
    "setup_iron_template_VW01_UV": "Fe_UVtemplt_A.asc",
    "setup_iron_template_BG92_OPT": "Fe_OPT_BR92_linear.txt",
    "setup_iron_template_Tsuz_UV": "Tsuzuki06.txt",
    "setup_iron_template_Tsuz_OPT": "Tsuzuki06.txt",
}


@pytest.mark.parametrize("func_name", list(_IRON_TEMPLATE_DATA_FILE))
def test_qso_autofit_iron_template_default_disp_range_matches_original(
    original_qso_autofit, monkeypatch, func_name
):
    """The ported functions gained an additive `disp_range=None` parameter
    (merged in from out.py). At the default (no `disp_range` passed), they
    must select the exact same dispersion/flux arrays as the original fgu
    function, which doesn't have that parameter at all.
    """
    data_file = (
        SCULPTOR_DATA_DIR / "iron_templates" / _IRON_TEMPLATE_DATA_FILE[func_name]
    )
    if not data_file.exists():
        pytest.skip(f"{data_file.name} is a known pre-existing missing data file")

    captured = {}

    def _capture_initialise_fe(prefix, disp, flux, fe_dict=None, median_params=None):
        captured["disp"] = disp
        captured["flux"] = flux
        return None, None

    monkeypatch.setattr(original_qso_autofit, "initialise_Fe", _capture_initialise_fe)
    getattr(original_qso_autofit, func_name)("test_prefix")
    orig_disp, orig_flux = captured["disp"], captured["flux"]

    monkeypatch.setattr(new_qso_autofit, "initialise_Fe", _capture_initialise_fe)
    getattr(new_qso_autofit, func_name)("test_prefix")
    new_disp, new_flux = captured["disp"], captured["flux"]

    np.testing.assert_allclose(orig_disp, new_disp)
    np.testing.assert_allclose(orig_flux, new_flux)


def test_qso_autofit_setup_MgII_lines_default_matches_original(
    original_qso_autofit, monkeypatch
):
    """The ported function gained an additive `MgII_broad_shifted=0`
    parameter/loop. At the default, it must add the same components as the
    original.
    """

    def _stub_line_model(prefix, params):
        return f"comp::{prefix}", f"pars::{prefix}"

    monkeypatch.setattr(
        original_qso_autofit, "initialise_gaussian_line_model", _stub_line_model
    )
    orig_model = _RecordingModel()
    original_qso_autofit.setup_MgII_lines(
        spec=None, line_dict=_AnyDict(), narrow=True, MgII_broad=2, model=orig_model
    )

    monkeypatch.setattr(
        new_qso_autofit, "initialise_gaussian_line_model", _stub_line_model
    )
    new_model = _RecordingModel()
    new_qso_autofit.setup_MgII_lines(
        spec=None, line_dict=_AnyDict(), narrow=True, MgII_broad=2, model=new_model
    )

    assert orig_model.added_components == new_model.added_components


def test_qso_autofit_setup_hb_o3_lines_default_matches_original(
    original_qso_autofit, monkeypatch
):
    """The ported function gained an additive `OIII_broad=0` parameter/loop.
    At the default, it must add the same components as the original.
    """

    def _stub_line_model(prefix, params):
        return f"comp::{prefix}", f"pars::{prefix}"

    monkeypatch.setattr(
        original_qso_autofit, "initialise_gaussian_line_model", _stub_line_model
    )
    orig_model = _RecordingModel()
    original_qso_autofit.setup_hb_o3_lines(
        spec=None,
        line_dict=_AnyDict(),
        narrow=True,
        Hbeta_broad=1,
        Hbeta_broad_shifted=1,
        OIII_wind=1,
        model=orig_model,
    )

    monkeypatch.setattr(
        new_qso_autofit, "initialise_gaussian_line_model", _stub_line_model
    )
    new_model = _RecordingModel()
    new_qso_autofit.setup_hb_o3_lines(
        spec=None,
        line_dict=_AnyDict(),
        narrow=True,
        Hbeta_broad=1,
        Hbeta_broad_shifted=1,
        OIII_wind=1,
        model=new_model,
    )

    assert orig_model.added_components == new_model.added_components


def test_qso_autofit_get_civ_line_prop_was_dropped():
    """Deliberately not ported: dead code guarded by an unconditional
    `raise DeprecationWarning`."""
    assert not hasattr(new_qso_autofit, "get_civ_line_prop")


def test_qso_autofit_new_config_and_functions_present():
    for name in [
        "setup_CIV_lines",
        "setup_iron_template_V01_FUV",
        "setup_iron_template_V01_CIV",
        "setup_iron_template_Park22_OPT_Hbeta",
        "qso_contfe_MgII_Xshooter",
        "qso_cont_feII_opt",
    ]:
        assert hasattr(new_qso_autofit, name)


def test_qso_autofit_civ_mask_range_edits_applied():
    assert new_qso_autofit.qso_contfe_CIV_Shen11["mask_ranges"] == [
        [1340, 1375],
        [1445, 1465],
        [1700, 1705],
    ]
    assert new_qso_autofit.CIV_range["mask_ranges"] == [[1465, 1600]]


def test_original_balmer_with_hol_was_broken_and_is_replaced(original_qso_autofit):
    """The original `Qso_autofit.Balmer_continuum_with_hol` called
    `np.loadtxt(datadir, "wave_2.txt")` (filename passed as `dtype`), so it
    always raised `TypeError`. The local copy, fixed, was then replaced by
    `qso_advanced_models.BalmerContinuumWithHOL` (review §3): check that
    the original still fails and that `initialise_balmer` builds the new
    model and evaluates on the packaged data.
    """
    x = np.linspace(3000, 4000, 50)
    kwargs = {
        "z": 0.0,
        "amp_be": 1.0,
        "Te": 15000.0,
        "tau_be": 1.0,
        "lambda_be": 3646.0,
        "fwhm": 2000.0,
    }
    with pytest.raises(TypeError, match="not understood"):
        original_qso_autofit.Balmer_continuum_with_hol(x, **kwargs)

    assert not hasattr(new_qso_autofit, "Balmer_continuum_with_hol")

    new_kwargs = _redsh(kwargs)
    balmer_dict = {par: {"value": v, "vary": False} for par, v in new_kwargs.items()}
    comp, params = new_qso_autofit.initialise_balmer("BC", None, balmer_dict)
    assert isinstance(comp.function, new_qam.BalmerContinuumWithHOL)
    assert sorted(params) == sorted(f"BC_{p}" for p in new_kwargs)

    result = comp.function(x, **new_kwargs)
    expected = new_qam.BalmerContinuumWithHOL()(x, **new_kwargs)
    np.testing.assert_allclose(result, expected)
    assert np.all(np.isfinite(result)) and result.max() > 0


# --------------------------------------------------------------------------
# qso_advanced_models.py -- new physics ported from out.py; should match
# the fgu source's numeric output exactly.
# --------------------------------------------------------------------------


def test_qam_planck_lambda_cgs_matches_original(original_out):
    wavelength_aa = np.linspace(2000, 5000, 200)
    orig = original_out._planck_lambda_cgs(wavelength_aa, Te=15000.0)
    new = new_qam._planck_lambda_cgs(wavelength_aa, Te=15000.0)
    np.testing.assert_allclose(orig, new, rtol=1e-12)


def test_qam_power_law_plus_fractional_bc_matches_original(original_out):
    x = np.linspace(2000, 5000, 200)
    kwargs = {
        "amp": 1.0,
        "slope": -1.5,
        "z": 2.0,
        "f": 0.3,
        "Te": 15000.0,
        "tau_be": 1.0,
        "lambda_be": 3646.0,
    }
    orig = original_out.power_law_at_2500_plus_fractional_bc(x, **kwargs)
    new = new_qam.power_law_at_2500_plus_fractional_bc(x, **_redsh(kwargs))
    np.testing.assert_allclose(orig, new, rtol=1e-10)


def test_qam_voigt_optical_depth_matches_original(original_out):
    wave_obs_aa = np.linspace(1210, 1220, 200)
    kwargs = {
        "z_abs": 2.0,
        "rest_wave_aa": 1215.67,
        "log_N": 20.0,
        "b_kms": 30.0,
        "osc_strength": 0.4164,
        "damping_gamma": 6.265e8,
    }
    orig = original_out._voigt_optical_depth(wave_obs_aa, **kwargs)
    new = new_qam._voigt_optical_depth(wave_obs_aa, **kwargs)
    np.testing.assert_allclose(orig, new, rtol=1e-10)


def test_qam_lya_absorption_model_matches_original(original_out):
    x = np.linspace(1210, 1220, 200)
    orig = original_out.lya_absorption_model(x, z_abs=2.0, log_N=20.0, b_kms=30.0)
    new = new_qam.lya_absorption_model(x, z_abs=2.0, log_N=20.0, b_kms=30.0)
    np.testing.assert_allclose(orig, new, rtol=1e-10)
    # Transmission must stay within [0, 1].
    assert np.all((new >= 0.0) & (new <= 1.0))


_QAM_PATH = (
    REPO_ROOT
    / "sculptor_extensions"
    / "src"
    / "sculptor_extensions"
    / "qso_advanced_models.py"
)


def test_qam_balmer_continuum_with_hol_matches_original(original_out, hol_data_file):
    # The data file is packaged, so the module-level data is loaded.
    assert new_qam._HOL_TRANSITIONS is not None
    assert len(new_qam._HOL_TRANSITIONS) == 44  # Balmer n = 7..50
    fresh_new_qam = new_qam

    # z=0 so the rest-frame axis (x here) actually brackets lambda_be=3646
    # and the transitions (3650-3970 A) -- otherwise the Balmer-edge
    # normalization interpolates outside the data range.
    x = np.linspace(2800, 4200, 500)
    kwargs = {
        "z": 0.0,
        "amp_be": 1.0,
        "Te": 15000.0,
        "tau_be": 1.0,
        "lambda_be": 3646.0,
        "fwhm": 2000.0,
    }
    orig = original_out.BalmerContinuumWithHOL()(x, **kwargs)
    new = fresh_new_qam.BalmerContinuumWithHOL()(x, **_redsh(kwargs))
    np.testing.assert_allclose(orig, new, rtol=1e-8)

    # The cached variant must give the same numbers as the uncached one.
    cached = fresh_new_qam.CachedBalmerContinuumWithHOL()(x, **_redsh(kwargs))
    np.testing.assert_allclose(new, cached, rtol=1e-8)


def test_qam_hol_models_raise_clearly_without_data_file():
    """Without the high-order Balmer line data, the HOL models must fail
    predictably (FileNotFoundError) rather than silently producing wrong
    numbers. Simulated on a fresh copy of the module, so the packaged data
    file is never touched.
    """
    fresh = _load_module_from_path("_qam_no_hol_data", _QAM_PATH)
    fresh._HOL_TRANSITIONS = None
    fresh._HOL_EMISS = None

    x = np.linspace(2800, 4200, 50)
    kwargs = {
        "redsh": 2.0,
        "amp_be": 1.0,
        "Te": 15000.0,
        "tau_be": 1.0,
        "lambda_be": 3646.0,
        "fwhm": 2000.0,
    }
    for model in (fresh.BalmerContinuumWithHOL(), fresh.CachedBalmerContinuumWithHOL()):
        with pytest.raises(FileNotFoundError, match=fresh.HOL_DATA_FILE):
            model(x, **kwargs)


def test_qam_absorption_transmission_not_registered_as_absorption_component():
    """Documents the known core-package gap: these functions are portable
    math, but `FitComponent`/`FitModel` don't support multiplicative
    (`is_absorption`) components yet.
    """

    from sculptor.component import FitComponent

    assert "is_absorption" not in inspect.signature(FitComponent.__init__).parameters


# --------------------------------------------------------------------------
# `sculptor_extensions.qso` used to define ~30 `setup_*` functions that built
# `lmfit.Model`/`lmfit.Parameters` objects, referencing `Model`/`Parameters`
# names that were never actually importable (the `lmfit` import was
# commented out, and `lmfit` isn't a project dependency) -- calling any of
# them raised `NameError`. They were dead code, superseded by
# `Qso_autofit.py`'s `initialise_*`/`setup_iron_template_*` functions, and
# have been removed.
# --------------------------------------------------------------------------


def test_qso_module_has_no_leftover_lmfit_setup_functions():
    assert not hasattr(new_qso, "setup_power_law_at_2500")
    assert not hasattr(new_qso, "setup_line_model_gaussian")
    assert not hasattr(new_qso, "setup_iron_template_model")
    assert not hasattr(new_qso, "add_redshift_param")
    assert not hasattr(new_qso, "model_func_list")
    assert not hasattr(new_qso, "model_setup_list")


def test_qso_module_does_not_reference_lmfit():
    source = inspect.getsource(new_qso)
    assert "lmfit" not in source
    assert "Parameters()" not in source


def test_model_functions_name_the_redshift_redsh():
    """Convention (review §4): model functions call the redshift fit
    parameter `redsh`; only the absorber redshift is `z_abs`."""

    from sculptor import analysis as ana
    from sculptor.component import FitComponent

    functions = [
        ana.power_law_at2500,
        ana.line_model_gaussian,
        ana.template_model,
        new_qso.power_law_at_2500_plus_bc,
        new_qso.line_model_gaussian_oiii_doublet,
        new_qso.line_model_gaussian_nii_doublet,
        new_qso.line_model_gaussian_sii_doublet,
        new_qso.CIII_complex_model_func,
        new_qso_autofit.line_model_gaussian_NII_doublet,
        new_qso_autofit.line_model_gaussian_SII_doublet,
        new_qam.fast_balmer_continuum_model,
        new_qam.power_law_at_2500_plus_fractional_bc,
        new_qam.BalmerContinuumWithHOL(),
        new_qam.CachedBalmerContinuumWithHOL(),
    ]
    for func in functions:
        args = FitComponent("c", func).function_args
        assert "redsh" in args and "z" not in args, (func, args)


# --------------------------------------------------------------------------
# setup_Hgamma_lines / setup_OI_lines (ported from upstream qso_auto_fit)
# --------------------------------------------------------------------------


def _gaussian_line_params(cen, redsh=0.5):
    return {
        "redsh": {"value": redsh, "vary": False},
        "flux": {"value": 10.0, "vary": False},
        "cen": {"value": cen, "vary": False},
        "fwhm_km_s": {"value": 3000.0, "vary": False},
    }


def test_setup_hgamma_and_oi_lines_follow_the_line_setup_pattern():
    from sculptor.model import FitModel

    line_dict = {
        "Hgamma": {
            "narrow": {"params": _gaussian_line_params(4341.7)},
            "broad": {"params": _gaussian_line_params(4341.7)},
        },
        "OI": {"broad": {"params": _gaussian_line_params(8446.4)}},
    }
    model = FitModel()

    new_qso_autofit.setup_Hgamma_lines(
        None, line_dict, narrow=True, Hgamma_broad=2, model=model
    )
    new_qso_autofit.setup_OI_lines(None, line_dict, OI_broad=1, model=model)

    assert [c.name for c in model.components] == [
        "Hgamma_narrow",
        "Hgamma_broad_a",
        "Hgamma_broad_b",
        "OI_broad_a",
    ]
    assert "Hgamma_broad_b_fwhm_km_s" in model.parameters
    assert "OI_broad_a_redsh" in model.parameters

    x = np.linspace(6000, 13500, 4000)
    y = model.eval(x, [])
    # Each line integrates to its flux (4 components x 10).
    assert np.trapezoid(y, x) == pytest.approx(40.0, rel=1e-3)
