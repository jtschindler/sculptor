"""Behavior of the QSO model setups, relations and line-property functions
in `sculptor_extensions` (comparisons with the original code are in
`test_reference_values.py`)."""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from astropy import units as u

from sculptor import analysis as ana
from sculptor.component import FitComponent
from sculptor.model import FitModel
from sculptor_extensions import Qso_autofit as qa
from sculptor_extensions import qso, qso_presets
from sculptor_extensions import qso_advanced_models as qam

KM_S = u.km / u.s
QAM_PATH = Path(qam.__file__)


def _fixed(**values):
    return {par: {"value": v, "vary": False} for par, v in values.items()}


# --- component setups ------------------------------------------------------------


@pytest.mark.parametrize(
    ("prefix", "tied_func", "untied_func", "ratio"),
    [
        (
            "NII",
            qa.line_model_gaussian_NII_doublet,
            qso.line_model_gaussian_nii_doublet,
            2.96,
        ),
        (
            "SII",
            qa.line_model_gaussian_SII_doublet,
            qso.line_model_gaussian_sii_doublet,
            1.0,
        ),
    ],
)
def test_doublet_switch_tied_by_default(prefix, tied_func, untied_func, ratio):
    tied_dict = _fixed(redsh=0.1, flux=5.0, fwhm_km_s=300.0, fluxratio=ratio)
    comp, params = qa.initialize_gaussian_line_model(prefix, tied_dict)
    assert comp.function is tied_func
    assert sorted(params) == sorted(f"{prefix}_{p}" for p in tied_dict)

    untied_dict = _fixed(
        redsh=0.1, flux_a=5.0, flux_b=5.0 * ratio, fwhm_km_s_a=300.0, fwhm_km_s_b=300.0
    )
    comp_u, params_u = qa.initialize_gaussian_line_model(
        prefix, untied_dict, tied_doublet=False
    )
    assert comp_u.function is untied_func
    assert sorted(params_u) == sorted(f"{prefix}_{p}" for p in untied_dict)

    # With equal widths and flux_b = ratio * flux_a, both models agree.
    x = np.linspace(7000, 7600, 2000)
    tied = comp.function(x, 0.1, 5.0, 300.0, ratio)
    untied = comp_u.function(x, 0.1, 5.0, 5.0 * ratio, 300.0, 300.0)
    np.testing.assert_allclose(tied, untied, rtol=1e-12)


BALMER = {
    "redsh": 2.0,
    "amp_be": 1.0,
    "Te": 15000.0,
    "tau_be": 1.0,
    "lambda_be": 3646.0,
}


def test_initialize_balmer_without_hol_uses_fast_model():
    comp, params = qa.initialize_balmer_without_hol("BC", _fixed(**BALMER))
    assert comp.function is qam.fast_balmer_continuum_model
    assert sorted(params) == sorted(f"BC_{p}" for p in BALMER)


def test_initialize_balmer_builds_the_hol_model():
    values = dict(BALMER, redsh=0.0, fwhm=2000.0)
    comp, params = qa.initialize_balmer("BC", None, _fixed(**values))
    assert isinstance(comp.function, qam.BalmerContinuumWithHOL)
    assert sorted(params) == sorted(f"BC_{p}" for p in values)

    y = comp.function(np.linspace(3000, 4000, 50), **values)
    assert np.all(np.isfinite(y)) and y.max() > 0


def test_hol_models_raise_clearly_without_data_file():
    """Without the high-order Balmer line data, the HOL models fail with
    FileNotFoundError. Simulated on a fresh copy of the module."""
    spec = importlib.util.spec_from_file_location("_qam_no_hol_data", QAM_PATH)
    fresh = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fresh)
    fresh._HOL_TRANSITIONS = None
    fresh._HOL_EMISS = None

    x = np.linspace(2800, 4200, 50)
    for model in (fresh.BalmerContinuumWithHOL(), fresh.CachedBalmerContinuumWithHOL()):
        with pytest.raises(FileNotFoundError, match=fresh.HOL_DATA_FILE):
            model(x, **BALMER, fwhm=2000.0)


def test_model_functions_name_the_redshift_redsh():
    """Convention: model functions call the redshift fit parameter `redsh`;
    only the absorber redshift is `z_abs`."""
    functions = [
        ana.power_law_at2500,
        ana.line_model_gaussian,
        ana.template_model,
        qso.power_law_at_2500_plus_bc,
        qso.line_model_gaussian_oiii_doublet,
        qso.line_model_gaussian_nii_doublet,
        qso.line_model_gaussian_sii_doublet,
        qso.CIII_complex_model_func,
        qa.line_model_gaussian_NII_doublet,
        qa.line_model_gaussian_SII_doublet,
        qam.fast_balmer_continuum_model,
        qam.power_law_at_2500_plus_fractional_bc,
        qam.BalmerContinuumWithHOL(),
        qam.CachedBalmerContinuumWithHOL(),
    ]
    for func in functions:
        args = FitComponent("c", func).function_args
        assert "redsh" in args and "z" not in args, (func, args)


def test_setup_hgamma_and_oi_lines():
    def line(cen):
        return {"params": _fixed(redsh=0.5, flux=10.0, cen=cen, fwhm_km_s=3000.0)}

    line_dict = {
        "Hgamma": {"narrow": line(4341.7), "broad": line(4341.7)},
        "OI": {"broad": line(8446.4)},
    }
    model = FitModel()
    qso_presets.setup_Hgamma_lines(
        None, line_dict, narrow=True, Hgamma_broad=2, model=model
    )
    qso_presets.setup_OI_lines(None, line_dict, OI_broad=1, model=model)

    assert [c.name for c in model.components] == [
        "Hgamma_narrow",
        "Hgamma_broad_a",
        "Hgamma_broad_b",
        "OI_broad_a",
    ]
    x = np.linspace(6000, 13500, 4000)
    # Each line integrates to its flux (4 components x 10).
    assert np.trapezoid(model.eval(x, []), x) == pytest.approx(40.0, rel=1e-3)


def test_civ_mask_ranges():
    assert qa.qso_contfe_CIV_Shen11["mask_ranges"] == [
        [1340, 1375],
        [1445, 1465],
        [1700, 1705],
    ]
    assert qa.CIV_range["mask_ranges"] == [[1465, 1600]]


# --- relations ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "func,coefficient,reference",
    [
        (qso.se_bhmass_civ_vp06_fwhm, 6.66, "CIV_VP06_fwhm"),
        (qso.se_bhmass_civ_vp06_sigma, 6.73, "CIV_VP06_sigma"),
        (qso.se_bhmass_civ_c17_fwhm, 6.71, "CIV_C17_fwhm"),
    ],
)
def test_civ_bh_masses(func, coefficient, reference):
    """The original CIV relations used `u.kms` (not an astropy unit) and
    always failed; check the fixed ones against the formula."""
    mass, ref = func(5000 * KM_S, 1e44 * u.erg / u.s / u.AA, 1350 * u.AA)
    assert ref == reference
    expected = 10**coefficient * 5.0**2 * 1350**0.53
    assert mass.to_value(u.Msun) == pytest.approx(expected)


def test_dalla_bonta_2020_values_and_units():
    # At the pivot (L = 1e42 erg/s, FWHM = 10^3.5 km/s) only the zero point
    # remains: log M = 7.015.
    mass, _ = qso.se_bhmass_db20(10**3.5 * KM_S, 1e42 * u.erg / u.s)
    assert np.log10(mass.to_value(u.Msun)) == pytest.approx(7.015)

    other, _ = qso.se_bhmass_db20(10**3.5 * 1e3 * u.m / u.s, 1e35 * u.W)
    assert other.to_value(u.Msun) == pytest.approx(mass.to_value(u.Msun))

    with pytest.raises(ValueError, match="not supported"):
        qso.se_bhmass_db20(3000 * KM_S, 1e42 * u.erg / u.s, "Ha_LHa")

    pivot, _ = qso.calc_bolometric_luminosity_db20(10**41.746 * u.erg / u.s)
    assert np.log10(pivot.to_value(u.erg / u.s)) == pytest.approx(44.717)


# --- line properties ----------------------------------------------------------------


def _civ_table(fwhm=4000.0, lum=1e45, z_civ=2.98):
    cols = {}
    for key in ("_median", "_low", "_upp"):
        cols[f"CIV_fwhm{key}"] = [fwhm, np.nan]
        cols[f"L1350{key}"] = [lum, lum]
        cols[f"CIV_redsh{key}"] = [z_civ, z_civ]
    return pd.DataFrame(cols)


CIV_COLUMNS = {"fwhm": "CIV_fwhm", "luminosity": "L1350", "redshift": "CIV_redsh"}


def test_get_civ_line_prop_values():
    df = qso_presets.get_line_prop_CIV(_civ_table(), 3.0, CIV_COLUMNS)

    cont_lwav = 1e45 * u.erg / u.s / u.AA
    mass = qso.se_bhmass_civ_vp06_fwhm(4000 * KM_S, cont_lwav, 1350 * u.AA)[0]
    lbol = qso.calc_bolometric_luminosity(cont_lwav, 1350 * u.AA)[0]
    edd = qso.calc_eddington_luminosity(mass)
    row = df.iloc[0]
    assert row["CIV_BH_mass_1350_vp06_median"] == pytest.approx(mass.to_value(u.Msun))
    assert row["L_bol_1350_median"] == pytest.approx(lbol.to_value(u.erg / u.s))
    assert row["Edd_lum_CIV_1350_vp06_median"] == pytest.approx(
        edd.to_value(u.erg / u.s)
    )
    assert row["Eddington_ratio_1350_vp06_median"] == pytest.approx(
        (lbol / edd).decompose().value
    )
    assert row["delta_v_CIV_median"] == pytest.approx(
        qso.calc_velocity_shifts(2.98, z_sys=3.0).to_value(KM_S)
    )
    # The NaN FWHM row is skipped for the BH mass, but gets a velocity shift.
    assert np.isnan(df.iloc[1]["CIV_BH_mass_1350_vp06_median"])
    assert np.isfinite(df.iloc[1]["delta_v_CIV_median"])


def test_get_civ_line_prop_writes_csv_only_with_save_dir(tmp_path):
    qso_presets.get_line_prop_CIV(_civ_table(), 3.0, CIV_COLUMNS)
    assert list(tmp_path.iterdir()) == []

    qso_presets.get_line_prop_CIV(
        _civ_table(), 3.0, CIV_COLUMNS, save_dir=tmp_path, source="J0001"
    )
    assert (tmp_path / "J0001_results_CIV.csv").exists()


def test_feii_over_mgii_ratio(tmp_path):
    names = {"fe_amp": "FeII_amp", "peak_flux": "MgII_peak_flux"}
    df = pd.DataFrame(
        {
            "FeII_amp_median": [3.0e17],
            "MgII_peak_flux_median": [5.0],
            "FeII_amp_low": [2.9e17],
            "MgII_peak_flux_low": [4.9],
            "FeII_amp_upp": [3.1e17],
            "MgII_peak_flux_upp": [5.1],
        }
    )
    result = qso_presets.get_line_prop_MgII(
        df.copy(), 2.0, names, str(tmp_path), "src", "Vest"
    )
    for key in ["_median", "_low", "_upp"]:
        expected = df[f"FeII_amp{key}"] * 1e-17 / df[f"MgII_peak_flux{key}"]
        np.testing.assert_allclose(result[f"FeII_over_MgII{key}"], expected)

    # Without the columns in param_name_dict the ratio is skipped.
    result = qso_presets.get_line_prop_MgII(
        pd.DataFrame({"a": [1.0]}), 2.0, {}, str(tmp_path), "src", "Vest"
    )
    assert "FeII_over_MgII_median" not in result.columns


def test_get_line_prop_hbeta_preset():
    cols = {}
    for key in ("_median", "_low", "_upp"):
        cols[f"Hbeta_fwhm{key}"] = [3000.0]
        cols[f"L5100{key}"] = [1e44]
        cols[f"Hbeta_redsh{key}"] = [0.5]
    names = {"fwhm": "Hbeta_fwhm", "luminosity": "L5100", "redshift": "Hbeta_redsh"}
    df = qso_presets.get_line_prop_Hbeta_OIII(pd.DataFrame(cols), 0.5, names)

    mass = qso.se_bhmass_hbeta_vp06(3000 * KM_S, 1e44 * qso.LWAV_UNIT)[0]
    row = df.iloc[0]
    assert row["Hbeta_BH_mass_5100_median"] == pytest.approx(mass.to_value(u.Msun))
    assert "L_bol_5100_upp" in df.columns
    assert "Edd_lum_Hbeta_5100_low" in df.columns
    assert row["delta_v_Hbeta_median"] == pytest.approx(0.0, abs=1e-9)


def _mgii_table(fwhm=3000.0, lum=1e44, z=2.99):
    cols = {}
    for key in ("_median", "_low", "_upp"):
        cols[f"MgII_fwhm{key}"] = [fwhm]
        cols[f"L3000{key}"] = [lum]
        cols[f"MgII_redsh{key}"] = [z]
    return pd.DataFrame(cols)


def test_get_line_prop_mgii_values(tmp_path):
    df = qso_presets.get_line_prop_MgII(
        _mgii_table(), 3.0, MGII_COLUMNS, str(tmp_path), "J0001", "Vest"
    )
    row = df.iloc[0]

    cont_lwav, cont_wav = 1e44 * qso.LWAV_UNIT, 3000 * u.AA
    s11 = qso.se_bhmass_mgii_s11_fwhm(3000 * KM_S, cont_lwav, cont_wav)[0]
    vo09 = qso.se_bhmass_mgii_vo09_fwhm(3000 * KM_S, cont_lwav, cont_wav)[0]
    lbol = qso.calc_bolometric_luminosity(cont_lwav, cont_wav)[0]

    assert row["MgII_BH_mass_3000_s11_median"] == pytest.approx(s11.to_value(u.Msun))
    assert row["MgII_BH_mass_3000_vo09_median"] == pytest.approx(vo09.to_value(u.Msun))
    assert row["L_bol_3000_median"] == pytest.approx(lbol.to_value(u.erg / u.s))
    assert row["Eddington_ratio_3000_s11_median"] == pytest.approx(
        (lbol / qso.calc_eddington_luminosity(s11)).decompose().value
    )
    assert row["delta_v_MgII_median"] == pytest.approx(
        qso.calc_velocity_shifts(2.99, 3.0).to_value(KM_S)
    )


MGII_COLUMNS = {"fwhm": "MgII_fwhm", "luminosity": "L3000", "redshift": "MgII_redsh"}
