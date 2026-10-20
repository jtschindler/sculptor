"""Tests that the `sculptor_extensions.qso` scaling relations and the
`Qso_autofit` line-property functions require astropy Quantities, convert
compatible units, and store results in fixed units.
"""

import numpy as np
import pandas as pd
import pytest
from astropy import units as u
from speconed import speconed as sod

from sculptor import measurements
from sculptor_extensions import Qso_autofit as qa
from sculptor_extensions import qso

KM_S = u.km / u.s
LWAV = u.erg / u.s / u.AA

# (function, keyword arguments in the canonical units)
RELATIONS = [
    (
        qso.se_bhmass_hbeta_vp06,
        {"hbeta_fwhm": 3000 * KM_S, "cont_lwav": 1e44 * LWAV, "cont_wav": 5100 * u.AA},
    ),
    (
        qso.se_bhmass_hbeta_park12,
        {"hbeta_fwhm": 3000 * KM_S, "cont_lwav": 1e44 * LWAV, "cont_wav": 5100 * u.AA},
    ),
    (
        qso.se_bhmass_civ_vp06_fwhm,
        {"civ_fwhm": 4000 * KM_S, "cont_lwav": 1e45 * LWAV, "cont_wav": 1350 * u.AA},
    ),
    (
        qso.se_bhmass_civ_vp06_sigma,
        {"civ_sigma": 2000 * KM_S, "cont_lwav": 1e45 * LWAV, "cont_wav": 1450 * u.AA},
    ),
    (
        qso.se_bhmass_civ_c17_fwhm,
        {"civ_fwhm": 4000 * KM_S, "cont_lwav": 1e45 * LWAV, "cont_wav": 1350 * u.AA},
    ),
    (
        qso.se_bhmass_mgii_vo09_fwhm,
        {"mgii_fwhm": 3000 * KM_S, "cont_lwav": 1e44 * LWAV, "cont_wav": 3000 * u.AA},
    ),
    (
        qso.se_bhmass_mgii_s11_fwhm,
        {"mgii_fwhm": 3000 * KM_S, "cont_lwav": 1e44 * LWAV, "cont_wav": 3000 * u.AA},
    ),
    (
        qso.calc_bolometric_luminosity,
        {"cont_lwav": 1e44 * LWAV, "cont_wav": 3000 * u.AA},
    ),
    (
        qso.se_bhmass_gh05,
        {"line_fwhm": 3000 * KM_S, "lwav": 1e42 * u.erg / u.s},
    ),
    (
        qso.se_bhmass_db20,
        {"line_fwhm": 3000 * KM_S, "lwav": 1e42 * u.erg / u.s},
    ),
]


def _in_other_units(kwargs):
    """The same quantities in different, compatible units."""
    other = {"km / s": u.m / u.s, "erg / (Angstrom s)": u.W / u.nm}
    other.update({"Angstrom": u.nm, "erg / s": u.W})
    return {k: v.to(other[v.unit.to_string()]) for k, v in kwargs.items()}


@pytest.mark.parametrize(
    "func,kwargs", RELATIONS, ids=lambda x: getattr(x, "__name__", "")
)
def test_relations_reject_bare_numbers(func, kwargs):
    for name in kwargs:
        bare = dict(kwargs, **{name: kwargs[name].value})
        with pytest.raises(TypeError, match=f"`{name}` must be an astropy Quantity"):
            func(**bare)


@pytest.mark.parametrize(
    "func,kwargs", RELATIONS, ids=lambda x: getattr(x, "__name__", "")
)
def test_relations_reject_incompatible_units(func, kwargs):
    for name in kwargs:
        wrong = dict(kwargs, **{name: kwargs[name].value * u.kg})
        with pytest.raises(u.UnitConversionError):
            func(**wrong)


@pytest.mark.parametrize(
    "func,kwargs", RELATIONS, ids=lambda x: getattr(x, "__name__", "")
)
def test_relations_convert_compatible_units(func, kwargs):
    expected, expected_ref = func(**kwargs)
    result, ref = func(**_in_other_units(kwargs))
    assert ref == expected_ref
    assert result.unit == expected.unit
    np.testing.assert_allclose(result.value, expected.value, rtol=1e-12)


def test_relation_results_are_in_fixed_units():
    mass, _ = qso.se_bhmass_mgii_s11_fwhm(
        3e6 * u.m / u.s, 1e37 * u.W / u.nm, cont_wav=300 * u.nm
    )
    assert mass.unit == u.Msun
    lbol, _ = qso.calc_bolometric_luminosity(1e37 * u.W / u.nm, 300 * u.nm)
    assert lbol.unit == u.erg / u.s
    ratio = qso.calc_eddington_ratio(1e39 * u.W, 1e9 * u.Msun)
    assert ratio.unit == u.dimensionless_unscaled
    assert qso.calc_velocity_shifts(2.01, 2.0).unit == KM_S


def test_vo09_reference_keeps_its_format():
    _, ref = qso.se_bhmass_mgii_vo09_fwhm(3000 * KM_S, 1e44 * LWAV, cont_wav=300 * u.nm)
    assert ref == "MgII_VO09_fwhm_3000.0"


def test_eddington_and_blueshift_require_quantities():
    with pytest.raises(TypeError):
        qso.calc_eddington_luminosity(1e9)
    with pytest.raises(TypeError):
        qso.calc_eddington_ratio(1e46, 1e9 * u.Msun)
    with pytest.raises(TypeError):
        qso.correct_CIV_fwhm_for_blueshift(5000 * KM_S, 500)


def _mgii_table(fwhm=3000.0, lum=1e44, z=2.99):
    cols = {}
    for key in ("_median", "_low", "_upp"):
        cols[f"MgII_fwhm{key}"] = [fwhm]
        cols[f"L3000{key}"] = [lum]
        cols[f"MgII_redsh{key}"] = [z]
    return pd.DataFrame(cols)


MGII_COLUMNS = {"fwhm": "MgII_fwhm", "luminosity": "L3000", "redshift": "MgII_redsh"}


def test_get_line_prop_mgii_values(tmp_path):
    df = qa.get_line_prop_MgII(
        _mgii_table(), 3.0, MGII_COLUMNS, str(tmp_path), "J0001", "Vest"
    )
    row = df.iloc[0]

    cont_lwav, cont_wav = 1e44 * LWAV, 3000 * u.AA
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


def _line_spec(fluxden_unit, scale):
    disp = np.linspace(5800, 6200, 4001)
    flux = scale * np.exp(-0.5 * ((disp - 6000) / 10) ** 2)
    return sod.SpecOneD(
        dispersion=disp,
        fluxden=flux,
        dispersion_unit=u.AA,
        fluxden_unit=fluxden_unit,
    )


def test_get_fwhm_int_flux_does_not_depend_on_the_flux_unit():
    """The integrated flux is returned in 1e-17 erg/s/cm^2 whatever the
    spectrum's flux density unit. It used to be `.value * 1e17`, which is
    only right for a spectrum in erg/s/cm^2/A."""
    cgs = u.erg / u.s / u.cm**2 / u.AA
    _, flux_cgs = qa.get_fwhm_int_flux(_line_spec(cgs, 1e-16), 5800, 6200, 0.0)
    spec_17 = _line_spec(1e-17 * cgs, 10.0)
    fwhm_17, flux_17 = qa.get_fwhm_int_flux(spec_17, 5800, 6200, 0.0)

    expected_flux = 10.0 * 10 * np.sqrt(2 * np.pi)  # in 1e-17 erg/s/cm^2
    assert flux_cgs == pytest.approx(expected_flux, rel=1e-6)
    assert flux_17 == pytest.approx(expected_flux, rel=1e-6)
    # FWHM as a float in km/s (only for the 1e-17 spectrum: the spline
    # smoothing s=100 is too strong for fluxes ~1e-16)
    expected_fwhm = measurements.get_fwhm(spec_17, [5800, 6200], s=100)
    assert fwhm_17 == pytest.approx(expected_fwhm.to_value(KM_S))
    assert isinstance(fwhm_17, float)


def test_get_fwhm_returns_nan_in_km_s():
    disp = np.linspace(5800, 6200, 4001)
    two_peaks = np.exp(-0.5 * ((disp - 5900) / 5) ** 2) + np.exp(
        -0.5 * ((disp - 6100) / 5) ** 2
    )
    spec = sod.SpecOneD(
        dispersion=disp,
        fluxden=two_peaks,
        dispersion_unit=u.AA,
        fluxden_unit=u.erg / u.s / u.cm**2 / u.AA,
    )
    fwhm = measurements.get_fwhm(spec, method="sign")
    assert fwhm.unit == KM_S
    assert np.isnan(fwhm)
