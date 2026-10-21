"""Tests that the `sculptor_extensions.qso` scaling relations and the
`qso_autofit` line-property functions require astropy Quantities, convert
compatible units, and store results in fixed units.
"""

import numpy as np
import pytest
from astropy import units as u
from speconed import speconed as sod

from sculptor import measurements
from sculptor_extensions import qso
from sculptor_extensions import qso_autofit as qa

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
    """FWHM (km/s) and integrated flux (1e-17 erg/s/cm^2) do not depend on the
    spectrum's flux density unit. The flux used to be `.value * 1e17` (only
    right for erg/s/cm^2/A), and the fixed spline smoothing s=100 broadened
    or smoothed away the line depending on the flux scale."""
    cgs = u.erg / u.s / u.cm**2 / u.AA
    fwhm_cgs, flux_cgs = qa.get_fwhm_int_flux(_line_spec(cgs, 1e-16), 5800, 6200, 0.0)
    fwhm_17, flux_17 = qa.get_fwhm_int_flux(
        _line_spec(1e-17 * cgs, 10.0), 5800, 6200, 0.0
    )

    expected_flux = 10.0 * 10 * np.sqrt(2 * np.pi)  # in 1e-17 erg/s/cm^2
    assert flux_cgs == pytest.approx(expected_flux, rel=1e-6)
    assert flux_17 == pytest.approx(expected_flux, rel=1e-6)
    # Gaussian sigma = 10 A at 6000 A
    expected_fwhm = 2 * np.sqrt(2 * np.log(2)) * 10 / 6000 * 299792.458
    assert fwhm_cgs == pytest.approx(expected_fwhm, rel=1e-3)
    assert fwhm_17 == pytest.approx(fwhm_cgs, rel=1e-9)
    assert isinstance(fwhm_17, float)


def test_get_fwhm_int_flux_subtracts_the_continuum():
    """Flux and FWHM are the line's, not line plus continuum: a sloped
    continuum under the line is subtracted first."""
    cgs = 1e-17 * u.erg / u.s / u.cm**2 / u.AA
    spec = _line_spec(cgs, 10.0)
    line_only = qa.get_fwhm_int_flux(spec, 5800, 6200, 0.0)
    spec.fluxden = spec.fluxden + 5.0 + 0.01 * (spec.dispersion - 6000)
    fwhm, flux = qa.get_fwhm_int_flux(spec, 5800, 6200, 0.0)
    assert flux == pytest.approx(line_only[1], rel=1e-3)
    assert fwhm == pytest.approx(line_only[0], rel=1e-2)
    _, raw_flux = qa.get_fwhm_int_flux(spec, 5800, 6200, 0.0, subtract_continuum=False)
    assert raw_flux > 2 * flux  # continuum x 400 A


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
