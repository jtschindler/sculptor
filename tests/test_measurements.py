"""Tests for `sculptor.measurements`, ported from `dep_v1.specanalysis`.

The v1 helpers called `np.trapz`, which no longer exists in numpy >= 2.4,
so `get_integrated_flux`, `get_equivalent_width` and
`calc_integrated_luminosity` (and with them the default measurements of
`analysis.analyze_emission_feature`) raised `AttributeError`. These tests
check each helper against an analytic expectation on a Gaussian line over a
flat continuum, and that `analyze_emission_feature` runs with its defaults.
"""

import numpy as np
import pytest
from astropy import units as u
from astropy.cosmology import FlatLambdaCDM
from speconed import speconed as sod

from sculptor import analysis, measurements
from sculptor.component import FitComponent
from sculptor.model import FitModel

REDSHIFT = 2.0
REST_CEN = 1549.06  # CIV
LINE_FLUX = 50.0  # in fluxden unit * AA
FWHM_KM_S = 3000.0
CONT_LEVEL = 2.0
COSMO = FlatLambdaCDM(H0=70, Om0=0.3, Tcmb0=2.725)

DISPERSION_UNIT = u.AA
FLUXDEN_UNIT = 1e-17 * u.erg / u.s / u.cm**2 / u.AA


@pytest.fixture
def dispersion():
    obs_cen = REST_CEN * (1 + REDSHIFT)
    return np.linspace(obs_cen - 300, obs_cen + 300, 6001)


def _spec(dispersion, fluxden):
    return sod.SpecOneD(
        dispersion=dispersion,
        fluxden=fluxden,
        dispersion_unit=DISPERSION_UNIT,
        fluxden_unit=FLUXDEN_UNIT,
    )


@pytest.fixture
def line_spec(dispersion):
    fluxden = analysis.line_model_gaussian(
        dispersion, REDSHIFT, LINE_FLUX, REST_CEN, FWHM_KM_S
    )
    return _spec(dispersion, fluxden)


@pytest.fixture
def cont_spec(dispersion):
    return _spec(dispersion, np.full_like(dispersion, CONT_LEVEL))


def test_get_integrated_flux_recovers_line_flux(line_spec):
    flux = measurements.get_integrated_flux(line_spec)

    assert flux.unit == DISPERSION_UNIT * FLUXDEN_UNIT
    assert flux.value == pytest.approx(LINE_FLUX, rel=1e-6)


def test_get_integrated_flux_disp_range_limits_integration(line_spec):
    obs_cen = REST_CEN * (1 + REDSHIFT)
    half = measurements.get_integrated_flux(line_spec, disp_range=[obs_cen, 1e9])

    assert half.value == pytest.approx(LINE_FLUX / 2, rel=1e-3)


def test_get_average_fluxden_of_flat_continuum(cont_spec):
    avg = measurements.get_average_fluxden(
        cont_spec, REST_CEN, width=20, redshift=REDSHIFT
    )

    # A scaled unit (1e-17 erg/...) is folded into the returned Quantity.
    assert avg.to(FLUXDEN_UNIT).value == pytest.approx(CONT_LEVEL)


def test_get_peak_redshift(line_spec):
    assert measurements.get_peak_redshift(line_spec, REST_CEN) == pytest.approx(
        REDSHIFT, abs=1e-4
    )


def test_get_equivalent_width_is_rest_frame(cont_spec, line_spec):
    ew = measurements.get_equivalent_width(cont_spec, line_spec, redshift=REDSHIFT)

    assert ew.unit == DISPERSION_UNIT
    assert ew.value == pytest.approx(LINE_FLUX / CONT_LEVEL / (1 + REDSHIFT), rel=1e-6)


@pytest.mark.parametrize("method", ["spline", "sign"])
def test_get_fwhm_recovers_input_fwhm(line_spec, method):
    fwhm = measurements.get_fwhm(line_spec, method=method)

    assert fwhm.unit == u.km / u.s
    assert fwhm.value == pytest.approx(FWHM_KM_S, rel=0.01)


def test_get_fwhm_rejects_unknown_method(line_spec):
    with pytest.raises(ValueError, match="not recognized"):
        measurements.get_fwhm(line_spec, method="bogus")


def test_get_nonparametric_measurements_symmetric_line(line_spec):
    v50, v05, v10, v90, v95, _, _, _, z_v50 = (
        measurements.get_nonparametric_measurements(line_spec, REST_CEN, REDSHIFT)
    )

    # Symmetric line: median velocity at the line center, and the
    # percentile velocities mirror each other around it.
    assert abs(v50.to(u.km / u.s).value) < 10
    assert v05.value == pytest.approx(-v95.value, rel=0.01)
    assert v10.value == pytest.approx(-v90.value, rel=0.01)
    assert z_v50 == pytest.approx(REDSHIFT, abs=1e-4)


def test_calc_lwav_from_fwav():
    fluxden = 1.0 * FLUXDEN_UNIT
    lwav = measurements.calc_lwav_from_fwav(fluxden, REDSHIFT, COSMO)

    d_l = COSMO.luminosity_distance(REDSHIFT)
    expected = (fluxden * 4 * np.pi * d_l**2 * (1 + REDSHIFT)).to(u.erg / u.s / u.AA)
    assert lwav.to(u.erg / u.s / u.AA).value == pytest.approx(expected.value)


def test_calc_integrated_luminosity(line_spec):
    lum = measurements.calc_integrated_luminosity(line_spec, REDSHIFT, COSMO)

    # L = F * 4 pi d_L^2: the (1+z) of L_lambda cancels the 1/(1+z) of the
    # rest-frame dispersion.
    d_l = COSMO.luminosity_distance(REDSHIFT)
    expected = LINE_FLUX * DISPERSION_UNIT * FLUXDEN_UNIT * 4 * np.pi * d_l**2
    assert lum.to(u.erg / u.s).value == pytest.approx(
        expected.to(u.erg / u.s).value, rel=1e-6
    )


def test_calc_integrated_luminosity_disp_range_is_observed_frame(line_spec):
    """Regression: `disp_range` was applied to the rest-frame spectrum, so
    an observed-frame range (the convention of every other measurement and
    of `analysis.analyze_emission_feature`) selected nothing and the
    luminosity came out as 0."""

    obs_cen = REST_CEN * (1 + REDSHIFT)
    full = measurements.calc_integrated_luminosity(line_spec, REDSHIFT, COSMO)
    covering = measurements.calc_integrated_luminosity(
        line_spec, REDSHIFT, COSMO, disp_range=[obs_cen - 100, obs_cen + 100]
    )
    red_half = measurements.calc_integrated_luminosity(
        line_spec, REDSHIFT, COSMO, disp_range=[obs_cen, obs_cen + 100]
    )

    assert (covering / full).decompose().value == pytest.approx(1.0, rel=1e-3)
    assert (red_half / full).decompose().value == pytest.approx(0.5, rel=1e-2)

    # Same fraction as the flux over the same (observed-frame) range.
    flux_frac = measurements.get_integrated_flux(
        line_spec, disp_range=[obs_cen, obs_cen + 100]
    ) / measurements.get_integrated_flux(line_spec)
    assert (red_half / full).decompose().value == pytest.approx(
        flux_frac.decompose().value, rel=1e-3
    )


def test_calc_apparent_mag_from_fluxden_ab_zero_point():
    wave = 5000 * u.AA
    # Flux density per unit wavelength equivalent to 3631 Jy -> m_AB = 0.
    fluxden = (3631 * u.Jy).to(
        u.erg / u.s / u.cm**2 / u.AA, equivalencies=u.spectral_density(wave)
    )

    mag = measurements.calc_apparent_mag_from_fluxden(fluxden, wave)

    assert mag.unit == u.mag
    assert mag.value == pytest.approx(0.0, abs=1e-3)


def test_analyze_emission_feature_default_measurements(dispersion):
    """Regression: the default measurements (EW, flux, lum) used to crash."""

    cont = FitComponent("cont", analysis.power_law_at2500)
    line = FitComponent("CIV", analysis.line_model_gaussian)
    params = {**cont.create_params(), **line.create_params()}
    values = {
        "cont_redsh": REDSHIFT,
        "cont_amp": CONT_LEVEL,
        "cont_slope": 0.0,
        "CIV_redsh": REDSHIFT,
        "CIV_flux": LINE_FLUX,
        "CIV_cen": REST_CEN,
        "CIV_fwhm_km_s": FWHM_KM_S,
    }
    for name, param in params.items():
        param.value = values[name]
        param.vary = False

    model = FitModel(components=[cont, line], parameters=params)

    result = analysis.analyze_emission_feature(
        model,
        "CIV",
        [line],
        [],
        REST_CEN,
        cont_components=[cont],
        redshift=REDSHIFT,
        dispersion=dispersion,
        cosmology=COSMO,
        dispersion_unit=DISPERSION_UNIT,
        fluxden_unit=FLUXDEN_UNIT,
    )

    assert set(result) == {f"CIV_{m}" for m in analysis.emfeat_measures_default}
    assert result["CIV_flux"].value == pytest.approx(LINE_FLUX, rel=1e-6)
    assert result["CIV_EW"].value == pytest.approx(
        LINE_FLUX / CONT_LEVEL / (1 + REDSHIFT), rel=1e-6
    )
    assert result["CIV_FWHM"].value == pytest.approx(FWHM_KM_S, rel=0.01)
    assert result["CIV_peak_redsh"] == pytest.approx(REDSHIFT, abs=1e-4)


def test_get_nonparametric_measurements_copies_by_default(line_spec):
    before = line_spec.copy()

    measurements.get_nonparametric_measurements(line_spec, REST_CEN, REDSHIFT)

    assert line_spec.dispersion_unit == before.dispersion_unit
    assert np.array_equal(line_spec.dispersion, before.dispersion)
    assert np.array_equal(line_spec.fluxden, before.fluxden)


def test_get_nonparametric_measurements_inplace(line_spec):
    n_before = line_spec.dispersion.size
    obs_cen = REST_CEN * (1 + REDSHIFT)

    measurements.get_nonparametric_measurements(
        line_spec,
        REST_CEN,
        REDSHIFT,
        disp_range=[obs_cen - 100, obs_cen + 100],
        inplace=True,
    )

    # The v1 behavior: trimmed and converted to frequency in place.
    assert line_spec.dispersion_unit.physical_type == "frequency"
    assert line_spec.dispersion.size < n_before


def test_get_nonparametric_measurements_disp_range_in_angstrom(line_spec):
    """Regression: `disp_range` (observed-frame Angstrom, as for the other
    measurements) was applied after the conversion to frequency, so it
    trimmed away the whole spectrum and the function raised ValueError."""

    obs_cen = REST_CEN * (1 + REDSHIFT)
    full = measurements.get_nonparametric_measurements(line_spec, REST_CEN, REDSHIFT)
    covering = measurements.get_nonparametric_measurements(
        line_spec, REST_CEN, REDSHIFT, disp_range=[obs_cen - 100, obs_cen + 100]
    )

    # A range covering the whole line gives the same velocities.
    for a, b in zip(full[:5], covering[:5], strict=True):
        assert a.value == pytest.approx(b.value, abs=5)
    assert covering[-1] == pytest.approx(REDSHIFT, abs=1e-4)

    # Only the red half of the line: the median velocity is redshifted.
    red_half = measurements.get_nonparametric_measurements(
        line_spec, REST_CEN, REDSHIFT, disp_range=[obs_cen, obs_cen + 100]
    )
    assert red_half[0].value > 300
    assert red_half[-1] > REDSHIFT


def test_analyze_emission_feature_nonparam_with_disp_range(dispersion):
    cont = FitComponent("cont", analysis.power_law_at2500)
    line = FitComponent("CIV", analysis.line_model_gaussian)
    params = {**cont.create_params(), **line.create_params()}
    values = {
        "cont_redsh": REDSHIFT,
        "cont_amp": CONT_LEVEL,
        "cont_slope": 0.0,
        "CIV_redsh": REDSHIFT,
        "CIV_flux": LINE_FLUX,
        "CIV_cen": REST_CEN,
        "CIV_fwhm_km_s": FWHM_KM_S,
    }
    for name, param in params.items():
        param.value = values[name]
        param.vary = False
    model = FitModel(components=[cont, line], parameters=params)
    obs_cen = REST_CEN * (1 + REDSHIFT)

    result = analysis.analyze_emission_feature(
        model,
        "CIV",
        [line],
        [],
        REST_CEN,
        cont_components=[cont],
        redshift=REDSHIFT,
        dispersion=dispersion,
        emfeat_meas=["flux", "nonparam"],
        disp_range=[obs_cen - 100, obs_cen + 100],
        cosmology=COSMO,
        dispersion_unit=DISPERSION_UNIT,
        fluxden_unit=FLUXDEN_UNIT,
    )

    assert result["CIV_redsh_v50"] == pytest.approx(REDSHIFT, abs=1e-4)
    assert abs(result["CIV_v50"].value) < 10


def test_get_fwhm_is_unbiased_on_noisy_lines():
    """The half maximum comes from the smoothed spectrum: the raw maximum of
    noisy data sits on a noise spike and biased the FWHM low (-8% at S/N 20
    per pixel)."""
    disp = np.linspace(5400, 5800, 4001)
    truth = analysis.line_model_gaussian(disp, 1.0, 50.0, 2798.75, 3000.0)
    noise = truth.max() / 20
    fwhms = []
    for seed in range(10):
        rng = np.random.default_rng(seed)
        spec = _spec(disp, truth + rng.normal(0, noise, disp.size))
        s = disp.size * noise**2  # smoothing from the noise, as in get_fwhm_int_flux
        fwhms.append(measurements.get_fwhm(spec, s=s).to_value(u.km / u.s))
    assert np.median(fwhms) == pytest.approx(3000.0, rel=0.02)
