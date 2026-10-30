"""Tests for `sculptor.analysis`.

- `analyze_mcmc_results` is the working routine (it used to be a stub that
  only printed; the real one was the private `_mcmc_analyze`, kept as an
  alias).
- The extinction correction lives in one helper.
- A missing redshift raises before it is used.
"""

import extinction
import numpy as np
import pytest
from astropy import units as u
from astropy.cosmology import FlatLambdaCDM
from astropy.table import QTable

from sculptor import analysis
from sculptor.component import FitComponent
from sculptor.model import FitModel

REDSHIFT = 2.0
REST_CEN = 1549.06
COSMO = FlatLambdaCDM(H0=70, Om0=0.3, Tcmb0=2.725)
UNITS = {"dispersion_unit": u.AA, "fluxden_unit": u.erg / u.s / u.cm**2 / u.AA}


@pytest.fixture
def setup():
    cont = FitComponent("cont", analysis.power_law_at2500)
    line = FitComponent("CIV", analysis.line_model_gaussian)
    params = {**cont.create_params(), **line.create_params()}
    values = {
        "cont_redsh": REDSHIFT,
        "cont_amp": 2.0,
        "cont_slope": 0.0,
        "CIV_redsh": REDSHIFT,
        "CIV_flux": 50.0,
        "CIV_cen": REST_CEN,
        "CIV_fwhm_km_s": 3000.0,
    }
    for name, param in params.items():
        param.value = values[name]
        param.vary = False

    model = FitModel(components=[cont, line], parameters=params)
    obs_cen = REST_CEN * (1 + REDSHIFT)
    dispersion = np.linspace(obs_cen - 300, obs_cen + 300, 3001)
    return model, cont, line, dispersion


def test_analyze_mcmc_results_returns_a_row_per_sample(setup):
    model, cont, line, dispersion = setup
    flat_chain = np.zeros((3, 0))  # all parameters fixed

    table = analysis.analyze_mcmc_results(
        model,
        flat_chain,
        {"components": [cont], "rest_frame_wavelengths": [1450.0]},
        [
            {
                "feature_name": "CIV",
                "components": [line],
                "rest_frame_wavelength": REST_CEN,
            }
        ],
        REDSHIFT,
        COSMO,
        dispersion=dispersion,
        **UNITS,
    )

    assert isinstance(table, QTable)
    assert len(table) == 3
    assert {"CIV_flux", "CIV_EW", "1450.0_fluxden_avg"} <= set(table.colnames)
    assert table["CIV_flux"].value == pytest.approx(50.0, rel=1e-6)


def test_old_private_name_is_a_deprecated_alias(setup):
    model, cont, _line, dispersion = setup
    with pytest.warns(DeprecationWarning):
        table = analysis._mcmc_analyze(
            model,
            np.zeros((1, 0)),
            {"components": [cont], "rest_frame_wavelengths": [1450.0]},
            [],
            REDSHIFT,
            COSMO,
            dispersion=dispersion,
            **UNITS,
        )
    assert len(table) == 1


def test_analysis_defaults_to_the_spectrum_axis(setup):
    from speconed import speconed as sod

    model, cont, line, dispersion = setup
    model.spec = sod.SpecOneD(
        dispersion=dispersion,
        fluxden=np.ones_like(dispersion),
        fluxden_err=np.ones_like(dispersion),
        **UNITS,
    )
    explicit = analysis.analyze_emission_feature(
        model,
        "CIV",
        [line],
        [],
        REST_CEN,
        redshift=REDSHIFT,
        cosmology=COSMO,
        dispersion=dispersion,
        emfeat_meas=["flux"],
        **UNITS,
    )
    default = analysis.analyze_emission_feature(
        model,
        "CIV",
        [line],
        [],
        REST_CEN,
        redshift=REDSHIFT,
        cosmology=COSMO,
        emfeat_meas=["flux"],
    )
    assert default["CIV_flux"] == explicit["CIV_flux"]

    cont_default = analysis.analyze_continuum(
        model, [cont], [], [1450.0], redshift=REDSHIFT
    )
    assert "1450.0_Lwav" in cont_default


def test_analysis_without_axis_or_spectrum_raises(setup):
    model, cont, _, _ = setup
    with pytest.raises(ValueError, match="No spectrum set"):
        analysis.analyze_continuum(model, [cont], [], [1450.0], COSMO, REDSHIFT)


def test_remove_extinction_matches_extinction_package():
    dispersion = np.linspace(3000, 9000, 100)
    fluxden = np.ones_like(dispersion)

    corrected = analysis._remove_extinction(fluxden, dispersion, 1.0, 0.5)

    ext = extinction.calzetti00(dispersion / 2.0, 0.5, 4.05)
    assert corrected == pytest.approx(extinction.remove(ext, fluxden))
    assert np.all(corrected > fluxden)


def test_remove_extinction_rejects_unknown_law():
    with pytest.raises(ValueError, match="extinction law"):
        analysis._remove_extinction(np.ones(3), np.ones(3) * 5000, 1.0, 0.5, "x")


def test_emission_feature_extinction_corrects_line_and_continuum(setup):
    model, cont, line, dispersion = setup
    kwargs = {
        "cont_components": [cont],
        "redshift": REDSHIFT,
        "dispersion": dispersion,
        "cosmology": COSMO,
        "emfeat_meas": ["flux", "EW"],
        **UNITS,
    }

    plain = analysis.analyze_emission_feature(
        model, "CIV", [line], [], REST_CEN, **kwargs
    )
    dereddened = analysis.analyze_emission_feature(
        model, "CIV", [line], [], REST_CEN, a_v=0.3, **kwargs
    )

    assert dereddened["CIV_flux"] > plain["CIV_flux"]
    # Line and continuum get the same correction, so the EW barely moves.
    assert dereddened["CIV_EW"].value == pytest.approx(plain["CIV_EW"].value, rel=0.01)


def test_emission_feature_without_redshift_raises_before_extinction(setup):
    model, _, line, dispersion = setup

    # With a_v set, the old code used the redshift for the extinction
    # correction first and failed with a TypeError.
    with pytest.raises(ValueError, match="No redshift"):
        analysis.analyze_emission_feature(
            model, "CIV", [line], [], REST_CEN, dispersion=dispersion, a_v=0.3
        )


def test_continuum_without_redshift_raises(setup):
    model, cont, _, dispersion = setup

    with pytest.raises(ValueError, match="No redshift"):
        analysis.analyze_continuum(
            model, [cont], [], [1450.0], COSMO, dispersion=dispersion
        )


class _FakeSpec:
    def __init__(self, dispersion, fluxden, fluxden_err):
        self.dispersion = np.asarray(dispersion, dtype=float)
        self.fluxden = np.asarray(fluxden, dtype=float)
        self.fluxden_err = np.asarray(fluxden_err, dtype=float)

    def copy(self):
        return _FakeSpec(
            self.dispersion.copy(), self.fluxden.copy(), self.fluxden_err.copy()
        )

    def trim_dispersion(self, disp_range):
        lo, hi = sorted(disp_range)
        mask = (self.dispersion >= lo) & (self.dispersion <= hi)
        return _FakeSpec(
            self.dispersion[mask], self.fluxden[mask], self.fluxden_err[mask]
        )


class _FakeModel:
    def __init__(self, spec):
        self.spec = spec

    def eval(self, dispersion, params, components=None, broaden_by_resolution=False):
        return np.zeros_like(dispersion, dtype=float)


def _spike_snr(l_min=None, l_max=None):
    """`get_line_snr` around a line at 5000 A for two samples, FWHM 300 and
    30000 km/s, with a spike at 5100 A (outside the narrow sample's window,
    inside the wide one's); returns the S/N and the line dict."""
    disp = np.linspace(4000, 6000, 20001)
    flux = np.zeros_like(disp)
    flux[np.argmin(np.abs(disp - 5100.0))] = 100.0
    model = _FakeModel(_FakeSpec(disp, flux, np.ones_like(disp)))
    fwhm = np.array([300.0, 30000.0])
    line_dict = {"line1": {"cen": 5000.0, "fwhm": fwhm, "l_min": l_min, "l_max": l_max}}
    line_snr, peak_snr = analysis.get_line_snr(
        model,
        np.zeros((2, 1)),
        cont_component=None,
        redshift=0.0,
        line_dict=line_dict,
        sigma=2.0,
    )
    return line_snr, peak_snr, line_dict


def test_get_line_snr_uses_per_sample_fwhm_window_and_does_not_mutate_input():
    """If the window were frozen at sample 0's (the bug), sample 1 would
    miss the spike too."""
    line_snr, peak_snr, line_dict = _spike_snr()
    assert line_snr[0, 0] == 0.0 and peak_snr[0, 0] == 0.0
    assert line_snr[0, 1] > 0.5 and peak_snr[0, 1] > 50.0
    # The input dict is not mutated with a cached window
    assert line_dict["line1"]["l_min"] is None
    assert line_dict["line1"]["l_max"] is None


def test_get_line_snr_respects_fixed_window_when_provided():
    """With l_min / l_max pinned, every sample uses that window."""
    line_snr, peak_snr, line_dict = _spike_snr(5090.0, 5110.0)
    assert np.all(line_snr[0] > 0.0) and np.all(peak_snr[0] > 50.0)
    assert (line_dict["line1"]["l_min"], line_dict["line1"]["l_max"]) == (
        5090.0,
        5110.0,
    )
