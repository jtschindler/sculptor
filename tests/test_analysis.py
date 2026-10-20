"""Tests for the `sculptor.analysis` clean-up (review §2.9, 2.10, §5.2).

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
