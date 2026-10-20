"""Correctness of `sculptor.utils.broaden_spectrum` (resolution broadening).

Replaces the interactive `examples/test_resolution.py` scratch script (now
in `deprecated/examples/`); its timing comparison lives in
`benchmarks/bench_broaden_spectrum.py`.
"""

import numpy as np
import pytest
from astropy import constants as const

from sculptor import analysis as ana
from sculptor import component as sccomp
from sculptor import model as scmod
from sculptor import utils as scut

C_KM_S = const.c.to("km/s").value
FWHM_TO_SIGMA = 1 / (2 * np.sqrt(2 * np.log(2)))


def _reference_broaden(dispersion, fluxden, resolution, fwhm_lim=5.0):
    """Plain-numpy version of the numba kernel: for each pixel, a Gaussian
    of FWHM = dispersion / R, truncated at +-fwhm_lim FWHM and normalised
    over the pixels it covers."""

    out = np.empty_like(fluxden, dtype=float)
    for i, (wave, r) in enumerate(zip(dispersion, resolution, strict=True)):
        fwhm = wave / r
        sigma = fwhm * FWHM_TO_SIGMA
        sel = (dispersion > wave - fwhm_lim * fwhm) & (
            dispersion < wave + fwhm_lim * fwhm
        )
        weights = np.exp(-0.5 * ((dispersion[sel] - wave) / sigma) ** 2)
        out[i] = np.sum(weights * fluxden[sel]) / np.sum(weights)
    return out


@pytest.fixture
def line():
    dispersion = np.linspace(6400, 6700, 3001)
    fluxden = ana.line_model_gaussian(dispersion, 0.0, 10.0, 6550.0, 100.0)
    return dispersion, fluxden


def test_matches_reference_implementation(line):
    dispersion, fluxden = line
    rng = np.random.default_rng(1)
    resolution = rng.uniform(800, 3000, dispersion.size)

    result = scut.broaden_spectrum(dispersion, fluxden, resolution)

    assert result == pytest.approx(_reference_broaden(dispersion, fluxden, resolution))


def test_broadened_width_adds_in_quadrature(line):
    dispersion, fluxden = line
    resolution = np.full_like(dispersion, 2000.0)

    result = scut.broaden_spectrum(dispersion, fluxden, resolution)

    # Gaussian line of 100 km/s convolved with an R = 2000 (150 km/s)
    # instrumental profile: FWHMs add in quadrature.
    expected_fwhm_km_s = np.hypot(100.0, C_KM_S / 2000)
    # Half-maximum crossings, linearly interpolated on each flank.
    ipeak = np.argmax(result)
    half = result[ipeak] / 2
    left = np.interp(half, result[: ipeak + 1], dispersion[: ipeak + 1])
    right = np.interp(half, result[ipeak:][::-1], dispersion[ipeak:][::-1])
    fwhm_km_s = (right - left) / dispersion[ipeak] * C_KM_S

    assert fwhm_km_s == pytest.approx(expected_fwhm_km_s, rel=0.01)
    # Flux is conserved for a line well inside the spectrum.
    assert np.trapezoid(result, dispersion) == pytest.approx(10.0, rel=1e-3)


def test_constant_spectrum_is_unchanged():
    dispersion = np.linspace(4000, 5000, 500)
    fluxden = np.full_like(dispersion, 3.0)

    result = scut.broaden_spectrum(dispersion, fluxden, np.full(500, 500.0))

    assert result == pytest.approx(fluxden)


def test_fitmodel_scalar_and_array_resolution_agree(line):
    dispersion, _ = line
    comp = sccomp.FitComponent("Ha", ana.line_model_gaussian)
    params = comp.create_params()
    for name, value in {
        "Ha_redsh": 0.0,
        "Ha_flux": 10.0,
        "Ha_cen": 6550.0,
        "Ha_fwhm_km_s": 100.0,
    }.items():
        params[name].value = value
        params[name].vary = False

    scalar = scmod.FitModel(components=[comp], parameters=params, resolution=2000)
    array = scmod.FitModel(
        components=[comp],
        parameters=params,
        resolution=np.full_like(dispersion, 2000.0),
    )

    assert scalar.eval(dispersion, []) == pytest.approx(array.eval(dispersion, []))
    assert scalar.eval(dispersion, []) == pytest.approx(
        scut.broaden_spectrum(
            dispersion,
            scalar.eval(dispersion, [], broaden_by_resolution=False),
            np.full_like(dispersion, 2000.0),
        )
    )


def test_big_endian_input_gives_the_same_result(line):
    """FITS data (e.g. read with astropy) is often big-endian, which numba
    cannot type; `broaden_spectrum` casts its inputs itself."""
    dispersion, fluxden = line
    resolution = np.full_like(dispersion, 2000.0)
    native = scut.broaden_spectrum(dispersion, fluxden, resolution)
    big_endian = scut.broaden_spectrum(dispersion.astype(">f8"), fluxden, resolution)
    assert np.all(np.isfinite(big_endian))
    np.testing.assert_array_equal(native, big_endian)
