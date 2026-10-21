"""Correctness of the resolution broadening: `sculptor.resolution` (used
by `FitModel.eval`) and the older `sculptor.utils.broaden_spectrum`.

Replaces the interactive `examples/test_resolution.py` scratch script (now
in `deprecated/examples/`); its timing comparison lives in
`benchmarks/bench_broaden_spectrum.py`.
"""

import numpy as np
import pytest
from astropy import constants as const
from astropy import units as u
from scipy.stats import norm
from speconed import speconed as sod

from sculptor import analysis as ana
from sculptor import component as sccomp
from sculptor import model as scmod
from sculptor import resolution as scres
from sculptor import utils as scut

C_KM_S = const.c.to("km/s").value
FWHM_TO_SIGMA = 1 / (2 * np.sqrt(2 * np.log(2)))


def _reference_broaden(dispersion, fluxden, resolution, fwhm_lim=5.0):
    """Plain-numpy version of the numba kernel: for each pixel, a Gaussian
    of FWHM = dispersion / R, truncated at +-fwhm_lim FWHM and normalized
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


def _line_model(resolution, **kwargs):
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
    return scmod.FitModel(
        components=[comp], parameters=params, resolution=resolution, **kwargs
    )


def _pixel_average_of_gaussian(edges, flux, cen, sigma):
    """Exact average over each pixel of a normalized Gaussian line."""
    cdf = norm.cdf(edges, cen, sigma)
    return flux * np.diff(cdf) / np.diff(edges)


def test_fitmodel_scalar_and_array_resolution_agree(line):
    dispersion, _ = line
    scalar = _line_model(2000)
    array = _line_model(np.full_like(dispersion, 2000.0))
    assert scalar.eval(dispersion, []) == pytest.approx(array.eval(dispersion, []))


@pytest.mark.parametrize("pixel_km_s", [20.0, 80.0, 180.0])
def test_fitmodel_resolution_matches_the_analytic_pixel_average(pixel_km_s):
    """A Gaussian line convolved with a Gaussian LSF is a Gaussian with the
    widths added in quadrature; each pixel holds its average. Also when the
    pixels are as wide as the line (undersampled), where evaluating at the
    pixel centers is far off."""
    dlog = pixel_km_s / C_KM_S
    dispersion = 6550.0 * np.exp(np.arange(-300, 301) * dlog)
    res = 1000.0
    model = _line_model(res)
    edges = scres.pixel_edges(dispersion)

    y = model.eval(dispersion, [])
    sigma_line = 6550.0 * 100.0 / C_KM_S / scres.FWHM_PER_SIGMA
    sigma_lsf = dispersion / res / scres.FWHM_PER_SIGMA
    # The LSF width changes by < 0.1% across the line: take it at the center
    sigma_tot = np.hypot(sigma_line, 6550.0 / res / scres.FWHM_PER_SIGMA)
    expected = _pixel_average_of_gaussian(edges, 10.0, 6550.0, sigma_tot)

    peak = expected.max()
    assert np.max(np.abs(y - expected)) < 2e-3 * peak
    assert np.sum(y * np.diff(edges)) == pytest.approx(10.0, rel=1e-4)
    assert np.all(sigma_lsf > 0)


def test_resolution_matrix_keeps_a_constant_and_is_cached():
    dispersion = np.geomspace(4000, 5000, 300)
    matrix, fine = scres.resolution_matrix(dispersion, 500.0)
    assert fine.size == 5 * dispersion.size
    assert matrix @ np.full(fine.size, 3.0) == pytest.approx(3.0)
    again, _ = scres.resolution_matrix(dispersion.copy(), 500.0)
    assert again is matrix
    with pytest.raises(ValueError, match="positive"):
        scres.resolution_matrix(dispersion, np.zeros_like(dispersion))
    with pytest.raises(ValueError, match="edges"):
        scres.resolution_matrix(dispersion, 500.0, edges=dispersion)


def test_fitmodel_pixel_edges_and_other_grids():
    """Given edges are used for the spectrum's pixels; another grid gets
    its own (e.g. a plot on a finer grid) and the resolution interpolated."""
    dispersion = np.linspace(6400, 6700, 301)
    spec = sod.SpecOneD(
        dispersion=dispersion,
        fluxden=np.zeros_like(dispersion),
        fluxden_err=np.ones_like(dispersion),
        dispersion_unit=u.AA,
        fluxden_unit=u.erg / u.s / u.cm**2 / u.AA,
    )
    res = np.linspace(900, 1100, dispersion.size)
    wide = scres.pixel_edges(dispersion)
    wide[1:-1] += 0.2  # shifted edges change the pixel averages
    model = _line_model(res, pixel_edges=wide)
    model.spec = spec
    shifted = model.eval(dispersion, [])
    model.pixel_edges = None
    plain = model.eval(dispersion, [])
    assert not np.allclose(shifted, plain)

    fine = np.linspace(6400, 6700, 1001)
    on_fine = model.eval(fine, [])
    assert np.interp(dispersion, fine, on_fine) == pytest.approx(
        plain, rel=0.05, abs=1e-3
    )


def test_resolution_on_grid_stitches_curves():
    blue = (np.array([1000.0, 2000.0]), np.array([100.0, 200.0]))
    red = (np.array([1800.0, 4000.0]), np.array([1000.0, 1000.0]))
    x = np.array([1500.0, 1900.0, 3000.0])
    res = scres.resolution_on_grid(x, [blue, red], boundaries=[1850.0], scale=2.0)
    assert res == pytest.approx([300.0, 2000.0, 2000.0])
    with pytest.raises(ValueError, match="covers"):
        scres.resolution_on_grid(np.array([500.0]), [blue, red], boundaries=[1850.0])


def test_big_endian_input_gives_the_same_result(line):
    """FITS data (e.g. read with astropy) is often big-endian, which numba
    cannot type; `broaden_spectrum` casts its inputs itself."""
    dispersion, fluxden = line
    resolution = np.full_like(dispersion, 2000.0)
    native = scut.broaden_spectrum(dispersion, fluxden, resolution)
    big_endian = scut.broaden_spectrum(dispersion.astype(">f8"), fluxden, resolution)
    assert np.all(np.isfinite(big_endian))
    np.testing.assert_array_equal(native, big_endian)
