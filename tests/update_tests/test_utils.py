
import numpy as np
import pytest

from sculptor import utils as scut_new
from sculptor.updated_modules import utils as scut_orig


def make_test_spectrum(n=2000, seed=0):
    rng = np.random.default_rng(seed)
    dispersion = np.linspace(3.5, 5.2, n).astype(np.float64)
    fluxden = (
        1.0
        + 0.5 * np.exp(-0.5 * ((dispersion - 4.3) / 0.02) ** 2)
        + 0.05 * rng.normal(size=n)
    ).astype(np.float64)
    resolution = np.full(n, 1000.0, dtype=np.float64)
    return dispersion, fluxden, resolution


def test_broaden_spectrum_matches_original():
    dispersion, fluxden, resolution = make_test_spectrum()

    out_orig = scut_orig.broaden_spectrum(dispersion, fluxden, resolution, fwhm_lim=5)
    out_new = scut_new.broaden_spectrum(dispersion, fluxden, resolution, fwhm_lim=5.0)

    assert out_orig.shape == out_new.shape
    np.testing.assert_allclose(out_orig, out_new, rtol=1e-8, atol=1e-10)


def test_broaden_spectrum_scalar_resolution_path():
    """Original allows a scalar resolution; new requires the caller (model.eval)
    to broadcast it first. Confirm both give the same numbers when done that way."""
    dispersion, fluxden, _ = make_test_spectrum()

    out_orig = scut_orig.broaden_spectrum(dispersion, fluxden, 1000.0, fwhm_lim=5)

    resolution_arr = np.full_like(dispersion, 1000.0)
    out_new = scut_new.broaden_spectrum(
        dispersion, fluxden, resolution_arr, fwhm_lim=5.0
    )

    np.testing.assert_allclose(out_orig, out_new, rtol=1e-8, atol=1e-10)


def test_single_pixel_convolution_matches():
    dispersion, fluxden, resolution = make_test_spectrum()
    idx = 500

    val_orig = scut_orig._resolution_convolution(
        idx, dispersion, fluxden, resolution[idx], 5.0
    )
    val_new = scut_new._resolution_convolution(
        idx, dispersion, fluxden, resolution[idx], 5.0
    )
    assert np.isclose(val_orig, val_new, rtol=1e-8, atol=1e-10)


def test_output_is_finite_everywhere():
    dispersion, fluxden, resolution = make_test_spectrum(n=50)
    out_new = scut_new.broaden_spectrum(dispersion, fluxden, resolution, fwhm_lim=5.0)
    assert np.all(np.isfinite(out_new))


def test_bigendian_dispersion_still_breaks_the_new_code():
    """Documents that issue #2 above is real: without an explicit cast,
    a big-endian dispersion array still trips numba's typing error."""
    dispersion, fluxden, resolution = make_test_spectrum()
    dispersion_be = dispersion.astype(">f8")
    with pytest.raises(Exception):
        scut_new.broaden_spectrum(dispersion_be, fluxden, resolution, fwhm_lim=5.0)


def test_bigendian_dispersion_fixed_by_explicit_cast():
    dispersion, fluxden, resolution = make_test_spectrum()
    dispersion_be = dispersion.astype(">f8")
    dispersion_fixed = dispersion_be.astype(np.float64)  # native order
    out = scut_new.broaden_spectrum(dispersion_fixed, fluxden, resolution, fwhm_lim=5.0)
    assert np.all(np.isfinite(out))


