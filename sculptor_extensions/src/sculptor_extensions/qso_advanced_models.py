"""Experimental QSO continuum/line models.

This module holds new, not-yet-validated physics ported from
``tests/ground_truth/fgu/out.py``: a Balmer-continuum model that
includes the higher-order Balmer lines (HOL) blending into a pseudo-
continuum, and a Voigt-profile damping-wing absorption model (used for
e.g. intervening/associated Lya absorption).

These are kept separate from ``Qso_autofit.py`` because they are
experimental and, in the case of the Balmer+HOL model, depend on a data
file (``wave_2.txt``) that is not yet part of the packaged data.
"""

import importlib.resources

import numpy as np
from astropy.constants import c, h, k_B
from scipy.special import voigt_profile

from sculptor import component as sccomp

resource_path = importlib.resources.files("sculptor")
datadir = str(resource_path / "data") + "/"

c_kms = c.to("km/s").value
c_AngstroemPerSecond = c.to("AA/s").value

# Physical constants in cgs, computed once at import time (not per call)
_H_CGS = h.cgs.value  # erg s
_C_CGS = c.cgs.value  # cm/s
_KB_CGS = k_B.cgs.value  # erg/K

_SIGMA_TO_FWHM = 2.0 * np.sqrt(2.0 * np.log(2.0))

# High-order Balmer line transitions/emissivities (Storey & Hummer 1995,
# case B, model B, 7<=n<=50), used by BalmerContinuumWithHOL /
# CachedBalmerContinuumWithHOL. The data file is not yet included in
# `sculptor/data/` -- these stay None (and the HOL models raise a clear
# error if actually called) until `wave_2.txt` is added there.
try:
    _HOL_DATA = np.loadtxt(datadir + "wave_2.txt")
    _HOL_TRANSITIONS = _HOL_DATA[:, 0]
    _HOL_EMISS = _HOL_DATA[:, 1]
except OSError:
    _HOL_TRANSITIONS = None
    _HOL_EMISS = None

# -----------------------------------------------------------------------
# Atomic data for the absorbing transition(s). Lya absorption in an
# intervening/associated system is itself just HI Lya -- same rest
# wavelength as the emission line, but at the absorber's own redshift.
# -----------------------------------------------------------------------
LYA_REST_WAVE = 1215.67  # Angstrom
LYA_OSC_STRENGTH = 0.4164
LYA_DAMPING_GAMMA = 6.265e8  # s^-1

_E_CHARGE_ESU = 4.803e-10
_M_ELECTRON_G = 9.109e-28

C_KM_S = c.to("km/s").value


# ------------------------------------------------------------------------------
# Balmer continuum with higher order lines (up to n=50)
# ------------------------------------------------------------------------------


class _FakeCode:
    """Mimics the two attributes FitComponent needs from a real code object."""

    def __init__(self, co_argcount, co_varnames):
        self.co_argcount = co_argcount
        self.co_varnames = co_varnames


class BalmerContinuumWithHOL:
    """
    Callable Balmer pseudo-continuum + high-order-line (HOL) model.

    Drop-in replacement for `fast_balmer_continuum_model` anywhere a plain
    `f(x, z, amp_be, Te, tau_be, lambda_be, fwhm)` callable is expected
    (e.g. `sccomp.FitComponent`). Stateless -- every call recomputes
    everything from scratch.

    Exposes a fake `__code__`/`__name__` so `FitComponent.__init__`'s
    `self.function.__code__.co_argcount` / `co_varnames` introspection
    works the same way it did for a plain function.
    """

    def __init__(self):
        # Mirrors the signature of __call__ below: x, z, amp_be, Te,
        # tau_be, lambda_be, fwhm -> co_argcount=7. FitComponent slices
        # co_varnames[1:co_argcount] to get the param names, excluding
        # the first ("x"), exactly as it does for a plain function.
        self.__code__ = _FakeCode(
            co_argcount=7,
            co_varnames=("x", "z", "amp_be", "Te", "tau_be", "lambda_be", "fwhm"),
        )
        self.__name__ = "balmer_continuum_with_hol"

    def _continuum_shape(self, x_rest, Te, tau_be, lambda_be):
        lambda_cm = x_rest * 1e-8
        nu = _C_CGS / lambda_cm
        exponent = (_H_CGS * nu) / (_KB_CGS * Te)
        with np.errstate(over="ignore"):
            black_body_nu = (2.0 * _H_CGS * nu**3 / _C_CGS**2) / np.expm1(exponent)
        black_body_lambda = black_body_nu * c_AngstroemPerSecond / x_rest**2
        optical_depth_term = -tau_be * (x_rest / lambda_be) ** 3
        shape = black_body_lambda * (-np.expm1(optical_depth_term))
        return np.where(x_rest < lambda_be, shape, 0.0)

    def _hol_shape(self, x_rest, fwhm, lambda_be, continuum_shape):
        if _HOL_TRANSITIONS is None:
            raise FileNotFoundError(
                "wave_2.txt (high-order Balmer line transitions/emissivities) "
                "is missing from sculptor/data/ -- add it to use the "
                "Balmer+HOL models."
            )
        sigma = (fwhm / c_kms) * _HOL_TRANSITIONS / _SIGMA_TO_FWHM
        diff2 = (x_rest[:, None] - _HOL_TRANSITIONS[None, :]) ** 2
        profile = np.exp(-0.5 * diff2 / sigma[None, :] ** 2)
        flux_density = np.sum(_HOL_EMISS[None, :] * profile / sigma[None, :], axis=1)
        edge_hol = np.interp(lambda_be, x_rest, flux_density)
        edge_continuum = np.interp(lambda_be, x_rest, continuum_shape)
        flux_density *= 0.3 * edge_continuum / edge_hol
        return flux_density

    def __call__(self, x, z, amp_be, Te, tau_be, lambda_be, fwhm):
        x_rest = x / (1.0 + z)
        continuum_shape = self._continuum_shape(x_rest, Te, tau_be, lambda_be)
        hol_shape = self._hol_shape(x_rest, fwhm, lambda_be, continuum_shape)
        return amp_be * (continuum_shape + hol_shape) * 1e-8


class CachedBalmerContinuumWithHOL:
    """
    Callable Balmer pseudo-continuum + high-order-line (HOL) model.

    Drop-in replacement for `fast_balmer_continuum_model` anywhere a plain
    `f(x, z, amp_be, Te, tau_be, lambda_be, fwhm)` callable is expected
    (e.g. `sccomp.FitComponent`).

    Caches the (n_pixels, n_lines) wavelength-difference matrix used in the
    HOL Gaussian sum -- the single biggest cost in __call__ -- and
    automatically invalidates that cache whenever `x` or `z` change between
    calls, so it stays correct even if `z` is a free/varying fit parameter
    (it just won't provide a caching benefit in that case).

    IMPORTANT: give each `FitComponent` its own instance. Don't share one
    instance across components fit to different `x`/`z`, since the cache
    is per-instance state.

    Exposes a fake `__code__`/`__name__` so `FitComponent.__init__`'s
    `self.function.__code__.co_argcount` / `co_varnames` introspection
    works the same way it did for a plain function.
    """

    def __init__(self):
        self.__code__ = _FakeCode(
            co_argcount=7,
            co_varnames=("x", "z", "amp_be", "Te", "tau_be", "lambda_be", "fwhm"),
        )
        self.__name__ = "balmer_continuum_with_hol"

        self._cache_key = None
        self._x_rest = None
        self._diff2 = None

    def _update_cache(self, x, z):
        # id(x) is a cheap identity check; relies on x not being mutated
        # in place between calls with the same identity but different
        # contents. If your fitting framework ever does that, swap this
        # for a content-based check (e.g. hashing x.tobytes()).
        if _HOL_TRANSITIONS is None:
            raise FileNotFoundError(
                "wave_2.txt (high-order Balmer line transitions/emissivities) "
                "is missing from sculptor/data/ -- add it to use the "
                "Balmer+HOL models."
            )
        cache_key = (id(x), z)
        if cache_key != self._cache_key:
            self._x_rest = x / (1.0 + z)
            self._diff2 = (self._x_rest[:, None] - _HOL_TRANSITIONS[None, :]) ** 2
            self._cache_key = cache_key

    def _continuum_shape(self, Te, tau_be, lambda_be):
        lambda_cm = self._x_rest * 1e-8
        nu = _C_CGS / lambda_cm
        exponent = (_H_CGS * nu) / (_KB_CGS * Te)
        with np.errstate(over="ignore"):
            black_body_nu = (2.0 * _H_CGS * nu**3 / _C_CGS**2) / np.expm1(exponent)
        black_body_lambda = black_body_nu * c_AngstroemPerSecond / self._x_rest**2
        optical_depth_term = -tau_be * (self._x_rest / lambda_be) ** 3
        shape = black_body_lambda * (-np.expm1(optical_depth_term))
        return np.where(self._x_rest < lambda_be, shape, 0.0)

    def _hol_shape(self, fwhm, lambda_be, continuum_shape):
        sigma = (fwhm / c_kms) * _HOL_TRANSITIONS / _SIGMA_TO_FWHM
        profile = np.exp(-0.5 * self._diff2 / sigma[None, :] ** 2)
        flux_density = np.sum(_HOL_EMISS[None, :] * profile / sigma[None, :], axis=1)
        edge_hol = np.interp(lambda_be, self._x_rest, flux_density)
        edge_continuum = np.interp(lambda_be, self._x_rest, continuum_shape)
        flux_density *= 0.3 * edge_continuum / edge_hol
        return flux_density

    def __call__(self, x, z, amp_be, Te, tau_be, lambda_be, fwhm):
        self._update_cache(x, z)
        continuum_shape = self._continuum_shape(Te, tau_be, lambda_be)
        hol_shape = self._hol_shape(fwhm, lambda_be, continuum_shape)
        return amp_be * (continuum_shape + hol_shape) * 1e-8


def initialise_balmer_with_hol(prefix, balmer_dict=None):
    """
    Initialise a Balmer continuum + higher-order-lines component.
    Function requires a dictionary that contains necessary parameters to set up
    a Balmer template such as "amp_be", "redshift", "Te", "tau_be", "lambda_be".
    In this dictionary, each component should have "initial_value", "vary" and "prior"
    which provides information on initial value of the parameter, whether the parameter should be varied, and
    prior to be used to contrain the parameter, respectively.
    :param prefix: Model prefix
    :type prefix: string
    :param balmer_dict: dictionary containing parameters to setup the Balmer continuum model
    :type Balmer_dict: dictionary

    :return: Balmer continuuum model component and parameters
    """
    comp = sccomp.FitComponent(prefix, BalmerContinuumWithHOL())
    params = comp.create_params()

    if balmer_dict:
        for key in params:
            par = key.split(f"{prefix}_")[1]
            input = balmer_dict[par]

            if "parameter" in input:
                params.pop(key)
                comp.param_mapping.update({par: input["parameter"]})
            else:
                params[key].value = input["value"]
                params[key].vary = input["vary"]
                if input["vary"]:
                    params[key].prior = input["prior"]

    else:
        raise ValueError("Provide `balmer_dict`")

    return comp, params


def _planck_lambda_cgs(wavelength_aa, Te):
    """
    B_lambda(T) at the given wavelength(s), reproducing exactly what
    astropy.modeling.models.BlackBody + the B_nu -> B_lambda Jacobian
    conversion computes -- but directly in cgs, without building a
    BlackBody model object or doing unit-aware Quantity arithmetic on
    every call.

    astropy's BlackBody model internally evaluates in frequency space
    (B_nu) regardless of whether it's called with a wavelength or
    frequency input; calling code then converts B_nu -> B_lambda via the
    standard Jacobian, B_lambda = B_nu * c / lambda^2. This helper
    reproduces that same two-step calculation.

    :param wavelength_aa: Wavelength(s) in Angstrom
    :type wavelength_aa: float or np.ndarray
    :param Te: Electron/blackbody temperature in K
    :type Te: float
    :return: B_lambda(T) at the given wavelength(s)
    :rtype: float or np.ndarray
    """
    lambda_cm = wavelength_aa * 1e-8  # Angstrom -> cm
    nu = _C_CGS / lambda_cm
    exponent = (_H_CGS * nu) / (_KB_CGS * Te)
    with np.errstate(over="ignore"):
        black_body_nu = (2.0 * _H_CGS * nu**3 / _C_CGS**2) / np.expm1(exponent)

    return black_body_nu * c_AngstroemPerSecond / wavelength_aa**2


def fast_balmer_continuum_model(x, z, amp_be, Te, tau_be, lambda_be):
    """Model of the Balmer continuum (Dietrich 2003), without higher-order
    lines.

    This model is defined for a spectral dispersion axis in Angstroem.

    This functions implements the Balmer continuum model presented in
    Dietrich 2003. The model follows a blackbody below the Balmer edge (
    3646A) and is zero above.

    The amplitude parameter amp_be defines the amplitude at the balmer edge.
    The strength of the Balmer continuum can be estimated from the fluxden
    density at 3675A after subtraction of the power-law continuum component
    for reference see Grandi et al.(1982), Wills et al.(1985) or Verner et al.(
    1999).

    At wavelengths of 3646A higher order Balmer lines are merging. This has
    not been included in this model and thus it will produce a sharp break at
    the Balmer edge.

    Not wired in as the default Balmer-continuum model anywhere -- available
    for a maintainer to benchmark/validate against `qso.balmer_continuum_model`
    before switching a `setup_*`/`initialise_*` function to use it.

    :param x: Dispersion of the Balmer continuum
    :type x: np.ndarray
    :param z: Redshift
    :type z: float
    :param amp_be: Amplitude of the Balmer continuum at the Balmer edge
    :type amp_be: float
    :param Te: Electron temperature
    :type Te: float
    :param tau_be: Optical depth at the Balmer edge
    :type tau_be: float
    :param lambda_be: Wavelength of the Balmer edge
    :type lambda_be: float
    :return: Balmer continuum model
    :rtype: np.ndarray

    """

    x_rest = x / (1.0 + z)

    # Planck function B_nu(T), evaluated at nu = c / lambda, computed
    # directly in cgs rather than via astropy's BlackBody model (which
    # rebuilds a Model object and does unit-aware Quantity arithmetic
    # on every call -- expensive when called thousands of times in an
    # MCMC loop). This reproduces the same underlying formula.
    lambda_cm = x_rest * 1e-8  # Angstrom -> cm
    nu = _C_CGS / lambda_cm

    exponent = (_H_CGS * nu) / (_KB_CGS * Te)
    # np.expm1 is more numerically stable than exp(x)-1 for small x,
    # and errstate suppresses benign overflow warnings for the
    # (masked-out) short-wavelength tail before we zero it below.
    with np.errstate(over="ignore"):
        black_body_nu = (2.0 * _H_CGS * nu**3 / _C_CGS**2) / np.expm1(exponent)

    # Convert B_nu -> B_lambda via the standard Jacobian, using the
    # same c_AngstroemPerSecond / x**2 factor as the original.
    black_body_lambda = black_body_nu * c_AngstroemPerSecond / x_rest**2

    optical_depth_term = -tau_be * (x_rest / lambda_be) ** 3
    fluxden = amp_be * black_body_lambda * (-np.expm1(optical_depth_term))

    # Zero out everything at/above the Balmer edge in one vectorized
    # pass instead of a separate fancy-indexed assignment.
    fluxden = np.where(x_rest < lambda_be, fluxden, 0.0)

    fluxden *= 1e-20

    return fluxden


def power_law_at_2500_plus_fractional_bc(x, amp, slope, z, f, Te, tau_be, lambda_be):
    """QSO continuum model consisting of a power law anchored at 2500 A and
    a balmer continuum contribution.

    This model is defined for a spectral dispersion axis in Angstroem.

    The amplitude of the Balmer continuum is set to be a fraction of the
    power law component at the Balmer edge (3646A) using the variabe f.

    :param x: Dispersion of the continuum model
    :type x: np.ndarray
    :param amp: Amplitude of the power law (at 2500 A)
    :type amp: float
    :param slope: Slope of the power law
    :type slope: float
    :param z: Redshift
    :type z: float
    :param f: Amplitude of the Balmer continuum as a fraction of the power \
        law component
    :type f: float
    :param Te: Electron temperature
    :type Te: float
    :param tau_be: Optical depth at the Balmer edge
    :type tau_be: float
    :param lambda_be: Wavelength of the Balmer edge
    :type lambda_be: float
    :return: QSO continuum model
    :rtype: np.ndarray

    """

    pl_fluxden = amp * (x / (2500.0 * (z + 1.0))) ** slope

    pl_fluxden_at_be = amp * (lambda_be / 2500.0) ** slope

    x = x / (1.0 + z)

    black_body_lambda = _planck_lambda_cgs(x, Te)
    black_body_lambda_be = _planck_lambda_cgs(lambda_be, Te)

    bc_fluxden_at_lambda_be = black_body_lambda_be * (1.0 - np.exp(-tau_be))

    bc_fluxden = (
        f
        * pl_fluxden_at_be
        / bc_fluxden_at_lambda_be
        * black_body_lambda
        * (1.0 - np.exp(-tau_be * (x / lambda_be) ** 3))
    )

    bc_fluxden[x >= lambda_be] = bc_fluxden[x >= lambda_be] * 0

    return pl_fluxden + bc_fluxden


# ------------------------------------------------------------------------------
# Damping-wing absorption model (e.g. intervening/associated Lya)
# ------------------------------------------------------------------------------


def _voigt_optical_depth(
    wave_obs_aa, z_abs, rest_wave_aa, log_N, b_kms, osc_strength, damping_gamma
):
    wave_obs_cm = wave_obs_aa * 1e-8
    obs_wave_aa = rest_wave_aa * (1.0 + z_abs)
    obs_wave_cm = obs_wave_aa * 1e-8
    rest_wave_cm = rest_wave_aa * 1e-8  # <-- needed for the corrected tau_0

    b_cms = b_kms * 1e5
    N = 10.0**log_N

    sigma_cm = obs_wave_cm * b_cms / _C_CGS / np.sqrt(2.0)
    gamma_cm = damping_gamma * obs_wave_cm**2 / (4.0 * np.pi * _C_CGS)

    phi = voigt_profile(wave_obs_cm - obs_wave_cm, sigma_cm, gamma_cm)

    # rest_wave_cm * obs_wave_cm (equivalently rest_wave_cm**2 * (1+z_abs)),
    # not obs_wave_cm**2, to correctly account for the profile being
    # normalized over the OBSERVED-frame wavelength axis while the
    # underlying oscillator-strength cross-section is a rest-frame quantity.
    tau_0 = (
        (np.pi * _E_CHARGE_ESU**2)
        / (_M_ELECTRON_G * _C_CGS**2)
        * osc_strength
        * N
        * rest_wave_cm
        * obs_wave_cm
    )

    return tau_0 * phi


def _absorption_transmission(
    wave_obs, z_abs, log_N, b_kms, rest_wave, osc_strength, damping_gamma
):
    """Transmission (0-1) from a single absorbing transition."""
    tau = _voigt_optical_depth(
        wave_obs, z_abs, rest_wave, log_N, b_kms, osc_strength, damping_gamma
    )
    return np.exp(-tau)


def absorption_transmission_model(
    x, z_abs, log_N, b_kms, rest_wave, osc_strength, damping_gamma, **kwargs
):
    """
    Returns TRANSMISSION (0-1), not additive flux.

    NOTE: this is meant to be registered as a multiplicative component
    (e.g. via an `is_absorption=True` flag) so that `FitModel.eval`
    multiplies it into the summed emission instead of adding it. As of
    this port, `sculptor.component.FitComponent` and `sculptor.model.FitModel`
    do not support such a flag -- this function is portable math, but is
    not yet usable as a fit component in the current fitting framework.
    Wiring that up is a separate, larger follow-up in the core package.
    """
    return _absorption_transmission(
        x, z_abs, log_N, b_kms, rest_wave, osc_strength, damping_gamma
    )


def lya_absorption_model(x, z_abs, log_N, b_kms, **kwargs):
    """Convenience wrapper fixing the Lya transition's atomic data."""
    return absorption_transmission_model(
        x, z_abs, log_N, b_kms, LYA_REST_WAVE, LYA_OSC_STRENGTH, LYA_DAMPING_GAMMA
    )
