"""QSO continuum/line models ported from ``tests/ground_truth/fgu/out.py``.

A Balmer-continuum model that includes the higher-order Balmer lines (HOL)
blending into a pseudo-continuum (used by
``qso_autofit.initialize_balmer``), faster versions of the plain Balmer
continuum and of the power law + fractional Balmer continuum, and a
Voigt-profile damping-wing absorption model (e.g. intervening/associated
Lya absorption). The HOL model reads
``sculptor/data/balmer_hol_storey_hummer1995.txt`` (see its header).
"""

import warnings

import numpy as np
from astropy.constants import c, h, k_B
from scipy.special import voigt_profile

from sculptor import analysis as ana
from sculptor import component as sccomp
from sculptor.utils import c_AngstroemPerSecond, c_km_s, datadir
from sculptor_extensions.component_setup import initialize_component

# Physical constants in cgs, computed once at import time (not per call)
_H_CGS = h.cgs.value  # erg s
_C_CGS = c.cgs.value  # cm/s
_KB_CGS = k_B.cgs.value  # erg/K

_SIGMA_TO_FWHM = 2.0 * np.sqrt(2.0 * np.log(2.0))

# High-order Balmer line transitions/emissivities (Storey & Hummer 1995,
# case B, 7<=n<=50), used by BalmerContinuumWithHOL /
# CachedBalmerContinuumWithHOL. Origin and columns are documented in the
# file header. If the file is missing these stay None and the HOL models
# raise a clear error when called.
HOL_DATA_FILE = "balmer_hol_storey_hummer1995.txt"
try:
    _HOL_DATA = np.loadtxt(datadir + HOL_DATA_FILE)
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


# ------------------------------------------------------------------------------
# Balmer continuum with higher order lines (up to n=50)
# ------------------------------------------------------------------------------


def _require_hol_data():
    """Raise a clear error if the HOL data file is missing."""
    if _HOL_TRANSITIONS is None:
        raise FileNotFoundError(
            f"{HOL_DATA_FILE} (high-order Balmer line transitions/"
            "emissivities) is missing from sculptor/data/ -- add it to "
            "use the Balmer+HOL models."
        )


def _hol_lines(diff2, fwhm):
    """Sum of the high-order Balmer lines (Gaussians at `_HOL_TRANSITIONS`
    weighted by `_HOL_EMISS`), unnormalized. `diff2` is ``(x_rest[:, None]
    - _HOL_TRANSITIONS[None, :]) ** 2``."""
    sigma = (fwhm / c_km_s) * _HOL_TRANSITIONS / _SIGMA_TO_FWHM
    profile = np.exp(-0.5 * diff2 / sigma[None, :] ** 2)
    return np.sum(_HOL_EMISS[None, :] * profile / sigma[None, :], axis=1)


def _hol_shape(x_rest, diff2, fwhm, lambda_be, continuum_shape):
    """High-order Balmer lines: `_hol_lines` scaled to 0.3 times the
    continuum at the Balmer edge, both interpolated on the pixel grid (as
    in the original model; the continuum is zero from the edge on, so the
    scale depends on where the pixels fall: see
    `BalmerContinuumHOLRatio`)."""
    flux_density = _hol_lines(diff2, fwhm)
    edge_hol = np.interp(lambda_be, x_rest, flux_density)
    edge_continuum = np.interp(lambda_be, x_rest, continuum_shape)
    flux_density *= 0.3 * edge_continuum / edge_hol
    return flux_density


class BalmerContinuumWithHOL:
    """
    Callable Balmer pseudo-continuum + high-order-line (HOL) model.

    Drop-in replacement for `fast_balmer_continuum_model` anywhere a plain
    `f(x, redsh, amp_be, Te, tau_be, lambda_be, fwhm)` callable is expected
    (e.g. `sccomp.FitComponent`). Stateless -- every call recomputes
    everything from scratch; `CachedBalmerContinuumWithHOL` caches the
    wavelength grid.

    Exposes a fake `__code__`/`__name__` so `FitComponent.__init__`'s
    `self.function.__code__.co_argcount` / `co_varnames` introspection
    works the same way it did for a plain function.
    """

    def __init__(self):
        # Mirrors the signature of __call__ below: x, redsh, amp_be, Te,
        # tau_be, lambda_be, fwhm -> co_argcount=7. FitComponent slices
        # co_varnames[1:co_argcount] to get the param names, excluding
        # the first ("x"), exactly as it does for a plain function.
        self.__code__ = sccomp.FakeCode(
            co_argcount=7,
            co_varnames=("x", "redsh", "amp_be", "Te", "tau_be", "lambda_be", "fwhm"),
        )
        self.__name__ = "balmer_continuum_with_hol"

    def _rest_frame(self, x, redsh):
        """Rest-frame wavelengths and their squared distances to the HOL
        transitions."""
        _require_hol_data()
        x_rest = x / (1.0 + redsh)
        return x_rest, (x_rest[:, None] - _HOL_TRANSITIONS[None, :]) ** 2

    def __call__(self, x, redsh, amp_be, Te, tau_be, lambda_be, fwhm):
        """Evaluate the model.

        :param x: Observed-frame dispersion in Angstroem
        :type x: np.ndarray
        :param redsh: Redshift
        :type redsh: float
        :param amp_be: Amplitude of the Balmer continuum
        :type amp_be: float
        :param Te: Electron temperature in K
        :type Te: float
        :param tau_be: Optical depth at the Balmer edge
        :type tau_be: float
        :param lambda_be: Wavelength of the Balmer edge in Angstroem
        :type lambda_be: float
        :param fwhm: FWHM of the high-order lines in km/s
        :type fwhm: float
        :return: Balmer continuum + high-order lines
        :rtype: np.ndarray
        """
        x_rest, diff2 = self._rest_frame(x, redsh)
        continuum_shape = balmer_continuum_shape(x_rest, Te, tau_be, lambda_be)
        hol_shape = _hol_shape(x_rest, diff2, fwhm, lambda_be, continuum_shape)
        return amp_be * (continuum_shape + hol_shape) * 1e-8


class CachedBalmerContinuumWithHOL(BalmerContinuumWithHOL):
    """
    `BalmerContinuumWithHOL` that caches the (n_pixels, n_lines)
    wavelength-difference matrix used in the HOL Gaussian sum -- the single
    biggest cost in __call__ -- and automatically invalidates that cache
    whenever `x` or `redsh` change between calls, so it stays correct even
    if `redsh` is a free/varying fit parameter (it just won't provide a
    caching benefit in that case).

    IMPORTANT: give each `FitComponent` its own instance. Don't share one
    instance across components fit to different `x`/`redsh`, since the cache
    is per-instance state.
    """

    def __init__(self):
        super().__init__()
        self._cache_key = None
        self._x_rest = None
        self._diff2 = None

    def _rest_frame(self, x, redsh):
        # id(x) is a cheap identity check; relies on x not being mutated
        # in place between calls with the same identity but different
        # contents. If your fitting framework ever does that, swap this
        # for a content-based check (e.g. hashing x.tobytes()).
        cache_key = (id(x), redsh)
        if cache_key != self._cache_key:
            self._x_rest, self._diff2 = super()._rest_frame(x, redsh)
            self._cache_key = cache_key
        return self._x_rest, self._diff2


class BalmerContinuumHOLRatio:
    """Balmer continuum + high-order lines with their relative strength as
    a parameter: `hol_ratio` is the lines' flux density at the Balmer edge
    over the continuum's just below it. Both are evaluated at the edge
    itself, not on the pixel grid, so the ratio means the same for any
    grid (`BalmerContinuumWithHOL` fixes 0.3 on grid-interpolated values).
    Caches the wavelength grid like `CachedBalmerContinuumWithHOL`: give
    each component its own instance."""

    def __init__(self):
        self._grid = CachedBalmerContinuumWithHOL()
        self.__code__ = sccomp.FakeCode(
            co_argcount=8,
            co_varnames=(
                "x",
                "redsh",
                "amp_be",
                "Te",
                "tau_be",
                "lambda_be",
                "fwhm",
                "hol_ratio",
            ),
        )
        self.__name__ = "balmer_continuum_hol_ratio"

    def __call__(self, x, redsh, amp_be, Te, tau_be, lambda_be, fwhm, hol_ratio):
        """Evaluate the model (arguments as `BalmerContinuumWithHOL`, plus
        `hol_ratio`, the lines over the continuum at the edge)."""
        x_rest, diff2 = self._grid._rest_frame(x, redsh)
        continuum = balmer_continuum_shape(x_rest, Te, tau_be, lambda_be)
        lines = _hol_lines(diff2, fwhm)
        edge_lines = _hol_lines((lambda_be - _HOL_TRANSITIONS[None, :]) ** 2, fwhm)[0]
        edge_continuum = balmer_continuum_at_edge(Te, tau_be, lambda_be)
        lines *= hol_ratio * edge_continuum / edge_lines
        return amp_be * (continuum + lines) * 1e-8


class BalmerHOLLines:
    """The high-order Balmer lines alone (n = 7 - 50, Storey & Hummer 1995
    case B emissivities, one FWHM), without the Balmer continuum: a template
    with a linear `flux`, the integrated flux of all the lines (as
    `analysis.line_model_gaussian`'s `flux`). Caches the wavelength grid
    like `CachedBalmerContinuumWithHOL`: give each component its own
    instance."""

    def __init__(self):
        self._grid = CachedBalmerContinuumWithHOL()
        self.__code__ = sccomp.FakeCode(
            co_argcount=4, co_varnames=("x", "redsh", "flux", "fwhm")
        )
        self.__name__ = "balmer_hol_lines"

    def __call__(self, x, redsh, flux, fwhm):
        """Evaluate the model.

        :param x: Observed-frame dispersion in Angstroem
        :type x: np.ndarray
        :param redsh: Redshift
        :type redsh: float
        :param flux: Integrated flux of all the lines
        :type flux: float
        :param fwhm: FWHM of the lines in km/s
        :type fwhm: float
        :return: The lines
        :rtype: np.ndarray
        """
        _, diff2 = self._grid._rest_frame(x, redsh)
        # Each line's Gaussian has area emissivity * sqrt(2 pi) in the rest
        # frame; 1 + z converts the rest-frame density to the observed one
        norm = np.sqrt(2.0 * np.pi) * np.sum(_HOL_EMISS) * (1.0 + redsh)
        return flux * _hol_lines(diff2, fwhm) / norm


def initialize_balmer_hol_lines(prefix, hol_dict=None):
    """Initialize a `BalmerHOLLines` component; `hol_dict` with "redsh",
    "flux" and "fwhm" (see `component_setup.initialize_component`).

    :return: Component and parameters
    """
    return initialize_component(
        prefix, BalmerHOLLines(), hol_dict, dict_name="hol_dict"
    )


def initialize_balmer_hol_ratio(prefix, balmer_dict=None):
    """Initialize a `BalmerContinuumHOLRatio` component; `balmer_dict` as
    for `initialize_balmer_with_hol`, plus "hol_ratio".

    :return: Component and parameters
    """
    return initialize_component(
        prefix, BalmerContinuumHOLRatio(), balmer_dict, dict_name="balmer_dict"
    )


def initialize_balmer_with_hol(prefix, balmer_dict=None):
    """
    Initialize a Balmer continuum + higher-order-lines component.
    Function requires a dictionary that contains necessary parameters to set up
    a Balmer template such as "amp_be", "redsh", "Te", "tau_be", "lambda_be"
    and "fwhm" (see `component_setup.initialize_component`).

    :param prefix: Model prefix
    :type prefix: string
    :param balmer_dict: dictionary containing parameters to setup the Balmer continuum model
    :type balmer_dict: dictionary

    :return: Balmer continuuum model component and parameters
    """
    return initialize_component(
        prefix, BalmerContinuumWithHOL(), balmer_dict, dict_name="balmer_dict"
    )


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


def balmer_continuum_shape(x_rest, Te, tau_be, lambda_be):
    """Shape of the Balmer continuum (Dietrich 2003): a blackbody times
    ``1 - exp(-tau_be (x / lambda_be)^3)`` below the Balmer edge, zero at
    and above it. The Balmer continuum models in this module and
    `qso.power_law_at_2500_plus_bc` scale this shape.

    :param x_rest: Rest-frame dispersion in Angstroem
    :type x_rest: np.ndarray
    :param Te: Electron temperature in K
    :type Te: float
    :param tau_be: Optical depth at the Balmer edge
    :type tau_be: float
    :param lambda_be: Wavelength of the Balmer edge in Angstroem
    :type lambda_be: float
    :return: B_lambda(Te) (cgs) times the optical-depth term
    :rtype: np.ndarray
    """
    optical_depth_term = -tau_be * (x_rest / lambda_be) ** 3
    shape = _planck_lambda_cgs(x_rest, Te) * (-np.expm1(optical_depth_term))
    return np.where(x_rest < lambda_be, shape, 0.0)


def balmer_continuum_at_edge(Te, tau_be, lambda_be):
    """`balmer_continuum_shape` just below the Balmer edge (its value for
    x -> lambda_be), used to normalize the continuum at the edge.

    :param Te: Electron temperature in K
    :type Te: float
    :param tau_be: Optical depth at the Balmer edge
    :type tau_be: float
    :param lambda_be: Wavelength of the Balmer edge in Angstroem
    :type lambda_be: float
    :return: B_lambda(Te) at lambda_be times ``1 - exp(-tau_be)``
    :rtype: float
    """
    return _planck_lambda_cgs(lambda_be, Te) * (-np.expm1(-tau_be))


def fast_balmer_continuum_model(x, redsh, amp_be, Te, tau_be, lambda_be):
    """Model of the Balmer continuum (Dietrich 2003), without higher-order
    lines.

    This model is defined for a spectral dispersion axis in Angstroem.

    This functions implements the Balmer continuum model presented in
    Dietrich 2003. The model follows a blackbody below the Balmer edge (
    3646A) and is zero above (`balmer_continuum_shape`).

    The amplitude parameter amp_be defines the amplitude at the balmer edge.
    The strength of the Balmer continuum can be estimated from the fluxden
    density at 3675A after subtraction of the power-law continuum component
    for reference see Grandi et al.(1982), Wills et al.(1985) or Verner et al.(
    1999).

    At wavelengths of 3646A higher order Balmer lines are merging. This has
    not been included in this model and thus it will produce a sharp break at
    the Balmer edge.

    Used by `qso_autofit.initialize_balmer_without_hol`. Replaces the former
    `qso.balmer_continuum_model` (same formula, ~7x faster; matches it to
    ~1e-15, see `tests/test_qso_extension_port.py`).

    Possible improvement (not done): cache the shape for fixed `redsh`,
    `Te`, `tau_be` and `lambda_be`, so that only `amp_be` is applied per
    call, as upstream's `Qso_autofit_2.balmer_continuum_model2` does (~9x
    faster when only `amp_be` varies). That version uses one module-wide
    cache keyed on `x.tobytes()`; a per-model cache would be safer.

    :param x: Dispersion of the Balmer continuum
    :type x: np.ndarray
    :param redsh: Redshift
    :type redsh: float
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
    x_rest = x / (1.0 + redsh)
    return amp_be * balmer_continuum_shape(x_rest, Te, tau_be, lambda_be) * 1e-20


def power_law_at_2500_plus_fractional_bc(
    x, amp, slope, redsh, f, Te, tau_be, lambda_be
):
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
    :param redsh: Redshift
    :type redsh: float
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
    pl_fluxden = ana.power_law_at2500(x, redsh, amp, slope)
    pl_fluxden_at_be = amp * (lambda_be / 2500.0) ** slope

    bc_fluxden = (
        f
        * pl_fluxden_at_be
        / balmer_continuum_at_edge(Te, tau_be, lambda_be)
        * balmer_continuum_shape(x / (1.0 + redsh), Te, tau_be, lambda_be)
    )

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


def __getattr__(name):
    """Forward `initialise_balmer_with_hol` (British spelling, renamed
    2026-10-20) with a `DeprecationWarning`.

    :param name: attribute name
    :type name: str
    :return: the function
    :rtype: callable
    :raises AttributeError: for any other name
    """
    if name == "initialise_balmer_with_hol":
        warnings.warn(
            "`initialise_balmer_with_hol` was renamed to `initialize_balmer_with_hol`",
            DeprecationWarning,
            stacklevel=2,
        )
        return initialize_balmer_with_hol
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
