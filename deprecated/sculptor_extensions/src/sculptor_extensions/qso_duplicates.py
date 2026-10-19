"""Model functions removed from `sculptor_extensions/qso.py` on 2026-10-19
(review §3), because a maintained copy exists elsewhere:

| Removed from `qso.py`                  | Use instead                                              |
|----------------------------------------|----------------------------------------------------------|
| `power_law_at_2500(x, amp, slope, z)`  | `sculptor.analysis.power_law_at2500(x, redsh, amp, slope)` (identical output) |
| `line_model_gaussian(x, z, ...)`       | `sculptor.analysis.line_model_gaussian(x, redsh, ...)` (identical output) |
| `template_model(x, amp, z, ...)`       | `sculptor.analysis.template_model(x, redsh, amp, ...)` (identical output) |
| `balmer_continuum_model`               | `sculptor_extensions.qso_advanced_models.fast_balmer_continuum_model` (same formula, ~7x faster) |
| `power_law_at_2500_plus_fractional_bc` | `sculptor_extensions.qso_advanced_models.power_law_at_2500_plus_fractional_bc` (same formula, ~10x faster) |

The maintained versions name the redshift argument `redsh` (core) or `z`
(advanced models); `FitComponent` derives parameter names from the argument
names, so components built from the core functions get `<prefix>_redsh`.
The regression tests compare the maintained versions against the original
reference copies in `tests/ground_truth/`.

Kept for reference only; this file is not importable as-is (it relied on
`qso.py`'s module-level imports and constants).
"""


def power_law_at_2500(x, amp, slope, z):
    """Power law model anchored at 2500 AA

    This model is defined for a spectral dispersion axis in Angstroem.

    :param x: Dispersion of the power law
    :type x: np.ndarray
    :param amp: Amplitude of the power law (at 2500 A)
    :type amp: float
    :param slope: Slope of the power law
    :type slope: float
    :param z: Redshift
    :type z: float
    :return: Power law model
    :rtype: np.ndarray

    """

    return amp * (x / (2500.0 * (z + 1.0))) ** slope


def balmer_continuum_model(x, z, amp_be, Te, tau_be, lambda_be):
    """Model of the Balmer continuum (Dietrich 2003)

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

    x = x / (1.0 + z)

    bb = BlackBody(temperature=Te * u.K)
    black_body_lambda = bb(x * u.AA).value * c_AngstroemPerSecond / x**2

    fluxden = (
        amp_be * black_body_lambda * (1.0 - np.exp(-tau_be * (x / lambda_be) ** 3))
    )

    fluxden[x >= lambda_be] = fluxden[x >= lambda_be] * 0

    fluxden *= 1e-20

    return fluxden


def power_law_at_2500_plus_fractional_bc(x, amp, slope, z, f, Te, tau_be, lambda_be):
    """ QSO continuum model consisting of a power law anchored at 2500 A and
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

    pl_fluxden_at_be = amp * ((lambda_be * (z + 1.0)) / (2500.0 * (z + 1.0))) ** slope

    x = x / (1.0 + z)

    bb = BlackBody(temperature=Te * u.K)
    black_body_lambda = bb(x * u.AA).value * c_AngstroemPerSecond / x**2
    black_body_lambda_be = (
        bb(lambda_be * u.AA).value * c_AngstroemPerSecond / lambda_be**2
    )

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


def line_model_gaussian(x, z, flux, cen, fwhm_km_s):
    """Gaussian line model

    The central wavelength of the Gaussian line model is determined by the
    central wavelength cen and the redshift, z. These parameters are degenerate
    in a line fit and it is adviseable to fix one of them (to predetermined
    values e.g., the redshift or the central wavelength).

    The width of the line is set by the FWHM in km/s.

    The Gaussian is normalized.

    :param x: Dispersion of the continuum model
    :type x: np.ndarray
    :param z: Redshift
    :type z: float
    :param flux: Amplitude of the Gaussian
    :type flux: float
    :param cen: Central wavelength
    :type cen: float
    :param fwhm_km_s: FWHM of the Gaussian in km/s
    :type fwhm_km_s: float
    :return: Gaussian line model
    :rtype: np.ndarray

    """

    # Redshift central wavelength
    cen = cen * (1 + z)

    # Calculate sigma from fwhm
    fwhm = fwhm_km_s / c_km_s * cen
    sigma = fwhm / np.sqrt(8 * np.log(2))

    return (
        flux / (sigma * np.sqrt(2 * np.pi)) * np.exp(-((x - cen) ** 2) / (2 * sigma**2))
    )


def template_model(
    x,
    amp,
    z,
    fwhm,
    intr_fwhm,
    templ_disp=None,
    templ_fluxden=None,
    templ_disp_unit_str=None,
    templ_fluxden_unit_str=None,
):
    """Template model

    :param x: Dispersion of the template model
    :type x: np.ndarray
    :param amp: Amplitude of the template model
    :type amp: float
    :param z: Redshift
    :type z: float
    :param fwhm: FWHM the template should be broadened to
    :type fwhm: float
    :param intr_fwhm: Intrinsic FWHM of the template
    :type intr_fwhm: float
    :param templ_disp: Dispersion axis of the template. This must match the \
        same dispersion unit as the model
    :type templ_disp: np.ndarray
    :param templ_fluxden: Flux density of the template.
    :type templ_fluxden: templ_fluxden: np.ndarray
    :param templ_disp_unit_str: Dispersion unit of the template as a string
        in astropy cds format.
    :type templ_fluxden_unit_str: str
    :param templ_fluxden_unit: Flux density unit of the template as a string
        in astropy cds format.
    :type templ_disp_unit_str: str
    :return: Template model as a Scipy interpolation

    """

    if templ_disp_unit_str is not None and type(templ_disp_unit_str) is str:
        dispersion_unit = u.Unit(templ_disp_unit_str, format="cds")
    else:
        dispersion_unit = None
    if templ_fluxden_unit_str is not None and type(templ_fluxden_unit_str) is str:
        fluxden_unit = u.Unit(templ_fluxden_unit_str, format="cds")
    else:
        fluxden_unit = None

    # Initialize a SpecOneD model for the template
    template_spec = sod.SpecOneD(
        dispersion=templ_disp,
        fluxden=templ_fluxden,
        dispersion_unit=dispersion_unit,
        fluxden_unit=fluxden_unit,
    )

    # Calculate the FWHM for the convolution
    convol_fwhm = np.sqrt(fwhm**2 - intr_fwhm**2)
    # Convolve the template spectrum to the goal FWHM
    spec = template_spec.broaden_by_gaussian(convol_fwhm)

    # Shift the dispersion axis to the specified redshift. This does not
    # change the amplitude of the template spectrum as is normally done for
    # cosmological redshifting. As we want to determine the amplitude it is a
    # free parameter anyway.
    spec.dispersion = spec.dispersion * (1.0 + z)

    # Return interpolation function
    f = sp.interpolate.interp1d(
        spec.dispersion,
        spec.fluxden,
        kind="linear",
        bounds_error=False,
        fill_value=(0, 0),
    )

    return f(x) * amp
