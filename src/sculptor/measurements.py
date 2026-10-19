"""Spectral measurements on `speconed.SpecOneD` spectra.

Low-level helpers used by `sculptor.analysis` to measure continuum and
emission-line properties of model spectra. Ported from the v1
`sculptor.dep_v1.specanalysis` module (now under `deprecated/`), keeping the
same function names and signatures.
"""

import numpy as np
from astropy import constants as const
from astropy import units
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import UnivariateSpline
from speconed import speconed as sod

c_km_s = const.c.to("km/s").value

# ------------------------------------------------------------------------------
# General model measurements
# ------------------------------------------------------------------------------


def get_integrated_flux(input_spec, disp_range=None):
    """Calculate the integrated flux of a spectrum.

    The keyword argument disp_range allows to specify the dispersion
    boundaries for the integration. The trapezoidal rule
    (`numpy.trapezoid`) is used for the integration.

    :param input_spec: SpecOneD object holding the spectrum to integrate
    :type input_spec: speconed.speconed.SpecOneD
    :param disp_range: 2 element list holding the lower and upper dispersion
        boundaries for the integration
    :type disp_range: [float, float]
    :return: Integrated flux
    :rtype: astropy.units.Quantity
    """

    if disp_range is not None:
        spec = input_spec.trim_dispersion(disp_range, inplace=False)
    else:
        spec = input_spec.copy()

    return (
        np.trapezoid(spec.fluxden, x=spec.dispersion)
        * spec.dispersion_unit
        * spec.fluxden_unit
    )


def get_average_fluxden(input_spec, dispersion, width=10, redshift=0):
    """Calculate the average flux density of a spectrum in a window centered
    at the specified dispersion and with a given width.

    The central dispersion and width can be specified in the rest-frame and
    then are redshifted to the observed frame using the redshift keyword
    argument.

    Warning: this function currently only works for spectra in wavelength
    units. For spectra in frequency units the conversion to rest-frame will
    be incorrect.

    :param input_spec: Input spectrum
    :type input_spec: speconed.speconed.SpecOneD
    :param dispersion: Central dispersion
    :type dispersion: float
    :param width: Width of the dispersion window
    :type width: float
    :param redshift: Redshift argument to redshift the dispersion window into
        the observed frame.
    :type redshift: float
    :return: Average flux density
    :rtype: astropy.units.Quantity
    """

    disp_range = [
        (dispersion - width / 2.0) * (1.0 + redshift),
        (dispersion + width / 2.0) * (1.0 + redshift),
    ]

    unit = input_spec.fluxden_unit

    return input_spec.average_fluxden(dispersion_range=disp_range) * unit


# ------------------------------------------------------------------------------
# Emission line model measurements
# ------------------------------------------------------------------------------


def get_peak_redshift(input_spec, rest_wave):
    """Calculate the redshift of the flux density peak in the spectrum by
    specifying the expected rest frame wavelength of the emission feature.

    :param input_spec: Input spectrum
    :type input_spec: speconed.speconed.SpecOneD
    :param rest_wave: Rest-frame wavelength of the expected emission feature.
    :type rest_wave: float
    :return: Redshift of the peak flux density
    :rtype: float
    """
    # TODO: Implement frequency unit support.

    return input_spec.peak_dispersion() / rest_wave - 1


def get_equivalent_width(cont_spec, line_spec, disp_range=None, redshift=0):
    """Calculate the rest-frame equivalent width of a spectral feature.

    Warning: this function currently only works for spectra in wavelength
    units. For spectra in frequency units the conversion to rest-frame will
    be incorrect.

    :param cont_spec: Continuum spectrum
    :type cont_spec: speconed.speconed.SpecOneD
    :param line_spec: Spectrum with the feature (e.g. emission line)
    :type line_spec: speconed.speconed.SpecOneD
    :param disp_range: Observed-frame dispersion range (2 element list of
        floats) over which the equivalent width will be calculated.
    :type disp_range: [float, float]
    :param redshift: Redshift of the astronomical source
    :type redshift: float
    :return: Rest-frame equivalent width
    :rtype: astropy.units.Quantity
    """

    # TODO: Implement frequency unit support.

    rest_dispersion = cont_spec.dispersion / (1 + redshift)
    rest_cont_flux = cont_spec.fluxden * (1 + redshift)
    rest_line_flux = line_spec.fluxden * (1 + redshift)

    if disp_range is not None:
        l_idx = np.argmin(np.abs(rest_dispersion - disp_range[0] / (1 + redshift)))
        u_idx = np.argmin(np.abs(rest_dispersion - disp_range[1] / (1 + redshift)))

        ew = np.trapezoid(
            rest_line_flux[l_idx:u_idx] / rest_cont_flux[l_idx:u_idx],
            rest_dispersion[l_idx:u_idx],
        )
    else:
        ew = np.trapezoid(rest_line_flux / rest_cont_flux, rest_dispersion)

    return ew * cont_spec.dispersion_unit


def get_fwhm(input_spec, disp_range=None, resolution=None, s=0, method="spline"):
    """Calculate the FWHM (in km/s) of an emission feature from the spectrum.

    The user can specify a dispersion range to limit the FWHM calculation to
    this part of the spectrum. If a resolution (R) is specified the FWHM is
    corrected for the broadening due to the resolution.

    The function will subtract a flux density value of half of the maximum
    and then find the two roots (flux density = 0) of the new flux density
    axis. If the emission feature has multiple components more than two
    roots can be found in which case a np.nan value will be returned.

    :param input_spec: Input spectrum
    :type input_spec: speconed.speconed.SpecOneD
    :param disp_range: Dispersion range to which the calculation is limited.
    :type disp_range: [float, float]
    :param resolution: Resolution in R = Lambda/Delta Lambda
    :type resolution: float
    :param s: Smoothing factor of the spline (only used by method 'spline').
    :type s: float
    :param method: Method to use in retrieving the FWHM. There are two
        methods available. The default method 'spline' uses a spline to
        interpolate the original spectrum and find the zero points using a
        root finding algorithm on the spline. The second method 'sign' finds
        sign changes in the half peak flux subtracted spectrum.
    :type method: string
    :return: FWHM of the spectral feature
    :rtype: astropy.units.Quantity
    """

    if disp_range is not None:
        spec = input_spec.trim_dispersion(disp_range, inplace=False)
    else:
        spec = input_spec.copy()

    fluxden = spec.fluxden
    dispersion = spec.dispersion

    if method == "spline":
        spline = UnivariateSpline(dispersion, fluxden - np.max(fluxden) / 2.0, s=s)
        roots = spline.roots()
    elif method == "sign":
        roots_idx = np.where(np.diff(np.sign(fluxden - np.max(fluxden) / 2.0)))[0]
        roots = dispersion[roots_idx]
    else:
        raise ValueError(
            '[ERROR] Value for "method" keyword argument not '
            'recognized. Possible values are "sign" or "spline".'
        )

    if len(roots) != 2:
        print(f"[ERROR] Found {len(roots)} roots. Cannot determine FWHM")
        print("[ERROR] Setting FWHM to NaN")
        return np.nan

    max_idx = np.where(fluxden == np.max(fluxden))
    max_wav = dispersion[max_idx]

    fwhm = (abs(roots[0] - max_wav) + abs(roots[1] - max_wav)) / max_wav * c_km_s

    if resolution is None:
        return fwhm[0] * units.km / units.s

    print(f"[INFO] FWHM is corrected for the provided resolution of R={resolution}")
    resolution_km_s = c_km_s / resolution
    return np.sqrt(fwhm**2 - resolution_km_s**2)[0] * units.km / units.s


def get_nonparametric_measurements(
    input_spec, line_rest_wavel, redshift, disp_range=None
):
    """Measure the velocities at different ratios of the total emission line
    flux.

    These velocity measurements are referenced in the literature by (e.g.)
    Whittle+1985, Liu+2013, Zakamska & Greene 2014.

    This function calculates the cumulative integral of the emission line
    flux and then determines the closest dispersion values in velocity space
    to the 5%, 10%, 50%, 90% and 95% total flux ratios.

    Note: `input_spec` is modified in place (converted to flux density per
    unit frequency and, if `disp_range` is given, trimmed).

    :param input_spec: Input spectrum
    :type input_spec: speconed.speconed.SpecOneD
    :param line_rest_wavel: rest-frame wavelength of the line in Angstroem
    :type line_rest_wavel: float
    :param redshift: Redshift of the source
    :type redshift: float
    :param disp_range: Observed-frame dispersion range to which the
        calculation is limited.
    :type disp_range: [float, float]
    :return: median velocity, 5% velocity, 10% velocity, 90% velocity,
        95% velocity, velocity resolution at median velocity, frequency of
        median velocity, wavelength of median velocity, redshift of median
        velocity
    :rtype: astropy.units.Quantity, astropy.units.Quantity,
        astropy.units.Quantity, astropy.units.Quantity,
        astropy.units.Quantity, astropy.units.Quantity,
        astropy.units.Quantity, astropy.units.Quantity, float
    """

    # Convert input spectrum to frequency
    input_spec.to_fluxden_per_unit_frequency_cgs()

    # Total flux
    flux_total = get_integrated_flux(input_spec, disp_range=disp_range)

    # Trim spectrum to disp_range if not None
    if disp_range is not None:
        input_spec.trim_dispersion(disp_range, inplace=True)

    # Calculate velocity dispersion axis
    line_wavel = (1 + redshift) * line_rest_wavel
    line_cen_freq = const.c / (line_wavel * units.AA)
    freq_to_vel = units.doppler_optical(line_cen_freq)
    velocity_disp = (input_spec.dispersion * input_spec.dispersion_unit).to(
        units.km / units.s, equivalencies=freq_to_vel
    )

    # Calculate cumulative flux
    flux_fraction = cumulative_trapezoid(
        input_spec.fluxden, input_spec.dispersion, initial=0
    )
    flux_fraction *= input_spec.dispersion_unit * input_spec.fluxden_unit
    flux_fraction /= flux_total

    # Determine velocities for flux fractions
    idx_90 = np.argmin(abs(flux_fraction - 0.1))
    idx_10 = np.argmin(abs(flux_fraction - 0.9))
    idx_95 = np.argmin(abs(flux_fraction - 0.05))
    idx_05 = np.argmin(abs(flux_fraction - 0.95))
    idx_50 = np.argmin(abs(flux_fraction - 0.5))

    v90 = velocity_disp[idx_90]
    v10 = velocity_disp[idx_10]
    v95 = velocity_disp[idx_95]
    v05 = velocity_disp[idx_05]
    v50 = velocity_disp[idx_50]
    v_res_at_line = np.abs(velocity_disp[idx_50 - 1] - velocity_disp[idx_50 + 1]) / 2.0

    # Calculate the centroid frequency
    freq_v50 = input_spec.dispersion[idx_50] * input_spec.dispersion_unit
    # Calculate the centroid wavelength
    wave_v50 = freq_v50.to(units.AA, equivalencies=units.spectral())
    # Calculate the centroid redshift
    z_v50 = wave_v50.value / line_rest_wavel - 1.0

    return v50, v05, v10, v90, v95, v_res_at_line, freq_v50, wave_v50, z_v50


# ------------------------------------------------------------------------------
# Astrophysical spectral measurements
# ------------------------------------------------------------------------------


def calc_lwav_from_fwav(fluxden, redshift, cosmology):
    """Calculate the monochromatic luminosity from the monochromatic flux
    density.

    :param fluxden: Monochromatic flux density at a given wavelength.
    :type fluxden: astropy.units.Quantity
    :param redshift: Redshift of the source.
    :type redshift: float
    :param cosmology: Cosmology as an astropy Cosmology object
    :type cosmology: astropy.cosmology.Cosmology
    :return: Monochromatic luminosity in units of erg s^-1 Angstroem^-1
    :rtype: astropy.units.Quantity
    """

    # Calculate luminosity distance in Mpc
    lum_distance = cosmology.luminosity_distance(redshift)

    return (fluxden * lum_distance**2 * 4 * np.pi * (1.0 + redshift)).decompose(
        bases=[units.erg, units.s, units.AA]
    )


def calc_integrated_luminosity(input_spec, redshift, cosmology, disp_range=None):
    """Calculate the integrated model spectrum luminosity.

    :param input_spec: Input spectrum
    :type input_spec: speconed.speconed.SpecOneD
    :param redshift: Redshift of the source.
    :type redshift: float
    :param cosmology: Cosmology as an astropy Cosmology object
    :type cosmology: astropy.cosmology.Cosmology
    :param disp_range: 2 element list holding the lower and upper dispersion
        boundaries for the integration
    :type disp_range: [float, float]
    :return: Return the integrated luminosity for (dispersion range in) the
        input spectrum.
    :rtype: astropy.units.Quantity
    """

    # Convert to rest-frame flux and dispersion
    rest_dispersion = input_spec.dispersion / (1.0 + redshift)
    rest_fluxden = calc_lwav_from_fwav(
        input_spec.fluxden * input_spec.fluxden_unit, redshift, cosmology
    )
    rest_frame_spec = sod.SpecOneD(
        dispersion=rest_dispersion,
        dispersion_unit=input_spec.dispersion_unit,
        fluxden=rest_fluxden.value,
        fluxden_unit=rest_fluxden.unit,
    )

    # Only integrate part of the model spectrum if disp_range is specified
    if disp_range is not None:
        rest_frame_spec.trim_dispersion(disp_range, inplace=True)

    # Integrate the model flux
    integrated_line_luminosity = np.trapezoid(
        rest_frame_spec.fluxden, x=rest_frame_spec.dispersion
    )

    return (
        integrated_line_luminosity
        * rest_frame_spec.fluxden_unit
        * rest_frame_spec.dispersion_unit
    )


def calc_apparent_mag_from_fluxden(fluxden, dispersion):
    """Calculate the apparent AB magnitude from the monochromatic flux
    density at a specified dispersion value.

    :param fluxden: Monochromatic flux density at a given wavelength.
    :type fluxden: astropy.units.Quantity
    :param dispersion: Dispersion value (usually in wavelength).
    :type dispersion: astropy.units.Quantity
    :return: Returns apparent AB magnitude.
    :rtype: astropy.units.Quantity
    """

    f_nu = fluxden * dispersion**2 / const.c

    value = (f_nu / units.ABflux).decompose()

    return -2.5 * np.log10(value) * units.mag
