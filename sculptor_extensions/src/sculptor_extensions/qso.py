"""The Sculptor Quasar Extension

This module defines models and analysis routines specific for the
analysis of type-I QSOs (quasars, type-I AGN).

Complex models can be constructed by combining multiple of the basic model
functions defined here (e.g. the Hbeta and [OIII] lines consisting of six
Gaussian emission lines) using `sculptor.component.FitComponent` /
`sculptor.model.FitModel`; see `sculptor_extensions.Qso_autofit` for the
`initialize_*` functions that wire these model functions into fit
components.

"""

import logging

import numpy as np
from astropy import constants as const
from astropy import units as u

from sculptor import analysis as ana
from sculptor.utils import c_AngstroemPerSecond, c_km_s, datadir  # noqa: F401
from sculptor_extensions import qso_advanced_models as qam

logger = logging.getLogger(__name__)

KM_S = u.km / u.s
LWAV_UNIT = u.erg / u.s / u.AA
LUM_UNIT = u.erg / u.s


def _quantity(value, unit, name):
    """Return `value` converted to `unit`, requiring an astropy Quantity.

    The scaling relations in this module are only valid in specific units:
    a bare number, or a Quantity in other units read with `.value`, would
    silently give a wrong result. Every physical input must therefore carry
    its unit; it is converted here, so any compatible unit works (e.g. m/s
    for a FWHM, W/nm for a monochromatic luminosity).

    :param value: input quantity
    :type value: astropy.units.Quantity
    :param unit: unit the relation needs
    :type unit: astropy.units.Unit
    :param name: argument name, for the error message
    :type name: str
    :return: `value` in `unit`
    :rtype: astropy.units.Quantity
    :raises TypeError: if `value` is not an `astropy.units.Quantity`
    :raises astropy.units.UnitConversionError: if its unit cannot be
        converted to `unit`
    """
    if not isinstance(value, u.Quantity):
        raise TypeError(
            f"`{name}` must be an astropy Quantity in units of {unit}, "
            f"got {type(value).__name__} {value!r}"
        )
    return value.to(unit)


def virial_bh_mass(
    line_width,
    luminosity,
    zero_point,
    width_exp,
    lum_exp,
    width_pivot=1000.0 * KM_S,
    lum_pivot=1e44 * LUM_UNIT,
):
    """Single-epoch virial BH mass from a power-law scaling relation:

    M_BH / M_sun = zero_point * (line_width / width_pivot) ** width_exp
                   * (luminosity / lum_pivot) ** lum_exp

    This is the base of the `se_bhmass_*` relations below, which fix the
    coefficients from the literature and check which luminosity they are
    valid for.

    :param line_width: line FWHM or dispersion
    :type line_width: astropy.units.Quantity
    :param luminosity: luminosity, e.g. lambda L_lambda of the continuum or
        a line luminosity
    :type luminosity: astropy.units.Quantity
    :param zero_point: mass at the pivot width and luminosity, in M_sun
    :type zero_point: float
    :param width_exp: exponent of the line width
    :type width_exp: float
    :param lum_exp: exponent of the luminosity
    :type lum_exp: float
    :param width_pivot: pivot line width (default 1000 km/s)
    :type width_pivot: astropy.units.Quantity
    :param lum_pivot: pivot luminosity (default 1e44 erg/s)
    :type lum_pivot: astropy.units.Quantity
    :return: BH mass
    :rtype: astropy.units.Quantity
    """
    line_width = _quantity(line_width, KM_S, "line_width")
    luminosity = _quantity(luminosity, LUM_UNIT, "luminosity")
    return (
        zero_point
        * (line_width / width_pivot).to_value(u.dimensionless_unscaled) ** width_exp
        * (luminosity / lum_pivot).to_value(u.dimensionless_unscaled) ** lum_exp
        * u.Msun
    )


def _is_wav(cont_wav, *wavs):
    """Return whether `cont_wav` (a Quantity in Angstroem) equals one of
    `wavs` (Angstroem).

    :param cont_wav: continuum wavelength
    :type cont_wav: astropy.units.Quantity
    :param wavs: allowed wavelengths in Angstroem
    :type wavs: float
    :rtype: bool
    """
    return any(np.isclose(cont_wav.to_value(u.AA), wav) for wav in wavs)


# ------------------------------------------------------------------------------
# Model functions
# ------------------------------------------------------------------------------


def power_law_at_2500_plus_bc(x, amp, slope, redsh, amp_be, Te, tau_be, lambda_be):
    """QSO continuum model consisting of a power law anchored at 2500 A and
    a balmer continuum contribution.

    This model is defined for a spectral dispersion axis in Angstroem.

    The amplitude of the Balmer continuum is set independently of the
    power law component by the amplitude of the balmer continuum at the
    balmer edge amp_be.

    :param x: Dispersion of the continuum model
    :type x: np.ndarray
    :param amp: Amplitude of the power law (at 2500 A)
    :type amp: float
    :param slope: Slope of the power law
    :type slope: float
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
    :return: QSO continuum model
    :rtype: np.ndarray

    """

    pl_fluxden = ana.power_law_at2500(x, redsh, amp, slope)

    bc_fluxden = (
        amp_be
        / qam.balmer_continuum_at_edge(Te, tau_be, lambda_be)
        * qam.balmer_continuum_shape(x / (1.0 + redsh), Te, tau_be, lambda_be)
    )

    return pl_fluxden + bc_fluxden


def line_model_gaussian_oiii_doublet(x, redsh, flux, fwhm_km_s, fluxratio):
    """Doublet line model for the [OIII] lines at 4960.30 A and 5008.24 A.

    This model ties the redshift, the FWHM and the fluxes (ratio 1:3) of the
    forbidden transitions of [OIII] at 4960.30 A and 5008.24 A together.

    :param x: Dispersion of the continuum model
    :type x: np.ndarray
    :param redsh: Redshift
    :type redsh: float
    :param flux: Amplitude of the Gaussian
    :type flux: float
    :param cen: Central wavelength
    :type cen: float
    :param fwhm_km_s: FWHM of the Gaussian in km/s
    :type fwhm_km_s: float
    :return: Gaussian doublet line model
    :rtype: np.ndarray
    """

    return ana.line_model_gaussian(
        x, redsh, flux, 4960.30, fwhm_km_s
    ) + ana.line_model_gaussian(x, redsh, fluxratio * flux, 5008.24, fwhm_km_s)


def line_model_gaussian_nii_doublet(x, redsh, flux_a, flux_b, fwhm_km_s_a, fwhm_km_s_b):
    """Doublet line model for the [NII] lines at 6549.85 A and 6585.28 A.

    This model ties the redshift of the forbidden transitions of [NII] at
    6549.85 A and 6585.28 A together. FWHM and fluxes are free parameters for
    each Gaussian line model.

    :param x: Dispersion of the continuum model
    :type x: np.ndarray
    :param redsh: Redshift
    :type redsh: float
    :param flux_a: Amplitude of the 1st Gaussian component
    :type flux_a: float
    :param flux_b: Amplitude of the 2nd Gaussian component
    :type flux_b: float
    :param fwhm_km_s_a: FWHM of the 1st Gaussian component in km/s
    :type fwhm_km_s_a: float
    :param fwhm_km_s_b: FWHM of the 2nd Gaussian component in km/s
    :type fwhm_km_s_b: float
    :return: Gaussian doublet line model
    :rtype: np.ndarray
    """

    return ana.line_model_gaussian(
        x, redsh, flux_a, 6549.85, fwhm_km_s_a
    ) + ana.line_model_gaussian(x, redsh, flux_b, 6585.28, fwhm_km_s_b)


def line_model_gaussian_sii_doublet(x, redsh, flux_a, flux_b, fwhm_km_s_a, fwhm_km_s_b):
    """Doublet line model for the [SII] lines at 6718.29 A and 6732.67 A.

    This model ties the redshift,  of the forbidden transitions of [SII] at
    6718.29 A and 6732.67 A together. FWHM and fluxes are free parameters for
    each Gaussian line model.

    :param x: Dispersion of the continuum model
    :type x: np.ndarray
    :param redsh: Redshift
    :type redsh: float
    :param flux_a: Amplitude of the 1st Gaussian component
    :type flux_a: float
    :param flux_b: Amplitude of the 2nd Gaussian component
    :type flux_b: float
    :param fwhm_km_s_a: FWHM of the 1st Gaussian component in km/s
    :type fwhm_km_s_a: float
    :param fwhm_km_s_b: FWHM of the 2nd Gaussian component in km/s
    :type fwhm_km_s_b: float
    :return: Gaussian doublet line model
    :rtype: np.ndarray

    """

    return ana.line_model_gaussian(
        x, redsh, flux_a, 6718.29, fwhm_km_s_a
    ) + ana.line_model_gaussian(x, redsh, flux_b, 6732.67, fwhm_km_s_b)


def CIII_complex_model_func(
    x,
    redsh,
    cen,
    cen_alIII,
    cen_siIII,
    flux,
    fwhm_km_s,
    shift_km_s,
    flux_alIII,
    fwhm_km_s_alIII,
    shift_km_s_alIII,
    flux_siIII,
    fwhm_km_s_siIII,
    shift_km_s_siIII,
):
    """Model function for the CIII] emission line complex, consisting of the
    Gaussian line models with a combined redshift parameter.

    The width of the line is set by the FWHM in km/s.

    The Gaussians are normalized (`analysis.line_model_gaussian`); each
    line's center is shifted by its velocity shift.

    :param x: Dispersion of the template model
    :type x: np.ndarray
    :param redsh: Redshift for CIII], AlIII, SiIII]
    :type redsh: float
    :param cen: CIII] central wavelength
    :type cen: float
    :param cen_alIII: AlIII central wavelength
    :type cen_alIII: float
    :param cen_siIII:  SiIII] central wavelength
    :type cen_siIII: float
    :param flux: Integrated flux of the CIII] line
    :type flux: float
    :param fwhm_km_s:  Full Width at Half Maximum (FWHM) of CIII] in km/s
    :type fwhm_km_s: float
    :param shift_km_s: Doppler velocity shift of the central wavelength
    :type shift_km_s: float
    :param flux_alIII: Integrated flux of the AlIII line
    :type flux_alIII: float
    :param fwhm_km_s_alIII:  Full Width at Half Maximum (FWHM) of AlIII in km/s
    :type fwhm_km_s_alIII: float
    :param shift_km_s_alIII: Doppler velocity shift of the central wavelength
    :type shift_km_s_alIII: float
    :param flux_siIII: Integrated flux of the SiIII] line
    :type flux_siIII: float
    :param fwhm_km_s_siIII: Full Width at Half Maximum (FWHM) of SiIII] in km/s
    :type fwhm_km_s_siIII: float
    :param shift_km_s_siIII: Doppler velocity shift of the central wavelength
    :type shift_km_s_siIII: float
    :return: CIII] complex model

    """

    def _line(flux, cen, fwhm_km_s, shift_km_s):
        shifted_cen = cen * (1 + shift_km_s / c_km_s)
        return ana.line_model_gaussian(x, redsh, flux, shifted_cen, fwhm_km_s)

    return (
        _line(flux, cen, fwhm_km_s, shift_km_s)
        + _line(flux_alIII, cen_alIII, fwhm_km_s_alIII, shift_km_s_alIII)
        + _line(flux_siIII, cen_siIII, fwhm_km_s_siIII, shift_km_s_siIII)
    )


# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# Analysis routines
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------


def calc_velocity_shifts(z, z_sys, relativistic=True):
    """Calculate the velocity difference of a feature redshift with respect
    to the systemic redshift:

    dv = c ((1 + z)^2 - (1 + z_sys)^2) / ((1 + z)^2 + (1 + z_sys)^2)

    (relativistic), or dv = c (z - z_sys) / (1 + z_sys). Same as
    `linetools.utils.dv_from_z`, which this used to wrap.

    :param z: The redshift of the spectroscopic feature (e.g., absorption or \
        emission line).
    :type z: float or np.ndarray
    :param z_sys: The systemic redshift
    :type z_sys: float or np.ndarray
    :param relativistic: Boolean indicating whether the doppler velocity is \
        calculated assuming relativistic velocities.
    :type relativistic: bool
    :return: Returns the velocity difference in km/s.
    :rtype: astropy.units.Quantity

    """
    z = np.asarray(z, dtype=float)
    z_sys = np.asarray(z_sys, dtype=float)
    if relativistic:
        dv = ((1 + z) ** 2 - (1 + z_sys) ** 2) / ((1 + z) ** 2 + (1 + z_sys) ** 2)
    else:
        dv = (z - z_sys) / (1.0 + z_sys)
    return dv * const.c.to(KM_S)


def calc_eddington_luminosity(bh_mass):
    """Calculate the Eddington luminosity for a given black hole mass.

    :param bh_mass: Black hole mass as an astropy Quantity
    :type bh_mass: u.Quantity
    :return: Returns the Eddington luminosity in cgs units (erg/s).
    :rtype: u.Quantity

    """

    bh_mass = _quantity(bh_mass, u.Msun, "bh_mass")

    factor = (4 * np.pi * const.G * const.c * const.m_p) / const.sigma_T
    factor = factor.to(u.erg / u.s / u.Msun)

    return factor * bh_mass


def calc_eddington_ratio(lbol, bh_mass):
    """Calculate the Eddington ratio for a provided bolometric luminosity
    and black hole mass.

    :param lbol: Bolometric luminosity
    :type lbol: u.Quantity
    :param bh_mass: Black hole mass
    :type bh_mass: u.Quantity
    :return: Eddington ratio
    :rtype: astropy.units.Quantity (dimensionless)

    """

    lbol = _quantity(lbol, LUM_UNIT, "lbol")
    edd_lum = calc_eddington_luminosity(bh_mass)
    return (lbol / edd_lum).to(u.dimensionless_unscaled)


def se_bhmass_hbeta_vp06(hbeta_fwhm, cont_lwav, cont_wav=5100 * u.AA):
    """Calculate the single-epoch virial BH mass based on the Hbeta FWHM and
    monochromatic continuum luminosity at 5100A.

    This relationship is taken from Vestergaard & Peterson 2006, ApJ 641, 689

    Note that the Hbeta line width to establish this single-epoch virial
    estimator was established by using the FWHM of only the broad component.
    The line width was corrected for the spectral resolution as described in
    Peterson et al. 2004.

    The relationship is based on line width measurements of quasars published in
    Boroson & Green 1992 and Marziani 2003.

    "The sample standard deviation of the weighted average zero point offset,
    which shows the intrinsic scatter in the saample is +-0.43 dex. This
    value is more representative of the uncertainty zer point than is the
    formal error."

    :param hbeta_fwhm: FWHM of the Hbeta line in km/s
    :type hbeta_fwhm: astropy.units.Quantity
    :param cont_lwav: Monochromatic continuum luminosity at 5100A in erg/s/A
    :type cont_lwav: astropy.units.Quantity
    :param cont_wav: Wavelength of the monochromatic continuum luminosity \
        (default = 5100A).
    :type cont_wav: astropy.units.Quantity
    :return: Returns a tuple of the BH mass estimate based on the Hbeta FWHM \
        and a reference string for the single-epoch scaling relationship.
    :rtype: astropy.units.Quantity, string

    """

    hbeta_fwhm = _quantity(hbeta_fwhm, KM_S, "hbeta_fwhm")
    cont_lwav = _quantity(cont_lwav, LWAV_UNIT, "cont_lwav")
    cont_wav = _quantity(cont_wav, u.AA, "cont_wav")

    if _is_wav(cont_wav, 5100):
        reference = "Hbeta_VP06_fwhm"

        bhmass = virial_bh_mass(
            hbeta_fwhm,
            cont_wav * cont_lwav,
            zero_point=10**6.91,
            width_exp=2,
            lum_exp=0.5,
        )

        return bhmass, reference

    else:
        raise ValueError(
            "The monochromatic continuum luminosity should be estimated at 5100A."
        )


def se_bhmass_hbeta_park12(hbeta_fwhm, cont_lwav, cont_wav=5100 * u.AA):
    """Calculate the single-epoch virial BH mass based on the Hbeta FWHM and
    monochromatic continuum luminosity at 5100A.

    These relationships are taken from Park et al. 2012, ApJ 747, 30

    To measure the line width of Hβ and the continuum luminosity at 5100 Å,
    the authors follow the procedure given by Woo et al. (2006) and
    McGill et al. (2008), but with significant modifications:

    - converted to in rest-frame
    - three component continuum (PL, FeII of Boroson & Green 1982,
     host-galaxy starlight from Buruzual & Charlo 2003)
    - continuum models subtracted from spectra
    - Hβ narrow-line profile by fitting a tenth-order Gauss–Hermite series
     to [OIII] A 5007 line.
    - Hbeta component subtracted by scaling the [OIII] A 5007 line.
    - broad component of the Hβ line using a sixth-order Gauss–Hermite series


    To obtain the AGN continuum luminosity (nuclear luminosity, L5100,n ),
    the host-galaxy contribution to the total luminosity should be subtracted
    from the measured total luminosity.
    Authors use information from spectral decomposition.
    The starlight contributions are on the same order as the AGN
    contributions (see Table 2)!

    The authors find a discrepancy in the FWHM of mean and rms spectra of
    their sources (multiple-epoch of spectroscopy). They derive a correction
    to convert the broader SE FWHM to the narrower RMS FWHM, which they argue
    should be used in the SE virial estimators.


    :param hbeta_fwhm: FWHM of the Hbeta line in km/s
    :type hbeta_fwhm: astropy.units.Quantity
    :param cont_lwav: Monochromatic continuum luminosity at 5100A in erg/s/A
    :type cont_lwav: astropy.units.Quantity
    :param cont_wav: Wavelength of the monochromatic continuum luminosity \
        (default = 5100A).
    :type cont_wav: astropy.units.Quantity
    :return: Returns a tuple of the BH mass estimate based on the Hbeta FWHM \
        and a reference string for the single-epoch scaling relationship.
    :rtype: astropy.units.Quantity, string

    """

    hbeta_fwhm = _quantity(hbeta_fwhm, KM_S, "hbeta_fwhm")
    cont_lwav = _quantity(cont_lwav, LWAV_UNIT, "cont_lwav")
    cont_wav = _quantity(cont_wav, u.AA, "cont_wav")

    if _is_wav(cont_wav, 5100):
        if np.median(hbeta_fwhm.value) > 3000:
            reference = "Hbeta_Park12_fwhm_eq15"

            bhmass = virial_bh_mass(
                hbeta_fwhm,
                cont_wav * cont_lwav,
                zero_point=10**6.966,
                width_exp=1.734,
                lum_exp=0.518,
            )

            return bhmass, reference

        else:
            reference = "Hbeta_Park12_fwhm_eq16"

            bhmass = virial_bh_mass(
                hbeta_fwhm,
                cont_wav * cont_lwav,
                zero_point=10**6.985,
                width_exp=1.666,
                lum_exp=0.518,
            )

            return bhmass, reference

    else:
        raise ValueError(
            "The monochromatic continuum luminosity should be estimated at 5100A."
        )


def se_bhmass_gh05(line_fwhm, lwav, single_epoch_rel="Ha_LHa"):
    """Calculate the single-epoch virial BH mass based on the Hbeta FWHM and
    monochromatic continuum luminosity at 5100A.

    These relationships are taken from Greene & Ho, ApJ 630, 122

    Continuum was fitted using a double power law with a spectral break at
    5000A and an empirical FeII template (Boroson & Green 1992).
    Balmer continuum was not modeled as the fit does not extend blueward of
    3685A.

    Halpha:
    Lines are deconstructed in narrow and broad components, beginning with
    the [SII] AA 6716, 6731 lines. Mult-Gaussian models are fit to the narrow
    lines.
    The broad component of Halpha is fit with as many Gaussian components as
    needed to provide an acceptable fit.

    Hbeta:
    The template built from [SII] is used to model the narrow Hbeta component,
    whose centroid is fixed to the Halpha position, while its flux is limited
    by the value appropriate for Case B' recombination (Ha = 3.1 Hbeta,
    Osterbrock 1989). The [OIII] lines are fit simultaneously using a
    two-component Gaussian (core + blue wings). The broad Hbeta profile is
    modeled as a multi-component Gaussian analogous to Halpha.

    The single epoch virial estimators in this work are based on transforming
    the virial mass expression of Kaspi et al. 2000 (L_5100 and FWHM_Hb) into
    a new formalism that depends solely on the observed properties of the Ha
    line (or Hb line).

    JTS comment: It is unclear to what extent the LSF has been subtracted from
    the line FWHM.

    :param line_fwhm: FWHM of the Balmer line in km/s
    :type line_fwhm: astropy.units.Quantity
    :param lwav: Luminosity of either the continuum (erg/s/A, for
        "Hb_L5100") or the line (narrow+broad, erg/s)
    :type lwav: astropy.units.Quantity
    :param single_epoch_rel: Selection of the mode of the SE virial estimator
    :type single_epoch_rel: str
    :return: Returns a tuple of the BH mass estimate based on the Hbeta FWHM \
        and a reference string for the single-epoch scaling relationship.
    :rtype: astropy.units.Quantity, string

    """

    line_fwhm = _quantity(line_fwhm, KM_S, "line_fwhm")
    lwav_unit = LWAV_UNIT if single_epoch_rel == "Hb_L5100" else LUM_UNIT
    lwav = _quantity(lwav, lwav_unit, "lwav")

    if single_epoch_rel == "Ha_LHa":
        reference = "Halpha_GH05_HaLHa"

        bhmass = virial_bh_mass(
            line_fwhm,
            lwav,
            zero_point=2.0e6,
            width_exp=2.06,
            lum_exp=0.55,
            lum_pivot=1e42 * LUM_UNIT,
        )

        return bhmass, reference

    elif single_epoch_rel == "Hb_L5100":
        reference = "Hbeta_GH05_HaL5100"

        cont_wav = 5100 * u.AA

        bhmass = virial_bh_mass(
            line_fwhm,
            cont_wav * lwav,
            zero_point=4.4e6,
            width_exp=2,
            lum_exp=0.64,
        )

        return bhmass, reference

    elif single_epoch_rel == "Hb_LHb":
        reference = "Halpha_GH05_HbLHb"

        bhmass = virial_bh_mass(
            line_fwhm,
            lwav,
            zero_point=3.6e6,
            width_exp=2.0,
            lum_exp=0.56,
            lum_pivot=1e42 * LUM_UNIT,
        )

        return bhmass, reference

    else:
        raise ValueError(
            "Single epoch relation {} not supported. "
            "Possible relations are Ha_LHa, Ha_L5100, Hb_LHb"
        )


def se_bhmass_db20(line_fwhm, lwav, single_epoch_rel="Hb_LHb", virial_factor=1.0):
    """Calculate the single-epoch virial BH mass based on the Hbeta FWHM and
    the Hbeta line luminosity, following Dalla Bonta et al. 2020, ApJ 903, 112
    (https://ui.adsabs.harvard.edu/abs/2020ApJ...903..112D/abstract):

    log(M_BH / M_sun) = log(f) + 7.015 + 0.784 log(L_Hb / 10^42 erg/s)
                        + 1.387 log(FWHM / 10^3.5 km/s)

    Ported from upstream `v2_dev` (commits 33b78d5, a5a1749).

    :param line_fwhm: FWHM of the Hbeta line in km/s
    :type line_fwhm: astropy.units.Quantity
    :param lwav: Hbeta line luminosity in erg/s
    :type lwav: astropy.units.Quantity
    :param single_epoch_rel: Single-epoch scaling relation to use; only
        "Hb_LHb" (default)
    :type single_epoch_rel: str
    :param virial_factor: Virial factor f to scale the BH mass estimate
    :type virial_factor: float
    :return: Returns a tuple of the BH mass estimate and a reference string
        for the single-epoch scaling relationship.
    :rtype: astropy.units.Quantity, string
    """

    line_fwhm = _quantity(line_fwhm, KM_S, "line_fwhm")
    lwav = _quantity(lwav, LUM_UNIT, "lwav")

    logger.info(f"The virial factor is set to {virial_factor}")

    if single_epoch_rel == "Hb_LHb":
        reference = "DB20_Hb_LHb"

        bhmass = virial_bh_mass(
            line_fwhm,
            lwav,
            zero_point=virial_factor * 10**7.015,
            width_exp=1.387,
            lum_exp=0.784,
            width_pivot=10**3.5 * KM_S,
            lum_pivot=1e42 * LUM_UNIT,
        )

        return bhmass, reference

    else:
        raise ValueError(
            f"Single epoch relation {single_epoch_rel} not supported. "
            "Possible relations are Hb_LHb"
        )


def se_bhmass_civ_vp06_fwhm(civ_fwhm, cont_lwav, cont_wav):
    """Calculate the single-epoch virial BH mass based on the CIV FWHM and
    monochromatic continuum luminosity at 1350A.

    The monochromatic continuum luminosity at 1450A can be used without error or
    penalty in accuracy.

    This relationship is taken from Vestergaard & Peterson 2006, ApJ 641, 689

    The FWHM of the CIV line was measured with the methodology described in
    Peterson et al. 2004. The line width measurements to establish the CIV
    single-epoch relation are corrected for resolution effects as described in
    Peterson et al. 2004.

    "The sample standard deviation of the weighted average zero point offset,
    which shows the intrinsic scatter in the saample is +-0.36 dex. This
    value is more representative of the uncertainty zer point than is the
    formal error."

    :param civ_fwhm: FWHM of the CIV line in km/s
    :type civ_fwhm: astropy.units.Quantity
    :param cont_lwav: Monochromatic continuum luminosity at 1350A/1450A in \
        erg/s/A.
    :type cont_lwav: astropy.units.Quantity
    :param cont_wav: Wavelength of the monochromatic continuum luminosity in A.
    :type cont_wav: astropy.units.Quantity
    :return: Returns a tuple of the BH mass estimate based on the CIV FWHM \
        and a reference string for the single-epoch scaling relationship.
    :rtype: astropy.units.Quantity, string

    """

    civ_fwhm = _quantity(civ_fwhm, KM_S, "civ_fwhm")
    cont_lwav = _quantity(cont_lwav, LWAV_UNIT, "cont_lwav")
    cont_wav = _quantity(cont_wav, u.AA, "cont_wav")

    if _is_wav(cont_wav, 1350, 1450):
        reference = "CIV_VP06_fwhm"

        bhmass = virial_bh_mass(
            civ_fwhm,
            cont_wav * cont_lwav,
            zero_point=10**6.66,
            width_exp=2,
            lum_exp=0.53,
        )

        return bhmass, reference

    else:
        raise ValueError(
            "The monochromatic continuum luminosity "
            "should be estimated at either 1350A or 1450A."
        )


def se_bhmass_civ_vp06_sigma(civ_sigma, cont_lwav, cont_wav):
    """Calculate the single-epoch virial BH mass based on the CIV line
    dispersion (sigma) and monochromatic continuum luminosity at 1350A.

    The monochromatic continuum luminosity at 1450A can be used without error or
    penalty in accuracy.

    This relationship is taken from Vestergaard & Peterson 2006, ApJ 641, 689

    The FWHM of the CIV line was measured with the methodology described in
    Peterson et al. 2004. The line width measurements to establish the CIV
    single-epoch relation are corrected for resolution effects as described in
    Peterson et al. 2004.

    Peterson et al. (2004) note a number of advantages to using sigma rathern
    than the FWHM as the line width measure.

    "The sample standard deviation of the weighted average zero point offset,
    which shows the intrinsic scatter in the saample is +-0.33 dex. This
    value is more representative of the uncertainty zer point than is the
    formal error."

    :param civ_sigma: Line dispersion (sigma) of the CIV line in km/s
    :type civ_sigma: astropy.units.Quantity
    :param cont_lwav: Monochromatic continuum luminosity at 1350A/1450A in \
        erg/s/A.
    :type cont_lwav: astropy.units.Quantity
    :param cont_wav: Wavelength of the monochromatic continuum luminosity in A.
    :type cont_wav: astropy.units.Quantity
    :return: Returns a tuple of the BH mass estimate based on the CIV sigma \
        and a reference string for the single-epoch scaling relationship.
    :rtype: astropy.units.Quantity, string

    """

    civ_sigma = _quantity(civ_sigma, KM_S, "civ_sigma")
    cont_lwav = _quantity(cont_lwav, LWAV_UNIT, "cont_lwav")
    cont_wav = _quantity(cont_wav, u.AA, "cont_wav")

    if _is_wav(cont_wav, 1350, 1450):
        reference = "CIV_VP06_sigma"

        bhmass = virial_bh_mass(
            civ_sigma,
            cont_wav * cont_lwav,
            zero_point=10**6.73,
            width_exp=2,
            lum_exp=0.53,
        )

        return bhmass, reference

    else:
        raise ValueError(
            "The monochromatic continuum luminosity "
            "should be estimated at either 1350A or 1450A."
        )


def se_bhmass_mgii_vo09_fwhm(mgii_fwhm, cont_lwav, cont_wav):
    """Calculate the single-epoch virial BH mass based on the MgII FWHM
    and monochromatic continuum luminosity at 1350A, 2100A, 3000A, or 5100A.

    This relationship is taken from Vestergaard & Osmer 2009, ApJ 641, 689

    To determine the FWHM of the MgII line, the authors modeled the FeII
    emission beneath the MgII line with the Vestergaard & Wilkes 2001 and the
    Boroson & Green 1992 iron templates.

    Most of the MgII lines were modeled with a single Gaussian component,
    in cases of high-quality spectra two Gaussian components were used. For
    the single-Gaussian components the authors adopted the measurements of
    the FWHM and uncertainties tabulated by Forster et al. (their Table 5).
    For multi-Gaussian components the FWHM of the full modeled profile was
    measured.

    As the subtraction of the underlying FeII continuum can have systematic
    effects on the measurement of the MgII FWHM and therefore the BH mass
    estimate it is advised to always employ the same continuum construction
    procedure as in the reference sample that established the single-epoch
    virial relationship.

    "The absolute 1 sigma scatter in the zero points is 0.55dex,
    which includes the factor ~2.9 uncertainties of the reverberation mapping
    masses to which these mass estimation relations are anchored (see
    Vestergaard & Peterson 2006 and Onken et al. 2004 for details)"

    :param mgii_fwhm: FWHM of the MgII line in km/s
    :type mgii_fwhm: astropy.units.Quantity
    :param cont_lwav: Monochromatic continuum luminosity in erg/s/A.
    :type cont_lwav: astropy.units.Quantity
    :param cont_wav: Wavelength of the monochromatic continuum luminosity in A.
    :type cont_wav: astropy.units.Quantity
    :return: Returns a tuple of the BH mass estimate based on the MgII FWHM \
        and a reference string for the single-epoch scaling relationship.
    :rtype: astropy.units.Quantity, string

    """

    zero_points = {1350: 6.72, 2100: 6.79, 3000: 6.86, 5100: 6.96}

    mgii_fwhm = _quantity(mgii_fwhm, KM_S, "mgii_fwhm")
    cont_lwav = _quantity(cont_lwav, LWAV_UNIT, "cont_lwav")
    cont_wav = _quantity(cont_wav, u.AA, "cont_wav")
    matches = [wav for wav in zero_points if _is_wav(cont_wav, wav)]

    if matches:
        reference = f"MgII_VO09_fwhm_{float(matches[0])}"

        zp = zero_points[matches[0]]

        bhmass = virial_bh_mass(
            mgii_fwhm,
            cont_wav * cont_lwav,
            zero_point=10**zp,
            width_exp=2,
            lum_exp=0.5,
        )

        return bhmass, reference

    else:
        raise ValueError(
            "The monochromatic continuum luminosity "
            "should be estimated at 1350A, 2100A, 3000A, or "
            "5100A."
        )


def se_bhmass_mgii_s11_fwhm(mgii_fwhm, cont_lwav, cont_wav):
    """Calculate the single-epoch virial BH mass based on the MgII FWHM
    and monochromatic continuum luminosity at 3000A.

    This relationship is taken from Shen et al. 2011, ApJ, 194, 45

    To model the FeII contribution beneath MgII line the authors use
    empirical FeII templates from Borosn & Grenn 1992, Vestergaard & Wilkes
    2001, and Salviander 2007.

    Salviander modified the Vestergaard & Wilkes (2001) template in the
    region of 2780–2830A centered on MgII, where the Vestergaard & Wilkes (2001)
    template is set to zero. For this region, they incorporate a theoretical
    FeII model ofSigut & Pradhan (2003) scaled to match the Vestergaard &
    Wilkes(2001) template at neighboring wavelengths.

    As the subtraction of the underlying FeII continuum can have systematic
    effects on the measurement of the MgII FWHM and therefore the BH mass
    estimate it is advised to always employ the same continuum construction
    procedure as in the reference sample that established the single-epoch
    virial relationship.

    :param mgii_fwhm: FWHM of the MgII line in km/s
    :type mgii_fwhm: astropy.units.Quantity
    :param cont_lwav: Monochromatic continuum luminosity in erg/s/A.
    :type cont_lwav: astropy.units.Quantity
    :param cont_wav: Wavelength of the monochromatic continuum luminosity in A.
    :type cont_wav: astropy.units.Quantity
    :return: Returns a tuple of the BH mass estimate based on the MgII FWHM \
        and a reference string for the single-epoch scaling relationship.
    :rtype: astropy.units.Quantity, string

    """

    mgii_fwhm = _quantity(mgii_fwhm, KM_S, "mgii_fwhm")
    cont_lwav = _quantity(cont_lwav, LWAV_UNIT, "cont_lwav")
    cont_wav = _quantity(cont_wav, u.AA, "cont_wav")

    if _is_wav(cont_wav, 3000):
        reference = "MgII_S11_fwhm_3000"

        zp = 6.74

        bhmass = virial_bh_mass(
            mgii_fwhm,
            cont_wav * cont_lwav,
            zero_point=10**zp,
            width_exp=2,
            lum_exp=0.62,
        )

        return bhmass, reference

    else:
        raise ValueError(
            "The monochromatic continuum luminosity should be estimated at 3000A."
        )


def calc_bolometric_luminosity(cont_lwav, cont_wav, reference="Shen2011"):
    """Calculate the bolometric luminosity from the monochromatic continuum
    luminosity (erg/s/A) using bolometric correction factors from the
    literature.

    The following bolometric corrections are available
    cont_wav = 1350, reference = Shen2011
    cont_wav = 3000, reference = Shen2011
    cont_wav = 5100, reference = Shen2011

    The Shen et al. 2011 (ApJ, 194, 45) bolometric corrections are based on the
    composite spectral energy distribution (SED) in Richards et al. 2006 (
    ApJ, 166,470).

    :param cont_lwav: Monochromatic continuum luminosity in erg/s/A.
    :type cont_lwav: astropy.units.Quantity
    :param cont_wav: Wavelength of the monochromatic continuum luminosity in A.
    :type cont_wav: astropy.units.Quantity
    :param reference: A reference string to select from the available \
        bolometric corrections.
    :type reference: string
    :return: Returns a tuple of the bolometric luminosity in erg/s and a \
        reference string indicating the publication and continuum wavelength \
        of the bolometric correction.
    :rtype: astropy.units.Quantity, string

    """

    cont_lwav = _quantity(cont_lwav, LWAV_UNIT, "cont_lwav")
    cont_wav = _quantity(cont_wav, u.AA, "cont_wav")

    if _is_wav(cont_wav, 1350) and reference == "Shen2011":
        reference = "Shen2011_1350"
        return (cont_lwav * cont_wav * 3.81).to(LUM_UNIT), reference

    if _is_wav(cont_wav, 3000) and reference == "Shen2011":
        reference = "Shen2011_3000"
        return (cont_lwav * cont_wav * 5.15).to(LUM_UNIT), reference

    if _is_wav(cont_wav, 5100) and reference == "Shen2011":
        reference = "Shen2011_5100"
        return (cont_lwav * cont_wav * 9.26).to(LUM_UNIT), reference
    else:
        raise ValueError(
            "No bolometric correction available for the "
            "supplied combination of continuum wavelength and "
            "reference."
        )


def calc_bolometric_luminosity_db20(line_lwav):
    """Calculate the bolometric luminosity from the Hbeta line luminosity
    with the bolometric correction of Dalla Bonta et al. 2020:

    log(L_bol / erg/s) = 44.717 + 0.802 (log(L_Hb / erg/s) - 41.746)

    Ported from upstream `v2_dev` (commit 33b78d5).

    :param line_lwav: Hbeta line luminosity in erg/s
    :type line_lwav: astropy.units.Quantity
    :return: Returns a tuple of the bolometric luminosity in erg/s and a
        reference string indicating the publication of the bolometric
        correction.
    :rtype: astropy.units.Quantity, string
    """

    line_lwav = _quantity(line_lwav, LUM_UNIT, "line_lwav")
    log_lbol = 44.717 + 0.802 * (np.log10(line_lwav.value) - 41.746)

    lbol = 10**log_lbol * u.erg / u.s

    reference = "DB20_Hb"

    return lbol, reference


def correct_CIV_fwhm_for_blueshift(civ_fwhm, blueshift):
    """Correct the CIV FWHM for the CIV blueshifts using the relation
    determined in Coatman et al. (2017, MNRAS, 465, 2120).

    The correction follows Eq. 4 of Coatman et al. (2017).

    :param civ_fwhm: FWHM of the CIV line in km/s
    :type civ_fwhm: astropy.units.Quantity
    :param blueshift: Blueshift of the CIV line in km/s
    :type blueshift: astropy.units.Quantity
    :return: Corrected FWHM in km/s
    :rtype: astropy.units.Quantity

    """

    civ_fwhm = _quantity(civ_fwhm, KM_S, "civ_fwhm")
    blueshift = _quantity(blueshift, KM_S, "blueshift")
    return civ_fwhm / (0.41 * blueshift / (1000 * u.km / u.s) + 0.62)


def se_bhmass_civ_c17_fwhm(civ_fwhm, cont_lwav, cont_wav):
    """Calculate the single-epoch virial BH mass based on the CIV FWHM and
    monochromatic continuum luminosity at 1350A.

    This relationship follows Eq.6 of from Coatman et al. (2017, MNRAS, 465,
    2120)

    The FWHM of the CIV line was corrected by the CIV blueshift using their
    Eq. 4. In this study the CIV line was modeled with sixth order
    Gauss-Hermite (GH) polynomials using the normalization of van der Marel
    & Franx (1993) and the functional forms of Cappelari et al. (2002). GH
    polynomials were chose because they are flexble enough to model the often
    very asymmetric CIV line profile.

    The authors state that using commonly employed three Gaussian components,
    rather than the GH polynomials, resulted in only marginal differences in
    the line parameters.

    :param civ_fwhm: FWHM of the CIV line in km/s
    :type civ_fwhm: astropy.units.Quantity
    :param cont_lwav: Monochromatic continuum luminosity at 1350A in erg/s/A.
    :type cont_lwav: astropy.units.Quantity
    :param cont_wav: Wavelength of the monochromatic continuum luminosity in A.
    :type cont_wav: astropy.units.Quantity
    :return: Returns a tuple of the BH mass estimate based on the CIV FWHM \
        and a reference string for the single-epoch scaling relationship.
    :rtype: astropy.units.Quantity, string

    """

    civ_fwhm = _quantity(civ_fwhm, KM_S, "civ_fwhm")
    cont_lwav = _quantity(cont_lwav, LWAV_UNIT, "cont_lwav")
    cont_wav = _quantity(cont_wav, u.AA, "cont_wav")

    if _is_wav(cont_wav, 1350):
        reference = "CIV_C17_fwhm"

        bhmass = virial_bh_mass(
            civ_fwhm,
            cont_wav * cont_lwav,
            zero_point=10**6.71,
            width_exp=2,
            lum_exp=0.53,
        )

        return bhmass, reference

    else:
        raise ValueError(
            "The monochromatic continuum luminosity should be estimated at 1350A."
        )
