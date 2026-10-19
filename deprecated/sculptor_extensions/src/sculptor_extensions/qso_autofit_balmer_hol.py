"""`Balmer_continuum_with_hol`, removed from
`sculptor_extensions/Qso_autofit.py` on 2026-10-19 (review §3).

`Qso_autofit.initialise_balmer` now uses
`qso_advanced_models.BalmerContinuumWithHOL`, the newer model (ported from
the later `out.py`), ~3x faster. The two do not give the same numbers:

- scale: this one normalises the continuum to its peak and multiplies by
  1e17, the new one keeps the physical shape and multiplies by 1e-8
  (~2e17 apart), so `amp_be` values/priors need rescaling;
- shape: the high-order lines are normalised differently (up to ~40% of
  the peak below the Balmer edge);
- the "smooth cutoff" above the edge compared flux values with a
  wavelength and never did anything; `wave_2.txt` was re-read on every
  call; FWHM-to-sigma used 2.35 instead of 2 sqrt(2 ln 2).

Kept for reference only; not importable as-is (it relied on
`Qso_autofit.py`'s module-level imports and constants).
"""


def Balmer_continuum_with_hol(x, z, amp_be, Te, tau_be, lambda_be, fwhm):
    """
    Compute the Balmer pseudo-continuum with high-order line (HOL) emission created due to super-position
    of higher order Balmer lines (from n >= 7).

    This function returns the total flux from Balmer contribution in units of 1e-17 erg/s/cm2.
    Line intensities are provided in Storey and Hummer 1995 (http://adsabs.harvard.edu/abs/1995MNRAS.272...41S).
    Data files downloaded from (http://cdsarc.u-strasbg.fr/ftp/cats/VI/64/).
    The function uses their case B approximation for Hydrogen,
    with temperature 15000K and electon density of log nH =9.
    One can customise the parameters depending on the case.

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
    :param fwhm: FWHM of the Gaussian profile to broaden the template (km/s)
    :type fwhm: int

    :return total_flux: Balmer continuum model
    :rtype: np.ndarray
    """
    rest_wavelength = x / (1 + z)

    # Blackbody continuum till Balmer_limit
    bb = BlackBody(temperature=Te * u.K)
    black_body_lambda = (
        bb(rest_wavelength * u.AA).value * c_AngstroemPerSecond / rest_wavelength**2
    )
    raw_continuum = black_body_lambda * (
        1 - np.exp(-tau_be * (rest_wavelength / lambda_be) ** 3)
    )

    transition_width = 20.0
    smooth_cutoff = 1 / (1 + np.exp((rest_wavelength - lambda_be) / transition_width))
    continuum = raw_continuum
    continuum[continuum > lambda_be] = continuum[continuum > lambda_be] * smooth_cutoff
    continuum /= np.max(continuum)

    # High-order line emissivities
    df = np.loadtxt(datadir + "wave_2.txt")
    transitions = df[:, 0]
    emiss = df[:, 1]
    flux_density = np.zeros_like(rest_wavelength)

    for line, em in zip(transitions, emiss, strict=True):
        sigma = (fwhm / c_kms) * line / 2.35
        profile = np.exp(-0.5 * ((rest_wavelength - line) / sigma) ** 2)
        flux_density += em * profile / sigma

    # Normalizing hol to continuum at the Balmer edge
    edge_val = np.interp(lambda_be, rest_wavelength, flux_density)
    flux_density /= (1 / 0.3) * edge_val

    total_flux = (amp_be * (continuum + flux_density)) * 1e17
    return total_flux
