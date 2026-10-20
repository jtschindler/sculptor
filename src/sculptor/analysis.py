import logging
import warnings

import extinction
import numpy as np
import scipy as sp
from astropy import units as u
from astropy.cosmology import FlatLambdaCDM
from astropy.table import QTable
from scipy.ndimage import gaussian_filter1d
from speconed import speconed as sod
from tqdm import tqdm

from sculptor import measurements as meas
from sculptor.component import FakeCode
from sculptor.utils import c_km_s

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------------------
# Model functions
# ------------------------------------------------------------------------------


def power_law_at2500(x, redsh, amp, slope, **kwargs):
    """Power law anchored at rest-frame 2500 Angstroem.

    :param x: Observed-frame dispersion in Angstroem
    :type x: np.ndarray
    :param redsh: Redshift
    :type redsh: float
    :param amp: Amplitude at rest-frame 2500 Angstroem
    :type amp: float
    :param slope: Power-law slope (in f_lambda)
    :type slope: float
    :return: Power law model
    :rtype: np.ndarray
    """

    return amp * ((x / (2500 * (redsh + 1))) ** slope)


def ext_power_law_at2500(x, redsh, amp, slope, a_v, **kwargs):
    """Power law anchored at rest-frame 2500 Angstroem, reddened by a Calzetti
    et al. (2000) attenuation law (R_V = 4.05) in the rest frame.

    :param x: Observed-frame dispersion in Angstroem
    :type x: np.ndarray
    :param redsh: Redshift
    :type redsh: float
    :param amp: Amplitude at rest-frame 2500 Angstroem
    :type amp: float
    :param slope: Power-law slope (in f_lambda)
    :type slope: float
    :param a_v: V-band attenuation in magnitudes
    :type a_v: float
    :return: Reddened power law model
    :rtype: np.ndarray
    """

    flux = power_law_at2500(x, redsh, amp, slope)

    ext = extinction.calzetti00(x / (redsh + 1), a_v, 4.05)

    return extinction.apply(ext, flux)


def scattered_power_law_at2500(x, redsh, amp, slope, f_sc, **kwargs):
    """Power law anchored at rest-frame 2500 Angstroem, scaled by 10**f_sc
    (e.g. a scattered-light fraction, in dex).

    :param x: Observed-frame dispersion in Angstroem
    :type x: np.ndarray
    :param redsh: Redshift
    :type redsh: float
    :param amp: Amplitude at rest-frame 2500 Angstroem
    :type amp: float
    :param slope: Power-law slope (in f_lambda)
    :type slope: float
    :param f_sc: Scaling in dex
    :type f_sc: float
    :return: Scaled power law model
    :rtype: np.ndarray
    """

    return 10**f_sc * power_law_at2500(x, redsh, amp, slope)


def template_model(
    x,
    redsh,
    amp,
    fwhm,
    intr_fwhm,
    templ_disp=None,
    templ_fluxden=None,
    templ_disp_unit_str=None,
    templ_fluxden_unit_str=None,
    **kwargs,
):
    """Template spectrum (e.g. an iron template) broadened, redshifted and
    scaled.

    The template is convolved to a total FWHM of `fwhm` (with a Gaussian of
    FWHM sqrt(fwhm^2 - intr_fwhm^2), `SpecOneD.broaden_by_gaussian`), its
    dispersion is multiplied by (1 + redsh) without changing the flux
    density (the amplitude is a free parameter), and it is linearly
    interpolated onto `x` (0 outside the template).

    :param x: Observed-frame dispersion
    :type x: np.ndarray
    :param redsh: Redshift
    :type redsh: float
    :param amp: Amplitude scaling of the template
    :type amp: float
    :param fwhm: Target FWHM in km/s
    :type fwhm: float
    :param intr_fwhm: Intrinsic FWHM of the template in km/s
    :type intr_fwhm: float
    :param templ_disp: Rest-frame template dispersion, in the same unit as
        `x`
    :type templ_disp: np.ndarray
    :param templ_fluxden: Template flux density
    :type templ_fluxden: np.ndarray
    :param templ_disp_unit_str: Template dispersion unit (astropy "cds"
        format)
    :type templ_disp_unit_str: str
    :param templ_fluxden_unit_str: Template flux density unit (astropy
        "cds" format)
    :type templ_fluxden_unit_str: str
    :return: Template model
    :rtype: np.ndarray
    """

    if isinstance(templ_disp_unit_str, str):
        dispersion_unit = u.Unit(templ_disp_unit_str, format="cds")
    else:
        dispersion_unit = None
    if isinstance(templ_fluxden_unit_str, str):
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
    spec.dispersion = spec.dispersion * (1.0 + redsh)

    # Return interpolation function
    f = sp.interpolate.interp1d(
        spec.dispersion,
        spec.fluxden,
        kind="linear",
        bounds_error=False,
        fill_value=(0, 0),
    )

    return f(x) * amp


class TemplateModel:
    """Fast version of `template_model` for one fixed template.

    `template_model` rebuilds a `SpecOneD` from the template and convolves
    it in log-wavelength on every call. This class converts the template to
    a uniform log-wavelength grid once, so a call is one Gaussian filter
    plus two interpolations (~20x faster for the packaged iron templates).
    Use an instance as the function of a `FitComponent`; its parameters are
    `redsh`, `amp`, `fwhm` and `intr_fwhm` (as for `template_model`).
    Instances can be pickled (for `FitModel.save` and `pool=`).

    The output reproduces `template_model`, including its normalization,
    which comes from `speconed`'s `SpecOneD.broaden_by_gaussian`: the
    convolution is divided by the number of template pixels and by
    (2 sqrt(2 ln 2))^2, so the amplitude scale depends on the template's
    wavelength range (a factor 1 / (N * dlnlambda) ~ 1 / ln(lambda_max /
    lambda_min)). Kept for consistency with `template_model`; flagged for
    review in `TODO_speconed.md`.

    It also reproduces a quirk of `broaden_by_gaussian`: for a template
    with an even number of pixels, `np.convolve(..., mode="same")` shifts
    the broadened spectrum by +1/2 pixel in ln(wavelength) (~53 km/s for
    the BG92 and Tsuzuki optical iron templates; no shift for the UV
    templates, which have an odd number of pixels). Also flagged in
    `TODO_speconed.md`. With both reproduced, the output agrees with
    `template_model` to ~1e-12 (odd) / ~2e-3 (even pixel count) of the peak.

    Ported from the upstream `qso_auto_fit` branch (`Qso_autofit_2.py`,
    3b85394), with the normalization matched to `template_model` (the
    original was ~5.6% higher for the UV iron template) and without
    `SpecOneD`'s private `_to_log_wavelength`.

    :param templ_disp: Rest-frame template dispersion in Angstroem
    :type templ_disp: np.ndarray
    :param templ_fluxden: Template flux density per Angstroem
    :type templ_fluxden: np.ndarray
    """

    def __init__(self, templ_disp, templ_fluxden):
        self.templ_disp = np.asarray(templ_disp, dtype=float)
        templ_fluxden = np.asarray(templ_fluxden, dtype=float)

        # Flux per ln(wavelength) on a uniform ln(wavelength) grid, as in
        # SpecOneD.broaden_by_gaussian.
        n = self.templ_disp.size
        self.log_disp = np.linspace(
            np.log(self.templ_disp.min()), np.log(self.templ_disp.max()), num=n
        )
        self.log_fluxden = np.interp(
            self.log_disp, np.log(self.templ_disp), templ_fluxden * self.templ_disp
        )
        self.dloglam = self.log_disp[1] - self.log_disp[0]
        # Grid the convolved spectrum is read back on: shifted by +1/2 pixel
        # for an even number of pixels, as np.convolve(mode="same") does in
        # SpecOneD.broaden_by_gaussian (see the class docstring).
        self._out_log_disp = self.log_disp + (0.5 * self.dloglam if n % 2 == 0 else 0)
        self._kernel_offsets = self.log_disp - (
            (self.log_disp.max() - self.log_disp.min()) / 2.0 + self.log_disp.min()
        )

        self.__code__ = FakeCode(
            co_argcount=5, co_varnames=("x", "redsh", "amp", "fwhm", "intr_fwhm")
        )
        self.__name__ = "TemplateModel"

    def __call__(self, x, redsh, amp, fwhm, intr_fwhm, **kwargs):
        """Evaluate the broadened, redshifted and scaled template.

        :param x: Observed-frame dispersion in Angstroem
        :type x: np.ndarray
        :param redsh: Redshift
        :type redsh: float
        :param amp: Amplitude scaling of the template
        :type amp: float
        :param fwhm: Target FWHM in km/s
        :type fwhm: float
        :param intr_fwhm: Intrinsic FWHM of the template in km/s
        :type intr_fwhm: float
        :return: Template model
        :rtype: np.ndarray
        """

        convol_fwhm = np.sqrt(fwhm**2 - intr_fwhm**2)
        sigma_log = convol_fwhm / c_km_s / (2 * np.sqrt(2 * np.log(2)))

        conv = gaussian_filter1d(
            self.log_fluxden,
            sigma=sigma_log / self.dloglam,
            mode="constant",
            cval=0.0,
            truncate=8.0,
        )
        # SpecOneD.broaden_by_gaussian's normalization: its kernel is the
        # normalized Gaussian sampled on the grid (sum ~ 1/dlnlambda), and
        # the convolution is divided by the number of pixels and (8 ln 2).
        kernel_sum = np.sum(np.exp(-0.5 * (self._kernel_offsets / sigma_log) ** 2)) / (
            np.sqrt(2 * np.pi) * sigma_log
        )
        conv *= kernel_sum / self.log_disp.size / (8 * np.log(2))

        # Back to flux density per Angstroem on the template's own grid, then
        # redshifted onto x (0 outside the template), as template_model does.
        out_disp = np.exp(self._out_log_disp)
        fluxden = np.interp(self.templ_disp, out_disp, conv / out_disp)
        return (
            np.interp(x, self.templ_disp * (1.0 + redsh), fluxden, left=0, right=0)
            * amp
        )


def line_model_gaussian(x, redsh, flux, cen, fwhm_km_s, **kwargs):
    """Gaussian line model

    The central wavelength of the Gaussian line model is determined by the
    central wavelength cen and the redshift, z. These parameters are degenerate
    in a line fit and it is adviseable to fix one of them (to predetermined
    values e.g., the redshift or the central wavelength).

    The width of the line is set by the FWHM in km/s.

    The Gaussian is normalized.

    :param x: Dispersion of the continuum model
    :type x: np.ndarray
    :param redsh: Redshift
    :type redsh: float
    :param flux: Integrated flux of the Gaussian
    :type flux: float
    :param cen: Central wavelength
    :type cen: float
    :param fwhm_km_s: FWHM of the Gaussian in km/s
    :type fwhm_km_s: float
    :return: Gaussian line model
    :rtype: np.ndarray

    """

    # Redshift central wavelength
    cen = cen * (1.0 + redsh)

    # Calculate sigma from fwhm
    fwhm = fwhm_km_s / c_km_s * cen
    sigma = fwhm / np.sqrt(8 * np.log(2))

    return (
        flux / (sigma * np.sqrt(2 * np.pi)) * np.exp(-((x - cen) ** 2) / (2 * sigma**2))
    )


# ------------------------------------------------------------------------------
# High-level analysis routines
# ------------------------------------------------------------------------------


def _default_axis(model, dispersion, dispersion_unit, fluxden_unit):
    """Default the analysis axis and units to the model's spectrum."""
    if dispersion is None:
        model._require_spectrum("analyze the model without `dispersion`")
        dispersion = model.spec.dispersion
        dispersion_unit = dispersion_unit or model.spec.dispersion_unit
        fluxden_unit = fluxden_unit or model.spec.fluxden_unit
    return dispersion, dispersion_unit, fluxden_unit


def _default_cosmology(cosmology):
    """FlatLambdaCDM(H0=70, Om0=0.3, Tcmb0=2.725), with a warning, if None."""
    if cosmology is None:
        cosmology = FlatLambdaCDM(H0=70, Om0=0.3, Tcmb0=2.725)
        logger.warning(
            "No Cosmology specified. Assuming default "
            "FlatLambdaCDM cosmology with H0=70, Om0=0.3, Tcmb0=2.725."
        )
    return cosmology


def _remove_extinction(fluxden, dispersion, redshift, a_v, ext_law="calzetti00"):
    """Correct an observed-frame model flux density for host extinction.

    :param fluxden: Model flux density
    :param dispersion: Observed-frame dispersion in Angstroem
    :param redshift: Redshift, to compute the rest-frame wavelengths
    :param a_v: V-band extinction in magnitudes
    :param ext_law: Extinction law; only "calzetti00" (R_V = 4.05)
    :return: Extinction-corrected flux density
    """

    if ext_law == "calzetti00":
        ext = extinction.calzetti00(dispersion / (1 + redshift), a_v, 4.05)
    else:
        raise ValueError("No valid extinction law specified")
    return extinction.remove(ext, fluxden)


def analyze_mcmc_results(
    model,
    flat_chain,
    cont_dict,
    emfeat_dictlist,
    redshift,
    cosmology,
    dispersion=None,
    dispersion_unit=None,
    fluxden_unit=None,
    width=10,
    emfeat_meas=None,
    a_v=None,
    ext_law="calzetti00",
):
    """Analyze the continuum and emission features for every posterior sample.

    :param model: Fitted model
    :type model: sculptor.model.FitModel
    :param flat_chain: Posterior samples, shape (n_samples, n_variable_params)
    :type flat_chain: np.ndarray
    :param cont_dict: Continuum setup with keys "components" and
        "rest_frame_wavelengths" (see `analyze_continuum`)
    :type cont_dict: dict
    :param emfeat_dictlist: One dict per emission feature with keys
        "feature_name", "components", "rest_frame_wavelength" and optionally
        "disp_range" and "fwhm_method" (see `analyze_emission_feature`)
    :type emfeat_dictlist: list(dict)
    :return: One row of measurements (with units) per sample
    :rtype: astropy.table.QTable
    """

    logger.info("Analyzing MCMC results")

    cont_component = cont_dict["components"]
    cont_wavelengths = cont_dict["rest_frame_wavelengths"]

    result_table = None

    for idx in tqdm(range(flat_chain.shape[0])):
        params = flat_chain[idx, :]

        result_dict = {}

        cont_result_dict = analyze_continuum(
            model,
            cont_component,
            params,
            cont_wavelengths,
            cosmology,
            redshift=redshift,
            dispersion_unit=dispersion_unit,
            fluxden_unit=fluxden_unit,
            dispersion=dispersion,
            width=width,
        )

        for emfeat_dict in emfeat_dictlist:
            feature_name = emfeat_dict["feature_name"]
            em_components = emfeat_dict["components"]
            rest_frame_wavelength = emfeat_dict["rest_frame_wavelength"]
            if "disp_range" in emfeat_dict:
                disp_range = emfeat_dict["disp_range"]
            else:
                disp_range = None
            if "fwhm_method" in emfeat_dict:
                fwhm_method = emfeat_dict["fwhm_method"]
            else:
                fwhm_method = "spline"

            emfeat_result_dict = analyze_emission_feature(
                model,
                feature_name,
                em_components,
                params,
                rest_frame_wavelength,
                cont_components=cont_component,
                redshift=redshift,
                dispersion=dispersion,
                emfeat_meas=emfeat_meas,
                disp_range=disp_range,
                cosmology=cosmology,
                fwhm_method=fwhm_method,
                dispersion_unit=dispersion_unit,
                fluxden_unit=fluxden_unit,
                a_v=a_v,
                ext_law=ext_law,
            )

            if emfeat_result_dict is not None:
                result_dict.update(emfeat_result_dict)

        if cont_result_dict is not None:
            result_dict.update(cont_result_dict)

        if result_table is None:
            # Initialize QTable for results
            # Initialize column names
            result_table = QTable(names=result_dict.keys())
            # Initialize column units
            for column in result_table.columns:
                if isinstance(
                    result_dict[column],
                    (u.Quantity, u.IrreducibleUnit, u.CompositeUnit),
                ):
                    result_table[column].unit = result_dict[column].unit

        result_table.add_row(result_dict)

    return result_table


def _mcmc_analyze(*args, **kwargs):
    """Deprecated old private name of `analyze_mcmc_results`."""
    warnings.warn(
        "_mcmc_analyze is deprecated; use analyze_mcmc_results",
        DeprecationWarning,
        stacklevel=2,
    )
    return analyze_mcmc_results(*args, **kwargs)


emfeat_measures_default = ["peak_fluxden", "peak_redsh", "EW", "FWHM", "flux", "lum"]


def analyze_emission_feature(
    model,
    feature_name,
    components,
    params_values,
    rest_frame_wavelength,
    cont_components=None,
    redshift=None,
    dispersion=None,
    emfeat_meas=None,
    disp_range=None,
    cosmology=None,
    fwhm_method="spline",
    dispersion_unit=None,
    fluxden_unit=None,
    a_v=None,
    ext_law="calzetti00",
):
    """Measure an emission feature on the model built from one parameter
    vector (e.g. one posterior sample).

    At present this analysis assumes that the spectra are in the following
    units: flux density - erg/s/cm^2/AA, dispersion - AA.

    :param model: Fitted model
    :type model: sculptor.model.FitModel
    :param feature_name: Name of the feature, used as the prefix of the
        result keys (e.g. "CIV_FWHM")
    :type feature_name: str
    :param components: Components that make up the feature
    :type components: list(sculptor.component.FitComponent)
    :param params_values: Values of the model's variable parameters
    :type params_values: np.ndarray
    :param rest_frame_wavelength: Rest-frame wavelength of the feature
    :type rest_frame_wavelength: float
    :param cont_components: Continuum components, needed for the
        equivalent width
    :type cont_components: list(sculptor.component.FitComponent)
    :param redshift: Redshift of the source (required)
    :type redshift: float
    :param dispersion: Dispersion axis on which the models are evaluated
        (default: the model's spectrum, with its units)
    :type dispersion: np.ndarray
    :param emfeat_meas: Measurements to compute, any of "peak_fluxden",
        "peak_redsh", "EW", "FWHM", "flux", "lum" (the default,
        `emfeat_measures_default`) and "nonparam" (v05...v95 velocities,
        slower)
    :type emfeat_meas: list(str)
    :param disp_range: Observed-frame dispersion range to restrict the
        measurements to
    :type disp_range: [float, float]
    :param cosmology: Cosmology for luminosities (default: FlatLambdaCDM
        with H0=70, Om0=0.3, Tcmb0=2.725, with a warning)
    :type cosmology: astropy.cosmology.Cosmology
    :param fwhm_method: "spline" (default) or "sign", see
        `sculptor.measurements.get_fwhm`
    :type fwhm_method: str
    :param dispersion_unit: Unit of `dispersion`
    :type dispersion_unit: astropy.units.Unit
    :param fluxden_unit: Unit of the model flux density
    :type fluxden_unit: astropy.units.Unit
    :param a_v: If given, correct line and continuum for this V-band
        attenuation (magnitudes) before measuring
    :type a_v: float
    :param ext_law: Extinction law; only "calzetti00"
    :type ext_law: str
    :raises ValueError: if `redshift` is None or `ext_law` is unknown
    :return: Measurement name -> value (with units)
    :rtype: dict
    """

    if redshift is None:
        raise ValueError("No redshift specified for the analysis.")
    dispersion, dispersion_unit, fluxden_unit = _default_axis(
        model, dispersion, dispersion_unit, fluxden_unit
    )

    # Build the model flux
    model_fluxden = model.eval(
        dispersion, params_values, components=components, broaden_by_resolution=False
    )

    if a_v is not None:
        model_fluxden = _remove_extinction(
            model_fluxden, dispersion, redshift, a_v, ext_law
        )

    model_spec = sod.SpecOneD(
        dispersion=dispersion,
        fluxden=model_fluxden,
        dispersion_unit=dispersion_unit,
        fluxden_unit=fluxden_unit,
    )

    if cont_components is not None:
        cont_fluxden = model.eval(
            dispersion,
            params_values,
            components=cont_components,
            broaden_by_resolution=False,
        )

        if a_v is not None:
            cont_fluxden = _remove_extinction(
                cont_fluxden, dispersion, redshift, a_v, ext_law
            )

        cont_spec = sod.SpecOneD(
            dispersion=dispersion,
            fluxden=cont_fluxden,
            dispersion_unit=dispersion_unit,
            fluxden_unit=fluxden_unit,
        )

    # Start analysis
    result_dict = {}

    if emfeat_meas is None:
        emfeat_meas = emfeat_measures_default

    cosmology = _default_cosmology(cosmology)

    if "peak_fluxden" in emfeat_meas:
        # Calculate the peak flux density
        result_dict.update(
            {
                feature_name + "_peak_fluxden": np.max(model_spec.fluxden)
                * model_spec.fluxden_unit
            }
        )

    if "peak_redsh" in emfeat_meas:
        # Calculate the peak redshift
        result_dict.update(
            {
                feature_name + "_peak_redsh": meas.get_peak_redshift(
                    model_spec, rest_frame_wavelength
                )
            }
        )
    if "EW" in emfeat_meas and cont_components is not None:
        # Calculate the rest-frame equivalent width
        ew = meas.get_equivalent_width(
            cont_spec, model_spec, redshift=redshift, disp_range=disp_range
        )
        result_dict.update({feature_name + "_EW": ew})

    if "FWHM" in emfeat_meas:
        # Calculate the (composite) line model FWHM.
        fwhm = meas.get_fwhm(model_spec, method=fwhm_method, disp_range=disp_range)
        result_dict.update({feature_name + "_FWHM": fwhm})

        if np.isnan(fwhm):
            logger.warning(f"FWHM could not be calculated for feature {feature_name}")

    if "flux" in emfeat_meas:
        # Calculate the integrated line flux
        flux = meas.get_integrated_flux(model_spec, disp_range=disp_range)
        result_dict.update({feature_name + "_flux": flux})

    if "lum" in emfeat_meas:
        # Calculate the integrated line luminosity
        lum = meas.calc_integrated_luminosity(
            model_spec, redshift=redshift, cosmology=cosmology, disp_range=disp_range
        )
        result_dict.update({feature_name + "_lum": lum})

    if "nonparam" in emfeat_meas:
        # Calculate basic non-parametric line measurements
        # Note that this might be computationally expensive!

        v50, v05, v10, v90, v95, v_res_at_line, freq_v50, wave_v50, z_v50 = (
            meas.get_nonparametric_measurements(
                model_spec, rest_frame_wavelength, redshift, disp_range=disp_range
            )
        )

        result_dict.update({feature_name + "_v50": v50})
        result_dict.update({feature_name + "_v05": v05})
        result_dict.update({feature_name + "_v10": v10})
        result_dict.update({feature_name + "_v90": v90})
        result_dict.update({feature_name + "_v95": v95})
        result_dict.update({feature_name + "_vres_at_line": v_res_at_line})
        result_dict.update({feature_name + "_freq_v50": freq_v50})
        result_dict.update({feature_name + "_wave_v50": wave_v50})
        result_dict.update({feature_name + "_redsh_v50": z_v50})

    return result_dict


cont_measures_default = ["fluxden_avg", "Lwav", "appmag"]


def analyze_continuum(
    model,
    components,
    params_values,
    rest_frame_wavelengths,
    cosmology=None,
    redshift=None,
    dispersion_unit=None,
    fluxden_unit=None,
    dispersion=None,
    cont_meas=None,
    width=10,
):
    """Measure the continuum at a list of rest-frame wavelengths on the model
    built from one parameter vector (e.g. one posterior sample).

    At present this analysis assumes that the spectra are in the following
    units: flux density - erg/s/cm^2/AA, dispersion - AA.

    :param model: Fitted model
    :type model: sculptor.model.FitModel
    :param components: Continuum components
    :type components: list(sculptor.component.FitComponent)
    :param params_values: Values of the model's variable parameters
    :type params_values: np.ndarray
    :param rest_frame_wavelengths: Rest-frame wavelengths to measure at;
        the result keys start with the wavelength (e.g. "1450.0_Lwav")
    :type rest_frame_wavelengths: list(float)
    :param cosmology: Cosmology for luminosities (default: FlatLambdaCDM
        with H0=70, Om0=0.3, Tcmb0=2.725, with a warning)
    :type cosmology: astropy.cosmology.Cosmology
    :param redshift: Redshift of the source (required)
    :type redshift: float
    :param dispersion_unit: Unit of `dispersion`
    :type dispersion_unit: astropy.units.Unit
    :param fluxden_unit: Unit of the model flux density
    :type fluxden_unit: astropy.units.Unit
    :param dispersion: Dispersion axis on which the model is evaluated
        (default: the model's spectrum, with its units)
    :type dispersion: np.ndarray
    :param cont_meas: Measurements to compute, any of "fluxden_avg",
        "Lwav", "appmag" (default: all, `cont_measures_default`)
    :type cont_meas: list(str)
    :param width: Rest-frame width of the averaging window, in dispersion
        units
    :type width: float
    :raises ValueError: if `redshift` is None
    :return: Measurement name -> value (with units)
    :rtype: dict
    """

    if redshift is None:
        raise ValueError("No redshift specified for the analysis.")
    dispersion, dispersion_unit, fluxden_unit = _default_axis(
        model, dispersion, dispersion_unit, fluxden_unit
    )
    cosmology = _default_cosmology(cosmology)

    # Build the continuum model spectrum
    cont_flux = model.eval(
        dispersion, params_values, components=components, broaden_by_resolution=False
    )
    cont_spec = sod.SpecOneD(
        dispersion=dispersion,
        fluxden=cont_flux,
        dispersion_unit=dispersion_unit,
        fluxden_unit=fluxden_unit,
    )
    # Define unit for flux density
    fluxden_unit = cont_spec.fluxden_unit

    # Analyze the continuum
    if cont_meas is None:
        cont_meas = cont_measures_default

    # Start analysis
    result_dict = {}

    for wave in rest_frame_wavelengths:
        wave_name = str(wave)
        fluxden_avg = meas.get_average_fluxden(
            cont_spec, wave, redshift=redshift, width=width
        )

        if "fluxden_avg" in cont_meas:
            result_dict.update({wave_name + "_fluxden_avg": fluxden_avg})

        if "Lwav" in cont_meas:
            lwav = meas.calc_lwav_from_fwav(
                fluxden_avg, redshift=redshift, cosmology=cosmology
            )
            result_dict.update({wave_name + "_Lwav": lwav})

        if "appmag" in cont_meas:
            appmag = meas.calc_apparent_mag_from_fluxden(
                fluxden_avg, wave * (1.0 + redshift) * cont_spec.dispersion_unit
            )
            result_dict.update({wave_name + "_appmag": appmag})

    return result_dict


def get_line_snr(model, flat_chain, cont_component, redshift, line_dict, sigma=2):
    """Signal-to-noise ratio of emission lines for every posterior sample.

    For each sample the continuum model is subtracted from the observed
    spectrum and, for each line, the flux and the peak are compared with
    the uncertainties within a window of +-`sigma` Gaussian sigmas (from
    that sample's FWHM) around the line center, unless a fixed window is
    given.

    :param model: Fitted model (its `spec` is used)
    :type model: sculptor.model.FitModel
    :param flat_chain: Posterior samples, shape (n_samples, n_params)
    :type flat_chain: np.ndarray
    :param cont_component: Continuum component to subtract
    :type cont_component: sculptor.component.FitComponent
    :param redshift: Redshift, to place the rest-frame window
    :type redshift: float
    :param line_dict: Line name -> {"cen": rest-frame center,
        "fwhm": FWHM in km/s per sample, "l_min"/"l_max": fixed rest-frame
        window bounds or None}
    :type line_dict: dict
    :param sigma: Half-width of the window in Gaussian sigmas
    :type sigma: float
    :return: Integrated-flux SNR and peak SNR, each of shape
        (n_lines, n_samples)
    :rtype: tuple(np.ndarray, np.ndarray)
    """

    dispersion = model.spec.dispersion
    resid_spec = model.spec.copy()

    n_lines = len(line_dict)
    line_snr = np.zeros((n_lines, flat_chain.shape[0]))
    peak_snr = np.zeros((n_lines, flat_chain.shape[0]))

    for idx in tqdm(range(flat_chain.shape[0])):
        params = flat_chain[idx, :]

        cont_flux = model.eval(
            dispersion, params, components=[cont_component], broaden_by_resolution=False
        )

        resid_spec.fluxden = model.spec.fluxden - cont_flux

        for jdx, line in enumerate(line_dict):
            # l_min/l_max are per-sample windows derived from the sample's
            # FWHM, unless the caller pinned them to a fixed value. They
            # must NOT be cached back into line_dict: doing so would freeze
            # the window at whatever the first sample computed and reuse
            # that stale window for every subsequent MCMC sample.
            if line_dict[line]["l_min"] is not None:
                l_min = line_dict[line]["l_min"]
            else:
                l_min = line_dict[line]["cen"] * (
                    1 - line_dict[line]["fwhm"][idx] / 2.355 * sigma / c_km_s
                )
            if line_dict[line]["l_max"] is not None:
                l_max = line_dict[line]["l_max"]
            else:
                l_max = line_dict[line]["cen"] * (
                    1 + line_dict[line]["fwhm"][idx] / 2.355 * sigma / c_km_s
                )

            # Sum the flux in the line region
            trimmed_spec = resid_spec.trim_dispersion(
                [l_min * (1 + redshift), l_max * (1 + redshift)]
            )
            # Calculate the SNR for the line
            errsum = np.sqrt(np.sum(trimmed_spec.fluxden_err**2))
            line_snr[jdx, idx] = np.sum(trimmed_spec.fluxden) / errsum

            peak_snr[jdx, idx] = (
                np.max(trimmed_spec.fluxden)
                / trimmed_spec.fluxden_err[np.argmax(trimmed_spec.fluxden)]
            )

    return line_snr, peak_snr
