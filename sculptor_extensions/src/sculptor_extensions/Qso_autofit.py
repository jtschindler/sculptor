"""
The Sculptor Quasar Extension for fitting of the rest frame UV/optical type-I AGN spectrum.
This module defines models, masks and analysis routines specific for the analysis of type-I AGN.

The routine sets up continnum and emission line components using basic model functions defined in Sculptor.
Users can build customised models using functions defined in this module.

Complex models can be constructed by combining various functions defined below.
For example, we define a setup function to initialize a model to perform a combined fit for
the Hbeta and [OIII] lines consisting multiple Gaussian emission lines for broad and narrow components.
Furthermore, thr routine allows to extract the fit parameters and extract the AGN properties
such as black hole mass, continuum luminosity, Eddington ratio etc. from a given emission line parameters.

Ready-made variants of the base functions here (one per iron template,
emission line, ...) are in `sculptor_extensions.qso_presets`.

"""

import logging
import os
import string

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy import units as u

from sculptor import analysis as ana
from sculptor import measurements as meas
from sculptor import prior as scpri
from sculptor.logging_config import warn_once
from sculptor.utils import SIGMA_PERCENTILES, datadir
from sculptor_extensions import qso
from sculptor_extensions import qso_advanced_models as qam
from sculptor_extensions.component_setup import initialize_component

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------------------
# Required functions during fitting
# ------------------------------------------------------------------------------


DEFAULT_DISCARD = 2000


def _default_discard(discard):
    """`discard`, or the default 2000 (with a one-time warning, as these
    helpers used to default to 0)."""
    if discard is None:
        warn_once(
            logger,
            f"No `discard` given: discarding the first {DEFAULT_DISCARD} steps "
            "as burn-in (the default of these helpers used to be 0).",
        )
        return DEFAULT_DISCARD
    return discard


def get_1sigma_range(data):
    """
    Compute the median and 1-sigma (68% confidence interval) range from a dataset.

    :param data: data
    :type: np.array

    :return : median and 1-sigma upper and lower bounds
    :type: tuple
    """
    data = np.asarray(data)
    data = data[~np.isnan(data)]
    median = np.median(data)
    lower, upper = np.percentile(data, SIGMA_PERCENTILES)
    return median, lower, upper


def extract_median_params(prefix, median_param_dict):
    """
    Extract median values and 1-sigma uncertainties for all parameters of a given component prefix.
    This function requires a dictionary containing all parameters of all components of a particular model.
    For example, one can extract median values and 1-sigma uncertainties of all parameters of "Power_law" componet in a given model.

    :param prefix: parameter
    :type prefix: str
    :param median_param_dict: dictionary containing parameter names and their values from MCMC analysis
    :type median_param_dict: dict

    :return param_subset: dictionary in the format {param_name: value, param_name_sigma: 1-sigma}
    :type param_subset: dict
    """

    param_subset = {}
    for key, val in median_param_dict.items():
        if key.startswith(prefix):
            param_subset[key] = val
    return param_subset


def get_median_params(model, discard=None):
    """
    Calculate median value, 1-sigma uncertainties and upper,lower bounds for all parameters from the MCMC chain.

    :param model: model (with a sampler or a stored chain)
    :type model: sculptor.model.FitModel
    :param discard: number of steps to discard (default 2000, like
        `FitModel`; it used to be 0, so a warning is logged when it is not
        given)
    :type discard: int

    :return param_dict: Dictionary with keys as parameter names and values as:
        - param_name: median value
        - param_name_sigma: 1-sigma uncertainty (half the 68% interval)
        - param_name_upper: 84.13 percentile
        - param_name_lower: 15.87 percentile
    :type param_dict: dict
    """

    flat_chain = model.get_flat_chain(discard=_default_discard(discard))
    param_names = list(model.params_variable.keys())

    param_dict = {}

    for i, pname in enumerate(param_names):
        samples = flat_chain[:, i]
        median = np.median(samples)
        lower, upper = np.percentile(samples, SIGMA_PERCENTILES)

        sigma = 0.5 * (upper - lower)
        param_dict[pname] = median
        param_dict[f"{pname}_sigma"] = sigma
        param_dict[f"{pname}_upper"] = upper
        param_dict[f"{pname}_lower"] = lower

    return param_dict


def extract_median_params_from_summary(prefix, summary, median_cal=None):
    """
    Extract median values and 1-sigma uncertainties for all parameters for a given component prefix from a chainconsumer summary dictionary.

    This function retrieves median, upper, and lower bounds for MCMC parameters from a chainconsumer summary dictionary.
    If bounds are not available in the summary, it optionally falls back to a provided dictionary of precomputed median values.

    :param prefix : parameter prefix
    :type prefix: str
    :param summary: summary dictionary returned by chainconsumer (using chainconsumer.ChainConsumer.analysis.get_summary())
    :type summary: dict
    :param median_cal : Optional dictionary of precomputed median values and uncertainties (from get_median_params()). This is used as a fallback if bounds are missing in the summary.
    :type median_cal: dict

    :return result: Dictionary containing:
        - param_name: median value
        - param_name_sigma: 1-sigma uncertainty
        - param_name_upper: upper bound
        - param_name_lower: lower bound
    """

    result = {}
    chain = summary.get("MCMC chain", {})

    for key, bound in chain.items():
        key_clean = key.replace(" ", "_")
        if not key_clean.startswith(prefix):
            continue

        if bound.lower is not None and bound.upper is not None:
            median = bound.center
            upper = bound.upper
            lower = bound.lower
            sigma = max(abs(bound.upper - median), abs(median - bound.lower))
        elif median_cal is not None:
            median = median_cal.get(key_clean)
            sigma = median_cal.get(f"{key_clean}_sigma")
            upper = median_cal.get(f"{key_clean}_upper")
            lower = median_cal.get(f"{key_clean}_lower")
            if median is None or sigma is None:
                logger.warning(f"No calculated values found for {key_clean}")
                continue
        else:
            logger.warning(f"Missing bounds for {key_clean}.")
            continue

        result[key_clean] = median
        result[f"{key_clean}_sigma"] = sigma
        result[f"{key_clean}_upper"] = upper
        result[f"{key_clean}_lower"] = lower

    return result


def print_variable_parameters(model):
    """
    Log a summary of all the variable parameters used for MCMC: initial
    value and prior of each.

    :param model: model
    :type model: sculptor.model.FitModel
    :return: the summary text
    :rtype: str
    """

    model.get_params_to_sample()

    lines = ["Variable Parameters Summary:", "=" * 60]
    for name, param in model.params_variable.items():
        lines += [f"- {name}", f"    Initial value : {param.value}"]

        prior = param.prior
        if prior is None:
            lines.append("    Prior         : None")
        else:
            lines.append(f"    Prior type    : {prior.distribution}")
            if isinstance(prior, scpri.UniformPrior):
                lines += [
                    f"    Lower bound   : {prior.low}",
                    f"    Upper bound   : {prior.upp}",
                ]
            elif isinstance(prior, scpri.GaussianPrior):
                lines += [
                    f"    Mean          : {prior.mean}",
                    f"    Sigma         : {prior.sigma}",
                ]
            else:
                lines.append(f"    Prior details : {prior.kwargs}")
    lines.append("=" * 60)

    summary = "\n".join(lines)
    logger.info(summary)
    return summary


def plot_burnin(model, save_dir=".", plot_name="burnin_all_params.pdf", max_walkers=10):
    """
    Generate a plot to assess how much burn-in is necessary for a given MCMC run:
    the chain of every variable parameter against the step number, for up to
    `max_walkers` walkers.

    :param model: model with a sampler (the unflattened chain is needed)
    :type model: sculptor.model.FitModel
    :param save_dir: path to save the output
    :type save_dir: str
    :param plot_name: output filename
    :type plot_name: str
    :param max_walkers: number of walkers to plot (default 10; None for all)
    :type max_walkers: int
    :raises ValueError: if the model has no sampler (e.g. a loaded chain)
    """

    if model.sampler is None:
        raise ValueError(
            "plot_burnin needs the sampler (the per-walker chain); this model "
            "only has a flat chain."
        )
    samples = model.sampler.get_chain(discard=0, flat=False)

    param_names = list(model.params_variable.keys())
    n_params = len(param_names)
    n_steps, n_walkers, _ = samples.shape

    _fig, axes = plt.subplots(n_params, 1, figsize=(12, 2.0 * n_params), sharex=True)

    if n_params == 1:
        axes = [axes]

    if max_walkers is not None:
        n_walkers = min(n_walkers, max_walkers)
    for i, param_name in enumerate(param_names):
        ax = axes[i]
        for w in range(n_walkers):
            ax.plot(np.arange(n_steps), samples[:, w, i], alpha=0.5, lw=0.3)
        ax.set_ylabel(param_name)

    axes[-1].set_xlabel("MCMC Step")
    plt.tight_layout()

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, plot_name)
    plt.savefig(save_path)
    plt.close()
    logger.info(f"Burn-in plot saved to: {save_path}")


def get_sampled_values(model, param_name, discard=None):
    """
    Extract the MCMC samples of a specific parameter from a given model.

    :param model: model (with a sampler or a stored chain)
    :type model: sculptor.model.FitModel
    :param param_name: parameter
    :type param_name: str
    :param discard: number of steps as burn-in (default 2000, like
        `FitModel`; it used to be 0, so a warning is logged when it is not
        given)
    :type discard: int

    :return: 1D array of parameter values from the MCMC chain.
    :rtype: np.ndarray
    """

    chain = model.get_flat_chain(discard=_default_discard(discard))
    param_names = list(model.params_variable.keys())

    if param_name not in param_names:
        raise ValueError(f"Parameter '{param_name}' not found in variable parameters.")

    param_idx = param_names.index(param_name)
    return chain[:, param_idx]


# ------------------------------------------------------------------------------
# Wavelength masks
# ------------------------------------------------------------------------------

# QSO continuum windows EUV (Vestergaard & Peterson 2006 and Shen et al 2011)
"""Description of the EUV continuum windows:
VP06: We fitted the rest-frame UV spectra with a power-law continuum
in nominally line-free windows typically in the wavelength
ranges 1265–1290, 1340–1375, 1425–1470, 1680–1705, and 1950–2050 Å.

Shen11: We fit the pseudo-continuum model to a set of continuum windows free of strong emission lines (except for FeII):
1350–1360 Å, 1445–1465 Å, 1700–1705 Å, 2155–2400 Å,
2480–2675 Å, 2925–3500 Å, 4200–4230 Å, 4435–4700 Å,
5100–5535 Å, 6000–6250 Å, and 6800–7000 Å.
"""

# QSO continuum windows from Vestergaard and Peterson 2006 and Shen et al. 2011 covering rest-frame UV and optical line-free regions
qso_cont_feII = {
    "name": "QSO Continuum+FeII",
    "rest_frame": True,
    "mask_ranges": [
        [1265, 1290],  # Vestergaard 2006
        [1350, 1360],
        [1445, 1465],
        [1690, 1705],  # modified from Shen11: [1700, 1705]
        [1950, 2050],  # Vestergaard 2006
        [2155, 2400],
        [2480, 2675],
        [2925, 3500],
        [4200, 4230],
        [4435, 4700],
        [5100, 5535],
        [6000, 6250],
        [6800, 7000],
    ],
}


# QSO continuum windows from Vestergaard & Peterson 2006 covering rest-frame UV region till 2050 Å
qso_cont_VP06 = {
    "name": "QSO Cont. VP06",
    "rest_frame": True,
    "mask_ranges": [
        [1265, 1290],
        [1340, 1375],
        [1425, 1470],
        [1680, 1705],
        [1950, 2050],
    ],
}

# QSO continuum windows from Shen et al 2011 covering line-free continuum+iron windows around particular emission lines
qso_contfe_CIV_Shen11 = {
    "name": "QSO Cont. CIV Shen11",
    "rest_frame": True,
    "mask_ranges": [
        [1340, 1375],
        [1445, 1465],
        [1700, 1705],
    ],  ##additional window [1340, 1375] in order to get the correct continuum luminosity at 1350 for BH masses
}

qso_contfe_MgII_Shen11 = {
    "name": "QSO Cont. MgII Shen11",
    "rest_frame": True,
    "mask_ranges": [[2200, 2700], [2900, 3090]],
}

qso_contfe_MgII_Xshooter = {
    "name": "QSO Cont. MgII Schindler2020",  # adapted from Schindler 2020 and Shen 2011 paper
    "rest_frame": True,
    "mask_ranges": [[2480, 2700], [2900, 3090]],
}

qso_contfe_HBeta_Shen11 = {
    "name": "QSO Cont. HBeta Shen11",
    "rest_frame": True,
    "mask_ranges": [[4435, 4700], [5100, 5535]],
}

qso_contfe_Halpha_Shen11 = {
    "name": "QSO Cont. Halpha Shen11",
    "rest_frame": True,
    "mask_ranges": [[6000, 6250], [6800, 7000]],
}

qso_cont_feII_opt = {
    "name": "QSO Continuum+FeII (optical)",
    "rest_frame": True,
    "mask_ranges": [
        [4435, 4700],
        [5100, 5535],
        [6000, 6250],
        [6800, 7000],
    ],
}

# ------------------------------------------------------------------------------

# This is the region where there is no iron template available if you are using Tsuzuki 2006 and BG92 templates for MgII and Hbeta fir regions together.
No_FeII_range = {
    "name": "No_FeII_range",
    "rest_frame": True,
    "mask_ranges": [[3500, 3685]],
}

# ------------------------------------------------------------------------------

# QSO emission line windows taken from Shen et al 2011 (MgII, CIV, Hbeta, Halpha) and Shen et al 2019 (Lya, CIII], CIV, SiIV).
# One can add other emission lines that are not included here while fitting.

Lya_range = {"name": "Lya_range", "rest_frame": True, "mask_ranges": [[1150, 1290]]}

SiIV_range = {"name": "SiIV_range", "rest_frame": True, "mask_ranges": [[1290, 1450]]}

CIV_range = {"name": "CIV_range", "rest_frame": True, "mask_ranges": [[1465, 1600]]}

CIII_range = {
    "name": "CIII_range",
    "rest_frame": True,
    "mask_ranges": [[1700, 1970]],
}

MgII_range = {"name": "MgII_range", "rest_frame": True, "mask_ranges": [[2700, 2900]]}

Hbeta_range = {
    "name": "Hbeta_range",
    "rest_frame": True,
    "mask_ranges": [[4700, 5100]],
}

Halpha_range = {
    "name": "Halpha_range",
    "rest_frame": True,
    "mask_ranges": [[6400, 6800]],
}

# ------------------------------------------------------------------------------

# Ground-based near-infrared observations are highly affected by the atmospheric aborptions in ceritain regions.
# We define two of the two affected windows that fall between
# J-H band and H-K bands. During fitting, these regions would be masked out.
# Users can modify the windows based on their data.

Telluric_ranges = {
    "name": "Telluric_ranges",
    "rest_frame": False,
    "mask_ranges": [
        [13450, 14300],
        [18000, 19500],
    ],
}

# ------------------------------------------------------------------------------
# Initialize model components (continuum, iron and emission lines)
# ------------------------------------------------------------------------------


def initialize_power_law(prefix, redsh, powerlaw_dict=None):
    """
    Setup power law component with pre-defined parameters defined in the given dictionary.
    The function takes a power law defined with slope and amplitude which is normalized at 2500 Å.
    This model is defined for a spectral dispersion axis in Angstroem.

    :param prefix: model prefix
    :type prefix: string
    :param redsh: Unused (the redshift is set in `powerlaw_dict`); kept so
        existing calls keep working.
    :type redsh: float
    :param powerlaw_dict: Dictionary containing power law components "amp", "slope" and "redsh",
        see `component_setup.initialize_component`.
    :type powerlaw_dict: Dictionary

    :return comp: power law components
    :return pars: power law parameters
    """
    return initialize_component(
        prefix, ana.power_law_at2500, powerlaw_dict, dict_name="powerlaw_dict"
    )


def initialize_Fe(prefix, Fe_dispersion, Fe_fluxden, Fe_dict=None):
    """
    Setup an iron template using parameters defined in a dictionary. These parameters include "amp",
    "fwhm", "redsh" and "intr_fwhm" (see `component_setup.initialize_component`);
    "templ_disp" and "templ_fluxden" are set from the template arrays.

    :param prefix: Model prefix
    :type prefix: string
    :param Fe_dispersion: dispersion range for a given iron template
    :type Fe_dispersion: array
    :param Fe_fluxden: flux values over the dispersion range for a given iron template
    :type Fe_fluxden: array
    :param Fe_dict: dictionary containing parameters to setup the iron template
    :type Fe_dict: dictionary

    :return: iron template component and parameters
    """
    fixed = {"templ_disp": Fe_dispersion, "templ_fluxden": Fe_fluxden}
    return initialize_component(
        prefix, ana.template_model, Fe_dict, fixed=fixed, dict_name="Fe_dict"
    )


def initialize_Fe_fast(prefix, Fe_dispersion, Fe_fluxden, Fe_dict=None):
    """
    Like `initialize_Fe`, but builds the component from
    `sculptor.analysis.TemplateModel`, a precomputed version of
    `template_model` that is ~10-20x faster per evaluation and gives the
    same model (see its docstring). The template arrays are stored in the
    model object instead of as constant parameters, so the component only
    has the parameters "redsh", "amp", "fwhm" and "intr_fwhm", and
    `Fe_dict` only needs those keys.

    Ported from the upstream `qso_auto_fit` branch (`Qso_autofit_2.py`).

    :param prefix: Model prefix
    :type prefix: string
    :param Fe_dispersion: Rest-frame dispersion of the iron template (Angstroem)
    :type Fe_dispersion: array
    :param Fe_fluxden: Flux density of the iron template
    :type Fe_fluxden: array
    :param Fe_dict: dictionary containing parameters to setup the iron template
    :type Fe_dict: dictionary
    :return: iron template component and parameters
    """
    return initialize_component(
        prefix,
        ana.TemplateModel(Fe_dispersion, Fe_fluxden),
        Fe_dict,
        dict_name="Fe_dict",
    )


def initialize_balmer_without_hol(prefix, balmer_dict=None):
    """
    Initialize Balmer continuum component without higher order lines.
    It constrain Balmer flux contribution till the Balmer edge (3646 Å) and drops to 0 showing sharp edge at the Balmer edge.
    This function requires a dictionary that contains necessary parameters to set up
    a Balmer template such as "amp_be", "redsh", "Te", "tau_be", "lambda_be"
    (see `component_setup.initialize_component`).

    :param prefix: Model prefix
    :type prefix: string
    :param balmer_dict: dictionary containing parameters to setup the Balmer continuum model
    :type balmer_dict: dictionary

    :return comp: Balmer continuuum without HOL component
    :return pars: Balmer continuuum without HOL parameters
    """
    return initialize_component(
        prefix, qam.fast_balmer_continuum_model, balmer_dict, dict_name="balmer_dict"
    )


def initialize_balmer(prefix, model=None, balmer_dict=None):
    """
    Initialize Balmer continuum including the higher order balmer lines that
    blend as a pseudo-continuum (Storey & Hummer 1995, case B, 7 <= n <= 50).

    Builds the component from
    `qso_advanced_models.BalmerContinuumWithHOL` (via
    `qso_advanced_models.initialize_balmer_with_hol`). The previous model,
    `Balmer_continuum_with_hol`, was removed (see
    `deprecated/sculptor_extensions/src/sculptor_extensions/qso_autofit_balmer_hol.py`);
    its output was normalized and scaled differently, so `amp_be` values
    and priors from older setups need rescaling.

    `balmer_dict` must contain "redsh", "amp_be", "Te", "tau_be", "lambda_be"
    and "fwhm"; each entry has "value", "vary" and "prior", or "parameter"
    to map it onto an existing (global) parameter.

    :param prefix: Model prefix
    :type prefix: string
    :param model: Unused; kept so existing calls keep working.
    :param balmer_dict: dictionary containing parameters to setup the Balmer
        continuum model
    :type balmer_dict: dictionary
    :return comp: Balmer continuum with HOL component
    :return pars: Balmer continuum with HOL parameters
    """
    return qam.initialize_balmer_with_hol(prefix, balmer_dict)


# ------------------------------------------------------------------------------
# Initialize emission line components
# ------------------------------------------------------------------------------


def initialize_gaussian_line_model(prefix, line_dict, tied_doublet=True):
    """
    Setup a gaussian model for a given emission line feature in the spectrum.
    The central wavelength of the Gaussian line model is determined by the
    central wavelength cen and the redshift, z. These parameters are degenerate
    in a line fit and it is adviseable to fix one of them (to predetermined
    values e.g., the redshift or the central wavelength).

    The width of the line is set by the FWHM in km/s. The model function is
    chosen from the prefix: [OIII], [NII] and [SII] doublets for prefixes
    containing "OIII", "NII", "SII", a single Gaussian otherwise.

    :param prefix: Model prefix
    :type prefix: string
    :param line_dict: Dictionary containing all the emission line parameters "flux", "fwhm_km_s","cen", "redsh", and "vel_shift".
        Each entry is set up as in `component_setup.initialize_component`.
        Depending upon the lines in consideration, separate components such as narrow, broad and blue-shifted can be provided.
    :type line_dict: dict
    :param tied_doublet: For [NII] and [SII] (prefix containing "NII" /
        "SII"): if True (default), both lines share one FWHM and the flux
        ratio is a parameter (`line_model_gaussian_NII_doublet` /
        `line_model_gaussian_SII_doublet`; `line_dict` keys "redsh", "flux",
        "fwhm_km_s", "fluxratio"). If False, each line has its own flux and
        FWHM (`qso.line_model_gaussian_nii_doublet` /
        `qso.line_model_gaussian_sii_doublet`; keys "redsh", "flux_a",
        "flux_b", "fwhm_km_s_a", "fwhm_km_s_b"). Ignored for other lines.
    :type tied_doublet: bool
    :return comps: Gaussian line components
    :return params: Gaussian line parameters
    """

    if "OIII" in prefix:
        function = qso.line_model_gaussian_oiii_doublet
    elif "NII" in prefix:
        function = (
            line_model_gaussian_NII_doublet
            if tied_doublet
            else qso.line_model_gaussian_nii_doublet
        )
    elif "SII" in prefix:
        function = (
            line_model_gaussian_SII_doublet
            if tied_doublet
            else qso.line_model_gaussian_sii_doublet
        )
    else:
        function = ana.line_model_gaussian
    return initialize_component(prefix, function, line_dict, dict_name="line_dict")


def line_model_gaussian_NII_doublet(x, redsh, flux, fwhm_km_s, fluxratio=2.96):
    """Doublet line model for the [NII] lines at 6549.85 A and 6585.28 A.

    This model ties the redshift of the forbidden transitions of [NII] at
    6549.85 A and 6585.28 A together. Line widths are tied torgether for both the components and
    the relative flux ratio of the two components is fixed to 2.96.
    Same as `qso.line_model_gaussian_nii_doublet` with one FWHM and
    `flux_b = fluxratio * flux`.

    :param x: Dispersion of the continuum model
    :type x: np.ndarray
    :param redsh: Redshift
    :type redsh: float
    :param flux: Amplitude of the [NII] Gaussian component at 6549.85
    :type flux: float
    :param fwhm_km_s: FWHM of both the components in km/s
    :type fwhm_km_s: float
    :param fluxratio: Flux ratio of the 6585.28 A to the 6549.85 A line
    :type fluxratio: float
    :return: Gaussian doublet line model
    :rtype: np.ndarray
    """
    return qso.line_model_gaussian_nii_doublet(
        x, redsh, flux, fluxratio * flux, fwhm_km_s, fwhm_km_s
    )


def line_model_gaussian_SII_doublet(x, redsh, flux, fwhm_km_s, fluxratio=1.0):
    """Doublet line model for the [SII] lines at 6718.29 A and 6732.67 A.

    This model ties the redshift of the forbidden transitions of [SII] at
    6718.29 A and 6732.67 A together. Line widths are tied torgether for both the components and
    the relative flux ratio of the two components is fixed to 1.0. Literature studies show that
    the flux intensity ratio varies as it is sensitive to NLR density.
    Same as `qso.line_model_gaussian_sii_doublet` with one FWHM and
    `flux_b = fluxratio * flux`.

    :param x: Dispersion of the continuum model
    :type x: np.ndarray
    :param redsh: Redshift
    :type redsh: float
    :param flux: Amplitude of the [SII] Gaussian component at 6718.29
    :type flux: float
    :param fwhm_km_s: FWHM of both the components in km/s
    :type fwhm_km_s: float
    :param fluxratio: Flux ratio of the 6732.67 A to the 6718.29 A line
    :type fluxratio: float
    :return: Gaussian doublet line model
    :rtype: np.ndarray
    """
    return qso.line_model_gaussian_sii_doublet(
        x, redsh, flux, fluxratio * flux, fwhm_km_s, fwhm_km_s
    )


def add_line_components(model, line_dict, line, kind, n=None):
    """
    Add Gaussian components of one kind (e.g. "narrow", "broad") of an
    emission line to `model`. This is the base of the `setup_<line>_lines`
    functions in `sculptor_extensions.qso_presets`.

    The components use the settings in ``line_dict[line][kind]["params"]``
    (see `initialize_gaussian_line_model`) and are named "<line>_<kind>"
    if `n` is None (one component), or "<line>_<kind>_a",
    "<line>_<kind>_b", ... for `n` components.

    :param model: model to add the components and parameters to
    :type model: sculptor.model.FitModel
    :param line_dict: emission line dictionary
    :type line_dict: dict
    :param line: line name, e.g. "MgII"
    :type line: str
    :param kind: component kind, e.g. "narrow", "broad", "broadshifted"
        (a line dictionary with the old key "broad_shifted" is still read,
        with a `DeprecationWarning`)
    :type kind: str
    :param n: number of lettered components, or None for a single
        unlettered one
    :type n: int or None
    """
    if n is None:
        names = [f"{line}_{kind}"]
    else:
        names = [f"{line}_{kind}_{string.ascii_lowercase[i]}" for i in range(n)]

    settings = line_dict[line]
    if (
        names
        and kind == "broadshifted"
        and kind not in settings
        and "broad_shifted" in settings
    ):
        import warnings

        warnings.warn(
            f'line_dict["{line}"]["broad_shifted"] is deprecated; '
            'use the key "broadshifted"',
            DeprecationWarning,
            stacklevel=3,
        )
        kind = "broad_shifted"

    for name in names:
        comp, params = initialize_gaussian_line_model(name, settings[kind]["params"])
        model.add_component(comp)
        model.add_parameters(params)


# ------------------------------------------------------------------------------
# Iron templates
# ------------------------------------------------------------------------------

# Iron is fit in the same windows as the continuum/power law. The
# `setup_iron_template_<name>` variants for the packaged templates are in
# `sculptor_extensions.qso_presets`.


def setup_iron_template(
    prefix, template_file, disp_range=None, fe_dict=None, fast=True, fluxden_scale=1e17
):
    """
    Setup an iron template component from a file in sculptor's
    `data/iron_templates/`. This is the base of the
    `setup_iron_template_<name>` functions in
    `sculptor_extensions.qso_presets`, which fix the file and the default
    wavelength range for each template.

    :param prefix: component prefix
    :type prefix: str
    :param template_file: file name in `data/iron_templates/`, two columns:
        rest-frame wavelength (Å) and flux density
    :type template_file: str
    :param disp_range: [min, max] rest-frame range (Å, inclusive) to
        restrict the template to; None uses the full template
    :type disp_range: list
    :param fe_dict: Dictionary containing the parameters for iron template which are amplitude, FWHM,
                    redshift and intrinsic FWHM. The dictionary should
                    contain each parameter value, the priors and boolean stating whether to vary the parameter.
    :type fe_dict: dict
    :param fast: Use the faster `initialize_Fe_fast` (`TemplateModel`)
        instead of `initialize_Fe` (`template_model`); same model.
        Default True
    :type fast: bool
    :param fluxden_scale: factor applied to the template flux density
        (default 1e17, for the templates stored in erg/s/cm2/Å)
    :type fluxden_scale: float
    :return: model component and parameters for the iron template
    :rtype: sculptor.component.FitComponent, dict
    """
    template = np.loadtxt(datadir + "iron_templates/" + template_file)
    Fe_dispersion = template[:, 0]
    Fe_fluxden = template[:, 1] * fluxden_scale

    if disp_range is not None:
        disp_mask = (Fe_dispersion >= disp_range[0]) & (Fe_dispersion <= disp_range[1])
        Fe_dispersion = Fe_dispersion[disp_mask]
        Fe_fluxden = Fe_fluxden[disp_mask]

    init = initialize_Fe_fast if fast else initialize_Fe
    return init(prefix, Fe_dispersion, Fe_fluxden, fe_dict)


# ------------------------------------------------------------------------------
# Necessary calculations for fitting
# ------------------------------------------------------------------------------


def get_fwhm_int_flux(input_spec, low, upp, redsh, **kwargs):
    """
    Calculate the FWHM (in km/s) and integrated flux (1e-17 erg/s/cm^2) for
    a given emission line from the spectrum over a specified dispersion range.
    The FWHM is measured on a smoothing spline of the spectrum (smoothing
    from the flux density errors) with a root finding algorithm
    (`sculptor.measurements.get_fwhm`, method "spline").

    To calculate integrated flux over a dispersion range, the standard
    numpy.trapz function is used for the integration.

    :param input_spec: input spectrum
    :type input_spec: speconed.speconed.SpecOneD
    :param low: lower limit of the dispersion range
    :param upp: upper limit of the dispersion range
    :param redsh: redshift of the source
    :return: FWHM (km/s, NaN if it cannot be measured) and integrated flux
        (1e-17 erg/s/cm^2) of the spectral feature, as plain floats for the
        fit parameters
    :rtype: float, float
    """
    disp_range = [low * (1 + redsh), upp * (1 + redsh)]
    # Spline smoothing from the noise (scipy: s ~ number of points x error
    # variance), so the FWHM does not depend on the flux scale; s=0 (an
    # interpolating spline) for a spectrum without errors. A fixed s=100 used
    # to broaden lines with fluxes of order unity (+25% FWHM) and smooth away
    # lines in plain cgs units.
    err = input_spec.trim_dispersion(disp_range, inplace=False).fluxden_err
    s = float(np.nansum(err**2)) if err is not None else 0.0
    fwhm = meas.get_fwhm(input_spec, disp_range, s=s, method="spline")
    fwhm = fwhm.to_value(qso.KM_S)

    int_flux = meas.get_integrated_flux(input_spec, disp_range).to_value(
        1e-17 * u.erg / u.s / u.cm**2
    )

    return fwhm, int_flux


def create_emission_line_dict(input_spec, redsh, line_dict):
    """
    Create an updated emission line dictionary that containing the parameters that will further be used
    to fit an emission line. The fuction calculates the integrated flux and FWHM for a given emission line and
    updates the line priors based on these calculations.
    The below funcion currently creates a dictionary that has MgII, Hbeta, [OIII], CIII], CIV, SiIV, Halpha and Lya parameters.
    The same approach can be extended for other emission lines as well.

    :param input_spec: input spectrum
    :type input_spec: speconed.speconed.SpecOneD
    :param redsh: redshift of the source
    :type redsh: float
    :param line_dict: emission line dictionary containing the parameters required to fit each of the mentioned emission lines.
    These parameters include flux, redshift, central_wavelength, FWHM and velocity shift.
    :type line_dict: dict

    :return line_dict: an update line dictionary
    :rtype: dict
    """
    line_range = {
        "Lya": [1150, 1290],
        "SiIV": [1290, 1450],
        "CIV": [1500, 1600],
        "CIII": [1700, 1970],
        "MgII": [2760, 2850],
        "Hbeta": [4800, 4930],
        "OIII": [4990, 5025],
        "Halpha": [6500, 6600],
        "SII": [6700, 6800],
    }

    for line, range in line_range.items():
        if line not in line_dict:
            continue
        fwhm, int_flux = get_fwhm_int_flux(input_spec, range[0], range[1], redsh)

        for component in line_dict[line]:
            params = line_dict[line][component]["params"]
            if "flux" in params:
                params["flux"]["value"] = int_flux
                if "prior_upp_factor" in params["flux"]:
                    factor = params["flux"]["prior_upp_factor"]
                    params["flux"]["prior"].upp = int_flux * factor

            if "fwhm_km_s" in params and not np.isnan(fwhm):
                params["fwhm_km_s"]["value"] = fwhm
                if "prior_upp_factor" in params["fwhm_km_s"]:
                    factor = params["fwhm_km_s"]["prior_upp_factor"]
                    params["fwhm_km_s"]["prior"].upp = fwhm * factor

    return line_dict


# ------------------------------------------------------------------------------
# Analysis routines
# ------------------------------------------------------------------------------


def get_line_properties(
    df, redshift, param_name_dict, line, cont_wav, bh_mass_relations
):
    """
    Calculate AGN properties from the measurements of one broad emission
    line: single-epoch virial BH masses, the bolometric luminosity, the
    Eddington luminosity and ratio, and the line velocity shift with respect
    to `redshift`. This is the base of the `get_line_prop_<line>` functions
    in `sculptor_extensions.qso_presets`.

    The columns of `df` are plain numbers in fixed units (below); they are
    turned into astropy Quantities for the calculations. For every row and
    each of the "_median", "_low", "_upp" columns, these columns are added
    (`<wav>` is `cont_wav` in Å, `<tag>` a key of `bh_mass_relations`):

    - "<line>_BH_mass_<wav><tag>" (M_sun)
    - "L_bol_<wav>" (erg/s, Shen et al. 2011 bolometric correction)
    - "Eddington_ratio_<wav><tag>", "Edd_lum_<line>_<wav><tag>" (erg/s)
    - "delta_v_<line>" (km/s)

    Rows with a missing or NaN FWHM or luminosity are skipped for the BH
    mass and luminosities; a missing redshift skips the velocity shift.

    :param df: input dataframe containing extracted emission line parameters
    :type df: pd.DataFrame
    :param redshift: reference (systemic) redshift of the source
    :type redshift: float
    :param param_name_dict: column name prefixes in `df`:
        - "fwhm": line FWHM in km/s (e.g. "MgII_fwhm")
        - "luminosity": continuum luminosity at `cont_wav` in erg/s/Å
          (e.g. "L3000")
        - "redshift": line redshift (e.g. "MgII_redsh")
    :type param_name_dict: dict
    :param line: line name used in the output columns, e.g. "MgII"
    :type line: str
    :param cont_wav: wavelength of the continuum luminosity
    :type cont_wav: astropy.units.Quantity
    :param bh_mass_relations: column tag -> BH-mass relation from
        `sculptor_extensions.qso`, called as
        ``relation(fwhm, cont_lwav=..., cont_wav=...)``; e.g.
        ``{"_s11": qso.se_bhmass_mgii_s11_fwhm}``. May be empty.
    :type bh_mass_relations: dict
    :return: `df` with the added columns
    :rtype: pd.DataFrame
    """
    fwhm_col = param_name_dict.get("fwhm")
    lum_col = param_name_dict.get("luminosity")
    z_col = param_name_dict.get("redshift")
    wav = f"{cont_wav.to_value(u.AA):.0f}"

    for idx, row in df.iterrows():
        for key in ["_median", "_low", "_upp"]:
            # BH masses, bolometric luminosity and Eddington ratios
            if fwhm_col and lum_col:
                fwhm_key = fwhm_col + key
                lum_key = lum_col + key

                if fwhm_key in df.columns and lum_key in df.columns:
                    fwhm = np.float64(row[fwhm_key])
                    lum = np.float64(row[lum_key])

                    if not (np.isnan(fwhm) or np.isnan(lum)):
                        fwhm = fwhm * qso.KM_S
                        cont_lwav = lum * qso.LWAV_UNIT
                        l_bol = qso.calc_bolometric_luminosity(
                            cont_lwav=cont_lwav, cont_wav=cont_wav
                        )[0]
                        df.at[idx, f"L_bol_{wav}{key}"] = l_bol.to_value(qso.LUM_UNIT)

                        for tag, relation in bh_mass_relations.items():
                            bh_mass = relation(
                                fwhm, cont_lwav=cont_lwav, cont_wav=cont_wav
                            )[0]
                            edd_lum = qso.calc_eddington_luminosity(bh_mass)
                            edd_ratio = qso.calc_eddington_ratio(l_bol, bh_mass)
                            df.at[idx, f"{line}_BH_mass_{wav}{tag}{key}"] = (
                                bh_mass.to_value(u.Msun)
                            )
                            df.at[idx, f"Eddington_ratio_{wav}{tag}{key}"] = (
                                edd_ratio.to_value(u.dimensionless_unscaled)
                            )
                            df.at[idx, f"Edd_lum_{line}_{wav}{tag}{key}"] = (
                                edd_lum.to_value(qso.LUM_UNIT)
                            )
                    else:
                        logger.info(
                            f"NaN in '{fwhm_key}' or '{lum_key}' at row {idx} — "
                            "skipping BH mass & L_bol."
                        )
                else:
                    logger.info(
                        f"Missing columns '{fwhm_key}' or '{lum_key}' — "
                        "skipping BH mass & L_bol."
                    )
            else:
                logger.info(
                    "'fwhm' or 'luminosity' not in param_name_dict — "
                    "skipping BH mass & L_bol."
                )

            # Velocity shift
            if z_col:
                z_key = z_col + key
                if z_key in df.columns:
                    try:
                        delta_v = qso.calc_velocity_shifts(
                            row[z_key], z_sys=redshift, relativistic=True
                        )
                        df.at[idx, f"delta_v_{line}{key}"] = delta_v.to_value(qso.KM_S)
                    except Exception as e:  # noqa: BLE001 -- best-effort per-row batch calc, log & continue
                        logger.warning(f"Velocity shift calc failed at row {idx}: {e}")
                else:
                    logger.info(f"Column '{z_key}' missing — skipping velocity shift.")
            else:
                logger.info(
                    "'redshift' not in param_name_dict — skipping velocity shift."
                )

    return df


def add_line_snr_stats(
    df,
    model,
    flat_chain,
    redshift,
    fwhm,
    comp_pl,
    source,
    save_dir,
    line_name,
    line_center,
):
    """
    Calculate the signal-to-noise for a given emission line. The function extracts line fluxes by subtracting the contributions from the provided continuum components (PL/FeII/FeIII/Balmer) and/or
    contributions from any other emission line. It further calculates the peak line SNR and the total SNR within 2sigma range calculated from the line center provide.

    :param df: input dataframe containing extracted emission line parameters
    :type df: pd.DataFrame
    :param model: fit model
    :type model: sculptor.model.FitModel
    :param flat_chain: flat chain from MCMC
    :type flat_chain: np.ndarray
    :param redshift: redshift of the source
    :type redshift: float
    :param fwhm: FWHM of the line in km/s, one value per sample of `flat_chain`
    :type fwhm: np.ndarray
    :param comp_pl: continuum component to be subtracted before SNR calculation
    :type comp_pl: sculptor.component.FitComponent
    :param source: name of the source, used in the file names
    :type source: str
    :param save_dir: path to save the per-sample SNRs
        (`<source>_<line_name>_line_snr.csv`, `..._peak_snr.csv`)
    :type save_dir: str
    :param line_name: emission line in consideration for SNR calculation
    :type line_name: str
    :param line_center: central wavelength of the emission line
    :type line_center: float

    :return df: `df` with the median and 1-sigma range of the line and peak
        SNR (the same values in every row)
    :rtype: pd.DataFrame
    """

    line_dict = {
        line_name: {"cen": line_center, "fwhm": fwhm, "l_min": None, "l_max": None}
    }
    line_snr, peak_snr = ana.get_line_snr(
        model=model,
        flat_chain=flat_chain,
        cont_component=comp_pl,
        redshift=redshift,
        line_dict=line_dict,
    )

    for snr, kind in ((line_snr, "line"), (peak_snr, "peak")):
        pd.DataFrame(snr.T, columns=list(line_dict)).to_csv(
            os.path.join(save_dir, f"{source}_{line_name}_{kind}_snr.csv"),
            index=False,
        )
        med, low, upp = get_1sigma_range(snr[0])
        df[f"{line_name}_{kind}_snr_median"] = med
        df[f"{line_name}_{kind}_snr_low"] = low
        df[f"{line_name}_{kind}_snr_upp"] = upp

    return df


# ------------------------------------------------------------------------------
# Names moved to `sculptor_extensions.qso_presets`
# ------------------------------------------------------------------------------

_MOVED_TO_PRESETS = (
    "setup_iron_template_VW01_UV",
    "setup_iron_template_V01_FUV",
    "setup_iron_template_V01_CIV",
    "setup_iron_template_VW01_UV_MgII",
    "setup_iron_template_VW01_Fe3",
    "setup_iron_template_BG92_OPT",
    "setup_iron_template_BG92_OPT_Hbeta",
    "setup_iron_template_Tsuz_UV",
    "setup_iron_template_Tsuz_OPT",
    "setup_iron_template_Park22_OPT_Hbeta",
    "setup_Halpha_NII_lines",
    "setup_MgII_lines",
    "setup_CIV_lines",
    "setup_hb_o3_lines",
    "setup_Hgamma_lines",
    "setup_OI_lines",
    "get_line_prop_MgII",
    "get_civ_line_prop",
    "get_line_prop_Hbeta_OIII",
)


# British spellings used before 2026-10-20
RENAMED = {
    f"initialise_{n}": f"initialize_{n}"
    for n in (
        "power_law",
        "Fe",
        "Fe_fast",
        "balmer_without_hol",
        "balmer",
        "gaussian_line_model",
    )
}


def __getattr__(name):
    """Forward the names moved to `sculptor_extensions.qso_presets` and the
    renamed functions (`RENAMED`), with a `DeprecationWarning`.

    :param name: attribute name
    :type name: str
    :return: the function
    :rtype: callable
    :raises AttributeError: for any other name
    """
    if name in RENAMED:
        import warnings

        warnings.warn(
            f"`Qso_autofit.{name}` was renamed to `{RENAMED[name]}`",
            DeprecationWarning,
            stacklevel=2,
        )
        return globals()[RENAMED[name]]
    if name in _MOVED_TO_PRESETS:
        import warnings

        from sculptor_extensions import qso_presets

        new_name = qso_presets.RENAMED.get(name, name)
        warnings.warn(
            f"`Qso_autofit.{name}` moved to "
            f"`sculptor_extensions.qso_presets.{new_name}`",
            DeprecationWarning,
            stacklevel=2,
        )
        return getattr(qso_presets, new_name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
