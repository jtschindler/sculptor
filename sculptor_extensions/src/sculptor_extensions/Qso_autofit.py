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

"""

import os
import string
import numpy as np
import pandas as pd
import scipy as sp

from sculptor import model as scmod
from sculptor import prior as scpri
from sculptor import component as sccomp
from sculptor import analysis as ana
from sculptor_extensions import qso
from speconed import speconed as sod
from sculptor.dep_v1 import specanalysis as scana

from astropy import units as u
from astropy.constants import c
from astropy import constants as const
from astropy.modeling.models import BlackBody
from astropy.convolution import convolve, Gaussian1DKernel

from tqdm import tqdm
import matplotlib.pyplot as plt
from sculptor import colors as tolc

cmap = tolc.tol_cmap(colormap="rainbow_PuRd")
from importlib import resources

datadir = str(resources.files('sculptor') / 'data') + '/'
c_kms = c.to("km/s").value
c_AngstroemPerSecond = c.to(u.AA / u.s).value


# ------------------------------------------------------------------------------
# Required functions during fitting
# ------------------------------------------------------------------------------


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
    lower, upper = np.percentile(data, [15.9, 84.1])
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


def get_median_params(model, discard=0):
    """
    Calculate median value, 1-sigma uncertainties and upper,lower bounds for all parameters from the MCMC chain.

    :param model: model
    :param discard: number of steps to discard

    :return param_dict: Dictionary with keys as parameter names and values as:
        - param_name: median value
        - param_name_sigma: 1-sigma uncertainty (half the 68% interval)
        - param_name_upper: 84.1 percentile
        - param_name_lower: 15.9 percentile
    :type param_dict: dict
    """
    import numpy as np

    flat_chain = model.sampler.get_chain(discard=discard, thin=1, flat=True)
    param_names = list(model.params_variable.keys())

    param_dict = {}

    for i, pname in enumerate(param_names):
        samples = flat_chain[:, i]
        median = np.median(samples)
        lower = np.percentile(samples, 15.9)
        upper = np.percentile(samples, 84.1)

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
                print(f"[WARNING] No calculated values found for {key_clean}")
                continue
        else:
            print(f"[WARNING] Missing bounds for {key_clean}.")
            continue

        result[key_clean] = median
        result[f"{key_clean}_sigma"] = sigma
        result[f"{key_clean}_upper"] = upper
        result[f"{key_clean}_lower"] = lower

    return result


def print_variable_parameters(model):
    """
    Get a summary for all the variable parameters used for MCMC.

    :param model: model
    :type model: sculptor.model.FitModel
    """

    print("\n[INFO] Variable Parameters Summary:")
    print("=" * 60)

    model.get_params_to_sample()

    for name, param in model.params_variable.items():
        print(f"- {name}")
        print(f"    Initial value : {param.value}")

        if param.prior is None:
            print("    Prior         : None")
        else:
            prior = param.prior
            print(f"    Prior type    : {prior.distribution}")

            if isinstance(prior, scpri.UniformPrior):
                print(f"    Lower bound   : {prior.low}")
                print(f"    Upper bound   : {prior.upp}")

            elif isinstance(prior, scpri.GaussianPrior):
                print(f"    Mean          : {prior.mean}")
                print(f"    Sigma         : {prior.sigma}")

            else:
                print(f"    Prior details : {prior.kwargs}")

        print()

    print("=" * 60)


def plot_burnin(model, save_dir=".", plot_name="burnin_all_params.pdf"):
    """
    Generate a plot to assess how much burn-in is necessary for a given MCMC run.

    :param model: model
    :type model: sculptor.model.FitModel
    :param save_dir: path to save the output
    :type save_dir: str
    :param plot_name: output filename
    :type plot_name: str
    """

    samples = model.sampler.get_chain(discard=0, flat=False)

    param_names = list(model.params_variable.keys())
    n_params = len(param_names)
    n_steps, n_walkers, _ = samples.shape

    fig, axes = plt.subplots(n_params, 1, figsize=(12, 2.0 * n_params), sharex=True)

    if n_params == 1:
        axes = [axes]

    low = 0
    upp = n_steps

    n_walkers = 10
    for i, param_name in enumerate(param_names):
        ax = axes[i]
        for w in range(n_walkers):
            ax.plot(np.arange(low, upp), samples[low:upp, w, i], alpha=0.5, lw=0.3)
        ax.set_ylabel(param_name)

    axes[-1].set_xlabel("MCMC Step")
    plt.tight_layout()

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, plot_name)
    plt.savefig(save_path)
    plt.close()
    print(f"[INFO] Burn-in plot saved to: {save_path}")


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
    df = np.loadtxt(datadir, "wave_2.txt")
    transitions = df[:, 0]
    emiss = df[:, 1]
    flux_density = np.zeros_like(rest_wavelength)

    for line, em in zip(transitions, emiss):
        sigma = (fwhm / c_kms) * line / 2.35
        profile = np.exp(-0.5 * ((rest_wavelength - line) / sigma) ** 2)
        flux_density += em * profile / sigma

    # Normalizing hol to continuum at the Balmer edge
    edge_val = np.interp(lambda_be, rest_wavelength, flux_density)
    flux_density /= (1 / 0.3) * edge_val

    total_flux = (amp_be * (continuum + flux_density)) * 1e17
    return total_flux


def get_sampled_values(model, param_name, discard=0):
    """
    Extract the MCMC samples of a specific parameter from a given model.

    :param model: model
    :type model: sculptor.model.FitModel
    :param param_name: parameter
    :type param_name: str
    :param discard: number of steps as burn-in
    :type discard: int

    :return: 1D array of parameter values from the MCMC chain.
    :type : array
    """

    if model.sampler is None:
        raise RuntimeError("Sampler not found. Run model.run_emcee() first.")

    chain = model.sampler.get_chain(discard=discard, flat=True)
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
    "mask_ranges": [[1445, 1465], [1700, 1705]],
}

qso_contfe_MgII_Shen11 = {
    "name": "QSO Cont. MgII Shen11",
    "rest_frame": True,
    "mask_ranges": [[2200, 2700], [2900, 3090]],
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

CIV_range = {"name": "CIV_range", "rest_frame": True, "mask_ranges": [[1500, 1700]]}

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
# Initialise model components (continuum, iron and emission lines)
# ------------------------------------------------------------------------------


def initialise_power_law(prefix, redsh, powerlaw_dict=None, median_params=None):
    """
    Setup power law component with pre-defined parameters defined in the given dictionary.
    The function takes a power law defined with slope and amplitude which is normalized at 2500 Å.
    This model is defined for a spectral dispersion axis in Angstroem.

    :param prefix: model prefix
    :type prefix: string
    :param redsh: redshift of the source
    :type redsh: float
    :param powerlaw_dict: Dictionary containing power law components "amp", "slope" and "redshift".
    In this dictionary, each component should have "initial_value", "vary" and "prior"
    which provides information on initial value of the parameter, whether the parameter should be varied, and
    information on priors to be used to contrain the parameter, respectively.
    :type powerlaw_dict: Dictionary
    :param median_params: Dictionary containing median parameters from any previous fits to constrain the parameter values.
    :type median_params: Dictionary

    :return comp: power law components
    :return pars: power law parameters
    """
    comp = sccomp.FitComponent(prefix, ana.power_law_at2500)
    params = comp.create_params()

    if median_params:
        amp = median_params[f"{prefix}_amp"]
        slope = median_params[f"{prefix}_slope"]

        sigma_amp = median_params.get(f"{prefix}_amp_sigma")
        sigma_slope = median_params.get(f"{prefix}_slope_sigma")

        params[f"{prefix}_amp"].value = amp
        params[f"{prefix}_amp"].vary = True
        params[f"{prefix}_amp"].prior = scpri.GaussianPrior(
            f"{prefix}_amp", amp, sigma_amp
        )

        params[f"{prefix}_slope"].value = slope
        params[f"{prefix}_slope"].vary = True
        params[f"{prefix}_slope"].prior = scpri.GaussianPrior(
            f"{prefix}_slope", slope, sigma_slope
        )

        params[f"{prefix}_redsh"].value = redsh
        params[f"{prefix}_redsh"].vary = False

    elif powerlaw_dict:
        for key in params.keys():
            par = key.split(f"{prefix}_")[1]
            input = powerlaw_dict[par]

            if "parameter" in input:
                params.pop(key)
                comp.param_mapping.update({par: input["parameter"]})
            else:
                params[key].value = input["value"]
                params[key].vary = input["vary"]
                if input["vary"]:
                    params[key].prior = input["prior"]

    else:
        raise ValueError("Provide either `median_params` or `powerlaw_dict`")

    return comp, params


def initialise_Fe(prefix, Fe_dispersion, Fe_fluxden, Fe_dict=None, median_params=None):
    """
    Setup an iron template using parameters defined in a dictionary. These parameters include "amp",
    "fwhm", "redshift", "intr_fwhm", "templ_disp" and "templ_fluxden".
    In this dictionary, each component should have "initial_value", "vary" and "prior"
    which provides information on initial value of the parameter, whether the parameter should be varied, and
    prior to be used to contrain the parameter, respectively.
    :param prefix: Model prefix
    :type prefix: string
    :param Fe_dispersion: dispersion range for a given iron template
    :type Fe_dispersion: array
    :param Fe_fluxden: flux values over the dispersion range for a given iron template
    :type Fe_fluxden: array
    :param Fe_dict: dictionary containing parameters to setup the iron template
    :type Fe_dict: dictionary
    :param median_params: dictionary containing median parameters to setup the iron template
    :type median_params: dictionary

    :return: iron template component and parameters
    """
    comp = sccomp.FitComponent(prefix, ana.template_model)
    params = comp.create_params()

    if median_params:
        amp = median_params[f"{prefix}_amp"]
        fwhm = median_params[f"{prefix}_fwhm"]
        redsh = median_params[f"{prefix}_redsh"]

        sigma_amp = median_params.get(f"{prefix}_amp_sigma")
        upper_fwhm = median_params.get(f"{prefix}_fwhm_upper")
        lower_fwhm = median_params.get(f"{prefix}_fwhm_lower")
        sigma_redsh = median_params.get(f"{prefix}_redsh_sigma")

        params[f"{prefix}_amp"].value = amp
        params[f"{prefix}_amp"].vary = True
        params[f"{prefix}_amp"].prior = scpri.GaussianPrior(
            f"{prefix}_amp", amp, sigma_amp
        )

        params[f"{prefix}_fwhm"].value = fwhm
        params[f"{prefix}_fwhm"].vary = True
        params[f"{prefix}_fwhm"].prior = scpri.UniformPrior(
            f"{prefix}_fwhm", lower_fwhm, upper_fwhm
        )

        params[f"{prefix}_redsh"].value = redsh
        params[f"{prefix}_redsh"].vary = True
        params[f"{prefix}_redsh"].prior = scpri.GaussianPrior(
            f"{prefix}_redsh", redsh, sigma_redsh
        )

    if Fe_dict:
        for key in params.keys():
            par = key.split(f"{prefix}_")[1]
            if par in ["amp", "fwhm", "redsh"] and median_params:
                continue
            if par == "templ_disp":
                params[key].value = Fe_dispersion
                params[key].vary = False
                continue
            if par == "templ_fluxden":
                params[key].value = Fe_fluxden
                params[key].vary = False
                continue
            input = Fe_dict[par]

            if "parameter" in input:
                params.pop(key)
                comp.param_mapping.update({par: input["parameter"]})
            else:
                params[key].value = input["value"]
                params[key].vary = input["vary"]
                if input["vary"]:
                    params[key].prior = input["prior"]

    else:
        raise ValueError("Provide either `median_params` or `FeII_dict`")

    return comp, params


def initialise_balmer_without_hol(prefix, balmer_dict=None, median_params=None):
    """
    Initialise Balmer continuum component without higher order lines.
    It constrain Balmer flux contribution till the Balmer edge (3646 Å) and drops to 0 showing sharp edge at the Balmer edge.
    This function requires a dictionary that contains necessary parameters to set up
    a Balmer template such as "amp_be", "redshift", "Te", "tau_be", "lambda_be".
    In this dictionary, each component should have "initial_value", "vary" and "prior"
    which provides information on initial value of the parameter, whether the parameter should be varied, and
    prior to be used to contrain the parameter, respectively.

    :param prefix: Model prefix
    :type prefix: string
    :param balmer_dict: dictionary containing parameters to setup the Balmer continuum model
    :type Balmer_dict: dictionary
    :param median_params: dictionary containing median parameters to setup the Balmer continuum model from any previous fit.
    :type median_params: dictionary

    :return comp: Balmer continuuum without HOL component
    :return pars: Balmer continuuum without HOL parameters
    """
    comp = sccomp.FitComponent(prefix, qso.balmer_continuum_model)
    params = comp.create_params()

    if median_params:
        amp_be = median_params[f"{prefix}_amp_be"]
        sigma_amp = median_params.get(f"{prefix}_amp_sigma", 0.1)

        params[f"{prefix}_amp_be"].value = amp_be
        params[f"{prefix}_amp_be"].vary = True
        params[f"{prefix}_amp_be"].prior = scpri.GaussianPrior(
            f"{prefix}_amp_be", amp_be, sigma_amp
        )

    if balmer_dict:
        for key in params.keys():
            par = key.split(f"{prefix}_")[1]
            if par in ["amp_be"] and median_params:
                continue
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
        raise ValueError("Provide either `median_params` or `powerlaw_dict`")

    return comp, params


def initialise_balmer(prefix, model, balmer_dict=None, median_params=None):
    """
    Initialise Balmer continuum including the higher order balmer lines that blend as a pseudo-continuum.
    Following Storey and Hummer 1995, we consider higher order lines generatinf from transitions with (7<=n<=50). Higher order line emissivities were calculated
    from the database provided in under the case B approximation for Hydrogen, for a given temperature, electron density, and braodening Doppler velocity.
    A Balmer template is defined using "amp_be", "redshift", "Te", "Tau_be", "lambda_be" and "FWHM".
    A dictionary should be provided wherein each parameter needed for Balmer template has "initial_value", "vary" and "prior"
    which provides information on initial value of the parameter, whether the parameter should be varied, and the
    priors to be used to contrain the parameter, respectively.

    :param prefix: Model prefix
    :type prefix: string
    :param balmer_dict: dictionary containing parameters to setup the Balmer continuum model
    :type Balmer_dict: dictionary
    :param median_params: dictionary containing median parameters to setup the Balmer continuum model from any previous fit.
    :type median_params: dictionary

    :return comp: Balmer continuuum with HOL component
    return pars: Balmer continuuum with HOL parameters
    """
    comp = sccomp.FitComponent(prefix, Balmer_continuum_with_hol)
    params = comp.create_params()

    if median_params:
        amp_be = median_params[f"{prefix}_amp_be"]
        sigma_amp = median_params.get(f"{prefix}_amp_sigma")

        params[f"{prefix}_amp_be"].value = amp_be
        params[f"{prefix}_amp_be"].vary = True
        params[f"{prefix}_amp_be"].prior = scpri.GaussianPrior(
            f"{prefix}_amp_be", amp_be, sigma_amp
        )

    if balmer_dict:
        for key in params.keys():
            par = key.split(f"{prefix}_")[1]
            if par in ["amp_be"] and median_params:
                continue
            if par in ["fwhm"] and median_params:
                continue
            if par == "x":
                params[key].value = model.gpm_fit
                params[key].vary = False
                continue
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
        raise ValueError("Provide either `median_params` or `balmer_dict`")

    return comp, params


# ------------------------------------------------------------------------------
# Initialise emission line components
# ------------------------------------------------------------------------------


def initialise_gaussian_line_model(prefix, line_dict):
    """
    Setup a gaussian model for a given emission line feature in the spectrum.
    The central wavelength of the Gaussian line model is determined by the
    central wavelength cen and the redshift, z. These parameters are degenerate
    in a line fit and it is adviseable to fix one of them (to predetermined
    values e.g., the redshift or the central wavelength).

    The width of the line is set by the FWHM in km/s.

    :param prefix: Model prefix
    :type prefix: string
    :param line_dict: Dictionary containing all the emission line parameters "flux", "fwhm_km_s","cen", "redsh", and "vel_shift".
    Each component from this dictionary has "initial_value", "vary" and "prior"
    which provides information on initial value of the parameter, whether the parameter should be varied, and
    prior to be used to contrain the parameter, respectively.
    Depending upon the lines in consideration, separate components such as narrow, broad and blue-shifted can be provided.

    :return comps: Gaussian line components
    :return params: Gaussian line parameters
    """

    if "OIII" in prefix:
        comp = sccomp.FitComponent(prefix, qso.line_model_gaussian_oiii_doublet)
    elif "NII" in prefix:
        comp = sccomp.FitComponent(prefix, line_model_gaussian_NII_doublet)
    elif "SII" in prefix:
        comp = sccomp.FitComponent(prefix, line_model_gaussian_SII_doublet)
    else:
        comp = sccomp.FitComponent(prefix, ana.line_model_gaussian)
    params = comp.create_params()

    for key in list(params.keys()):
        par = key.split(f"{prefix}_")[1]
        input = line_dict[par]

        if "parameter" in input.keys():
            params.pop(key)
            comp.param_mapping.update({par: input["parameter"]})
        else:
            params[key].value = input["value"]
            params[key].vary = input["vary"]
            if input["vary"]:
                params[key].prior = input["prior"]

    return comp, params


def line_model_gaussian_NII_doublet(x, redsh, flux, fwhm_km_s, fluxratio=2.96):
    """Doublet line model for the [NII] lines at 6549.85 A and 6585.28 A.

    This model ties the redshift of the forbidden transitions of [NII] at
    6549.85 A and 6585.28 A together. Line widths are tied torgether for both the components and
    the relative flux ratio of the two components is fixed to 2.96.

    :param x: Dispersion of the continuum model
    :type x: np.ndarray
    :param redsh: Redshift
    :type redsh: float
    :param flux: Amplitude of the [NII] Gaussian component at 6549.85
    :type flux: float
    :param fwhm_km_s: FWHM of both the components in km/s
    :type fwhm_km_s: float
    :return: Gaussian doublet line model
    :rtype: np.ndarray
    """

    # Redshift central wavelengths
    cen_a = 6549.85 * (1 + redsh)
    cen_b = 6585.28 * (1 + redsh)

    # Calculate sigma from fwhm
    fwhm_a = fwhm_km_s / c_kms * cen_a
    fwhm_b = fwhm_km_s / c_kms * cen_b
    sigma_a = fwhm_a / np.sqrt(8 * np.log(2))
    sigma_b = fwhm_b / np.sqrt(8 * np.log(2))

    flux_a = flux
    flux_b = fluxratio * flux_a

    comp_a = (
        flux_a
        / (sigma_a * np.sqrt(2 * np.pi))
        * np.exp(-((x - cen_a) ** 2) / (2 * sigma_a**2))
    )

    comp_b = (
        flux_b
        / (sigma_b * np.sqrt(2 * np.pi))
        * np.exp(-((x - cen_b) ** 2) / (2 * sigma_b**2))
    )

    return comp_a + comp_b


def line_model_gaussian_SII_doublet(x, redsh, flux, fwhm_km_s, fluxratio=1.0):
    """Doublet line model for the [SII] lines at 6718.29 A and 6732.67 A.

    This model ties the redshift of the forbidden transitions of [NII] at
    6718.29 A and 6732.67 A together. Line widths are tied torgether for both the components and
    the relative flux ratio of the two components is fixed to 1.0. Literature studies show that
    the flux intensity ratio varies as it is sensitive to NLR density.

    :param x: Dispersion of the continuum model
    :type x: np.ndarray
    :param redsh: Redshift
    :type redsh: float
    :param flux: Amplitude of the [SII] Gaussian component at 6549.85
    :type flux: float
    :param fwhm_km_s: FWHM of both the components in km/s
    :type fwhm_km_s: float
    :return: Gaussian doublet line model
    :rtype: np.ndarray
    """

    # Redshift central wavelengths
    cen_a = 6718.29 * (1 + redsh)
    cen_b = 6732.67 * (1 + redsh)

    # Calculate sigma from fwhm
    fwhm_a = fwhm_km_s / c_kms * cen_a
    fwhm_b = fwhm_km_s / c_kms * cen_b
    sigma_a = fwhm_a / np.sqrt(8 * np.log(2))
    sigma_b = fwhm_b / np.sqrt(8 * np.log(2))

    flux_a = flux
    flux_b = fluxratio * flux_a

    comp_a = (
        flux_a
        / (sigma_a * np.sqrt(2 * np.pi))
        * np.exp(-((x - cen_a) ** 2) / (2 * sigma_a**2))
    )

    comp_b = (
        flux_b
        / (sigma_b * np.sqrt(2 * np.pi))
        * np.exp(-((x - cen_b) ** 2) / (2 * sigma_b**2))
    )

    return comp_a + comp_b


def setup_Halpha_NII_lines(spec, line_dict, narrow=True, Halpha_broad=2, model=None):
    """
    Setup Halpha+[NII] lines simultaneously at 6564.61 A, 6549.85 A and 6585.28 A.
    The function requires an emission line dictionary with prior information on narrow (Halpha, [NII]) and broad (Halpha) components for each line.

    The narrow component FWHM limits are set between 300-1000 km/s.
    Otherwise, the function calculates FWHM and integrated flux of Halpha and [NII] line regions separately which are further incorporated in determining the priors.
    By default, the model fits 1 narrow component for Halpha and each of the [NII] lines and 2 broad component for Halpha.
    The narrow components of Halpha and each [NII] are tied together as they are known to be originated from the same gas in the NLR.
    Therefore, all narrow components have the same redshift and FWHM.
    Users can specify the number of gaussians to fit depending upon the nature of their source.

    :param spec: input spectrum
    :type spec: sculptor.speconed.SpecOneD
    :param line_dict: dictionary containing emission line parameters that contains necessary parameters to fit the Halpha and [NII] emission lines.
    such as "flux", "redshift", "fwhm_km_s", "cen", "vel_shift".
    In this dictionary, each component should have "initial_value", "vary" and "prior" which provides information on initial value of the parameter, whether the parameter should be varied, and
    prior to be used to contrain the parameter, respectively.
    :type line_dict: dictionary
    :param narrow: fit narrow component for Halpha and [NII] lines
    :type narrow: Boolean
    :param Halpha_broad: number of Halpha broad components to fit (default 2)
    :type Halpha_broad: int

    :param model: Model to fit

    """
    if narrow:
        comp_Halpha_narrow, pars_Halpha_narrow = initialise_gaussian_line_model(
            "Halpha_narrow", line_dict["Halpha"]["narrow"]["params"]
        )
        model.add_component(comp_Halpha_narrow)
        model.add_parameters(pars_Halpha_narrow)

        comp_NII_narrow, pars_NII_narrow = initialise_gaussian_line_model(
            "NII_narrow", line_dict["NII"]["narrow"]["params"]
        )
        model.add_component(comp_NII_narrow)
        model.add_parameters(pars_NII_narrow)

    for i in range(Halpha_broad):
        comp_Halpha_broad, pars_Halpha_broad = initialise_gaussian_line_model(
            f"Halpha_broad_{string.ascii_lowercase[i]}",
            line_dict["Halpha"]["broad"]["params"],
        )
        model.add_component(comp_Halpha_broad)
        model.add_parameters(pars_Halpha_broad)


def setup_MgII_lines(spec, line_dict, narrow=False, MgII_broad=1, model=None):
    """
    Setup emission line model for MgII emission line (2798.75 Å) using multiple broad and narrow gaussian components.
    MgII line paramters are constarined using the priors set in the emission line dictionary provided.
    Generally, MgII lines are not well resolved and thus, not fit using two separate gaussians. Instead, a single or multi-gaussian
    components are used to capture the broader feature originating from BLR.

    :param spec: input spectrum
    :type spec: sculptor.speconed.SpecOneD
    :param line_dict: dictionary containing emission line parameters that contains necessary parameters to fit the MgII emission line
    such as "flux", "redshift", "fwhm_km_s", "cen", "vel_shift". In this dictionary, each component should have "initial_value", "vary" and "prior"
    which provides information on initial value of the parameter, whether the parameter should be varied, and
    prior to be used to contrain the parameter, respectively.
    It is recommended to fit the narrow component only when the signal-to-noise for the provided is considerably high.

    :type line_dict: dictionary
    :param narrow: fit narrow component for MgII line
    :type narrow: Boolean
    :param MgII_broad: number of broad components to fit
    :type MgII_broad: int
    :param model: Model to fit
    """
    if narrow:
        comp_MgII_narrow, pars_MgII_narrow = initialise_gaussian_line_model(
            "MgII_narrow", line_dict["MgII"]["narrow"]["params"]
        )
        model.add_component(comp_MgII_narrow)
        model.add_parameters(pars_MgII_narrow)

    for i in range(MgII_broad):
        comp_MgII_broad, pars_MgII_broad = initialise_gaussian_line_model(
            f"MgII_broad_{string.ascii_lowercase[i]}",
            line_dict["MgII"]["broad"]["params"],
        )
        model.add_component(comp_MgII_broad)
        model.add_parameters(pars_MgII_broad)


def setup_hb_o3_lines(
    spec,
    line_dict,
    narrow=True,
    Hbeta_broad=1,
    Hbeta_broad_shifted=0,
    OIII_wind=0,
    model=None,
):
    """
    Setup Hbeta+[OIII] lines simultaneously at 4862.30 A, 4960.30 A and 5008.24 A.
    The function requires an emission line dictionary with prior information on narrow (Hbeta, [OIII]), broad (Hbeta) and blue-shifted (Hbeta, [OIII]) components for each line.

    The narrow component FWHM limits are set between 300-1000 km/s. Otherwise, the function calculates FWHM and integrated flux of Hbeta and [OIII] line regions separately which are further incorporated in determining the priors.
    By default, the model fits 1 narrow component for Hbeta and each of the [OIII] lines and 1 broad component for Hbeta.
    The narrow components of Hbeta and each [OIII] are tied together as they are known to be originated from the same gas in the NLR. Therefore, all narrow components have the same redshift and FWHM.
    Users can specify the number of gaussians to fit depending upon the nature of their source.

    :param spec: Input spectrum
    :type spec: sculptor.speconed.SpecOneD
    :param line_dict: dictionary containing emission line parameters that contains necessary parameters to fit the Hbeta and [OIII] emission lines.
    such as "flux", "redshift", "fwhm_km_s", "cen", "vel_shift".
    In this dictionary, each component should have "initial_value", "vary" and "prior" which provides information on initial value of the parameter, whether the parameter should be varied, and
    prior to be used to contrain the parameter, respectively.
    :type line_dict: dictionary
    :param narrow: fit narrow component for Hbeta and [OIII] lines
    :type narrow: Boolean
    :param Hbeta_broad: number of Hbeta broad components to fit
    :type Hbeta_broad: int
    :param Hbeta_broad_shifted: number of velocity shifted Hbeta broad components to fit
    :type Hbeta_broad: int
    :param OIII_wind: number of [OIII] blue-shifted wind components to fit
    :type OIII_wind: int
    :param model: Model to fit
    """

    if narrow:
        comp_Hbeta_narrow, pars_Hbeta_narrow = initialise_gaussian_line_model(
            "Hbeta_narrow", line_dict["Hbeta"]["narrow"]["params"]
        )
        model.add_component(comp_Hbeta_narrow)
        model.add_parameters(pars_Hbeta_narrow)

        comp_OIII_narrow, pars_OIII_narrow = initialise_gaussian_line_model(
            "OIII_narrow", line_dict["OIII"]["narrow"]["params"]
        )
        model.add_component(comp_OIII_narrow)
        model.add_parameters(pars_OIII_narrow)

    for i in range(Hbeta_broad):
        comp_Hbeta_broad, pars_Hbeta_broad = initialise_gaussian_line_model(
            f"Hbeta_broad_{string.ascii_lowercase[i]}",
            line_dict["Hbeta"]["broad"]["params"],
        )
        model.add_component(comp_Hbeta_broad)
        model.add_parameters(pars_Hbeta_broad)

    for i in range(Hbeta_broad_shifted):
        comp_Hbeta_broad_shifted, pars_Hbeta_broad_shifted = (
            initialise_gaussian_line_model(
                f"Hbeta_broad_shifted_{string.ascii_lowercase[i]}",
                line_dict["Hbeta"]["broad_shifted"]["params"],
            )
        )
        model.add_component(comp_Hbeta_broad_shifted)
        model.add_parameters(pars_Hbeta_broad_shifted)

    for k in range(OIII_wind):
        comp_OIII_wind, pars_OIII_wind = initialise_gaussian_line_model(
            f"OIII_wind_{string.ascii_lowercase[k]}",
            line_dict["OIII"]["wind"]["params"],
        )
        model.add_component(comp_OIII_wind)
        model.add_parameters(pars_OIII_wind)


# ------------------------------------------------------------------------------
# Iron templates
# ------------------------------------------------------------------------------

"""
Iron is fit in the same windows as the continuum/powerlaw. We setup different iron templates here
that cover different wavelengths in rest-frame.
"""


def setup_iron_template_VW01_UV(prefix, fe_dict=None, median_params=None):
    """
    Setup the Vestergaard et al. 2001 iron template covering UV range between 1075 - 3089 Å.

    :param prefix: component prefix
    :param type: str
    :param fe_dict: Dictionary containing the parameters for iron template which are amplitude, FWWHM,
                    redshift, intrinsic_FWHM, templ_disp_unit_str and templ_fluxden_unit_str. The dictionary should
                    contain each parameter value, the priors and boolean stating whether to vary the parameter.
    :param fe_dict:
    :param median_params: Dictionary containing median parameters to setup the iron template. This dict contains the median values for parameters taken from the initial fit.
    :return comp: model components for the iron template
    :return pars: model parameters for the iron template

    Note: Flux values from the iron template are in units of 1e-17 erg/s/cm2
    """

    template = np.loadtxt(datadir + "iron_templates/" + "Fe_UVtemplt_A.asc")
    Fe_dispersion = template[:, 0]
    Fe_fluxden = template[:, 1] * 1.0e17

    comp, params = initialise_Fe(
        prefix, Fe_dispersion, Fe_fluxden, fe_dict, median_params
    )
    return comp, params


def setup_iron_template_VW01_UV_MgII(prefix, fe_dict=None, median_params=None):
    """
    Setup the Vestergaard et al. 2001 iron template covering region around MgII emission line within the range of 2200 - 3090 Å.

    :param prefix: component prefix
    :param type: str
    :param fe_dict: Dictionary containing the parameters for iron template which are amplitude, FWWHM,
                    redshift, intrinsic_FWHM, templ_disp_unit_str and templ_fluxden_unit_str. The dictionary should
                    contain each parameter value, the priors and boolean stating whether to vary the parameter.
    :param fe_dict:
    :param median_params: Dictionary containing median parameters to setup the iron template. This dict contains the median values for parameters taken from the initial fit.
    :return comp: model components for the iron template
    :return pars: model parameters for the iron template

    Note: Flux values from the iron template are in units of 1e-17 erg/s/cm2
    """
    MgII_disp = [2200, 3090]

    template = np.loadtxt(datadir + "iron_templates/" + "Fe_UVtemplt_A.asc")
    Fe_dispersion = template[:, 0]
    MgII_disp_mask = (Fe_dispersion >= MgII_disp[0]) & (Fe_dispersion <= MgII_disp[1])
    Fe_dispersion = Fe_dispersion[MgII_disp_mask]
    Fe_fluxden = template[:, 1][MgII_disp_mask] * 1.0e17

    comp, params = initialise_Fe(
        prefix, Fe_dispersion, Fe_fluxden, fe_dict, median_params
    )
    return comp, params


def setup_iron_template_BG92_OPT(prefix, fe_dict=None, median_params=None):
    """
    Setup the Boroson and Green et al. 1992 iron template covering the optical region between 3685 - 7844 Å.

    :param prefix: component prefix
    :param type: str
    :param fe_dict: Dictionary containing the parameters for iron template which are amplitude, FWWHM,
                    redshift, intrinsic_FWHM, templ_disp_unit_str and templ_fluxden_unit_str. The dictionary should
                    contain each parameter value, the priors and boolean stating whether to vary the parameter.
    :param fe_dict:
    :param median_params: Dictionary containing median parameters to setup the iron template. This dict contains the median values for parameters taken from the initial fit.
    :return comp: model components for the iron template
    :return pars: model parameters for the iron template

    Note: Flux values from the iron template are in units of 1e-17 erg/s/cm2
    """

    template = np.loadtxt(datadir + "iron_templates/" + "Fe_OPT_BR92_linear.txt")
    Fe_dispersion = template[:, 0]
    Fe_fluxden = template[:, 1] * 1.0e17

    comp, params = initialise_Fe(
        prefix, Fe_dispersion, Fe_fluxden, fe_dict, median_params
    )
    return comp, params


def setup_iron_template_Tsuz_UV(prefix, fe_dict=None, median_params=None):
    """
    Setup UV part of the Tsuzuki et al. 2006 template that coveres the region between 2200-3500 Å.

    :param prefix: component prefix
    :param type: str
    :param fe_dict: Dictionary containing the parameters for iron template which are amplitude, FWWHM,
                    redshift, intrinsic_FWHM, templ_disp_unit_str and templ_fluxden_unit_str. The dictionary should
                    contain each parameter value, the priors and boolean stating whether to vary the parameter.
    :param fe_dict:
    :param median_params: Dictionary containing median parameters to setup the iron template. This dict contains the median values for parameters taken from the initial fit.
    :return comp: model components for the iron template
    :return pars: model parameters for the iron template

    Note: Flux values from the iron template are in units of 1e-17 erg/s/cm2
    """
    MgII_disp = [2200, 3500]

    template = np.loadtxt(datadir + "iron_templates/" + "Tsuzuki06.txt")
    Fe_dispersion = template[:, 0]
    MgII_disp_mask = (Fe_dispersion >= MgII_disp[0]) & (Fe_dispersion <= MgII_disp[1])
    Fe_dispersion = Fe_dispersion[MgII_disp_mask]
    Fe_fluxden = template[:, 1][MgII_disp_mask] * 1.0e17

    comp, params = initialise_Fe(
        prefix, Fe_dispersion, Fe_fluxden, fe_dict, median_params
    )
    return comp, params


def setup_iron_template_Tsuz_OPT(prefix, fe_dict=None, median_params=None):
    """
    Setup the optical region of the Tsuzuki et al. 2006 template. This template is essentially based on BG 1992 iron template.

    :param prefix: component prefix
    :param type: str
    :param fe_dict: Dictionary containing the parameters for iron template which are amplitude, FWWHM,
                    redshift, intrinsic_FWHM, templ_disp_unit_str and templ_fluxden_unit_str. The dictionary should
                    contain each parameter value, the priors and boolean stating whether to vary the parameter.
    :param fe_dict:
    :param median_params: Dictionary containing median parameters to setup the iron template. This dict contains the median values for parameters taken from the initial fit.
    :return comp: model components for the iron template
    :return pars: model parameters for the iron template

    Note: Flux values from the iron template are in units of 1e-17 erg/s/cm2
    """
    opt_disp = [3500]

    template = np.loadtxt(datadir + "iron_templates/" + "Tsuzuki06.txt")
    Fe_dispersion = template[:, 0]
    opt_disp_mask = Fe_dispersion > opt_disp[0]
    Fe_dispersion = Fe_dispersion[opt_disp_mask]
    Fe_fluxden = template[:, 1][opt_disp_mask] * 1.0e17

    comp, params = initialise_Fe(
        prefix, Fe_dispersion, Fe_fluxden, fe_dict, median_params
    )
    return comp, params


def setup_iron_template_BG92_OPT_Hbeta(prefix, fe_dict=None, median_params=None):
    """
    Setup the Boroson and Green et al. 1992 iron template covering the region around Hbeta between 4435 - 5535 Å.
    The region is chosen from Shen et al. 2011 continnum windows.

    :param prefix: component prefix
    :param type: str
    :param fe_dict: Dictionary containing the parameters for iron template which are amplitude, FWWHM,
                    redshift, intrinsic_FWHM, templ_disp_unit_str and templ_fluxden_unit_str. The dictionary should
                    contain each parameter value, the priors and boolean stating whether to vary the parameter.
    :param fe_dict:
    :param median_params: Dictionary containing median parameters to setup the iron template. This dict contains the median values for parameters taken from the initial fit.
    :return comp: model components for the iron template
    :return pars: model parameters for the iron template

    Note: Flux values from the iron template are in units of 1e-17 erg/s/cm2
    """
    Hbeta_disp = [4435, 5535]

    template = np.loadtxt(datadir + "iron_templates/" + "Fe_OPT_BR92_linear.txt")
    Fe_dispersion = template[:, 0]
    Hbeta_disp_mask = (Fe_dispersion >= Hbeta_disp[0]) & (
        Fe_dispersion <= Hbeta_disp[1]
    )
    Fe_dispersion = Fe_dispersion[Hbeta_disp_mask]
    Fe_fluxden = template[:, 1][Hbeta_disp_mask] * 1.0e17

    comp, params = initialise_Fe(
        prefix, Fe_dispersion, Fe_fluxden, fe_dict, median_params
    )
    return comp, params


def setup_iron_template_VW01_Fe3(prefix, fe_dict=None, median_params=None):
    """
    Setup the Vestergaard et al. 2001 FeIII template within the range 2379 - 2456 Å.

    :param prefix: component prefix
    :param type: str
    :param fe_dict: Dictionary containing the parameters for iron template which are amplitude, FWWHM,
                    redshift, intrinsic_FWHM, templ_disp_unit_str and templ_fluxden_unit_str. The dictionary should
                    contain each parameter value, the priors and boolean stating whether to vary the parameter.
    :param fe_dict:
    :param median_params: Dictionary containing median parameters to setup the iron template. This dict contains the median values for parameters taken from the initial fit.
    :return comp: model components for the iron template
    :return pars: model parameters for the iron template

    Note: Flux values from the iron template are in units of 1e-17 erg/s/cm2
    """

    template = np.loadtxt(datadir + "iron_templates/" + "Fe3_UV47.asc")
    Fe_dispersion = template[:, 0]
    Fe_fluxden = template[:, 1] * 1.0e17

    comp, params = initialise_Fe(
        prefix, Fe_dispersion, Fe_fluxden, fe_dict, median_params
    )
    return comp, params


# ------------------------------------------------------------------------------
# Necessary calculations for fitting
# ------------------------------------------------------------------------------


def get_fwhm_int_flux(input_spec, low, upp, redsh, **kwargs):
    """
    Calculate the FWHM (in km/s) and integrated flux (1e-17 erg/s/cm^2) for
    a given emission line from the spectrum over a specified dispersion range.
    By default, the function uses "spline" method to interpolate
    the original spectrum and find the zero points using a root finding
    algorithm on the spline. The second method 'sign' finds sign changes in
    the half peak flux subtracted spectrum.

    To calculate integrated flux over a dispersion range, the standard
    numpy.trapz function is used for the integration.

    :param input_spec: input spectrum
    :type input_spec: sculptor.speconed.SpecOneD
    :param low: lower limit of the dispersion range
    :param upp: upper limit of the dispersion range
    :param redsh: redshift of the source
    :return: FWHM and integrated flux of the spectral feature
    :rtype: astropy.units.Quantity
    """
    fwhm = scana.get_fwhm(
        input_spec, [low * (1 + redsh), upp * (1 + redsh)], s=100, method="spline"
    )
    if not np.isnan(fwhm):
        fwhm = fwhm.value

    int_flux = (
        scana.get_integrated_flux(
            input_spec, [low * (1 + redsh), upp * (1 + redsh)]
        ).value
        * 1e17
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
    :param type: sculptor.speconed.SpecOneD
    :param redsh: redshift of the source
    :param type: float
    :param line_dict: emission line dictionary containing the parameters required to fit each of the mentioned emission lines.
    These parameters include flux, redshift, central_wavelength, FWHM and velocity shift.
    :param type: dict

    :return line_dict: an update line dictionary
    :return type: dict
    """
    line_range = {
        "Lya": [1150, 1290],
        "SiIV": [1290, 1450],
        "CIV": [1500, 1700],
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
                    params["flux"]["prior"].upper = int_flux * factor

            if "fwhm_km_s" in params and not np.isnan(fwhm):
                params["fwhm_km_s"]["value"] = fwhm
                if "prior_upp_factor" in params["fwhm_km_s"]:
                    factor = params["fwhm_km_s"]["prior_upp_factor"]
                    params["fwhm_km_s"]["prior"].upper = fwhm * factor

    return line_dict


# ------------------------------------------------------------------------------
# Analysis routines
# ------------------------------------------------------------------------------


def get_line_prop_MgII(df, redshift, param_name_dict, save_dir, source, fit_name):
    """
    Calculate AGN properties from the extracted emission line features (MgII) from the fit.
    The function provides measurements for the BH masses, Bolometric luminosity calculated using continuum luminosity at 3000 Å,
    Eddington luminosity, Eddington ratio, and line velocity shifts wrt to the reference redshift provided.
    Additionally, it calculates the iron contribution between 2200-3090 Å excluding any contribution from continuum or emission line feature and FeII/MgII ratio.
    From MgII line properties, we calculate the single-epoch virial BH mass based on the MgII FWHM and
    monochromatic continuum luminosity at 3000 Å. Depending on the iron template used, black hole masses are calculated using relation from
    Vestergaard & Osmer 2009, ApJ 641, 689 in case of Vestergaard template or Shen et al. 2011 in case of Tsuzuki 2006 template.

    :param df: input dataframe containing extracted emission line parameters
    :type df: pd.DataFrame
    :param redshift: redshift of the source
    :type redshift: float
    :param param_name_dict: dictionary specifying the columns names of the parameters in the provided dataframe containing -
        - "fwhm": str ("MgII_fwhm")
        - "luminosity": str ("L3000")
        - "redshift": str ("MgII_redsh")
        - "fe_amp": str ("FeII_amp")
        - "peak_flux": str ("MgII_peak_flux")
    :type param_name_dict: dict
    :param save_dir: path to save the output
    :type save_dir: str
    :param source: name of the source
    :type source: str
    :param model: fit model used
    :typpe model:
    :param flat_samples: flat chain from the MCMC fit

    :return df: output dataframe
    :type df: pd.DataFrames
    """

    fwhm_col = param_name_dict.get("fwhm")
    lum_col = param_name_dict.get("luminosity")
    z_col = param_name_dict.get("redshift")
    fe_amp_col = param_name_dict.get("fe_amp")
    peak_flux_col = param_name_dict.get("peak_flux")

    for idx, row in df.iterrows():
        for key in ["_median", "_low", "_upp"]:
            # MgII BH mass (depends on fit_name)
            if fwhm_col and lum_col:
                fwhm_key = fwhm_col + key
                lum_key = lum_col + key
                if fwhm_key in df.columns and lum_key in df.columns:
                    fwhm = np.float64(row[fwhm_key])
                    lum = np.float64(row[lum_key])
                    if not (np.isnan(fwhm) or np.isnan(lum)):
                        if fit_name == "Tsuzuki":
                            BH_mass = qso.se_bhmass_mgii_s11_fwhm(
                                fwhm, cont_lwav=lum, cont_wav=3000 * u.AA
                            )
                        elif fit_name == "Vest":
                            BH_mass = qso.se_bhmass_mgii_vo09_fwhm(
                                fwhm, cont_lwav=lum, cont_wav=3000 * u.AA
                            )
                        else:
                            print(
                                f"[INFO] Unknown fit_name '{fit_name}' — skipping BH mass calc."
                            )
                            BH_mass = [np.nan * u.Msun]

                        df.at[idx, "MgII_BH_mass_3000" + key] = BH_mass[0].value

                        # Bolometric luminosity and Eddington ratio
                        L_bol = qso.calc_bolometric_luminosity(
                            cont_lwav=lum, cont_wav=3000 * u.AA
                        )
                        df.at[idx, "L_bol_3000" + key] = L_bol[0].value

                        Edd_ratio = qso.calc_eddington_ratio(L_bol[0], BH_mass[0])
                        df.at[idx, "Eddington_ratio_3000" + key] = Edd_ratio.value

                        # Eddington luminosity
                        Edd_lum = qso.calc_eddington_luminosity(BH_mass[0])
                        df.at[idx, "Edd_lum_MgII_3000" + key] = Edd_lum.value
                    else:
                        print(
                            f"[INFO] NaN in '{fwhm_key}' or '{lum_key}' at row {idx} — skipping BH mass & L_bol."
                        )
                else:
                    print(
                        f"[INFO] Missing columns '{fwhm_key}' or '{lum_key}' — skipping BH mass & L_bol."
                    )
            else:
                print(
                    "[INFO] 'fwhm' or 'luminosity' not in param_name_dict — skipping BH mass & L_bol."
                )

            # Velocity shift
            if z_col:
                z_key = z_col + key
                if z_key in df.columns:
                    try:
                        delta_v = qso.calc_velocity_shifts(
                            row[z_key], z_sys=redshift, relativistic=True
                        )
                        df.at[idx, "delta_v_MgII" + key] = delta_v.value
                    except Exception as e:
                        print(f"[INFO] Velocity shift calc failed at row {idx}: {e}")
                else:
                    print(f"[INFO] Column '{z_key}' missing — skipping velocity shift.")
            else:
                print(
                    "[INFO] 'redshift' not in param_name_dict — skipping velocity shift."
                )

            # FeII/MgII flux ratio
            if fe_amp_col and peak_flux_col:
                fe_key = fe_amp_col + key
                flux_key = peak_flux_col + key
                if fe_key in df.columns and flux_key in df.columns:
                    try:
                        fe_flux_ratio = row[fe_key] * 1e-17 / row[flux_key]
                        df.at[idx, "FeII_over_MgII" + key] = fe_flux_ratio
                    except Exception as e:
                        print(f"[INFO] FeII/MgII flux ratio failed at row {idx}: {e}")
                else:
                    print(
                        f"[INFO] Missing columns '{fe_key}' or '{flux_key}' — skipping FeII/MgII."
                    )
            else:
                print(
                    "[INFO] 'fe_amp' or 'peak_flux' not in param_name_dict — skipping FeII/MgII."
                )

    df_path = os.path.join(save_dir, f"{source}_results_MgII.csv")
    df.to_csv(df_path, index=False, float_format="%.4e")
    return df


def get_line_prop_Hbeta_OIII(
    df, redshift, param_name_dict, save_dir, source, model, flat_samples
):
    """
    Calculate AGN properties from the extracted emission line features (Hbeta) from the fit.
    The function provides measurements for the BH masses, Bolometric luminosity calculated using continuum luminosity at 5100 Å,
    Eddington luminosity, Eddington ratio, and line velocity shifts wrt to the reference redshift provided.
    From Hbeta line properties, we calculate the single-epoch virial BH mass based on the Hbeta FWHM and
    monochromatic continuum luminosity at 5100 Å. This relation is taken from Vestergaard & Peterson 2006, ApJ 641, 689 and
    based on line width measurements of quasars published in Boroson & Green 1992 and Marziani 2003.

    :param df: input dataframe containing extracted emission line parameters
    :type df: pd.DataFrame
    :param redshift: redshift of the source
    :type redshift: float
    :param param_name_dict: dictionary specifying the columns names of the parameters in the provided dataframe containing -
        - "fwhm" : str ("Hbeta_fwhm")
        - "luminosity" : str ("L5100")
        - "redshift" : str ("Hbeta_redsh")
    :type param_name_dict: dictionary
    :param save_dir: path to save the output
    :type save_dir: str
    :param source: name of the source
    :type source: str
    :param model: fit model used
    :typpe model:
    :param flat_samples: flat chain from the MCMC fit
    """

    fwhm_col = param_name_dict.get("fwhm")
    lum_col = param_name_dict.get("luminosity")
    z_col = param_name_dict.get("redshift")

    for idx, row in df.iterrows():
        for key in ["_median", "_low", "_upp"]:
            # FWHM and Luminosity-based calculations
            if fwhm_col and lum_col:
                fwhm_key = fwhm_col + key
                lum_key = lum_col + key

                if fwhm_key in df.columns and lum_key in df.columns:
                    fwhm_Hbeta = np.float64(row[fwhm_key])
                    cont_lwave_5100 = np.float64(row[lum_key])

                    if not (np.isnan(fwhm_Hbeta) or np.isnan(cont_lwave_5100)):
                        hbeta_BH_mass_5100 = qso.se_bhmass_hbeta_vp06(
                            fwhm_Hbeta, cont_lwav=cont_lwave_5100, cont_wav=5100 * u.AA
                        )
                        L_bol_5100 = qso.calc_bolometric_luminosity(
                            cont_lwav=cont_lwave_5100, cont_wav=5100 * u.AA
                        )
                        Edd_lum_Hbeta_5100 = qso.calc_eddington_luminosity(
                            hbeta_BH_mass_5100[0]
                        )
                        Eddington_ratio_5100 = qso.calc_eddington_ratio(
                            L_bol_5100[0], hbeta_BH_mass_5100[0]
                        )

                        df.at[idx, "Hbeta_BH_mass_5100" + key] = hbeta_BH_mass_5100[
                            0
                        ].value
                        df.at[idx, "Edd_lum_Hbeta_5100" + key] = (
                            Edd_lum_Hbeta_5100.value
                        )
                        df.at[idx, "Eddington_ratio_5100" + key] = (
                            Eddington_ratio_5100.value
                        )
                        df.at[idx, "L_bol_5100" + key] = L_bol_5100[0].value
                    else:
                        print(
                            f"[INFO] NaN in '{fwhm_key}' or '{lum_key}' at row {idx} — skipping BH mass and Eddington calculations."
                        )
                else:
                    print(
                        f"[INFO] One or both columns '{fwhm_key}', '{lum_key}' missing in DataFrame — skipping BH mass and Eddington calculations."
                    )
            else:
                print(
                    "[INFO] 'fwhm' or 'luminosity' prefix not provided — skipping BH mass and Eddington calculations."
                )

            # Velocity shift calculation
            if z_col:
                z_key = z_col + key
                if z_key in df.columns:
                    try:
                        delta_v_Hbeta = qso.calc_velocity_shifts(
                            row[z_key], z_sys=redshift, relativistic=True
                        )
                        df.at[idx, "delta_v_Hbeta" + key] = delta_v_Hbeta.value
                    except Exception as e:
                        print(
                            f"[INFO] Failed to calculate velocity shift for '{z_key}' at row {idx}: {e}"
                        )
                else:
                    print(
                        f"[INFO] Column '{z_key}' missing in DataFrame — skipping velocity shift."
                    )
            else:
                print(
                    "[INFO] 'redshift' key not provided in param_name_dict — skipping velocity shift calculation."
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
    :type flat_chain:
    :param redshift: redshift of the source
    :type redshift: float
    :param fwhm: FWHM of the line in consideration
    :type FWHM: float
    :param comp_pl: continuum components to be subtracted before SNR calculation
    :type comp_pl:
    :param save_dir: path to save the output
    :type save_dir: str
    :param line_name: emission line in consideration for SNR calculation
    :type line_name: str
    :param line_center: central wavelength of the emission line
    :type line_center: float

    :return df: dataframe containing peak and total line SNR
    :type df: pd.DataFrame
    """

    for idx, row in df.iterrows():

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

        df_snr = pd.DataFrame(line_snr.T, columns=list(line_dict.keys()))
        df_snr.to_csv(
            os.path.join(save_dir, f"{source}_{line_name}_line_snr.csv"), index=False
        )
        df_snr = pd.DataFrame(peak_snr.T, columns=list(line_dict.keys()))
        df_snr.to_csv(
            os.path.join(save_dir, f"{source}_{line_name}_peak_snr.csv"), index=False
        )

        med, low, upp = get_1sigma_range(line_snr[0])
        df.at[idx, f"{line_name}_line_snr_median"] = med
        df.at[idx, f"{line_name}_line_snr_low"] = low
        df.at[idx, f"{line_name}_line_snr_upp"] = upp

        med, low, upp = get_1sigma_range(peak_snr[0])
        df.at[idx, f"{line_name}_peak_snr_median"] = med
        df.at[idx, f"{line_name}_peak_snr_low"] = low
        df.at[idx, f"{line_name}_peak_snr_upp"] = upp

    return df
