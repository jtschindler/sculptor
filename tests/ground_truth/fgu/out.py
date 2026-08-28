"""
The Sculptor Quasar Extension

This module defines models, masks and analysis routines specific for the
analysis of type-I QSOs (quasars, type-I AGN).

At first we define the basic model functions and their setups. The setup
functions initialize models and parameters using the basic model
functions defined here. Users can build customised models using functions defined in this module.

Complex models can be constructed by combining multiple of the basic model
functions. For example, we define a setup function to initialize a model for
the Hbeta and [OIII] lines consisting multiple Gaussian emission lines for broad and narrow components.
"""

import importlib.resources
import os
import string

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy import units as u
from astropy.constants import c, h, k_B
from astropy.modeling.models import BlackBody
from scipy.special import voigt_profile

from sculptor import analysis as ana
from sculptor import colors as tolc
from sculptor import component as sccomp
from sculptor import prior as scpri

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

# -----------------------------------------------------------------------------
# Added by FGU
from sculptor.dep_v1 import specanalysis as scana
from sculptor_extensions import qso

# -----------------------------------------------------------------------------

resource_path = importlib.resources.files("sculptor")
cmap = tolc.tol_cmap(colormap="rainbow_PuRd")
datadir = str(resource_path / "data") + "/"

c_kms = c.to("km/s").value
c_AngstroemPerSecond = c.to(u.AA / u.s).value

# Physical constants in cgs, computed once at import time (not per call)
_H_CGS = h.cgs.value  # erg s
_C_CGS = c.cgs.value  # cm/s
_KB_CGS = k_B.cgs.value  # erg/K

_HOL_DATA = np.loadtxt(datadir + "wave_2.txt")
_HOL_TRANSITIONS = _HOL_DATA[:, 0]
_HOL_EMISS = _HOL_DATA[:, 1]
_SIGMA_TO_FWHM = 2.0 * np.sqrt(2.0 * np.log(2.0))

# ------------------------------------------------------------------------------
# Masks
# ------------------------------------------------------------------------------

# QSO continuum windows EUV (Vestergaard & Peterson 2006)

"""Description of the EUV continuum windows:
VP06: We fitted the rest-frame UV spectra with a power-law contin-
uum in nominally line-free windows typically in the wavelength
ranges 1265–1290, 1340–1375, 1425–1470, 1680–1705, and
1950–2050 8
Shen12: We fit the pseudo-continuum model to a set of contin-
uum windows free of strong emission lines (except for Fe ii):
1350–1360 Å, 1445–1465 Å, 1700–1705 Å, 2155–2400 Å,
2480–2675 Å, 2925–3500 Å, 4200–4230 Å, 4435–4700 Å,
5100–5535 Å, 6000–6250 Å, and 6800–7000 Å.
"""

# QSO continuum windows from Vestergaard and Peterson 2006 and Shen et al. 2012
qso_cont_feII = {
    "name": "QSO Continuum+FeII",
    "rest_frame": True,
    "mask_ranges": [
        [1265, 1290],
        [1950, 2050],
        [1350, 1360],
        [1445, 1465],
        [1690, 1705],
        [2155, 2400],
        [2480, 2675],
        [2925, 3500],
        [4200, 4230],
        [4150, 4230],
        [4435, 4700],
        [5100, 5535],
        [5100, 5200],
        [6000, 6250],
        [6800, 7000],
    ],
}

# QSO continuum windows from Vestergaard & Peterson 2006
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

# QSO continuum + iron windows, see Shen+2011
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

qso_contfe_CIV_Shen11 = {
    "name": "QSO Cont. CIV Shen11",
    "rest_frame": True,
    "mask_ranges": [[1445, 1465], [1700, 1705]],
}

# Emission line windows
MgII_range = {"name": "MgII_range", "rest_frame": True, "mask_ranges": [[2675, 2925]]}

Hbeta_range = {"name": "Hbeta_range", "rest_frame": True, "mask_ranges": [[4700, 5100]]}

Hgamma_range = {
    "name": "Hgamma_range",
    "rest_frame": True,
    "mask_ranges": [[4300, 4400]],
}

Hdelta_range = {
    "name": "Hdelta_range",
    "rest_frame": True,
    "mask_ranges": [[4000, 4200]],
}

HeI_range = {"name": "HeI_range", "rest_frame": True, "mask_ranges": [[3800, 3990]]}

# Atmospheric windows for highly affected regions and iron3 - from telfit10000, focusing on flux<0.8
Telluric_ranges = {
    "name": "Telluric_ranges",
    "rest_frame": False,
    "mask_ranges": [
        [13450, 14300],
        [18000, 19500],
    ],
}

FeIII_range = {
    "name": "FeIII",
    "rest_frame": True,
    "mask_ranges": [
        # [2379, 2455],
        [2250, 2675],
    ],
}


###exclude additional ranges 1. no iron template available 2.reward of [OIII]
No_FeII_range = {
    "name": "No_FeII_range",
    "rest_frame": True,
    "mask_ranges": [[3500, 3685]],
}

No_fit_range = {
    "name": "No_fit",
    "rest_frame": True,
    "mask_ranges": [[5400, 5700]],
}

# ------------------------------------------------------------------------------
# Iron template models
# ------------------------------------------------------------------------------


def setup_iron_template_VW01_UV(
    prefix,
    disp_range=None,
    fe_dict=None,
    median_params=None,
):
    """
    setup vestergaard uv iron template
    """

    template = np.loadtxt(datadir + "iron_templates/" + "Fe_UVtemplt_A.asc")
    Fe_dispersion = template[:, 0]
    Fe_fluxden = template[:, 1] * 1.0e17

    if disp_range is not None:
        cond_Fe_MgII = (Fe_dispersion < disp_range[1]) & (Fe_dispersion > disp_range[0])
        Fe_dispersion = Fe_dispersion[cond_Fe_MgII]
        Fe_fluxden = Fe_fluxden[cond_Fe_MgII]

    comp, params = initialise_Fe(
        prefix,
        Fe_dispersion,
        Fe_fluxden,
        fe_dict,
        median_params,
    )

    return comp, params


def setup_iron_template_BG92_OPT(
    prefix,
    disp_range=None,
    fe_dict=None,
    median_params=None,
):
    """
    setup borson iron template
    """

    template = np.loadtxt(datadir + "iron_templates/" + "Fe_OPT_BR92_linear.txt")
    Fe_dispersion = template[:, 0]
    Fe_fluxden = template[:, 1] * 1.0e17

    if disp_range is not None:
        cond_Fe_MgII = (Fe_dispersion < disp_range[1]) & (Fe_dispersion > disp_range[0])
        Fe_dispersion = Fe_dispersion[cond_Fe_MgII]
        Fe_fluxden = Fe_fluxden[cond_Fe_MgII]

    comp, params = initialise_Fe(
        prefix, Fe_dispersion, Fe_fluxden, fe_dict, median_params
    )
    return comp, params


def setup_iron_template_Tsuz_UV(prefix, fe_dict=None, median_params=None):
    """
    Setup Tsuzuki et al. 2006 UV iron template
    """

    template = np.loadtxt(datadir + "iron_templates/" + "Tsuzuki06.txt")
    Fe_dispersion = template[:, 0]
    cond_Fe_MgII = Fe_dispersion < 3500
    Fe_dispersion = Fe_dispersion[cond_Fe_MgII]
    Fe_fluxden = template[:, 1][cond_Fe_MgII] * 1.0e17

    comp, params = initialise_Fe(
        prefix, Fe_dispersion, Fe_fluxden, fe_dict, median_params
    )
    return comp, params


def setup_iron_template_Tsuz_Uv_MgII(
    prefix, disp_range=[2200, 3500], fe_dict=None, median_params=None
):
    """
    Setup Tsuzuki et al. 2006 UV iron template
    """

    template = np.loadtxt(datadir + "iron_templates/" + "Tsuzuki06.txt")
    Fe_dispersion = template[:, 0]
    cond_Fe_MgII = (Fe_dispersion < disp_range[1]) & (Fe_dispersion > disp_range[0])
    Fe_dispersion = Fe_dispersion[cond_Fe_MgII]
    Fe_fluxden = template[:, 1][cond_Fe_MgII] * 1.0e17

    comp, params = initialise_Fe(
        prefix, Fe_dispersion, Fe_fluxden, fe_dict, median_params
    )
    return comp, params


def setup_iron_template_Tsuz_MgII_region(
    prefix, disp_range=[2200, 3090], fe_dict=None, median_params=None
):
    """
    Setup Tsuzuki et al. 2006 UV iron template
    """

    template = np.loadtxt(datadir + "iron_templates/" + "Tsuzuki06.txt")
    Fe_dispersion = template[:, 0]
    cond_Fe_MgII = (Fe_dispersion < disp_range[1]) & (Fe_dispersion > disp_range[0])
    Fe_dispersion = Fe_dispersion[cond_Fe_MgII]
    Fe_fluxden = template[:, 1][cond_Fe_MgII] * 1.0e17

    comp, params = initialise_Fe(
        prefix, Fe_dispersion, Fe_fluxden, fe_dict, median_params
    )
    return comp, params


def setup_iron_template_Tsuz_MgII_Hband(
    prefix, disp_range=[3190, 3500], fe_dict=None, median_params=None
):
    """
    Setup Tsuzuki et al. 2006 UV iron template
    """

    template = np.loadtxt(datadir + "iron_templates/" + "Tsuzuki06.txt")
    Fe_dispersion = template[:, 0]
    cond_Fe_MgII = (Fe_dispersion < disp_range[1]) & (Fe_dispersion > disp_range[0])
    Fe_dispersion = Fe_dispersion[cond_Fe_MgII]
    Fe_fluxden = template[:, 1][cond_Fe_MgII] * 1.0e17

    comp, params = initialise_Fe(
        prefix, Fe_dispersion, Fe_fluxden, fe_dict, median_params
    )
    return comp, params


def setup_iron_template_BG92_OPT_Hband(
    prefix, disp_range=[3685, 4025], fe_dict=None, median_params=None
):
    """
    Setup Tsuzuki et al. 2006 UV iron template
    """

    template = np.loadtxt(datadir + "iron_templates/" + "Fe_OPT_BR92_linear.txt")
    Fe_dispersion = template[:, 0]
    cond_Fe_MgII = (Fe_dispersion < disp_range[1]) & (Fe_dispersion > disp_range[0])
    Fe_dispersion = Fe_dispersion[cond_Fe_MgII]
    Fe_fluxden = template[:, 1][cond_Fe_MgII] * 1.0e17

    comp, params = initialise_Fe(
        prefix, Fe_dispersion, Fe_fluxden, fe_dict, median_params
    )
    return comp, params


def setup_iron_template_BG92_OPT_Hbeta_region(
    prefix, disp_range=[4350, 5400], fe_dict=None, median_params=None
):
    """
    Setup Tsuzuki et al. 2006 UV iron template
    """

    template = np.loadtxt(datadir + "iron_templates/" + "Fe_OPT_BR92_linear.txt")
    Fe_dispersion = template[:, 0]
    cond_Fe_MgII = (Fe_dispersion < disp_range[1]) & (Fe_dispersion > disp_range[0])
    Fe_dispersion = Fe_dispersion[cond_Fe_MgII]
    Fe_fluxden = template[:, 1][cond_Fe_MgII] * 1.0e17

    comp, params = initialise_Fe(
        prefix, Fe_dispersion, Fe_fluxden, fe_dict, median_params
    )
    return comp, params


def setup_iron_template_VW01_MgII(
    prefix, disp_range=[2200, 3089], fe_dict=None, median_params=None
):
    """ """
    template = np.loadtxt(datadir + "iron_templates/" + "Fe_UVtemplt_A.asc")
    Fe_dispersion = template[:, 0]
    cond_Fe_MgII = (Fe_dispersion < disp_range[1]) & (Fe_dispersion > disp_range[0])
    Fe_dispersion = Fe_dispersion[cond_Fe_MgII]
    Fe_fluxden = template[:, 1][cond_Fe_MgII] * 1.0e17

    comp, params = initialise_Fe(
        prefix, Fe_dispersion, Fe_fluxden, fe_dict, median_params
    )
    return comp, params


def setup_iron_template_Tsuz_OPT(prefix, fe_dict=None, median_params=None):
    """
    Setup Tsuzuki et al. 2006 OPT iron template - based on boroson and green et al. 1992 iron template
    """

    template = np.loadtxt(datadir + "iron_templates/" + "Tsuzuki06.txt")
    Fe_dispersion = template[:, 0]
    cond_Fe_Hbeta = Fe_dispersion > 3500
    Fe_dispersion = Fe_dispersion[cond_Fe_Hbeta]
    Fe_fluxden = template[:, 1][cond_Fe_Hbeta] * 1.0e17

    comp, params = initialise_Fe(
        prefix, Fe_dispersion, Fe_fluxden, fe_dict, median_params
    )
    return comp, params


def setup_iron_template_VW01_Fe3_47(prefix, fe_dict=None, median_params=None):
    """
    setup feiii iron template
    """

    template = np.loadtxt(datadir + "iron_templates/" + "Fe3_UV47.asc")
    Fe_dispersion = template[:, 0]
    Fe_fluxden = template[:, 1] * 1.0e17

    comp, params = initialise_Fe(
        prefix,
        Fe_dispersion,
        Fe_fluxden,
        fe_dict,
        median_params,
    )
    return comp, params


def setup_iron_template_VW01_Fe3_34(prefix, fe_dict=None, median_params=None):
    """
    setup feiii iron template
    """

    template = np.loadtxt(datadir + "iron_templates/" + "Fe3UV34modelB2.asc")
    Fe_dispersion = template[:, 0]
    Fe_fluxden = template[:, 1] * 1.0e17

    comp, params = initialise_Fe(
        prefix,
        Fe_dispersion,
        Fe_fluxden,
        fe_dict,
        median_params,
    )
    return comp, params


def setup_iron_template_VW01_Fe3_2038_2113(prefix, fe_dict=None, median_params=None):
    """
    setup feiii iron template
    """

    template = np.loadtxt(datadir + "iron_templates/" + "Fe3UV34_VW01_2038-2113.asc")
    Fe_dispersion = template[:, 0]
    Fe_fluxden = template[:, 1] * 1.0e17

    comp, params = initialise_Fe(
        prefix,
        Fe_dispersion,
        Fe_fluxden,
        fe_dict,
        median_params,
    )
    return comp, params


def setup_iron_template_cloudy(prefix, fe_dict=None, median_params=None):
    """
    setup feiii iron template
    """

    template = np.loadtxt(datadir + "iron_templates/" + "phi17.0_nH9.0_m0.dat")
    Fe_dispersion = template[:, 0]
    Fe_fluxden = template[:, 1] * 1.0e17

    comp, params = initialise_Fe(
        prefix,
        Fe_dispersion,
        Fe_fluxden,
        fe_dict,
        median_params,
    )

    return comp, params


# ------------------------------------------------------------------------------
# Necessary calculations before fitting
# ------------------------------------------------------------------------------


def get_fwhm_int_flux(input_spec, low, upp, redsh, **kwargs):
    """
    Calculate the FWHM (in km/s) and integrated flux (1e-17 erg/s/cm^2/AA) of
    a given emission line from the spectrum over a specified dispersion range.
    By default, the function uses "spline" method to interpolate
    the original spectrum and find the zero points using a root finding
    algorithm on the spline. The second method 'sign' finds sign changes in
    the half peak flux subtracted spectrum.

    To calculate integrated flux over a dispersion range, the standard
    numpy.trapz function is used for the integration.

    :param input_spec: Input spectrum
    :type input_spec: sculptor.speconed.SpecOneD
    :param low: lower limit of dispersion range
    :param upp: upper limit of dispersion range
    :param redsh: Redshift of the source
    :return: FWHM and integrated flux of the spectral feature
    :rtype: astropy.units.Quantity
    """
    fwhm = scana.get_fwhm(
        input_spec, [low * (1 + redsh), upp * (1 + redsh)], method="spline"
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
    fwhm and int flux for
    """
    line_range = {
        "MgII": [2760, 2850],
        "Hbeta": [4800, 4930],
        "OIII": [4990, 5025],
    }

    for line, range in line_range.items():
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
# Initialise model parameters (continuum, iron and emission lines)
# ------------------------------------------------------------------------------


def initialise_power_law_old(prefix, redsh, powerlaw_dict=None, median_params=None):
    """
    Setup power law model with pre-defined parameters from the user given dictionary.
    This model is defined for a spectral dispersion axis in Angstroem.

    :param prefix: Model prefix         ########
    :type prefix: string
    :param redsh: Redshift of the source
    :type redsh: float
    :param powerlaw_dict: Dictionary containing power law components "amp", "slope".
    In this dictionary, each component should have "initial_value", "vary" and "prior"
    which provides information on initial value of the parameter, whether the parameter should be varied, and
    prior to be used to contrain the parameter, respectively.
    :type amp: Dictionary
    :param ml_params: Dictionary containing median or maximum_likelihood parameters
    from any previous fits to constrain the parameter values.
    :type ml_params: Dictionary
    :return: Power law model components and parameters
    :rtype:                 ##confirm this one
    """
    comp = sccomp.FitComponent(prefix, ana.power_law_at2500)
    params = comp.create_params()

    if median_params:
        amp = median_params[f"{prefix}_amp"]
        slope = median_params[f"{prefix}_slope"]

        params[f"{prefix}_amp"].value = amp
        params[f"{prefix}_amp"].vary = True
        params[f"{prefix}_amp"].prior = scpri.GaussianPrior(f"{prefix}_amp", amp, 0.1)
        params[f"{prefix}_slope"].value = slope
        params[f"{prefix}_slope"].vary = True
        params[f"{prefix}_slope"].prior = scpri.GaussianPrior(
            f"{prefix}_slope", slope, 0.1
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
        raise ValueError("Provide either `ml_params` or `powerlaw_dict`")

    return comp, params


def initialise_power_law(prefix, redsh, powerlaw_dict=None, median_params=None):
    """
    Setup power law model with pre-defined parameters from the user given dictionary.
    This model is defined for a spectral dispersion axis in Angstroem.

    :param prefix: Model prefix
    :type prefix: string
    :param redsh: Redshift of the source
    :type redsh: float
    :param powerlaw_dict: Dictionary containing power law components "amp", "slope".
    In this dictionary, each component should have "initial_value", "vary" and "prior"
    which provides information on initial value of the parameter, whether the parameter should be varied, and
    prior to be used to contrain the parameter, respectively.
    :type amp: Dictionary
    :param ml_params: Dictionary containing median or maximum_likelihood parameters
    from any previous fits to constrain the parameter values.
    :type ml_params: Dictionary
    :return: Power law model components and parameters
    """
    comp = sccomp.FitComponent(prefix, ana.power_law_at2500)
    params = comp.create_params()

    if median_params:
        amp = median_params[f"{prefix}_amp"]
        slope = median_params[f"{prefix}_slope"]

        sigma_amp = median_params.get(f"{prefix}_amp_sigma", 0.01)
        sigma_slope = median_params.get(f"{prefix}_slope_sigma", 0.01)

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
        raise ValueError("Provide either `ml_params` or `powerlaw_dict`")

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
    fast_model = ana.make_template_model(Fe_dispersion, Fe_fluxden)
    comp = sccomp.FitComponent(prefix, fast_model)

    # comp = sccomp.FitComponent(prefix, ana.template_model)
    params = comp.create_params()

    if median_params:
        amp = median_params[f"{prefix}_amp"]
        fwhm = median_params[f"{prefix}_fwhm"]
        redsh = median_params[f"{prefix}_redsh"]

        sigma_amp = median_params.get(f"{prefix}_amp_sigma", 0.01)
        sigma_fwhm = median_params.get(f"{prefix}_fwhm_sigma", 50)
        sigma_redsh = median_params.get(f"{prefix}_redsh_sigma", 0.05)

        params[f"{prefix}_amp"].value = amp
        params[f"{prefix}_amp"].vary = True
        params[f"{prefix}_amp"].prior = scpri.GaussianPrior(
            f"{prefix}_amp", amp, sigma_amp
        )

        params[f"{prefix}_fwhm"].value = fwhm
        params[f"{prefix}_fwhm"].vary = True
        # params[f"{prefix}_fwhm"].prior = scpri.GaussianPrior(f"{prefix}_fwhm", fwhm, 5)
        params[f"{prefix}_fwhm"].prior = scpri.UniformPrior(
            f"{prefix}_fwhm", fwhm - 100, fwhm + 100
        )

        params[f"{prefix}_redsh"].value = redsh
        params[f"{prefix}_redsh"].vary = True
        params[f"{prefix}_redsh"].prior = scpri.GaussianPrior(
            f"{prefix}_redsh", redsh, sigma_redsh
        )  ##0.08

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
        raise ValueError("Provide either `ml_params` or `FeII_dict`")

    return comp, params


def initialise_balmer(prefix, model, balmer_dict=None, median_params=None):
    """
    Initialise Balmer continuum including the higher order balmer lines that blend as a pseudo-continuum.
    Following Storey and Hummer 1995, we consider higher order lines from (7<=n<=50). Higher order line emissivities were calculated
    fromthe database provided in under the case B approximation for Hydrogen, for a given temperature, electron density, and braodening Doppler velocity.
    a Balmer template such as "amp_be", "redshift", "Te", "tau_be", "lambda_be" and "fwhm".
    In this dictionary, each component should have "initial_value", "vary" and "prior"
    which provides information on initial value of the parameter, whether the parameter should be varied, and
    prior to be used to contrain the parameter, respectively.
    :param prefix: Model prefix
    :type prefix: string
    :param balmer_dict: dictionary containing parameters to setup the Balmer continuum model
    :type Balmer_dict: dictionary
    :param median_params: dictionary containing median parameters to setup the Balmer continuum model
    :type median_params: dictionary

    :return: Balmer continuuum model component and parameters

    """
    comp = sccomp.FitComponent(prefix, Balmer_continuum_with_hol)
    params = comp.create_params()

    if median_params:
        amp_be = median_params[f"{prefix}_amp_be"]
        sigma_amp = median_params.get(f"{prefix}_amp_sigma", 0.01)
        # fwhm = median_params[f"{prefix}_fwhm"]

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
        raise ValueError("Provide either `ml_params` or `balmer_dict`")

    return comp, params


def initialise_Fe_old(
    prefix, Fe_dispersion, Fe_fluxden, Fe_dict=None, median_params=None
):
    """
    Setup iron templates and ml params is a dict
    """
    comp = sccomp.FitComponent(prefix, ana.template_model)
    params = comp.create_params()

    if median_params:
        amp = median_params[f"{prefix}_amp"]
        fwhm = median_params[f"{prefix}_fwhm"]
        redsh = median_params[f"{prefix}_redsh"]

        params[f"{prefix}_amp"].value = amp
        params[f"{prefix}_amp"].vary = True
        params[f"{prefix}_amp"].prior = scpri.GaussianPrior(f"{prefix}_amp", amp, 0.15)

        params[f"{prefix}_fwhm"].value = fwhm
        params[f"{prefix}_fwhm"].vary = True
        # params[f"{prefix}_fwhm"].prior = scpri.GaussianPrior(
        #     f"{prefix}_fwhm", fwhm, 100  ##use 100
        # )
        params[f"{prefix}_fwhm"].prior = scpri.UniformPrior(
            f"{prefix}_fwhm",
            fwhm - 50,
            fwhm + 50,  ##use 100
        )

        params[f"{prefix}_redsh"].value = redsh
        params[f"{prefix}_redsh"].vary = True
        params[f"{prefix}_redsh"].prior = scpri.GaussianPrior(
            f"{prefix}_fwhm",
            redsh,
            0.05,  ##previously 0.08
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
        raise ValueError("Provide either `ml_params` or `FeII_dict`")

    return comp, params


# Faster version
def fast_balmer_continuum_model(x, z, amp_be, Te, tau_be, lambda_be):
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


def initialise_balmer_without_hol(prefix, balmer_dict=None, median_params=None):
    """
    Initialise Balmer continuum component without higher order lines.
    Function requires a dictionary that contains necessary parameters to set up
    a Balmer template such as "amp_be", "redshift", "Te", "tau_be", "lambda_be".
    In this dictionary, each component should have "initial_value", "vary" and "prior"
    which provides information on initial value of the parameter, whether the parameter should be varied, and
    prior to be used to contrain the parameter, respectively.
    :param prefix: Model prefix
    :type prefix: string
    :param balmer_dict: dictionary containing parameters to setup the Balmer continuum model
    :type Balmer_dict: dictionary
    :param median_params: dictionary containing median parameters to setup the Balmer continuum model
    :type median_params: dictionary

    :return: Balmer continuuum model component and parameters
    """
    comp = sccomp.FitComponent(prefix, fast_balmer_continuum_model)
    # qso.balmer_continuum_model)
    params = comp.create_params()

    if median_params:
        amp_be = median_params[f"{prefix}_amp_be"]

        params[f"{prefix}_amp_be"].value = amp_be
        params[f"{prefix}_amp_be"].vary = True
        params[f"{prefix}_amp_be"].prior = scpri.GaussianPrior(
            f"{prefix}_amp_be", amp_be, 0.05
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
        raise ValueError("Provide either `ml_params` or `powerlaw_dict`")

    return comp, params


def initialise_balmer_old(prefix, model, balmer_dict=None, median_params=None):
    comp = sccomp.FitComponent(prefix, Balmer_continuum_with_hol)
    params = comp.create_params()

    if median_params:
        amp_be = median_params[f"{prefix}_amp_be"]
        # fwhm = median_params[f"{prefix}_fwhm"]

        params[f"{prefix}_amp_be"].value = amp_be
        params[f"{prefix}_amp_be"].vary = True
        params[f"{prefix}_amp_be"].prior = scpri.GaussianPrior(
            f"{prefix}_amp_be", amp_be, 0.01
        )  ##0.05

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
        raise ValueError("Provide either `ml_params` or `balmer_dict`")

    return comp, params


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
    :return: Components and parameters for a gaussian line model
    :rtype:                 #### confirm this again
    """
    if "OIII" in prefix:
        comp = sccomp.FitComponent(prefix, qso.line_model_gaussian_oiii_doublet)
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


def setup_MgII_lines(
    spec, line_dict, narrow=True, MgII_broad=1, MgII_broad_shifted=0, model=None
):

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

    for j in range(MgII_broad_shifted):
        comp_MgII_broad_shifted, pars_MgII_broad_shifted = (
            initialise_gaussian_line_model(
                f"MgII_broad_shifted_{string.ascii_lowercase[j]}",
                line_dict["MgII"]["broad_shifted"]["params"],
            )
        )
        model.add_component(comp_MgII_broad_shifted)
        model.add_parameters(pars_MgII_broad_shifted)


def setup_hb_o3_lines(
    spec,
    line_dict,
    narrow=True,
    Hbeta_broad=1,
    Hbeta_broad_shifted=0,
    OIII_broad=0,
    OIII_wind=0,
    model=None,
):
    """
    Setup Hbeta+[OIII] lines simultaneously at 4862.30 A, 4960.30 A and 5008.24 A.
    The function creates an emission line dictionary with narrow, broad and blue-shifted components for each line.

    The narrow component has FWHM between 300-1000 km/s and broad component from 1000 to 10000 km/s.
    It internally calculates FWHM and integrated flux of Hbeta and [OIII] line regions separately which are further incorporated in determining the priors.
    By default, the model fits 1 narrow component for Hbeta and each of the [OIII] lines and 1 broad component for Hbeta.
    Users can specify the number of components to fit depending upon the nature of their source.
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

    for j in range(OIII_broad):
        comp_OIII_broad, pars_OIII_broad = initialise_gaussian_line_model(
            f"OIII_broad_{string.ascii_lowercase[j]}",
            line_dict["OIII"]["broad"]["params"],
        )
        model.add_component(comp_OIII_broad)
        model.add_parameters(pars_OIII_broad)

    for k in range(OIII_wind):
        comp_OIII_wind, pars_OIII_wind = initialise_gaussian_line_model(
            f"OIII_wind_{string.ascii_lowercase[k]}",
            line_dict["OIII"]["wind"]["params"],
        )
        model.add_component(comp_OIII_wind)
        model.add_parameters(pars_OIII_wind)


# ------------------------------------------------------------------------------
# Extract parameters
# ------------------------------------------------------------------------------


def get_median_params_old(model, discard):
    """
    get median params
    """
    flat_samples = model.sampler.get_chain(discard=discard, thin=1, flat=True)
    median_params = np.median(flat_samples, axis=0)
    median_params_dict = dict(zip(model.params_variable.keys(), median_params))

    return median_params_dict


def get_median_params(model, discard=0):
    """
    Extract median and 1-sigma uncertainty for each parameter from the MCMC chain.

    Returns
    -------
    dict
        {param_name: median, param_name_sigma: 1-sigma}
    """

    flat_chain = model.sampler.get_chain(discard=discard, thin=1, flat=True)
    param_names = list(model.params_variable.keys())

    param_dict = {}

    for i, pname in enumerate(param_names):
        samples = flat_chain[:, i]
        median = np.median(samples)
        lower = np.percentile(samples, 16)
        upper = np.percentile(samples, 84)
        sigma = 0.5 * (upper - lower)
        param_dict[pname] = median
        param_dict[f"{pname}_sigma"] = sigma
        # print(pname)
        # print(param_dict[pname])
        # print(param_dict[f"{pname}_sigma"])

    return param_dict


def extract_median_params_old(prefix, param_dict):
    """
    returns a dictionary for median parameters for each component
    """
    return {k: v for k, v in param_dict.items() if k.startswith(prefix)}


def extract_median_params(prefix, median_param_dict):
    """
    Extract parameter values and 1-sigma uncertainties for a given prefix.

    Returns
    -------
    dict: {param_name: value, param_name_sigma: 1-sigma}
    """
    param_subset = {}
    for key, val in median_param_dict.items():
        if key.startswith(prefix):
            param_subset[key] = val
    return param_subset


def extract_redshift_from_OIII_Hbeta(model):
    """
    Calculate redshift from OIII or Hbeta if OIII is not well constrained
    """
    if "OIII_narrow_redsh" in model.parameters:
        return model.parameters["OIII_narrow_redsh"].value
    elif "Hbeta_broad_a_redsh" in model.parameters:
        return model.parameters["Hbeta_broad_a_redsh"].value
    else:
        return None


def update_redshift(model):
    """
    Update redshift calculated using [OIII] or Hbeta in the model
    """
    redshift = extract_redshift_from_OIII_Hbeta(model)
    for name, param in model.parameters.items():
        if name.endswith("_redsh"):
            param.value = redshift


def get_diff_median_model_and_fluxerr(model, disp_min, disp_max):
    """
    Returns (dispersion, med_model, fluxden_err, difference)
    for a given wavelength region.

    :param model: FitModel instance after running MCMC
    :param disp_min: lower bound of dispersion
    :param disp_max: upper bound of dispersion
    :return: np.ndarray with columns: dispersion, med_model, fluxden_err, difference
    """
    if model.mcmc_model_posterior is None:
        raise ValueError(
            "MCMC model posterior not available. Run plot_mcmc_result() first."
        )

    dispersion = model.spec.dispersion
    fluxden_err = model.spec.fluxden_err
    med_model, _, _ = (
        model.mcmc_model_posterior
    )  ## not taking low and upp model here. taking only median model

    mask = (dispersion >= disp_min) & (dispersion <= disp_max)

    return np.column_stack(
        [
            dispersion[mask],
            med_model[mask],
            fluxden_err[mask],
            med_model[mask] - fluxden_err[mask],
        ]
    )


# ------------------------------------------------------------------------------
# Extra functions to read files
# ------------------------------------------------------------------------------


def get_redsh_from_csv(csv_path, object_name):
    df = pd.read_csv(csv_path)
    match = df[df["wise_designation"] == object_name]

    if not match.empty:
        return float(match.iloc[0]["reference_z"])
    else:
        raise ValueError(f"Object '{object_name}' not found in {csv_path}.")


# ------------------------------------------------------------------------------
# Balmer continuum with higher order lines (up to n=50)
# ------------------------------------------------------------------------------


def Balmer_continuum_with_hol(x, z, amp_be, Te, tau_be, lambda_be, fwhm):
    rest_wavelength = x / (1 + z)

    # Blackbody continuum till 3646
    bb = BlackBody(temperature=Te * u.K)
    black_body_lambda = (
        bb(rest_wavelength * u.AA).value * c_AngstroemPerSecond / rest_wavelength**2
    )
    raw_continuum = black_body_lambda * (
        1 - np.exp(-tau_be * (rest_wavelength / lambda_be) ** 3)
    )

    # smoothening the hard cutoff that I get because of raw_conti >= 3646 = 0
    transition_width = 20.0
    smooth_cutoff = 1 / (1 + np.exp((rest_wavelength - lambda_be) / transition_width))
    continuum = raw_continuum
    continuum[continuum > lambda_be] = continuum[continuum > lambda_be] * smooth_cutoff
    continuum /= np.max(continuum)

    # high-order line emissivities with Shorey and HUmmer 1995 file from model b with 15000K and log nH =9
    df = np.loadtxt("/hshome/bbe2414/Model_test/wave_2.txt")
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


######################### plot to check burn-in in mcmc
def plot_burnin(model, save_dir=".", plot_name="burnin_all_params.pdf"):
    """
    Plots the burn-in for all variable parameters in the model.

    Parameters:
    model: A fitted Sculptor FitModel
    save_dir: Directory where the plot will be saved
    plot_name: Filename of the saved plot
    """

    ##get chain
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
        # ax.plot(np.arange(low, upp), samples[low:upp, 0, i], alpha=0.3, lw=0.3)
        ax.set_ylabel(param_name)

    axes[-1].set_xlabel("MCMC Step")
    plt.tight_layout()

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, plot_name)
    plt.savefig(save_path)
    plt.close()
    print(f"[INFO] Burn-in plot saved to: {save_path}")


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------


# Added by FGU - highly experimental! And made with Claude
# def setup_iron_template_Fe3_blend_2039_2113(prefix, fe_dict=None, median_params=None):
#     """
#     Setup the Fe III lambda-lambda 2039-2113 A blend component, following
#     Fian et al. (2021). Unlike the other setup_iron_template_* functions,
#     this does not load a static .asc template file -- it wires up the
#     fe3_blend_model function directly, since the underlying model is a
#     parametric sum of 19 fixed-relative-amplitude Gaussians rather than a
#     pre-broadened static spectrum.

#     :param prefix: Model prefix
#     :type prefix: string
#     :param fe_dict: dictionary containing parameters "amp", "fwhm_km_s",
#         "redsh", and "lam_shift" to set up the component
#     :type fe_dict: dictionary
#     :param median_params: dictionary containing median parameters from a
#         previous fit, to constrain the parameter values
#     :type median_params: dictionary
#     :return: Fe III blend component and parameters
#     """

#     comp = sccomp.FitComponent(prefix, fe3_blend_model)
#     params = comp.create_params()

#     if median_params:
#         amp = median_params[f"{prefix}_amp"]
#         fwhm = median_params[f"{prefix}_fwhm_km_s"]
#         redsh = median_params[f"{prefix}_redsh"]

#         sigma_amp = median_params.get(f"{prefix}_amp_sigma", 0.01)
#         sigma_fwhm = median_params.get(f"{prefix}_fwhm_km_s_sigma", 200)
#         sigma_redsh = median_params.get(f"{prefix}_redsh_sigma", 0.05)

#         params[f"{prefix}_amp"].value = amp
#         params[f"{prefix}_amp"].vary = True
#         params[f"{prefix}_amp"].prior = scpri.GaussianPrior(
#             f"{prefix}_amp", amp, sigma_amp
#         )

#         params[f"{prefix}_fwhm_km_s"].value = fwhm
#         params[f"{prefix}_fwhm_km_s"].vary = True
#         params[f"{prefix}_fwhm_km_s"].prior = scpri.UniformPrior(
#             f"{prefix}_fwhm_km_s", fwhm - 500, fwhm + 500
#         )

#         params[f"{prefix}_redsh"].value = redsh
#         params[f"{prefix}_redsh"].vary = True
#         params[f"{prefix}_redsh"].prior = scpri.GaussianPrior(
#             f"{prefix}_redsh", redsh, sigma_redsh
#         )

#     if fe_dict:
#         for key in params.keys():
#             par = key.split(f"{prefix}_")[1]
#             print(par)
#             if par in ["amp", "fwhm", "redsh"] and median_params:
#                 continue
#             input = fe_dict[par]

#             if "parameter" in input:
#                 params.pop(key)
#                 comp.param_mapping.update({par: input["parameter"]})
#             else:
#                 params[key].value = input["value"]
#                 params[key].vary = input["vary"]
#                 if input["vary"]:
#                     params[key].prior = input["prior"]
#     else:
#         raise ValueError("Provide either `median_params` or `fe_dict`")

#     return comp, params


# def setup_iron_template_VW01_Fe3_full(
#     prefix,
#     disp_range=None,
#     fe_dict=None,
#     median_params=None,
# ):
#     """
#     setup full FeIII template reconstructed from VW01 (2001) Table 4
#     Claude made this!
#     """
#     template = np.loadtxt(datadir + "iron_templates/" + "Fe3_full_VW01_table4.asc")
#     Fe_dispersion = template[:, 0]
#     Fe_fluxden = template[:, 1]

#     if disp_range is not None:
#         cond = (Fe_dispersion < disp_range[1]) & (Fe_dispersion > disp_range[0])
#         Fe_dispersion = Fe_dispersion[cond]
#         Fe_fluxden = Fe_fluxden[cond]

#     comp, params = initialise_Fe(
#         prefix, Fe_dispersion, Fe_fluxden, fe_dict, median_params
#     )
#     return comp, params


def setup_iron_template_BG92_Park2022_OPT(prefix, fe_dict=None, median_params=None):
    """
    setup Park et al. (2022) optical iron template
    """

    template = np.loadtxt(datadir + "iron_templates/" + "Fe_OPT_BG+Park.txt")
    Fe_dispersion = template[:, 0]
    Fe_fluxden = template[:, 1]

    comp, params = initialise_Fe(
        prefix, Fe_dispersion, Fe_fluxden, fe_dict, median_params
    )
    return comp, params


def setup_iron_template_Park2022_OPT(prefix, fe_dict=None, median_params=None):
    """
    setup Park et al. (2022) optical iron template
    """

    template = np.loadtxt(datadir + "iron_templates/" + "Park_2022.asc")
    Fe_dispersion = template[:, 0]
    Fe_fluxden = template[:, 1]

    comp, params = initialise_Fe(
        prefix, Fe_dispersion, Fe_fluxden, fe_dict, median_params
    )
    return comp, params


def setup_iron_template_Tsuz_range(
    prefix,
    disp_range=[2200, 3500],
    fe_dict=None,
    median_params=None,
):
    """
    Setup Tsuzuki et al. 2006 UV iron template
    """

    template = np.loadtxt(datadir + "iron_templates/" + "Tsuzuki06.txt")
    Fe_dispersion = template[:, 0]
    cond_Fe_MgII = (Fe_dispersion < disp_range[1]) & (Fe_dispersion > disp_range[0])
    Fe_dispersion = Fe_dispersion[cond_Fe_MgII]
    Fe_fluxden = template[:, 1][cond_Fe_MgII] * 1.0e17

    comp, params = initialise_Fe(
        prefix, Fe_dispersion, Fe_fluxden, fe_dict, median_params
    )
    return comp, params


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
        sigma = (fwhm / c_kms) * _HOL_TRANSITIONS / _SIGMA_TO_FWHM
        diff2 = (x_rest[:, None] - _HOL_TRANSITIONS[None, :]) ** 2
        profile = np.exp(-0.5 * diff2 / sigma[None, :] ** 2)
        flux_density = np.sum(_HOL_EMISS[None, :] * profile / sigma[None, :], axis=1)
        edge_hol = np.interp(lambda_be, x_rest, flux_density)
        edge_continuum = np.interp(lambda_be, x_rest, continuum_shape)
        flux_density *= 0.3 * edge_continuum / edge_hol
        # see normalization note from earlier
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
        # see normalization note from earlier
        return flux_density

    def __call__(self, x, z, amp_be, Te, tau_be, lambda_be, fwhm):
        self._update_cache(x, z)
        continuum_shape = self._continuum_shape(Te, tau_be, lambda_be)
        hol_shape = self._hol_shape(fwhm, lambda_be, continuum_shape)
        return amp_be * (continuum_shape + hol_shape) * 1e-8


def initialise_balmer_with_hol(prefix, balmer_dict=None, median_params=None):
    """
    Initialise Balmer continuum component without higher order lines.
    Function requires a dictionary that contains necessary parameters to set up
    a Balmer template such as "amp_be", "redshift", "Te", "tau_be", "lambda_be".
    In this dictionary, each component should have "initial_value", "vary" and "prior"
    which provides information on initial value of the parameter, whether the parameter should be varied, and
    prior to be used to contrain the parameter, respectively.
    :param prefix: Model prefix
    :type prefix: string
    :param balmer_dict: dictionary containing parameters to setup the Balmer continuum model
    :type Balmer_dict: dictionary
    :param median_params: dictionary containing median parameters to setup the Balmer continuum model
    :type median_params: dictionary

    :return: Balmer continuuum model component and parameters
    """
    comp = sccomp.FitComponent(prefix, BalmerContinuumWithHOL())
    # qso.balmer_continuum_model)
    params = comp.create_params()

    if median_params:
        amp_be = median_params[f"{prefix}_amp_be"]

        params[f"{prefix}_amp_be"].value = amp_be
        params[f"{prefix}_amp_be"].vary = True
        params[f"{prefix}_amp_be"].prior = scpri.GaussianPrior(
            f"{prefix}_amp_be", amp_be, 0.05
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
        raise ValueError("Provide either `ml_params` or `powerlaw_dict`")

    return comp, params


# ================================================================
# ================================================================
# ================================================================


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


# ================================================================


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

    # FIXED: rest_wave_cm * obs_wave_cm (equivalently rest_wave_cm**2 * (1+z_abs)),
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
    Returns TRANSMISSION (0-1), not additive flux. Marked as an
    absorption component via is_absorption=True when registered, so
    FitModel.eval multiplies this into the summed emission instead of
    adding it.
    """
    return _absorption_transmission(
        x, z_abs, log_N, b_kms, rest_wave, osc_strength, damping_gamma
    )


def lya_absorption_model(x, z_abs, log_N, b_kms, **kwargs):
    """Convenience wrapper fixing the Lya transition's atomic data."""
    return absorption_transmission_model(
        x, z_abs, log_N, b_kms, LYA_REST_WAVE, LYA_OSC_STRENGTH, LYA_DAMPING_GAMMA
    )
