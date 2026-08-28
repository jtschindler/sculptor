import os
import pathlib
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from tqdm import tqdm

sys.path.append("/Users/jtschindler/Projects/2205_JWST_Light_Echo/jwst_lightecho/")

import extinction
from astropy import constants as const
from astropy import units as u
from astropy.cosmology import FlatLambdaCDM
from astropy.io import fits
from astropy.table import QTable
from atelier import plot_defaults
from paper_seyfert import utils
from speconed import speconed as sod

from sculptor import analysis as ana
from sculptor import component as sccomp
from sculptor import model as scmod
from sculptor import parameter as scpar
from sculptor import prior as scpri
from sculptor.dep_v1 import specanalysis as scana

# from sculptor import specmodel
# from sculptor import specfit as scfit
from sculptor_extensions import qso

c_light = (const.c.to("km/s")).value


black = (0, 0, 0)
orange = (230 / 255.0, 159 / 255.0, 0)
blue = (86 / 255.0, 180 / 255.0, 233 / 255.0)
green = (0, 158 / 255.0, 115 / 255.0)
yellow = (240 / 255.0, 228 / 255.0, 66 / 255.0)
dblue = (0, 114 / 255.0, 178 / 255.0)
vermillion = (213 / 255.0, 94 / 255.0, 0)
purple = (181 / 255.0, 107 / 255.0, 148 / 255.0)

# GeoDataViz Colors
gdv_div_a = [
    "#009392",
    "#39b185",
    "#9ccb86",
    "#e9e29c",
    "#eeb479",
    "#e88471",
    "#cf597e",
]
gdv_div_b = [
    "#045275",
    "#089099",
    "#7ccba2",
    "#fcde9c",
    "#f0746e",
    "#dc3977",
    "#7c1d6f",
]
gdv_div_d = [
    "#008042",
    "#6fa253",
    "#b7c370",
    "#fce498",
    "#d78287",
    "#bf5688",
    "#7c1d6f",
]

# http://tsitsul.in/blog/coloropt/
xgfs_normal6 = [
    "#4053d3",
    "#ddb310",
    "#b51d14",
    "#00beff",
    "#fb49b0",
    "#00b25d",
    "#cacaca",
]


def get_L5100_from_Hb_lum(lum_Hb):
    """Use the Greene & Ho 2005 relation to estimate the
    continuum luminosity at 5100A from the Hb line luminosity

    :param lum_Hb:
    :return:
    """

    log_term = np.log10(lum_Hb / 1.425 / 1e42)

    return 1e44 * 10 ** (log_term / 1.133)


def get_Hb_lum_from_L5100(lum_5100):

    return 1.425 * 1e42 * (lum_5100 / 1e44) ** 1.133


def lband_from_lbol(lbol, c1, k1, c2, k2):

    term_a = (lbol / const.L_sun.to(u.erg / u.s)) ** k1
    term_b = (lbol / const.L_sun.to(u.erg / u.s)) ** k2

    lbol_over_lband = c1 * term_a + c2 * term_b

    return lbol / lbol_over_lband


def get_wave_grid(wave_min, wave_max, dvpix):
    """
    Adopted from enigma-igm/qso_fitting/data/utils.py

    Get a wavelength grid with uniform velocity separation. Wavelengths
    can either be in rest-frame or observed frame

    Args:
        wave_min (float):
            Minimum wavelength
        wave_max (float):
            Maximum wavelength
        dvpix (float):
            Pixel scale in velocity units (km/s)

    Returns:
        wave_grid (ndarray):
            Wavelength grid, shape = (npix)
    """

    # Create the observed wavelength grid with uniform velocity scale
    dloglam = dvpix / c_light / np.log(10.0)
    loglam_min = np.log10(wave_min)
    loglam_max = np.log10(wave_max)
    npix = int(np.round((loglam_max - loglam_min) / dloglam))
    loglam = loglam_min + np.arange(npix) * dloglam
    wave_grid = np.power(10.0, loglam)
    return wave_grid


# ------------------------------------------------------------------------------
# Functions to fit the AGN model
# ------------------------------------------------------------------------------


def preliminary_fit(
    spec,
    redshift,
    nwalkers=40,
    nsteps=4000,
    discard=2000,
    save_dir="preliminary_fit_mcmc",
    refit=False,
):
    """Preliminary fit to the AGN spectrum to determine the redshift.

    :param spec:
    :param redshift:
    :param nwalkers:
    :param nsteps:
    :param discard:
    :param save_dir:
    :param refit:
    :return:
    """

    # Generate output directory and generate directory if it does not exist
    dir_path = pathlib.Path(f"../../data/{save_dir}")
    dir_path.mkdir(parents=True, exist_ok=True)

    chain_path = os.path.join(dir_path, "flat_chain.csv")
    result_path = os.path.join(dir_path, "fit_results.escv")

    if (not os.path.exists(chain_path) and not os.path.exists(result_path)) or refit:
        # Initialize the empty FitModel
        model = scmod.FitModel(spectrum=spec)

        # Define wavelength ranges for the fit
        model.add_wavelength_range_to_fit_mask(4750 * (1 + redshift), 51800)
        model.add_wavelength_range_to_fit_mask(36700, 38400)
        model.add_wavelength_range_to_fit_mask(32500, 35500)

        # Remove the wavelength range around the Hd and 3968.58 [NeIII] lines
        model.remove_wavelength_range_from_fit_mask(
            4070 * (1 + redshift), 4148 * (1 + redshift)
        )
        model.remove_wavelength_range_from_fit_mask(
            3956 * (1 + redshift), 4015 * (1 + redshift)
        )

        # Define the continuum model
        comp_pl = sccomp.FitComponent("pl", ana.power_law_at2500)
        pars_pl = comp_pl.create_params()

        pars_pl["pl_amp"].prior = scpri.UniformPrior("amp", 0, 0.1)
        pars_pl["pl_slope"].prior = scpri.UniformPrior("slope", -2, 1)
        pars_pl["pl_redsh"].value = redshift
        pars_pl["pl_redsh"].vary = False

        # Global narrow line redshift parameter
        par_redsh_nl = scpar.FitParameter("redsh_nl", value=redshift, vary=True)
        par_redsh_nl.prior = scpri.UniformPrior("redsh_nl", 7.18, 7.30)

        global_pars = {"redsh_nl": par_redsh_nl}

        # [OIII] 5007
        comp_o3_a = sccomp.FitComponent("O3_5007", ana.line_model_gaussian)
        pars_o3_a = comp_o3_a.create_params()

        pars_o3_a["O3_5007_flux"].value = 0.0006
        pars_o3_a["O3_5007_flux"].prior = scpri.UniformPrior("O3_5007_flux", 0, 1)
        pars_o3_a["O3_5007_cen"].value = 5008.24
        pars_o3_a["O3_5007_cen"].vary = False
        pars_o3_a.pop("O3_5007_redsh")
        pars_o3_a["O3_5007_fwhm_km_s"].value = 1200
        pars_o3_a["O3_5007_fwhm_km_s"].prior = scpri.UniformPrior(
            "O3_5007_fwhm_km_s", 300, 3000
        )

        comp_o3_a.param_mapping.update({"redsh": "redsh_nl"})

        # [OIII] 4960
        comp_o3_b = sccomp.FitComponent("O3_4960", ana.line_model_gaussian)
        pars_o3_b = comp_o3_b.create_params()

        pars_o3_b["O3_4960_flux"].value = 0.0006
        pars_o3_b["O3_4960_flux"].prior = scpri.UniformPrior("O3_4960_flux", 0, 1)
        pars_o3_b["O3_4960_cen"].value = 4960.30
        pars_o3_b["O3_4960_cen"].vary = False
        pars_o3_b.pop("O3_4960_redsh")
        pars_o3_b["O3_4960_fwhm_km_s"].value = 1200
        pars_o3_b["O3_4960_fwhm_km_s"].prior = scpri.UniformPrior(
            "O3_4960_fwhm_km_s", 300, 3000
        )

        comp_o3_b.param_mapping.update({"redsh": "redsh_nl"})

        # Hb
        comp_hb = sccomp.FitComponent("Hb", ana.line_model_gaussian)
        pars_hb = comp_hb.create_params()

        pars_hb["Hb_flux"].value = 0.0001
        pars_hb["Hb_flux"].prior = scpri.UniformPrior("Hb_flux", 0, 1)
        pars_hb["Hb_cen"].value = 4862.68
        pars_hb["Hb_cen"].vary = False
        pars_hb["Hb_redsh"].value = redshift
        pars_hb["Hb_redsh"].prior = scpri.UniformPrior(
            "Hb_redsh", redshift - 0.05, redshift + 0.05
        )
        pars_hb["Hb_fwhm_km_s"].value = 1200
        pars_hb["Hb_fwhm_km_s"].prior = scpri.UniformPrior("Hb_fwhm_km_s", 300, 10000)

        # # Reading in the iron template
        # iron_template_path = '/Users/jtschindler/Software/sculptor/sculptor/data/iron_templates/Fe_OPT_BR92_linear.txt'
        # iron_template = np.genfromtxt(iron_template_path, names=['wave', 'flux'])
        # flux = iron_template['flux']
        # nflux = flux / np.median(flux)
        # wave = iron_template['wave']
        #
        # # Adding an iron template
        # comp_fe = sccomp.FitComponent('Fe', fit.template_model)
        # pars_fe = comp_fe.create_params()
        #
        # pars_fe['Fe_amp'].value = 0.0001
        # pars_fe['Fe_amp'].prior = scpri.UniformPrior('Fe_amp', 0, 1)
        # pars_fe['Fe_redsh'].value = redshift
        # pars_fe['Fe_redsh'].vary = False
        # pars_fe['Fe_fwhm'].value = 1200
        # pars_fe['Fe_fwhm'].prior = scpri.UniformPrior('Fe_fwhm', 901, 10000)
        # pars_fe['Fe_intr_fwhm'].value = 900
        # pars_fe['Fe_intr_fwhm'].vary = False
        # pars_fe['Fe_templ_fluxden'].value = nflux
        # pars_fe['Fe_templ_fluxden'].vary = False
        # pars_fe['Fe_templ_disp'].value = wave
        # pars_fe['Fe_templ_disp'].vary = False
        # pars_fe['Fe_templ_disp_unit_str'].value = 'AA'
        # pars_fe['Fe_templ_disp_unit_str'].vary = False
        # pars_fe['Fe_templ_fluxden_unit_str'].value = 'erg/s/cm/cm/AA'
        # pars_fe['Fe_templ_fluxden_unit_str'].vary = False

        # Consolidate model components
        model.components = [
            comp_pl,
            comp_o3_a,
            comp_o3_b,
            comp_hb,
        ]
        # Consolidate model parameters
        params = {}
        for pars in [pars_pl, pars_o3_a, pars_o3_b, pars_hb, global_pars]:
            params.update(pars)
        model.parameters = params

        # Run the MCMC
        model.run_emcee(nsteps, nwalkers)

        flat_samples = model.sampler.get_chain(discard=discard, thin=1, flat=True)

        model.plot_mcmc_result(discard=discard, save_dir=dir_path, save=True)
        model.plot_mcmc_result(
            discard=discard,
            save_dir=dir_path,
            save=True,
            xlim=[38000, 45000],
            save_name="fit_result_zoom.pdf",
            show_components=True,
        )
        model.plot_posterior_corner(save_dir=dir_path, discard=discard, save=True)

        df = pd.DataFrame(data=flat_samples, columns=model.params_variable.keys())

        df.to_csv(chain_path, index=False)

        # ----------------------------------------------------------------------
        # Analyze the preliminary fit
        # ----------------------------------------------------------------------

        # Analyse the [OIII] lines
        continuum_listdict = {
            "feature_name": ["pl"],
            "components": [comp_pl],
            "rest_frame_wavelengths": [5100],
        }

        emission_feature_listdict = [
            {
                "feature_name": "O3_5007",
                "components": [comp_o3_a],
                "rest_frame_wavelength": 5008.24,
            },
            {
                "feature_name": "O3_4960",
                "components": [comp_o3_b],
                "rest_frame_wavelength": 4960.30,
            },
        ]

        cosmology = FlatLambdaCDM(70, 0.3)

        rt = ana._mcmc_analyze(
            model,
            flat_samples,
            continuum_listdict,
            emission_feature_listdict,
            redshift,
            cosmology,
            dispersion=spec.dispersion,
            dispersion_unit=spec.dispersion_unit,
            fluxden_unit=spec.fluxden_unit,
        )
    else:
        rt = QTable.read(result_path, format="ascii.ecsv")
        df = pd.read_csv(chain_path)

    rt.write(
        result_path.format(save_suffix),
        format="ascii.ecsv",
        overwrite=True,
        delimiter=",",
    )

    result_dict = {}
    p = np.percentile(rt["O3_5007_FWHM"], [15.87, 50, 84.13])
    result_dict["O3_5007_FWHM"] = [p[1], p[0] - p[1], p[2] - p[1]]
    p = np.percentile(rt["O3_4960_FWHM"], [15.87, 50, 84.13])
    result_dict["O3_4960_FWHM"] = [p[1], p[0] - p[1], p[2] - p[1]]

    p = np.percentile(df["redsh_nl"], [15.87, 50, 84.13])
    result_dict["redsh"] = [p[1], p[0] - p[1], p[2] - p[1]]

    return result_dict


def agn_model(spec, redshift, fit_range, flux_factor=1, resolution=None):
    """Define the AGN model for the fit.

    The emission lines are:
    - [OIII] 5008.24 (redsh, fwhm coupled)
    - [OIII] 4960.30 (redsh, fwhm coupled)
    - Hb 4862.68 (redsh, fwhm coupled)
    - Hg 4341.68 (redsh, fwhm coupled)
    - [OIII] 4364.44 (redsh, fwhm coupled)
    - Hb broad 4862.68 (redsh_br, fwhm_br coupled)
    - Hg broad 4341.68 (redsh_br, fwhm_br coupled)
    - NeIII 3869.85 (fwhm restr. 300-10000, redshift restricted 7.27+-0.06)
    - HeII 4687.02 (fwhm restr. 300-10000, redshift coupled)

    :param spec:
    :param redshift:
    :param fit_range:
    :param flux_factor:
    :return:
    """

    # Initialize the empty FitModel
    model = scmod.FitModel(spectrum=spec)

    # Define wavelength ranges for the fit
    model.add_wavelength_range_to_fit_mask(fit_range[0], fit_range[1])

    # Add the resolution to the model
    model.resolution = resolution

    # Remove the wavelength range around the Hd and 3968.58 [NeIII] lines
    # model.remove_wavelength_range_from_fit_mask(4070 * (1 + redshift),
    #                                             4148 * (1 + redshift))
    model.remove_wavelength_range_from_fit_mask(
        3956 * (1 + redshift), 4015 * (1 + redshift)
    )
    model.remove_wavelength_range_from_fit_mask(
        4650 * (1 + redshift), 4725 * (1 + redshift)
    )

    # ------------------------------------
    # Define the continuum model
    # ------------------------------------
    comp_pl = sccomp.FitComponent("pl", ana.power_law_at2500)
    pars_pl = comp_pl.create_params()

    pars_pl["pl_amp"].prior = scpri.UniformPrior("amp", 0, 0.002)
    pars_pl["pl_slope"].prior = scpri.UniformPrior("slope", -2, 1)
    pars_pl["pl_redsh"].value = redshift
    pars_pl["pl_redsh"].vary = False

    # ------------------------------------
    # Define the narrow line emission model
    # ------------------------------------

    # Global narrow line redshift parameter
    par_redsh_nl = scpar.FitParameter("redsh_nl", value=redshift, vary=True)
    par_redsh_nl.prior = scpri.UniformPrior("redsh_nl", 7.25, 7.28)

    par_fwhm_nl = scpar.FitParameter("fwhm_nl", value=600, vary=True)
    par_fwhm_nl.prior = scpri.UniformPrior("fwhm_nl", 150, 1500)

    par_redsh_bl = scpar.FitParameter("redsh_bl", value=redshift, vary=True)
    par_redsh_bl.prior = scpri.UniformPrior("redsh_bl", 7.22, 7.28)

    par_fwhm_bl = scpar.FitParameter("fwhm_bl", value=3000, vary=True)
    par_fwhm_bl.prior = scpri.UniformPrior("fwhm_bl", 1200, 8000)

    global_pars = {
        "redsh_nl": par_redsh_nl,
        "fwhm_nl": par_fwhm_nl,
        "redsh_bl": par_redsh_bl,
        "fwhm_bl": par_fwhm_bl,
    }

    # ------------------------------------
    # [OIII] 5007
    comp_o3_a = sccomp.FitComponent("O3_5007", ana.line_model_gaussian)
    pars_o3_a = comp_o3_a.create_params()

    pars_o3_a["O3_5007_flux"].value = 0.0006 * flux_factor
    pars_o3_a["O3_5007_flux"].prior = scpri.UniformPrior("O3_5007_flux", 0, 1)
    pars_o3_a["O3_5007_cen"].value = 5008.24
    pars_o3_a["O3_5007_cen"].vary = False
    pars_o3_a.pop("O3_5007_redsh")
    pars_o3_a.pop("O3_5007_fwhm_km_s")

    comp_o3_a.param_mapping.update({"redsh": "redsh_nl", "fwhm_km_s": "fwhm_nl"})

    # ------------------------------------
    # [OIII] 4960
    comp_o3_b = sccomp.FitComponent("O3_4960", ana.line_model_gaussian)
    pars_o3_b = comp_o3_b.create_params()

    pars_o3_b["O3_4960_flux"].value = 0.0006 * flux_factor
    pars_o3_b["O3_4960_flux"].prior = scpri.UniformPrior("O3_4960_flux", 0, 1)
    pars_o3_b["O3_4960_cen"].value = 4960.30
    pars_o3_b["O3_4960_cen"].vary = False
    pars_o3_b.pop("O3_4960_redsh")
    pars_o3_b.pop("O3_4960_fwhm_km_s")

    comp_o3_b.param_mapping.update({"redsh": "redsh_nl", "fwhm_km_s": "fwhm_nl"})

    # ------------------------------------
    # Hb narrow
    comp_hb_nl = sccomp.FitComponent("Hb", ana.line_model_gaussian)
    pars_hb_nl = comp_hb_nl.create_params()

    pars_hb_nl["Hb_flux"].value = 0.0001 * flux_factor
    pars_hb_nl["Hb_flux"].prior = scpri.UniformPrior("Hb_flux", 0, 1.5)
    pars_hb_nl["Hb_cen"].value = 4862.68
    pars_hb_nl["Hb_cen"].vary = False

    pars_hb_nl.pop("Hb_redsh")
    pars_hb_nl.pop("Hb_fwhm_km_s")

    comp_hb_nl.param_mapping.update({"redsh": "redsh_nl", "fwhm_km_s": "fwhm_nl"})

    # ------------------------------------
    # Hg narrow
    comp_hg_nl = sccomp.FitComponent("Hg", ana.line_model_gaussian)
    pars_hg_nl = comp_hg_nl.create_params()

    pars_hg_nl["Hg_flux"].value = 0.0001 * flux_factor
    pars_hg_nl["Hg_flux"].prior = scpri.UniformPrior("Hg_flux", 0, 1.0)
    pars_hg_nl["Hg_cen"].value = 4341.68
    pars_hg_nl["Hg_cen"].vary = False

    pars_hg_nl.pop("Hg_redsh")
    pars_hg_nl.pop("Hg_fwhm_km_s")

    comp_hg_nl.param_mapping.update({"redsh": "redsh_nl", "fwhm_km_s": "fwhm_nl"})

    # ------------------------------------
    # Hg narrow (fully coupled FWHM and redshift)
    comp_hd_nl = sccomp.FitComponent("Hd", ana.line_model_gaussian)
    pars_hd_nl = comp_hd_nl.create_params()

    pars_hd_nl["Hd_flux"].value = 0.0001 * flux_factor
    pars_hd_nl["Hd_flux"].prior = scpri.UniformPrior("Hd_flux", 0, 1.0)
    pars_hd_nl["Hd_cen"].value = 4102.89
    pars_hd_nl["Hd_cen"].vary = False

    pars_hd_nl.pop("Hd_redsh")
    pars_hd_nl.pop("Hd_fwhm_km_s")

    comp_hd_nl.param_mapping.update({"redsh": "redsh_nl", "fwhm_km_s": "fwhm_nl"})

    # ------------------------------------
    # O3 4364
    comp_o3_c = sccomp.FitComponent("O3_4364", ana.line_model_gaussian)
    pars_o3_c = comp_o3_c.create_params()

    pars_o3_c["O3_4364_flux"].value = 0.0001 * flux_factor
    pars_o3_c["O3_4364_flux"].prior = scpri.UniformPrior("O3_4364_flux", 0, 1.0)
    pars_o3_c["O3_4364_cen"].value = 4364.44
    pars_o3_c["O3_4364_cen"].vary = False

    pars_o3_c.pop("O3_4364_redsh")
    pars_o3_c.pop("O3_4364_fwhm_km_s")

    comp_o3_c.param_mapping.update({"redsh": "redsh_nl", "fwhm_km_s": "fwhm_nl"})

    # ------------------------------------
    # Hb broad
    comp_hb_br = sccomp.FitComponent("Hb_br", ana.line_model_gaussian)
    pars_hb_br = comp_hb_br.create_params()

    pars_hb_br["Hb_br_flux"].value = 0.0001 * flux_factor
    pars_hb_br["Hb_br_flux"].prior = scpri.UniformPrior("Hb_br_flux", 0, 1.5)
    pars_hb_br["Hb_br_cen"].value = 4862.68
    pars_hb_br["Hb_br_cen"].vary = False

    pars_hb_br.pop("Hb_br_redsh")
    pars_hb_br.pop("Hb_br_fwhm_km_s")

    comp_hb_br.param_mapping.update({"redsh": "redsh_bl", "fwhm_km_s": "fwhm_bl"})

    # ------------------------------------
    # Hg broad
    comp_hg_br = sccomp.FitComponent("Hg_br", ana.line_model_gaussian)
    pars_hg_br = comp_hg_br.create_params()

    pars_hg_br["Hg_br_flux"].value = 0.0001 * flux_factor
    pars_hg_br["Hg_br_flux"].prior = scpri.UniformPrior("Hg_br_flux", 0, 1.5)
    pars_hg_br["Hg_br_cen"].value = 4341.68
    pars_hg_br["Hg_br_cen"].vary = False

    pars_hg_br.pop("Hg_br_redsh")
    pars_hg_br.pop("Hg_br_fwhm_km_s")

    comp_hg_br.param_mapping.update({"redsh": "redsh_bl", "fwhm_km_s": "fwhm_bl"})

    # ------------------------------------
    # Ne III 3869
    comp_ne3 = sccomp.FitComponent("Ne3_a", ana.line_model_gaussian)
    pars_ne3 = comp_ne3.create_params()

    pars_ne3["Ne3_a_flux"].value = 0.0001 * flux_factor
    pars_ne3["Ne3_a_flux"].prior = scpri.UniformPrior("Ne3_a_flux", 0, 1.5)
    pars_ne3["Ne3_a_cen"].value = 3869.85
    pars_ne3["Ne3_a_cen"].vary = False
    pars_ne3["Ne3_a_redsh"].value = redshift
    pars_ne3["Ne3_a_redsh"].prior = scpri.UniformPrior(
        "Ne3_a_redsh", redshift - 0.05, redshift + 0.05
    )
    pars_ne3["Ne3_a_fwhm_km_s"].value = 1200
    pars_ne3["Ne3_a_fwhm_km_s"].prior = scpri.UniformPrior("Ne3_a_fwhm_km_s", 300, 5000)

    # # ------------------------------------
    # # He II
    # comp_he2 = sccomp.FitComponent('He2', ana.line_model_gaussian)
    # pars_he2 = comp_he2.create_params()
    #
    # pars_he2['He2_flux'].value = 0.0001 * flux_factor
    # pars_he2['He2_flux'].prior = scpri.UniformPrior('He2_flux', 0, 1.5)
    # pars_he2['He2_cen'].value = 4687.02
    # pars_he2['He2_cen'].vary = False
    # pars_he2['He2_fwhm_km_s'].value = 1200
    # pars_he2['He2_fwhm_km_s'].prior = scpri.UniformPrior('He2_fwhm_km_s', 200, 4000)
    # pars_he2.pop('He2_redsh')
    #
    # comp_he2.param_mapping.update({'redsh': 'redsh_nl'})

    # # Reading in the iron template
    # iron_template_path = '/Users/jtschindler/Software/sculptor/sculptor/data/iron_templates/Fe_OPT_BR92_linear.txt'
    # iron_template = np.genfromtxt(iron_template_path, names=['wave', 'flux'])
    # flux = iron_template['flux']
    # nflux = flux / np.median(flux)
    # wave = iron_template['wave']
    #
    # # Adding an iron template
    # comp_fe = sccomp.FitComponent('Fe', fit.template_model)
    # pars_fe = comp_fe.create_params()
    #
    # pars_fe['Fe_amp'].value = 0.0001
    # pars_fe['Fe_amp'].prior = scpri.UniformPrior('Fe_amp', 0, 1)
    # pars_fe['Fe_redsh'].value = redshift
    # pars_fe['Fe_redsh'].vary = False
    # pars_fe['Fe_fwhm'].value = 3000
    # pars_fe['Fe_fwhm'].vary = False
    # pars_fe['Fe_intr_fwhm'].value = 900
    # pars_fe['Fe_intr_fwhm'].vary = False
    # pars_fe['Fe_templ_fluxden'].value = nflux
    # pars_fe['Fe_templ_fluxden'].vary = False
    # pars_fe['Fe_templ_disp'].value = wave
    # pars_fe['Fe_templ_disp'].vary = False
    # pars_fe['Fe_templ_disp_unit_str'].value = 'AA'
    # pars_fe['Fe_templ_disp_unit_str'].vary = False
    # pars_fe['Fe_templ_fluxden_unit_str'].value = 'erg/s/cm/cm/AA'
    # pars_fe['Fe_templ_fluxden_unit_str'].vary = False

    # Consolidate model components
    model.components = [
        comp_pl,
        comp_o3_a,
        comp_o3_b,
        comp_hb_nl,
        comp_hg_nl,
        comp_o3_c,
        comp_hb_br,
        comp_hg_br,
        comp_ne3,
        # comp_he2,
        comp_hd_nl,
    ]
    # Consolidate model parameters
    params = {}
    for pars in [
        pars_pl,
        pars_o3_a,
        pars_o3_b,
        global_pars,
        pars_hb_nl,
        pars_hg_nl,
        pars_o3_c,
        pars_hb_br,
        pars_hg_br,
        pars_ne3,
        # pars_he2,
        pars_hd_nl,
    ]:
        params.update(pars)
    model.parameters = params

    return model


def fit_spectrum(
    spec,
    redshift,
    nsteps=1000,
    discard=500,
    nwalkers=40,
    flux_factor=1,
    fit_range=[31000, 43000],
    dered=False,
    save_dir=".",
    refit=False,
    resolution=None,
):
    """Fit the spectrum with a model consisting of a power law continuum
    and a number of emission lines.

    :param spec:
    :param redshift:
    :param narrow_line_fwhm:
    :param save_suffix:
    :param flux_factor:
    :param fit_range:
    :param dered:
    :return:
    """

    # Generate the path to the output directory
    dir_path = pathlib.Path(f"../../data/{save_dir}")
    dir_path.mkdir(parents=True, exist_ok=True)

    chain_path = os.path.join(dir_path, "flat_chain.csv")
    result_path = os.path.join(dir_path, "fit_results.escv")

    if (not os.path.exists(chain_path) and not os.path.exists(result_path)) or refit:
        model = agn_model(spec, redshift, fit_range, flux_factor, resolution=resolution)

        (
            comp_pl,
            comp_o3_a,
            comp_o3_b,
            comp_hb_nl,
            comp_hg_nl,
            comp_o3_c,
            comp_hb_br,
            comp_hg_br,
            comp_ne3,
            # comp_he2,
            comp_hd_nl,
        ) = model.components

        # Run the MCMC
        model.run_emcee(nsteps, nwalkers)

        flat_samples = model.sampler.get_chain(discard=discard, thin=1, flat=True)
        pv = list(model.params_variable.keys())

        for idx, name in enumerate(pv):
            plt.figure()
            plt.hist(flat_samples[:, idx], bins=50, color="k", histtype="step", lw=2)
            plt.xlabel(name)
            plt.ylabel("Posterior")

            plt.savefig(os.path.join(dir_path, f"{name}_posterior.pdf"))

        model.plot_mcmc_result(
            discard=discard,
            save=True,
            save_dir=dir_path,
            xlim=fit_range,
            show_components=False,
            save_data=True,
        )

        model.plot_posterior_corner(
            discard=discard, save=True, save_dir=dir_path, save_name="corner_plot.pdf"
        )

        # --------------------------
        # Analyze the fit results
        # --------------------------
        params_values = flat_samples[0, :]
        dispersion = get_wave_grid(fit_range[0], fit_range[1], 100)

        cont_dict = {
            "feature_name": "pl",
            "components": [comp_pl],
            "rest_frame_wavelengths": [5100],
        }

        cosmology = FlatLambdaCDM(H0=70, Om0=0.3)

        emfeat_dictlist = [
            {
                "feature_name": "Hb_broad",
                "components": [comp_hb_br],
                "rest_frame_wavelength": 4862.68,
            },
            {
                "feature_name": "Hb_narrow",
                "components": [comp_hb_nl],
                "rest_frame_wavelength": 4862.68,
            },
            {
                "feature_name": "Hb_total",
                "components": [comp_hb_nl, comp_hb_br],
                "rest_frame_wavelength": 4862.68,
            },
            {
                "feature_name": "O3_5007",
                "components": [comp_o3_a],
                "rest_frame_wavelength": 5008.24,
            },
            {
                "feature_name": "O3_4960",
                "components": [comp_o3_b],
                "rest_frame_wavelength": 4960.30,
            },
            {
                "feature_name": "Ne3_a",
                "components": [comp_ne3],
                "rest_frame_wavelength": 3869.85,
            },
            # {'feature_name': 'He2',
            #  'components': [comp_he2],
            #  'rest_frame_wavelength': 4687.02},
            {
                "feature_name": "Hg_total",
                "components": [comp_hg_nl, comp_hg_br],
                "rest_frame_wavelength": 4341.68,
            },
            {
                "feature_name": "Hg_narrow",
                "components": [comp_hg_nl],
                "rest_frame_wavelength": 4341.68,
            },
            {
                "feature_name": "Hg_broad",
                "components": [comp_hg_br],
                "rest_frame_wavelength": 4341.68,
            },
            {
                "feature_name": "Hd_narrow",
                "components": [comp_hd_nl],
                "rest_frame_wavelength": 4102.89,
            },
        ]

        rt = ana._mcmc_analyze(
            model,
            flat_samples,
            cont_dict,
            emfeat_dictlist,
            redshift,
            cosmology,
            dispersion=dispersion,
            dispersion_unit=spec.dispersion_unit,
            fluxden_unit=spec.fluxden_unit,
        )

        rt.write(result_path, format="ascii.ecsv", overwrite=True, delimiter=",")

        df = pd.DataFrame(data=flat_samples, columns=model.params_variable.keys())

        df.to_csv(chain_path, index=False)

        # Get the line SNR
        line_dict = {
            "Hd": {
                "cen": 4102.89,
                "fwhm": df["fwhm_bl"].values,
                "l_min": None,
                "l_max": None,
            },
            "Hb": {
                "cen": 4862.68,
                "fwhm": df["fwhm_bl"].values,
                "l_min": None,
                "l_max": None,
            },
            "Hg": {
                "cen": 4341.68,
                "fwhm": df["fwhm_bl"].values,
                "l_min": None,
                "l_max": None,
            },
            "NeIII": {
                "cen": 3869.85,
                "fwhm": df["Ne3_a_fwhm_km_s"].values,
                "l_min": None,
                "l_max": None,
            },
        }

        line_snr, peak_snr = ana.get_line_snr(
            model, flat_samples, comp_pl, redshift, line_dict
        )

        df_snr = pd.DataFrame(line_snr.T, columns=list(line_dict.keys()))
        df_snr.to_csv(os.path.join(dir_path, "line_snr.csv"), index=False)
        df_snr = pd.DataFrame(peak_snr.T, columns=list(line_dict.keys()))
        df_snr.to_csv(os.path.join(dir_path, "peak_snr.csv"), index=False)

    else:
        rt = QTable.read(result_path, format="ascii.ecsv")
        df = pd.read_csv(chain_path)

        model = agn_model(spec, redshift, fit_range, flux_factor, resolution=resolution)

        model.get_params_to_sample()

    # Plot the spectral fit for the paper
    plot_spectral_fit_zoomin(
        model,
        discard=discard,
        flat_chain=df.values,
        save_dir=dir_path,
        resolution=resolution,
        save=True,
    )
    plot_spectral_fit_zoomin(
        model,
        discard=discard,
        flat_chain=df.values,
        resolution=resolution,
        save_dir=dir_path,
        mode="presentation",
        dark=True,
        save_name="spectral_fit_zoomin_pres.pdf",
        save=True,
    )

    # --------------------------------------------------------------------------
    # Analysis of the spectrum
    # --------------------------------------------------------------------------

    # Read in the dispersion information
    hdu = fits.open("../../data/jwst_nirspec_prism_disp.fits")
    disp = hdu[1].data["WAVELENGTH"]
    res_arr = hdu[1].data["R"]
    c_km_s = const.c.to("km/s")

    # Analyze the fit results for the AGN spectrum
    lum_5100 = rt["5100_Lwav"]

    O3_fwhm = rt["O3_5007_FWHM"]

    hb_fwhm = rt["Hb_broad_FWHM"]
    fwhm = rt["Hb_broad_FWHM"]

    hg_fwhm = rt["Hg_broad_FWHM"]

    m_bh = qso.se_bhmass_hbeta_vp06(fwhm, lum_5100)
    l_bol = qso.calc_bolometric_luminosity(lum_5100, 5100 * u.AA)
    edd_r = qso.calc_eddington_ratio(l_bol[0], m_bh[0])

    m_bh_p12 = qso.se_bhmass_hbeta_park12(fwhm, lum_5100)
    edd_r_p12 = qso.calc_eddington_ratio(l_bol[0], m_bh_p12[0])

    # TODO: This should be the total Hb line luminosity
    hb_lwav = rt["Hb_total_lum"]
    L_Hb = np.nanpercentile(hb_lwav, q=[15.87, 50, 84.13])
    L5100_from_Hb = get_L5100_from_Hb_lum(
        np.nanpercentile(hb_lwav.value, q=[15.87, 50, 84.13])
    )
    lbol_from_Hb = 9.26 * L5100_from_Hb

    m_bh_gh05_LHb = qso.se_bhmass_gh05(fwhm, hb_lwav, "Hb_LHb")
    edd_r_gh05_LHb = qso.calc_eddington_ratio(9.26 * L5100_from_Hb[0], m_bh_gh05_LHb[0])

    m_bh_gh05_L5100 = qso.se_bhmass_gh05(fwhm, lum_5100, "Hb_L5100")
    edd_r_gh05_L5100 = qso.calc_eddington_ratio(l_bol[0], m_bh_gh05_L5100[0])

    pl_slope = np.nanpercentile(df["pl_slope"], [15.87, 50, 84.13])

    print("L5100: ", np.nanpercentile(lum_5100, [15.87, 50, 84.13]))
    print("L5100_Hb: ", L5100_from_Hb)
    print("Hb FWHM: ", np.nanpercentile(hb_fwhm, [15.87, 50, 84.13]))
    # print('Hb FWHM: ', np.nanpercentile(Hg_fwhm, [15.87, 50, 84.13]))
    print("Hg FWHM: ", np.nanpercentile(hg_fwhm, [15.87, 50, 84.13]))
    print("O3 FWHM:", np.nanpercentile(O3_fwhm, [15.87, 50, 84.13]))
    print(f"BH mass (VP06): {np.nanpercentile(m_bh[0], 50):.2e}")
    print(f"BH mass (GH05, L5100): {np.nanpercentile(m_bh_gh05_L5100[0], 50):.2e}")
    print(f"BH mass (GH05, LHb): {np.nanpercentile(m_bh_gh05_LHb[0], 50):.2e}")
    print(f"BH mass (P12): {np.nanpercentile(m_bh_p12[0], 50):.2e}")
    print("Lbol: ", np.nanpercentile(l_bol[0], 50))
    print("Lbol(Hb): ", lbol_from_Hb[1])
    print("Eddington ratio: ", np.nanpercentile(edd_r, 50))

    # Calculate UV luminosity from L5100 Lbol
    # Shen+2020 (Xuejian Shen)
    c1 = 1.862
    k1 = -0.361
    c2 = 4.870
    k2 = -0.0063

    L1450_lbol = lband_from_lbol(
        np.nanpercentile(l_bol[0], q=[15.87, 50, 84.13]), c1, k1, c2, k2
    )
    print(
        f"L1450(Lbol) = {L1450_lbol[1]:.2e} {L1450_lbol[0] - L1450_lbol[1]:.2e} {L1450_lbol[2] - L1450_lbol[1]:.2e}"
    )

    M1450_lbol = scana.calc_absolute_mag_from_monochromatic_luminosity(
        L1450_lbol / 1450 / u.AA, 1450 * u.AA * (1 + redshift), redshift
    )
    print(
        f"M1450(Lbol) = {M1450_lbol[1]:.2f} {M1450_lbol[0] - M1450_lbol[1]:.2f} {M1450_lbol[2] - M1450_lbol[1]:.2f}"
    )

    L1450_Hb = lband_from_lbol(lbol_from_Hb * u.erg / u.s, c1, k1, c2, k2)
    print(
        f"L1450(Lbol Hb) = {L1450_Hb[1]:.2e} {L1450_Hb[0] - L1450_Hb[1]:.2e} {L1450_Hb[2] - L1450_Hb[1]:.2e}"
    )

    M1450_Hb = scana.calc_absolute_mag_from_monochromatic_luminosity(
        L1450_Hb / 1450 / u.AA, 1450 * u.AA * (1 + redshift), redshift
    )

    print(
        f"M1450(Lbol Hb) = {M1450_Hb[1]:.2f} {M1450_Hb[0] - M1450_Hb[1]:.2f} {M1450_Hb[2] - M1450_Hb[1]:.2f}"
    )

    L5100 = np.nanpercentile(lum_5100, [15.87, 50, 84.13])
    Lbol = np.nanpercentile(l_bol[0], [15.87, 50, 84.13])
    fwhm_b = np.nanpercentile(fwhm, [15.87, 50, 84.13])
    fwhm_hg = np.nanpercentile(hg_fwhm, [15.87, 50, 84.13])
    fwhm_O3 = np.nanpercentile(O3_fwhm, [15.87, 50, 84.13])

    mbh_vp06 = np.nanpercentile(m_bh[0], [15.87, 50, 84.13]) / 1e7
    eddr_vp06 = np.nanpercentile(edd_r, [15.87, 50, 84.13])

    mbh_gh05_L5100 = np.nanpercentile(m_bh_gh05_L5100[0], [15.87, 50, 84.13]) / 1e7
    eddr_gh05_L5100 = np.nanpercentile(edd_r_gh05_L5100, [15.87, 50, 84.13])

    mbh_gh05_LHb = np.nanpercentile(m_bh_gh05_LHb[0], [15.87, 50, 84.13]) / 1e7
    eddr_gh05_LHb = np.nanpercentile(edd_r_gh05_LHb, [15.87, 50, 84.13])

    mbh_p12 = np.nanpercentile(m_bh_p12[0], [15.87, 50, 84.13]) / 1e7
    eddr_p12 = np.nanpercentile(edd_r_p12, [15.87, 50, 84.13])

    # Equivalent width
    ew_o3_5007 = np.nanpercentile(rt["O3_5007_EW"], [15.87, 50, 84.13])
    ew_o3_4960 = np.nanpercentile(rt["O3_4960_EW"], [15.87, 50, 84.13])
    # ew_he2 = np.nanpercentile(rt['He2_EW'], [15.87, 50, 84.13])
    ew_ne3_a = np.nanpercentile(rt["Ne3_a_EW"], [15.87, 50, 84.13])

    flux_hb_br = np.nanpercentile(rt["Hb_broad_flux"], [15.87, 50, 84.13])
    flux_hb_nl = np.nanpercentile(rt["Hb_narrow_flux"], [15.87, 50, 84.13])
    flux_hg_br = np.nanpercentile(rt["Hg_broad_flux"], [15.87, 50, 84.13])
    flux_hg_nl = np.nanpercentile(rt["Hg_narrow_flux"], [15.87, 50, 84.13])
    flux_hd_nl = np.nanpercentile(rt["Hd_narrow_flux"], [15.87, 50, 84.13])

    result_dict = {
        "L1450_Lbol": L1450_lbol[1],
        "L1450_Lbol_low": L1450_lbol[0] - L1450_lbol[1],
        "L1450_Lbol_high": L1450_lbol[2] - L1450_lbol[1],
        "M1450_Lbol": M1450_lbol[1],
        "M1450_Lbol_low": M1450_lbol[0] - M1450_lbol[1],
        "M1450_Lbol_high": M1450_lbol[2] - M1450_lbol[1],
        "L1450_Hb": L1450_Hb[1],
        "L1450_Hb_low": L1450_Hb[0] - L1450_Hb[1],
        "L1450_Hb_high": L1450_Hb[2] - L1450_Hb[1],
        "M1450_Hb": M1450_Hb[1],
        "M1450_Hb_low": M1450_Hb[0] - M1450_Hb[1],
        "M1450_Hb_high": M1450_Hb[2] - M1450_Hb[1],
        "PL_slope": pl_slope[1],
        "PL_slope_low": pl_slope[0] - pl_slope[1],
        "PL_slope_high": pl_slope[2] - pl_slope[1],
        "L5100": np.log10(L5100[1].value),
        "L5100_low": np.log10(L5100[0].value) - np.log10(L5100[1].value),
        "L5100_high": np.log10(L5100[2].value) - np.log10(L5100[1].value),
        "L_Hb": np.log10(L_Hb[1].value),
        "L_Hb_low": np.log10(L_Hb[0].value) - np.log10(L_Hb[1].value),
        "L_Hb_high": np.log10(L_Hb[2].value) - np.log10(L_Hb[1].value),
        "Lbol": np.log10(Lbol[1].value),
        "Lbol_low": np.log10(Lbol[0].value) - np.log10(Lbol[1].value),
        "Lbol_high": np.log10(Lbol[2].value) - np.log10(Lbol[1].value),
        "Lbol_Hb": np.log10(lbol_from_Hb[1]),
        "Lbol_Hb_low": np.log10(lbol_from_Hb[0]) - np.log10(lbol_from_Hb[1]),
        "Lbol_Hb_high": np.log10(lbol_from_Hb[2]) - np.log10(lbol_from_Hb[1]),
        "fwhm_b": fwhm_b[1],
        "fwhm_low": fwhm_b[0] - fwhm_b[1],
        "fwhm_high": fwhm_b[2] - fwhm_b[1],
        "fwhm_hg": fwhm_hg[1],
        "fwhm_hg_low": fwhm_hg[0] - fwhm_hg[1],
        "fwhm_hg_high": fwhm_hg[2] - fwhm_hg[1],
        "fwhm_O3": fwhm_O3[1],
        "fwhm_O3_low": fwhm_O3[0] - fwhm_O3[1],
        "fwhm_O3_high": fwhm_O3[2] - fwhm_O3[1],
        "mbh_vp06": mbh_vp06[1],
        "mbh_vp06_low": mbh_vp06[0] - mbh_vp06[1],
        "mbh_vp06_high": mbh_vp06[2] - mbh_vp06[1],
        "eddr_vp06": eddr_vp06[1],
        "eddr_vp06_low": eddr_vp06[0] - eddr_vp06[1],
        "eddr_vp06_high": eddr_vp06[2] - eddr_vp06[1],
        "mbh_gh05_L5100": mbh_gh05_L5100[1],
        "mbh_gh05_L5100_low": mbh_gh05_L5100[0] - mbh_gh05_L5100[1],
        "mbh_gh05_L5100_high": mbh_gh05_L5100[2] - mbh_gh05_L5100[1],
        "eddr_gh05_L5100": eddr_gh05_L5100[1],
        "eddr_gh05_L5100_low": eddr_gh05_L5100[0] - eddr_gh05_L5100[1],
        "eddr_gh05_L5100_high": eddr_gh05_L5100[2] - eddr_gh05_L5100[1],
        "mbh_gh05_LHb": mbh_gh05_LHb[1],
        "mbh_gh05_LHb_low": mbh_gh05_LHb[0] - mbh_gh05_LHb[1],
        "mbh_gh05_LHb_high": mbh_gh05_LHb[2] - mbh_gh05_LHb[1],
        "eddr_gh05_LHb": eddr_gh05_LHb[1],
        "eddr_gh05_LHb_low": eddr_gh05_LHb[0] - eddr_gh05_LHb[1],
        "eddr_gh05_LHb_high": eddr_gh05_LHb[2] - eddr_gh05_LHb[1],
        "mbh_p12": mbh_p12[1],
        "mbh_p12_low": mbh_p12[0] - mbh_p12[1],
        "mbh_p12_high": mbh_p12[2] - mbh_p12[1],
        "eddr_p12": eddr_p12[1],
        "eddr_p12_low": eddr_p12[0] - eddr_p12[1],
        "eddr_p12_high": eddr_p12[2] - eddr_p12[1],
        "ew_o3_5007": ew_o3_5007[1],
        "ew_o3_5007_low": ew_o3_5007[0] - ew_o3_5007[1],
        "ew_o3_5007_high": ew_o3_5007[2] - ew_o3_5007[1],
        "ew_o3_4960": ew_o3_4960[1],
        "ew_o3_4960_low": ew_o3_4960[0] - ew_o3_4960[1],
        "ew_o3_4960_high": ew_o3_4960[2] - ew_o3_4960[1],
        # 'ew_he2': ew_he2[1],
        # 'ew_he2_low': ew_he2[0] - ew_he2[1],
        # 'ew_he2_high': ew_he2[2] - ew_he2[1],
        "ew_ne3_a": ew_ne3_a[1],
        "ew_ne3_a_low": ew_ne3_a[0] - ew_ne3_a[1],
        "ew_ne3_a_high": ew_ne3_a[2] - ew_ne3_a[1],
        "flux_hb_br": flux_hb_br[1],
        "flux_hb_br_low": flux_hb_br[0] - flux_hb_br[1],
        "flux_hb_br_high": flux_hb_br[2] - flux_hb_br[1],
        "flux_hb_nl": flux_hb_nl[1],
        "flux_hb_nl_low": flux_hb_nl[0] - flux_hb_nl[1],
        "flux_hb_nl_high": flux_hb_nl[2] - flux_hb_nl[1],
        "flux_hg_br": flux_hg_br[1],
        "flux_hg_br_low": flux_hg_br[0] - flux_hg_br[1],
        "flux_hg_br_high": flux_hg_br[2] - flux_hg_br[1],
        "flux_hg_nl": flux_hg_nl[1],
        "flux_hg_nl_low": flux_hg_nl[0] - flux_hg_nl[1],
        "flux_hg_nl_high": flux_hg_nl[2] - flux_hg_nl[1],
        "flux_hd_nl": flux_hd_nl[1],
        "flux_hd_nl_low": flux_hd_nl[0] - flux_hd_nl[1],
        "flux_hd_nl_high": flux_hd_nl[2] - flux_hd_nl[1],
        "L5100_from_hb": L5100_from_Hb[1],
        "L5100_from_hb_low": L5100_from_Hb[0] - L5100_from_Hb[1],
        "L5100_from_hb_high": L5100_from_Hb[2] - L5100_from_Hb[1],
    }

    return result_dict


def plot_spectral_fit_zoomin(
    model,
    discard,
    flat_chain=None,
    save=False,
    save_dir=None,
    save_name="spectral_fit_zoomin.pdf",
    a_v=0,
    mode="paper",
    dark=False,
    resolution=None,
):
    """Plot the spectral fit in a zoomed in region around the Hbeta line.

    :param model:
    :param discard:
    :param save:
    :param save_dir:
    :param save_name:
    :param a_v:
    :param mode:
    :param dark:
    :return:
    """

    if mode == "paper":
        plot_defaults.set_paper_defaults()
    elif mode == "presentation":
        plot_defaults.set_presentation_defaults()
        mpl.rcParams["xtick.top"] = False

    if dark:
        plt.style.use("dark_background")
        line_bg = "black"
        col_linetext = blue
        col_spec = "white"
        col_err = "0.8"
        col_fit = vermillion
    else:
        line_bg = "white"
        col_linetext = "C0"
        col_spec = "black"
        col_err = "0.5"
        col_fit = "C1"

    flux_fac = 1000  # Factor to multiply into the flux

    ymin = 0.10
    ymax = 0.95 * 1.2
    ddy = 0.03
    dy = (ymax - ymin) / 3 - ddy

    fig = plt.figure(constrained_layout=True, figsize=(9, 6))
    gs1 = fig.add_gridspec(
        nrows=5, ncols=4, bottom=0.12, top=0.9, hspace=0.0, right=0.97, left=0.12
    )
    ax1 = fig.add_subplot(gs1[:-1, :])
    ax2 = fig.add_subplot(gs1[-1, :], sharex=ax1)

    redshift = 7.2583

    def obswave_to_restwave(x):
        return x / (1 + redshift)

    def restwave_to_obswave(x):
        return x * (1 + redshift)

    spec = model.spec

    if a_v != 0:
        ext = extinction.calzetti00(
            spec.dispersion.newbyteorder().byteswap() / (1 + redshift), a_v, 4.05
        )

        new_fluxden = extinction.remove(ext, spec.fluxden)
        new_fluxden_err = extinction.remove(ext, spec.fluxden_err)

        spec.fluxden = new_fluxden
        spec.fluxden_err = new_fluxden_err

    ax1.step(
        spec.dispersion,
        spec.fluxden * flux_fac,
        color=col_spec,
        where="mid",
        lw=2,
        label=r"$f_{\lambda, \rm{obs}}$",
    )
    ax1.step(
        spec.dispersion,
        spec.fluxden_err * flux_fac,
        color=col_err,
        where="mid",
        alpha=0.5,
        lw=3,
        label=r"$\sigma_{f_{\lambda, \rm{obs}}}$",
    )

    # Set dispersion axis for the line fits
    dispersion = np.arange(30000, 50000, 20)
    dispersion = spec.dispersion
    if resolution is not None:
        model.resolution = np.interp(dispersion, spec.dispersion, resolution)

    # Generate the line fit components
    line_dict = {}
    comp_names = [comp.name for comp in model.components]
    comp_dict = dict(zip(comp_names, model.components))
    if flat_chain is None:
        flat_samples = model.sampler.get_chain(discard=discard, thin=1, flat=True)
    else:
        flat_samples = flat_chain
    line_array = np.zeros((flat_samples.shape[0], len(comp_names), dispersion.shape[0]))

    for idx in tqdm(range(flat_samples.shape[0])):
        # for idx in range(flat_samples.shape[0]):
        for ldx, line_comp in enumerate(comp_names):
            line_array[idx, ldx, :] = model.eval(
                dispersion, flat_samples[idx, :], components=[comp_dict[line_comp]]
            )

    for ldx, line in enumerate(comp_names):
        sub_array = line_array[:, ldx, :]
        line_dict.update({line: sub_array})

    if resolution is not None:
        model.resolution = resolution

    # Plot the median model fit and the 1-sigma range
    if flat_chain is not None:
        model.flat_chain = flat_chain
        model.get_mcmc_model_results(discard=discard)
    median_fit = model.mcmc_model_posterior[0]
    low_fit = model.mcmc_model_posterior[1]
    upp_fit = model.mcmc_model_posterior[2]

    ax1.fill_between(
        spec.dispersion,
        low_fit * flux_fac,
        upp_fit * flux_fac,
        step="mid",
        color=gdv_div_b[4],
        ls="-",
        alpha=0.5,
    )

    ax1.step(
        model.spec.dispersion,
        median_fit * flux_fac,
        color=gdv_div_b[4],
        lw=2,
        where="mid",
        ls="-",
        alpha=0.8,
        label=r"$f_{\lambda, \rm{medfit}}$",
    )

    # Plot the resampled results
    median, low, upp = np.percentile(line_dict["pl"], q=[50, 16, 84], axis=0)
    ax1.fill_between(
        dispersion,
        low * flux_fac,
        upp * flux_fac,
        step="mid",
        color=gdv_div_b[0],
        ls="-",
        alpha=0.5,
    )
    ax1.step(
        dispersion,
        median * flux_fac,
        color=gdv_div_b[0],
        ls="--",
        lw=2,
        label=r"$\rm{Power-law}$",
        where="mid",
    )

    # median, low, upp = np.percentile(line_dict['He2'], q=[50, 16, 84], axis=0)
    # ax1.fill_between(dispersion, low * flux_fac, upp * flux_fac, step='mid',
    #                  color=gdv_div_b[2], ls='-',
    #                  alpha=0.5)
    # ax1.step(dispersion, median * flux_fac, color=gdv_div_b[2], ls='-',
    #          lw=2, label=r'$\rm{Single/Narrow\ component}$', where='mid')

    median, low, upp = np.percentile(line_dict["Ne3_a"], q=[50, 16, 84], axis=0)
    ax1.fill_between(
        dispersion,
        low * flux_fac,
        upp * flux_fac,
        step="mid",
        color=gdv_div_b[2],
        ls="-",
        label=r"$\rm{Single/Narrow\ component}$",
        alpha=0.5,
    )
    ax1.step(
        dispersion, median * flux_fac, color=gdv_div_b[2], ls="-", lw=2, where="mid"
    )

    median, low, upp = np.percentile(line_dict["O3_4960"], q=[50, 16, 84], axis=0)
    ax1.fill_between(
        dispersion,
        low * flux_fac,
        upp * flux_fac,
        step="mid",
        color=gdv_div_b[2],
        ls="-",
        alpha=0.5,
    )
    ax1.step(
        dispersion, median * flux_fac, color=gdv_div_b[2], ls="-", lw=2, where="mid"
    )

    median, low, upp = np.percentile(line_dict["O3_5007"], q=[50, 16, 84], axis=0)
    ax1.fill_between(
        dispersion,
        low * flux_fac,
        upp * flux_fac,
        step="mid",
        color=gdv_div_b[2],
        ls="-",
        alpha=0.5,
    )
    ax1.step(
        dispersion, median * flux_fac, color=gdv_div_b[2], ls="-", lw=2, where="mid"
    )

    median, low, upp = np.percentile(line_dict["O3_4364"], q=[50, 16, 84], axis=0)
    ax1.fill_between(
        dispersion,
        low * flux_fac,
        upp * flux_fac,
        step="mid",
        color=gdv_div_b[2],
        ls="-",
        alpha=0.5,
    )
    ax1.step(
        dispersion, median * flux_fac, color=gdv_div_b[2], ls="-", lw=2, where="mid"
    )

    median, low, upp = np.percentile(line_dict["Hg"], q=[50, 16, 84], axis=0)
    ax1.fill_between(
        dispersion,
        low * flux_fac,
        upp * flux_fac,
        step="mid",
        color=gdv_div_b[2],
        ls="-",
        alpha=0.5,
    )
    ax1.step(
        dispersion, median * flux_fac, color=gdv_div_b[2], ls="-", lw=2, where="mid"
    )

    median, low, upp = np.percentile(line_dict["Hb"], q=[50, 16, 84], axis=0)
    ax1.fill_between(
        dispersion,
        low * flux_fac,
        upp * flux_fac,
        step="mid",
        color=gdv_div_b[2],
        ls="-",
        alpha=0.5,
    )
    ax1.step(
        dispersion, median * flux_fac, color=gdv_div_b[2], ls="-", lw=2, where="mid"
    )

    median, low, upp = np.percentile(line_dict["Hd"], q=[50, 16, 84], axis=0)
    ax1.fill_between(
        dispersion,
        low * flux_fac,
        upp * flux_fac,
        step="mid",
        color=gdv_div_b[2],
        ls="-",
        alpha=0.5,
    )
    ax1.step(
        dispersion, median * flux_fac, color=gdv_div_b[2], ls="-", lw=2, where="mid"
    )

    median, low, upp = np.percentile(line_dict["Hb_br"], q=[50, 16, 84], axis=0)
    ax1.fill_between(
        dispersion,
        low * flux_fac,
        upp * flux_fac,
        step="mid",
        color=gdv_div_b[1],
        ls="-",
        alpha=0.5,
    )
    ax1.step(
        dispersion,
        median * flux_fac,
        color=gdv_div_b[1],
        ls="-",
        lw=2,
        label=r"$\rm{Broad\ component}$",
        where="mid",
    )

    median, low, upp = np.percentile(line_dict["Hg_br"], q=[50, 16, 84], axis=0)
    ax1.fill_between(
        dispersion,
        low * flux_fac,
        upp * flux_fac,
        step="mid",
        color=gdv_div_b[1],
        ls="-",
        alpha=0.5,
    )
    ax1.step(
        dispersion, median * flux_fac, color=gdv_div_b[1], ls="-", lw=2, where="mid"
    )

    ax1.axhline(0, color="0.8", ls="-", lw=1.5)

    # Plot line texts
    fs = 12
    ax1.text(
        restwave_to_obswave(4687.02),
        -0.25,
        r"$\rm{HeII}$",
        va="center",
        ha="center",
        fontsize=fs,
    )
    ax1.text(
        restwave_to_obswave(3869.85),
        -0.25,
        r"$\rm{[NeIII]}$",
        va="center",
        ha="center",
        fontsize=fs,
    )
    ax1.text(
        restwave_to_obswave(3968.58),
        -0.25,
        r"$\rm{[NeIII]}$",
        va="center",
        ha="center",
        fontsize=fs,
    )
    ax1.text(
        restwave_to_obswave(4102.89),
        -0.25,
        r"$\rm{H}\delta$",
        va="center",
        ha="center",
        fontsize=fs,
    )
    ax1.text(
        restwave_to_obswave(4351.68),
        -0.25,
        r"$\rm{H}\gamma{+}\rm{[OIII]}$",
        va="center",
        ha="center",
        fontsize=fs,
    )
    ax1.text(
        restwave_to_obswave(4862.68),
        -0.25,
        r"$\rm{H}\beta$",
        va="center",
        ha="center",
        fontsize=fs,
    )
    ax1.text(
        restwave_to_obswave(4952.30),
        -0.25,
        r"$\rm{[OIII]}$",
        va="center",
        ha="center",
        fontsize=fs,
    )
    ax1.text(
        restwave_to_obswave(5018.24),
        -0.25,
        r"$\rm{[OIII]}$",
        va="center",
        ha="center",
        fontsize=fs,
    )

    # Plot the residuals

    ax2.step(
        spec.dispersion,
        (spec.fluxden - median_fit) * flux_fac,
        color=col_spec,
        lw=2,
        where="mid",
        ls="-",
    )
    ax2.fill_between(
        spec.dispersion,
        -spec.fluxden_err * flux_fac,
        spec.fluxden_err * flux_fac,
        color="0.5",
        alpha=0.3,
        step="mid",
    )
    ax2.axhline(0, color="0.8", ls="-", lw=1.5)

    # Plot adjustments
    ax1.set_xlim(restwave_to_obswave(3800), restwave_to_obswave(5200))
    ax1.set_ylim(-0.5, 5)
    ax2.set_ylim(-0.5, 0.5)

    ax1.tick_params(labelbottom=False)
    ax1.tick_params(axis="x", which="both", top=False)

    # Add rest-frame wavelength axis
    secax = ax1.secondary_xaxis(
        "top", functions=(obswave_to_restwave, restwave_to_obswave)
    )
    secax.set_xlabel(
        r"$\rm{Rest{-}frame\ wavelength}$ $\left(\textrm{\AA}\right)$", fontsize=20
    )
    secax.tick_params(axis="both", which="major", labelsize=19)

    # Plot modifications
    ax1.tick_params(axis="both", which="major", labelsize=19)
    ax2.tick_params(axis="both", which="major", labelsize=19)
    ax2.set_xlabel(
        r"$\rm{Observed\ Wavelength}$ $\left(\textrm{\AA}\right)$", fontsize=20
    )
    ax1.set_ylabel(
        r"$f_\lambda\ (10^{-20}\,\textrm{erg}\,\textrm{s}^{-1}\,\textrm{cm}^{-2}\,"
        r"\textrm{\AA}^{-1})$",
        fontsize=16,
        labelpad=20,
    )
    ax2.set_ylabel(r"$\rm{Residuals}$", fontsize=20, labelpad=0)

    ax1.legend(loc=2, fontsize=16, ncols=3)

    insets = True

    if insets:
        pl_median, low, upp = np.percentile(line_dict["pl"], q=[50, 16, 84], axis=0)

        # Define insets

        axins = inset_axes(
            ax1,
            width="40%",
            height="60%",
            bbox_to_anchor=(0.08, 0.29, 0.6, 0.70),
            bbox_transform=ax1.transAxes,
            loc=3,
        )

        axins2 = inset_axes(
            ax1,
            width="40%",
            height="60%",
            bbox_to_anchor=(0.45, 0.29, 0.6, 0.70),
            bbox_transform=ax1.transAxes,
            loc=3,
        )

        axins.step(
            spec.dispersion, spec.fluxden * flux_fac, color=col_spec, where="mid", lw=2
        )

        median, low, upp = np.percentile(line_dict["Hg"], q=[50, 16, 84], axis=0)
        axins.fill_between(
            dispersion,
            (pl_median + low) * flux_fac,
            (pl_median + upp) * flux_fac,
            step="mid",
            color=gdv_div_b[2],
            ls="-",
            alpha=0.5,
        )
        axins.step(
            dispersion,
            (pl_median + median) * flux_fac,
            color=gdv_div_b[2],
            ls="-",
            lw=2,
            where="mid",
        )

        median, low, upp = np.percentile(line_dict["Hg_br"], q=[50, 16, 84], axis=0)
        axins.fill_between(
            dispersion,
            (pl_median + low) * flux_fac,
            (pl_median + upp) * flux_fac,
            step="mid",
            color=gdv_div_b[1],
            ls="-",
            alpha=0.5,
        )
        axins.step(
            dispersion,
            (pl_median + median) * flux_fac,
            color=gdv_div_b[1],
            ls="-",
            lw=2,
            where="mid",
        )

        median, low, upp = np.percentile(line_dict["O3_4364"], q=[50, 16, 84], axis=0)
        axins.fill_between(
            dispersion,
            (pl_median + low) * flux_fac,
            (pl_median + upp) * flux_fac,
            step="mid",
            color=gdv_div_b[2],
            ls="-",
            alpha=0.5,
        )
        axins.step(
            dispersion,
            (pl_median + median) * flux_fac,
            color=gdv_div_b[2],
            ls="-",
            lw=2,
            where="mid",
        )

        axins.fill_between(
            spec.dispersion,
            low_fit * flux_fac,
            upp_fit * flux_fac,
            step="mid",
            color=gdv_div_b[4],
            ls="-",
            alpha=0.5,
        )

        axins.step(
            spec.dispersion,
            median_fit * flux_fac,
            color=gdv_div_b[4],
            lw=2,
            where="mid",
            ls="-",
            alpha=0.8,
        )

        axins.set_xlim(35400, 36500)
        axins.set_ylim(0.2, 1.8 * 1.2)
        axins.tick_params(axis="both", which="major", labelsize=14)

        axins2.step(
            spec.dispersion, spec.fluxden * flux_fac, color=col_spec, where="mid", lw=2
        )

        median, low, upp = np.percentile(line_dict["Hb"], q=[50, 16, 84], axis=0)
        axins2.fill_between(
            dispersion,
            (pl_median + low) * flux_fac,
            (pl_median + upp) * flux_fac,
            step="mid",
            color=gdv_div_b[2],
            ls="-",
            alpha=0.5,
        )
        axins2.step(
            dispersion,
            (pl_median + median) * flux_fac,
            color=gdv_div_b[2],
            ls="-",
            lw=2,
            where="mid",
        )

        median, low, upp = np.percentile(line_dict["Hb_br"], q=[50, 16, 84], axis=0)
        axins2.fill_between(
            dispersion,
            (pl_median + low) * flux_fac,
            (pl_median + upp) * flux_fac,
            step="mid",
            color=gdv_div_b[1],
            ls="-",
            alpha=0.5,
        )
        axins2.step(
            dispersion,
            (pl_median + median) * flux_fac,
            color=gdv_div_b[1],
            ls="-",
            lw=2,
            where="mid",
        )

        axins2.fill_between(
            spec.dispersion,
            low_fit * flux_fac,
            upp_fit * flux_fac,
            step="mid",
            color=gdv_div_b[4],
            ls="-",
            alpha=0.5,
        )

        axins2.step(
            spec.dispersion,
            median_fit * flux_fac,
            color=gdv_div_b[4],
            lw=2,
            where="mid",
            ls="-",
            alpha=0.8,
        )

        axins2.set_xlim(39500, 40750)
        axins2.set_ylim(0.2, 2.5 * 1.2)
        axins2.tick_params(axis="both", which="major", labelsize=14)

        axins.text(
            0.1,
            0.85,
            r"$\rm{H\gamma{+}[OIII]}$",
            va="center",
            ha="left",
            transform=axins.transAxes,
            fontsize=16,
        )
        axins2.text(
            0.1,
            0.85,
            r"$\rm{H\beta}$",
            va="center",
            ha="left",
            transform=axins2.transAxes,
            fontsize=16,
        )

    if save:
        plt.savefig(os.path.join(save_dir, save_name))
    else:
        plt.show()


def get_basic_spectral_properties(spec, redshift=7.266):
    """Get basic spectral properties from a spectrum

    :param spec: Spectrum1D object
    :param a_lambda: power law slope in the UV
    :param a_lambda_red: power law slope in the optical
    :return:
    """

    cosmology = FlatLambdaCDM(H0=70, Om0=0.3)

    a_nu = 0
    a_nu_red = 0

    print(a_nu, a_nu_red)

    # Measure the continuum at 1450A/M_UV
    flux_1450 = scana.get_average_fluxden(spec, 1450, redshift=redshift, width=50)
    disp_range = [
        (1450 - 50 / 2.0) * (1.0 + redshift),
        (1450 + 50 / 2.0) * (1.0 + redshift),
    ]
    flux_1450_err = (
        np.average(spec.trim_dispersion(disp_range).fluxden_err) * spec.fluxden_unit
    )
    m_1450 = scana.calc_apparent_mag_from_fluxden(
        flux_1450, 1450 * u.AA * (1 + redshift)
    )
    absmag_1450 = scana.calc_absolute_mag_from_fluxden(
        flux_1450,
        1450 * u.AA * (1 + redshift),
        redshift=redshift,
        cosmology=cosmology,
        kcorrection=True,
        a_nu=a_nu,
    )

    absmag_1450_low = scana.calc_absolute_mag_from_fluxden(
        flux_1450 - flux_1450_err,
        1450 * u.AA * (1 + redshift),
        redshift=redshift,
        cosmology=cosmology,
        kcorrection=True,
        a_nu=a_nu,
    )
    absmag_1450_upp = scana.calc_absolute_mag_from_fluxden(
        flux_1450 + flux_1450_err,
        1450 * u.AA * (1 + redshift),
        redshift=redshift,
        cosmology=cosmology,
        kcorrection=True,
        a_nu=a_nu,
    )

    lwav_1450 = scana.calc_lwav_from_fwav(
        flux_1450, redshift=redshift, cosmology=cosmology
    )
    absmag2_1450 = scana.calc_absolute_mag_from_monochromatic_luminosity(
        lwav_1450, 1450 * u.AA * (1 + redshift), redshift=redshift
    )

    flux_5100 = scana.get_average_fluxden(spec, 5100, redshift=redshift, width=50)
    lwav_5100 = scana.calc_lwav_from_fwav(
        flux_5100, redshift=redshift, cosmology=cosmology
    )
    absmag_5100 = scana.calc_absolute_mag_from_fluxden(
        flux_5100,
        5100 * u.AA * (1 + redshift),
        redshift=redshift,
        cosmology=cosmology,
    )

    print(f"f_1450 = {flux_1450:.2e}")
    print(f"f_1450_err = {flux_1450_err:.2e}")
    print(f"L1450 = {lwav_1450 * 1450:.2e}")
    print(f"L5100 = {lwav_5100 * 5100:.2e}")
    print(f"m1450 = {m_1450:.2f}")
    print(
        f"M1450 = {absmag_1450:.2f} {absmag_1450_low - absmag_1450:+.2f} {absmag_1450_upp - absmag_1450:+.2f}"
    )
    print(f"M1450 (alt) = {absmag2_1450:.2f}")

    # Plus the absolute UV magnitude from the F115W photometry
    mf115w = 26.79 * u.mag
    mf115w_err = 0.07 * u.mag
    Muv = scana.calc_absolute_mag_from_apparent_mag(
        mf115w, cosmology, redshift, a_nu=a_nu
    )
    Muv_low = scana.calc_absolute_mag_from_apparent_mag(
        mf115w - mf115w_err, cosmology, redshift, a_nu=a_nu
    )
    Muv_upp = scana.calc_absolute_mag_from_apparent_mag(
        mf115w + mf115w_err, cosmology, redshift, a_nu=a_nu
    )
    print(f"M_F115W = {Muv:.2f}-{Muv_low - Muv:.2f}+{Muv_upp - Muv:.2f}")

    # Get SNR
    width = 50
    dispersion = 1450
    disp_range = [
        (dispersion - width / 2.0) * (1.0 + redshift),
        (dispersion + width / 2.0) * (1.0 + redshift),
    ]

    snr_1450 = np.average(
        spec.trim_dispersion(disp_range).fluxden
        / spec.trim_dispersion(disp_range).fluxden_err
    )

    width = 50
    dispersion = 5100
    disp_range = [
        (dispersion - width / 2.0) * (1.0 + redshift),
        (dispersion + width / 2.0) * (1.0 + redshift),
    ]

    snr_5100 = np.average(
        spec.trim_dispersion(disp_range).fluxden
        / spec.trim_dispersion(disp_range).fluxden_err
    )

    print(f"SNR 1450 = {snr_1450:.2f}")
    print(f"SNR 5100 = {snr_5100:.2f}")

    # Get nominal resolution at wavelength
    hdu = fits.open("../../data/jwst_nirspec_prism_disp.fits")
    disp = hdu[1].data["WAVELENGTH"]
    res_arr = hdu[1].data["R"]

    wavelength = 1450
    idx = np.argmin(np.abs(disp * 10000 - wavelength * (1 + redshift)))
    res_1450 = res_arr[idx]

    wavelength = 5100
    idx = np.argmin(np.abs(disp * 10000 - wavelength * (1 + redshift)))
    res_5100 = res_arr[idx]

    print(f"R_1450 = {res_1450:.2f}")
    print(f"R_5100 = {res_5100:.2f}")

    print(f"SNR at G395M = {snr_5100 / np.sqrt(1000 / res_5100):.2f}")

    result_dict = {
        "f_1450": flux_1450,
        "f_1450_err": flux_1450_err,
        "L1450": lwav_1450 * 1450,
        "L5100": lwav_5100 * 5100,
        "m1450": m_1450,
        "M1450": absmag_1450,
        "M1450_low": absmag_1450_low - absmag_1450,
        "M1450_upp": absmag_1450_upp - absmag_1450,
        "Muv": Muv,
        "Muv_low": Muv_low - Muv,
        "Muv_upp": Muv_upp - Muv,
        "SNR_1450": snr_1450,
        "SNR_5100": snr_5100,
        "R_1450": res_1450,
        "R_5100": res_5100,
    }

    return result_dict


def create_paper_table(
    prel_result, basic_dict, result_dict, table_name="agn_spec_results.tex"
):
    """Create the tex table for the paper

    :param prel_result:
    :param basic_dict:
    :param result_dict:
    :return:
    """

    # Create a table with the basic spectral properties
    filename = os.path.join("../../tex_tables", table_name)
    table = open(filename, "w+")

    table.write("\\begin{table}[h!]\n")
    table.write("\\renewcommand{\\arraystretch}{1.3}\n")
    table.write("\\footnotesize \n")
    table.write("\\centering \n")
    table.write("\\caption{\\textbf{AGN Properties}}\n")

    table.write("\\begin{tabular}{ll} \n")
    table.write("\\small \n")

    # tablehead
    table.write(" Property &  Value \\\\ \n")

    table.write("\\hline \n")

    # table content
    table.write(
        "Redshift & ${:.4f}_{{{:+.4f}}}^{{{:+.4f}}}$  \\\\ \n".format(
            prel_result["redsh"][0], prel_result["redsh"][1], prel_result["redsh"][2]
        )
    )

    # table.write(
    #     '$M_{\\rm{UV}}/(\\rm{mag})$' + ' & ${:.2f}_{{{:+.2f}}}^{{{:+.2f}}}$  \\\\ \n'.format(basic_dict['Muv'].value,
    #                                                                                          basic_dict[
    #                                                                                              'Muv_low'].value,
    #                                                                                          basic_dict[
    #                                                                                              'Muv_upp'].value))
    #
    # table.write('\\hline \n')

    table.write(
        "$M_{1450}/(\\rm{mag})$"
        + " & ${:.2f}_{{{:+.2f}}}^{{{:+.2f}}}$ \\\\ \n".format(
            basic_dict["M1450"].value,
            basic_dict["M1450_upp"].value,
            basic_dict["M1450_low"].value,
        )
    )

    # table.write('$\\alpha_{\\rm{UV}}$' + '& ${:.2f}_{{{:+.2f}}}^{{{:+.2f}}}$ \\\\ \n'.format(blue_cont_result[
    #                                                                                              'PL_slope'][0],
    #                                                                                          blue_cont_result[
    #                                                                                              'PL_slope'][1],
    #                                                                                          blue_cont_result[
    #                                                                                              'PL_slope'][2]))
    table.write(
        "$\\alpha_{\\rm{OPT}}$"
        + " & ${:.2f}_{{{:+.2f}}}^{{{:+.2f}}}$ \\\\ \n".format(
            result_dict["PL_slope"],
            result_dict["PL_slope_low"],
            result_dict["PL_slope_high"],
        )
    )

    table.write("\\hline \n")

    # table.write('$\\rm{FWHM}_{\\rm{H}\\gamma, \\rm{br}}/(\\rm{km}\,\\rm{s}^{-1})$' + ' & ${:.2f}_{{{:+.2f}}}^{{{'
    #                                                                                   ':+.2f}}}$ \\\\ \n'.format(
    #     result_dict['fwhm_hg'].value,
    #     result_dict['fwhm_hg_low'].value,
    #     result_dict['fwhm_hg_high'].value))
    table.write(
        "$\\rm{FWHM}_{\\rm{[OIII]}5007, \\rm{narrow}}/(\\rm{km}\\,\\rm{s}^{-1})$"
        + " & ${:.2f}_{{{:+.2f}}}^{{{:+.2f}}}$  \\\\ \n".format(
            result_dict["fwhm_O3"].value,
            result_dict["fwhm_O3_low"].value,
            result_dict["fwhm_O3_high"].value,
        )
    )

    table.write(
        "$\\rm{FWHM}_{\\rm{H}\\beta, \\rm{broad}}/(\\rm{km}\\,\\rm{s}^{-1})$"
        + " & ${:.2f}_{{{:+.2f}}}^{{{:+.2f}}}$  \\\\ \n".format(
            result_dict["fwhm_b"].value,
            result_dict["fwhm_low"].value,
            result_dict["fwhm_high"].value,
        )
    )

    table.write(
        "$\\rm{EW}_{\\rm{[OIII]}5007}/(\\textrm{\\AA})$"
        + " & ${:.2f}_{{{:+.2f}}}^{{{:+.2f}}}$  \\\\ \n".format(
            result_dict["ew_o3_5007"].value,
            result_dict["ew_o3_5007_low"].value,
            result_dict["ew_o3_5007_high"].value,
        )
    )

    table.write(
        "$\\rm{EW}_{\\rm{[OIII]}4960}/(\\textrm{\\AA})$"
        + " & ${:.2f}_{{{:+.2f}}}^{{{:+.2f}}}$  \\\\ \n".format(
            result_dict["ew_o3_4960"].value,
            result_dict["ew_o3_4960_low"].value,
            result_dict["ew_o3_4960_high"].value,
        )
    )

    # table.write('$\\rm{EW}_{\\rm{HeII}4687}/(\\textrm{\AA})$' + ' & ${:.2f}_{{{:+.2f}}}^{{{:+.2f}}}$  \\\\ \n'.format(
    #     result_dict['ew_he2'].value,
    #     result_dict['ew_he2_low'].value,
    #     result_dict['ew_he2_high'].value))

    table.write(
        "$\\rm{EW}_{\\rm{[NeIII]}3869}/(\\textrm{\\AA})$"
        + " & ${:.2f}_{{{:+.2f}}}^{{{:+.2f}}}$  \\\\ \n".format(
            result_dict["ew_ne3_a"].value,
            result_dict["ew_ne3_a_low"].value,
            result_dict["ew_ne3_a_high"].value,
        )
    )

    table.write(
        "$\\rm{Flux}_{\\rm{H}\\beta, \\rm{broad}}/(10^{-20}\\,\\rm{erg}\\,\\rm{s}^{-1}\\,\\rm{cm}^{-2})"
        + " & ${:.2f}_{{{:+.2f}}}^{{{:+.2f}}}$  \\\\ \n".format(
            result_dict["flux_hb_br"].value / 1e-20,
            result_dict["flux_hb_br_low"].value / 1e-20,
            result_dict["flux_hb_br_high"].value / 1e-20,
        )
    )

    table.write(
        "$\\rm{Flux}_{\\rm{H}\\beta, \\rm{narrow}}/(10^{-20}\\,\\rm{erg}\\,\\rm{s}^{-1}\\,\\rm{cm}^{-2})"
        + " & ${:.2f}_{{{:+.2f}}}^{{{:+.2f}}}$  \\\\ \n".format(
            result_dict["flux_hb_nl"].value / 1e-20,
            result_dict["flux_hb_nl_low"].value / 1e-20,
            result_dict["flux_hb_nl_high"].value / 1e-20,
        )
    )

    table.write(
        "$\\rm{Flux}_{\\rm{H}\\gamma, \\rm{broad}}/(10^{-20}\\,\\rm{erg}\\,\\rm{s}^{-1}\\,\\rm{cm}^{-2})"
        + " & ${:.2f}_{{{:+.2f}}}^{{{:+.2f}}}$  \\\\ \n".format(
            result_dict["flux_hg_br"].value / 1e-20,
            result_dict["flux_hg_br_low"].value / 1e-20,
            result_dict["flux_hg_br_high"].value / 1e-20,
        )
    )

    table.write(
        "$\\rm{Flux}_{\\rm{H}\\gamma, \\rm{narrow}}/(10^{-20}\\,\\rm{erg}\\,\\rm{s}^{-1}\\,\\rm{cm}^{-2})"
        + " & ${:.2f}_{{{:+.2f}}}^{{{:+.2f}}}$  \\\\ \n".format(
            result_dict["flux_hg_nl"].value / 1e-20,
            result_dict["flux_hg_nl_low"].value / 1e-20,
            result_dict["flux_hg_nl_high"].value / 1e-20,
        )
    )

    table.write("\\hline \n")

    # table content
    table.write("$\\log(L_{5100}/( \\rm{erg}\\,\\rm{s}^{-1}\\,\\rm{\\AA}^{-1}))$ &")
    table.write(
        "${:.2f}_{{{:+.2f}}}^{{{:+.2f}}}$ \\\\ \n".format(
            result_dict["L5100"], result_dict["L5100_low"], result_dict["L5100_high"]
        )
    )

    table.write("$\\log(L_{\\rm{bol}}/(\\rm{erg}\\,\\rm{s}^{-1}))$ &")
    table.write(
        "${:.2f}_{{{:+.2f}}}^{{{:+.2f}}}$ \\\\ \n".format(
            result_dict["Lbol"], result_dict["Lbol_low"], result_dict["Lbol_high"]
        )
    )

    table.write("$\\log(L_{\\rm{H}\\beta}/(\\rm{erg}\\,\\rm{s}^{-1}))$ &")
    table.write(
        "${:.2f}_{{{:+.2f}}}^{{{:+.2f}}}$ \\\\ \n".format(
            result_dict["L_Hb"], result_dict["L_Hb_low"], result_dict["L_Hb_high"]
        )
    )

    table.write("$\\log(L_{\\rm{bol, H}\\beta}/(\\rm{erg}\\,\\rm{s}^{-1}))$ &")
    table.write(
        "${:.2f}_{{{:+.2f}}}^{{{:+.2f}}}$ \\\\ \n".format(
            result_dict["Lbol_Hb"],
            result_dict["Lbol_Hb_low"],
            result_dict["Lbol_Hb_high"],
        )
    )

    #
    # table.write('$\\rm{FWHM}_{\\rm{H}\\beta, \\rm{br}}/(\\rm{km}\,\\rm{s}^{-1})$ &')
    # table.write('${:.2f}_{{{:+.2f}}}^{{{:+.2f}}}$ \\\\ \n'.format(
    #     result_dict['fwhm_b'].value, result_dict['fwhm_low'].value, result_dict['fwhm_high'].value))

    table.write("$M_{\\rm{BH, GH05, LH}\\beta}/(10^{7}\\,\\rm{M}_{\\odot})$ &")
    table.write(
        "${:.2f}_{{{:+.2f}}}^{{{:+.2f}}}$ \\\\ \n".format(
            result_dict["mbh_gh05_LHb"].value,
            result_dict["mbh_gh05_LHb_low"].value,
            result_dict["mbh_gh05_LHb_high"].value,
        )
    )

    table.write("$\\lambda_{\\rm{Edd, GH05, LH}\\beta}$ &")
    table.write(
        "${:.2f}_{{{:+.2f}}}^{{{:+.2f}}}$ \\\\ \n".format(
            result_dict["eddr_gh05_LHb"].value,
            result_dict["eddr_gh05_LHb_low"].value,
            result_dict["eddr_gh05_LHb_high"].value,
        )
    )

    table.write("\\hline \n")

    table.write("$M_{\\rm{BH, VP06}}/(10^{7}\\,\\rm{M}_{\\odot})$ &")
    table.write(
        "${:.2f}_{{{:+.2f}}}^{{{:+.2f}}}$ \\\\ \n".format(
            result_dict["mbh_vp06"].value,
            result_dict["mbh_vp06_low"].value,
            result_dict["mbh_vp06_high"].value,
        )
    )

    table.write("$\\lambda_{\\rm{Edd, VP06}}$ &")
    table.write(
        "${:.2f}_{{{:+.2f}}}^{{{:+.2f}}}$ \\\\ \n".format(
            result_dict["eddr_vp06"].value,
            result_dict["eddr_vp06_low"].value,
            result_dict["eddr_vp06_high"].value,
        )
    )

    table.write("$M_{\\rm{BH, GH05}}/(10^{7}\\,\\rm{M}_{\\odot})$ &")
    table.write(
        "${:.2f}_{{{:+.2f}}}^{{{:+.2f}}}$ \\\\ \n".format(
            result_dict["mbh_gh05_L5100"].value,
            result_dict["mbh_gh05_L5100_low"].value,
            result_dict["mbh_gh05_L5100_high"].value,
        )
    )

    table.write("$\\lambda_{\\rm{Edd, GH05}}$ &")
    table.write(
        "${:.2f}_{{{:+.2f}}}^{{{:+.2f}}}$ \\\\ \n".format(
            result_dict["eddr_gh05_L5100"].value,
            result_dict["eddr_gh05_L5100_low"].value,
            result_dict["eddr_gh05_L5100_high"].value,
        )
    )

    table.write("$M_{\\rm{BH, P12}}/(10^{7}\\,\\rm{M}_{\\odot})$ &")
    table.write(
        "${:.2f}_{{{:+.2f}}}^{{{:+.2f}}}$ \\\\ \n".format(
            result_dict["mbh_p12"].value,
            result_dict["mbh_p12_low"].value,
            result_dict["mbh_p12_high"].value,
        )
    )

    table.write("$\\lambda_{\\rm{Edd, P12}}$ &")
    table.write(
        "${:.2f}_{{{:+.2f}}}^{{{:+.2f}}}$ \\\\ \n".format(
            result_dict["eddr_p12"].value,
            result_dict["eddr_p12_low"].value,
            result_dict["eddr_p12_high"].value,
        )
    )

    table.write("\\hline \n")
    table.write("\\hline \n")
    table.write("\\label{tab:spec_info} \n")
    table.write("\\end{tabular} \n")
    # table.write('\\\\ \n')
    # table.write('\\footnotesize{ \\textbf{Notes.}} \n')
    table.write("\\end{table} \n")
    table.close()


def create_paper_table_large(
    prel_result, basic_dict, result_dict_a, result_dict_b, result_dict_c
):
    """Create the tex table for the paper

    :param prel_result:
    :param basic_dict:
    :param result_dict:
    :return:
    """

    # Create a table with the basic spectral properties
    table = open("../../tex_tables/agn_spec_results_comp.tex", "w+")

    table.write("\\begin{table}[h!]\n")
    table.write("\\renewcommand{\\arraystretch}{1.3}\n")
    table.write("\\footnotesize \n")
    table.write("\\centering \n")
    table.write("\\caption{\\textbf{AGN Properties}}\n")

    table.write("\\begin{tabular}{llll} \n")
    table.write("\\small \n")

    # tablehead
    table.write(
        " Property as a function of & $R\\approx273$ (default) &  $R\\approx278$ & $R\\approx268$ \\\\ \n"
    )
    table.write(
        " resolution at [OIII]\\,5007 &  FWHM=$1100\\,\\rm{km}\\,\\rm{s}^{-1}$ & FWHM=$1080\\,\\rm{km}\\,"
        "\\rm{s}^{-1}$ & "
        "FWHM=$1120\\,\\rm{km}\\,\\rm{s}^{-1}$ \\\\ \n"
    )

    table.write("\\hline \n")

    # table content
    table.write(
        "Redshift &  \\multicolumn{{3}}{{c}}{{ ${:.4f}_{{{:+.4f}}}^{{{:+.4f}}}$ }}  \\\\ \n".format(
            prel_result["redsh"][0], prel_result["redsh"][1], prel_result["redsh"][2]
        )
    )

    # table.write(
    #     '$M_{\\rm{UV}}/(\\rm{mag})$' + ' & \\multicolumn{3}{c}{${:.2f}_{{{:+.2f}}}^{{{:+.2f}}}$}  \\\\ \n'.format(
    #         basic_dict['Muv'].value,
    #                                                                                          basic_dict[
    #                                                                                              'Muv_low'].value,
    #                                                                                          basic_dict[
    #                                                                                              'Muv_upp'].value))
    #
    # table.write('\\hline \n')

    # table.write('$\\alpha_{\\rm{UV}}$' + '& ${:.2f}_{{{:+.2f}}}^{{{:+.2f}}}$ \\\\ \n'.format(blue_cont_result[
    #                                                                                              'PL_slope'][0],
    #                                                                                          blue_cont_result[
    #                                                                                              'PL_slope'][1],
    #                                                                                          blue_cont_result[
    #                                                                                              'PL_slope'][2]))

    table.write(
        "$M_{1450}/(\\rm{mag})$"
        + " & \\multicolumn{{3}}{{c}}{{${:.2f}_{{{:+.2f}}}^{{{:+.2f}}}$}}"
        " \\\\ \n".format(
            basic_dict["M1450"].value,
            basic_dict["M1450_upp"].value,
            basic_dict["M1450_low"].value,
        )
    )

    table.write(
        "$\\alpha_{\\rm{OPT}}$"
        + " & ${:.2f}_{{{:+.2f}}}^{{{:+.2f}}}$  & ${:.2f}_{{{:+.2f}}}^{{{:+.2f}}}$ & ${:.2f}_{{{:+.2f}}}^{{{:+.2f}}}$  \\\\ \n".format(
            result_dict_a["PL_slope"].values[0],
            result_dict_a["PL_slope_low"].values[0],
            result_dict_a["PL_slope_high"].values[0],
            result_dict_b["PL_slope"].values[0],
            result_dict_b["PL_slope_low"].values[0],
            result_dict_b["PL_slope_high"].values[0],
            result_dict_c["PL_slope"].values[0],
            result_dict_c["PL_slope_low"].values[0],
            result_dict_c["PL_slope_high"].values[0],
        )
    )

    table.write("\\hline \n")

    table.write(
        "$\\rm{FWHM}_{\\rm{narrow}}/(\\rm{km}\\,\\rm{s}^{-1})$"
        + " & ${:.2f}_{{{:+.2f}}}^{{{:+.2f}}}$  & ${:.2f}_{{{:+.2f}}}^{{{:+.2f}}}$ & ${:.2f}_{{{:+.2f}}}^{{{:+.2f}}}$  \\\\ \n".format(
            result_dict_a["fwhm_O3"].values[0],
            result_dict_a["fwhm_O3_low"].values[0],
            result_dict_a["fwhm_O3_high"].values[0],
            result_dict_b["fwhm_O3"].values[0],
            result_dict_b["fwhm_O3_low"].values[0],
            result_dict_b["fwhm_O3_high"].values[0],
            result_dict_c["fwhm_O3"].values[0],
            result_dict_c["fwhm_O3_low"].values[0],
            result_dict_c["fwhm_O3_high"].values[0],
        )
    )

    table.write(
        "$\\rm{FWHM}_{\\rm{broad}}/(\\rm{km}\\,\\rm{s}^{-1})$"
        + " & ${:.2f}_{{{:+.2f}}}^{{{:+.2f}}}$  & ${:.2f}_{{{:+.2f}}}^{{{:+.2f}}}$ & ${:.2f}_{{{:+.2f}}}^{{{:+.2f}}}$  \\\\ \n".format(
            result_dict_a["fwhm_b"].values[0],
            result_dict_a["fwhm_low"].values[0],
            result_dict_a["fwhm_high"].values[0],
            result_dict_b["fwhm_b"].values[0],
            result_dict_b["fwhm_low"].values[0],
            result_dict_b["fwhm_high"].values[0],
            result_dict_c["fwhm_b"].values[0],
            result_dict_c["fwhm_low"].values[0],
            result_dict_c["fwhm_high"].values[0],
        )
    )

    table.write(
        "$\\rm{EW}_{\\rm{[OIII]}5007}/(\\textrm{\\AA})$"
        + " & ${:.2f}_{{{:+.2f}}}^{{{:+.2f}}}$  & ${:.2f}_{{{:+.2f}}}^{{{:+.2f}}}$ & ${:.2f}_{{{:+.2f}}}^{{{:+.2f}}}$  \\\\ \n".format(
            result_dict_a["ew_o3_5007"].values[0],
            result_dict_a["ew_o3_5007_low"].values[0],
            result_dict_a["ew_o3_5007_high"].values[0],
            result_dict_b["ew_o3_5007"].values[0],
            result_dict_b["ew_o3_5007_low"].values[0],
            result_dict_b["ew_o3_5007_high"].values[0],
            result_dict_c["ew_o3_5007"].values[0],
            result_dict_c["ew_o3_5007_low"].values[0],
            result_dict_c["ew_o3_5007_high"].values[0],
        )
    )

    table.write(
        "$\\rm{EW}_{\\rm{[OIII]}4960}/(\\textrm{\\AA})$"
        + " & ${:.2f}_{{{:+.2f}}}^{{{:+.2f}}}$  & ${:.2f}_{{{:+.2f}}}^{{{:+.2f}}}$ & ${:.2f}_{{{:+.2f}}}^{{{:+.2f}}}$  \\\\ \n".format(
            result_dict_a["ew_o3_4960"].values[0],
            result_dict_a["ew_o3_4960_low"].values[0],
            result_dict_a["ew_o3_4960_high"].values[0],
            result_dict_b["ew_o3_4960"].values[0],
            result_dict_b["ew_o3_4960_low"].values[0],
            result_dict_b["ew_o3_4960_high"].values[0],
            result_dict_c["ew_o3_4960"].values[0],
            result_dict_c["ew_o3_4960_low"].values[0],
            result_dict_c["ew_o3_4960_high"].values[0],
        )
    )

    # table.write('$\\rm{EW}_{\\rm{HeII}4687}/(\\textrm{\AA})$' +
    #             ' & ${:.2f}_{{{:+.2f}}}^{{{:+.2f}}}$  & ${:.2f}_{{{:+.2f}}}^{{{:+.2f}}}$ & ${:.2f}_{{{:+.2f}}}^{{{:+.2f}}}$  \\\\ \n'.format(
    #     result_dict_a['ew_he2'].values[0], result_dict_a['ew_he2_low'].values[0],
    #     result_dict_a['ew_he2_high'].values[0], result_dict_b['ew_he2'].values[0],
    #     result_dict_b['ew_he2_low'].values[0], result_dict_b['ew_he2_high'].values[0],
    #     result_dict_c['ew_he2'].values[0], result_dict_c['ew_he2_low'].values[0],
    #     result_dict_c['ew_he2_high'].values[0]))

    table.write(
        "$\\rm{EW}_{\\rm{[NeIII]}3869}/(\\textrm{\\AA})$"
        + " & ${:.2f}_{{{:+.2f}}}^{{{:+.2f}}}$  & ${:.2f}_{{{:+.2f}}}^{{{:+.2f}}}$ & ${:.2f}_{{{:+.2f}}}^{{{:+.2f}}}$  \\\\ \n".format(
            result_dict_a["ew_ne3_a"].values[0],
            result_dict_a["ew_ne3_a_low"].values[0],
            result_dict_a["ew_ne3_a_high"].values[0],
            result_dict_b["ew_ne3_a"].values[0],
            result_dict_b["ew_ne3_a_low"].values[0],
            result_dict_b["ew_ne3_a_high"].values[0],
            result_dict_c["ew_ne3_a"].values[0],
            result_dict_c["ew_ne3_a_low"].values[0],
            result_dict_c["ew_ne3_a_high"].values[0],
        )
    )

    table.write(
        "$\\rm{Flux}_{\\rm{H}\\beta, \\rm{broad}}/(10^{-20}\\,\\rm{erg}\\,\\rm{s}^{-1}\\,\\rm{cm}^{-2})"
        + " & ${:.2f}_{{{:+.2f}}}^{{{:+.2f}}}$ "
        " & ${:.2f}_{{{:+.2f}}}^{{{:+.2f}}}$ "
        " & ${:.2f}_{{{:+.2f}}}^{{{:+.2f}}}$ "
        " \\\\ \n".format(
            result_dict_a["flux_hb_br"].values[0] / 1e-20,
            result_dict_a["flux_hb_br_low"].values[0] / 1e-20,
            result_dict_a["flux_hb_br_high"].values[0] / 1e-20,
            result_dict_b["flux_hb_br"].values[0] / 1e-20,
            result_dict_b["flux_hb_br_low"].values[0] / 1e-20,
            result_dict_b["flux_hb_br_high"].values[0] / 1e-20,
            result_dict_c["flux_hb_br"].values[0] / 1e-20,
            result_dict_c["flux_hb_br_low"].values[0] / 1e-20,
            result_dict_c["flux_hb_br_high"].values[0] / 1e-20,
        )
    )

    table.write(
        "$\\rm{Flux}_{\\rm{H}\\beta, \\rm{narrow}}/(10^{-20}\\,\\rm{erg}\\,\\rm{s}^{-1}\\,\\rm{cm}^{-2})"
        + " & ${:.2f}_{{{:+.2f}}}^{{{:+.2f}}}$ "
        " & ${:.2f}_{{{:+.2f}}}^{{{:+.2f}}}$ "
        " & ${:.2f}_{{{:+.2f}}}^{{{:+.2f}}}$ "
        " \\\\ \n".format(
            result_dict_a["flux_hb_nl"].values[0] / 1e-20,
            result_dict_a["flux_hb_nl_low"].values[0] / 1e-20,
            result_dict_a["flux_hb_nl_high"].values[0] / 1e-20,
            result_dict_b["flux_hb_nl"].values[0] / 1e-20,
            result_dict_b["flux_hb_nl_low"].values[0] / 1e-20,
            result_dict_b["flux_hb_nl_high"].values[0] / 1e-20,
            result_dict_c["flux_hb_nl"].values[0] / 1e-20,
            result_dict_c["flux_hb_nl_low"].values[0] / 1e-20,
            result_dict_c["flux_hb_nl_high"].values[0] / 1e-20,
        )
    )

    table.write(
        "$\\rm{Flux}_{\\rm{H}\\gamma, \\rm{broad}}/(10^{-20}\\,\\rm{erg}\\,\\rm{s}^{-1}\\,\\rm{cm}^{-2})"
        + " & ${:.2f}_{{{:+.2f}}}^{{{:+.2f}}}$ "
        " & ${:.2f}_{{{:+.2f}}}^{{{:+.2f}}}$ "
        " & ${:.2f}_{{{:+.2f}}}^{{{:+.2f}}}$ "
        " \\\\ \n".format(
            result_dict_a["flux_hg_br"].values[0] / 1e-20,
            result_dict_a["flux_hg_br_low"].values[0] / 1e-20,
            result_dict_a["flux_hg_br_high"].values[0] / 1e-20,
            result_dict_b["flux_hg_br"].values[0] / 1e-20,
            result_dict_b["flux_hg_br_low"].values[0] / 1e-20,
            result_dict_b["flux_hg_br_high"].values[0] / 1e-20,
            result_dict_c["flux_hg_br"].values[0] / 1e-20,
            result_dict_c["flux_hg_br_low"].values[0] / 1e-20,
            result_dict_c["flux_hg_br_high"].values[0] / 1e-20,
        )
    )

    table.write(
        "$\\rm{Flux}_{\\rm{H}\\gamma, \\rm{narrow}}/(10^{-20}\\,\\rm{erg}\\,\\rm{s}^{-1}\\,\\rm{cm}^{-2})"
        + " & ${:.2f}_{{{:+.2f}}}^{{{:+.2f}}}$ "
        " & ${:.2f}_{{{:+.2f}}}^{{{:+.2f}}}$ "
        " & ${:.2f}_{{{:+.2f}}}^{{{:+.2f}}}$ "
        " \\\\ \n".format(
            result_dict_a["flux_hg_nl"].values[0] / 1e-20,
            result_dict_a["flux_hg_nl_low"].values[0] / 1e-20,
            result_dict_a["flux_hg_nl_high"].values[0] / 1e-20,
            result_dict_b["flux_hg_nl"].values[0] / 1e-20,
            result_dict_b["flux_hg_nl_low"].values[0] / 1e-20,
            result_dict_b["flux_hg_nl_high"].values[0] / 1e-20,
            result_dict_c["flux_hg_nl"].values[0] / 1e-20,
            result_dict_c["flux_hg_nl_low"].values[0] / 1e-20,
            result_dict_c["flux_hg_nl_high"].values[0] / 1e-20,
        )
    )

    table.write("\\hline \n")

    # table content
    table.write("$\\log(L_{5100}/( \\rm{erg}\\,\\rm{s}^{-1}\\,\\rm{\\AA}^{-1}))$ ")
    table.write(
        " & ${:.2f}_{{{:+.2f}}}^{{{:+.2f}}}$  & ${:.2f}_{{{:+.2f}}}^{{{:+.2f}}}$ & ${:.2f}_{{{:+.2f}}}^{{{:+.2f}}}$  \\\\ \n".format(
            result_dict_a["L5100"].values[0],
            result_dict_a["L5100_low"].values[0],
            result_dict_a["L5100_high"].values[0],
            result_dict_b["L5100"].values[0],
            result_dict_b["L5100_low"].values[0],
            result_dict_b["L5100_high"].values[0],
            result_dict_c["L5100"].values[0],
            result_dict_c["L5100_low"].values[0],
            result_dict_c["L5100_high"].values[0],
        )
    )

    table.write("$\\log(L_{\\rm{bol}}/(\\rm{erg}\\,\\rm{s}^{-1}))$ ")
    table.write(
        " & ${:.2f}_{{{:+.2f}}}^{{{:+.2f}}}$  & ${:.2f}_{{{:+.2f}}}^{{{:+.2f}}}$ & ${:.2f}_{{{:+.2f}}}^{{{:+.2f}}}$  \\\\ \n".format(
            result_dict_a["Lbol"].values[0],
            result_dict_a["Lbol_low"].values[0],
            result_dict_a["Lbol_high"].values[0],
            result_dict_b["Lbol"].values[0],
            result_dict_b["Lbol_low"].values[0],
            result_dict_b["Lbol_high"].values[0],
            result_dict_c["Lbol"].values[0],
            result_dict_c["Lbol_low"].values[0],
            result_dict_c["Lbol_high"].values[0],
        )
    )

    table.write("$\\log(L_{\\rm{H}\\beta}/(\\rm{erg}\\,\\rm{s}^{-1}))$ ")
    table.write(
        " & ${:.2f}_{{{:+.2f}}}^{{{:+.2f}}}$  & ${:.2f}_{{{:+.2f}}}^{{{:+.2f}}}$ & ${:.2f}_{{{:+.2f}}}^{{{:+.2f}}}$  \\\\ \n".format(
            result_dict_a["L_Hb"].values[0],
            result_dict_a["L_Hb_low"].values[0],
            result_dict_a["L_Hb_high"].values[0],
            result_dict_b["L_Hb"].values[0],
            result_dict_b["L_Hb_low"].values[0],
            result_dict_b["L_Hb_high"].values[0],
            result_dict_c["L_Hb"].values[0],
            result_dict_c["L_Hb_low"].values[0],
            result_dict_c["L_Hb_high"].values[0],
        )
    )

    table.write("$\\log(L_{\\rm{bol, H}\\beta}/(\\rm{erg}\\,\\rm{s}^{-1}))$ ")
    table.write(
        " & ${:.2f}_{{{:+.2f}}}^{{{:+.2f}}}$  & ${:.2f}_{{{:+.2f}}}^{{{:+.2f}}}$ & ${:.2f}_{{{:+.2f}}}^{{{:+.2f}}}$  \\\\ \n".format(
            result_dict_a["Lbol_Hb"].values[0],
            result_dict_a["Lbol_Hb_low"].values[0],
            result_dict_a["Lbol_Hb_high"].values[0],
            result_dict_b["Lbol_Hb"].values[0],
            result_dict_b["Lbol_Hb_low"].values[0],
            result_dict_b["Lbol_Hb_high"].values[0],
            result_dict_c["Lbol_Hb"].values[0],
            result_dict_c["Lbol_Hb_low"].values[0],
            result_dict_c["Lbol_Hb_high"].values[0],
        )
    )

    table.write("$M_{\\rm{BH, GH05, LH}\\beta}/(10^{7}\\,\\rm{M}_{\\odot})$ ")
    table.write(
        " & ${:.2f}_{{{:+.2f}}}^{{{:+.2f}}}$  & ${:.2f}_{{{:+.2f}}}^{{{:+.2f}}}$ & ${:.2f}_{{{:+.2f}}}^{{{:+.2f}}}$  \\\\ \n".format(
            result_dict_a["mbh_gh05_LHb"].values[0],
            result_dict_a["mbh_gh05_LHb_low"].values[0],
            result_dict_a["mbh_gh05_LHb_high"].values[0],
            result_dict_b["mbh_gh05_LHb"].values[0],
            result_dict_b["mbh_gh05_LHb_low"].values[0],
            result_dict_b["mbh_gh05_LHb_high"].values[0],
            result_dict_c["mbh_gh05_LHb"].values[0],
            result_dict_c["mbh_gh05_LHb_low"].values[0],
            result_dict_c["mbh_gh05_LHb_high"].values[0],
        )
    )

    table.write("$\\lambda_{\\rm{Edd, GH05, LH}\\beta}$ ")
    table.write(
        " & ${:.2f}_{{{:+.2f}}}^{{{:+.2f}}}$  & ${:.2f}_{{{:+.2f}}}^{{{:+.2f}}}$ & ${:.2f}_{{{:+.2f}}}^{{{:+.2f}}}$  \\\\ \n".format(
            result_dict_a["eddr_gh05_LHb"].values[0],
            result_dict_a["eddr_gh05_LHb_low"].values[0],
            result_dict_a["eddr_gh05_LHb_high"].values[0],
            result_dict_b["eddr_gh05_LHb"].values[0],
            result_dict_b["eddr_gh05_LHb_low"].values[0],
            result_dict_b["eddr_gh05_LHb_high"].values[0],
            result_dict_c["eddr_gh05_LHb"].values[0],
            result_dict_c["eddr_gh05_LHb_low"].values[0],
            result_dict_c["eddr_gh05_LHb_high"].values[0],
        )
    )

    table.write("\\hline \n")

    table.write("$M_{\\rm{BH, VP06}}/(10^{7}\\,\\rm{M}_{\\odot})$ ")
    table.write(
        " & ${:.2f}_{{{:+.2f}}}^{{{:+.2f}}}$  & ${:.2f}_{{{:+.2f}}}^{{{:+.2f}}}$ & ${:.2f}_{{{:+.2f}}}^{{{:+.2f}}}$  \\\\ \n".format(
            result_dict_a["mbh_vp06"].values[0],
            result_dict_a["mbh_vp06_low"].values[0],
            result_dict_a["mbh_vp06_high"].values[0],
            result_dict_b["mbh_vp06"].values[0],
            result_dict_b["mbh_vp06_low"].values[0],
            result_dict_b["mbh_vp06_high"].values[0],
            result_dict_c["mbh_vp06"].values[0],
            result_dict_c["mbh_vp06_low"].values[0],
            result_dict_c["mbh_vp06_high"].values[0],
        )
    )

    table.write("$\\lambda_{\\rm{Edd, VP06}}$ ")
    table.write(
        " & ${:.2f}_{{{:+.2f}}}^{{{:+.2f}}}$  & ${:.2f}_{{{:+.2f}}}^{{{:+.2f}}}$ & ${:.2f}_{{{:+.2f}}}^{{{:+.2f}}}$  \\\\ \n".format(
            result_dict_a["eddr_vp06"].values[0],
            result_dict_a["eddr_vp06_low"].values[0],
            result_dict_a["eddr_vp06_high"].values[0],
            result_dict_b["eddr_vp06"].values[0],
            result_dict_b["eddr_vp06_low"].values[0],
            result_dict_b["eddr_vp06_high"].values[0],
            result_dict_c["eddr_vp06"].values[0],
            result_dict_c["eddr_vp06_low"].values[0],
            result_dict_c["eddr_vp06_high"].values[0],
        )
    )

    table.write("$M_{\\rm{BH, GH05}}/(10^{7}\\,\\rm{M}_{\\odot})$ ")
    table.write(
        " & ${:.2f}_{{{:+.2f}}}^{{{:+.2f}}}$  & ${:.2f}_{{{:+.2f}}}^{{{:+.2f}}}$ & ${:.2f}_{{{:+.2f}}}^{{{:+.2f}}}$  \\\\ \n".format(
            result_dict_a["mbh_gh05_L5100"].values[0],
            result_dict_a["mbh_gh05_L5100_low"].values[0],
            result_dict_a["mbh_gh05_L5100_high"].values[0],
            result_dict_b["mbh_gh05_L5100"].values[0],
            result_dict_b["mbh_gh05_L5100_low"].values[0],
            result_dict_b["mbh_gh05_L5100_high"].values[0],
            result_dict_c["mbh_gh05_L5100"].values[0],
            result_dict_c["mbh_gh05_L5100_low"].values[0],
            result_dict_c["mbh_gh05_L5100_high"].values[0],
        )
    )

    table.write("$\\lambda_{\\rm{Edd, GH05}}$")
    table.write(
        " & ${:.2f}_{{{:+.2f}}}^{{{:+.2f}}}$  & ${:.2f}_{{{:+.2f}}}^{{{:+.2f}}}$ & ${:.2f}_{{{:+.2f}}}^{{{:+.2f}}}$  \\\\ \n".format(
            result_dict_a["eddr_gh05_L5100"].values[0],
            result_dict_a["eddr_gh05_L5100_low"].values[0],
            result_dict_a["eddr_gh05_L5100_high"].values[0],
            result_dict_b["eddr_gh05_L5100"].values[0],
            result_dict_b["eddr_gh05_L5100_low"].values[0],
            result_dict_b["eddr_gh05_L5100_high"].values[0],
            result_dict_c["eddr_gh05_L5100"].values[0],
            result_dict_c["eddr_gh05_L5100_low"].values[0],
            result_dict_c["eddr_gh05_L5100_high"].values[0],
        )
    )

    table.write("$M_{\\rm{BH, P12}}/(10^{7}\\,\\rm{M}_{\\odot})$ ")
    table.write(
        " & ${:.2f}_{{{:+.2f}}}^{{{:+.2f}}}$  & ${:.2f}_{{{:+.2f}}}^{{{:+.2f}}}$ & ${:.2f}_{{{:+.2f}}}^{{{:+.2f}}}$  \\\\ \n".format(
            result_dict_a["mbh_p12"].values[0],
            result_dict_a["mbh_p12_low"].values[0],
            result_dict_a["mbh_p12_high"].values[0],
            result_dict_b["mbh_p12"].values[0],
            result_dict_b["mbh_p12_low"].values[0],
            result_dict_b["mbh_p12_high"].values[0],
            result_dict_c["mbh_p12"].values[0],
            result_dict_c["mbh_p12_low"].values[0],
            result_dict_c["mbh_p12_high"].values[0],
        )
    )

    table.write("$\\lambda_{\\rm{Edd, P12}}$ ")
    table.write(
        " & ${:.2f}_{{{:+.2f}}}^{{{:+.2f}}}$  & ${:.2f}_{{{:+.2f}}}^{{{:+.2f}}}$ & ${:.2f}_{{{:+.2f}}}^{{{:+.2f}}}$  \\\\ \n".format(
            result_dict_a["eddr_p12"].values[0],
            result_dict_a["eddr_p12_low"].values[0],
            result_dict_a["eddr_p12_high"].values[0],
            result_dict_b["eddr_p12"].values[0],
            result_dict_b["eddr_p12_low"].values[0],
            result_dict_b["eddr_p12_high"].values[0],
            result_dict_c["eddr_p12"].values[0],
            result_dict_c["eddr_p12_low"].values[0],
            result_dict_c["eddr_p12_high"].values[0],
        )
    )

    table.write("\\hline \n")
    table.write("\\hline \n")
    table.write("\\label{tab:spec_info} \n")
    table.write("\\end{tabular} \n")
    # table.write('\\\\ \n')
    # table.write('\\footnotesize{ \\textbf{Notes.}} \n')
    table.write("\\end{table} \n")
    table.close()


def get_dereddened_properties(
    spec, redshift, fit_range=[31000, 43000], flux_factor=1.0, save_dir="v2_fit_mcmc"
):
    """Get the dereddened properties of the AGN.

    :param spec:
    :param redshift:
    :param fit_range:
    :param flux_factor:
    :param save_dir:
    :param a_v:
    :return:
    """

    # Generate the path to the output directory
    dir_path = pathlib.Path(f"../../data/{save_dir}")
    dir_path.mkdir(parents=True, exist_ok=True)

    chain_path = os.path.join(dir_path, "flat_chain.csv")

    # Instantiate the model
    model = agn_model(spec, redshift, fit_range, flux_factor)

    (
        comp_pl,
        comp_o3_a,
        comp_o3_b,
        comp_hb_nl,
        comp_hg_nl,
        comp_o3_c,
        comp_hb_br,
        comp_hg_br,
        comp_ne3,
        comp_he2,
    ) = model.components

    # Build the list of variable parameters
    model.get_params_to_sample()

    # Load the flat chain
    dispersion = get_wave_grid(fit_range[0], fit_range[1], 100)
    flat_chain = pd.read_csv(chain_path)
    params_values = flat_chain.values
    model.flat_chain = flat_chain.values

    model.nwalkers = 40
    model.nsteps = 3000

    cont_dict = {
        "feature_name": "power_law",
        "components": [comp_pl],
        "rest_frame_wavelengths": [1450, 5100],
    }

    emfeat_dictlist = [
        {
            "feature_name": "Hb_broad",
            "components": [comp_hb_br],
            "rest_frame_wavelength": 4862.68,
        },
        {
            "feature_name": "Hb_total",
            "components": [comp_hb_br, comp_hb_nl],
            "rest_frame_wavelength": 4862.68,
        },
    ]

    cosmology = FlatLambdaCDM(H0=70, Om0=0.3)

    # Setting analysis variables
    cont_components = [comp_pl]
    fwhm_method = "spline"
    disp_range = None
    result_table = None
    dispersion_unit = spec.dispersion_unit
    fluxden_unit = spec.fluxden_unit

    # Read in the a_v values from the other fit and sample them
    df = pd.read_csv("../../data/scattered_light_fit/scattered_light_flat_chain.csv")
    a_v_arr = df.sample(flat_chain.shape[0])["plav_a_v"].values

    # Manual rewrite of the analysis routines
    for idx in tqdm(range(flat_chain.shape[0])):
        # Sample av here from other chain
        a_v = a_v_arr[idx]
        # Get chain values
        params_values = flat_chain.values[idx, :]

        # Continuum component (the same for all emission features)
        cont_fluxden = model.eval(
            dispersion,
            params_values,
            components=cont_components,
            broaden_by_resolution=False,
        )
        ext = extinction.calzetti00(dispersion / (1 + redshift), a_v, 4.05)
        cont_fluxden = extinction.remove(ext, cont_fluxden)
        cont_spec = sod.SpecOneD(
            dispersion=dispersion,
            fluxden=cont_fluxden,
            dispersion_unit=dispersion_unit,
            fluxden_unit=fluxden_unit,
        )

        result_dict = {}

        for dict in emfeat_dictlist:
            components = dict["components"]
            feature_name = dict["feature_name"]
            rest_frame_wavelength = dict["rest_frame_wavelength"]

            # Emission line component
            model_fluxden = model.eval(
                dispersion,
                params_values,
                components=components,
                broaden_by_resolution=False,
            )

            ext = extinction.calzetti00(dispersion / (1 + redshift), a_v, 4.05)
            model_fluxden = extinction.remove(ext, model_fluxden)

            model_spec = sod.SpecOneD(
                dispersion=dispersion,
                fluxden=model_fluxden,
                dispersion_unit=dispersion_unit,
                fluxden_unit=fluxden_unit,
            )

            # Line analysis routine here
            ew = scana.get_equivalent_width(cont_spec, model_spec, redshift=redshift)
            result_dict.update({feature_name + "_EW": ew})

            fwhm = scana.get_fwhm(model_spec, method=fwhm_method)
            result_dict.update({feature_name + "_FWHM": fwhm})

            flux = scana.get_integrated_flux(model_spec, disp_range=disp_range)
            result_dict.update({feature_name + "_flux": flux})

            lum = scana.calc_integrated_luminosity(
                model_spec, redshift=redshift, cosmology=cosmology
            )
            result_dict.update({feature_name + "_lum": lum})

        if result_table is None:
            # Initialize QTable for results
            # Initialize column names
            result_table = QTable(names=result_dict.keys())
            # Initialize column units
            for column in result_table.columns:
                if (
                    isinstance(result_dict[column], u.Quantity)
                    or isinstance(result_dict[column], u.IrreducibleUnit)
                    or isinstance(result_dict[column], u.CompositeUnit)
                ):
                    result_table[column].unit = result_dict[column].unit

        result_table.add_row(result_dict)

    rt = result_table

    # Read in the dispersion information
    hdu = fits.open("../../data/jwst_nirspec_prism_disp.fits")
    disp = hdu[1].data["WAVELENGTH"]
    res_arr = hdu[1].data["R"]
    c_km_s = const.c.to("km/s")
    idx = np.argmin(np.abs(disp * 10000 - 4862.68 * (1 + redshift)))
    res = res_arr[idx]

    resolution_km_s = c_km_s / res

    print(f"Resolution at Hb: {res:.2f}")
    print(f"Resolution at Hb: {resolution_km_s:.2f}")

    # Analze the results
    Hb_fwhm = rt["Hb_broad_FWHM"]
    fwhm = np.sqrt(Hb_fwhm.value**2 - resolution_km_s.value**2) * u.km / u.s
    fwhm_b = np.nanpercentile(fwhm, [15.87, 50, 84.13])
    hb_lwav = rt["Hb_total_lum"]
    hb_lwav_broad = rt["Hb_broad_lum"]
    L_Hb = np.nanpercentile(hb_lwav, q=[15.87, 50, 84.13])
    # L_Hb_broad = np.nanpercentile(hb_lwav_broad, q=[15.87, 50, 84.13])
    L5100_from_Hb = get_L5100_from_Hb_lum(
        np.nanpercentile(hb_lwav.value, q=[15.87, 50, 84.13])
    )
    # L5100_from_Hb_broad = get_L5100_from_Hb_lum(np.nanpercentile(hb_lwav_broad.value, q=[15.87, 50, 84.13]))
    lbol_from_Hb = 9.26 * L5100_from_Hb
    # lbol_from_Hb_broad = 9.26 * L5100_from_Hb_broad

    # Using the total Hb line luminosity
    m_bh_gh05_LHb = qso.se_bhmass_gh05(fwhm, hb_lwav, "Hb_LHb")
    edd_r_gh05_LHb = qso.calc_eddington_ratio(9.26 * L5100_from_Hb[0], m_bh_gh05_LHb[0])

    mbh_gh05_LHb = np.nanpercentile(m_bh_gh05_LHb[0], [15.87, 50, 84.13]) / 1e7
    eddr_gh05_LHb = np.nanpercentile(edd_r_gh05_LHb, [15.87, 50, 84.13])

    # Save the results
    result_dict = {
        "fwhm_b": fwhm[1],
        "fwhm_low": fwhm[0],
        "fwhm_high": fwhm[2],
        "L_Hb": np.log10(L_Hb[1].value),
        "L_Hb_low": np.log10(L_Hb[0].value) - np.log10(L_Hb[1].value),
        "L_Hb_high": np.log10(L_Hb[2].value) - np.log10(L_Hb[1].value),
        "L5100_from_Hb": L5100_from_Hb[1],
        "L5100_from_Hb_low": L5100_from_Hb[0],
        "L5100_from_Hb_high": L5100_from_Hb[2],
        "Lbol_Hb": np.log10(lbol_from_Hb[1]),
        "Lbol_Hb_low": np.log10(lbol_from_Hb[0]) - np.log10(lbol_from_Hb[1]),
        "Lbol_Hb_high": np.log10(lbol_from_Hb[2]) - np.log10(lbol_from_Hb[1]),
        # 'Lbol_Hb': np.log10(lbol_from_Hb_broad[1]),
        # 'Lbol_Hb_low': np.log10(lbol_from_Hb_broad[0]) - np.log10(lbol_from_Hb_broad[1]),
        # 'Lbol_Hb_high': np.log10(lbol_from_Hb_broad[2]) - np.log10(lbol_from_Hb_broad[1]),
        "mbh_gh05_LHb": mbh_gh05_LHb[1],
        "mbh_gh05_LHb_low": mbh_gh05_LHb[0] - mbh_gh05_LHb[1],
        "mbh_gh05_LHb_high": mbh_gh05_LHb[2] - mbh_gh05_LHb[1],
        "eddr_gh05_LHb": eddr_gh05_LHb[1],
        "eddr_gh05_LHb_low": eddr_gh05_LHb[0] - eddr_gh05_LHb[1],
        "eddr_gh05_LHb_high": eddr_gh05_LHb[2] - eddr_gh05_LHb[1],
    }

    return result_dict


def create_paper_table_dered(dered_results):

    # Create a table with the basic spectral properties
    table = open("../../tex_tables/agn_spec_results_dered.tex", "w+")

    table.write("\\begin{table}[h!]\n")
    table.write("\\renewcommand{\\arraystretch}{1.3}\n")
    table.write("\\footnotesize \n")
    table.write("\\centering \n")
    table.write("\\caption{\\textbf{Dereddened AGN Properties}}\n")

    table.write("\\begin{tabular}{ll} \n")
    table.write("\\small \n")

    # tablehead
    table.write(" Property &  Value \\\\ \n")

    table.write("\\hline \n")

    # Result from the scattered light fit
    df = pd.read_csv("../../data/scattered_light_fit/agn_luminosities.csv")

    table.write("$m_{1450}/(\\rm{mag})$ &")
    table.write(
        "${:.2f}_{{{:+.2f}}}^{{{:+.2f}}}$ \\\\ \n".format(
            df.loc[2, "med"],
            (df.loc[2, "low"] - df.loc[2, "med"]),
            (df.loc[2, "upp"] - df.loc[2, "med"]),
        )
    )

    table.write("$M_{1450}/(\\rm{mag})$ &")
    table.write(
        "${:.2f}_{{{:+.2f}}}^{{{:+.2f}}}$ \\\\ \n".format(
            df.loc[3, "med"],
            (df.loc[3, "low"] - df.loc[3, "med"]),
            (df.loc[3, "upp"] - df.loc[3, "med"]),
        )
    )

    table.write("$\\log(L_{5100}/(\\rm{erg}\\,\\rm{s}^{-1}\\,\\rm{\\AA}^{-1}))$ &")
    table.write(
        "${:.2f}_{{{:+.2f}}}^{{{:+.2f}}}$ \\\\ \n".format(
            np.log10(df.loc[0, "med"]),
            np.log10(df.loc[0, "low"]) - np.log10(df.loc[0, "med"]),
            np.log10(df.loc[0, "upp"]) - np.log10(df.loc[0, "med"]),
        )
    )

    table.write("$\\log(L_{\\rm{bol}}/(\\rm{erg}\\,\\rm{s}^{-1}))$ &")
    table.write(
        "${:.2f}_{{{:+.2f}}}^{{{:+.2f}}}$ \\\\ \n".format(
            np.log10(df.loc[1, "med"]),
            np.log10(df.loc[1, "low"]) - np.log10(df.loc[1, "med"]),
            np.log10(df.loc[1, "upp"]) - np.log10(df.loc[1, "med"]),
        )
    )

    table.write("\\hline \n")

    # table content

    table.write("$\\log(L_{\\rm{H}\\beta}/(\\rm{erg}\\,\\rm{s}^{-1}))$ &")
    table.write(
        "${:.2f}_{{{:+.2f}}}^{{{:+.2f}}}$ \\\\ \n".format(
            dered_results["L_Hb"], dered_results["L_Hb_low"], dered_results["L_Hb_high"]
        )
    )

    table.write("$\\log(L_{\\rm{bol, H}\\beta}/(\\rm{erg}\\,\\rm{s}^{-1}))$ &")
    table.write(
        "${:.2f}_{{{:+.2f}}}^{{{:+.2f}}}$ \\\\ \n".format(
            dered_results["Lbol_Hb"],
            dered_results["Lbol_Hb_low"],
            dered_results["Lbol_Hb_high"],
        )
    )

    # table.write('$\log(L_{\\rm{bol, H}\\beta, \\rm{broad}}/(\\rm{erg}\,\\rm{s}^{-1}))$ &')
    # table.write('${:.2f}_{{{:+.2f}}}^{{{:+.2f}}}$ \\\\ \n'.format(
    #     dered_results['Lbol_Hb_broad'],
    #     dered_results['Lbol_Hb_broad_low'],
    #     dered_results['Lbol_Hb_broad_high']))

    table.write("$M_{\\rm{BH, GH05, LH}\\beta}/(10^{7}\\,\\rm{M}_{\\odot})$ &")
    table.write(
        "${:.2f}_{{{:+.2f}}}^{{{:+.2f}}}$ \\\\ \n".format(
            dered_results["mbh_gh05_LHb"].value,
            dered_results["mbh_gh05_LHb_low"].value,
            dered_results["mbh_gh05_LHb_high"].value,
        )
    )

    table.write("$\\lambda_{\\rm{Edd, GH05, LH}\\beta}$ &")
    table.write(
        "${:.2f}_{{{:+.2f}}}^{{{:+.2f}}}$ \\\\ \n".format(
            dered_results["eddr_gh05_LHb"].value,
            dered_results["eddr_gh05_LHb_low"].value,
            dered_results["eddr_gh05_LHb_high"].value,
        )
    )

    table.write("\\hline \n")
    table.write("\\hline \n")
    table.write("\\label{tab:dereddened_prop} \n")
    table.write("\\end{tabular} \n")
    # table.write('\\\\ \n')
    # table.write('\\footnotesize{ \\textbf{Notes.}} \n')
    table.write("\\end{table} \n")

    table.close()


if __name__ == "__main__":
    # Load new reduction
    # spec1d_path = os.path.join('/Users/jtschindler/Projects/2205_JWST_Light_Echo/jwst_lightecho/redux_new/spec_J1007'
    #                            '/pypeit/Science_coadd',
    #                            'spec1d_jw02073008001_03101_00001_nrs1_rate-jw02073008001_03101_00003_nrs1_rate-2073_4562.fits')
    # spec = utils.load_pypeit_spectrum(spec1d_path)

    # Load Fred's rerereduction of the J1007 AGN spectrum
    master_path = os.getenv("JWST_LIGHT_ECHO_PATH")
    data_path = os.path.join(master_path, "paper_seyfert/data")
    spec1d_path = os.path.join(data_path, "spec1d_J1007_AGN_WAVECORR.fits")
    spec = utils.load_pypeit_spectrum(spec1d_path)

    spec.fluxden = spec.fluxden.astype("float64")
    spec.dispersion = spec.dispersion.astype("float64")

    # Multiply by flux correction factor, empirically determined on the
    # by the average of the F115W, F277W, and F444W bands in the discovery plot
    spec.fluxden *= 1.0 / 0.8260
    spec.fluxden_err *= 1.0 / 0.8260
    spec.fluxden_ivar *= 0.8260**2

    # Read in the NIRSpec PRISM resolution file
    hdu = fits.open("../../data/jwst_nirspec_prism_disp.fits")
    wave = hdu[1].data["WAVELENGTH"] * 10000
    res = hdu[1].data["R"]

    # First, resample the resolving power to the wavelength grid of the spectrum
    res_interp = np.interp(spec.dispersion, wave, res)

    # Plot save directory
    save_dir = os.path.join("../../figures/")

    # Set initial fit redshift
    redshift = 7.2583

    # Set the suffix for the save directory
    save_suffix = "v1_new_fit_mcmc"

    # # Preliminary fit to the spectrum to determine the redshift
    prel_result_dict = preliminary_fit(
        spec,
        redshift,
        nwalkers=40,
        nsteps=2000,
        discard=1000,
        refit=False,
        save_dir="preliminary_v4_new_fit_mcmc",
    )

    # Calculate the observed fwhm @ 5007 AA and increase resolution by the
    # ratio to to allow for the measured [OIII]5007 fwhm to be resolved
    fwhm_o3_5007 = prel_result_dict["O3_5007_FWHM"][0].value
    c = const.c.to("km/s").value
    fwhm_o3 = c / np.interp((redshift + 1) * 5007, wave, res)
    fwhm = c / res_interp

    # TODO: The ratio is now set assuming an intrinsic FWHM @ 5007 AA of 1000 km/s
    #   This is slightly below the measured value of ~1130 km/s to allow for broader
    #   narrow line components

    fwhm_res = 1100

    ratio = fwhm_o3 / fwhm_res

    # # Quick plot of spectrum and resolving power
    # fig = plt.figure(figsize=(8, 6))
    # ax = fig.add_subplot(111)
    # ax2 = ax.twinx()
    #
    # ax3 = ax2.twinx()
    #
    # ax.plot(spec.dispersion, spec.fluxden, color='black', lw=1.5)
    # ax2.plot(wave, res, color='blue', lw=1.5, label='original')
    # ax2.plot(spec.dispersion, res_interp, color='red', lw=1.5, label='interpolated')
    # ax2.plot(spec.dispersion, res_interp*ratio, color=vermillion, lw=1.5, label='interp scaled')
    #
    # plt.legend()
    #
    # # Plot resolving power as fwhm in km/s
    # ax3.plot(spec.dispersion, fwhm, color=dblue, lw=1.5, label='fwhm')
    # ax3.plot((redshift+1)*5007, fwhm_o3_5007, 'o', color='green', label='[OIII]5007')
    # ax3.plot(spec.dispersion, fwhm/ratio, color='green', lw=1.5, label='fwhm')
    # plt.legend()
    #
    # plt.show()

    # Corrected resolution
    prism_resolution = res_interp * ratio

    print(f"[INFO The FWHM of the [OIII] 5007 line is {fwhm_o3_5007:.2f}")
    print(
        f"[INFO] The original resolution at [OIII]5007 is {res_interp[np.argmin(np.abs((redshift + 1) * 5007 - spec.dispersion))]:.2f}"
    )
    print(
        f"[INFO] The corrected resolution at [OIII]5007 is {prism_resolution[np.argmin(np.abs((redshift + 1) * 5007 - spec.dispersion))]:.2f}"
    )

    # Update redshift from fit
    redshift = prel_result_dict["redsh"][0]
    print("[INFO] New [OIII] redshift = {:.4f}".format(prel_result_dict["redsh"][0]))

    # Get basic spectral properties
    basic_dict = get_basic_spectral_properties(spec, redshift=redshift)

    # # Full fit to the spectrum, including the majority of emission lines
    fit_result = fit_spectrum(
        spec,
        redshift,
        nsteps=4000,
        discard=2000,
        nwalkers=40,
        save_dir=f"v4_new_fit_mcmc_r{fwhm_res}_test",
        refit=False,
        resolution=prism_resolution,
    )

    # Save the results to a table
    fr = fit_result.copy()
    for key in fr:
        if isinstance(fr[key], u.Quantity):
            fr[key] = fr[key].value
    pd.DataFrame(fr, index=[0]).to_csv(
        f"../../data/new_spectral_properties_r{fwhm_res}_test.csv"
    )

    # Create the latex table for the paper
    create_paper_table(
        prel_result_dict,
        basic_dict,
        fit_result,
        table_name=f"agn_spec_results_r{fwhm_res}_test.tex",
    )

    # Create the larger comparison version
    result_r1120 = pd.read_csv("../../data/new_spectral_properties_r1120_test.csv")
    result_r1100 = pd.read_csv("../../data/new_spectral_properties_r1100_test.csv")
    result_r1050 = pd.read_csv("../../data/new_spectral_properties_r1080_test.csv")

    create_paper_table_large(
        prel_result_dict, basic_dict, result_r1100, result_r1050, result_r1120
    )

    # # Get the dereddened properties according to the scattered light fit
    # fwhm_res = 1100
    # dered_result = get_dereddened_properties(spec, redshift,
    #                                          fit_range=[31000, 43000],
    #                                          flux_factor=1.0,
    #                                          save_dir='v4_new_fit_mcmc_r{}_test'.format(fwhm_res))
    #
    # # Save results to a table
    # fr = dered_result.copy()
    # for key in fr:
    #     if isinstance(fr[key], u.Quantity):
    #         fr[key] = fr[key].value
    # pd.DataFrame(fr, index=[0]).to_csv('../../data/new_dereddened_properties.csv'.format(fwhm_res))
    #
    # create_paper_table_dered(dered_result)
    #
