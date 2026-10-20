"""Ready-made QSO fit setups and line-property calculations.

Each function here is a named variant of a base function in
`sculptor_extensions.Qso_autofit`, with the template file, wavelength
range, emission line or scaling relations filled in:

- `setup_iron_template_<name>`: `Qso_autofit.setup_iron_template` for one
  iron template and wavelength range.
- `setup_<line>_lines`: `Qso_autofit.add_line_components` for the narrow,
  broad, ... components of one emission line (or line complex).
- `get_line_prop_<line>`:
  `Qso_autofit.get_line_properties` for one line and its BH-mass relations.

To add a variant, call the base function with your own settings.
"""

import logging
import os
import warnings

import numpy as np
from astropy import units as u

from sculptor_extensions import Qso_autofit as qa
from sculptor_extensions import qso

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------------------
# Iron templates (`Qso_autofit.setup_iron_template`)
# ------------------------------------------------------------------------------

# Iron is fit in the same windows as the continuum/power law. All arguments
# are passed on to `Qso_autofit.setup_iron_template` (see there for
# `fe_dict` and `fast`). Flux values of the templates are in units of
# 1e-17 erg/s/cm2 after the default scaling.


def setup_iron_template_VW01_UV(prefix, disp_range=None, fe_dict=None, fast=True):
    """
    Setup the Vestergaard & Wilkes 2001 iron template covering the UV range
    1075 - 3089 Å (`Fe_UVtemplt_A.asc`).

    :param prefix: component prefix
    :type prefix: str
    :param disp_range: optional [min, max] dispersion range (Å) to restrict
        the template to. Defaults to the full template range.
    :type disp_range: list
    :param fe_dict: iron template parameters, see
        `Qso_autofit.setup_iron_template`
    :type fe_dict: dict
    :param fast: use `TemplateModel` (default) instead of `template_model`
    :type fast: bool
    :return: model component and parameters for the iron template
    :rtype: sculptor.component.FitComponent, dict
    """
    return qa.setup_iron_template(
        prefix, "Fe_UVtemplt_A.asc", disp_range, fe_dict, fast
    )


def setup_iron_template_VW01_FUV(prefix, fe_dict=None, fast=True):
    """
    Setup the far-UV part (1074 - 2200 Å) of the Vestergaard & Wilkes 2001
    iron template.

    :param prefix: component prefix
    :type prefix: str
    :param fe_dict: iron template parameters, see
        `Qso_autofit.setup_iron_template`
    :type fe_dict: dict
    :param fast: use `TemplateModel` (default) instead of `template_model`
    :type fast: bool
    :return: model component and parameters for the iron template
    :rtype: sculptor.component.FitComponent, dict
    """
    return qa.setup_iron_template(
        prefix, "Fe_UVtemplt_A.asc", [1074, 2200], fe_dict, fast
    )


def setup_iron_template_VW01_CIV(prefix, fe_dict=None, fast=True):
    """
    Setup the part of the Vestergaard & Wilkes 2001 iron template around
    the CIV emission line, 1300 - 1800 Å. This covers all windows of
    `Qso_autofit.qso_contfe_CIV_Shen11` (1340 - 1705 Å, including the
    1340 - 1375 Å window for L1350); an older docstring said 1440 - 1710 Å
    (the span of the Shen et al. 2011 CIV windows alone). To be confirmed.

    :param prefix: component prefix
    :type prefix: str
    :param fe_dict: iron template parameters, see
        `Qso_autofit.setup_iron_template`
    :type fe_dict: dict
    :param fast: use `TemplateModel` (default) instead of `template_model`
    :type fast: bool
    :return: model component and parameters for the iron template
    :rtype: sculptor.component.FitComponent, dict
    """
    return qa.setup_iron_template(
        prefix, "Fe_UVtemplt_A.asc", [1300, 1800], fe_dict, fast
    )


def setup_iron_template_VW01_UV_MgII(prefix, fe_dict=None, fast=True):
    """
    Setup the part of the Vestergaard & Wilkes 2001 iron template around
    the MgII emission line, 2200 - 3090 Å.

    :param prefix: component prefix
    :type prefix: str
    :param fe_dict: iron template parameters, see
        `Qso_autofit.setup_iron_template`
    :type fe_dict: dict
    :param fast: use `TemplateModel` (default) instead of `template_model`
    :type fast: bool
    :return: model component and parameters for the iron template
    :rtype: sculptor.component.FitComponent, dict
    """
    return qa.setup_iron_template(
        prefix, "Fe_UVtemplt_A.asc", [2200, 3090], fe_dict, fast
    )


def setup_iron_template_VW01_Fe3(prefix, fe_dict=None, fast=True):
    """
    Setup the Vestergaard & Wilkes 2001 FeIII UV47 template
    (`Fe3_UV47.asc`; non-zero between 2379 - 2456 Å).

    :param prefix: component prefix
    :type prefix: str
    :param fe_dict: iron template parameters, see
        `Qso_autofit.setup_iron_template`
    :type fe_dict: dict
    :param fast: use `TemplateModel` (default) instead of `template_model`
    :type fast: bool
    :return: model component and parameters for the iron template
    :rtype: sculptor.component.FitComponent, dict
    """
    return qa.setup_iron_template(prefix, "Fe3_UV47.asc", None, fe_dict, fast)


def setup_iron_template_BG92_OPT(prefix, disp_range=None, fe_dict=None, fast=True):
    """
    Setup the Boroson & Green 1992 iron template covering the optical
    region 3685 - 7484 Å (`Fe_OPT_BR92_linear.txt`).

    :param prefix: component prefix
    :type prefix: str
    :param disp_range: optional [min, max] dispersion range (Å) to restrict
        the template to. Defaults to the full template range.
    :type disp_range: list
    :param fe_dict: iron template parameters, see
        `Qso_autofit.setup_iron_template`
    :type fe_dict: dict
    :param fast: use `TemplateModel` (default) instead of `template_model`
    :type fast: bool
    :return: model component and parameters for the iron template
    :rtype: sculptor.component.FitComponent, dict
    """
    return qa.setup_iron_template(
        prefix, "Fe_OPT_BR92_linear.txt", disp_range, fe_dict, fast
    )


def setup_iron_template_BG92_OPT_Hbeta(prefix, fe_dict=None, fast=True):
    """
    Setup the part of the Boroson & Green 1992 iron template around Hbeta,
    4435 - 5535 Å (from the Shen et al. 2011 continuum windows).

    :param prefix: component prefix
    :type prefix: str
    :param fe_dict: iron template parameters, see
        `Qso_autofit.setup_iron_template`
    :type fe_dict: dict
    :param fast: use `TemplateModel` (default) instead of `template_model`
    :type fast: bool
    :return: model component and parameters for the iron template
    :rtype: sculptor.component.FitComponent, dict
    """
    return qa.setup_iron_template(
        prefix, "Fe_OPT_BR92_linear.txt", [4435, 5535], fe_dict, fast
    )


def setup_iron_template_Tsuz_UV(prefix, disp_range=None, fe_dict=None, fast=True):
    """
    Setup the UV part of the Tsuzuki et al. 2006 iron template
    (`Tsuzuki06.txt`), 2200 - 3500 Å by default.

    :param prefix: component prefix
    :type prefix: str
    :param disp_range: optional [min, max] dispersion range (Å) to restrict
        the template to. Defaults to [2200, 3500].
    :type disp_range: list
    :param fe_dict: iron template parameters, see
        `Qso_autofit.setup_iron_template`
    :type fe_dict: dict
    :param fast: use `TemplateModel` (default) instead of `template_model`
    :type fast: bool
    :return: model component and parameters for the iron template
    :rtype: sculptor.component.FitComponent, dict
    """
    if disp_range is None:
        disp_range = [2200, 3500]
    return qa.setup_iron_template(prefix, "Tsuzuki06.txt", disp_range, fe_dict, fast)


def setup_iron_template_Tsuz_OPT(prefix, disp_range=None, fe_dict=None, fast=True):
    """
    Setup the optical part of the Tsuzuki et al. 2006 iron template
    (`Tsuzuki06.txt`, essentially based on the Boroson & Green 1992
    template), above 3500 Å by default.

    :param prefix: component prefix
    :type prefix: str
    :param disp_range: optional [min, max] dispersion range (Å) to restrict
        the template to. Defaults to [3500, max].
    :type disp_range: list
    :param fe_dict: iron template parameters, see
        `Qso_autofit.setup_iron_template`
    :type fe_dict: dict
    :param fast: use `TemplateModel` (default) instead of `template_model`
    :type fast: bool
    :return: model component and parameters for the iron template
    :rtype: sculptor.component.FitComponent, dict
    """
    if disp_range is None:
        disp_range = [3500, np.inf]
    return qa.setup_iron_template(prefix, "Tsuzuki06.txt", disp_range, fe_dict, fast)


def setup_iron_template_Park22_OPT_Hbeta(prefix, fe_dict=None, fast=True):
    """
    Setup the Park et al. 2022 optical iron template around Hbeta,
    4000 - 5600 Å (`Park_2022.asc`, flux density in
    1e-15 erg/s/cm2/Å, used unscaled).

    :param prefix: component prefix
    :type prefix: str
    :param fe_dict: iron template parameters, see
        `Qso_autofit.setup_iron_template`
    :type fe_dict: dict
    :param fast: use `TemplateModel` (default) instead of `template_model`
    :type fast: bool
    :return: model component and parameters for the iron template
    :rtype: sculptor.component.FitComponent, dict
    """
    return qa.setup_iron_template(
        prefix, "Park_2022.asc", None, fe_dict, fast, fluxden_scale=1.0
    )


# ------------------------------------------------------------------------------
# Emission lines (`Qso_autofit.add_line_components`)
# ------------------------------------------------------------------------------


def _renamed_kwarg(old, new, value):
    """Warn about a renamed keyword argument and return its value."""
    warnings.warn(f"`{old}` was renamed to `{new}`", DeprecationWarning, stacklevel=3)
    return value


# All `setup_<line>_lines` functions take the spectrum (unused, kept for a
# uniform signature), the emission line dictionary (see
# `Qso_autofit.initialize_gaussian_line_model` and
# `Qso_autofit.create_emission_line_dict`), the number of components of
# each kind, and the model to add the components to.


def setup_Halpha_NII_lines(spec, line_dict, narrow=True, Halpha_broad=2, model=None):
    """
    Setup Halpha+[NII] lines simultaneously at 6564.61 A, 6549.85 A and 6585.28 A:
    by default one narrow component for Halpha and the [NII] doublet
    ("Halpha_narrow", "NII_narrow") and two broad Halpha components
    ("Halpha_broad_a", "Halpha_broad_b"). Narrow components of Halpha and
    [NII] can be tied (same redshift and FWHM) in `line_dict`, as they
    originate from the same gas in the NLR.

    :param spec: input spectrum (unused)
    :type spec: speconed.speconed.SpecOneD
    :param line_dict: emission line dictionary; uses
        line_dict["Halpha"]["narrow" / "broad"] and line_dict["NII"]["narrow"]
    :type line_dict: dict
    :param narrow: fit narrow component for Halpha and [NII] lines
    :type narrow: bool
    :param Halpha_broad: number of Halpha broad components to fit (default 2)
    :type Halpha_broad: int
    :param model: model to add the components and parameters to
    :type model: sculptor.model.FitModel
    """
    if narrow:
        qa.add_line_components(model, line_dict, "Halpha", "narrow")
        qa.add_line_components(model, line_dict, "NII", "narrow")
    qa.add_line_components(model, line_dict, "Halpha", "broad", Halpha_broad)


def setup_MgII_lines(
    spec,
    line_dict,
    narrow=False,
    MgII_broad=1,
    MgII_broadshifted=0,
    model=None,
    *,
    MgII_broad_shifted=None,
):
    """
    Setup emission line model for MgII emission line (2798.75 Å) using
    broad, velocity-shifted broad and (optionally) narrow gaussian
    components ("MgII_narrow", "MgII_broad_a", ...,
    "MgII_broadshifted_a", ...). MgII is generally not resolved into
    separate lines; it is recommended to fit the narrow component only for
    a high signal-to-noise spectrum.

    :param spec: input spectrum (unused)
    :type spec: speconed.speconed.SpecOneD
    :param line_dict: emission line dictionary; uses
        line_dict["MgII"]["narrow" / "broad" / "broadshifted"]
    :type line_dict: dict
    :param narrow: fit narrow component for MgII line
    :type narrow: bool
    :param MgII_broad: number of broad components to fit
    :type MgII_broad: int
    :param MgII_broadshifted: number of velocity shifted broad components to fit
    :type MgII_broadshifted: int
    :param model: model to add the components and parameters to
    :type model: sculptor.model.FitModel
    :param MgII_broad_shifted: deprecated name of `MgII_broadshifted`
    :type MgII_broad_shifted: int
    """
    if MgII_broad_shifted is not None:
        MgII_broadshifted = _renamed_kwarg(
            "MgII_broad_shifted", "MgII_broadshifted", MgII_broad_shifted
        )
    if narrow:
        qa.add_line_components(model, line_dict, "MgII", "narrow")
    qa.add_line_components(model, line_dict, "MgII", "broad", MgII_broad)
    qa.add_line_components(model, line_dict, "MgII", "broadshifted", MgII_broadshifted)


def setup_CIV_lines(
    spec, line_dict, narrow=False, CIV_broad=1, CIV_broadshifted=1, model=None
):
    """
    Setup emission line model for CIV emission line (1549.06 Å) using
    broad, velocity-shifted broad and (optionally) narrow gaussian
    components ("CIV_narrow", "CIV_broad_a", ..., "CIV_broadshifted_a",
    ...). CIV, a high-ionization line, is usually well described by up to
    three broad gaussians (Shen et al. 2011), accounting for outflows and
    winds that make the profile asymmetric.

    :param spec: input spectrum (unused)
    :type spec: speconed.speconed.SpecOneD
    :param line_dict: emission line dictionary; uses
        line_dict["CIV"]["narrow" / "broad" / "broadshifted"]
    :type line_dict: dict
    :param narrow: fit narrow component for CIV line
    :type narrow: bool
    :param CIV_broad: number of broad components to fit
    :type CIV_broad: int
    :param CIV_broadshifted: number of velocity shifted broad components to fit
    :type CIV_broadshifted: int
    :param model: model to add the components and parameters to
    :type model: sculptor.model.FitModel
    """
    if narrow:
        qa.add_line_components(model, line_dict, "CIV", "narrow")
    qa.add_line_components(model, line_dict, "CIV", "broad", CIV_broad)
    qa.add_line_components(model, line_dict, "CIV", "broadshifted", CIV_broadshifted)


def setup_Hbeta_OIII_lines(
    spec,
    line_dict,
    narrow=True,
    Hbeta_broad=1,
    Hbeta_broadshifted=0,
    OIII_broad=0,
    OIII_wind=0,
    model=None,
    *,
    Hbeta_broad_shifted=None,
):
    """
    Setup Hbeta+[OIII] lines simultaneously at 4862.30 A, 4960.30 A and
    5008.24 A: by default one narrow component for Hbeta and the [OIII]
    doublet ("Hbeta_narrow", "OIII_narrow") and one broad Hbeta component
    ("Hbeta_broad_a"); optionally velocity-shifted broad Hbeta, broad
    [OIII] and blue-shifted [OIII] wind components. Narrow components of
    Hbeta and [OIII] can be tied (same redshift and FWHM) in `line_dict`,
    as they originate from the same gas in the NLR.

    :param spec: input spectrum (unused)
    :type spec: speconed.speconed.SpecOneD
    :param line_dict: emission line dictionary; uses
        line_dict["Hbeta"]["narrow" / "broad" / "broadshifted"] and
        line_dict["OIII"]["narrow" / "broad" / "wind"]
    :type line_dict: dict
    :param narrow: fit narrow component for Hbeta and [OIII] lines
    :type narrow: bool
    :param Hbeta_broad: number of Hbeta broad components to fit
    :type Hbeta_broad: int
    :param Hbeta_broadshifted: number of velocity shifted Hbeta broad components to fit
    :type Hbeta_broadshifted: int
    :param OIII_broad: number of [OIII] broad components to fit
    :type OIII_broad: int
    :param OIII_wind: number of [OIII] blue-shifted wind components to fit
    :type OIII_wind: int
    :param model: model to add the components and parameters to
    :type model: sculptor.model.FitModel
    :param Hbeta_broad_shifted: deprecated name of `Hbeta_broadshifted`
    :type Hbeta_broad_shifted: int
    """
    if Hbeta_broad_shifted is not None:
        Hbeta_broadshifted = _renamed_kwarg(
            "Hbeta_broad_shifted", "Hbeta_broadshifted", Hbeta_broad_shifted
        )
    if narrow:
        qa.add_line_components(model, line_dict, "Hbeta", "narrow")
        qa.add_line_components(model, line_dict, "OIII", "narrow")
    qa.add_line_components(model, line_dict, "Hbeta", "broad", Hbeta_broad)
    qa.add_line_components(
        model, line_dict, "Hbeta", "broadshifted", Hbeta_broadshifted
    )
    qa.add_line_components(model, line_dict, "OIII", "broad", OIII_broad)
    qa.add_line_components(model, line_dict, "OIII", "wind", OIII_wind)


def setup_Hgamma_lines(spec, line_dict, narrow=False, Hgamma_broad=1, model=None):
    """
    Setup emission line model for the Hgamma emission line (4341.7 Å) using
    broad and (optionally) narrow gaussian components ("Hgamma_narrow",
    "Hgamma_broad_a", ...).

    Ported from the upstream `qso_auto_fit` branch (`Qso_autofit_2.py`).

    :param spec: input spectrum (unused)
    :type spec: speconed.speconed.SpecOneD
    :param line_dict: emission line dictionary; uses
        line_dict["Hgamma"]["narrow" / "broad"]
    :type line_dict: dict
    :param narrow: add a narrow component
    :type narrow: bool
    :param Hgamma_broad: number of broad components
    :type Hgamma_broad: int
    :param model: model to add the components and parameters to
    :type model: sculptor.model.FitModel
    """
    if narrow:
        qa.add_line_components(model, line_dict, "Hgamma", "narrow")
    qa.add_line_components(model, line_dict, "Hgamma", "broad", Hgamma_broad)


def setup_OI_lines(spec, line_dict, OI_broad=1, model=None):
    """
    Setup emission line model for the OI emission line (8446 Å) using broad
    gaussian components ("OI_broad_a", ...).

    Ported from the upstream `qso_auto_fit` branch (`Qso_autofit_2.py`).

    :param spec: input spectrum (unused)
    :type spec: speconed.speconed.SpecOneD
    :param line_dict: emission line dictionary; uses line_dict["OI"]["broad"]
    :type line_dict: dict
    :param OI_broad: number of broad components
    :type OI_broad: int
    :param model: model to add the components and parameters to
    :type model: sculptor.model.FitModel
    """
    qa.add_line_components(model, line_dict, "OI", "broad", OI_broad)


# ------------------------------------------------------------------------------
# Line properties (`Qso_autofit.get_line_properties`)
# ------------------------------------------------------------------------------


def get_line_prop_MgII(df, redshift, param_name_dict, save_dir, source, fit_name):
    """
    Calculate AGN properties from the MgII line measurements of a fit (see
    `Qso_autofit.get_line_properties` for the columns), with the
    continuum luminosity at 3000 Å. The BH-mass relation depends on the
    iron template used: Shen et al. 2011 ("_s11") for the Tsuzuki 2006 or
    Vestergaard & Wilkes 2001 template, and Vestergaard & Osmer 2009
    ("_vo09") for the latter. Additionally adds "FeII_over_MgII", the iron
    template amplitude over the MgII flux, and writes
    `<save_dir>/<source>_results_MgII.csv`.

    :param df: input dataframe containing extracted emission line parameters
    :type df: pd.DataFrame
    :param redshift: redshift of the source
    :type redshift: float
    :param param_name_dict: column name prefixes in `df`, as for
        `Qso_autofit.get_line_properties` ("fwhm": "MgII_fwhm",
        "luminosity": "L3000", "redshift": "MgII_redsh"), plus
        "fe_amp" ("FeII_amp") and "peak_flux" ("MgII_broad_flux") for the
        FeII/MgII ratio
    :type param_name_dict: dict
    :param save_dir: path to save the output
    :type save_dir: str
    :param source: name of the source
    :type source: str
    :param fit_name: iron template used in the fit: "Vest" (Vestergaard &
        Wilkes 2001; S11 and VO09 masses) or containing "Tsuzuki" (S11 mass)
    :type fit_name: str
    :return df: output dataframe
    :rtype: pd.DataFrame
    """
    relations = {}
    if "Tsuzuki" in fit_name or fit_name == "Vest":
        relations["_s11"] = qso.se_bhmass_mgii_s11_fwhm
    if fit_name == "Vest":
        relations["_vo09"] = qso.se_bhmass_mgii_vo09_fwhm
    if not relations:
        logger.info(f"Unknown fit_name '{fit_name}' — skipping BH mass calc.")

    df = qa.get_line_properties(
        df, redshift, param_name_dict, "MgII", 3000 * u.AA, relations
    )

    # FeII/MgII flux ratio
    fe_amp_col = param_name_dict.get("fe_amp")
    peak_flux_col = param_name_dict.get("peak_flux")
    if fe_amp_col and peak_flux_col:
        for key in ["_median", "_low", "_upp"]:
            fe_key = fe_amp_col + key
            flux_key = peak_flux_col + key
            if fe_key in df.columns and flux_key in df.columns:
                df["FeII_over_MgII" + key] = df[fe_key] * 1e-17 / df[flux_key]
            else:
                logger.info(
                    f"Missing columns '{fe_key}' or '{flux_key}' — skipping FeII/MgII."
                )
    else:
        logger.info(
            "'fe_amp' or 'peak_flux' not in param_name_dict — skipping FeII/MgII."
        )

    df_path = os.path.join(save_dir, f"{source}_results_MgII.csv")
    df.to_csv(df_path, index=False, float_format="%.4e")
    return df


def get_line_prop_CIV(df, redshift, param_name_dict, save_dir=None, source=None):
    """
    Calculate AGN properties from the CIV line measurements of a fit (see
    `Qso_autofit.get_line_properties` for the columns), with the
    continuum luminosity at 1350 Å and the Vestergaard & Peterson 2006
    BH-mass relation ("_vp06").

    Ported from the upstream `qso_auto_fit` branch (`Qso_autofit_2.py`),
    where it passed unitless numbers to the BH-mass relation and failed
    (`UnitConversionError`); upstream's columns were "civ_BH_mass_1350",
    "L_bol_1350", "Eddington_ratio_1350", "Edd_lum_civ_1350" and
    "delta_v_civ". Its unused `model` / `flat_samples` arguments are gone.

    :param df: input dataframe containing extracted emission line parameters
    :type df: pd.DataFrame
    :param redshift: reference (systemic) redshift of the source
    :type redshift: float
    :param param_name_dict: column name prefixes in `df`, as for
        `Qso_autofit.get_line_properties` (e.g. "CIV_fwhm", "L1350",
        "CIV_redsh")
    :type param_name_dict: dict
    :param save_dir: if given, also write `<save_dir>/<source>_results_CIV.csv`
    :type save_dir: str
    :param source: name of the source, used in the file name
    :type source: str
    :return: `df` with the added columns
    :rtype: pd.DataFrame
    """
    df = qa.get_line_properties(
        df,
        redshift,
        param_name_dict,
        "CIV",
        1350 * u.AA,
        {"_vp06": qso.se_bhmass_civ_vp06_fwhm},
    )

    if save_dir is not None:
        df_path = os.path.join(save_dir, f"{source}_results_CIV.csv")
        df.to_csv(df_path, index=False, float_format="%.4e")
    return df


def get_line_prop_Hbeta_OIII(
    df,
    redshift,
    param_name_dict,
    save_dir=None,
    source=None,
    model=None,
    flat_samples=None,
):
    """
    Calculate AGN properties from the Hbeta line measurements of a fit (see
    `Qso_autofit.get_line_properties` for the columns), with the
    continuum luminosity at 5100 Å and the Vestergaard & Peterson 2006
    BH-mass relation (no column tag, e.g. "Hbeta_BH_mass_5100").

    :param df: input dataframe containing extracted emission line parameters
    :type df: pd.DataFrame
    :param redshift: redshift of the source
    :type redshift: float
    :param param_name_dict: column name prefixes in `df`, as for
        `Qso_autofit.get_line_properties` (e.g. "Hbeta_fwhm", "L5100",
        "Hbeta_redsh")
    :type param_name_dict: dict
    :param save_dir: unused; kept so existing calls keep working
    :type save_dir: str
    :param source: unused; kept so existing calls keep working
    :type source: str
    :param model: unused; kept so existing calls keep working
    :type model: sculptor.model.FitModel
    :param flat_samples: unused; kept so existing calls keep working
    :type flat_samples: np.ndarray
    :return: `df` with the added columns
    :rtype: pd.DataFrame
    """
    return qa.get_line_properties(
        df,
        redshift,
        param_name_dict,
        "Hbeta",
        5100 * u.AA,
        {"": qso.se_bhmass_hbeta_vp06},
    )


# ------------------------------------------------------------------------------
# Old names
# ------------------------------------------------------------------------------

RENAMED = {
    "setup_iron_template_V01_FUV": "setup_iron_template_VW01_FUV",
    "setup_iron_template_V01_CIV": "setup_iron_template_VW01_CIV",
    "setup_hb_o3_lines": "setup_Hbeta_OIII_lines",
    "get_civ_line_prop": "get_line_prop_CIV",
}


def __getattr__(name):
    """Forward renamed functions (`RENAMED`), with a `DeprecationWarning`.

    :param name: attribute name
    :type name: str
    :return: the renamed function
    :rtype: callable
    :raises AttributeError: for any other name
    """
    if name in RENAMED:
        warnings.warn(
            f"`qso_presets.{name}` was renamed to `{RENAMED[name]}`",
            DeprecationWarning,
            stacklevel=2,
        )
        return globals()[RENAMED[name]]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
