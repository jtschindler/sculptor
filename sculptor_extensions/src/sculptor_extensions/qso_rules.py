"""Rules for automatic model building of type-I quasar spectra
(`sculptor.autofit`), built on the `qso_presets` / `qso_autofit` setups.

`qso_rules(redshift)` returns the default set (the one validated on the
development spectra): broad Gaussians at the strong lines in the fit
regions (`DEFAULT_LINES`, including the broad "HeII" red shelf), the CIII]
complex with SiIII] and AlIII (`ciii_complex_rule`) and its optional
second broad CIII] (`ciii_second_component_rule`), the UV and optical iron
templates (`IRON_TEMPLATES_TS06`), the Balmer continuum and, separately,
the high-order Balmer lines (`balmer_hol_rule`), and the narrow
components of Hbeta, Halpha and [OIII] proposed together at the systemic
redshift (`narrow_line_group`, the only route for a narrow Hbeta /
Halpha), then narrow Hgamma and Hdelta on the group's kinematics
(`narrow_line_companions`). Components narrower than `NARROW_FWHM_MAX`
count as narrow;
change it before building the rules::

    qso_rules.NARROW_FWHM_MAX = 1000.0

Start `AutoFit` from a power-law continuum::

    from sculptor.autofit import AutoFit
    from sculptor_extensions import qso_rules

    rules = qso_rules.qso_rules(redshift=7.16)
    result = AutoFit(base_model, rules, criterion="bic").run()

Each rule can also be built on its own (`line_rules`, `iron_template_rule`,
`balmer_continuum_rule`, ...).
"""

from sculptor import analysis as ana
from sculptor import utils as scut
from sculptor.autofit import (
    CompanionLineRule,
    ComponentRule,
    GaussianLineRule,
    LineGroupRule,
)
from sculptor.component import FitComponent
from sculptor.prior import UniformPrior
from sculptor_extensions import qso, qso_presets
from sculptor_extensions import qso_advanced_models as qam
from sculptor_extensions import qso_autofit as qa

# Rest-frame vacuum wavelengths (Angstrom) of the strong quasar lines
QSO_LINES = {
    "Lya": 1215.67,
    "NV": 1240.14,
    "OI": 1305.42,  # OI 1304 + SiII 1306 blend (Vanden Berk et al. 2001)
    "SiIV": 1399.80,
    "CIV": 1549.06,
    "HeII": 1640.42,
    "OIII1663": 1663.48,  # OIII] 1661 + 1666
    "AlIII": 1857.40,
    "SiIII": 1892.03,
    "CIII": 1908.73,
    "MgII": 2798.75,
    "Hdelta": 4102.89,
    "Hgamma": 4341.68,
    "Hbeta": 4862.68,
    "OIII_4960": 4960.30,
    "OIII_5008": 5008.24,
    "Halpha": 6564.61,
}

# Doublets fitted as one component: name -> (function, line used for the
# detection, fixed arguments, largest velocity offset in km/s).
# [OIII] 5008/4960 = 2.98 (Storey & Zeippen 2000); `flux` is the 4960
# line's. The offset stays below the lines' separation (2870 km/s): with
# more, the detection window around 5008 A reaches the 4960 line, and a
# doublet shifted by -2870 km/s fits it.
DOUBLETS = {
    "OIII": (
        qso.line_model_gaussian_oiii_doublet,
        5008.24,
        {"fluxratio": 2.98},
        2000.0,
    ),
}

# Lines fitted by default: Lya and NV need absorption modeling and are left
# out. In JWST spectra of z ~ 7 quasars a single broad (~10000 km/s)
# "HeII" Gaussian takes up the excess between CIV and CIII] (the red shelf:
# HeII, OIII] 1663, iron) rather than HeII alone. "CIII" is fitted as the
# CIII] complex by `qso_rules` (AlIII and SiIII] as separate Gaussians trade
# places with CIII] in broad blends).
DEFAULT_LINES = (
    "OI",
    "SiIV",
    "CIV",
    "HeII",
    "CIII",
    "MgII",
    "Hdelta",
    "Hgamma",
    "Hbeta",
    "OIII",
    "Halpha",
)

# Iron templates of `qso_rules`: component name -> `iron_template_rule`
# preset, or (preset, rest-frame range in A).
# VW01 (Vestergaard & Wilkes 2001) below 2200 A, TS06 (Tsuzuki et al.
# 2006) at 2200-3500 A (iron under MgII, and up to 3500 A where VW01_UV
# stops at 3089 A), BG92 from 3685 A. Round 27: TS06's 3190-3320 A bump
# is over-predicted by ~1 sigma in the dev objects; needs a human look.
IRON_TEMPLATES_TS06 = {
    "FeII_UV": "VW01_FUV",
    "FeII_TS": "Tsuz_UV",
    "FeII_opt": "BG92_OPT",
}
# Alternative: VW01 up to 3089 A (no iron at 3089-3685 A)
IRON_TEMPLATES_VW01 = {"FeII_UV": "VW01_UV", "FeII_opt": "BG92_OPT"}
# Forbidden lines get the narrow FWHM range
NARROW_LINES = ("OIII", "OIII_4960", "OIII_5008")

# Permitted lines with a narrow component: their own rule is broad only
# (FWHM above `NARROW_FWHM_MAX`); the narrow component comes from
# `narrow_line_group` or `narrow_line_companions` alone (one hypothesis,
# one route)
LINES_WITH_NARROW = ("Hbeta", "Halpha", "Hgamma", "Hdelta")

# Lines whose narrow components `narrow_line_companions` proposes one at a
# time on the narrow group's redshift and FWHM, once the group is in the
# model: too weak to earn a threshold with kinematics of their own
NARROW_COMPANION_LINES = ("Hgamma", "Hdelta")

# Components with a FWHM below this (km/s) count as narrow (the narrow
# group's largest width); change it before building the rules
NARROW_FWHM_MAX = 1200.0

# Lines whose narrow components `narrow_line_group` proposes together by
# default
NARROW_GROUP_LINES = ("Hbeta", "Halpha", "OIII")

# Allowed velocity offset of the narrow-line group from the systemic
# redshift, km/s: the group models cores at the systemic redshift, not
# shifted outflows (a group whose redshift ends at the bound is removed)
NARROW_VELOCITY_RANGE = 300.0

# Rest-frame wavelengths of the CIII] complex (`qso.CIII_complex_model_func`)
CIII_COMPLEX = {
    "cen": QSO_LINES["CIII"],
    "cen_alIII": QSO_LINES["AlIII"],
    "cen_siIII": QSO_LINES["SiIII"],
}


def _fixed(value):
    return {"value": value, "vary": False}


def line_rules(
    redshift,
    lines=DEFAULT_LINES,
    broad_fwhm=(1000.0, 15000.0),
    narrow_fwhm=(100.0, 2500.0),
    velocity_range=5000.0,
    max_components=3,
    **kwargs,
):
    """One `GaussianLineRule` per line, at the line's rest wavelength
    (fixed center, redshift free within +-`velocity_range`), with up to
    `max_components` Gaussians each. Doublets in `DOUBLETS` (e.g. "OIII")
    are one component per pair, with one redshift and width and a fixed
    flux ratio, and at most their own velocity offset (below the lines'
    separation). Lines in `LINES_WITH_NARROW` are broad only (above
    `NARROW_FWHM_MAX`): their narrow components come from
    `narrow_line_group`.

    :param redshift: Systemic redshift
    :type redshift: float
    :param lines: Names in `QSO_LINES`, or name -> rest wavelength
    :type lines: list(str) or dict
    :param broad_fwhm: FWHM range of the broad permitted lines, km/s
    :type broad_fwhm: (float, float)
    :param narrow_fwhm: FWHM range of the forbidden lines ([OIII]), km/s
    :type narrow_fwhm: (float, float)
    :param velocity_range: Allowed velocity offset, km/s (CIV is often
        blueshifted by a few 1000 km/s); doublets take the smaller of this
        and their own (`DOUBLETS`)
    :type velocity_range: float
    :param max_components: Maximum Gaussians per line
    :type max_components: int
    :param kwargs: Passed on to `GaussianLineRule`
    :return: The rules
    :rtype: list(sculptor.autofit.GaussianLineRule)
    """
    if not isinstance(lines, dict):
        lines = {
            name: (DOUBLETS[name][1] if name in DOUBLETS else QSO_LINES[name])
            for name in lines
        }
    rules = []
    for name, rest in lines.items():
        function, fixed, max_velocity = None, None, velocity_range
        if name in DOUBLETS:
            function, _, fixed, doublet_velocity = DOUBLETS[name]
            max_velocity = min(velocity_range, doublet_velocity)
        if name in NARROW_LINES:
            fwhm_range = narrow_fwhm
        elif name in LINES_WITH_NARROW:
            fwhm_range = (max(broad_fwhm[0], NARROW_FWHM_MAX), broad_fwhm[1])
        else:
            fwhm_range = broad_fwhm
        rules.append(
            GaussianLineRule(
                name,
                redshift=redshift,
                lines=[rest],
                fwhm_range=fwhm_range,
                velocity_range=max_velocity,
                max_components=max_components,
                function=function,
                fixed=fixed,
                **kwargs,
            )
        )
    return rules


def narrow_line_group(
    rules, redshift, lines=NARROW_GROUP_LINES, max_fwhm=None, min_fwhm=100.0, **kwargs
):
    """Rule that proposes the narrow components of `lines` together, at the
    systemic redshift, sharing redshift and FWHM (`narrow_redsh`,
    `narrow_fwhm_km_s`): weak narrow lines that would not earn the threshold
    one at a time may earn it
    together. See `sculptor.autofit.LineGroupRule`. By default it waits
    for the members with broad components (e.g. Hbeta), not for those that
    are narrow only (`NARROW_LINES`, [OIII]): a weak [OIII] may never be
    added alone, and the narrow Hbeta would then never be tested.

    :param rules: The line rules (from `line_rules`)
    :type rules: list
    :param redshift: Systemic redshift (the starting value)
    :type redshift: float
    :param lines: Line names (rule prefixes); those without a rule are
        skipped
    :type lines: list(str)
    :param max_fwhm: Largest FWHM, km/s (default `NARROW_FWHM_MAX`, read
        when called)
    :type max_fwhm: float
    :param min_fwhm: Smallest FWHM, km/s
    :type min_fwhm: float
    :param kwargs: Passed on to `LineGroupRule` (default `velocity_range`:
        `NARROW_VELOCITY_RANGE`; `after_members`: the members not in
        `NARROW_LINES`)
    :return: The rule, or None with fewer than two of the lines
    :rtype: sculptor.autofit.LineGroupRule
    """
    by_prefix = {
        r.prefix: r
        for r in rules
        if isinstance(r, GaussianLineRule) and r.lines is not None
    }
    members = [by_prefix[name] for name in lines if name in by_prefix]
    if len(members) < 2:
        return None
    kwargs.setdefault("velocity_range", NARROW_VELOCITY_RANGE)
    kwargs.setdefault(
        "after_members", [m.prefix for m in members if m.prefix not in NARROW_LINES]
    )
    return LineGroupRule(
        "narrow",
        members,
        redshift,
        fwhm_range=(min_fwhm, NARROW_FWHM_MAX if max_fwhm is None else max_fwhm),
        **kwargs,
    )


def narrow_line_companions(rules, group, lines=NARROW_COMPANION_LINES):
    """Rules that propose the narrow components of `lines` on the narrow
    group's redshift and FWHM, one line at a time, once the group is in the
    model (`sculptor.autofit.CompanionLineRule`): a narrow Hbeta implies
    narrow Hgamma and Hdelta, too weak to be found with kinematics of
    their own. Each has only its flux free, its own threshold, and is
    removed with the group.

    :param rules: The line rules (from `line_rules`)
    :type rules: list
    :param group: The narrow group (from `narrow_line_group`)
    :type group: sculptor.autofit.LineGroupRule
    :param lines: Line names (rule prefixes); those without a rule are
        skipped
    :type lines: list(str)
    :return: The rules (named "narrow Hgamma", ...)
    :rtype: list(sculptor.autofit.CompanionLineRule)
    """
    by_prefix = {
        r.prefix: r
        for r in rules
        if isinstance(r, GaussianLineRule) and r.lines is not None
    }
    return [
        CompanionLineRule(group, by_prefix[name]) for name in lines if name in by_prefix
    ]


def narrow_line_windows(
    redshift, lines=("Hbeta", "OIII_4960", "OIII_5008", "Halpha"), velocity=1000.0
):
    """Observed-frame ranges of +-`velocity` around the narrow lines at the
    systemic redshift, to keep out of sigma clipping
    (`FitModel.sigma_clip(protect=...)`, e.g. in `AutoFit(clip=...)`): an
    unresolved narrow core is as narrow as a cosmic ray, and the clipping
    against an unfinished model would remove it.

    :param redshift: Systemic redshift
    :type redshift: float
    :param lines: Names in `QSO_LINES`
    :type lines: list(str)
    :param velocity: Half-width, km/s
    :type velocity: float
    :return: (lo, hi) ranges, Angstrom
    :rtype: list(tuple(float, float))
    """
    factor = velocity / scut.c_km_s
    return [
        (
            QSO_LINES[name] * (1 + redshift) * (1 - factor),
            QSO_LINES[name] * (1 + redshift) * (1 + factor),
        )
        for name in lines
    ]


def ciii_complex_rule(
    redshift,
    prefix="CIIIc",
    fwhm=3000.0,
    fwhm_range=(1000.0, 15000.0),
    velocity_range=5000.0,
    start_fwhm=(1500.0, 6000.0),
    shared_width=True,
):
    """Rule that adds the CIII] complex (`qso.CIII_complex_model_func`):
    CIII], SiIII] and AlIII at one redshift, each with its own flux, with
    one FWHM (`shared_width`) or one each, velocity shifts fixed at 0;
    fluxes estimated jointly from the residuals. A shared width keeps
    SiIII] and AlIII from taking over a broad, blended CIII].

    :param redshift: Systemic redshift
    :type redshift: float
    :param prefix: Component name
    :type prefix: str
    :param fwhm: Starting FWHM of the three lines, km/s
    :type fwhm: float
    :param fwhm_range: FWHM prior range, km/s
    :type fwhm_range: (float, float)
    :param velocity_range: Allowed velocity offset, km/s
    :type velocity_range: float
    :param start_fwhm: Further starting FWHMs (all three lines), km/s
    :type start_fwhm: list(float)
    :param shared_width: One FWHM (`<prefix>_fwhm_km_s`) for the three
        lines
    :type shared_width: bool
    :return: The rule
    :rtype: sculptor.autofit.ComponentRule
    """
    widths = ("fwhm_km_s", "fwhm_km_s_alIII", "fwhm_km_s_siIII")
    if shared_width:
        widths = widths[:1]
    dz = velocity_range / scut.c_km_s * (1 + redshift)

    def factory(model):
        """The complex at the systemic redshift."""
        comp = FitComponent(prefix, qso.CIII_complex_model_func)
        pars = comp.create_params()
        pars[f"{prefix}_redsh"].value = redshift
        pars[f"{prefix}_redsh"].prior = UniformPrior(
            f"{prefix}_redsh", redshift - dz, redshift + dz
        )
        for arg, value in CIII_COMPLEX.items():
            pars[f"{prefix}_{arg}"].value = value
            pars[f"{prefix}_{arg}"].vary = False
        for arg in ("shift_km_s", "shift_km_s_alIII", "shift_km_s_siIII"):
            pars[f"{prefix}_{arg}"].value = 0.0
            pars[f"{prefix}_{arg}"].vary = False
        for arg in widths:
            pars[f"{prefix}_{arg}"].value = fwhm
            pars[f"{prefix}_{arg}"].prior = UniformPrior(f"{prefix}_{arg}", *fwhm_range)
        if shared_width:
            for arg in ("fwhm_km_s_alIII", "fwhm_km_s_siIII"):
                del pars[f"{prefix}_{arg}"]
                comp.param_mapping[arg] = f"{prefix}_fwhm_km_s"
        return comp, pars

    return ComponentRule(
        "CIII] complex",
        factory,
        linear_param=["flux", "flux_siIII", "flux_alIII"],
        starts=[{f"{prefix}_{arg}": w for arg in widths} for w in start_fwhm],
    )


def ciii_second_component_rule(
    complex_prefix="CIIIc",
    prefix="CIIIc_b",
    fwhm_range=(1000.0, 15000.0),
    start_fwhm=(1200.0, 3000.0, 8000.0),
):
    """Rule that adds a second broad CIII] Gaussian to the CIII] complex
    (`ciii_complex_rule`): a broad-line profile sharper or wider than one
    Gaussian (core and wings), not a narrow-line-region line. It shares the
    complex's redshift and has its own flux and FWHM, a constrained
    hypothesis that the calibration gives its own threshold. Proposed only
    while the complex is in the model; the candidate is rejected if its
    width or the complex's ends at a bound of `fwhm_range` (at the lower one
    it is not broad; the two Gaussians may swap roles).

    :param complex_prefix: Name of the complex component
    :type complex_prefix: str
    :param prefix: Name of the new component
    :type prefix: str
    :param fwhm_range: FWHM prior range, km/s
    :type fwhm_range: (float, float)
    :param start_fwhm: Starting FWHMs, km/s (the first is the default
        start)
    :type start_fwhm: list(float)
    :return: The rule
    :rtype: sculptor.autofit.ComponentRule
    """
    redsh = f"{complex_prefix}_redsh"
    width = f"{prefix}_fwhm_km_s"

    def factory(model):
        """A CIII] Gaussian at the complex's redshift, if it is there."""
        if redsh not in model.parameters or not any(
            c.name == complex_prefix for c in model.components
        ):
            return None
        comp = FitComponent(prefix, ana.line_model_gaussian)
        pars = comp.create_params()
        del pars[f"{prefix}_redsh"]
        comp.param_mapping["redsh"] = redsh
        pars[f"{prefix}_cen"].value = CIII_COMPLEX["cen"]
        pars[f"{prefix}_cen"].vary = False
        pars[width].value = start_fwhm[0]
        pars[width].prior = UniformPrior(width, *fwhm_range)
        return comp, pars

    return ComponentRule(
        "CIII] second component",
        factory,
        linear_param="flux",
        starts=[{width: w} for w in start_fwhm[1:]],
        reject_at_bound=[width, f"{complex_prefix}_fwhm_km_s"],
    )


def iron_template_rule(
    redshift,
    preset="VW01_UV",
    prefix="FeII",
    fwhm=2500.0,
    fwhm_range=(1000.0, 10000.0),
    intr_fwhm=900.0,
    vary_fwhm=True,
    disp_range=None,
):
    """Rule that adds an iron template from `qso_presets`
    (`setup_iron_template_<preset>`), at the systemic redshift, with its
    amplitude estimated from the residuals.

    :param redshift: Systemic redshift (fixed)
    :type redshift: float
    :param preset: Template name after `setup_iron_template_`, e.g.
        "VW01_UV" (Vestergaard & Wilkes 2001), "Tsuz_UV", "BG92_OPT"
    :type preset: str
    :param prefix: Component name
    :type prefix: str
    :param fwhm: Starting FWHM of the broadening, km/s
    :type fwhm: float
    :param fwhm_range: FWHM prior range, km/s
    :type fwhm_range: (float, float)
    :param intr_fwhm: Intrinsic FWHM of the template, km/s (900 for the
        I Zw 1 templates)
    :type intr_fwhm: float
    :param vary_fwhm: Fit the FWHM (slower) or keep it at `fwhm`
    :type vary_fwhm: bool
    :param disp_range: Rest-frame range (A) to restrict the template to;
        None for the preset's default
    :type disp_range: (float, float)
    :return: The rule
    :rtype: sculptor.autofit.ComponentRule
    :raises ValueError: for an unknown preset
    """
    setup = getattr(qso_presets, f"setup_iron_template_{preset}", None)
    if setup is None:
        raise ValueError(f"No iron template preset {preset!r} in qso_presets.")

    def factory(model):
        """The template component for `model`."""
        fe_dict = {
            "redsh": _fixed(redshift),
            "amp": {
                "value": 1.0,
                "vary": True,
                "prior": UniformPrior(f"{prefix}_amp", 0, 1e30),
            },
            "fwhm": (
                {
                    "value": fwhm,
                    "vary": True,
                    "prior": UniformPrior(f"{prefix}_fwhm", *fwhm_range),
                }
                if vary_fwhm
                else _fixed(fwhm)
            ),
            "intr_fwhm": _fixed(intr_fwhm),
        }
        if disp_range is None:
            return setup(prefix, fe_dict=fe_dict, fast=True)
        return setup(prefix, disp_range=list(disp_range), fe_dict=fe_dict, fast=True)

    name = preset
    if disp_range is not None:
        name += f" {disp_range[0]:.0f}-{disp_range[1]:.0f}"
    return ComponentRule(f"iron ({name})", factory, linear_param="amp")


def balmer_continuum_rule(
    redshift,
    prefix="BalmerCont",
    Te=15000.0,
    tau_be=1.0,
    lambda_be=3646.0,
    hol=False,
    fwhm=3000.0,
    fwhm_range=(1000.0, 10000.0),
    hol_ratio=0.5,
    hol_ratio_range=(0.0, 2.0),
):
    """Rule that adds a Balmer continuum (Dietrich et al. 2003), with its
    amplitude estimated from the residuals. Proposed wherever the component
    has flux on the fitted pixels, as every continuum-like component (the
    `ComponentRule` amplitude estimate): below the edge, or with `hol` also
    above it, where the high-order lines are (a fit range starting above
    the edge then gets the lines; the continuum part below the edge is
    unconstrained there and trades with `hol_ratio`).

    Without `hol` the continuum drops to zero at the edge
    (`qso_advanced_models.fast_balmer_continuum_model`). With `hol` the
    high-order Balmer lines (n = 7 - 50, Storey & Hummer 1995) are added,
    broadened by a free FWHM, with a free strength relative to the
    continuum (`qso_advanced_models.BalmerContinuumHOLRatio`: `hol_ratio`
    is the lines' flux density at the edge over the continuum's just below
    it): they fill the region just above the edge, which the sharp edge
    leaves to the other components. The candidate also starts from the
    ends of `hol_ratio_range` (inside it).

    :param redshift: Systemic redshift (fixed)
    :type redshift: float
    :param prefix: Component name
    :type prefix: str
    :param Te: Electron temperature, K (fixed)
    :type Te: float
    :param tau_be: Optical depth at the Balmer edge (fixed)
    :type tau_be: float
    :param lambda_be: Balmer edge, Angstrom
    :type lambda_be: float
    :param hol: Include the high-order Balmer lines
    :type hol: bool
    :param fwhm: With `hol`: starting FWHM of the high-order lines, km/s
    :type fwhm: float
    :param fwhm_range: With `hol`: their FWHM prior range, km/s
    :type fwhm_range: (float, float)
    :param hol_ratio: With `hol`: starting ratio of the lines to the
        continuum at the edge
    :type hol_ratio: float
    :param hol_ratio_range: With `hol`: its prior range (provisional: the
        dev objects want about 0.3 - 1)
    :type hol_ratio_range: (float, float)
    :return: The rule
    :rtype: sculptor.autofit.ComponentRule
    """

    def factory(model):
        """The Balmer continuum component."""
        balmer_dict = {
            "redsh": _fixed(redshift),
            "amp_be": {
                "value": 1.0,
                "vary": True,
                "prior": UniformPrior(f"{prefix}_amp_be", 0, 1e30),
            },
            "Te": _fixed(Te),
            "tau_be": _fixed(tau_be),
            "lambda_be": _fixed(lambda_be),
        }
        if not hol:
            return qa.initialize_balmer_without_hol(prefix, balmer_dict)
        balmer_dict["fwhm"] = {
            "value": fwhm,
            "vary": True,
            "prior": UniformPrior(f"{prefix}_fwhm", *fwhm_range),
        }
        balmer_dict["hol_ratio"] = {
            "value": hol_ratio,
            "vary": True,
            "prior": UniformPrior(f"{prefix}_hol_ratio", *hol_ratio_range),
        }
        return qam.initialize_balmer_hol_ratio(prefix, balmer_dict=balmer_dict)

    if not hol:
        return ComponentRule("Balmer continuum", factory, linear_param="amp_be")
    low, upp = hol_ratio_range
    span = upp - low
    starts = [
        {f"{prefix}_hol_ratio": low + 0.1 * span},
        {f"{prefix}_hol_ratio": upp - 0.1 * span},
    ]
    return ComponentRule(
        "Balmer continuum + HOL", factory, linear_param="amp_be", starts=starts
    )


def balmer_hol_rule(
    redshift, prefix="BalmerHOL", fwhm=3000.0, fwhm_range=(1000.0, 10000.0)
):
    """Rule that adds the high-order Balmer lines alone
    (`qso_advanced_models.BalmerHOLLines`: n = 7 - 50, fixed relative
    strengths, one free FWHM), with their flux estimated from the
    residuals, independently of the Balmer continuum. Proposed wherever the
    lines have flux on the fitted pixels (about 3646 - 3970 A rest).

    :param redshift: Systemic redshift (fixed)
    :type redshift: float
    :param prefix: Component name
    :type prefix: str
    :param fwhm: Starting FWHM, km/s
    :type fwhm: float
    :param fwhm_range: FWHM prior range, km/s
    :type fwhm_range: (float, float)
    :return: The rule
    :rtype: sculptor.autofit.ComponentRule
    """

    def factory(model):
        """The high-order lines component."""
        hol_dict = {
            "redsh": _fixed(redshift),
            "flux": {
                "value": 1.0,
                "vary": True,
                "prior": UniformPrior(f"{prefix}_flux", 0, 1e30),
            },
            "fwhm": {
                "value": fwhm,
                "vary": True,
                "prior": UniformPrior(f"{prefix}_fwhm", *fwhm_range),
            },
        }
        return qam.initialize_balmer_hol_lines(prefix, hol_dict=hol_dict)

    return ComponentRule("Balmer high-order lines", factory, linear_param="flux")


BALMER_OPTIONS = ("separate", "hol", True, False)


def qso_rules(
    redshift,
    lines=DEFAULT_LINES,
    iron=IRON_TEMPLATES_TS06,
    balmer="separate",
    narrow_group=True,
    ciii_complex=True,
    ciii_second=True,
    **line_kwargs,
):
    """The default rules for a quasar spectrum (see the module docstring).

    :param redshift: Systemic redshift
    :type redshift: float
    :param lines: See `line_rules`; with `ciii_complex`, "CIII" stands for
        the complex instead of free Gaussians
    :type lines: list(str) or dict
    :param iron: Component name -> iron template preset, or (preset,
        rest-frame range) (see `iron_template_rule`); None or {} for none.
        Default VW01 + TS06 + BG92 (`IRON_TEMPLATES_TS06`); alternative
        `IRON_TEMPLATES_VW01`
    :type iron: dict
    :param balmer: "separate" (default): the Balmer continuum (sharp edge)
        and the high-order Balmer lines as two independent components
        (`balmer_continuum_rule`, `balmer_hol_rule`); "hol": one component
        with the lines' ratio to the continuum free (only constrained if the
        fit reaches below the edge; for a later, manual stage); True: the
        continuum only; False: neither
    :type balmer: bool or str
    :param narrow_group: Include `narrow_line_group` (`NARROW_GROUP_LINES`
        among `lines`) and its `narrow_line_companions`
        (`NARROW_COMPANION_LINES`)
    :type narrow_group: bool
    :param ciii_complex: Fit "CIII" as `ciii_complex_rule`
    :type ciii_complex: bool
    :param ciii_second: With `ciii_complex`, also include
        `ciii_second_component_rule`
    :type ciii_second: bool
    :param line_kwargs: Passed on to `line_rules`
    :return: The rules
    :rtype: list
    """
    complex_ = ciii_complex and "CIII" in lines
    gaussian = [name for name in lines if not (complex_ and name == "CIII")]
    if isinstance(lines, dict):
        gaussian = {name: lines[name] for name in gaussian}
    rules = line_rules(redshift, gaussian, **line_kwargs)
    if narrow_group:
        group = narrow_line_group(rules, redshift)
        if group is not None:
            rules += [group, *narrow_line_companions(rules, group)]
    if complex_:
        rules.append(ciii_complex_rule(redshift))
        if ciii_second:
            rules.append(ciii_second_component_rule())
    for prefix, preset in (iron or {}).items():
        preset, disp_range = (preset, None) if isinstance(preset, str) else preset
        rules.append(
            iron_template_rule(
                redshift, preset=preset, prefix=prefix, disp_range=disp_range
            )
        )
    if not any(balmer is option or balmer == option for option in BALMER_OPTIONS):
        raise ValueError(f"balmer must be one of {BALMER_OPTIONS}, not {balmer!r}")
    if balmer:
        rules.append(balmer_continuum_rule(redshift, hol=balmer == "hol"))
    if balmer == "separate":
        rules.append(balmer_hol_rule(redshift))
    return rules
