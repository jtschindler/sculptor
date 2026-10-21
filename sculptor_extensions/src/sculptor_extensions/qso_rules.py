"""Rules for automatic model building of type-I quasar spectra
(`sculptor.autofit`), built on the `qso_presets` / `qso_autofit` setups.

`qso_rules(redshift)` returns a default set: Gaussians at the strong
broad and narrow lines in the fit regions, the UV iron template and the
Balmer continuum, the narrow components of Hbeta, Halpha and [OIII]
proposed together at the systemic redshift (`narrow_line_group`), and the
hypothesis that they share redshift and width (`narrow_line_tie`), which
AutoFit keeps only if the data accept it. The CIII] complex with SiIII]
and AlIII (`ciii_complex_rule`, `qso_rules(ciii_complex=True)`) and the
weaker lines in `QSO_LINES` (e.g. "HeII") are optional. Components
narrower than
`NARROW_FWHM_MAX` count as narrow; change it before building the rules::

    qso_rules.NARROW_FWHM_MAX = 1000.0

Start `AutoFit` from a power-law continuum::

    from sculptor.autofit import AutoFit
    from sculptor_extensions import qso_rules

    rules = qso_rules.qso_rules(redshift=7.16)
    result = AutoFit(base_model, rules, criterion="bic").run()

Each rule can also be built on its own (`line_rules`, `iron_template_rule`,
`balmer_continuum_rule`, `line_dict_rule`).
"""

import copy
import string

from sculptor import utils as scut
from sculptor.autofit import ComponentRule, GaussianLineRule, LineGroupRule, Tie
from sculptor.component import FitComponent
from sculptor.prior import UniformPrior
from sculptor_extensions import qso, qso_presets
from sculptor_extensions import qso_autofit as qa

# Rest-frame vacuum wavelengths (Angstrom) of the strong quasar lines
QSO_LINES = {
    "Lya": 1215.67,
    "NV": 1240.14,
    "SiIV": 1399.80,
    "CIV": 1549.06,
    "HeII": 1640.42,
    "OIII1663": 1663.48,  # OIII] 1661 + 1666
    "AlIII": 1857.40,
    "SiIII": 1892.03,
    "CIII": 1908.73,
    "MgII": 2798.75,
    "Hgamma": 4341.68,
    "Hbeta": 4862.68,
    "OIII_4960": 4960.30,
    "OIII_5008": 5008.24,
    "Halpha": 6564.61,
}

# Doublets fitted as one component: name -> (function, line used for the
# detection, fixed arguments). [OIII] 5008/4960 = 2.98 (Storey & Zeippen
# 2000); `flux` is the 4960 line's.
DOUBLETS = {
    "OIII": (qso.line_model_gaussian_oiii_doublet, 5008.24, {"fluxratio": 2.98}),
}

# Lines fitted by default: Lya and NV need absorption modeling and are left
# out
DEFAULT_LINES = ("SiIV", "CIV", "CIII", "MgII", "Hbeta", "OIII", "Halpha")

# Weaker lines, not fitted by default. In JWST spectra of z ~ 7 quasars a
# single broad (~10000 km/s) "HeII" Gaussian takes up the excess between
# CIV and CIII] (the red shelf: HeII, OIII] 1663, iron) rather than HeII
# alone. AlIII and SiIII] are best fitted with CIII] as one complex
# (`ciii_complex_rule`): as separate Gaussians they trade places with
# CIII] in broad blends.
# Forbidden lines get the narrow FWHM range
NARROW_LINES = ("OIII", "OIII_4960", "OIII_5008")

# Permitted lines with a narrow component: their Gaussians may be narrow or
# broad (FWHM from the narrow range's minimum to the broad range's maximum)
LINES_WITH_NARROW = ("Hbeta", "Halpha")

# Components with a FWHM up to this (km/s) count as narrow, e.g. for
# `narrow_line_tie`; change it before building the rules
NARROW_FWHM_MAX = 1200.0

# Lines whose narrow components `narrow_line_tie` ties (and
# `narrow_line_group` proposes together) by default
NARROW_TIE_LINES = ("Hbeta", "Halpha", "OIII")

# Allowed velocity offset of the narrow-line group from the systemic
# redshift, km/s: the group models cores at the systemic redshift, not
# shifted outflows (a group whose redshift ends at the bound is removed)
NARROW_VELOCITY_RANGE = 300.0

# Rest-frame wavelengths of the CIII] complex (`qso.CIII_complex_model_func`)
CIII_COMPLEX = {"cen": 1908.73, "cen_alIII": 1857.40, "cen_siIII": 1892.03}


def _fixed(value):
    return {"value": value, "vary": False}


def line_rules(
    redshift,
    lines=DEFAULT_LINES,
    broad_fwhm=(1000.0, 15000.0),
    narrow_fwhm=(100.0, 2500.0),
    velocity_range=5000.0,
    max_components=3,
    shared=None,
    **kwargs,
):
    """One `GaussianLineRule` per line, at the line's rest wavelength
    (fixed center, redshift free within +-`velocity_range`), with up to
    `max_components` Gaussians each. Doublets in `DOUBLETS` (e.g. "OIII")
    are one component per pair, with one redshift and width and a fixed
    flux ratio. Lines in `LINES_WITH_NARROW` cover both FWHM ranges; tie
    their narrow components with `narrow_line_tie`.

    :param redshift: Systemic redshift
    :type redshift: float
    :param lines: Names in `QSO_LINES`, or name -> rest wavelength
    :type lines: list(str) or dict
    :param broad_fwhm: FWHM range of the broad permitted lines, km/s
    :type broad_fwhm: (float, float)
    :param narrow_fwhm: FWHM range of the forbidden lines ([OIII]), km/s
    :type narrow_fwhm: (float, float)
    :param velocity_range: Allowed velocity offset, km/s (CIV is often
        blueshifted by a few 1000 km/s)
    :type velocity_range: float
    :param max_components: Maximum Gaussians per line
    :type max_components: int
    :param shared: Line name -> {argument: parameter name}: the first
        component of the line uses these parameters, shared with the lines
        that name the same ones; further components get their own. A fixed
        tie, unlike `narrow_line_tie`, which the data may reject.
    :type shared: dict
    :param kwargs: Passed on to `GaussianLineRule`
    :return: The rules
    :rtype: list(sculptor.autofit.GaussianLineRule)
    """
    shared = shared or {}
    if not isinstance(lines, dict):
        lines = {
            name: (DOUBLETS[name][1] if name in DOUBLETS else QSO_LINES[name])
            for name in lines
        }
    rules = []
    for name, rest in lines.items():
        function, fixed = None, None
        if name in DOUBLETS:
            function, _, fixed = DOUBLETS[name]
        if name in NARROW_LINES:
            fwhm_range = narrow_fwhm
        elif name in LINES_WITH_NARROW:
            fwhm_range = (narrow_fwhm[0], broad_fwhm[1])
        else:
            fwhm_range = broad_fwhm
        rules.append(
            GaussianLineRule(
                name,
                redshift=redshift,
                lines=[rest],
                fwhm_range=fwhm_range,
                velocity_range=velocity_range,
                max_components=max_components,
                function=function,
                fixed=fixed,
                shared=shared.get(name),
                n_shared=1,
                **kwargs,
            )
        )
    return rules


def narrow_line_tie(lines=NARROW_TIE_LINES, max_fwhm=None, args=None):
    """Hypothesis that the narrow components of `lines` (the narrowest
    Gaussian of each with FWHM <= `max_fwhm`) share redshift and FWHM, in
    parameters `narrow_redsh` and `narrow_fwhm_km_s`. AutoFit tests it once
    its search converges and keeps it unless the untied model earns the
    threshold; `AutoFitResult.ties["narrow"]` reports the outcome (e.g.
    rejected when [OIII] is dominated by a shifted outflow).

    :param lines: Line names (rule prefixes of `line_rules`)
    :type lines: list(str)
    :param max_fwhm: Largest narrow FWHM, km/s (default `NARROW_FWHM_MAX`,
        read when called)
    :type max_fwhm: float
    :param args: Shared arguments (default redshift and FWHM)
    :type args: list(str)
    :return: The tie (pass it among the rules)
    :rtype: sculptor.autofit.Tie
    """
    return Tie(
        "narrow",
        lines,
        args=("redsh", "fwhm_km_s") if args is None else args,
        max_width=NARROW_FWHM_MAX if max_fwhm is None else max_fwhm,
    )


def narrow_line_group(
    rules, redshift, lines=NARROW_TIE_LINES, max_fwhm=None, min_fwhm=100.0, **kwargs
):
    """Rule that proposes the narrow components of `lines` together, at the
    systemic redshift, sharing redshift and FWHM (`narrow_redsh`,
    `narrow_fwhm_km_s`, the names `narrow_line_tie` uses): weak narrow
    lines that would not earn the threshold one at a time may earn it
    together. See `sculptor.autofit.LineGroupRule`.

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
        `NARROW_VELOCITY_RANGE`)
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
    return LineGroupRule(
        "narrow",
        members,
        redshift,
        fwhm_range=(min_fwhm, NARROW_FWHM_MAX if max_fwhm is None else max_fwhm),
        **kwargs,
    )


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


def iron_template_rule(
    redshift,
    preset="VW01_UV",
    prefix="FeII",
    fwhm=2500.0,
    fwhm_range=(1000.0, 10000.0),
    intr_fwhm=900.0,
    vary_fwhm=True,
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
        return setup(prefix, fe_dict=fe_dict, fast=True)

    return ComponentRule(f"iron ({preset})", factory, linear_param="amp")


def balmer_continuum_rule(
    redshift, prefix="BalmerCont", Te=15000.0, tau_be=1.0, lambda_be=3646.0
):
    """Rule that adds a Balmer continuum (Dietrich et al. 2003, without the
    high-order lines: `qso_advanced_models.fast_balmer_continuum_model`),
    with its amplitude estimated from the residuals. Only proposed if the
    fit regions reach below the Balmer edge.

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
    :return: The rule
    :rtype: sculptor.autofit.ComponentRule
    """

    def factory(model):
        """The Balmer continuum component, if the fit covers the edge."""
        x = model.spec.dispersion[model.get_fit_mask()]
        if x.size == 0 or x.min() > lambda_be * (1 + redshift):
            return None
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
        return qa.initialize_balmer_without_hol(prefix, balmer_dict)

    return ComponentRule("Balmer continuum", factory, linear_param="amp_be")


def line_dict_rule(line_dict, line, kind, max_components=3):
    """Rule that adds Gaussians of one kind of a line from an emission
    line dictionary, as `qso_autofit.add_line_components` does, one per
    use: `<line>_<kind>_a`, `<line>_<kind>_b`, ... The flux starts at the
    least-squares amplitude on the residuals.

    Use it to let the criterion choose the number of components of a
    `setup_<line>_lines` setup, e.g. `line_dict_rule(line_dict, "CIV",
    "broad")`.

    :param line_dict: Emission line dictionary (see
        `qso_autofit.initialize_gaussian_line_model`)
    :type line_dict: dict
    :param line: Line name, e.g. "CIV"
    :type line: str
    :param kind: Component kind, e.g. "broad", "broadshifted", "narrow"
    :type kind: str
    :param max_components: Maximum components of this kind
    :type max_components: int
    :return: The rule
    :rtype: sculptor.autofit.ComponentRule
    """
    settings = line_dict[line][kind]["params"]

    def factory(model):
        """The next lettered component."""
        names = {c.name for c in model.components}
        for letter in string.ascii_lowercase[:max_components]:
            name = f"{line}_{kind}_{letter}"
            if name not in names:
                return qa.initialize_gaussian_line_model(name, copy.deepcopy(settings))
        return None

    return ComponentRule(
        f"{line} {kind}",
        factory,
        max_uses=max_components,
        linear_param="flux",
    )


def qso_rules(
    redshift,
    lines=DEFAULT_LINES,
    iron="VW01_UV",
    balmer=True,
    narrow_tie=True,
    narrow_group=True,
    ciii_complex=False,
    **line_kwargs,
):
    """Default rules for a quasar spectrum: Gaussians at `lines`, an iron
    template, a Balmer continuum, the narrow lines proposed together at
    the systemic redshift and the narrow-line tie.

    :param redshift: Systemic redshift
    :type redshift: float
    :param lines: See `line_rules`
    :type lines: list(str) or dict
    :param iron: Iron template preset (see `iron_template_rule`), or None
    :type iron: str
    :param balmer: Include the Balmer continuum rule
    :type balmer: bool
    :param narrow_tie: Include `narrow_line_tie` (for the tied lines among
        `lines`)
    :type narrow_tie: bool
    :param narrow_group: Include `narrow_line_group` (same lines)
    :type narrow_group: bool
    :param ciii_complex: Include `ciii_complex_rule` (with "CIII" in
        `lines`)
    :type ciii_complex: bool
    :param line_kwargs: Passed on to `line_rules`
    :return: The rules
    :rtype: list
    """
    rules = line_rules(redshift, lines, **line_kwargs)
    if narrow_group:
        group = narrow_line_group(rules, redshift)
        if group is not None:
            rules.append(group)
    if ciii_complex and "CIII" in lines:
        rules.append(ciii_complex_rule(redshift))
    if iron is not None:
        rules.append(iron_template_rule(redshift, preset=iron))
    if balmer:
        rules.append(balmer_continuum_rule(redshift))
    if narrow_tie:
        tied = [name for name in NARROW_TIE_LINES if name in lines]
        if len(tied) >= 2:
            rules.append(narrow_line_tie(tied))
    return rules
