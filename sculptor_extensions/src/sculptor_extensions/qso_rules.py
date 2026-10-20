"""Rules for automatic model building of type-I quasar spectra
(`sculptor.autofit`), built on the `qso_presets` / `qso_autofit` setups.

`qso_rules(redshift)` returns a default set: Gaussians at the strong
broad and narrow lines in the fit regions, the UV iron template and the
Balmer continuum. Start `AutoFit` from a power-law continuum::

    from sculptor.autofit import AutoFit
    from sculptor_extensions import qso_rules

    rules = qso_rules.qso_rules(redshift=7.16)
    result = AutoFit(base_model, rules, criterion="bic").run()

Each rule can also be built on its own (`line_rules`, `iron_template_rule`,
`balmer_continuum_rule`, `line_dict_rule`).
"""

import copy
import string

from sculptor.autofit import ComponentRule, GaussianLineRule
from sculptor.prior import UniformPrior
from sculptor_extensions import qso_autofit as qa
from sculptor_extensions import qso_presets

# Rest-frame vacuum wavelengths (Angstrom) of the strong quasar lines
QSO_LINES = {
    "Lya": 1215.67,
    "NV": 1240.14,
    "SiIV": 1399.80,
    "CIV": 1549.06,
    "HeII": 1640.42,
    "CIII": 1908.73,
    "MgII": 2798.75,
    "Hgamma": 4341.68,
    "Hbeta": 4862.68,
    "OIII_4960": 4960.30,
    "OIII_5008": 5008.24,
    "Halpha": 6564.61,
}

# Lines fitted by default: Lya and NV need absorption modeling and are left
# out
DEFAULT_LINES = ("SiIV", "CIV", "CIII", "MgII", "Hbeta", "OIII_5008", "Halpha")

# Narrow lines get a narrow FWHM range
NARROW_LINES = ("OIII_4960", "OIII_5008")


def _fixed(value):
    return {"value": value, "vary": False}


def line_rules(
    redshift,
    lines=DEFAULT_LINES,
    broad_fwhm=(1000.0, 15000.0),
    narrow_fwhm=(100.0, 1500.0),
    velocity_range=5000.0,
    max_components=3,
    **kwargs,
):
    """One `GaussianLineRule` per line, at the line's rest wavelength
    (fixed center, redshift free within +-`velocity_range`), with up to
    `max_components` Gaussians each.

    :param redshift: Systemic redshift
    :type redshift: float
    :param lines: Names in `QSO_LINES`, or name -> rest wavelength
    :type lines: list(str) or dict
    :param broad_fwhm: FWHM range of the permitted lines, km/s
    :type broad_fwhm: (float, float)
    :param narrow_fwhm: FWHM range of the forbidden lines ([OIII]), km/s
    :type narrow_fwhm: (float, float)
    :param velocity_range: Allowed velocity offset, km/s (CIV is often
        blueshifted by a few 1000 km/s)
    :type velocity_range: float
    :param max_components: Maximum Gaussians per line
    :type max_components: int
    :param kwargs: Passed on to `GaussianLineRule`
    :return: The rules
    :rtype: list(sculptor.autofit.GaussianLineRule)
    """
    if not isinstance(lines, dict):
        lines = {name: QSO_LINES[name] for name in lines}
    return [
        GaussianLineRule(
            name,
            redshift=redshift,
            lines=[rest],
            fwhm_range=narrow_fwhm if name in NARROW_LINES else broad_fwhm,
            velocity_range=velocity_range,
            max_components=max_components,
            **kwargs,
        )
        for name, rest in lines.items()
    ]


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
    **line_kwargs,
):
    """Default rules for a quasar spectrum: Gaussians at `lines`, an iron
    template and a Balmer continuum.

    :param redshift: Systemic redshift
    :type redshift: float
    :param lines: See `line_rules`
    :type lines: list(str) or dict
    :param iron: Iron template preset (see `iron_template_rule`), or None
    :type iron: str
    :param balmer: Include the Balmer continuum rule
    :type balmer: bool
    :param line_kwargs: Passed on to `line_rules`
    :return: The rules
    :rtype: list
    """
    rules = line_rules(redshift, lines, **line_kwargs)
    if iron is not None:
        rules.append(iron_template_rule(redshift, preset=iron))
    if balmer:
        rules.append(balmer_continuum_rule(redshift))
    return rules
