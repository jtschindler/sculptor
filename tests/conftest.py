import sys
from pathlib import Path

import numpy as np
import pytest
from astropy import units as u
from speconed import speconed as sod

import sculptor
from sculptor import analysis as ana
from sculptor import component as sccomp
from sculptor import model as scmod
from sculptor import prior as scpri

REPO_ROOT = Path(__file__).resolve().parents[1]
DEPRECATED_SCULPTOR_DIR = REPO_ROOT / "deprecated" / "src" / "sculptor"


@pytest.fixture(scope="module")
def deprecated_dep_v1():
    """Make `sculptor.dep_v1.*` (now under `deprecated/`) importable.

    The ground-truth reference copies in `tests/ground_truth/` import
    `sculptor.dep_v1.speconed` / `.specanalysis` under their old names. This
    exposes them only for the test module that requests the fixture, so the
    maintained package cannot silently depend on deprecated code.
    """
    sculptor.__path__.append(str(DEPRECATED_SCULPTOR_DIR))
    try:
        yield
    finally:
        sculptor.__path__.remove(str(DEPRECATED_SCULPTOR_DIR))
        for name in [m for m in sys.modules if m.startswith("sculptor.dep_v1")]:
            del sys.modules[name]
        if hasattr(sculptor, "dep_v1"):
            del sculptor.dep_v1


REDSHIFT = 3.227  # used by the model fixtures below


def _line_component(name="SiIV_a"):
    comp = sccomp.FitComponent(name, ana.line_model_gaussian)
    pars = comp.create_params()
    pars[f"{name}_redsh"].value = REDSHIFT
    pars[f"{name}_redsh"].vary = False
    pars[f"{name}_flux"].value = 6.0
    pars[f"{name}_flux"].prior = scpri.UniformPrior(f"{name}_flux", 0, 100)
    pars[f"{name}_cen"].value = 1399.8
    pars[f"{name}_cen"].vary = False
    pars[f"{name}_fwhm_km_s"].value = 1200
    pars[f"{name}_fwhm_km_s"].prior = scpri.UniformPrior(f"{name}_fwhm_km_s", 300, 6000)
    return comp, pars


def _fit_model():
    """Power law + Gaussian line on a synthetic spectrum, short emcee run."""

    comp_pl = sccomp.FitComponent("pl", ana.power_law_at2500)
    pars_pl = comp_pl.create_params()
    pars_pl["pl_redsh"].value = REDSHIFT
    pars_pl["pl_redsh"].vary = False
    pars_pl["pl_amp"].value = 0.02
    pars_pl["pl_amp"].prior = scpri.UniformPrior("pl_amp", 0, 1)
    pars_pl["pl_slope"].value = -1.5
    pars_pl["pl_slope"].prior = scpri.GaussianPrior("pl_slope", -1.5, 0.3)

    comp_line, pars_line = _line_component()

    dispersion = np.linspace(5500, 6300, 400)
    rng = np.random.default_rng(42)
    truth = ana.power_law_at2500(
        dispersion, REDSHIFT, 0.02, -1.5
    ) + ana.line_model_gaussian(dispersion, REDSHIFT, 6.0, 1399.8, 1200)
    error = np.full_like(dispersion, 0.002)
    spec = sod.SpecOneD(
        dispersion=dispersion,
        fluxden=truth + rng.normal(0, 0.002, dispersion.size),
        fluxden_err=error,
        dispersion_unit=u.AA,
        fluxden_unit=u.erg / u.s / u.cm**2 / u.AA,
    )

    model = scmod.FitModel(
        components=[comp_pl, comp_line],
        parameters={**pars_pl, **pars_line},
        spectrum=spec,
        redshift=REDSHIFT,
    )
    model.add_wavelength_range_to_fit_mask(5600, 6200)
    model.remove_wavelength_range_from_fit_mask(5800, 5820)
    model.run_emcee(nsteps=20, nwalkers=10, progress=False)
    return model


@pytest.fixture
def line_component():
    """Factory: a Gaussian line FitComponent and its parameters."""
    return _line_component


@pytest.fixture
def fit_model():
    """Power law + Gaussian line on a synthetic spectrum, after a short
    emcee run (20 steps, 10 walkers)."""
    return _fit_model()
