"""Example: fit the SiIV and CIV lines of an SDSS quasar spectrum with emcee.

A power-law continuum plus two Gaussians for each line; all lines share one
redshift parameter (`redsh_nl`). Run from anywhere:

    uv run python examples/example_fit_setup_script.py
"""

from pathlib import Path

from speconed import speconed as sod

from sculptor import analysis as ana
from sculptor import component as sccomp
from sculptor import model as scmod
from sculptor import parameter as scpar
from sculptor import prior as scpri

SPECTRUM = (
    Path(__file__).resolve().parents[1]
    / "src/sculptor/data/example_spectra/J030341.04-002321.8_0.fits"
)
REDSHIFT = 3.227
AMP_FACTOR = 1000


def gaussian_line(name, cen, flux_max, fwhm_range):
    """A Gaussian line at rest-frame `cen` whose redshift is the shared
    `redsh_nl` parameter."""

    comp = sccomp.FitComponent(name, ana.line_model_gaussian)
    pars = comp.create_params()

    pars[f"{name}_flux"].value = 0.0006
    pars[f"{name}_flux"].prior = scpri.UniformPrior(f"{name}_flux", 0, flux_max)
    pars[f"{name}_cen"].value = cen
    pars[f"{name}_cen"].vary = False
    pars[f"{name}_fwhm_km_s"].value = 1200
    pars[f"{name}_fwhm_km_s"].prior = scpri.UniformPrior(
        f"{name}_fwhm_km_s", *fwhm_range
    )

    # Use the shared redshift instead of the component's own.
    pars.pop(f"{name}_redsh")
    comp.param_mapping.update({"redsh": "redsh_nl"})
    return comp, pars


def setup_example_fit(nsteps=2000, nwalkers=50, discard=500, save_dir="."):
    """Set up and run the example fit, and save the result plots.

    :param nsteps: Number of MCMC steps
    :type nsteps: int
    :param nwalkers: Number of walkers
    :type nwalkers: int
    :param discard: Burn-in steps to discard in the plots
    :type discard: int
    :param save_dir: Directory for the plots
    :type save_dir: str
    :return: The fitted model
    :rtype: sculptor.model.FitModel
    """

    spec = sod.SpecOneD()
    spec.read_sdss_fits(str(SPECTRUM))

    model = scmod.FitModel(spectrum=spec, redshift=REDSHIFT)

    # Fit regions (observed frame, Angstrom)
    for lo, hi in [
        (8300, 8620),
        (7105, 7205),
        (5400, 5450),
        (5685, 5750),
        (6145, 6212),
        (5750, 7500),
    ]:
        model.add_wavelength_range_to_fit_mask(lo, hi)
    model.remove_wavelength_range_from_fit_mask(5760, 5780)

    # Continuum
    comp_pl = sccomp.FitComponent("pl", ana.power_law_at2500)
    pars_pl = comp_pl.create_params()
    pars_pl["pl_amp"].prior = scpri.UniformPrior("amp", 0, 0.1 * AMP_FACTOR)
    pars_pl["pl_slope"].prior = scpri.UniformPrior("slope", -2, 1)
    pars_pl["pl_redsh"].value = REDSHIFT
    pars_pl["pl_redsh"].vary = False

    # Shared narrow-line redshift
    redsh_nl = scpar.FitParameter("redsh_nl", value=REDSHIFT, vary=True)
    redsh_nl.prior = scpri.UniformPrior("redsh_nl", 3.2, 3.3)

    lines = [
        gaussian_line("SiIV_a", 1399.8, 1 * AMP_FACTOR, (300, 6000)),
        gaussian_line("SiIV_b", 1399.8, 1 * AMP_FACTOR, (500, 8000)),
        gaussian_line("CIV_a", 1549.06, 10 * AMP_FACTOR, (300, 6000)),
        gaussian_line("CIV_b", 1549.06, 10 * AMP_FACTOR, (1000, 12000)),
    ]

    model.add_component(comp_pl)
    model.add_parameters(pars_pl)
    model.add_parameters({"redsh_nl": redsh_nl})
    for comp, pars in lines:
        model.add_component(comp)
        model.add_parameters(pars)

    model.run_emcee(nsteps, nwalkers)

    model.plot_mcmc_result(discard=discard, save_dir=save_dir, save=True, ylim=[0, 120])
    model.plot_mcmc_result(
        discard=discard,
        save_dir=save_dir,
        save=True,
        save_name="fit_result_zoom.pdf",
        ylim=[0, 120],
        show_components=True,
    )
    model.plot_posterior_corner(save_dir=save_dir, discard=discard, save=True)
    return model


if __name__ == "__main__":
    setup_example_fit()
