#!/usr/bin/env python



from speconed import speconed as sod

from sculptor import model as scmod
from sculptor import prior as scpri
from sculptor import component as sccomp
from sculptor import parameter as scpar
from sculptor import analysis as ana


def setup_example_fit(nsteps=2000, nwalkers=50, discard=500,
                      save_dir='.', save=True):
    """Set up an example fit using SpecFit and SpecModels objects.

    :return:
    """

    # Load the example spectrum
    spec = sod.SpecOneD()
    filename = '../sculptor/data/example_spectra/J030341.04-002321.8_0.fits'
    spec.read_sdss_fits(filename)
    redshift = 3.227

    # spec.plot(ymax=200)

    # Initialize the empty FitModel
    model = scmod.FitModel(spectrum=spec)

    # Define wavelength ranges for the fit
    model.add_wavelength_range_to_fit_mask(8300, 8620)
    model.add_wavelength_range_to_fit_mask(7105, 7205)
    model.add_wavelength_range_to_fit_mask(5400, 5450)
    model.add_wavelength_range_to_fit_mask(5685, 5750)
    model.add_wavelength_range_to_fit_mask(6145, 6212)

    model.add_wavelength_range_to_fit_mask(5750, 7500)

    model.remove_wavelength_range_from_fit_mask(5760, 5780)

    # Define the continuum model
    comp_pl = sccomp.FitComponent('pl', ana.power_law_at2500)
    pars_pl = comp_pl.create_params()

    # Global parameters
    par_redsh_nl = scpar.FitParameter('redsh_nl', value=redshift, vary=True)
    par_redsh_nl.prior = scpri.UniformPrior('redsh_nl', 3.2, 3.3)

    global_pars = {'redsh_nl': par_redsh_nl}

    amp_factor = 1000

    pars_pl['pl_amp'].prior = scpri.UniformPrior('amp', 0, 0.1 * amp_factor)
    pars_pl['pl_slope'].prior = scpri.UniformPrior('slope', -2, 1)
    pars_pl['pl_redsh'].value = redshift
    pars_pl['pl_redsh'].vary = False

    # Define the SiIV model
    comp_siiv_a = sccomp.FitComponent('SiIV_a', ana.line_model_gaussian)
    pars_siiv_a = comp_siiv_a.create_params()

    pars_siiv_a['SiIV_a_flux'].value = 0.0006
    pars_siiv_a['SiIV_a_flux'].prior = scpri.UniformPrior('SiIV_flux', 0, 1 * amp_factor)
    pars_siiv_a['SiIV_a_cen'].value = 1399.8
    pars_siiv_a['SiIV_a_cen'].vary = False
    # pars_siiv_a['SiIV_a_redsh'].prior = scpri.UniformPrior('SiIV_redsh', 3.17, 3.27)
    pars_siiv_a.pop('SiIV_a_redsh')
    pars_siiv_a['SiIV_a_fwhm_km_s'].value = 1200
    pars_siiv_a['SiIV_a_fwhm_km_s'].prior = scpri.UniformPrior('SiIV_fwhm_km_s', 300, 6000)

    comp_siiv_a.param_mapping.update({'redsh': 'redsh_nl'})


    comp_siiv_b = sccomp.FitComponent('SiIV_b', ana.line_model_gaussian)
    pars_siiv_b = comp_siiv_b.create_params()

    pars_siiv_b['SiIV_b_flux'].value = 0.0006
    pars_siiv_b['SiIV_b_flux'].prior = scpri.UniformPrior('SiIV_flux', 0, 1 * amp_factor)
    pars_siiv_b['SiIV_b_cen'].value = 1399.8
    pars_siiv_b['SiIV_b_cen'].vary = False
    pars_siiv_b['SiIV_b_redsh'].prior = scpri.UniformPrior('SiIV_redsh', 3.17, 3.27)
    pars_siiv_b['SiIV_b_fwhm_km_s'].value = 1200
    pars_siiv_b['SiIV_b_fwhm_km_s'].prior = scpri.UniformPrior('SiIV_fwhm_km_s', 500, 8000)

    comp_siiv_b.param_mapping.update({'redsh': 'redsh_nl'})

    # Define the CIV model
    comp_civ_a = sccomp.FitComponent('CIV_a', ana.line_model_gaussian)
    pars_civ_a = comp_civ_a.create_params()

    pars_civ_a['CIV_a_flux'].value = 0.0006
    pars_civ_a['CIV_a_flux'].prior = scpri.UniformPrior('CIV_flux', 0, 10 * amp_factor)
    pars_civ_a['CIV_a_cen'].value = 1549.06
    pars_civ_a['CIV_a_cen'].vary = False
    pars_civ_a['CIV_a_redsh'].prior = scpri.UniformPrior('CIV_redsh', 3.18, 3.26)
    pars_civ_a['CIV_a_fwhm_km_s'].value = 1200
    pars_civ_a['CIV_a_fwhm_km_s'].prior = scpri.UniformPrior('CIV_fwhm_km_s', 300, 6000)

    comp_civ_a.param_mapping.update({'redsh': 'redsh_nl'})

    comp_civ_b = sccomp.FitComponent('CIV_b', ana.line_model_gaussian)
    pars_civ_b = comp_civ_b.create_params()

    pars_civ_b['CIV_b_flux'].value = 0.0006
    pars_civ_b['CIV_b_flux'].prior = scpri.UniformPrior('CIV_flux', 0, 10 * amp_factor)
    pars_civ_b['CIV_b_cen'].value = 1549.06
    pars_civ_b['CIV_b_cen'].vary = False
    pars_civ_b['CIV_b_redsh'].prior = scpri.UniformPrior('CIV_redsh', 3.18, 3.26)
    pars_civ_b['CIV_b_fwhm_km_s'].value = 1200
    pars_civ_b['CIV_b_fwhm_km_s'].prior = scpri.UniformPrior('CIV_fwhm_km_s', 1000, 12000)

    comp_civ_b.param_mapping.update({'redsh': 'redsh_nl'})

    # Consolidate model components
    model.components = [comp_pl, comp_siiv_a, comp_siiv_b, comp_civ_a, comp_civ_b]
    # Consolidate model parameters
    params = {}
    for pars in [pars_pl, pars_siiv_a, pars_siiv_b, pars_civ_a, pars_civ_b, global_pars]:
        params.update(pars)
    model.parameters = params

    # Run the MCMC
    model.run_emcee(nsteps, nwalkers)

    flat_samples = model.sampler.get_chain(discard=discard, thin=1, flat=True)

    model.plot_mcmc_result(discard=discard, save_dir=save_dir, save=True,
                           ylim=[0,120])
    model.plot_mcmc_result(discard=discard, save_dir=save_dir, save=True,
                           save_name='fit_result_zoom.pdf', ylim=[0,120],
                           show_components=True)
    model.plot_posterior_corner(save_dir=save_dir, discard=discard, save=True)


if __name__ == '__main__':

    setup_example_fit()
