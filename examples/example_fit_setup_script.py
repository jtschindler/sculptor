#!/usr/bin/env python



from speconed import speconed as sod

from sculptor import model as scmod
from sculptor import prior as scpri
from sculptor import component as sccomp
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
    pars_siiv_a['SiIV_a_redsh'].prior = scpri.UniformPrior('SiIV_redsh', 3.17, 3.27)
    pars_siiv_a['SiIV_a_fwhm_km_s'].value = 1200
    pars_siiv_a['SiIV_a_fwhm_km_s'].prior = scpri.UniformPrior('SiIV_fwhm_km_s', 300, 6000)

    # comp_siiv_a.param_mapping.update({'redsh': 'redsh_nl'})


    comp_siiv_b = sccomp.FitComponent('SiIV_b', ana.line_model_gaussian)
    pars_siiv_b = comp_siiv_b.create_params()

    pars_siiv_b['SiIV_b_flux'].value = 0.0006
    pars_siiv_b['SiIV_b_flux'].prior = scpri.UniformPrior('SiIV_flux', 0, 1 * amp_factor)
    pars_siiv_b['SiIV_b_cen'].value = 1399.8
    pars_siiv_b['SiIV_b_cen'].vary = False
    pars_siiv_b['SiIV_b_redsh'].prior = scpri.UniformPrior('SiIV_redsh', 3.17, 3.27)
    pars_siiv_b['SiIV_b_fwhm_km_s'].value = 1200
    pars_siiv_b['SiIV_b_fwhm_km_s'].prior = scpri.UniformPrior('SiIV_fwhm_km_s', 500, 8000)

    # comp_siiv_b.param_mapping.update({'redsh': 'redsh_nl'})

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

    # comp_civ_a.param_mapping.update({'redsh': 'redsh_nl'})

    comp_civ_b = sccomp.FitComponent('CIV_b', ana.line_model_gaussian)
    pars_civ_b = comp_civ_b.create_params()

    pars_civ_b['CIV_b_flux'].value = 0.0006
    pars_civ_b['CIV_b_flux'].prior = scpri.UniformPrior('CIV_flux', 0, 10 * amp_factor)
    pars_civ_b['CIV_b_cen'].value = 1549.06
    pars_civ_b['CIV_b_cen'].vary = False
    pars_civ_b['CIV_b_redsh'].prior = scpri.UniformPrior('CIV_redsh', 3.18, 3.26)
    pars_civ_b['CIV_b_fwhm_km_s'].value = 1200
    pars_civ_b['CIV_b_fwhm_km_s'].prior = scpri.UniformPrior('CIV_fwhm_km_s', 1000, 12000)

    # comp_civ_b.param_mapping.update({'redsh': 'redsh_nl'})

    # Consolidate model components
    model.components = [comp_pl, comp_siiv_a, comp_siiv_b, comp_civ_a, comp_civ_b]
    # Consolidate model parameters
    params = {}
    for pars in [pars_pl, pars_siiv_a, pars_siiv_b, pars_civ_a, pars_civ_b]:
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

    # # Initialize SpecFit object
    # fit = scfit.SpecFit(spec, redshift)
    #
    # # Add the continuum SpecModel
    # fit.add_specmodel()
    # contmodel = fit.specmodels[0]
    # contmodel.name = 'Continuum'
    #
    # contmodel.add_wavelength_range_to_fit_mask(8300, 8620)
    # contmodel.add_wavelength_range_to_fit_mask(7105, 7205)
    # contmodel.add_wavelength_range_to_fit_mask(5400, 5450)
    # contmodel.add_wavelength_range_to_fit_mask(5685, 5750)
    # contmodel.add_wavelength_range_to_fit_mask(6145, 6212)
    #
    # model_name = 'Power Law (2500A)'
    # model_prefix = 'PL_'
    # contmodel.add_model(model_name, model_prefix)
    # contmodel.fit()
    #
    # fit.plot()
    #
    # # Add the SiIV emission line model
    # fit.add_specmodel()
    # siiv_model = fit.specmodels[1]
    # siiv_model.name = 'SiIV_line'
    #
    # siiv_model.add_wavelength_range_to_fit_mask(5790, 5870)
    # siiv_model.add_wavelength_range_to_fit_mask(5910, 6015)
    #
    # model_name = 'SiIV (2G components)'
    # model_prefix = 'will be automatically replaced by model'
    # siiv_model.add_model(model_name, model_prefix, amplitude=20)
    #
    # # Make the redshift a variable parameters
    # params = siiv_model.params_list[0]
    # params['SiIV_A_z'].vary = True
    # params = siiv_model.params_list[1]
    # params['SiIV_B_z'].vary = True
    #
    # siiv_model.fit()
    #
    # # Add the CIV emission line model
    # fit.add_specmodel()
    # civ_model = fit.specmodels[2]
    # civ_model.name = 'CIV_line'
    #
    # civ_model.add_wavelength_range_to_fit_mask(6240, 6700)
    #
    # model_name = 'CIV (2G components)'
    # model_prefix = 'will be automatically replaced by model'
    # civ_model.add_model(model_name, model_prefix, amplitude=10)
    #
    # # Make the redshift a variable parameter
    # params = civ_model.params_list[0]
    # params['CIV_A_z'].vary = True
    # params = civ_model.params_list[1]
    # params['CIV_B_z'].vary = True
    #
    # civ_model.fit()
    #
    # # Add the CIII] complex emission line model
    # fit.add_specmodel()
    # ciii_model = fit.specmodels[3]
    # ciii_model.name = 'CIII]_complex'
    #
    # ciii_model.add_wavelength_range_to_fit_mask(7800, 8400)
    #
    # model_name = 'CIII] complex (3G components)'
    # model_prefix = 'will be automatically replaced by model'
    # ciii_model.add_model(model_name, model_prefix, amplitude=2)
    #
    # params = ciii_model.params_list[0]
    # params['CIII_z'].vary = True
    #
    # ciii_model.fit()
    # ciii_model.fit()
    #
    # # Add absorption line models
    # fit.add_specmodel()
    # abs_model = fit.specmodels[4]
    # abs_model.name = 'Abs_lines'
    #
    # abs_model.add_wavelength_range_to_fit_mask(5760, 5790)
    #
    # model_name = 'Line model Gaussian'
    # model_prefix = 'Abs_A'
    # abs_model.add_model(model_name, model_prefix, amplitude=-15,
    #                     cenwave=5766, fwhm=200, redshift=0)
    # model_name = 'Line model Gaussian'
    # model_prefix = 'Abs_B'
    # abs_model.add_model(model_name, model_prefix, amplitude=-15,
    #                     cenwave=5776, fwhm=200, redshift=0)
    #
    # abs_model.fit()
    #
    # fit.fit(save_results=True, foldername='example_spectrum_fit')
    # fit.plot()
    #
    # fit.save('example_spectrum_fit')


if __name__ == '__main__':

    setup_example_fit()
