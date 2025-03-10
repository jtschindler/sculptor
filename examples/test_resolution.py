from docutils.nodes import label

from sculptor import component as sccomp
from sculptor import parameter as scpar
from sculptor import analysis as ana
from sculptor import prior as scpri
from sculptor import model as scmod
from sculptor import utils as scut

from matplotlib import pyplot as plt

from IPython import embed

from astropy import units as u
from astropy import constants as const

from speconed import speconed as sod

import numpy as np


def create_a_spectral_model():


    redshift = 3.227
    amp_factor = 1000

    # Create the model
    model = scmod.FitModel()

    # Create the components

    # Define the continuum model
    comp_pl = sccomp.FitComponent('pl', ana.power_law_at2500)
    pars_pl = comp_pl.create_params()

    # Global parameters
    par_redsh_nl = scpar.FitParameter('redsh_nl', value=redshift, vary=True)
    par_redsh_nl.prior = scpri.UniformPrior('redsh_nl', 3.2, 3.3)

    global_pars = {'redsh_nl': par_redsh_nl}

    pars_pl['pl_amp'].prior = scpri.UniformPrior('amp', 0,
                                                 0.01 * amp_factor)
    pars_pl['pl_amp'].value = 0.00002 * amp_factor
    pars_pl['pl_slope'].prior = scpri.UniformPrior('slope', -2, 1)
    pars_pl['pl_slope'].value = -1.5
    pars_pl['pl_redsh'].value = redshift
    pars_pl['pl_redsh'].vary = False

    # # Define the SiIV model
    comp_siiv_a = sccomp.FitComponent('SiIV_a', ana.line_model_gaussian)
    pars_siiv_a = comp_siiv_a.create_params()

    pars_siiv_a['SiIV_a_flux'].value = 0.006 * amp_factor
    pars_siiv_a['SiIV_a_flux'].prior = scpri.UniformPrior('SiIV_flux',
                                                          0, 0.1 * amp_factor)
    pars_siiv_a['SiIV_a_cen'].value = 1399.8
    pars_siiv_a['SiIV_a_cen'].vary = False
    pars_siiv_a.pop('SiIV_a_redsh')
    pars_siiv_a['SiIV_a_fwhm_km_s'].value = 800
    pars_siiv_a['SiIV_a_fwhm_km_s'].prior = scpri.UniformPrior('SiIV_fwhm_km_s',
                                                               300, 1400)

    comp_siiv_a.param_mapping.update({'redsh': 'redsh_nl'})

    comp_siiv_b = sccomp.FitComponent('SiIV_b', ana.line_model_gaussian)
    pars_siiv_b = comp_siiv_b.create_params()

    pars_siiv_b['SiIV_b_flux'].value = 0.006 * amp_factor
    pars_siiv_b['SiIV_b_flux'].prior = scpri.UniformPrior('SiIV_flux',
                                                          0, 0.1 * amp_factor)
    pars_siiv_b['SiIV_b_cen'].value = 1399.8
    pars_siiv_b['SiIV_b_cen'].vary = False
    # pars_siiv_b['SiIV_b_redsh'].prior = scpri.UniformPrior('SiIV_redsh',
    #                                                        3.17, 3.27)
    pars_siiv_b.pop('SiIV_b_redsh')
    pars_siiv_b['SiIV_b_fwhm_km_s'].value = 3600
    pars_siiv_b['SiIV_b_fwhm_km_s'].prior = scpri.UniformPrior('SiIV_fwhm_km_s',
                                                               1200, 8000)

    comp_siiv_b.param_mapping.update({'redsh': 'redsh_nl'})

    # Define the CIV model
    comp_civ_a = sccomp.FitComponent('CIV_a', ana.line_model_gaussian)
    pars_civ_a = comp_civ_a.create_params()

    pars_civ_a['CIV_a_flux'].value = 0.006 * amp_factor
    pars_civ_a['CIV_a_flux'].prior = scpri.UniformPrior('CIV_flux',
                                                        0, 0.1 * amp_factor)
    pars_civ_a['CIV_a_cen'].value = 1549.06
    pars_civ_a['CIV_a_cen'].vary = False
    # pars_civ_a['CIV_a_redsh'].prior = scpri.UniformPrior('CIV_redsh',
    #                                                      3.18, 3.26)
    pars_civ_a.pop('CIV_a_redsh')
    # pars_civ_a['CIV_a_redsh'].value = redshift
    pars_civ_a['CIV_a_fwhm_km_s'].value = 800
    pars_civ_a['CIV_a_fwhm_km_s'].prior = scpri.UniformPrior('CIV_fwhm_km_s',
                                                             300, 1400)

    comp_civ_a.param_mapping.update({'redsh': 'redsh_nl'})

    comp_civ_b = sccomp.FitComponent('CIV_b', ana.line_model_gaussian)
    pars_civ_b = comp_civ_b.create_params()

    pars_civ_b['CIV_b_flux'].value = 0.006 * amp_factor
    pars_civ_b['CIV_b_flux'].prior = scpri.UniformPrior('CIV_flux',
                                                        0, 0.1 * amp_factor)
    pars_civ_b['CIV_b_cen'].value = 1549.06
    pars_civ_b['CIV_b_cen'].vary = False
    # pars_civ_b['CIV_b_redsh'].prior = scpri.UniformPrior('CIV_redsh',
    #                                                      3.18, 3.26)
    pars_civ_b.pop('CIV_b_redsh')
    pars_civ_b['CIV_b_fwhm_km_s'].value = 5800
    pars_civ_b['CIV_b_fwhm_km_s'].prior = scpri.UniformPrior('CIV_fwhm_km_s',
                                                             1200, 12000)

    comp_civ_b.param_mapping.update({'redsh': 'redsh_nl'})

    # Consolidate model components
    model.components = [comp_pl, comp_siiv_a, comp_siiv_b, comp_civ_a,
                        comp_civ_b]

    # Consolidate model parameters
    params = {}
    for pars in [pars_pl, pars_siiv_a, pars_siiv_b, pars_civ_a,
                 pars_civ_b, global_pars]:
        params.update(pars)
    model.parameters = params

    model.get_params_to_sample()
    param_values = [par.value for par in model.params_variable.values()]

    dispersion = np.linspace(1000*(redshift+1),
                             2000*(redshift+1), 1000)

    model_flux = model.eval(dispersion, param_values)

    model.save('test_model.pkl', 'test_folder')

    # Create constant SNR errors for the spectrum
    snr = 20
    error = model_flux / snr

    # Create a spectrum object
    spectrum = sod.SpecOneD(dispersion=dispersion,
                            fluxden=model_flux,
                            fluxden_err=error,
                            dispersion_unit=u.AA,
                            fluxden_unit=u.erg/u.s/u.cm**2/u.AA)


    # spectrum.plot()

    return spectrum


def test_resolution_broadening(spectrum, model_name, model_dir):

    # Model redshift
    redshift = 3.227



    # Load the old model
    model = scmod.FitModel()
    model.load(model_name, model_dir)

    # Constant resolution
    r = 350

    # Wavelength dependent resolution (step function)
    r = np.ones_like(spectrum.dispersion) * 350
    r[spectrum.dispersion < 1500 * (1+redshift)] = 500


    # Carry out a fit without the broadening and without the resolution
    model.spec = spectrum
    model.reset_fit_mask()

    # Add wavelength region to the model
    model.add_wavelength_range_to_fit_mask(1001*(redshift+1), 1999*(redshift+1))

    # Run the fit
    model.run_emcee(nsteps=2500, nwalkers=50)

    # Create the QA plots
    discard = 1500
    save_dir = 'test_unbroadened'

    model.plot_mcmc_result(discard=discard, save_dir=save_dir, save=True,
                           ylim=[0, 0.5])
    model.plot_mcmc_result(discard=discard, save_dir=save_dir, save=True,
                           save_name='fit_result_zoom.pdf', ylim=[0, 0.5],
                           show_components=True)
    model.plot_posterior_corner(save_dir=save_dir, discard=discard, save=True)

    # Broaden the spectrum by the resolution

    import timeit
    start = timeit.default_timer()
    for i in range(1000):
        spectrum_broadened = spectrum.broaden_by_resolution(r)
    stop = timeit.default_timer()
    print('Time: ', stop - start)
    start = timeit.default_timer()
    for i in range(1000):
        fluxden_broadened = scut.broaden_spectrum(spectrum.dispersion, spectrum.fluxden, r)
    stop = timeit.default_timer()
    print('Time: ', stop - start)
    # fluxden_broadened = scut.broaden_spectrum(spectrum.dispersion, spectrum.fluxden, r)



    # Load the old model (again)
    model2 = scmod.FitModel()
    model2.load(model_name, model_dir)

    # Update with the new spectrum
    model2.spec = spectrum_broadened

    # Add wavelength region to the model
    model2.reset_fit_mask()
    model2.add_wavelength_range_to_fit_mask(1011 * (redshift + 1), 1989 * (redshift + 1))

    # Add the constant resolution to the model
    model2.resolution = r # add constant resolution in km/s

    model2.get_params_to_sample()
    param_values = [par.value for par in model2.params_variable.values()]


    # Comparison plot
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(111)
    ax.plot(spectrum.dispersion, spectrum.fluxden, label='Original spectrum')
    ax.plot(spectrum.dispersion, model2.eval(spectrum.dispersion, param_values), label='Model spectrum', ls='--')
    ax.plot(spectrum_broadened.dispersion, fluxden_broadened, label='Broadened spectrum')

    ax2 = ax.twinx()

    ax2.plot(spectrum.dispersion, r, color='red', alpha=0.5)

    plt.show()

    # Run the fit
    model2.run_emcee(nsteps=2500, nwalkers=50)


    # Create the QA plots
    discard = 1500
    save_dir = 'test_broadened'

    model2.plot_mcmc_result(discard=discard, save_dir=save_dir, save=True,
                           ylim=[0, 0.5])
    # model.plot_mcmc_result(discard=discard, save_dir=save_dir, save=True,
    #                        save_name='fit_result_zoom.pdf', ylim=[0, 0.5],
    #                        show_components=True)
    model2.plot_posterior_corner(save_dir=save_dir, discard=discard, save=True)


if __name__ == '__main__':

    spectrum = create_a_spectral_model()

    test_resolution_broadening(spectrum, 'test_model.pkl', 'test_folder')

    # test_broadening()
