#!/usr/bin/env python

import pkg_resources

from sculptor import specfit as scfit
from sculptor import speconed as scspec


def setup_example_fit():
    """Set up an example fit using SpecFit and SpecModels objects.

    :return:
    """

    # Load the example spectrum
    spec = scspec.SpecOneD()
    filename = pkg_resources.resource_filename(
        'sculptor',
        'data/example_spectra/J030341.04-002321.8_0.fits')
    spec.read_sdss_fits(filename)
    redshift = 3.227

    # Initialize SpecFit object
    fit = scfit.SpecFit(spec, redshift)

    # Add the continuum SpecModel
    fit.add_specmodel()
    contmodel = fit.specmodels[0]
    contmodel.name = 'Continuum'

    contmodel.add_wavelength_range_to_fit_mask(8300, 8620)
    contmodel.add_wavelength_range_to_fit_mask(7105, 7205)
    contmodel.add_wavelength_range_to_fit_mask(5400, 5450)
    contmodel.add_wavelength_range_to_fit_mask(5685, 5750)
    contmodel.add_wavelength_range_to_fit_mask(6145, 6212)

    model_name = 'Power Law (2500A)'
    model_prefix = 'PL_'
    contmodel.add_model(model_name, model_prefix)
    contmodel.fit()

    fit.plot()

    # Add the SiIV emission line model
    fit.add_specmodel()
    siiv_model = fit.specmodels[1]
    siiv_model.name = 'SiIV_line'

    siiv_model.add_wavelength_range_to_fit_mask(5790, 5870)
    siiv_model.add_wavelength_range_to_fit_mask(5910, 6015)

    model_name = 'SiIV (2G components)'
    model_prefix = 'will be automatically replaced by model'
    siiv_model.add_model(model_name, model_prefix, amplitude=20)

    # Make the redshift a variable parameters
    params = siiv_model.params_list[0]
    params['SiIV_A_z'].vary = True
    params = siiv_model.params_list[1]
    params['SiIV_B_z'].vary = True

    siiv_model.fit()

    # Add the CIV emission line model
    fit.add_specmodel()
    civ_model = fit.specmodels[2]
    civ_model.name = 'CIV_line'

    civ_model.add_wavelength_range_to_fit_mask(6240, 6700)

    model_name = 'CIV (2G components)'
    model_prefix = 'will be automatically replaced by model'
    civ_model.add_model(model_name, model_prefix, amplitude=10)

    # Make the redshift a variable parameter
    params = civ_model.params_list[0]
    params['CIV_A_z'].vary = True
    params = civ_model.params_list[1]
    params['CIV_B_z'].vary = True

    civ_model.fit()

    # Add the CIII] complex emission line model
    fit.add_specmodel()
    ciii_model = fit.specmodels[3]
    ciii_model.name = 'CIII]_complex'

    ciii_model.add_wavelength_range_to_fit_mask(7800, 8400)

    model_name = 'CIII] complex (3G components)'
    model_prefix = 'will be automatically replaced by model'
    ciii_model.add_model(model_name, model_prefix, amplitude=2)

    params = ciii_model.params_list[0]
    params['CIII_z'].vary = True

    ciii_model.fit()
    ciii_model.fit()

    # Add absorption line models
    fit.add_specmodel()
    abs_model = fit.specmodels[4]
    abs_model.name = 'Abs_lines'

    abs_model.add_wavelength_range_to_fit_mask(5760, 5790)

    model_name = 'Line model Gaussian'
    model_prefix = 'Abs_A'
    abs_model.add_model(model_name, model_prefix, amplitude=-15,
                        cenwave=5766, fwhm=200, redshift=0)
    model_name = 'Line model Gaussian'
    model_prefix = 'Abs_B'
    abs_model.add_model(model_name, model_prefix, amplitude=-15,
                        cenwave=5776, fwhm=200, redshift=0)

    abs_model.fit()

    fit.fit(save_results=True, foldername='example_spectrum_fit')
    fit.plot()

    fit.save('example_spectrum_fit')


setup_example_fit()
