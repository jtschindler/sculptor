#!/usr/bin/env python

from astropy.cosmology import FlatLambdaCDM

from sculptor import specfit as scfit
from sculptor import specanalysis as scana


def analyze_mcmc_file():
    """Analyzing the example spectrum MCMC CIV line fit
    """

    # Define Cosmology for cosmological conversions
    cosmo = FlatLambdaCDM(H0=70, Om0=0.3, Tcmb0=2.725)

    # Instantiate an empty SpecFit object
    fit = scfit.SpecFit()
    # Load the example spectrum fit
    fit.load('example_spectrum_fit')
    emission_feature_listdict = [{'feature_name': 'CIV',
                             'model_names' : ['CIV_A_', 'CIV_B_'],
                             'rest_frame_wavelength': 1549.06}
                                 ]
    continuum_listdict = {'model_names': ['PL_'],
                          'rest_frame_wavelengths': [1450, 1280]}

    scana.analyse_mcmc_results('example_spectrum_fit', fit,
                               continuum_listdict,
                               emission_feature_listdict,
                               fit.redshift, cosmo, concatenate=True)


analyze_mcmc_file()
