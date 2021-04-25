#!/usr/bin/env python

import corner
import matplotlib.pyplot as plt
from astropy.cosmology import FlatLambdaCDM
from lmfit import Model, Parameters, fit_report

from sculptor import specfit as scfit


def example_fit_mcmc():
    """Fitting the Sculptor example spectrum CIV line using MCMC

    :return:
    """

    # Import the saved example spectrum
    fit = scfit.SpecFit()
    fit.load('example_spectrum_fit')

    # Setting the fit method to MCMC via emcee
    fit.fitting_method = 'Maximum likelihood via Monte-Carlo Markov Chain'

    # Set the MCMC keywords
    # They can be accessed via fit.emcee_kws
    fit.emcee_kws['steps'] = 5000
    fit.emcee_kws['burn'] = 500
    # We are fitting 6 parameters so nwalker=50 is fine
    fit.emcee_kws['nwalkers'] = 50
    # No multiprocessing for now
    fit.emcee_kws['workers'] = 1
    fit.emcee_kws['thin'] = 2
    fit.emcee_kws['progress'] = True
    # Take uncertainties into account
    fit.emcee_kws['is_weighted'] = True

    # Select the CIV emission line SpecModel
    civ_model = fit.specmodels[2]

    # Fit the SpecModel using the MCMC method and emcee_kws modified above
    civ_model.fit()

    # Print the fit result
    print(civ_model.fit_result.fit_report())

    # Retrieve the MCMC flat chain of the CIV model fit
    data = civ_model.fit_result.flatchain.to_numpy()
    # Visualize the flat chain fit results using the typical corner plot
    corner_plot = corner.corner(data,
                                labels=civ_model.fit_result.var_names,
                                quantiles=[0.16, 0.5, 0.84],
                                show_titles=True,
                                title_kwargs={"fontsize": 12}
                                )
    plt.show()

    # Save the MCMC flatchain to a file for analysis
    civ_model.save_mcmc_chain('example_spectrum_fit')


example_fit_mcmc()
