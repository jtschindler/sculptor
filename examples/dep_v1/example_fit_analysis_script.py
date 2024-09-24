
import numpy as np

from sculptor import specfit as scfit
from sculptor.dep_v1 import specanalysis as scana

from astropy import units
from astropy.cosmology import FlatLambdaCDM


def analyze_specfit_example_civ_line():
    """
    Analyze the CIV emission line properties from the example spectral fit.

    :return:
    """
    # Define Cosmology for cosmological conversions
    cosmo = FlatLambdaCDM(H0=70, Om0=0.3, Tcmb0=2.725)

    # Instantiate an empty SpecFit object
    fit = scfit.SpecFit()
    # Load the Sculptor model fit from its folder
    fit.load('example_spectrum_fit')

    # Build the continuum and CIV line spectrum
    cont_spec = scana.build_model_flux(fit, ['PL_'])

    civ_spec = scana.build_model_flux(fit, ['CIV_A_', 'CIV_B_'])
    
    # Analyze the continuum 
    fluxden_1450 = scana.get_average_fluxden(cont_spec, 1450,
                                             redshift=fit.redshift)

    lum_mono = scana.calc_lwav_from_fwav(fluxden_1450,
                                         redshift=fit.redshift,
                                         cosmology=cosmo)

    abmag = scana.calc_apparent_mag_from_fluxden(
        fluxden_1450, 1450*(1.+fit.redshift)*units.AA)

    abs_abmag = scana.calc_absolute_mag_from_monochromatic_luminosity(
        lum_mono, 1450*units.AA)

    abs_abmag2 = scana.calc_absolute_mag_from_apparent_mag(abmag, cosmo,
                                                       fit.redshift,
                                                       kcorrection=True,
                                                       a_nu=0)
    abs_abmag3 = scana.calc_absolute_mag_from_fluxden(
        fluxden_1450, 1450*(1.+fit.redshift) * units.AA,
        cosmo, fit.redshift, kcorrection=True, a_nu=0)

    # Print continuum analysis results
    print('--- Manual continuum analysis ---')
    print('Flux density at 1450AA: {:.2e}'.format(fluxden_1450 *
                                                  cont_spec.fluxden_unit))
    print('Monochromatic luminosity at 1450AA: {:.2e}'.format(lum_mono))
    print('Apparent AB magnitude (1450AA): {:.2f}'.format(abmag))
    print('Absolute AB magnitude (1) (1450AA): {:.2f}'.format(abs_abmag))
    print('Absolute AB magnitude (2) (1450AA): {:.2f}'.format(abs_abmag2))
    print('Absolute AB magnitude (3) (1450AA): {:.2f}'.format(abs_abmag3))

    # Analyze the CIV emission line properties
    civ_peak_fluxden = np.max(civ_spec.fluxden)*civ_spec.fluxden_unit
    civ_z = scana.get_peak_redshift(civ_spec, 1549.06)
    civ_ew = scana.get_equivalent_width(cont_spec, civ_spec, redshift=civ_z)
    civ_fwhm = scana.get_fwhm(civ_spec)
    civ_flux = scana.get_integrated_flux(civ_spec)
    civ_line_lum = scana.calc_integrated_luminosity(civ_spec,
                                                    fit.redshift,
                                                    cosmo)

    # Print line analysis results
    print('--- Manual CIV line analysis ---')
    print('CIV line analysis')
    print('CIV peak redshift: {:.3f}'.format(civ_z))
    print('CIV EW (rest-frame): {:.2f}'.format(civ_ew))
    print('CIV FWHM: {:.2f}'.format(civ_fwhm))
    print('CIV peak flux density: {:.2e}'.format(civ_peak_fluxden))
    print('CIV integrated flux: {:.2e}'.format(civ_flux))
    print('CIV integrated line luminosity: {:.2e} '.format(civ_line_lum))


def analyze_specfit_example():
    """Analze the CIV emission lines from the example spectrum fit using the
    built-in analyze_continuum and analyze_emission_feature_functions.

    :return:
    """
    # Define Cosmology for cosmological conversions
    cosmo = FlatLambdaCDM(H0=70, Om0=0.3, Tcmb0=2.725)

    # Instantiate an empty SpecFit object
    fit = scfit.SpecFit()
    # Load the Sculptor model fit from its folder
    fit.load('example_spectrum_fit')

    # Use the continuum analysis function
    cont_result = scana.analyze_continuum(fit, ['PL_'],
                                          [1450, 1280],
                                          cosmo, width=10)

    # Use the emission feature analysis function to analyze the CIV line
    civ_result = scana.analyze_emission_feature(
        fit, 'CIV', ['CIV_A_', 'CIV_B_'], 1549.06, cont_model_names=['PL_'],
        redshift=fit.redshift, emfeat_meas=None)

    cont_result.update(civ_result)

    print('\n')
    print('--- Automatic continuum and CIV line analysis ---')
    for key in cont_result.keys():
        print('{} = {:.2e}'.format(key, cont_result[key]))


analyze_specfit_example_civ_line()

analyze_specfit_example()


