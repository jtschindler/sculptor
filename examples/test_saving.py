
from sculptor import component as sccomp
from sculptor import parameter as scpar
from sculptor import analysis as ana
from sculptor import prior as scpri

from IPython import embed

def test_component_save():

    # Define the SiIV model
    comp_siiv_a = sccomp.FitComponent('SiIV_a', ana.line_model_gaussian)
    pars_siiv_a = comp_siiv_a.create_params()

    pars_siiv_a['SiIV_a_flux'].value = 0.0006
    pars_siiv_a['SiIV_a_flux'].prior = scpri.UniformPrior('UniPrior', 0, 1)
    pars_siiv_a['SiIV_a_cen'].value = 1399.8
    pars_siiv_a['SiIV_a_cen'].vary = False
    pars_siiv_a['SiIV_a_redsh'].prior = scpri.UniformPrior('UniPrior', 3.17, 3.27)
    pars_siiv_a['SiIV_a_fwhm_km_s'].value = 1200
    pars_siiv_a['SiIV_a_fwhm_km_s'].prior = scpri.UniformPrior('UniPrior', 300, 6000)

    pars_siiv_a['SiIV_a_flux'].save('test_folder')

    embed()

def test_component_load():

    param = scpar.FitParameter()

if __name__ == "__main__":

    test_component_save()