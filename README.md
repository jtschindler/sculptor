# Sculptor

## Interactive modeling of astronomic, electro-magnetic spectra

### (Version 0.1 BETA)
##### Disclaimer: Please use at your own risk. I greatly appreciate bug reports and any feedback to improve this tool further.  
 
---

The Sculptor python package provides the functionality to easily model astronomic 
spectra using the LMFIT fitting environment. A graphical user interface allows interactive 
modeling with a high degree of complexity. 

The main modules SpecFit and SpecModel also allow to build fitting pipelines without user
interference in an easy and intuitive way. 

### Installation & Requirements

#### Requirements: 
'numpy', 'matplotlib', 'scipy', 'astropy', 'pandas', 'lmfit', 'PyQt5'

#### Installation:

Download the git repository. Change to the downloaded sculptor main directory and execute

python setup.py install

To test if the installation was successful open a command line and type 'sculptor'. This command should open the main GUI.

### Main modules (for user interaction)

* SpecFit
* SpecModel
* SpecFitGui
* SpecOneD

### Masks and Models

The Sculptor GUI uses the masks and models defined in the sub-package "models_and_masks". 
 This sub-package currently holds the following modules:
 
 *  <u>basic.py</u> : Basic models, such as a constant, a power law, a Gaussian or a Lorentzian function.
 
 *  <u>quasar.py</u> : A large range of models for fitting quasar spectra.
 
### Introduction and Tutorials

Coming soon!