# Sculptor

Interactive modeling of astronomic, electro-magnetic spectra

##### (Version 0.1.0)

Sculptor is a high level API and Graphical User Interface around [LMFIT] tailored specifically for the analysis of astronomical spectra.

At the heart of this package are three python 3 modules that manipulate 1D astronomical spectra (speconed), help to construct spectral models using pre-defined or user-defined functions to fit a spectrum utilizing LMFIT (specmodel), and a higher level class that combines multiple spectral models for a complex fit of the astronomical spectrum (specfit).


![Sculptor example fit][logo]

[logo]: https://github.com/jtschindler/sculptor/blob/master/docs/images/example_fit.png "A Sculptor example fit of a quasar spectrum."



The Graphical User Interface offers interactive control to set up and combine multiple spectral models to fully fit the astronomical spectrum of choice. This includes masking of spectral features, defining fit regions, setting of fit parameter boundaries. The framework allows to add interdependent fit parameters (e.g., to couple the FWHM of multiple emission/absorption) lines.

If you are interested in being involved with this project, please contact Jan-Torge Schindler via github.

**Disclaimer:**
This project is currently undergoing rapid development. Be advised that fits done with the current version may not work with a future version of the software. The first stable, well-tested release will be version 1.0.0 anticipated for summer 2021.


## Installing Sculptor and further documentation

#### The documentation, including the installation instructions are hosted at [Sculptor's ReadTheDocs].


[LMFIT]: https://lmfit.github.io/lmfit-py/
[Sculptor's ReadTheDocs]: https://sculptor.readthedocs.io/en/latest/
