# Sculptor

[![Documentation Status](https://readthedocs.org/projects/sculptor/badge/?version=latest)](https://sculptor.readthedocs.io/en/latest/?badge=latest)

Interactive modeling of (electromagnetic) astronomical spectra.

For installation instructions and documentation visit [Sculptor's ReadTheDocs]

##### (Version 0.4b0)

**Sculptor** is a high level API and Graphical User Interface around [LMFIT](https://lmfit.github.io/lmfit-py/) tailored specifically for the analysis of astronomical spectra. This package is designed to facilitate reproducible scientific results and easy to inspect model fits in an open source framework. For this purpose the *Sculptor* package introduces four core python modules and a Graphical User Interface for interactive control:

1. **SpecOneD**:
The *SpecOneD* module introduces the SpecOneD class, which is designed to store and manipulate 1D astronomical spectral data.

2. **SpecFit**:
The core module of *Sculptor* introducing the *SpecFit* class, which holds complex models to fit to the 1D astronomical spectrum.

3. **SpecModel**:
A helper class, which holds one complex spectral model, which can consist of multiple pre-defined or user-defined model functions to be fit to the 1D spectrum.

4. **SpecAnalysis**:
A module focused on the analysis of continuum models of models of emission/absorption features. It interfaces with the SpecFit class and streamlines the process of analyzing the fitted spectral models.


![Sculptor example fit][logo]

[logo]: https://github.com/jtschindler/sculptor/blob/main/docs/images/example_fit.png "A Sculptor example fit of a quasar spectrum."

At the heart of the *Sculptor* package is the Graphical User Interface, which offers interactive control to set up and combine multiple spectral models to fully fit the astronomical spectrum of choice. This includes masking of spectral features, defining fit regions, and setting of fit parameter boundaries. The framework allows to add interdependent fit parameters (e.g., to couple the FWHM of multiple emission/absorption lines).

If you are interested in being involved with this project, please contact Jan-Torge Schindler via [github](https://github.com/jtschindler/sculptor).

**Disclaimer:**
This project is currently undergoing rapid development. Be advised as the API has not been finalized, yet, fits done with the current version may not work with a future version of the software. The first stable, well-tested release will be version 1.0.0 anticipated for summer 2021.


[LMFIT]: https://lmfit.github.io/lmfit-py/
[Sculptor's ReadTheDocs]: https://sculptor.readthedocs.io/en/latest/
