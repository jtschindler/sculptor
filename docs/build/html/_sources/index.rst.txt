Welcome to Sculptor's documentation!
====================================

**Version:** |version|

**Sculptor** is a high level API and Graphical User Interface around `LMFIT <https://lmfit.github.io/lmfit-py/>`_ tailored specifically for the analysis of astronomical spectra. This package is designed to facilitate reproducible scientific results and easy to inspect model fits in an open source framework. For this purpose the *Sculptor* package introduces four core python modules and a Graphical User Interface for interactive control:

1. **SpecOneD**:
The *SpecOneD* module introduces the SpecOneD class, which is designed to store and manipulate 1D astronomical spectral data.

2. **SpecFit**:
The core module of *Sculptor* introducing the *SpecFit* class, which holds complex models to fit to the 1D astronomical spectrum.

3. **SpecModel**:
A helper class, which holds one complex spectral model, which can consist of multiple pre-defined or user-defined model functions to be fit to the 1D spectrum.

4. **SpecAnalysis**:
A module focused on the analysis of continuum models of models of emission/absorption features. It interfaces with the SpecFit class and streamlines the process of analyzing the fitted spectral models.

.. image:: ../images/example_fit.png
  :width: 800
  :alt: Full fit of the example spectrum

At the heart of the *Sculptor* package is the Graphical User Interface, which offers interactive control to set up and combine multiple spectral models to fully fit the astronomical spectrum of choice. This includes masking of spectral features, defining fit regions, and setting of fit parameter boundaries. The framework allows to add interdependent fit parameters (e.g., to couple the FWHM of multiple emission/absorption lines).

If you are interested in being involved with this project, please contact Jan-Torge Schindler via `github <https://github.com/jtschindler/sculptor>`_.

**Disclaimer:**
Version 1.0.0 is the first stable release version of Sculptor. Be advised that all future 1.x.x versions will adhere to the same API. However API changes might occur between major releases.

.. toctree::
  :maxdepth: 1
  :caption: Getting Started:

  installation
  getting_started
  first_fit

.. toctree::
  :maxdepth: 1
  :caption: Tutorials:

  speconed_demonstration.nblink
  spectrum_preparation.nblink
  scripting_sculptor_1.nblink
  scripting_sculptor_2.nblink
  scripting_sculptor_3.nblink


.. toctree::
  :maxdepth: 1
  :caption: Module documentation:

  speconed
  specmodel
  specfit
  specanalysis
  masksmodels

.. toctree::
  :maxdepth: 1
  :caption: Extensions:

  qso_extension
  my_extension

.. toctree::
  :caption: License:

  license



.. Indices and tables
.. ==================

.. * :ref:`genindex`
.. * :ref:`modindex`
.. * :ref:`search`
