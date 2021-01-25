.. Sculptor documentation master file, created by
   sphinx-quickstart on Fri Jan 15 10:47:50 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to Sculptor's documentation!
====================================

Sculptor is a high level API and Graphical User Interface around `LMFIT <https://lmfit.github.io/lmfit-py/>`_ tailored specifically for the analysis of astronomical spectra.

At the heart of this package are three python 3 modules that manipulate 1D astronomical spectra (speconed), help to construct spectral models using pre-defined or user-defined functions to fit a spectrum utilizing LMFIT (specmodel), and a higher level class that combines multiple spectral models for a complex fit of the astronomical spectrum (specfit).

.. image:: ../images/example_fit.png
  :width: 800
  :alt: Full fit of the example spectrum 

The Graphical User Interface offers interactive control to set up and combine multiple spectral models to fully fit the astronomical spectrum of choice. This includes masking of spectral features, defining fit regions, setting of fit parameter boundaries. The framework allows to add interdependent fit parameters (e.g., to couple the FWHM of multiple emission/absorption) lines.

**Disclaimer:**
This project is currently undergoing rapid development. Be advised that fits done with the current version may not work with a future version of the software. The first stable, well-test release will be version 1.0.0 anticipated for summer 2021.

.. toctree::
  :maxdepth: 1
  :caption: Contents:

  installation
  getting_started
  first_fit

.. toctree::
  :maxdepth: 1
  :caption: Modules:

  speconed
  specmodel
  specfit

.. toctree::
  :caption: License:

  license



.. Indices and tables
.. ==================

.. * :ref:`genindex`
.. * :ref:`modindex`
.. * :ref:`search`
