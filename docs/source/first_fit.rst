===========================================
Spectral fitting with Sculptor - An example
===========================================

In this example we will fit the SDSS spectrum of quasar J030341.04-002321.8 at redshift z=3.227 step by step. The example is aimed at first time user to provide insight into the Sculptor workflow and is designed to present a starting point.

We will begin by starting up sculptor with the example spectrum already imported:

.. code-block::

  run_sculptor --example=True

The GUI will start with the SpecFit tab open displaying the quasar spectrum.

.. image:: ../images/example_specfit_0.png
  :width: 800
  :alt: SpecFitTab_example

1-The quasar continuum model
############################

In this example we will be working with the Sculptor basic models and the models defined in the Sculptor extension *my_extension.py*, which were specifically included for this example.


2-Modeling the SiIV emission line
#################################

3-Modeling the CIV emission line
################################
