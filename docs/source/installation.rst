Installation
============

1. Clone the github repository
##############################

This document describes how to install Sculptor and its dependencies. For now the project has not been published on PyPi, yet. Therefore, the first step is to clone the Sculptor repository from `github <https://github.com/jtschindler/sculptor>`_.

To do this simply clone the repository to your folder of choice.

.. code-block::

  git clone https://github.com/jtschindler/sculptor.git


2. Install requirements
#######################

Navigate to the main folder of sculptor. It should contain the *setup.py* file as well as *requirements.txt*, *conda_requirements.yml*, and *environment.yml*.

2.1 Installing requirements with a new conda environment (Recommended)
**********************************************************************

The sculptor github repository provides an environment.yml file. It automatically creates the  *sculptor-env* environment installing all necessary dependencies with the following command:

.. code-block::

  conda env create --file environment.yml

2.2 Installing requirements in an existing conda environment
************************************************************

There are basically two ways to make sure all requirements are installed, when working with an existing conda environment. The more convenient way makes use of the conda_requirements.yaml by updating the environment of choice [myenv]:

.. code-block::

  conda env update --name [myenv] --file conda_requirements.yml

Alternatively, one can open the conda_requirements.yml with a text editor and manually install all dependencies. The environment of choice should be activated first. Note, that *lmfit* and *corner* can only be installed via pip. Therefore one has to install pip in the active environment, if it is not installed already:

.. code-block::

  conda install pip


2.3 Installing requirements via pip
***********************************
In the sculptor github repository you will find a 'requirements.txt', which allows you to install the necessary requirements using pip from the main sculptor directory:

.. code-block::

  pip install -r requirements.txt

If you are managing your python installation with Anaconda, this can work as well as long as you have pip installed in your Anaconda working environment. However, it may lead to issues if a pip version and an anaconda version of the same package (e.g., astropy) is installed.


3. Install sculptor from the cloned repository
##############################################

With all requirements fulfilled, install Sculptor via

.. code-block::

  python setup.py install


4. Open up the sculptor GUI
###########################

To test whether the installation was successful, open your terminal and simply type

.. code-block::

  run_sculptor

If this opens up the Sculptor GUI, the installation was a success!
