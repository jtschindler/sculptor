Installation
============

1. Clone the github repository
##############################

This document describes how to install Sculptor and its dependencies. For now the project has not been published on PyPi, yet. Therefore, the first step is to clone the Sculptor repository from `github <https://github.com/jtschindler/sculptor>`_.

To do this simply clone the repository to your folder of choice.

.. code-block::

  git clone https://github.com/jtschindler/sculptor.git

The current version of Sculptor is designed to work with python >= 3.9.

2. Installing Sculptor and its requirements
###########################################

Navigate to the main folder of sculptor. It should contain the *setup.py* file as well as *requirements.txt*, *conda_requirements.yml*, and *environment.yml*.

2.1 Installing Sculptor via conda (Recommended)
***********************************************

While some may prefer a pure pip installation, I have run into issues with installing PyQt5 and pytables via pip on new Mac OSX M1/M2 machines. Therefore, I recommend to use the provided environment.yml file instead.

It automatically creates the  *sculptor* environment installing all necessary dependencies with the following command:

.. code-block::

  conda env create --file environment.yml

The environment is crearted using python 3.10. If a different python version is needed, please modify your version of the environment.yml file. Following the creating of the environment activate it via

.. code-block::

  conda activate sculptor

and then install Sculptor with

.. code-block::

  pip install -e .

This has been tested on a Mac OSX with a M2 chip, recently.

2.2 Installing Sculptor (using setup.py via pip)
************************************************

One can attempt a pure pip installation. Due to issues with pytables and PyQt5 installations via pip on Mac OSX M1/M2 machines, one needs to first install these packages independently:

.. code-block::

  pip install PyQt5
  pip install tables

Then you need to navigate into the main package folder with the setup.py file and execute:

.. code-block::

  pip install -e .

This will install all of the remaining dependencies and the Sculptor package itself.

2.3 Installing Sculptor (using requirements.txt via pip)
********************************************************

In the sculptor github repository you will find a 'requirements.txt', which allows you to install the necessary requirements using pip from the main sculptor directory:

.. code-block::

  pip install -r requirements.txt

If you are managing your python installation with Anaconda, this can work as well as long as you have pip installed in your Anaconda working environment. However, it may lead to issues if a pip version and an anaconda version of the same package (e.g., astropy) is installed.

In the same folder you then execute:

.. code-block::

  pip install -e .

This will install the Sculptor package.


3. Open up the sculptor GUI
###########################

To test whether the installation was successful, open your terminal and simply type

.. code-block::

  run_sculptor

If this opens up the Sculptor GUI, the installation was a success!

To test Sculptor further one can also load the example spectrum via

.. code-block::

  run_sculptor --ex=True