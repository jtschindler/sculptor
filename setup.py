#!/usr/bin/env python

from distutils.core import setup

try:  # Python 3.x
    from distutils.command.build_py import build_py_2to3 as build_py
except ImportError:  # Python 2.x
    from distutils.command.build_py import build_py

setup(name='sculptor',
      version='0.1.0',
      description='Interactive Modelling of astronomic electromagnetic spectra',
      author='Jan-Torge Schindler',
      author_email='jtschi@posteo.net',
      license='GPL',
      url='http://github.com/jtschindler/sculptor',
      packages=['sculptor', 'sculptor_extensions'],
      provides=['sculptor'],
      package_dir={'sculptor': 'sculptor', 'sculptor_extensions':
          'sculptor_extensions'},
      package_data={'sculptor': ['data/iron_templates/*.*',
                                 'data/example_spectra/*.*',
                                 'data/passbands/*.*'],
                    'sculptor_extensions': ['*.*']},
      requires=['numpy', 'matplotlib', 'scipy', 'astropy', 'pandas', 'lmfit',
                'PyQt5'],
      scripts=['sculptor/scripts/run_sculptor'],
      keywords=['Scientific/Engineering'])
