#!/usr/bin/env python

from distutils.core import setup
import pathlib

here = pathlib.Path(__file__).parent.resolve()
# Get the long description from the README file
long_description = (here / 'README.md').read_text(encoding='utf-8')


try:  # Python 3.x
    from distutils.command.build_py import build_py_2to3 as build_py
except ImportError:  # Python 2.x
    from distutils.command.build_py import build_py

setup(name='sculptor',
      version='0.3b0',
      description='Interactive modeling of (electromagnetic) astronomical spectra',
      long_description=long_description,
      long_description_content_type='text/markdown',
      author='Jan-Torge Schindler',
      author_email='schindler@mpia.de',
      license='BSD-3',
      url='http://github.com/jtschindler/sculptor',
      packages=['sculptor', 'sculptor_extensions'],
      provides=['sculptor'],
      package_dir={'sculptor': 'sculptor', 'sculptor_extensions':
          'sculptor_extensions'},
      package_data={'sculptor': ['data/iron_templates/*.*',
                                 'data/example_spectra/*.*',
                                 'data/passbands/*.*'],
                    'sculptor_extensions': ['*.*']},
      # requires='astropy>=4.0.2, matplotlib>=3.3.1, numpy>=1.19.1, pandas>=1.1.3, PyQt5>=5.9.2, scipy>=1.5.0, corner>=2.1.0, lmfit>=1.0.1, tables>=3.6.1, emcee>=3.0.0, PyYAML>=5.4.1, extinction>=0.4.5, tqdm>=4.6.0, spectres',
      scripts=['sculptor/scripts/run_sculptor'],
      keywords=['Sculptor', 'astronomy', 'spectroscopy', 'modeling', 'fitting'],
      classifiers = ['Development Status :: 4 - Beta',
                   'Intended Audience :: Science/Research',
                   'License :: OSI Approved :: BSD License',
                   'Operating System :: OS Independent',
                   'Programming Language :: Python :: 3.6',
                   'Programming Language :: Python :: 3.7',
                   'Programming Language :: Python :: 3.8',
                   'Programming Language :: Python :: 3.9',
                   'Topic :: Scientific/Engineering :: Astronomy',
                   'Topic :: Software Development :: User Interfaces',
                   'Topic :: Software Development :: Libraries :: Python Modules'
                   ]
    )