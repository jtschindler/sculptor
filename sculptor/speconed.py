#!/usr/bin/env python

"""
This module introduces the SpecOneD class, it's functions and the related
PassBand class.
The main purpose of the SpecOneD class and it's children classes is to provide
python functionality for the manipulation of 1D spectral data in astronomy.

"""


import numpy as np
import scipy as sp
import pandas as pd

import h5py
import extinction as ext
import pkg_resources
import spectres

from astropy.io import fits
from astropy import constants as const
from astropy import units as u
from astropy.convolution import convolve, Gaussian1DKernel, Box1DKernel

import matplotlib.pyplot as plt

black = (0, 0, 0)
orange = (230/255., 159/255., 0)
blue = (86/255., 180/255., 233/255.)
green = (0, 158/255., 115/255.)
yellow = (240/255., 228/255., 66/255.)
dblue = (0, 114/255., 178/255.)
vermillion = (213/255., 94/255., 0)
purple = (204/255., 121/255., 167/255.)

color_list = [vermillion, dblue, green, purple, yellow, orange, blue]

ln_AA = u.def_unit('ln(Angstroem)')


def gaussian(x, amp, cen, sigma, shift):
    """ 1-D Gaussian function"""
    central = cen + shift

    return (amp / (np.sqrt(2*np.pi) * sigma)) * np.exp(-(x-central)**2 / (2*sigma**2))


class SpecOneD(object):
    """The SpecOneD class provides a data structure for 1D astronomical
    spectra with extensive capabilities for spectrum analysis and manipulation.

    :param dispersion: A 1D array providing the dispersion axis of the \
    spectrum.
    :type dispersion: numpy.ndarray
    :param fluxden: A 1D array providing the flux density data of the \
    spectrum.
    :type fluxden: numpy.ndarray
    :param fluxden_err: A 1D array providing the 1-sigma flux density \
    error of the spectrum.
    :type fluxden_err: numpy.ndarray
    :param fluxden_ivar: A 1D array providing the inverse variance of the \
    flux density for the spectrum.
    :type fluxden_ivar: numpy.ndarray
    :param header: A pandas DataFrame containing additional information \
    on the spectrum.
    :type header: pandas.DataFrame
    :param dispersion_unit: The physical unit (including normalization \
    factors) of the dispersion axis of the spectrum.
    :type dispersion_unit: astropy.units.Unit or astropy.units.Quantity or \
    astropy.units.CompositeUnit or astropy.units.IrreducibleUnit
    :param fluxden_unit: The physical unit (including normalization \
    factors) of the flux density and associated properties (e.g. flux \
    density error) of the spectrum.
    :type fluxden_unit: astropy.units.Unit or astropy.units.Quantity or \
    astropy.units.CompositeUnit or astropy.units.IrreducibleUnit
    :param obj_model: Object model from the telluric correction routine \
    of PypeIt.
    :type obj_model: numpy.ndarray
    :param telluric: Telluric (atmospheric transmission) model from the \
    telluric correction routine of PypeIt.
    :type telluric: numpy.ndarray
    :param mask: A boolean 1D array specifying regions that allows to \
    mask region in the spectrum during spectral manipulation or for \
    display purposes.
    :type mask: numpy.ndarray

    :raise ValueError: Raises a ValueError if the supplied header is not a \
    pandas.DataFrame

    """

    def __init__(self, dispersion=None, fluxden=None, fluxden_err=None,
                 fluxden_ivar=None, header=None,
                 dispersion_unit=None, fluxden_unit=None, obj_model=None,
                 telluric=None, mask=None):
        """Initialize the SpecOneD class object.

        :param dispersion: A 1D array providing the dispersion axis of the \
        spectrum.
        :type dispersion: numpy.ndarray
        :param fluxden: A 1D array providing the flux density data of the \
        spectrum.
        :type fluxden: numpy.ndarray
        :param fluxden_err: A 1D array providing the 1-sigma flux density \
        error of the spectrum.
        :type fluxden_err: numpy.ndarray
        :param fluxden_ivar: A 1D array providing the inverse variance of the \
        flux density for the spectrum.
        :type fluxden_ivar: numpy.ndarray
        :param header: A pandas DataFrame containing additional information \
        on the spectrum.
        :type header: pandas.DataFrame
        :param dispersion_unit: The physical unit (including normalization \
        factors) of the dispersion axis of the spectrum.
        :type dispersion_unit: astropy.units.Unit or astropy.units.Quantity or \
        astropy.units.CompositeUnit or astropy.units.IrreducibleUnit
        :param fluxden_unit: The physical unit (including normalization \
        factors) of the flux density and associated properties (e.g. flux \
        density error) of the spectrum.
        :type fluxden_unit: astropy.units.Unit or astropy.units.Quantity or \
        astropy.units.CompositeUnit or astropy.units.IrreducibleUnit
        :param obj_model: Object model from the telluric correction routine \
        of PypeIt.
        :type obj_model: numpy.ndarray
        :param telluric: Telluric (atmospheric transmission) model from the \
        telluric correction routine of PypeIt.
        :type telluric: numpy.ndarray
        :param mask: A boolean 1D array specifying regions that allows to \
        mask region in the spectrum during spectral manipulation or for \
        display purposes.
        :type mask: numpy.ndarray

        :raise ValueError: Raises a ValueError if the supplied header is not a \
        pandas.DataFrame
        """

        self.fluxden = fluxden
        self.fluxden_err = fluxden_err
        self.dispersion = dispersion
        self.fluxden_ivar = fluxden_ivar

        if self.dispersion is not None and mask is None:
            self.mask = np.ones(self.dispersion.shape, dtype=bool)
        else:
            self.mask = mask

        if fluxden_err is None and fluxden_ivar is not None:
            self.get_fluxden_error_from_ivar()
        if fluxden_ivar is None and fluxden_err is not None:
            self.get_ivar_from_fluxden_error()

        if fluxden is not None \
            and (isinstance(dispersion_unit, u.Unit) or
                 isinstance(dispersion_unit, u.Quantity) or
                 isinstance(dispersion_unit, u.IrreducibleUnit) or
                 isinstance(dispersion_unit, u.CompositeUnit)) \
            and (isinstance(fluxden_unit, u.Unit) or
                 isinstance(fluxden_unit, u.Quantity) or
                 isinstance(fluxden_unit, u.IrreducibleUnit) or
                 isinstance(fluxden_unit, u.CompositeUnit)):
            self.dispersion_unit = dispersion_unit
            self.fluxden_unit = fluxden_unit
        elif fluxden is None:
            self.dispersion_unit = None
            self.fluxden_unit = None
        else:
            print('[WARNING] Flux density and dispersion units are '
                  'not specified or their types are not '
                  'supported. Any of the following astropy.units '
                  'are allowed: "Unit", "Quantity", "CompositeUnit", '
                  '"IrreducibleUnit". As a default the units will '
                  'be populated with erg/s/cm^2/AA and AA')

            self.dispersion_unit = 1.*u.AA
            self.fluxden_unit = 1.*u.erg/u.s/u.cm**2/u.AA

        if isinstance(header, pd.DataFrame):
            self.header = header
            self.fits_header = None
        elif header is None:
            self.header = pd.DataFrame(columns=['value'])
            self.fits_header = None
        elif isinstance(header, fits.Header):
            self.header = pd.DataFrame(list(header.items()),
                                       index=list(header.keys()),
                                       columns=['property', 'value'])
            self.header.drop(columns='property', inplace=True)
            self.fits_header = header
        else:
            raise ValueError('[ERROR] Header is not a pandas DataFrame or a '
                             'fits header.')

        self.obj_model = obj_model
        self.telluric = telluric

# ------------------------------------------------------------------------------
# READ AND WRITE
# ------------------------------------------------------------------------------

    def read_pypeit_fits(self, filename, exten=1):
        """Read in a pypeit fits spectrum as a SpecOneD object.

        :param filename: Filename of the fits file.
        :type filename: string
        :param exten: Extension of the pypeit fits file to read. This \
        defaults to exten=1.
        :type exten: int
        :return:

        :raises ValueError: Raises an error when the filename could not be \
        read in.
        """

        # Open the fits file
        try:
            hdu = fits.open(filename)
        except:
            raise ValueError("Filename not found", str(filename))

        # Add header information to SpecOneD
        header_df = pd.DataFrame(list(hdu[0].header.items()),
                                 index=list(hdu[0].header.keys()),
                                 columns=['property', 'value'])
        header_df.drop(columns='property', inplace=True)
        self.header = header_df

        # Retain fits header in SpecOneD
        self.fits_header = hdu[0].header

        # Check pypeit header keywords
        #
        # dispersion
        if 'OPT_WAVE' in hdu[exten].columns.names:
            self.dispersion = hdu[exten].data['OPT_WAVE']
        if 'wave' in hdu[exten].columns.names:
            self.dispersion = hdu[exten].data['wave']
        # flux density
        if 'OPT_FLAM' in hdu[exten].columns.names:
            self.fluxden = hdu[exten].data['OPT_FLAM']
        if 'flux' in hdu[exten].columns.names:
            self.fluxden = hdu[exten].data['flux']

        # mask
        if 'OPT_MASK' in hdu[exten].columns.names:
            self.mask = np.array(hdu[exten].data['OPT_MASK'], dtype=bool)
        if 'mask' in hdu[exten].columns.names:
            self.mask = np.array(hdu[exten].data['mask'], dtype=bool)

        # ivar
        if 'OPT_FLAM_IVAR' in hdu[exten].columns.names:
            self.fluxden_ivar = hdu[exten].data['OPT_FLAM_IVAR']
        if 'ivar' in hdu[exten].columns.names:
            self.fluxden_ivar = hdu[exten].data['ivar']
            if 'sigma' not in hdu[exten].columns.names:
                # No flux density 1 sigma error stored in this format
                # Calculate the 1 sigma error.
                self.get_fluxden_error_from_ivar()
        # 1 sigma flux density error
        if 'OPT_FLAM_SIG' in hdu[exten].columns.names:
            self.fluxden_err = hdu[exten].data['OPT_FLAM_SIG']



        # # Mask all pixels where the fluxden error is 0
        # new_mask = np.ones_like(self.mask, dtype=bool)
        # new_mask[self.fluxden_err == 0] = 0
        # self.mask = new_mask

        self.dispersion_unit = 1. * u.AA
        self.fluxden_unit = 1e-17*u.erg/u.s/u.cm**2/u.AA

        if 'TELLURIC' in hdu[exten].columns.names:
            self.telluric = hdu[exten].data['TELLURIC']
        if 'OBJ_MODEL' in hdu[exten].columns.names:
            self.obj_model = hdu[exten].data['OBJ_MODEL']

    def read_from_fits(self, filename):
        """Read in an iraf fits spectrum as a SpecOneD object.

        :param filename: Filename of the fits file.
        :type filename: str
        :return:

        :raises ValueError: Raises an error when the filename could not be \
        read in.
        """

        # Open the fits file
        try:
            hdu = fits.open(filename)
        except:
            raise ValueError("Filename not found", str(filename))

        # Read in header information
        crval = hdu[0].header['CRVAL1']
        try:
            cd = hdu[0].header['CD1_1']
        except:
            print("CD1_1 keyword not found, using CDELT1")
            cd = hdu[0].header['CDELT1']

        crpix = hdu[0].header['CRPIX1']
        naxis = hdu[0].header['NAXIS1']

        # Read in object fluxden
        if np.ndim(hdu[0].data) == 3:
            try:
                self.fluxden = np.array(hdu[0].data[0, 0, :])
                self.fluxden_err = np.array(hdu[0].data[3, 0, :])
            except:
                self.fluxden = np.array(hdu[0].data[0, 0, :])
                self.fluxden_err = np.array(hdu[0].data[1, 0, :])
        else:
            self.fluxden = np.array(hdu[0].data[:])
            self.fluxden_err = None

        # Calculate dispersion axis from header information
        crval = crval - crpix
        self.dispersion = crval + np.arange(naxis) * cd

        self.mask = np.ones(self.dispersion.shape, dtype=bool)

        # Add header information to SpecOneD
        header_df = pd.DataFrame(list(hdu[0].header.items()),
                                 index=list(hdu[0].header.keys()),
                                 columns=['property','value'])
        header_df.drop(columns='property', inplace=True)
        self.header = header_df
        self.fits_header = hdu[0].header

        if self.fits_header['BUNIT'] == 'erg/cm2/s/A':
            self.dispersion_unit = 1. * u.AA
            self.fluxden_unit = 1. * u.erg / u.s / u.cm ** 2 / u.AA
        else:
            print('[WARNING] Spectral units were not found and need to be '
                  'manually set. For example for a dispersion in Angstroem set '
                  '"self.dispersion_unit = u.AA" and for a corresponding cgs '
                  'flux density set "self.fluxden_unit = u.erg / u.s / u.cm ** 2 / u.AA"')


    def read_sdss_fits(self, filename):
        """Read in an SDSS/BOSS fits spectrum as a SpecOneD object.

        :param filename: Filename of the fits file.
        :type filename: str
        :return:

        :raises ValueError: Raises an error when the filename could not be \
        read in.
        """

        # Open the fits file
        try:
            hdu = fits.open(filename)
        except:
            raise ValueError("Filename not found", str(filename))

        self.fluxden = np.array(hdu[1].data['flux'], dtype=np.float64)
        self.dispersion = 10**np.array(hdu[1].data['loglam'], dtype=np.float64)
        self.ivar = np.array(hdu[1].data['ivar'], dtype=np.float64)
        self.fluxden_err = 1/np.sqrt(self.ivar)

        self.mask = np.ones(self.dispersion.shape, dtype=bool)

        # Add header information to SpecOneD
        header_df = pd.DataFrame(list(hdu[0].header.items()),
                                 index=list(hdu[0].header.keys()),
                                 columns=['property', 'value'])
        header_df.drop(columns='property', inplace=True)
        self.header = header_df
        self.fits_header = hdu[0].header

        self.dispersion_unit = 1. * u.AA
        self.fluxden_unit = 1e-17 * u.erg / u.s / u.cm ** 2 / u.AA

    def save_to_hdf(self, filename):
        """
        Save a SpecOneD object to a hdf5 file.

        SpecOneD hdf5 files have three extensions:
        - data: holding the array spectral information like dispersion,
        flux density, flux density error, flux density inverse variance, or mask
        - spec_meta: holding information on the spectral meta data, currently
        this includes the units of the dispersion and flux density axis.
        - header: If a header exists, it will be saved here.

        :param filename: Filename to save the current SpecOneD object to.
        :type filename: str
        :return:
        """

        df = pd.DataFrame(np.array([self.dispersion, self.fluxden]).T,
                          columns=['dispersion', 'fluxden'])

        if self.fluxden_err is not None:
            df['fluxden_err'] = self.fluxden_err
        if self.fluxden_ivar is not None:
            df['fluxden_ivar'] = self.fluxden_err
        if self.mask is not None:
            df['mask'] = self.mask
        if self.obj_model is not None:
            df['obj_model'] = self.obj_model

        df.to_hdf(filename, 'data')

        # Save the spectrum meta data
        dispersion_unit_str = self.dispersion_unit.to_string()
        fluxden_unit_str = self.fluxden_unit.to_string()
        df = pd.DataFrame({'dispersion_unit': dispersion_unit_str,
                           'fluxden_unit': fluxden_unit_str},
                          index=['value'])

        df.to_hdf(filename, 'spec_meta')

        # If a header exists, save it
        if self.header is not None:
            self.header.to_hdf(filename, 'header')

    def read_from_hdf(self, filename):
        """
        Read in a SpecOneD object from a hdf5 file.

        :param filename: Filename from which to read the new SpecOneD object in.
        :type filename: str
        :return:
        """

        h5file = h5py.File(filename, 'r')
        if 'data' in h5file.keys():

            df = pd.read_hdf(filename, 'data')

            self.dispersion = df['dispersion'].values
            # Backward compatibility with previous SpecOneD
            if 'fluxden' in df.columns:
                self.fluxden = df['fluxden'].values
            else:
                self.fluxden = df['flux'].values
            if 'fluxden_err' in df.columns:
                self.fluxden_err = df['fluxden_err'].values
            else:
                self.fluxden_err = df['flux_err'].values

            if 'obj_model' in df.columns:
                self.obj_model = df['obj_model'].values
            if 'mask' in df.columns:
                self.mask = df['mask'].values
            else:
                self.reset_mask()
        else:
            raise ValueError('[ERROR] No spectral data found in hdf5 file.')

        if 'spec_meta' in h5file.keys():
            df = pd.read_hdf(filename, 'spec_meta')

            dispersion_unit = u.Quantity(df.loc['value', 'dispersion_unit'])
                                            # format='cds')
            fluxden_unit = u.Quantity(df.loc['value', 'fluxden_unit'])
                                         # format='cds')
            self.dispersion_unit = dispersion_unit
            self.fluxden_unit = fluxden_unit
        else:
            raise ValueError('[ERROR] No spectral meta data found in hdf5 '
                             'file.')

        if 'header' in h5file.keys():
            df = pd.read_hdf(filename, 'header')
            self.header = df
        else:
            self.header = None

    def save_to_csv(self, filename, outputformat='linetools'):
        """Save SpecOneD object to a csv file.

        Output formats:
        - default: All array-like SpecOneD data will be saved to the csv
        file. Standard columns include dispersion, flux density, and flux
        density error. Additional columns can be telluric or object model
        from spectra reduced with pypeit.
        - "linetools": In the linetool format dispersion, flux density and
        flux density error (if exists) are saved in three columns with names
        'wave', 'flux', and 'error'.

        WARNING: At present the SpecOneD data will be saved independent of
        its physical units. They will not be automatically converted to a
        common format. User discretion is advised as unit information might
        get lost if saving to csv.

        :param filename: Filename to save the SpecOneD object to.
        :type filename: str
        :param outputformat: Format of the csv file. Possible formats include \
        "linetools". All other inputs will save it to the default format.
        :type outputformat: str
        :return:
        """

        data = [self.dispersion, self.fluxden]

        if self.fluxden_err is not None:
            data.append(self.fluxden_err)
        if self.telluric is not None:
            data.append(self.telluric)
        if self.obj_model is not None:
            data.append(self.obj_model)

        if outputformat == 'linetools':

            column_names = ['wave', 'flux']
            if self.fluxden_err is not None:
                column_names.append('error')

        else:
            column_names = ['wavelength', 'fluxden']
            if self.fluxden_err is not None:
                column_names.append('flux_error')

        if self.telluric is not None:
            column_names.append('telluric')
        if self.obj_model is not None:
            column_names.append('obj_model')

        df = pd.DataFrame(np.array(data).T, columns=column_names)

        df.to_csv(filename, index=False)

# ------------------------------------------------------------------------------
# BASIC CLASS FUNCTIONALITY
# ------------------------------------------------------------------------------

    def copy(self):
        """Copy the current SpecOneD object to a new instance and return it.

        :return:
        :rtype: SpecOneD
        """

        dispersion = self.dispersion.copy()
        fluxden = self.fluxden.copy()
        if self.fluxden_err is not None:
            flux_err = self.fluxden_err.copy()
        else:
            flux_err = self.fluxden_err
        if self.fluxden_ivar is not None:
            fluxden_ivar = self.fluxden_ivar.copy()
        else:
            fluxden_ivar = self.fluxden_ivar
        if self.mask is not None:
            mask = self.mask.copy()
        else:
            mask = self.mask
        if self.obj_model is not None:
            obj_model = self.obj_model.copy()
        else:
            obj_model = None
        if self.telluric is not None:
            telluric = self.telluric.copy()
        else:
            telluric = self.telluric

        if self.header is not None:
            header = self.header.copy(deep=True)
        else:
            header = self.header

        if isinstance(self.dispersion_unit, u.Unit) or \
                isinstance(self.dispersion_unit, u.CompositeUnit) or \
                isinstance(self.dispersion_unit, u.IrreducibleUnit):
            dispersion_unit = u.Quantity(1.,
                                         self.dispersion_unit)
        else:
            dispersion_unit = u.Quantity(self.dispersion_unit.value,
                                         self.dispersion_unit.unit)

        if isinstance(self.dispersion_unit, u.Unit) or \
                isinstance(self.dispersion_unit, u.CompositeUnit) or \
                isinstance(self.dispersion_unit, u.IrreducibleUnit):
            fluxden_unit = u.Quantity(1.,
                                      self.fluxden_unit)
        else:
            fluxden_unit = u.Quantity(self.fluxden_unit.value,
                                  self.fluxden_unit.unit)

        return SpecOneD(dispersion=dispersion,
                        fluxden=fluxden,
                        fluxden_err=flux_err,
                        fluxden_ivar=fluxden_ivar,
                        header=header,
                        mask=mask,
                        dispersion_unit=dispersion_unit,
                        fluxden_unit=fluxden_unit,
                        obj_model=obj_model,
                        telluric=telluric
                        )

    def reset_mask(self):
        """Reset the spectrum mask.

        :return:
        """
        self.mask = np.ones(self.dispersion.shape, dtype=bool)

    def mask_by_snr(self, signal_to_noise_ratio, inplace=False):
        """Mask all regions with a signal to noise below the specified limit

        :param signal_to_noise_ratio: All regions of the spectrum with a \
        value below this limit will be masked.
        :type signal_to_noise_ratio: float
        :param inplace: Boolean to indicate whether the active SpecOneD \
        object will be modified or a new SpecOneD object will be created and \
        returned.
        :type inplace: bool
        :return: SpecOneD
        """

        mask_index = self.fluxden / self.fluxden_err > signal_to_noise_ratio

        new_mask = np.zeros(self.dispersion.shape, dtype=bool)

        new_mask[mask_index] = 1

        new_mask = new_mask * np.array(self.mask, dtype=bool)

        if inplace:
            self.mask = new_mask

        else:
            spec = self.copy()
            spec.mask = new_mask
            return spec

    def mask_between(self, limits, inplace=False):
        """Mask spectrum between specified dispersion limits.

        :param limits: A list of two floats indicating the lower and upper \
        dispersion limit to mask between.
        :type limits: [float, float]
        :param inplace: Boolean to indicate whether the active SpecOneD \
        object will be modified or a new SpecOneD object will be created and \
        returned.
        :type inplace: bool
        :return:
        """

        lo_index = np.argmin(np.abs(self.dispersion - limits[0]))
        up_index = np.argmin(np.abs(self.dispersion - limits[1]))

        self.mask[lo_index:up_index] = 0

        if inplace:
            self.mask[lo_index:up_index] = 0
        else:
            spec = self.copy()
            spec.mask[lo_index:up_index] = 0

            return spec

    def get_ivar_from_fluxden_error(self):
        """Calculate inverse variance of the flux density from the flux
        density 1-sigma error.

        :return:
        """

        if self.fluxden_err is None:
            self.fluxden_ivar = None
        else:
            ivar = np.zeros_like(self.fluxden_err)
            valid = self.fluxden_err > 0
            ivar[valid] = 1. / (self.fluxden_err[valid]) ** 2

            self.fluxden_ivar = ivar

    def get_fluxden_error_from_ivar(self):
        """Calculate the flux density 1-sigma error from the inverse variance
        of the flux density/

        :return:
        """

        if self.fluxden_ivar is None:
            self.fluxden_err = None
        else:
            sigma = np.zeros_like(self.fluxden_ivar)
            valid = self.fluxden_ivar > 0
            sigma[valid] = 1./np.sqrt(self.fluxden_ivar[valid])

            self.fluxden_err = sigma

    def average_fluxden(self, dispersion_range=None):
        """Calculate the average flux density over the full spectrum or the
        specified dispersion range

        :param dispersion_range: Dispersion range over which to average the \
        flux density.
        :type dispersion_range: [float, float]
        :return: Average flux density of the full spectrum or specified \
        dispersion range
        :rtype: float
        """

        if dispersion_range is None:
            return np.average(self.fluxden)
        else:
            return np.average(self.trim_dispersion(dispersion_range).fluxden)

    def peak_fluxden(self):
        """Return the maximum flux density value in the spectrum.

        :return: Maximum flux density value
        :rtype: float
        """

        return np.max(self.fluxden)

    def peak_dispersion(self):
        """Return the dispersion of the maximum flux density value in the
        spectrum.

        :return: Dispersion value of maximum flux density
        :rtype: float
        """

        return self.dispersion[np.argmax(self.fluxden)]

# ------------------------------------------------------------------------------
# SPECTRAL UNIT CONVERSION
# ------------------------------------------------------------------------------

    def check_units(self, spectrum):
        """Raise a ValueError if current and input spectrum have different
        dispersion of flux density units.


        :param spectrum:
        :type spectrum: SpecOneD
        :return:
        """

        if isinstance(self.fluxden_unit, u.Quantity):
            self_fluxden_unit = self.fluxden_unit.unit
        else:
            self_fluxden_unit = self.fluxden_unit
        if isinstance(spectrum.fluxden_unit, u.Quantity):
            spectrum_fluxden_unit = spectrum.fluxden_unit.unit
        else:
            spectrum_fluxden_unit = spectrum.fluxden_unit

        if spectrum_fluxden_unit != u.dimensionless_unscaled and \
           self_fluxden_unit != u.dimensionless_unscaled:
            if self.fluxden_unit != spectrum.fluxden_unit or \
               self.dispersion_unit != spectrum.dispersion_unit:
                raise ValueError('[ERROR] Current and supplied spectrum have '
                                 'different dispersion and/or flux density units.')
        else:
            if self.dispersion_unit != spectrum.dispersion_unit:
                raise ValueError('[ERROR] Current and supplied spectrum have '
                                 'different dispersion units.')

    def _reorder_dispersion(self, verbosity=0):

        if self.dispersion[0] > self.dispersion[-1]:
            if verbosity > 0:
                print('[INFO] Reordering dispersion values in ascending order.')

            for attr in ['dispersion', 'fluxden', 'mask', 'fluxden_err',
                         'fluxden_ivar', 'telluric', 'obj_model']:
                if self.__dict__[attr] is not None:
                    self.__dict__[attr] = self.__dict__[attr][::-1]

    def convert_spectral_units(self, new_dispersion_unit, new_fluxden_unit,
                               verbosity=0):
        """Convert the spectrum to new physical dispersion and flux density
        units.

        The function converts the flux density, the dispersion, the flux
        density error and the inverse variance.
        Object model and telluric if they exist will not be converted.

        :param new_dispersion_unit: New dispersion unit (or quantity)
        :type new_dispersion_unit: astropy.units.Unit or \
        astropy.units.Quantity or astropy.units.CompositeUnit or \
        astropy.units.IrreducibleUnit
        :param new_fluxden_unit: New flux density unit (or quantity)
        :type new_fluxden_unit: astropy.units.Unit or \
        astropy.units.Quantity or astropy.units.CompositeUnit or \
        astropy.units.IrreducibleUnit
        :return:
        """

        # Setup physical spectral properties
        fluxden = self.fluxden * self.fluxden_unit
        dispersion = self.dispersion * self.dispersion_unit

        # Convert flux density
        new_fluxden = fluxden.to(new_fluxden_unit,
                                 equivalencies=u.spectral_density(dispersion))

        # Convert flux density 1-sigma errors
        if self.fluxden_err is not None:
            fluxden_err = self.fluxden_err * self.fluxden_unit
            new_fluxden_err = fluxden_err.to(new_fluxden_unit,
                                             equivalencies=u.spectral_density(
                                                           dispersion))
            self.fluxden_err = new_fluxden_err.value
            self.get_ivar_from_fluxden_error()

        # Convert object model if present
        if self.obj_model is not None:
            obj_model = self.obj_model * self.fluxden_unit
            # Convert object model
            new_obj_model = obj_model.to(new_fluxden_unit,
                                             equivalencies=u.spectral_density(
                                                           dispersion))
            self.obj_model = new_obj_model.value

        # Convert dispersion axis
        new_dispersion = dispersion.to(new_dispersion_unit,
                                       equivalencies=u.spectral())

        self.fluxden = new_fluxden.value
        self.fluxden_unit = new_fluxden.unit
        self.dispersion = new_dispersion.value
        self.dispersion_unit = new_dispersion.unit

        self._reorder_dispersion(verbosity=verbosity)

    def to_fluxden_per_unit_frequency_cgs(self):
        """Convert SpecOneD spectrum to flux density per unit frequency (Hz) in
        cgs units.

        :return:
        """

        new_fluxden_unit = 1 * u.erg / u.s / u.cm ** 2 / u.Hz
        new_dispersion_unit = 1 * u.Hz

        self.convert_spectral_units(new_dispersion_unit,
                                    new_fluxden_unit)

    def to_fluxden_per_unit_frequency_jy(self):
        """Convert SpecOneD spectrum to flux density per unit frequency (Hz) in
        Jy.

        :return:
        """

        new_fluxden_unit = 1 * u.Jy
        new_dispersion_unit = 1 * u.Hz

        self.convert_spectral_units(new_dispersion_unit,
                                    new_fluxden_unit)

    def to_fluxden_per_unit_frequency_si(self):
        """Convert SpecOneD spectrum to flux density per unit frequency (Hz) in
        SI units.

        :return:
        """

        new_fluxden_unit = 1 * u.Watt / u.m ** 2 / u.Hz
        new_dispersion_unit = 1 * u.Hz

        self.convert_spectral_units(new_dispersion_unit,
                                    new_fluxden_unit)

    def to_fluxden_per_unit_wavelength_cgs(self):
        """Convert SpecOneD spectrum to flux density per unit wavelength (
        Angstroem) in cgs units.

        :return:
        """

        new_fluxden_unit = 1 * u.erg / u.s / u.cm ** 2 / u.AA
        new_dispersion_unit = 1 * u.AA

        self.convert_spectral_units(new_dispersion_unit,
                                    new_fluxden_unit)

    def _to_log_wavelength(self):
        """ Convert the spectrum to logarithmic (natural logarithm) wavelength
        units.

        A helper function for the convolve_log_wavelength function.

        This method converts the flux density to flux and the dispersion to
        logarithmic wavelength (ln(AA)).

        :return:
        """

        # Convert the specrum to flux density per unit wavelength (Angstroem)
        self.to_fluxden_per_unit_wavelength_cgs()

        # Convert the flux density
        self.fluxden = self.fluxden * self.dispersion
        self.fluxden_unit = self.fluxden_unit * u.AA

        # Convert the dispersion axis
        self.dispersion = np.log(self.dispersion)
        self.dispersion_unit = ln_AA

    def _to_lin_wavelength(self):
        """ Convert the spectrum from logarithmic wavelength units back to
        linear wavelength units.

        A helper function for the convolve_log_wavelength function.

        This method converts the flux back to flux density and the dispersion
        from ln(AA) back to AA.

        :return:
        """

        # Convert the dispersion axis
        self.dispersion = np.exp(self.dispersion)
        self.dispersion_unit = 1 * u.AA

        # Convert the flux density
        self.fluxden = self.fluxden / self.dispersion
        self.fluxden_unit = self.fluxden_unit / (1 * u.AA)

    def normalize_fluxden_by_error(self, inplace=False):
        """Normalize the flux density, flux density error and object model
        numerical values by the median value of the flux density error array.

        The flux density unit will be scaled accordingly. Hence,
        this normalization does not affect the physical values of the flux
        density and only serves to normalize the values in the flux density
        array.

        This enables more efficient calculations on the flux density array by
        avoiding small numerical values.

        :param inplace: Boolean to indicate whether the active SpecOneD \
        object will be modified or a new SpecOneD object will be created and \
        returned.
        :type inplace: bool
        :return: SpecOneD
        """

        scale_factor = np.median(self.fluxden_err[self.mask])

        spec = self.normalize_fluxden_by_factor(scale_factor, inplace=inplace)

        return spec

    def normalize_fluxden_to_factor(self, factor, inplace=False):
        """Normalize the flux density, flux density error and object model
        numerical values to the specified unit factor.

        The flux density unit will be scaled accordingly. Hence,
        this normalization does not affect the physical values of the flux
        density and only serves to normalize the values in the flux density
        array.

        For example normalizing the flux density to a factor 1e-17 will
        assure that the flux density unit is 1e-17 times the original unit of
        the flux density.

        This enables more efficient calculations on the flux density array by
        avoiding small numerical values.

        :param factor: Scale factor by which the flux density, flux density \
        error and object model will be divided and the flux density unit will \
        be multiplied with.
        :type factor: float
        :param inplace: Boolean to indicate whether the active SpecOneD \
        object will be modified or a new SpecOneD object will be created and \
        returned.
        :type inplace: bool
        :return: SpecOneD
        """

        if isinstance(self.fluxden_unit, (u.Unit, u.IrreducibleUnit,
                                             u.CompositeUnit)):
            scale_factor = factor
        elif isinstance(self.fluxden_unit, u.Quantity):
            scale_factor = factor / self.fluxden_unit.value
        else:
            raise ValueError('[ERROR] Flux density unit type is not '
                             'supported. Supported types are astropy.units.Unit'
                             ' or astropy.units.Quantity or '
                             'astropy.units.CompositeUnit or '
                             'astropy.units.IrreducibleUnit')
        spec = self.normalize_fluxden_by_factor(scale_factor, inplace=inplace)

        return spec

    def normalize_fluxden_by_factor(self, factor, inplace=False):
        """Normalize the flux density, flux density error and object model
        numerical values by the specified numerical factor.

        The flux density unit will be scaled accordingly. Hence,
        this normalization does not affect the physical values of the flux
        density and only serves to normalize the values in the flux density
        array.

        This enables more efficient calculations on the flux density array by
        avoiding small numerical values.

        :param factor: Scale factor by which the flux density, flux density \
        error and object model will be divided and the flux density unit will \
        be multiplied with.
        :type factor: float
        :param inplace: Boolean to indicate whether the active SpecOneD \
        object will be modified or a new SpecOneD object will be created and \
        returned.
        :type inplace: bool
        :return: SpecOneD
        """

        if inplace:
            for attr in ['fluxden', 'fluxden_err', 'obj_model']:
                if self.__dict__[attr] is not None:
                    self.__dict__[attr] /= factor
            self.get_ivar_from_fluxden_error()
            self.fluxden_unit *= factor
        else:
            spec = self.copy()
            for attr in ['fluxden', 'fluxden_err', 'obj_model']:
                if spec.__dict__[attr] is not None:
                    spec.__dict__[attr] /= factor
            spec.get_ivar_from_fluxden_error()
            spec.fluxden_unit *= factor

            return spec



# ------------------------------------------------------------------------------
# CLASS FUNCTIONALITY
# ------------------------------------------------------------------------------

    def check_dispersion_overlap(self, secondary_spectrum):
        """Check the overlap between the active spectrum and the
        supplied secondary spectrum.

        This method determines whether the active spectrum (primary) and the
        supplied spectrum (secondary) have overlap in their dispersions.
        Possible cases include:
        i) The current spectrum dispersion is fully within the dispersion
        range of the 'secondary' spectrum -> 'primary' overlap.
        ii) The secondary spectrum dispersion is fully within the dispersion
        range of the current spectrum -> 'secondary' overlap.
        iii) and iv) There is only partial overlap between the spectra ->
        'partial' overlap.
        v) There is no overlap between the spectra -> 'none' overlap. In the
        case of no overlap np.NaN values are returned for the minimum and
        maximum dispersion limits.

        :param secondary_spectrum:
        :type secondary_spectrum: SpecOneD
        :return: overlap, overlap_min, overlap_max \
            Returns a string indicating the dispersion \
            overlap type according to the cases above 'overlap and the \
            minimum and maximum dispersion value of the overlap region of the \
            two spectra.
        :rtype: (str, float, float)
        """

        self.check_units(secondary_spectrum)

        spec_min = np.min(self.dispersion)
        spec_max = np.max(self.dispersion)

        secondary_min = np.min(secondary_spectrum.dispersion)
        secondary_max = np.max(secondary_spectrum.dispersion)

        if spec_min >= secondary_min and spec_max <= secondary_max:
            return 'primary', spec_min, spec_max
        elif spec_min < secondary_min and spec_max > secondary_max:
            return 'secondary', secondary_min, secondary_max
        elif spec_min <= secondary_min and secondary_min <= spec_max <= \
                secondary_max:
            return 'partial', secondary_min, spec_max
        elif secondary_max >= spec_min >= secondary_min and spec_max >= \
                secondary_max:
            return 'partial', spec_min, secondary_max
        else:
            return 'none', np.NaN, np.NaN

    def match_dispersions(self, secondary_spectrum, match_secondary=True,
                          force=False, method='interpolate',
                          interp_method='linear'):
        """Match the dispersion of the current spectrum and the secondary
        spectrum.

        Both, current and secondary, SpecOneD objects are modified in this
        process. The dispersion match identifies the maximum possible overlap
        in the dispersion direction of both spectra and automatically trims
        them to that range.

        If the current (primary) spectrum overlaps fully with the secondary
        spectrum the dispersion of the secondary will be interpolated/resampled
        to the primary dispersion.

        If the secondary spectrum overlaps fully with the primary, the primary
        spectrum will be interpolated/resampled on the secondary spectrum
        resolution, but this will only be executed if 'force==True' and
        'match_secondary==False'.

        If there is partial overlap between the spectra and 'force==True'
        the secondary spectrum will be interpolated/resampled to match the
        dispersion values of the primary spectrum.

        If there is no overlap a ValueError will be raised.

        :param secondary_spectrum: Secondary spectrum
        :type secondary_spectrum: SpecOneD
        :param match_secondary: The boolean indicates whether the secondary \
        will always be matched to the primary or whether reverse matching, \
        primary to secondary is allowed.
        :type match_secondary: bool
        :param force: The boolean sets whether the dispersions are matched if \
        only partial overlap between the spectral dispersions exists.
        :type force: bool
        :param method:
        :type method: str
        :param interp_method:
        :type interp_method: str
        :return:

        :raise ValueError: A ValueError will be raised if there is no overlap \
        between the spectra.
        """

        self.check_units(secondary_spectrum)

        overlap, s_min, s_max = self.check_dispersion_overlap(secondary_spectrum)

        if overlap == 'primary':
            if method == 'interpolate':
                secondary_spectrum.interpolate(self.dispersion,
                                               kind=interp_method,
                                               inplace=True)
            elif method == 'resample':
                secondary_spectrum.resample(self.dispersion, force=force,
                                            inplace=True)

        elif (overlap == 'secondary' and match_secondary is False and force
              is True):
            if method == 'interpolate':
                self.interpolate(secondary_spectrum.dispersion,
                                 kind=interp_method,
                                 inplace=True)
            elif method == 'resample':
                self.resample(secondary_spectrum.dispersion, force=force,
                              inplace=True)

        elif (overlap == 'secondary' and match_secondary is True and force is
              True):
            self.trim_dispersion(limits=[s_min, s_max], mode='physical',
                                 inplace=True)
            if method == 'interpolate':
                secondary_spectrum.interpolate(self.dispersion,
                                               kind=interp_method,
                                               inplace=True)
            elif method == 'resample':
                secondary_spectrum.resample(self.dispersion, force=force,
                                            inplace=True)

        elif overlap == 'partial' and force is True:
            self.trim_dispersion(limits=[s_min, s_max], mode='physical',
                                 inplace=True)
            if method == 'interpolate':
                secondary_spectrum.interpolate(self.dispersion,
                                               kind=interp_method,
                                               inplace=True)
            elif method == 'resample':
                secondary_spectrum.resample(self.dispersion, force=force,
                                            inplace=True)

        elif force is False and (overlap == 'secondary' or overlap ==
                                 'partial'):
            raise ValueError('[ERROR] There is secondary or partial overlap '
                             'between the spectra but force is False. Current '
                             'spectrum will not be modified')

        elif overlap == 'none':
            raise ValueError('[ERROR] There is no overlap between the current '
                             'and the secondary spectrum.')

    def trim_dispersion(self, limits, mode='physical', inplace=False):
        """Trim the spectrum according to the dispersion limits specified.

        :param limits: A list of two floats indicating the lower and upper \
        dispersion limit to trim the dispersion axis to.
        :type limits: [float, float] or [int, int]
        :param mode: A string specifying whether the limits are in 'physical' \
        values of the dispersion axis (e.g. Angstroem) or in pixel values.
        :type mode: str
        :param inplace: Boolean to indicate whether the active SpecOneD \
        object will be modified or a new SpecOneD object will be created and \
        returned.
        :type inplace: bool
        :return: Spectrum trimmed to the specified limits
        :rtype: SpecOneD
        """

        if mode == "physical":

            lo_index = np.argmin(np.abs(self.dispersion - limits[0]))
            up_index = np.argmin(np.abs(self.dispersion - limits[1]))

            # Warnings
            if limits[0] < self.dispersion[0]:
                print("[WARNING] Lower limit is below the lowest dispersion "
                      "value. The lower limit is set to the minimum "
                      "dispersion value.")
            if limits[1] > self.dispersion[-1]:
                print("[WARNING] Upper limit is above the highest dispersion "
                      "value. The upper limit is set to the maximum "
                      "dispersion value.")

        elif mode == "pixel":

            if limits[0] < 0:
                print("[WARNING] Lower limit is below the lowest pixel "
                      "value. The lower limit is set to the minimum "
                      "pixel value.")
            if limits[1] > len(self.dispersion-1):
                print("[WARNING] Upper limit is above the highest pixel "
                      "value. The upper limit is set to the maximum "
                      "pixel value.")

            lo_index = limits[0]
            up_index = limits[1]

        if inplace:
            for attr in ['dispersion', 'fluxden', 'mask', 'fluxden_err',
                         'fluxden_ivar', 'telluric', 'obj_model']:
                if self.__dict__[attr] is not None:
                    self.__dict__[attr] = self.__dict__[attr][lo_index:up_index]

        else:
            spec = self.copy()
            for attr in ['dispersion', 'fluxden', 'mask', 'fluxden_err',
                         'fluxden_ivar', 'telluric', 'obj_model']:
                if spec.__dict__[attr] is not None:
                    spec.__dict__[attr] = spec.__dict__[attr][lo_index:up_index]

            return spec

    def interpolate(self, new_dispersion, kind='linear', fill_value='const',
                    inplace=False, verbosity=0):
        """Interpolate spectrum to a new dispersion axis.

        The interpolation is done using scipy.interpolate.interp1d.

        Interpolating a spectrum to a new dispersion axis automatically
        resets the spectrum mask.

        :param new_dispersion: 1D array with the new dispersion axis
        :type new_dispersion: numpy.ndarray
        :param kind: String that indicates the interpolation function ( \
        default: 'linear')
        :type kind: str
        :param fill_value: A string indicating whether values outside the \
        dispersion range will be extrapolated ('extrapolate') or filled with \
        a constant value ('const') based on the median of the 10 values at \
        the edge.
        :type fill_value: str
        :param inplace: Boolean to indicate whether the active SpecOneD \
        object will be modified or a new SpecOneD object will be created and \
        returned.
        :type inplace: bool
        :param verbosity: Integer indecating the verbosity level
        :type verbosity: int
        :return:
        """

        # Test if values outside original range
        # Add support for ivar, obj_model, telluric etc.
        if verbosity > 0:
            print('[INFO] Interpolating spectrum to new dispersion axis.')
        if verbosity > 1:
            if fill_value == 'extrapolate':
                print('[INFO] Values outside the original dispersion range '
                      'will be extrapolated!')
            elif fill_value == 'const':
                print('[INFO] Values outside the original dispersion range  '
                      'will be padded with the median value of the first/last '
                      'ten pixels.')
            else:
                raise ValueError('[ERROR] Fill value mode not understood. '
                                 'Supported modes are "extrapolate" and '
                                 '"const".')

        snr_factor = np.sqrt(len(self.dispersion)/len(new_dispersion))

        if not inplace:
            spec = self.copy()

        for attr in ['fluxden', 'fluxden_err', 'fluxden_ivar', 'telluric',
                     'obj_model']:

            if self.__dict__[attr] is not None:

                if fill_value == 'extrapolate':
                    interp = sp.interpolate.interp1d(self.dispersion,
                                                    self.__dict__[attr],
                                                    kind=kind,
                                                    fill_value='extrapolate')

                if fill_value == 'const':
                    fill_lo = np.median(self.__dict__[attr][0:10])
                    fill_hi = np.median(self.__dict__[attr][-11:-1])
                    interp = sp.interpolate.interp1d(self.dispersion,
                                                     self.__dict__[attr],
                                                     kind=kind,
                                                     bounds_error=False,
                                                     fill_value=(fill_lo,
                                                                 fill_hi))

                if inplace:
                    self.__dict__[attr] = interp(new_dispersion)
                else:
                    spec.__dict__[attr] = interp(new_dispersion)

        if inplace:
            if self.fluxden_err is not None:
                self.fluxden_err /= snr_factor
            self.dispersion = new_dispersion
            self.reset_mask()
        else:
            if self.fluxden_err is not None:
                spec.fluxden_err /= snr_factor
            spec.dispersion = new_dispersion
            spec.reset_mask()
            return spec

    def smooth(self, width, kernel="boxcar", scale_sigma=True, inplace=False):
        """Smoothing the flux density of the spectrum using a boxcar oder
        gaussian kernel.

        This function uses astropy.convolution to convolve the spectrum with
        the selected kernel.

        If scale_sigma=True, the fluxden error is scaled down according to
        sqrt(width).

        :param width: Width (in pixels) of the kernel)
        :type: width: int
        :param kernel: String indicating whether to use the Boxcar ("boxcar") \
        or Gaussian ("gaussian") kernel.
        :type kernel: str
        :param scale_sigma: Boolean to indicate whether to scale the fluxden \
        error according to the width of the boxcar kernel.
        :type scale_sigma: bool
        :param inplace: Boolean to indicate whether the active SpecOneD \
        object will be modified or a new SpecOneD object will be created and \
        returned.
        :type inplace: bool
        :return:
        """

        if kernel == "boxcar" or kernel == "Boxcar":
            kernel = Box1DKernel(width)
        elif kernel == "gaussian" or kernel == "Gaussian":
            kernel = Gaussian1DKernel(width)

        new_flux = convolve(self.fluxden, kernel)
        if self.fluxden_err is not None:
            new_flux_err = convolve(self.fluxden_err, kernel)
            if scale_sigma:
                new_flux_err /= np.sqrt(width)
        else:
            new_flux_err = self.fluxden_err


        if inplace:
            self.fluxden = new_flux
            self.fluxden_err = new_flux_err
        else:
            spec = self.copy()
            spec.fluxden= new_flux
            spec.fluxden_err = new_flux_err
            return spec

    def create_dispersion_by_resolution(self, resolution):
        """
        This function creates a new dispersion axis in wavelength sampled by
        a fixed resolution, given in km/s.

        This should work for all spectra with flux densities per unit
        wavelength/frequency.

        :param resolution:
        :return: Returns new dispersion axis with a resolution in km/s as \
        given by the input value.
        :rtype: numpy.ndarray
        """

        new_dispersion = [self.dispersion[0]]
        lambda_new = 0
        while lambda_new < self.dispersion[-1]:

            d_lambda = new_dispersion[-1]/const.c.to(u.km/u.s).value * \
                       resolution
            lambda_new = d_lambda + new_dispersion[-1]
            new_dispersion.append(lambda_new)

        return np.array(new_dispersion[1:-1])

    def resample_to_resolution(self, resolution, buffer=2, inplace=False):
        """Resample spectrum at a specific resolution specified in km/s.

        This should work for all spectra with flux densities per unit
        wavelength/frequency.

        :param resolution: Specified resolution in km/s
        :type resolution: float
        :param buffer: Integer value indicating how many pixels at the \
        beginning and the end of the current spectrum will be omitted in the \
        resampling process.
        :type buffer: int
        :param inplace: Boolean to indicate whether the active SpecOneD \
        object will be modified or a new SpecOneD object will be created and \
        returned.
        :type inplace: bool
        :return: Returns the resampled spectrum as a SpecOneD object if \
        inplace==False.
        :rtype: SpecOneD
        """

        new_dispersion = self.create_dispersion_by_resolution(resolution)

        if inplace:
            self.resample(new_dispersion[buffer:-buffer], inplace=inplace)
        else:
            return self.resample(new_dispersion[buffer:-buffer],
                                 inplace=inplace)

    def resample(self, new_dispersion, force=False, inplace=False):
        """ Function for resampling spectra (and optionally associated
        uncertainties) onto a new wavelength basis.

        This code is making use of SpectRes
        https://github.com/ACCarnall/SpectRes
        by Adam Carnall - damc@roe.ac.uk

        The mask will be automatically reset.

        If obj_model and telluric exist for the spectrum these will be
        linearly interpolated onto the new dispersion axis and NOT resampled.

        :param new_dispersion: Array containing the new wavelength sampling \
         desired for the spectrum or spectra.
        :type new_dispersion: numpy.ndarray
        :param force: Boolean to force the resampling of the spectrum by \
        reducing the new dispersion axis range to the old dispersion axis range.
        :type force: bool
        :param inplace: Boolean to indicate whether the active SpecOneD \
        object will be modified or a new SpecOneD object will be created and \
        returned.
        :type inplace: bool
        :return: Returns the resampled spectrum as a SpecOneD object if \
        inplace==False.
        :rtype: SpecOneD
        """

        old_spec_wavs = self.dispersion
        spec_fluxes = self.fluxden
        if self.fluxden_err is not None:
            spec_errs = self.fluxden_err
        else:
            spec_errs = None

        new_spec_wavs = new_dispersion

        if force:
            print('[WARNING] Forcing new spectral dispersion axis to be within '
                  'the limits of the old spectral dispersion.')
            indices = np.where((new_spec_wavs < old_spec_wavs.max()) &
                               (new_spec_wavs > old_spec_wavs.min()))
            new_spec_wavs = new_spec_wavs[indices]

        output = spectres.spectres(new_spec_wavs, old_spec_wavs, spec_fluxes,
                          spec_errs=spec_errs,
                          fill=None, verbose=True)

        if spec_errs is None:
            new_fluxden = output
        else:
            new_fluxden, new_fluxden_err = output

        if inplace:
            self.interpolate(new_dispersion, inplace=True)
            self.dispersion = new_spec_wavs
            self.fluxden = new_fluxden
            if spec_errs is not None:
                self.fluxden_err = new_fluxden_err
            self.reset_mask()

        else:
            spec = self.copy()
            spec.interpolate(new_dispersion, inplace=True)
            spec.dispersion = new_spec_wavs
            spec.fluxden = new_fluxden
            if spec_errs is not None:
                spec.fluxden_err = new_fluxden_err
            spec.reset_mask()

            return spec

    def bin_by_npixels(self, npix, inplace=False):
        """Bin the spectrum by an integer number of pixel.

        The spectrum is binned by npix pixel. A new dispersion axis is
        calculated asumming that the old dispersion values marked the center
        positions of their bins.

        The flux density (obj_model, telluric) are averaged over the new bin
        width, whereas the flux density error is accordingly propagated.

        The spectrum mask will be automatically reset.

        :param npix: Number of pixels to be binned.
        :type npix: int
        :param inplace: Boolean to indicate whether the active SpecOneD \
        object will be modified or a new SpecOneD object will be created and \
        returned.
        :type inplace: boolean
        :return: Returns the binned spectrum as a SpecOneD object if \
        inplace==False.
        :rtype: SpecOneD
        """

        disp = self.dispersion
        dbins = disp[1:] - disp[:-1]
        bin_boundary = disp[:-1] + 0.5 * dbins

        lbins = bin_boundary[:-1]
        rbins = bin_boundary[1:]
        mbins = disp[1:-1]
        dbins = rbins - lbins
        flux = self.fluxden[1:-1]
        flux_err = self.fluxden_err[1:-1]
        num_bins = len(mbins)

        num_new_bins = int((num_bins - (num_bins % npix)) / npix)

        new_disp = np.zeros(num_new_bins)
        new_fluxden = np.zeros(num_new_bins)
        new_fluxden_err = np.zeros(num_new_bins)

        for idx in range(num_new_bins):

            _new_fluxden = 0
            _new_fluxden_err = 0
            _new_dbin = 0

            for jdx in range(npix):
                _new_fluxden += flux[idx * npix + jdx] * dbins[idx * npix + jdx]
                _new_dbin += dbins[idx * npix + jdx]
                _new_fluxden_err += (flux_err[idx * npix + jdx] * dbins[
                    idx * npix + jdx]) ** 2

            rbin = rbins[npix * idx + npix - 1]
            lbin = lbins[npix * idx]
            _new_disp = (rbin - lbin) * 0.5 + lbin

            new_disp[idx] = _new_disp
            new_fluxden[idx] = _new_fluxden / _new_dbin
            new_fluxden_err[idx] = np.sqrt(_new_fluxden_err) / _new_dbin

        attr_dict = {}
        for attr_name in ['obj_model', 'telluric']:
            if self.__dict__[attr_name] is not None:
                new_attr = np.zeros(num_new_bins)
                attr = self.__dict__[attr_name]
                for idx in range(num_new_bins):
                    _new_attr = 0
                    _new_dbin = 0
                    for jdx in range(npix):
                        _new_attr += attr[idx * npix + jdx] * \
                                     dbins[idx * npix + jdx]
                        _new_dbin += dbins[idx * npix + jdx]

                    new_attr[idx] = _new_attr/_new_dbin
            else:
                new_attr = None

            attr_dict[attr_name] = new_attr

        if inplace:
            self.dispersion = new_disp
            self.fluxden = new_fluxden
            self.fluxden_err = new_fluxden_err
            self.reset_mask()

            for attr in ['obj_model', 'telluric']:
                self.__dict__[attr] = attr_dict[attr]

        else:
            spec = self.copy()
            spec.dispersion = new_disp
            spec.fluxden = new_fluxden
            spec.fluxden_err = new_fluxden_err
            spec.reset_mask()

            for attr in ['obj_model', 'telluric']:
                spec.__dict__[attr] = attr_dict[attr]

            return spec

# ------------------------------------------------------------------------------
# PLOT FUNCTIONALITY
# ------------------------------------------------------------------------------

    def plot(self, show_fluxden_err=True, mask_values=True, ymin=None,
             ymax=None, show_obj_model=True, show_telluric=True):
        """Plot the spectrum.

        This plot is aimed for a quick visualization of the spectrum not for
        publication grade figures.

        :param show_fluxden_err: Boolean to indicate whether the error will \
        be plotted or not (default:True).
        :type show_fluxden_err: bool
        :param mask_values: Boolean to indicate whether the mask will be \
        applied when plotting the spectrum (default:True).
        :type mask_values: bool
        :param ymin: Minimum value for the y-axis of the plot (flux density \
        axis). This defaults to 'None'. If either ymin or ymax are 'None' the \
        y-axis range will be determined automatically.
        :type ymin: float
        :param ymax: Maximum value for the y-axis of the plot (flux density \
        axis). This defaults to 'None'. If either ymin or ymax are 'None' the \
        y-axis range will be determined automatically.
        :type ymax: float
        :param show_obj_model: Boolean to indicate whether the object model \
        will be plotted or not (default:True).
        :type show_obj_model: bool
        :param show_telluric: Boolean to indicate whether the atmospheric \
        model will be plotted or not (default:True).
        :type show_telluric: bool
        :return:
        """

        if mask_values:
            mask = self.mask
        else:
            mask = np.ones(self.dispersion.shape, dtype=bool)

        fig, ax = plt.subplots(nrows=1, ncols=1,
                               figsize=(15, 7),
                               dpi=140)
        fig.subplots_adjust(left=0.09, right=0.97, top=0.89, bottom=0.16)

        # Plot the Spectrum
        ax.axhline(y=0.0, linewidth=1.5, color='k', linestyle=':',
                   label='Line of 0 flux density')

        if show_fluxden_err and self.fluxden_err is not None:
            ax.plot(self.dispersion[mask], self.fluxden_err[mask], 'grey',
                    lw=1, label='Flux density error')

        ax.plot(self.dispersion[mask], self.fluxden[mask], 'k',
                linewidth=1, label='Flux density')

        # Additional plotting functionality for spectra with obj_models
        if self.obj_model is not None and show_obj_model:
            ax.plot(self.dispersion[mask], self.obj_model[mask],
                    color=vermillion,
                    label='Object model')
        # Additional plotting functionality for spectra with telluric models
        if self.telluric is not None and show_telluric:
            ax2 = ax.twinx()
            ax2.plot(self.dispersion[mask], self.telluric[mask], color=dblue,
                    label='Atmospheric model')
            ax2.set_ylabel('Atmospheric model transmission', fontsize=14)
            ax2.legend(loc=2, fontsize=14)


        ax.set_xlabel('Dispersion ({})'.format(
            self.dispersion_unit.to_string(format='latex')), fontsize=14)

        ax.set_ylabel('Flux density ({})'.format(
            self.fluxden_unit.to_string(format='latex')), fontsize=14)

        if ymin is None or ymax is None:
            lim_spec = self.copy()
            ymin, ymax = lim_spec.get_specplot_ylim()

        ax.set_ylim(ymin, ymax)

        ax.legend(loc=1, fontsize=14)

        plt.show()

    def get_specplot_ylim(self):
        """Calculate the minimum and maximum flux density values for plotting
         the spectrum.

         The minimum value is set to -0.5 * median of the flux density. The
         maximum value is set 4 times the 84-percentile value of the flux
         density.
         This is an 'approximate guess' for a quick visualization of the
         spectrum and may not be optimal for all purposes. For pulication
         grade plots, the user should devise their own plots.

        :return: (ylim_min, ylim_max) Return the minimum and maximum values \
        for the flux density (y-axis) for the plot function.
        :rtype: (float, float)
        """

        spec = self.copy()

        percentiles = np.percentile(spec.fluxden[spec.mask], [16, 84])
        median = np.median(spec.fluxden[spec.mask])

        ylim_min = -0.5*median
        ylim_max = 4*percentiles[1]

        return ylim_min, ylim_max

# ------------------------------------------------------------------------------
# ADVANCED FUNCTIONALITY
# ------------------------------------------------------------------------------

    def apply_extinction(self, a_v, r_v, extinction_law='ccm89', inplace=False):
        """Apply extinction to the spectrum (flux density ONLY).

        This function makes use of the python extinction package:
        https://github.com/kbarbary/extinction .

        Their documentation is available at
        https://extinction.readthedocs.io/en/latest/ .

        Please have a careful look at their implementation and regarding
        details on the use of a_v and r_v. Possible extinction laws to use
        are "ccm89", "odonnell94", "calzetti00", "fitzpatrick99", "fm07".

        :param a_v: Extinction value ein the V band.
        :type a_v: float
        :param r_v:  Ratio of total to selective extinction r_v = a_v/E(B-V)
        :type r_v: float
        :param extinction_law: Extinction law name as implemented in the \
        extinction package, see documentation.
        :type extinction_law: str
        :param inplace: Boolean to indicate whether the active SpecOneD \
        object will be modified or a new SpecOneD object will be created and \
        returned.
        :type inplace: bool
        :return: Returns the binned spectrum as a SpecOneD object if \
        inplace==False.
        :rtype: SpecOneD
        """

        if self.dispersion_unit / self.dispersion_unit.value != u.AA:
            raise ValueError('[ERROR] Spectrum must be in flux density per '
                             'Angstroem. Use the internal function '
                             '"to_fluxden_per_unit_wavelength_cgs" to convert'
                             ' the spectrum accordingly.')

        if extinction_law == 'ccm89':
            extinction = ext.ccm89(self.dispersion, a_v, r_v)
        elif extinction_law == 'odonnel94':
            extinction = ext.odonnel94(self.dispersion, a_v, r_v)
        elif extinction_law == 'calzetti00':
            extinction = ext.calzetti00(self.dispersion, a_v, r_v)
        elif extinction_law == 'fitzpatrick99':
            extinction = ext.fitzpatrick99(self.dispersion, a_v, r_v)
        elif extinction_law == 'fm07':
            print('[Warning] For Fitzpatrick & Massa 2007 R_V=3.1')
            extinction = ext.fm07(self.dispersion, a_v)
        else:
            raise ValueError('[ERROR] Specified extinction law not recognized')

        if inplace:
            self.fluxden = self.fluxden * 10.0 ** (-0.4 * extinction)
        else:
            spec = self.copy()
            spec.fluxden = spec.fluxden * 10.0 ** (-0.4 * extinction)

            return spec

    def remove_extinction(self, a_v, r_v, extinction_law='ccm89',
                          inplace=False):
        """Remove extinction from spectrum (flux density ONLY).

        This function makes use of the python extinction package:
        https://github.com/kbarbary/extinction .

        Their documentation is available at
        https://extinction.readthedocs.io/en/latest/ .

        Please have a careful look at their implementation and regarding
        details on the use of a_v and r_v. Possible extinction laws to use
        are "ccm89", "odonnell94", "calzetti00", "fitzpatrick99", "fm07".

        :param a_v: Extinction value ein the V band.
        :type a_v: float
        :param r_v:  Ratio of total to selective extinction r_v = a_v/E(B-V)
        :type r_v: float
        :param extinction_law: Extinction law name as implemented in the \
        extinction package, see documentation.
        :type extinction_law: str
        :param inplace: Boolean to indicate whether the active SpecOneD \
        object will be modified or a new SpecOneD object will be created and \
        returned.
        :type inplace: bool
        :return: Returns the binned spectrum as a SpecOneD object if \
        inplace==False.
        :rtype: SpecOneD
        """

        if isinstance(self.dispersion_unit, (u.Unit, u.IrreducibleUnit,
                                             u.CompositeUnit)):
            if self.dispersion_unit != u.AA:
                raise ValueError('[ERROR] Spectrum must be in flux density per '
                                 'Angstroem. Use the internal function '
                                 '"to_fluxden_per_unit_wavelength_cgs" to '
                                 'convert the spectrum accordingly.')
        elif isinstance(self.dispersion_unit, u.Quantity):
            if self.dispersion_unit / self.dispersion_unit.value != u.AA:
                raise ValueError('[ERROR] Spectrum must be in flux density per '
                                 'Angstroem. Use the internal function '
                                 '"to_fluxden_per_unit_wavelength_cgs" to '
                                 'convert the spectrum accordingly.')

        if extinction_law == 'ccm89':
            extinction = ext.ccm89(self.dispersion, a_v, r_v)
        elif extinction_law == 'odonnel94':
            extinction = ext.odonnel94(self.dispersion, a_v, r_v)
        elif extinction_law == 'calzetti00':
            extinction = ext.calzetti00(self.dispersion, a_v, r_v)
        elif extinction_law == 'fitzpatrick99':
            extinction = ext.fitzpatrick99(self.dispersion, a_v, r_v)
        elif extinction_law == 'fm07':
            print('[Warning] For Fitzpatrick & Massa 2007 R_V=3.1')
            extinction = ext.fm07(self.dispersion, a_v)
        else:
            raise ValueError('[ERROR] Specified extinction law not recognized')

        if inplace:
            self.fluxden = self.fluxden * 10.0 ** (0.4 * extinction)
        else:
            spec = self.copy()
            spec.fluxden = spec.fluxden * 10.0 ** (0.4 * extinction)

            return spec

    def broaden_by_gaussian(self, fwhm, inplace=False):
        """The spectrum is broadened by a Gaussian with the specified FWHM (
        in km/s).

        The convolution of the current spectrum and the Gaussian is performed
        in logarithmic wavelength. Therefore, the spectrum is first converted to
        flux per logarithmic wavelength, then convolved with the Gaussian
        kernel and then converted back.

        The conversion functions will automatically take care of the unit
        conversion and input spectra can be in flux density per unit
        frequency or wavelength.

        This function normalizes the output of the convolved spectrum in a
        way that a Gaussian input signal of FWHM X broadened by a Gaussian
        kernel of FWHM Y, results in a Gaussian output signal of FWHM sqrt(
        X**2+Y**2) with the same amplitude as the input signal. Due to the
        normalization factor of the Gaussian itself, this results in a lower
        peak height.

        The input spectrum and the Gaussian kernel are matched to the same
        dispersion axis using the 'interpolate' function.

        :param fwhm: FWHM of the Gaussian that the spectrum will be \
        convolved with in km/s.
        :type fwhm: float
        :param inplace: Boolean to indicate whether the active SpecOneD \
        object will be modified or a new SpecOneD object will be created and \
        returned.
        :type inplace: bool
        :return: Returns the binned spectrum as a SpecOneD object if \
        inplace==False.
        :rtype: SpecOneD
        """

        spec = self.copy()

        stddev = fwhm / const.c.to('km/s').value / (2 * np.sqrt(2 * np.log(2)))

        # Convert spectrum to logarithmic wavelength (velocity space)
        spec._to_log_wavelength()
        # Interpolate to linear scale in logarithmic wavelength
        new_disp = np.linspace(min(spec.dispersion), max(spec.dispersion),
                               num=len(spec.dispersion))
        spec.interpolate(new_disp, inplace=True)

        # Setup the normalized Gaussian kernel
        cen = (max(new_disp) - min(new_disp)) / 2. + min(new_disp)
        kernel = gaussian(new_disp, 1.0, cen, stddev, 0)
        conv = np.convolve(spec.fluxden, kernel, mode='same')

        # Normalize convolved flux
        conv = conv / len(kernel) / np.sqrt(8*np.log(2))**2

        spec.fluxden = conv

        # Convert back to linear wavelength units
        spec._to_lin_wavelength()
        spec.interpolate(self.dispersion, inplace=True)

        if inplace:
            self.fluxden = spec.fluxden
        else:
            rspec = self.copy()
            rspec.fluxden = spec.fluxden
            rspec.fluxden_unit = spec.fluxden_unit

            return rspec

    def calculate_passband_flux(self, passband,
                                match_method='interpolate', force=False):
        """Calculate the integrated flux in the specified passband.

        Disclaimer: This function is written for passbands in quantum
        efficiency. Therefore, the (h*nu)^-1 term is not included in the
        integral.

        :param passband: The astronomical passband with throughput in quantum \
        efficiencies.
        :type passband: PassBand
        :param match_method: Method for matching the dispersion axis of the \
        spectrum to the passband.
        :type match_method: str
        :param force: Boolean to indicate if they spectra will be forced to \
        match if the spectrum does not fully cover the passband. The \
        forced match will result in an inner match of the spectrum's and the \
        passband's dispersion axes. User discretion is advised.
        :type force: bool
        :return: Integrated spectrum flux in the passband
        :rtype: Quantity
        """

        # Copy spectrum
        spec = self.copy()

        # Convert passband to spectrum dispersion
        passband.convert_spectral_units(spec.dispersion_unit)

        overlap, disp_min, disp_max = passband.check_dispersion_overlap(spec)

        if not force:
            if overlap != 'primary':
                raise ValueError('[ERROR] The spectrum does not fill the '
                                 'passband')
        else:
            print("[Warning] Force was set to TRUE. The spectrum might not "
                  "fully fill the passband!")

        passband.match_dispersions(spec, force=force, method=match_method)
        spec.fluxden = passband.fluxden * spec.fluxden
        total_flux = np.trapz(spec.fluxden * spec.fluxden_unit,
                              spec.dispersion * spec.dispersion_unit)

        if total_flux <= 0.0:
            raise ValueError('[ERROR] Integrated flux is <= 0')
        if np.isnan(total_flux):
            raise ValueError('[ERROR] Integrated flux is NaN')
        if np.isinf(total_flux):
            raise ValueError('[ERROR] Integrated flux is infinite')

        return total_flux

    def calculate_passband_ab_magnitude(self, passband,
                                        match_method='interpolate',
                                        force=False):
        """Calculate the AB magnitude of the spectrum in the given passband.

        Disclaimer: This function is written for passbands in quantum
        efficiency. Therefore, the (h*nu)^-1 term is not included in the
        integral.

        :param passband: The astronomical passband with throughput in quantum \
        efficiencies.
        :type passband: PassBand
        :param match_method: Method for matching the dispersion axis of the \
        spectrum to the passband.
        :type match_method: str
        :param force: Boolean to indicate if they spectra will be forced to \
        match if the spectrum does not fully cover the passband. The \
        forced match will result in an inner match of the spectrum's and the \
        passband's dispersion axes. User discretion is advised.
        :type force: bool
        :return: AB magnitude of the spectrum in the specified passband \
        :rtype: float
        """
        #

        spec = self.copy()
        spec.to_fluxden_per_unit_frequency_cgs()

        spectrum_flux = spec.calculate_passband_flux(passband, force=force,
                                                     match_method=match_method)

        flat_flux = 3.631e-20 * np.ones_like(passband.dispersion) * \
                    u.erg/u.s/u.cm**2/u.Hz

        passband_flux = np.trapz(flat_flux * passband.fluxden,
                                 passband.dispersion * passband.dispersion_unit)

        return -2.5 * np.log10(spectrum_flux / passband_flux)

    def renormalize_by_ab_magnitude(self, magnitude, passband,
                                    match_method='interpolate', force=False,
                                    output_mode='spectrum', inplace=False):
        """Scale the spectrum flux density and 1-sigma errors to the specified
        magnitude in the provided passband.

        :param magnitude: Magnitude to scale the spectrum to.
        :type magnitude: float
        :param passband: The astronomical passband with throughput in quantum \
        efficiencies.
        :type passband: PassBand
        :param match_method: Method for matching the dispersion axis of the \
        spectrum to the passband.
        :type match_method: str
        :param force: Boolean to indicate if they spectra will be forced to \
        match if the spectrum does not fully cover the passband. The \
        forced match will result in an inner match of the spectrum's and the \
        passband's dispersion axes. User discretion is advised. \
        :type force: bool
        :param output_mode: Output mode of the function. The default mode \
        "Spectrum" returns the rescaled spectrum as a SpecOneD object or if \
        inplace=True updates the provided spectrum. The alternative output \
        mode "flux_factor" returns the factor to scale the flux with as a float.
        :type output_mode: str
        :param inplace: Boolean to indicate whether the active SpecOneD \
        object will be modified or a new SpecOneD object will be created and \
        returned.
        :type inplace: bool
        :return: Normalized spectrum or flux density normalization factor
        """

        spec_mag = self.calculate_passband_ab_magnitude(passband,
                                                        force=force,
                                                        match_method=
                                                        match_method)

        dmag = magnitude - spec_mag

        if output_mode == 'spectrum':

            if inplace:
                self.fluxden = self.fluxden * 10 ** (-0.4 * dmag.value)
                if self.fluxden_err is not None:
                    self.fluxden_err = self.fluxden_err * 10 ** (-0.4 *
                                                                 dmag.value)
            else:
                spec = self.copy()
                spec.fluxden = spec.fluxden * 10**(-0.4*dmag.value)
                if spec.fluxden_err is not None:
                    spec.fluxden_err = spec.fluxden_err * 10**(-0.4*dmag.value)
                return spec

        elif output_mode == 'flux_factor':

            return 10**(-0.4*dmag)

        else:
            raise ValueError("[ERROR] Output mode {} not supported. "
                             "Output modes are 'spectrum' or "
                             "'flux_factor'.".format(output))

    def renormalize_by_spectrum(self, spectrum, dispersion_limits=None,
                                output_mode='spectrum', inplace=False):
        """Scale the spectrum flux density and 1-sigma errors to match the
        provided spectrum in the full overlap region or in a specified
        dispersion range.

        The original SpecOneD spectrum and the normalization spectrum should be
        in the same units. If this is not the case, the normalization spectrum
        will be converted to the same units as the original SpecOneD spectrum.

        The dispersion limits are unitless (list of two floats) but need to
        be in the same units as the SpecOneD dispersion axis (dispersion_unit).

        :param spectrum: The provided spectrum to scale the SpecOneD spectrum \
        to.
        :type spectrum: SpecOneD
        :param dispersion_limits: A list of two floats indicating the lower \
        and upper dispersion limits between which the spectra are normalized.
        :type dispersion_limits: (float, float)
        :param output_mode: Output mode of the function. The default mode \
        "Spectrum" returns the rescaled spectrum as a SpecOneD object or if \
        inplace=True updates the provided spectrum. The alternative output \
        mode "flux_factor" returns the factor to scale the flux with as a float.
        :type output_mode: str
        :param inplace: Boolean to indicate whether the active SpecOneD \
        object will be modified or a new SpecOneD object will be created and \
        returned.
        :type inplace: bool
        :return:
        """

        # Convert the spectra to dummy variables
        spec = self.copy()
        spec2 = spectrum.copy()

        # Convert the renormalization spectrum to the same units as the
        # target spectrum.
        spec2.convert_spectral_units(spec.dispersion_unit, spec.fluxden_unit)

        if dispersion_limits is None:
            spec.match_dispersions(spec2, match_secondary=False,
                                   method='interpolate', force=True,
                                   interp_method='linear')
        elif isinstance(dispersion_limits, (list,)):
            spec.trim_dispersion(dispersion_limits, mode='physical',
                                 inplace=True)
            spec2.trim_dispersion(dispersion_limits, mode='physical',
                                  inplace=True)
        else:
            raise ValueError('[ERROR] Specified dispersion limits type not '
                             'understood. The function can take None or a '
                             'list of two floats.')

        average_self_flux = np.trapz(spec.fluxden, spec.dispersion)
        average_spec_flux = np.trapz(spec2.fluxden, spec2.dispersion)

        scale_factor = (average_spec_flux/average_self_flux)

        if output_mode == 'spectrum':
            if inplace:
                self.fluxden = self.fluxden * scale_factor
                if self.fluxden_err is not None:
                    self.fluxden_err = self.fluxden_err * scale_factor
            else:
                spec = self.copy()
                spec.fluxden = spec.fluxden * scale_factor
                if spec.fluxden_err is not None:
                    spec.fluxden_err = spec.fluxden_err * scale_factor

                return spec
        elif output_mode == 'flux_factor':
            return scale_factor

    # def redshift(self, z, method='doppler', inplace=False):
    #     """Redshift the spectrum.
    #
    #     :param z:
    #     :param method:
    #     :param inplace:
    #     :return:
    #     """
    #     pass
    #
    # def blueshift(self, z, method='doppler', inplace=False):
    #     """Blueshift the spectrum.
    #
    #     :param z:
    #     :param method:
    #     :param inplace:
    #     :return:
    #     """
    #
    #     pass


class PassBand(SpecOneD):
    """The PassBand class, a child of the SpecOneD class, is a data structure
    for storing and manipulating astronomical filter transmission curves.

    :param passband_name: Name of the passband. The passband names \
    provided with the Sculptor package are in the format [ \
    INSTRUMENT]-[BAND] and can be found in the Sculptor data folder.
    :type passband_name: str
    :param dispersion: A 1D array providing the dispersion axis of the \
    passband.
    :type dispersion: numpy.ndarray
    :param fluxden: A 1D array providing the transmission data of the \
    spectrum in quantum efficiency.
    :type fluxden: numpy.ndarray
    :param fluxden_err: A 1D array providing the 1-sigma error of the \
    passband's transmission curve.
    :type fluxden_err: numpy.ndarray
    :param header: A pandas DataFrame containing additional information \
    on the spectrum.
    :type header: pandas.DataFrame
    :param dispersion_unit: The physical unit (including normalization \
    factors) of the dispersion axis of the passband.
    :type dispersion_unit: astropy.units.Unit or astropy.units.Quantity or \
    astropy.units.CompositeUnit or astropy.units.IrreducibleUnit
    :param fluxden_unit: The physical unit (including normalization \
    factors) of the transmission curve and associated properties (e.g. \
    flux density error) of the spectrum.
    :type fluxden_unit: astropy.units.Unit or astropy.units.Quantity or \
    astropy.units.CompositeUnit or astropy.units.IrreducibleUnit

    """

    def __init__(self, passband_name=None, dispersion=None, fluxden=None,
                 fluxden_err=None, header=None, dispersion_unit=None,
                 fluxden_unit=None):
        """Initialize the PassBand object.

        :param passband_name: Name of the passband. The passband names \
        provided with the Sculptor package are in the format [ \
        INSTRUMENT]-[BAND] and can be found in the Sculptor data folder.
        :type passband_name: str
        :param dispersion: A 1D array providing the dispersion axis of the \
        passband.
        :type dispersion: numpy.ndarray
        :param fluxden: A 1D array providing the transmission data of the \
        spectrum in quantum efficiency.
        :type fluxden: numpy.ndarray
        :param fluxden_err: A 1D array providing the 1-sigma error of the \
        passband's transmission curve.
        :type fluxden_err: numpy.ndarray
        :param header: A pandas DataFrame containing additional information \
        on the spectrum.
        :type header: pandas.DataFrame
        :param dispersion_unit: The physical unit (including normalization \
        factors) of the dispersion axis of the passband.
        :type dispersion_unit: astropy.units.Unit or astropy.units.Quantity or \
        astropy.units.CompositeUnit or astropy.units.IrreducibleUnit
        :param fluxden_unit: The physical unit (including normalization \
        factors) of the transmission curve and associated properties (e.g. \
        flux density error) of the spectrum.
        :type fluxden_unit: astropy.units.Unit or astropy.units.Quantity or \
        astropy.units.CompositeUnit or astropy.units.IrreducibleUnit
        """

        if passband_name is not None:
            self.load_passband(passband_name)
        else:
            super(PassBand, self).__init__(dispersion=dispersion,
                                           fluxden=fluxden,
                                           fluxden_err=fluxden_err,
                                           dispersion_unit=dispersion_unit,
                                           fluxden_unit=fluxden_unit,
                                           header=header)

    def show_available_passbands(self):
        pass

    def load_passband(self, passband_name, tolerance=0.005):
        """Load a passband from the sculptor/data/passbands folder.

        The passband names are in the following format:
        [INSTRUMENT]-[BAND]

        :param passband_name: Name of the passband, e.g. WISE-W1
        :type passband_name: str
        :param tolerance: Value below which the passband throughput will be \
        ignored when reading the passband in. In many cases the original \
        passband files contain a large range of 0 values below and above the \
        passband. The default value for the tolerance is 0.005, i.e. 0.5% \
        throughput.
        :type tolerance: float
        :return:
        """

        passband_filename = \
            pkg_resources.resource_filename('sculptor',
                                    'data/passbands/{}.dat'.format(
                                        passband_name))
        passband_data = np.genfromtxt(passband_filename)

        wavelength = passband_data[:, 0]
        throughput = passband_data[:, 1]

        # Change wavelength to Angstroem for all passbands
        filter_group = passband_name.split('-')[0]

        if filter_group == "WISE":
            # micron to Angstroem
            wavelength = wavelength * 10000.

        elif filter_group in ["LSST", "SWIRC"]:
            # nm to Angstroem
            wavelength = wavelength * 10.

        # Correct percent to fraction of 1
        if filter_group in ["SWIRC"]:
            throughput = throughput / 100.

        # Order wavelength in increasing order
        if wavelength[0] > wavelength[-1]:
            wavelength = wavelength[::-1]
            throughput = throughput[::-1]

        # Only select passband ranges with contributions above tolerance
        tolerance_mask = throughput > tolerance
        throughput = throughput[tolerance_mask]
        wavelength = wavelength[tolerance_mask]

        self.dispersion = wavelength
        self.fluxden = throughput
        self.fluxden_err = None
        self.fluxden_ivar = None
        self.obj_model = None
        self.telluric = None
        self.header = pd.DataFrame(columns=['value'])
        self.header.loc['passband_name', 'value'] = passband_name
        self.mask = np.ones(self.dispersion.shape, dtype=bool)

        self.dispersion_unit = 1 * u.AA
        self.fluxden_unit = 1 * u.dimensionless_unscaled

    def convert_spectral_units(self, new_dispersion_unit):
        """Convert the passband to new physical dispersion units.

        This function only converts the passband dispersion axis.

        :param new_dispersion_unit:
        :type: astropy.units.Unit or astropy.units.Quantity or \
        astropy.units.CompositeUnit or astropy.units.IrreducibleUnit
        :return:
        """

        # Setup physical spectral properties
        dispersion = self.dispersion * self.dispersion_unit

        # Convert dispersion axis
        new_dispersion = dispersion.to(new_dispersion_unit,
                                       equivalencies=u.spectral())

        self.dispersion = new_dispersion.value
        self.dispersion_unit = new_dispersion.unit

        self._reorder_dispersion()

    def plot(self, mask_values=False, ymin=0,  ymax=1.1):
        """Plot the passband.

        This plot is aimed for a quick visualization of the passband spectrum
        and not for publication grade figures.

        :param mask_values: Boolean to indicate whether the mask will be \
        applied when plotting the spectrum (default:True).
        :param ymin: Minimum value for the y-axis of the plot (flux density \
        axis). This defaults to 'None'. If either ymin or ymax are 'None' the \
        y-axis range will be determined automatically.
        :type ymin: float
        :param ymax: Maximum value for the y-axis of the plot (flux density \
        axis). This defaults to 'None'. If either ymin or ymax are 'None' the \
        y-axis range will be determined automatically.
        :type ymax: float
        :type mask_values: bool
        :return:
        """

        if mask_values:
            mask = self.mask
        else:
            mask = np.ones(self.dispersion.shape, dtype=bool)

        fig, ax = plt.subplots(nrows=1, ncols=1,
                               figsize=(15, 7),
                               dpi=140)
        fig.subplots_adjust(left=0.09, right=0.97, top=0.89, bottom=0.16)

        # Plot the Spectrum
        ax.axhline(y=0.0, linewidth=1.5, color='k', linestyle=':',
                   label='Line of 0 transmission')

        passband_name = self.header.loc['passband_name', 'value']
        ax.plot(self.dispersion[mask], self.fluxden[mask], 'k',
                linewidth=1, label=passband_name)

        ax.set_xlabel('Dispersion ({})'.format(
            self.dispersion_unit.to_string(format='latex')), fontsize=14)

        ax.set_ylabel('Passband transmission', fontsize=14)

        if ymin is None or ymax is None:
            lim_spec = self.copy()
            ymin, ymax = lim_spec.get_specplot_ylim()

        ax.set_ylim(ymin, ymax)

        ax.legend(loc=1, fontsize=14)

        plt.show()




# ------------------------------------------------------------------------------
# OLD DEPRECATED CAPABILITIES, IN PART REPLACED OR IMPROVED
# ------------------------------------------------------------------------------
    # def save_to_speconed_fits(self, filename, comment=None, overwrite = False):
    #     """Save a SpecOneD spectrum to a fits file.
    #
    #     Note: This save function does not store flux_errors, masks, fits etc.
    #     Only the original header, the dispersion (via the header), and the fluxden
    #     are stored.
    #
    #     Parameters
    #     ----------
    #     filename : str
    #         A string providing the path and filename for the fits file.
    #
    #     Raises
    #     ------
    #     ValueError
    #         Raises an error when the filename exists and overwrite = False
    #     """
    #
    #     df = pd.DataFrame(np.array([self.dispersion, self.fluxden]).T,
    #                       columns=['dispersion', 'fluxden'])
    #
    #     if self.fluxden_err is not None:
    #         df['fluxden_err'] = self.fluxden_err
    #
    #     hdu = fits.PrimaryHDU(df)
    #
    #     if self.fits_header is not None:
    #         hdu.header = self.fits_header
    #
    #
    #     hdu.header['FLUXDEN_UNIT'] = self.unit[1].to_string(format='fits')
    #     hdu.header['DISP_UNIT'] = self.unit[0].to_string(format='fits')
    #
    #     hdu.header['HISTORY'] = '1D spectrum saved in SpecOneD format'
    #
    #     if comment:
    #         hdu.header['HISTORY'] = comment
    #
    #     hdul = fits.HDUList([hdu])
    #
    #     try:
    #         hdul.writeto(filename, overwrite=overwrite)
    #     except:
    #         raise ValueError("Spectrum could not be saved. Maybe a file with the same name already exists and overwrite is False")

    # def override_raw(self):
    #     """ Override the raw_dispersion, raw_fluxden and raw_fluxden_err
    #     variables in the SpecOneD class with the current dispersion, fluxden and
    #     fluxden_err values.
    #     """
    #
    #     self.raw_dispersion = self.dispersion
    #     self.raw_fluxden = self.fluxden
    #     self.raw_fluxden_err = self.fluxden_err
    #
    # def restore(self):
    #     """ Override the dispersion, fluxden and fluxden_err
    #     variables in the SpecOneD class with the raw_dispersion, raw_fluxden and
    #     raw_fluxden_err values.
    #     """
    #
    #     self.dispersion = self.raw_dispersion
    #     self.fluxden = self.raw_fluxden
    #     self.fluxden_err = self.raw_fluxden_err
    #     self.reset_mask()
    #
    # def add(self, secondary_spectrum, copy_header='first', force=True,
    #         copy_flux_err = 'No', method='interpolate'):
    #
    #     """Add the fluxden in the primary and secondary spectra.
    #
    #     Notes
    #     -----
    #     Users should be aware that in order to add the fluxden of the two spectra,
    #     the dispersions of the spectra need to be matched, see match_dispersions
    #     and beware of the caveats of dispersion interpolation.
    #
    #     Parameters
    #     ----------
    #     secondary_spectrum : obj:`SpecOneD`
    #         Secondary spectrum
    #     copy_header : 'str'
    #         A string indicating whether the primary('first') spectrum header or
    #         the secondary('last') spectrum header should be copied to the
    #         resulting spectrum.
    #     force : boolean
    #         The boolean sets whether the dispersions are matched if only
    #         partial overlap between the spectral dispersions exist.
    #     inplace : boolean
    #         The boolean indicates whether the resulting spectrum will overwrite
    #         the primary spectrum or whether it will be returned as a new
    #         spectrum argument by the method.
    #
    #     Returns
    #     -------
    #     obj:`SpecOneD`
    #         Returns a SpecOneD object in the default case, "inplace==False".
    #     """
    #
    #     self.check_units(secondary_spectrum)
    #     # fluxden error needs to be taken care of
    #     # needs to be tested
    #
    #     s_one = self.copy()
    #     s_two = secondary_spectrum.copy()
    #
    #
    #     if not np.array_equal(s_one.dispersion, s_two.dispersion):
    #         print("Warning: Dispersion does not match.")
    #         print("Warning: Flux will be interpolated/resampled.")
    #
    #         s_one.match_dispersions(s_two, force=force,
    #                                method=method)
    #
    #     new_flux = s_one.flux + s_two.flux
    #
    #     if copy_flux_err == 'No' and isinstance(s_one.flux_err,
    #                                             np.ndarray) and \
    #             isinstance(s_two.flux_err, np.ndarray):
    #         new_flux_err = np.sqrt(s_one.flux_err**2 + s_two.flux_err**2)
    #     elif copy_flux_err == 'first':
    #         new_flux_err = s_one.flux_err
    #     elif copy_flux_err == 'second':
    #         new_flux_err = s_two.flux_err
    #     else:
    #         new_flux_err = None
    #
    #     if copy_header == 'first':
    #         new_header = s_one.header
    #     elif copy_header == 'last':
    #         new_header = s_two.header
    #
    #
    #     return SpecOneD(dispersion=s_one.dispersion,
    #                     fluxden=new_flux,
    #                     fluxden_err=new_flux_err,
    #                     header=new_header,
    #                     unit=s_one.unit,
    #                     mask=s_one.mask)
    #
    # def subtract(self, secondary_spectrum, copy_header='first', force=True,
    #              copy_flux_err = 'No', method='interpolate'):
    #     """Subtract the fluxden of the secondary spectrum from the primary
    #     spectrum.
    #
    #     Notes
    #     -----
    #     Users should be aware that in order to subtract the fluxden,
    #     the dispersions of the spectra need to be matched, see match_dispersions
    #     and beware of the caveats of dispersion interpolation.
    #
    #     Parameters
    #     ----------
    #     secondary_spectrum : obj:`SpecOneD`
    #         Secondary spectrum
    #     copy_header : 'str'
    #         A string indicating whether the primary('first') spectrum header or
    #         the secondary('last') spectrum header should be copied to the
    #         resulting spectrum.
    #     force : boolean
    #         The boolean sets whether the dispersions are matched if only
    #         partial overlap between the spectral dispersions exist.
    #     inplace : boolean
    #         The boolean indicates whether the resulting spectrum will overwrite
    #         the primary spectrum or whether it will be returned as a new
    #         spectrum argument by the method.
    #
    #     Returns
    #     -------
    #     obj:`SpecOneD`
    #         Returns a SpecOneD object in the default case, "inplace==False".
    #     """
    #
    #     self.check_units(secondary_spectrum)
    #     # fluxden error needs to be taken care of
    #     # needs to be tested
    #     # check for negative values?
    #
    #     s_one = self.copy()
    #     s_two = secondary_spectrum.copy()
    #
    #     if not np.array_equal(s_one.dispersion, s_two.dispersion):
    #         print ("Warning: Dispersion does not match.")
    #         print ("Warning: Flux will be interpolated.")
    #
    #         s_one.match_dispersions(s_two, force=force,
    #                                method=method)
    #
    #     new_flux = s_one.flux - s_two.flux
    #
    #     if copy_flux_err == 'No' and isinstance(s_one.flux_err,
    #                                             np.ndarray) and \
    #             isinstance(s_two.flux_err, np.ndarray):
    #         new_flux_err = np.sqrt(s_one.flux_err**2 + s_two.flux_err**2)
    #     elif copy_flux_err == 'first':
    #         new_flux_err = s_one.flux_err
    #     elif copy_flux_err == 'second':
    #         new_flux_err = s_two.flux_err
    #     else:
    #         new_flux_err = None
    #
    #     if copy_header == 'first':
    #         new_header = s_one.header
    #     elif copy_header == 'last':
    #         new_header = s_two.header
    #
    #
    #     return SpecOneD(dispersion=s_one.dispersion,
    #                     fluxden=new_flux,
    #                     fluxden_err=new_flux_err,
    #                     header=new_header,
    #                     unit=s_one.unit,
    #                     mask=s_one.mask)
    #
    # def divide(self, secondary_spectrum, copy_header='first', force=True,
    #            copy_flux_err = 'No', method='interpolate'):
    #
    #     """Divide the fluxden of the primary spectrum by the fluxden
    #     of the secondary spectrum.
    #
    #     Notes
    #     -----
    #     Users should be aware that in order to divide the fluxden of the two
    #     spectra, the dispersions of the spectra need to be matched,
    #     (see match_dispersions) and beware of the caveats of dispersion
    #     interpolation.
    #
    #     Parameters
    #     ----------
    #     secondary_spectrum : obj: SpecOneD
    #         Secondary spectrum
    #     copy_header : 'str'
    #         A string indicating whether the primary('first') spectrum header or
    #         the secondary('last') spectrum header should be copied to the
    #         resulting spectrum.
    #     force : boolean
    #         The boolean sets whether the dispersions are matched if only
    #         partial overlap between the spectral dispersions exist.
    #     inplace : boolean
    #         The boolean indicates whether the resulting spectrum will overwrite
    #         the primary spectrum or whether it will be returned as a new
    #         spectrum argument by the method.
    #
    #     Returns
    #     -------
    #     obj: SpecOneD
    #         Returns a SpecOneD object in the default case, "inplace==False".
    #     """
    #
    #     self.check_units(secondary_spectrum)
    #
    #     s_one = self.copy()
    #     s_two = secondary_spectrum.copy()
    #
    #     if not np.array_equal(self.dispersion, s_two.dispersion):
    #         print ("Warning: Dispersion does not match.")
    #         print ("Warning: Flux will be interpolated.")
    #
    #         s_one.match_dispersions(s_two, force=force,
    #                                match_secondary=False, method=method)
    #
    #     new_flux = s_one.flux / s_two.flux
    #     print(copy_flux_err)
    #     if copy_flux_err == 'No' and isinstance(s_one.flux_err,
    #                                            np.ndarray) and \
    #             isinstance(s_two.flux_err, np.ndarray):
    #             new_flux_err = np.sqrt( (s_one.flux_err/ s_two.flux)**2  + (new_flux/s_two.flux*s_two.flux_err)**2 )
    #     elif copy_flux_err == 'first':
    #         new_flux_err = s_one.flux_err
    #     elif copy_flux_err == 'second':
    #         new_flux_err = s_two.flux_err
    #     else:
    #         new_flux_err = None
    #
    #     print(new_flux_err, s_one.flux_err)
    #
    #     if copy_header == 'first':
    #         new_header = s_one.header
    #     elif copy_header == 'last':
    #         new_header = s_two.header
    #
    #     return SpecOneD(dispersion=s_one.dispersion,
    #                     fluxden=new_flux,
    #                     fluxden_err=new_flux_err,
    #                     header=new_header,
    #                     unit=s_one.unit,
    #                     mask=s_one.mask)
    #
    # def multiply(self, secondary_spectrum, copy_header='first', force=True,
    #              copy_flux_err = 'No', method='interpolate'):
    #
    #     """Multiply the fluxden of primary spectrum with the secondary spectrum.
    #
    #     Notes
    #     -----
    #     Users should be aware that in order to add the fluxden of the two spectra,
    #     the dispersions of the spectra need to be matched, see match_dispersions
    #     and beware of the caveats of dispersion interpolation.
    #
    #     Parameters
    #     ----------
    #     secondary_spectrum : obj:`SpecOneD`
    #         Secondary spectrum
    #     copy_header : 'str'
    #         A string indicating whether the primary('first') spectrum header or
    #         the secondary('last') spectrum header should be copied to the
    #         resulting spectrum.
    #     force : boolean
    #         The boolean sets whether the dispersions are matched if only
    #         partial overlap between the spectral dispersions exist.
    #     inplace : boolean
    #         The boolean indicates whether the resulting spectrum will overwrite
    #         the primary spectrum or whether it will be returned as a new
    #         spectrum argument by the method.
    #
    #     Returns
    #     -------
    #     obj;`SpecOneD`
    #         Returns a SpecOneD object in the default case, "inplace==False".
    #     """
    #
    #     self.check_units(secondary_spectrum)
    #
    #     s_one = self.copy()
    #     s_two = secondary_spectrum.copy()
    #
    #     if not np.array_equal(s_one.dispersion, s_two.dispersion):
    #         print ("Warning: Dispersion does not match.")
    #         print ("Warning: Flux will be interpolated.")
    #
    #         s_one.match_dispersions(s_two, force=force,
    #                                method=method)
    #
    #     new_flux = s_one.flux * s_two.flux
    #
    #     if copy_flux_err == 'No' and isinstance(s_one.flux_err,
    #                                             np.ndarray) and \
    #             isinstance(s_two.flux_err, np.ndarray):
    #         new_flux_err = np.sqrt(s_two.flux**2 * s_one.flux_err**2 + s_one.flux**2 * s_two.flux_err**2)
    #     elif copy_flux_err == 'first':
    #         new_flux_err = s_one.flux_err
    #     elif copy_flux_err == 'second':
    #         new_flux_err = s_two.flux_err
    #     else:
    #         new_flux_err = None
    #
    #     if copy_header == 'first':
    #         new_header = s_one.header
    #     elif copy_header == 'last':
    #         new_header = s_two.header
    #
    #     return SpecOneD(dispersion=s_one.dispersion,
    #                     fluxden=new_flux,
    #                     fluxden_err=new_flux_err,
    #                     header=new_header,
    #                     unit=s_one.unit,
    #                     mask=s_one.mask)
    #
    #
    # def pypeit_plot(self, show_fluxden_err=False, show_raw_fluxden=False,
    #                 mask_values=False, ymax=None):
    #
    #     """Plot the spectrum assuming it is a pypeit spectrum
    #
    #     """
    #
    #     if mask_values:
    #         mask = self.mask
    #     else:
    #         mask = np.ones(self.dispersion.shape, dtype=bool)
    #
    #     self.fig, self.ax = plt.subplots(nrows=1, ncols=1, figsize=(15,7), dpi = 140)
    #     self.fig.subplots_adjust(left=0.09, right=0.97, top=0.89, bottom=0.16)
    #
    #     # Plot the Spectrum
    #     self.ax.axhline(y=0.0, linewidth=1.5, color='k', linestyle='--')
    #
    #     # Add second axis to plot telluric model
    #
    #
    #     if show_fluxden_err:
    #         self.ax.plot(self.dispersion[mask], self.fluxden_err[mask], 'grey',
    #                      lw=1, label='Flux Error')
    #     if show_raw_fluxden:
    #         self.ax.plot(self.raw_dispersion[mask], self.raw_fluxden[mask],
    #                      'grey', lw=3, label='Raw Flux')
    #
    #     self.ax.plot(self.dispersion[mask], self.fluxden[mask], 'k',
    #                  linewidth=1, label='Flux')
    #
    #     if self.unit=='f_lam':
    #         self.ax.set_xlabel(r'$\rm{Wavelength}\ [\rm{\AA}]$', fontsize=15)
    #         self.ax.set_ylabel(r'$\rm{Flux}\ f_{\lambda}\ [\rm{erg}\,\rm{s}^{'
    #                            r'-1}\,\rm{cm}^{-2}\,\rm{\AA}^{-1}]$', fontsize=15)
    #
    #     elif self.unit =='f_nu':
    #         self.ax.set_xlabel(r'$\rm{Frequency}\ [\rm{Hz}]$', fontsize=15)
    #         self.ax.set_ylabel(r'$\rm{Flux}\ f_{\nu}\ [\rm{erg}\,\rm{s}^{'
    #                            r'-1}\,\rm{cm}^{-2}\,\rm{Hz}^{-1}]$', fontsize=15)
    #
    #     elif self.unit =='f_loglam':
    #         self.ax.set_xlabel(r'$\log\rm{Wavelength}\ [\log\rm{\AA}]$', fontsize=15)
    #         self.ax.set_ylabel(r'$\rm{Flux}\ f_{\lambda}\ [\rm{erg}\,\rm{s}^{'
    #                            r'-1}\,\rm{cm}^{-2}\,(\log\rm{\AA})^{-1}]$', fontsize=15)
    #
    #     else:
    #         raise ValueError("Unrecognized units")
    #
    #
    #     # Add OBJ model if it exists
    #
    #
    #
    #     lim_spec = self.copy()
    #     lim_spec.restore()
    #     lim_spec = lim_spec.mask_sn(5)
    #     lim_spec = lim_spec.sigmaclip_flux(3, 3)
    #     ylim_min = 0
    #     if ymax == None:
    #         ylim_max = lim_spec.flux[lim_spec.mask].max()
    #     else:
    #         ylim_max = ymax
    #     self.ax.set_ylim(ylim_min, ylim_max)
    #
    #     self.ax.legend()
    #     plt.show()
    #
    # def calculate_snr(self):
    #
    #     pass

    # def redshift(self, z, inplace=False):
    #     # TODO Take care of fluxden conversion here as well!
    #     # TODO Taken care of fluxden conversion, check how IVAR behaves
    #
    #     if inplace:
    #         self.dispersion = self.dispersion * (1.+z)
    #         # self.fluxden /= (1.+z)
    #         # self.fluxden_err /= (1.+z)
    #     else:
    #         spec = self.copy()
    #         spec.dispersion = spec.dispersion * (1.+z)
    #         # spec.fluxden /= (1.+z)
    #         # spec.fluxden_err /= (1.+z)
    #
    #         return spec


    # def to_wavelength(self):
    #     """ Convert the spectrum from fluxden per frequency to fluxden per
    #     wavenlength.
    #
    #     This method converts the fluxden from erg/s/cm^2/Hz to
    #     erg/s/cm^2/Angstroem and the dispersion accordingly from Hz to
    #     Angstroem.
    #
    #     Raises
    #     ------
    #     ValueError
    #         Raises an error, if the fluxden is already in wavelength.
    #     """
    #
    #     if self.unit == 'f_lam':
    #         raise ValueError('Dispersion is arealdy in wavelength')
    #     elif self.unit == 'f_nu':
    #         self.fluxden = self.fluxden * self.dispersion ** 2 / (const.c.value * 1e+10)
    #         self.dispersion = (const.c.value * 1e+10) / self.dispersion
    #
    #         self.fluxden = np.flip(self.fluxden, axis=0)
    #         self.dispersion = np.flip(self.dispersion, axis=0)
    #
    #     elif self.unit == 'f_loglam':
    #         self.dispersion = np.exp(self.dispersion)
    #         self.fluxden = self.fluxden / self.dispersion
    #     else:
    #         raise ValueError('Spectrum unit not recognized: ', self.unit)
    #
    #     self.unit = 'f_lam'

    # def to_frequency(self):
    #     """ Convert the spectrum from fluxden per wavelength to fluxden per
    #     frequency.
    #
    #     This method converts the fluxden from erg/s/cm^2/Angstroem to
    #     erg/s/cm^2/Hz and the dispersion accordingly from Angstroem to Hz.
    #
    #     Raises
    #     ------
    #     ValueError
    #         Raises an error, if the fluxden is already in frequency.
    #     """
    #
    #     if self.unit == 'f_nu':
    #         raise ValueError('Dispersion is already in frequency.')
    #
    #     elif self.unit == 'f_lam':
    #         self.fluxden = self.fluxden * self.dispersion ** 2 / (const.c.value * 1e+10)
    #         self.dispersion = (const.c.value * 1e+10) / self.dispersion
    #
    #         self.fluxden = np.flip(self.fluxden, axis=0)
    #         self.dispersion = np.flip(self.dispersion, axis=0)
    #
    #     elif self.unit == 'f_loglam':
    #         self.to_wavelength()
    #         self.to_frequency()
    #     else:
    #         raise ValueError('Spectrum unit not recognized: ', self.unit)
    #
    #     self.unit = 'f_nu'


    # INCORRECT RESULTS, TESTED AGAINST SDSS QUASAR SPECTRUM
    # def calculate_passband_st_magnitude(self, passband, force=False,
    #                                     match_method='resample'):
    #     # This function is written for passbands in quantum efficiency
    #     # Therefore, the (h*nu)^-1 term is not included in the integral
    #
    #     spec = self.copy()
    #     spec.to_fluxden_per_unit_wavelength_cgs()
    #
    #     spectrum_flux = spec.calculate_passband_flux(passband, force=force,
    #                                                  match_method=match_method)
    #
    #     flat_flux = 3.631e-9 * np.ones_like(passband.dispersion) *  \
    #                 u.erg/u.s/u.cm**2/u.AA
    #     # flat_flux = 1.4454397707459234e-9 * np.ones_like(passband.dispersion) * \
    #     #             u.erg / u.s / u.cm ** 2 / u.AA
    #
    #     passband_flux = np.trapz(flat_flux * passband.fluxden,
    #                              passband.dispersion * passband.dispersion_unit)
    #
    #     print(spectrum_flux , passband_flux)
    #
    #     return -2.5 * np.log10(spectrum_flux / passband_flux)


    # def medianclip_flux(self, sigma=3, binsize=11, inplace=False):
    #     """
    #     Quick hack for sigma clipping using a running median
    #     :param sigma:
    #     :param binsize:
    #     :param inplace:
    #     :return:
    #     """
    #
    #     flux = self.fluxden.copy()
    #     flux_err = self.fluxden_err.copy()
    #
    #     median = medfilt(flux, kernel_size=binsize)
    #
    #     diff = np.abs(flux-median)
    #
    #     mask = self.mask.copy()
    #
    #     mask[diff > sigma * flux_err] = 0
    #
    #     if inplace:
    #         self.mask = mask
    #     else:
    #         spec = self.copy()
    #         spec.mask = mask
    #
    #         return spec
    #
    #
    # def sigmaclip_flux(self, low=3, up=3, binsize=120, niter=5, inplace=False):
    #
    #     hbinsize = int(binsize/2)
    #
    #     flux = self.fluxden
    #     dispersion = self.dispersion
    #
    #
    #     mask_index = np.arange(dispersion.shape[0])
    #
    #     # loop over sigma-clipping iterations
    #     for jdx in range(niter):
    #
    #         n_mean = np.zeros(flux.shape[0])
    #         n_std = np.zeros(flux.shape[0])
    #
    #         # calculating mean and std arrays
    #         for idx in range(len(flux[:-binsize])):
    #
    #             # fluxden subset
    #             f_bin = flux[idx:idx+binsize]
    #
    #             # calculate mean and std
    #             mean = np.median(f_bin)
    #             std = np.std(f_bin)
    #
    #             # set array value
    #             n_mean[idx+hbinsize] = mean
    #             n_std[idx+hbinsize] = std
    #
    #         # fill zeros at end and beginning
    #         # with first and last values
    #         n_mean[:hbinsize] = n_mean[hbinsize]
    #         n_mean[-hbinsize:] = n_mean[-hbinsize-1]
    #         n_std[:hbinsize] = n_std[hbinsize]
    #         n_std[-hbinsize:] = n_std[-hbinsize-1]
    #
    #         # create index array with included pixels ("True" values)
    #         mask = (flux-n_mean < n_std*up) & (flux-n_mean > -n_std*low)
    #         mask_index = mask_index[mask]
    #
    #         # mask the fluxden for the next iteration
    #         flux = flux[mask]
    #
    #     mask = np.zeros(len(self.mask), dtype='bool')
    #
    #     # mask = self.mask
    #     mask[:] = False
    #     mask[mask_index] = True
    #     mask = mask * np.array(self.mask, dtype=bool)
    #
    #     if inplace:
    #         self.mask = mask
    #     else:
    #         spec = self.copy()
    #         spec.mask = mask
    #
    #         return spec

# def combine_spectra(filenames, method='average', file_format='fits'):
#
#     s_list = []
#
#     for filename in filenames:
#         spec = SpecOneD()
#         if file_format == 'fits':
#             spec.read_from_fits(filename)
#             print(spec)
#         else:
#             raise NotImplementedError('File format not understood')
#
#         s_list.append(spec)
#
#     # TODO Test if all spectra are in same unit, if not convert
#     print(s_list)
#     # Resample all spectra onto slightly reduced dispersion of first spectrum
#     disp = s_list[0].dispersion[5:-5]
#     for spec in s_list:
#         spec.resample(disp, inplace=True)
#
#     comb_flux = np.zeros(len(disp))
#     comb_fluxerr = np.zeros(len(disp))
#
#     N = float(len(filenames))
#
#     if method == 'average':
#         for spec in s_list:
#             comb_flux += spec.fluxden
#             comb_fluxerr += spec.fluxden_err ** 2 / N ** 2
#
#         comb_flux = comb_flux / N
#         comb_fluxerr = np.sqrt(comb_fluxerr)
#
#         comb_spec = SpecOneD(dispersion=disp, fluxden=comb_flux,
#                              fluxden_err=comb_fluxerr, unit='f_lam',
#                              )
#         return comb_spec
#
#     else:
#         raise NotImplementedError(
#             'Selected method for combining spectra is '
#             'not implemented. Implemented methods: '
#             'average')
#
# def pypeit_spec1d_plot(filename, show_flux_err=True, mask_values=False,
#                         ex_value='OPT', show='fluxden', smooth=None):
#
#     # plot_setup
#     fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15, 7), dpi=140)
#     fig.subplots_adjust(left=0.09, right=0.97, top=0.89, bottom=0.16)
#
#     # Plot 0-line
#     ax.axhline(y=0.0, linewidth=1.5, color='k', linestyle='--')
#
#     # read the pypeit echelle file
#     hdu = fits.open(filename)
#
#     n_spec = hdu[0].header['NSPEC']
#     target = hdu[0].header['TARGET']
#     # instrument = hdu[0].header['INSTRUME']
#
#
#     ylim_min = []
#     ylim_max = []
#
#     for order in range(1, n_spec+1):
#
#         if order % 2 == 0:
#             color = vermillion
#         else:
#             color = dblue
#
#         wavelength = hdu[order].data['{}_WAVE'.format(ex_value)]
#         if mask_values:
#             mask = hdu[order].data['{}_MASK'.format(ex_value)]
#
#         else:
#             mask = np.ones_like(wavelength, dtype=bool)
#
#         # masking the value and wavelength = 0
#         wave_mask = wavelength > 1.0
#
#         mask = np.logical_and(mask, wave_mask)
#
#
#         if '{}_FLAM'.format(ex_value) in hdu[order].columns.names:
#             flux = hdu[order].data['{}_FLAM'.format(ex_value)]
#             flux_ivar = hdu[order].data['{}_FLAM_IVAR'.format(ex_value)]
#             flux_sigma = hdu[order].data['{}_FLAM_SIG'.format(ex_value)]
#         else:
#             counts = hdu[order].data['{}_COUNTS'.format(ex_value)]
#             counts_ivar = hdu[order].data['{}_COUNTS_IVAR'.format(ex_value)]
#             counts_sigma = hdu[order].data['{}_COUNTS_SIG'.format(ex_value)]
#             show = 'counts'
#
#         if show == 'counts':
#             if smooth is not None:
#                 counts = convolve(counts, Box1DKernel(smooth))
#                 counts_sigma /= np.sqrt(smooth)
#
#             ax.plot(wavelength[mask], counts[mask], color=color)
#             yy = counts[mask]
#             if show_flux_err:
#                 ax.plot(wavelength[mask], counts_sigma[mask], color=color,
#                         alpha=0.5)
#
#         elif show == 'fluxden':
#             if smooth is not None:
#                 flux = convolve(flux, Box1DKernel(smooth))
#                 flux_sigma /= np.sqrt(smooth)
#
#             ax.plot(wavelength[mask], flux[mask], color=color, alpha=0.8)
#             yy = flux[mask]
#             if show_flux_err:
#                 ax.plot(wavelength[mask], flux_sigma[mask], color=color,
#                         alpha=0.5)
#         else:
#             raise ValueError('Variable input show = {} not '
#                              'understood'.format(show))
#
#         percentiles = np.percentile(yy, [16, 84])
#         median = np.median(yy)
#         delta = np.abs(percentiles[1] - median)
#         # print('delta', delta)
#         # print('percentiles', percentiles)
#
#         ylim_min.append(-0.5 * median)
#         ylim_max.append(4 * percentiles[1])
#
#
#     if show == 'counts':
#         ax.set_ylabel(
#             r'$\rm{Counts}\ [\rm{ADU}]$', fontsize=15)
#     elif show == 'fluxden':
#         ax.set_ylabel(
#             r'$\rm{Flux}\ f_{\lambda}\ [\rm{erg}\,\rm{s}^{-1}\,\rm{cm}^{-2}\,'
#             r'\rm{\AA}^{-1}]$',
#             fontsize=15)
#     else:
#         raise ValueError('Variable input show = {} not '
#                          'understood'.format(show))
#
#     ax.set_xlabel(r'$\rm{Wavelength}\ [\rm{\AA}]$', fontsize=15)
#
#     ax.set_ylim(min(ylim_min), max(ylim_max))
#
#     # plt.title(r'{} {}'.format(target, instrument))
#
#     plt.legend()
#
#     plt.show()
#
#
# def pypeit_multi_plot(filenames, show_flux_err=True, show_tellurics=False,
#     mask_values=False, smooth=None, ymax=None):
#     """Plot the spectrum assuming it is a pypeit spectrum
#
#      """
#
#     fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15, 7), dpi=140)
#     fig.subplots_adjust(left=0.09, right=0.97, top=0.89, bottom=0.16)
#
#     # Plot 0-line
#     ax.axhline(y=0.0, linewidth=1.5, color='k', linestyle='--')
#
#     max_limits = []
#
#     for idx, filename in enumerate(filenames):
#
#         color = color_list[idx]
#
#         spec = SpecOneD()
#         spec.read_pypeit_fits(filename)
#
#         if mask_values:
#             mask = spec.mask
#         else:
#             mask = np.ones(spec.dispersion.shape, dtype=bool)
#
#         if smooth is not None and type(smooth) is int:
#             spec.smooth(smooth, inplace=True)
#
#         label = filename
#
#
#         # Add second axis to plot telluric model
#         if show_tellurics is True:
#             telluric = spec.telluric[mask] / spec.telluric.max() * np.median(
#                 spec.fluxden[mask]) * 2.5
#             ax.plot(spec.dispersion[mask], telluric[mask],
#                          label='Telluric', color=color, alpha=0.5, ls='..')
#
#         if show_flux_err:
#             ax.plot(spec.dispersion[mask], spec.fluxden_err[mask], 'grey',
#                     lw=1, label='Flux Error', color=color, alpha=0.5)
#
#
#         ax.plot(spec.dispersion[mask], spec.fluxden[mask], 'k',
#                 linewidth=1, label=label, color=color)
#
#         # # Add OBJ model if it exists
#         # if hasattr(spec, 'obj_model'):
#         #     ax.plot(spec.dispersion[mask], spec.obj_model, label='Obj '
#         #                                                               'model')
#
#         lim_spec = spec.copy()
#         lim_spec.restore()
#         lim_spec = lim_spec.mask_sn(5)
#         lim_spec = lim_spec.sigmaclip_flux(3, 3)
#
#         max_limits.append(lim_spec.flux[lim_spec.mask].max())
#
#     if spec.unit == 'f_lam':
#         ax.set_xlabel(r'$\rm{Wavelength}\ [\rm{\AA}]$', fontsize=15)
#         ax.set_ylabel(
#             r'$\rm{Flux}\ f_{\lambda}\ [\rm{erg}\,\rm{s}^{-1}\,\rm{cm}^{-2}\,'
#             r'\rm{\AA}^{-1}]$',
#             fontsize=15)
#
#     elif spec.unit == 'f_nu':
#         ax.set_xlabel(r'$\rm{Frequency}\ [\rm{Hz}]$', fontsize=15)
#         ax.set_ylabel(
#             r'$\rm{Flux}\ f_{\nu}\ [\rm{erg}\,\rm{s}^{-1}\,\rm{cm}^{-2}\,'
#             r'\rm{Hz}^{-1}]$',
#             fontsize=15)
#
#     elif spec.unit == 'f_loglam':
#         ax.set_xlabel(r'$\log\rm{Wavelength}\ [\log\rm{\AA}]$',
#                            fontsize=15)
#         ax.set_ylabel(
#             r'$\rm{Flux}\ f_{\lambda}\ [\rm{erg}\,\rm{s}^{-1}\,\rm{cm}^{-2}\,'
#             r'(\log\rm{\AA})^{-1}]$',
#             fontsize=15)
#
#     else:
#         raise ValueError("Unrecognized units")
#
#
#     ylim_min = 0
#     if ymax == None:
#         ylim_max = max(max_limits)
#     else:
#         ylim_max = ymax
#     ax.set_ylim(ylim_min, ylim_max)
#     ax.legend()
#     plt.show()
#
#
# def comparison_plot(spectrum_a, spectrum_b, spectrum_result,
#                     show_flux_err=True):
#
#     fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(15,7), dpi=140)
#     fig.subplots_adjust(left=0.09, right=0.97, top=0.89, bottom=0.16)
#
#     ax1.plot(spectrum_a.dispersion, spectrum_a.flux, color='k')
#     ax1.plot(spectrum_b.dispersion, spectrum_b.flux, color='r')
#
#     ax2.plot(spectrum_result.dispersion, spectrum_result.flux, color='k')
#
#     if show_flux_err:
#         ax1.plot(spectrum_a.dispersion, spectrum_a.flux_err, 'grey', lw=1)
#         ax1.plot(spectrum_b.dispersion, spectrum_b.flux_err, 'grey', lw=1)
#         ax2.plot(spectrum_result.dispersion, spectrum_result.flux_err, 'grey', lw=1)
#
#     if spectrum_result.unit=='f_lam':
#         ax2.set_xlabel(r'$\rm{Wavelength}\ [\rm{\AA}]$', fontsize=15)
#         ax1.set_ylabel(r'$\rm{Flux}\ f_{\lambda}\ [\rm{erg}\,\rm{s}^{-1}\,\rm{cm}^{2}\,\rm{\AA}^{-1}]$', fontsize=15)
#         ax2.set_ylabel(r'$\rm{Flux}\ f_{\lambda}\ [\rm{erg}\,\rm{s}^{-1}\,\rm{cm}^{2}\,\rm{\AA}^{-1}]$', fontsize=15)
#
#     elif spectrum_result.unit =='f_nu':
#         ax2.set_xlabel(r'$\rm{Frequency}\ [\rm{Hz}]$', fontsize=15)
#         ax1.set_ylabel(r'$\rm{Flux}\ f_{\nu}\ [\rm{erg}\,\rm{s}^{-1}\,\rm{cm}^{2}\,\rm{Hz}^{-1}]$', fontsize=15)
#         ax2.set_ylabel(r'$\rm{Flux}\ f_{\nu}\ [\rm{erg}\,\rm{s}^{-1}\,\rm{cm}^{2}\,\rm{Hz}^{-1}]$', fontsize=15)
#
#     else :
#         raise ValueError("Unrecognized units")
#
#     plt.show()
#
# class FlatSpectrum(SpecOneD):
#
#     def __init__(self, flat_dispersion, unit='f_nu'):
#
#         try:
#             flat_dispersion = np.array(flat_dispersion)
#             if flat_dispersion.ndim != 1:
#                 raise ValueError("Flux dimension is not 1")
#         except ValueError:
#             print("Flux could not be converted to 1D ndarray")
#
#         if unit == 'f_lam':
#             fill_value = 3.631e-9
#         if unit == 'f_nu':
#             fill_value = 3.631e-20
#
#         self.flux = np.full(flat_dispersion.shape, fill_value)
#         self.dispersion = flat_dispersion
