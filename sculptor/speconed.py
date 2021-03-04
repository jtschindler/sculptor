#!/usr/bin/env python

"""
This module introduces the SpecOneD class, it's functions and the related
FlatSpectrum and QuasarSpectrum classes.
The main purpose of the SpecOneD class and it's children classes is to provide
python functionality for the manipulation of 1D spectral data in astronomy.


Notes
-----
    The documentation for this module is not yet completed.


Attributes
----------
datadir : str
    The path to the data directory formatted as  a string.
"""

import os
import numpy as np
import scipy as sp
import pandas as pd

from astropy.io import fits
from astropy import constants as const
from astropy import units as u
from astropy.convolution import convolve, Gaussian1DKernel, Box1DKernel

from scipy.signal import medfilt

import matplotlib.pyplot as plt

from numpy.polynomial import Legendre, Chebyshev, Polynomial


black = (0, 0, 0)
orange = (230/255., 159/255., 0)
blue = (86/255., 180/255., 233/255.)
green = (0, 158/255., 115/255.)
yellow = (240/255., 228/255., 66/255.)
dblue = (0, 114/255., 178/255.)
vermillion = (213/255., 94/255., 0)
purple = (204/255., 121/255., 167/255.)

color_list = [vermillion, dblue, green, purple, yellow, orange, blue]

datadir = os.path.split(__file__)[0]
datadir = os.path.split(datadir)[0] + '/data/'


def gaussian(x, amp, cen, sigma, shift):
    """ 1-D Gaussian function"""
    central = cen + shift

    return (amp / (np.sqrt(2*np.pi) * sigma)) * np.exp(-(x-central)**2 / (2*sigma**2))


class SpecOneD(object):

    """The SpecOneD class stores 1D spectral information and allows it's
    manipulation with built-in functions.

    Attributes
    ----------
    raw_dispersion : ndarray
        A 1D array containing the original spectral dispersion data in 'float'
        type.
    raw_fluxden : ndarray
        A 1D array containing the original spectral fluxden data in 'float' type.
    raw_fluxden_err : ndarray
        A 1D array containing the original spectral fluxden error data in 'float'
        type.
    dispersion : ndarray
        The 1D array containing the sp`ectral dispersion data in 'float' type.
    fluxden : ndarray
        A 1D array containing the spectral fluxden data in 'float' type.
    fluxden_err : ndarray
        A 1D array containing the spectral fluxden error data in 'float' type.
    mask : ndarray
        A 1D mask array (boolean type).
    unit : str
        A string specifying if the if the fluxden unit is per wavelength 'f_lam'
        or frequency 'f_nu'.
    model_spectrum : obj:
        A lmfit Model object, which allows to fit the spectrum using lmfit
    model_pars : obj:
        A lmfit parameter object containing the parameters for the
        model_spectrum.
    fit_output : obj:
        The resulting fit based on the model_spectrum and the model_pars
    fit_dispersion :
        A 1D array containing the dispersion of the fitted spectrum.
    fit_fluxden : ndarray
        A 1D arrat containing the fluxden values for the fitted spectrum.
    header : obj
        The spectral header object, containing additional data with regard to
        the spectrum.

    """

    def __init__(self, dispersion=None, fluxden=None, fluxden_err=None, 
                 header=None, unit=None, mask=None, raw_dispersion=None, 
                 raw_fluxden=None, raw_fluxden_err=None):
        """The __init__ method for the SpecOneD class

        Parameters
        ----------
        dispersion : ndarray
            A 1D array providing the dispersion data of the spectrum in
            wavelength or frequency.
        fluxden : ndarray
            A 1D array providing the fluxden data for the spectrum.
            lines are supported.
        fluxden_err : ndarray
            A 1D array providing the error on the spectrum fluxden
        header : obj:`dict`
            The header object file for the spectrum. This should be a python
            dictionary or a fits format header file.
        unit : str
            A string defining the unit of the fluxden measurement. Currently
            fluxden density per wavelength or per frequency are supported as the
            following
            options: 'f_lam' for wavelength, 'f_nu' for frequency.
        mask : ndarray
            A 1D boolean array can be specified to provide a mask for the
            spectrum.

        Raises
        ------
        ValueError
            Raises an error when either the dispersion or fluxden dimension is not
            or could not be converted to a 1D ndarray.
        ValueError
            Raises an error when the supplied header is not a dictionary.

        """

        # disperion units need to be in Angstroem

        try:
            if raw_fluxden is None:
                self.raw_fluxden = fluxden
            else:
                self.raw_fluxden = np.array(raw_fluxden)
        except ValueError:
            print("Raw fluxden could not be converted to 1D ndarray")

        try:
            if raw_dispersion is None:
                self.raw_dispersion = dispersion
            else:
                self.raw_dispersion = np.array(raw_dispersion)
                # if dispersion.ndim != 1:
                #     raise ValueError("Flux dimension is not 1")
        except ValueError:
            print("Raw dispersion could not be converted to 1D ndarray")

        if raw_fluxden_err is None:
            self.raw_fluxden_err = fluxden_err
        else:
            self.raw_fluxden_err = np.array(raw_fluxden_err)

        self.fluxden = fluxden
        self.fluxden_err = fluxden_err
        self.dispersion = dispersion

        if mask is not None:
            self.mask = mask
        elif self.dispersion is None:
            self.mask = None
        else:
            self.mask = np.ones(self.dispersion.shape, dtype=bool)

        self.unit = unit
        self.model_spectrum = None
        self.model_pars = None
        self.fit_dispersion = None
        self.fit_fluxden = None
        self.fit_output = None


        if header == None:
            self.header = dict()
        else:
            self.header = header


    def read_pypeit_fits(self, filename, unit='f_lam', exten=1):
        """ Read a 1D fits pypeit fits file to populate the SpecOneD class.

        :param filename: str
            Path/Filename of the pypeit 1D fits file
        :param unit:
            The unit of the fluxden measurement in the fits file. This defaults
            to fluxden per wavelength (erg/s/cm^2/Angstroem)
        """

        # Open the fits file
        try:
            hdu = fits.open(filename)
        except:
            raise ValueError("Filename not found", str(filename))

        self.header = hdu[0].header
        self.unit = unit
        self.dispersion = hdu[exten].data['OPT_WAVE']
        self.raw_dispersion = hdu[exten].data['OPT_WAVE']
        self.fluxden = hdu[exten].data['OPT_FLAM'] * 1e-17
        self.raw_fluxden = hdu[exten].data['OPT_FLAM'] * 1e-17
        self.mask = np.array(hdu[exten].data['OPT_MASK'], dtype=bool)
        self.flux_ivar = hdu[exten].data['OPT_FLAM_IVAR']
        self.fluxden_err = hdu[exten].data['OPT_FLAM_SIG'] * 1e-17
        self.raw_fluxden_err = self.fluxden_err

        # Mask all pixels where the fluxden error is 0
        new_mask = np.ones_like(self.mask, dtype=bool)
        new_mask[self.fluxden_err == 0] = 0
        self.mask = new_mask


        if 'TELLURIC' in hdu[exten].columns.names:
            self.telluric = hdu[exten].data['TELLURIC']
        if 'OBJ_MODEL' in hdu[exten].columns.names:
            self.obj_model = hdu[exten].data['OBJ_MODEL']*1e-17


    def read_from_fits(self, filename, unit='f_lam'):
        """Read a 1D fits file to populate the SpecOneD class.

        Parameters
        ----------
        filename : str
            A string providing the path and filename for the fits file.
        unit : str
            The unit of the fluxden measurement in the fits file. This defaults
            to fluxden per wavelength (erg/s/cm^2/Angstroem)

        Raises
        ------
        ValueError
            Raises an error when the filename could not be read in.
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


        self.unit = unit

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

        self.raw_fluxden = self.fluxden
        self.mask = np.ones(self.dispersion.shape, dtype=bool)
        self.raw_fluxden_err = self.fluxden_err
        self.raw_dispersion = self.dispersion

        self.header = hdu[0].header

    def read_sdss_fits(self, filename, unit='f_lam'):
        """Read a 1D SDSS fits file to populate the SpecOneD class.

        Parameters
        ----------
        filename : str
            A string providing the path and filename for the fits file.
        unit : str
            The unit of the fluxden measurement in the fits file. This defaults
            to fluxden per wavelength (erg/s/cm^2/Angstroem)

        Raises
        ------
        ValueError
            Raises an error when the filename could not be read in.
        """

        # Open the fits file
        try:
            hdu = fits.open(filename)
        except:
            raise ValueError("Filename not found", str(filename))


        self.unit = unit


        self.fluxden = np.array(hdu[1].data['flux'], dtype=np.float64)*1e-17
        self.dispersion = 10**np.array(hdu[1].data['loglam'], dtype=np.float64)
        self.ivar = np.array(hdu[1].data['ivar'], dtype=np.float64)
        self.fluxden_err = 1/np.sqrt(self.ivar)*1e-17

        self.mask = np.ones(self.dispersion.shape, dtype=bool)

        self.header = hdu[0].header


    def save_to_fits(self, filename, comment=None, overwrite = False):
        """Save a SpecOneD spectrum to a fits file.

        Note: This save function does not store flux_errors, masks, fits etc.
        Only the original header, the dispersion (via the header), and the fluxden
        are stored.

        Parameters
        ----------
        filename : str
            A string providing the path and filename for the fits file.

        Raises
        ------
        ValueError
            Raises an error when the filename exists and overwrite = False
        """


        hdu  = fits.PrimaryHDU(self.fluxden)
        hdu.header = self.header

        # Update header information
        crval = self.dispersion[0]
        cd = self.dispersion[1]-self.dispersion[0]
        crpix = 1

        hdu.header['CRVAL1'] = crval
        hdu.header['CD1_1'] = cd
        hdu.header['CDELT1'] = cd
        hdu.header['CRPIX1'] = crpix

        hdu.header['HISTORY'] = '1D spectrum generated with SpecOneD'

        if comment:
            hdu.header['HISTORY'] = comment

        hdul = fits.HDUList([hdu])

        try:
            hdul.writeto(filename, overwrite = overwrite)
        except:
            raise ValueError("Spectrum could not be saved. Maybe a file with the same name already exists and overwrite is False")

    def save_to_hdf(self, filename, comment=None, overwrite=False):

        df = pd.DataFrame(np.array([self.dispersion, self.fluxden]).T, columns=['dispersion', 'fluxden'])

        if self.fluxden_err is not None:
            df['fluxden_err'] = self.fluxden_err

        df.to_hdf(filename, 'data')


    def read_from_hdf(self, filename):

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
        self.unit = 'f_lam'
        self.reset_mask()
        self.override_raw()



    def save_to_csv(self, filename, format='linetools'):

        data = [self.dispersion, self.fluxden]

        if self.fluxden_err is not None:
            data.append(self.fluxden_err)


        if hasattr(self, 'telluric'):
            data.append(self.telluric)
        if hasattr(self, 'obj_model'):
            data.append(self.obj_model)

        if format == 'linetools':

            column_names = ['wave', 'fluxden']
            if self.fluxden_err is not None:
                column_names.append('error')

        else:

            column_names = ['wavelength', 'fluxden']
            if self.fluxden_err is not None:
                column_names.append('flux_error')

        if hasattr(self, 'telluric'):
            column_names.append('telluric')
        if hasattr(self, 'obj_model'):
            column_names.append('obj_model')

        df = pd.DataFrame(np.array(data).T, columns=column_names)


        df.to_csv(filename, index=False)




    def reset_mask(self):
        """Reset the spectrum mask by repopulating it with a 1D array of
        boolean 1 = "True" values.
        """

        self.mask = np.ones(self.dispersion.shape, dtype=bool)


    def mask_sn(self, signal_to_noise_limit, inplace=False):
        """

        :param signal_to_noise_limit:
        :param inplace:
        :return:
        """

        mask_index = self.fluxden / self.fluxden_err > signal_to_noise_limit

        new_mask = np.zeros(self.dispersion.shape, dtype=bool)

        new_mask[mask_index] = 1

        new_mask = new_mask * np.array(self.mask, dtype=bool)

        if inplace:
            self.mask = new_mask

        else:
            spec = self.copy()
            spec.mask = new_mask
            return spec

    def get_ivar_from_sigma(self):

        if self.fluxden_err is None:
            self.flux_ivar = None
        else:
            ivar = np.zeros_like(self.fluxden_err)
            valid = self.fluxden_err > 0
            ivar[valid] = 1. / (self.fluxden_err[valid]) ** 2

            self.flux_ivar = ivar

    def get_sigma_from_ivar(self):

        if self.flux_ivar is None:
            self.fluxden_err = None
        else:
            sigma = np.zeros_like(self.flux_ivar)
            valid = self.flux_ivar > 0
            sigma[valid] = 1./np.sqrt(self.flux_ivar[valid])

            self.fluxden_err = sigma



    def copy(self):
        """Create a new SpecOneD instance, populate it with the values
        from the active spectrum and return it.


        Returns
        -------
        obj:'SpecOneD'
            Returns an new SpecOneD instance populated by the original spectrum.
        """
        new_dispersion = self.dispersion.copy()
        new_flux = self.fluxden.copy()
        if self.fluxden_err is not None:
            new_flux_err = self.fluxden_err.copy()
        else:
            new_flux_err = None
        new_header = self.header
        new_unit = self.unit
        new_mask = self.mask
        new_raw_dispersion = self.raw_dispersion
        new_raw_flux = self.raw_fluxden
        new_raw_flux_err = self.raw_fluxden_err

        return SpecOneD(dispersion=new_dispersion,
                        fluxden=new_flux,
                        fluxden_err=new_flux_err,
                        header=new_header,
                        unit=new_unit,
                        mask=new_mask,
                        raw_dispersion=new_raw_dispersion,
                        raw_fluxden=new_raw_flux,
                        raw_fluxden_err=new_raw_flux_err,
                        )

    def override_raw(self):
        """ Override the raw_dispersion, raw_fluxden and raw_fluxden_err
        variables in the SpecOneD class with the current dispersion, fluxden and
        fluxden_err values.
        """

        self.raw_dispersion = self.dispersion
        self.raw_fluxden = self.fluxden
        self.raw_fluxden_err = self.fluxden_err

    def restore(self):
        """ Override the dispersion, fluxden and fluxden_err
        variables in the SpecOneD class with the raw_dispersion, raw_fluxden and
        raw_fluxden_err values.
        """

        self.dispersion = self.raw_dispersion
        self.fluxden = self.raw_fluxden
        self.fluxden_err = self.raw_fluxden_err
        self.reset_mask()

    def check_units(self, secondary_spectrum):
        """ This method checks if the active spectrum and a second spectrum
        have the same fluxden units.

        Parameters
        ----------
        secondary_spectrum : obj:`SpecOneD`
            Secondary spectrum to compare units with.

        Raises
        ------
        ValueError
            Raises an Error when the spectra are in different units.

        """

        if self.unit != secondary_spectrum.unit:
            raise ValueError('Spectra are in different units!')

    def to_log_wavelength(self):
        """ Convert the spectrum into logarithmic wavelength units.

        This method converts the dispersion and fluxden axis to logarithmic
        wavelength units.

        Raises
        ------
        ValueError
            Raises an error, if the dispersion/fluxden is arealdy in logarithmic
            wavelength.
        """

        if self.unit == 'f_loglam':
            raise ValueError('Spectrum is already in logarithmic wavelength')

        elif self.unit == 'f_lam':
            self.fluxden = self.fluxden * self.dispersion
            self.dispersion = np.log(self.dispersion)

        elif self.unit == 'f_nu':
            self.to_wavelength()
            self.to_log_wavelength()

        self.unit='f_loglam'


    def to_wavelength(self):
        """ Convert the spectrum from fluxden per frequency to fluxden per
        wavenlength.

        This method converts the fluxden from erg/s/cm^2/Hz to
        erg/s/cm^2/Angstroem and the dispersion accordingly from Hz to
        Angstroem.

        Raises
        ------
        ValueError
            Raises an error, if the fluxden is already in wavelength.
        """

        if self.unit == 'f_lam':
            raise ValueError('Dispersion is arealdy in wavelength')
        elif self.unit == 'f_nu':
            self.fluxden = self.fluxden * self.dispersion ** 2 / (const.c.value * 1e+10)
            self.dispersion = (const.c.value * 1e+10) / self.dispersion

            self.fluxden = np.flip(self.fluxden, axis=0)
            self.dispersion = np.flip(self.dispersion, axis=0)

        elif self.unit == 'f_loglam':
            self.dispersion = np.exp(self.dispersion)
            self.fluxden = self.fluxden / self.dispersion
        else:
            raise ValueError('Spectrum unit not recognized: ', self.unit)

        self.unit = 'f_lam'

    def to_frequency(self):
        """ Convert the spectrum from fluxden per wavelength to fluxden per
        frequency.

        This method converts the fluxden from erg/s/cm^2/Angstroem to
        erg/s/cm^2/Hz and the dispersion accordingly from Angstroem to Hz.

        Raises
        ------
        ValueError
            Raises an error, if the fluxden is already in frequency.
        """

        if self.unit == 'f_nu':
            raise ValueError('Dispersion is already in frequency.')

        elif self.unit == 'f_lam':
            self.fluxden = self.fluxden * self.dispersion ** 2 / (const.c.value * 1e+10)
            self.dispersion = (const.c.value * 1e+10) / self.dispersion

            self.fluxden = np.flip(self.fluxden, axis=0)
            self.dispersion = np.flip(self.dispersion, axis=0)

        elif self.unit == 'f_loglam':
            self.to_wavelength()
            self.to_frequency()
        else:
            raise ValueError('Spectrum unit not recognized: ', self.unit)

        self.unit = 'f_nu'

    def check_dispersion_overlap(self, secondary_spectrum):
        """Check the overlap between the active spectrum and the
        supplied secondary spectrum.

        This method determines whether the active spectrum (primary) and the
        supplied spectrum (secondary) have overlap in their dispersions.
        Possible cases include:
        i) The primary spectrum is fully overlapping with the secondary.
        ii) The secondary is fully overlapping with this priamy spectrum, but
        not vice versa.
        iii) and iv) There is partial overlap between the spectra.
        v) There is no overlap between the spectra.

        Parameters
        ----------
        secondary_spectrum : obj:`SpecOneD`
            Secondary spectrum

        Returns
        -------
        overlap : str
            A string indicating what the dispersion overlap between the spectra
            is according to the cases above.
        overlap_min : float
            The minimum value of the overlap region of the two spectra.
        overlap_max : float
            The maximum value of the overlap region of the two spectra.
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
        elif spec_min <= secondary_min and secondary_min <= spec_max <= secondary_max:
            return 'partial', secondary_min, spec_max
        elif secondary_max >= spec_min >= secondary_min and spec_max >= secondary_max:
            return 'partial', spec_min, secondary_max
        else:
            return 'none', np.NaN, np.NaN

    def match_dispersions(self, secondary_spectrum, match_secondary=True,
                          force=False, interp_kind='linear',
                          method='interpolate'):
        """Match the dispersion of the primary and the supplied secondary
        spectrum.


        Notes
        -----
        TODO: Add fluxden error handling

        Both, primary and secondary, SpecOneD classes are modified in this
        process. The dispersion match identifies the maximum possible overlap
        in the dispersion direction of both spectra and trims them to that
        range.

        If the primary spectrum overlaps fully with the secondary spectrum the
        dispersion of the secondary will be interpolated/resampled to the
        primary dispersion.
        If the secondary spectrum overlaps fully with the primary, the primary
        spectrum will be interpolated/resampled on the secondary spectrum
        resolution, but
        this happens only if 'force==True' and 'match_secondary==False'.
        If there is partial overlap between the spectra and 'force==True'
        the secondary spectrum will be interpolated to match the
        dispersion values of the primary spectrum.
        If there is no overlap a ValueError will be raised.

        Parameters
        ----------
        secondary_spectrum : obj:`SpecOneD`
            Secondary spectrum
        match_secondary : boolean
            The boolean indicates whether the secondary will always be matched
            to the primary or whether reverse matching, primary to secondary is
            allowed.
        force : boolean
            The boolean sets whether the dispersions are matched if only
            partial overlap between the spectral dispersions exists.

        Raises
        ------
        ValueError
            A ValueError will be raised if there is no overlap between the
            spectra.

        """

        self.check_units(secondary_spectrum)

        overlap, s_min, s_max = self.check_dispersion_overlap(secondary_spectrum)

        if overlap == 'primary':
            if method == 'interpolate':
                secondary_spectrum.interpolate(self.dispersion, kind=interp_kind)
            elif method == 'resample':
                secondary_spectrum.resample(self.dispersion, force=force,
                inplace=True)


        elif (overlap == 'secondary' and match_secondary is False and force is
        True):
            if method == 'interpolate':
                self.interpolate(secondary_spectrum.dispersion, kind=interp_kind)
            elif method == 'resample':
                self.resample(secondary_spectrum.dispersion, force=force,
                              inplace=True)

        elif (overlap == 'secondary' and match_secondary is True and force is
        True):
            self.trim_dispersion(limits=[s_min, s_max], mode='wav', inplace=True)
            if method == 'interpolate':
                secondary_spectrum.interpolate(self.dispersion, kind=interp_kind)
            elif method == 'resample':
                secondary_spectrum.resample(self.dispersion, force=force,
                                            inplace=True)

        elif overlap == 'partial' and force is True:
            self.trim_dispersion(limits=[s_min, s_max], mode='wav', inplace=True)
            if method == 'interpolate':
                secondary_spectrum.interpolate(self.dispersion, kind=interp_kind)
            elif method == 'resample':
                secondary_spectrum.resample(self.dispersion, force=force,
                                            inplace=True)

        elif force is False and (overlap == 'secondary' or overlap == 'partial'):
            raise ValueError('There is overlap between the spectra but force is False.')

        elif overlap == 'none':
            raise ValueError('There is no overlap between the primary and \
                             secondary spectrum.')


    def add(self, secondary_spectrum, copy_header='first', force=True,
            copy_flux_err = 'No', method='interpolate'):

        """Add the fluxden in the primary and secondary spectra.

        Notes
        -----
        Users should be aware that in order to add the fluxden of the two spectra,
        the dispersions of the spectra need to be matched, see match_dispersions
        and beware of the caveats of dispersion interpolation.

        Parameters
        ----------
        secondary_spectrum : obj:`SpecOneD`
            Secondary spectrum
        copy_header : 'str'
            A string indicating whether the primary('first') spectrum header or
            the secondary('last') spectrum header should be copied to the
            resulting spectrum.
        force : boolean
            The boolean sets whether the dispersions are matched if only
            partial overlap between the spectral dispersions exist.
        inplace : boolean
            The boolean indicates whether the resulting spectrum will overwrite
            the primary spectrum or whether it will be returned as a new
            spectrum argument by the method.

        Returns
        -------
        obj:`SpecOneD`
            Returns a SpecOneD object in the default case, "inplace==False".
        """

        self.check_units(secondary_spectrum)
        # fluxden error needs to be taken care of
        # needs to be tested

        s_one = self.copy()
        s_two = secondary_spectrum.copy()


        if not np.array_equal(s_one.dispersion, s_two.dispersion):
            print ("Warning: Dispersion does not match.")
            print ("Warning: Flux will be interpolated/resampled.")

            s_one.match_dispersions(s_two, force=force,
                                   method=method)

        new_flux = s_one.flux + s_two.flux

        if copy_flux_err == 'No' and isinstance(s_one.flux_err,
                                                np.ndarray) and \
                isinstance(s_two.flux_err, np.ndarray):
            new_flux_err = np.sqrt(s_one.flux_err**2 + s_two.flux_err**2)
        elif copy_flux_err == 'first':
            new_flux_err = s_one.flux_err
        elif copy_flux_err == 'second':
            new_flux_err = s_two.flux_err
        else:
            new_flux_err = None

        if copy_header == 'first':
            new_header = s_one.header
        elif copy_header == 'last':
            new_header = s_two.header


        return SpecOneD(dispersion=s_one.dispersion,
                        fluxden=new_flux,
                        fluxden_err=new_flux_err,
                        header=new_header,
                        unit=s_one.unit,
                        mask=s_one.mask)

    def subtract(self, secondary_spectrum, copy_header='first', force=True,
                 copy_flux_err = 'No', method='interpolate'):
        """Subtract the fluxden of the secondary spectrum from the primary
        spectrum.

        Notes
        -----
        Users should be aware that in order to subtract the fluxden,
        the dispersions of the spectra need to be matched, see match_dispersions
        and beware of the caveats of dispersion interpolation.

        Parameters
        ----------
        secondary_spectrum : obj:`SpecOneD`
            Secondary spectrum
        copy_header : 'str'
            A string indicating whether the primary('first') spectrum header or
            the secondary('last') spectrum header should be copied to the
            resulting spectrum.
        force : boolean
            The boolean sets whether the dispersions are matched if only
            partial overlap between the spectral dispersions exist.
        inplace : boolean
            The boolean indicates whether the resulting spectrum will overwrite
            the primary spectrum or whether it will be returned as a new
            spectrum argument by the method.

        Returns
        -------
        obj:`SpecOneD`
            Returns a SpecOneD object in the default case, "inplace==False".
        """

        self.check_units(secondary_spectrum)
        # fluxden error needs to be taken care of
        # needs to be tested
        # check for negative values?

        s_one = self.copy()
        s_two = secondary_spectrum.copy()

        if not np.array_equal(s_one.dispersion, s_two.dispersion):
            print ("Warning: Dispersion does not match.")
            print ("Warning: Flux will be interpolated.")

            s_one.match_dispersions(s_two, force=force,
                                   method=method)

        new_flux = s_one.flux - s_two.flux

        if copy_flux_err == 'No' and isinstance(s_one.flux_err,
                                                np.ndarray) and \
                isinstance(s_two.flux_err, np.ndarray):
            new_flux_err = np.sqrt(s_one.flux_err**2 + s_two.flux_err**2)
        elif copy_flux_err == 'first':
            new_flux_err = s_one.flux_err
        elif copy_flux_err == 'second':
            new_flux_err = s_two.flux_err
        else:
            new_flux_err = None

        if copy_header == 'first':
            new_header = s_one.header
        elif copy_header == 'last':
            new_header = s_two.header


        return SpecOneD(dispersion=s_one.dispersion,
                        fluxden=new_flux,
                        fluxden_err=new_flux_err,
                        header=new_header,
                        unit=s_one.unit,
                        mask=s_one.mask)

    def divide(self, secondary_spectrum, copy_header='first', force=True,
               copy_flux_err = 'No', method='interpolate'):

        """Divide the fluxden of the primary spectrum by the fluxden
        of the secondary spectrum.

        Notes
        -----
        Users should be aware that in order to divide the fluxden of the two
        spectra, the dispersions of the spectra need to be matched,
        (see match_dispersions) and beware of the caveats of dispersion
        interpolation.

        Parameters
        ----------
        secondary_spectrum : obj: SpecOneD
            Secondary spectrum
        copy_header : 'str'
            A string indicating whether the primary('first') spectrum header or
            the secondary('last') spectrum header should be copied to the
            resulting spectrum.
        force : boolean
            The boolean sets whether the dispersions are matched if only
            partial overlap between the spectral dispersions exist.
        inplace : boolean
            The boolean indicates whether the resulting spectrum will overwrite
            the primary spectrum or whether it will be returned as a new
            spectrum argument by the method.

        Returns
        -------
        obj: SpecOneD
            Returns a SpecOneD object in the default case, "inplace==False".
        """

        self.check_units(secondary_spectrum)

        s_one = self.copy()
        s_two = secondary_spectrum.copy()

        if not np.array_equal(self.dispersion, s_two.dispersion):
            print ("Warning: Dispersion does not match.")
            print ("Warning: Flux will be interpolated.")

            s_one.match_dispersions(s_two, force=force,
                                   match_secondary=False, method=method)

        new_flux = s_one.flux / s_two.flux
        print(copy_flux_err)
        if copy_flux_err == 'No' and isinstance(s_one.flux_err,
                                               np.ndarray) and \
                isinstance(s_two.flux_err, np.ndarray):
                new_flux_err = np.sqrt( (s_one.flux_err/ s_two.flux)**2  + (new_flux/s_two.flux*s_two.flux_err)**2 )
        elif copy_flux_err == 'first':
            new_flux_err = s_one.flux_err
        elif copy_flux_err == 'second':
            new_flux_err = s_two.flux_err
        else:
            new_flux_err = None

        print(new_flux_err, s_one.flux_err)

        if copy_header == 'first':
            new_header = s_one.header
        elif copy_header == 'last':
            new_header = s_two.header

        return SpecOneD(dispersion=s_one.dispersion,
                        fluxden=new_flux,
                        fluxden_err=new_flux_err,
                        header=new_header,
                        unit=s_one.unit,
                        mask=s_one.mask)

    def multiply(self, secondary_spectrum, copy_header='first', force=True,
                 copy_flux_err = 'No', method='interpolate'):

        """Multiply the fluxden of primary spectrum with the secondary spectrum.

        Notes
        -----
        Users should be aware that in order to add the fluxden of the two spectra,
        the dispersions of the spectra need to be matched, see match_dispersions
        and beware of the caveats of dispersion interpolation.

        Parameters
        ----------
        secondary_spectrum : obj:`SpecOneD`
            Secondary spectrum
        copy_header : 'str'
            A string indicating whether the primary('first') spectrum header or
            the secondary('last') spectrum header should be copied to the
            resulting spectrum.
        force : boolean
            The boolean sets whether the dispersions are matched if only
            partial overlap between the spectral dispersions exist.
        inplace : boolean
            The boolean indicates whether the resulting spectrum will overwrite
            the primary spectrum or whether it will be returned as a new
            spectrum argument by the method.

        Returns
        -------
        obj;`SpecOneD`
            Returns a SpecOneD object in the default case, "inplace==False".
        """

        self.check_units(secondary_spectrum)

        s_one = self.copy()
        s_two = secondary_spectrum.copy()

        if not np.array_equal(s_one.dispersion, s_two.dispersion):
            print ("Warning: Dispersion does not match.")
            print ("Warning: Flux will be interpolated.")

            s_one.match_dispersions(s_two, force=force,
                                   method=method)

        new_flux = s_one.flux * s_two.flux

        if copy_flux_err == 'No' and isinstance(s_one.flux_err,
                                                np.ndarray) and \
                isinstance(s_two.flux_err, np.ndarray):
            new_flux_err = np.sqrt(s_two.flux**2 * s_one.flux_err**2 + s_one.flux**2 * s_two.flux_err**2)
        elif copy_flux_err == 'first':
            new_flux_err = s_one.flux_err
        elif copy_flux_err == 'second':
            new_flux_err = s_two.flux_err
        else:
            new_flux_err = None

        if copy_header == 'first':
            new_header = s_one.header
        elif copy_header == 'last':
            new_header = s_two.header

        return SpecOneD(dispersion=s_one.dispersion,
                        fluxden=new_flux,
                        fluxden_err=new_flux_err,
                        header=new_header,
                        unit=s_one.unit,
                        mask=s_one.mask)


    def plot(self, show_fluxden_err=False, show_raw_fluxden=False,
             mask_values=False):

        """Plot the spectrum

        """

        if mask_values:
            mask = self.mask
        else:
            mask = np.ones(self.dispersion.shape, dtype=bool)

        self.fig, self.ax = plt.subplots(nrows=1, ncols=1, figsize=(15,7), dpi = 140)
        self.fig.subplots_adjust(left=0.09, right=0.97, top=0.89, bottom=0.16)

        # Plot the Spectrum
        self.ax.axhline(y=0.0, linewidth=1.5, color='k', linestyle='--')

        if show_fluxden_err:
            self.ax.plot(self.dispersion[mask], self.fluxden_err[mask], 'grey', lw=1)
        if show_raw_fluxden:
            self.ax.plot(self.raw_dispersion[mask], self.raw_fluxden[mask], 'grey', lw=3)

        self.ax.plot(self.dispersion[mask], self.fluxden[mask], 'k', linewidth=1)

        if self.unit=='f_lam':
            self.ax.set_xlabel(r'$\rm{Wavelength}\ [\rm{\AA}]$', fontsize=15)
            self.ax.set_ylabel(r'$\rm{Flux}\ f_{\lambda}\ [\rm{erg}\,\rm{s}^{-1}\,\rm{cm}^{2}\,\rm{\AA}^{-1}]$', fontsize=15)

        elif self.unit =='f_nu':
            self.ax.set_xlabel(r'$\rm{Frequency}\ [\rm{Hz}]$', fontsize=15)
            self.ax.set_ylabel(r'$\rm{Flux}\ f_{\nu}\ [\rm{erg}\,\rm{s}^{-1}\,\rm{cm}^{2}\,\rm{Hz}^{-1}]$', fontsize=15)

        elif self.unit =='f_loglam':
            self.ax.set_xlabel(r'$\log\rm{Wavelength}\ [\log\rm{\AA}]$', fontsize=15)
            self.ax.set_ylabel(r'$\rm{Flux}\ f_{\lambda}\ [\rm{erg}\,\rm{s}^{-1}\,\rm{cm}^{2}\,(\log\rm{\AA})^{-1}]$', fontsize=15)

        else:
            raise ValueError("Unrecognized units")

        # If a model spectrum exists, print it
        if self.model_spectrum:
            model_flux = self.model_spectrum.eval(self.model_pars, x=self.dispersion)
            self.ax.plot(self.dispersion[mask], model_flux[mask])

        if self.fit_output:
            self.ax.plot(self.dispersion[mask], self.fit_output.best_fit[mask])

        lim_spec = self.copy()
        ylim_min, ylim_max = lim_spec.get_specplot_ylim()

        self.ax.set_ylim(ylim_min, ylim_max)

        plt.show()

    def get_specplot_ylim(self):

        spec = self.copy()

        percentiles = np.percentile(spec.fluxden[spec.mask], [16, 84])
        median = np.median(spec.fluxden[spec.mask])

        ylim_min = -0.5*median
        ylim_max = 4*percentiles[1]

        return ylim_min, ylim_max


    def pypeit_plot(self, show_fluxden_err=False, show_raw_fluxden=False,
                    mask_values=False, ymax=None):

        """Plot the spectrum assuming it is a pypeit spectrum

        """

        if mask_values:
            mask = self.mask
        else:
            mask = np.ones(self.dispersion.shape, dtype=bool)

        self.fig, self.ax = plt.subplots(nrows=1, ncols=1, figsize=(15,7), dpi = 140)
        self.fig.subplots_adjust(left=0.09, right=0.97, top=0.89, bottom=0.16)

        # Plot the Spectrum
        self.ax.axhline(y=0.0, linewidth=1.5, color='k', linestyle='--')

        # Add second axis to plot telluric model
        if hasattr(self, 'telluric'):
            telluric = self.telluric[mask] / self.telluric.max() * np.median(
                self.fluxden[mask]) * 2.5
            self.ax.plot(self.dispersion[mask], telluric[mask],
                         label='Telluric')

        if show_fluxden_err:
            self.ax.plot(self.dispersion[mask], self.fluxden_err[mask], 'grey',
                         lw=1, label='Flux Error')
        if show_raw_fluxden:
            self.ax.plot(self.raw_dispersion[mask], self.raw_fluxden[mask],
                         'grey', lw=3, label='Raw Flux')

        self.ax.plot(self.dispersion[mask], self.fluxden[mask], 'k',
                     linewidth=1, label='Flux')

        if self.unit=='f_lam':
            self.ax.set_xlabel(r'$\rm{Wavelength}\ [\rm{\AA}]$', fontsize=15)
            self.ax.set_ylabel(r'$\rm{Flux}\ f_{\lambda}\ [\rm{erg}\,\rm{s}^{'
                               r'-1}\,\rm{cm}^{-2}\,\rm{\AA}^{-1}]$', fontsize=15)

        elif self.unit =='f_nu':
            self.ax.set_xlabel(r'$\rm{Frequency}\ [\rm{Hz}]$', fontsize=15)
            self.ax.set_ylabel(r'$\rm{Flux}\ f_{\nu}\ [\rm{erg}\,\rm{s}^{'
                               r'-1}\,\rm{cm}^{-2}\,\rm{Hz}^{-1}]$', fontsize=15)

        elif self.unit =='f_loglam':
            self.ax.set_xlabel(r'$\log\rm{Wavelength}\ [\log\rm{\AA}]$', fontsize=15)
            self.ax.set_ylabel(r'$\rm{Flux}\ f_{\lambda}\ [\rm{erg}\,\rm{s}^{'
                               r'-1}\,\rm{cm}^{-2}\,(\log\rm{\AA})^{-1}]$', fontsize=15)

        else:
            raise ValueError("Unrecognized units")

        # If a model spectrum exists, plot it
        if self.model_spectrum:
            model_flux = self.model_spectrum.eval(self.model_pars, x=self.dispersion)
            self.ax.plot(self.dispersion[mask], model_flux[mask])

        if self.fit_output:
            self.ax.plot(self.dispersion[mask], self.fit_output.best_fit[mask])


        # Add OBJ model if it exists
        if hasattr(self, 'obj_model'):
            self.ax.plot(self.dispersion[mask], self.obj_model, label='Obj '
                                                                      'model')


        lim_spec = self.copy()
        lim_spec.restore()
        lim_spec = lim_spec.mask_sn(5)
        lim_spec = lim_spec.sigmaclip_flux(3, 3)
        ylim_min = 0
        if ymax == None:
            ylim_max = lim_spec.flux[lim_spec.mask].max()
        else:
            ylim_max = ymax
        self.ax.set_ylim(ylim_min, ylim_max)

        self.ax.legend()
        plt.show()


    def fit_model_spectrum(self, mask_values=True):

        if mask_values:
            self.fit_output = self.model_spectrum.fit(self.fluxden[self.mask], self.model_pars, x =self.dispersion[self.mask])
        else:
            self.fit_output = self.model_spectrum.fit(self.fluxden, self.model_pars, x =self.dispersion)

    def calculate_snr(self):

        pass

    def trim_dispersion(self, limits, mode='wav', trim_err=True, inplace=False):

        #change names of modes.... physical, pixel ?

        if mode == "wavelength" or mode == "wav":

            lo_index = np.argmin(np.abs(self.dispersion - limits[0]))
            up_index = np.argmin(np.abs(self.dispersion - limits[1]))

            # Warnings
            if limits[0] < self.dispersion[0]:
                print(self.dispersion[0], limits[0])
                print("WARNING: Lower limit is below the lowest dispersion value. The lower limit is set to the minimum dispersion value.")
            if limits[1] > self.dispersion[-1]:
                print("WARNING: Upper limit is above the highest dispersion value. The upper limit is set to the maximum dispersion value.")



        else:
            # Warnings
            if limits[0] < self.dispersion[0]:
                print("WARNING: Lower limit is below the lowest dispersion value. The lower limit is set to the minimum dispersion value.")
            if limits[1] > self.dispersion[-1]:
                print("WARNING: Upper limit is above the highest dispersion value. The upper limit is set to the maximum dispersion value.")

            lo_index = limits[0]
            up_index = limits[1]

        if inplace:
            self.dispersion = self.dispersion[lo_index:up_index]
            self.fluxden = self.fluxden[lo_index:up_index]
            self.mask = self.mask[lo_index:up_index]
            if trim_err:
                if self.fluxden_err is not None:
                    self.fluxden_err = self.fluxden_err[lo_index:up_index]
        else:
            spec = self.copy()
            spec.dispersion = spec.dispersion[lo_index:up_index]
            spec.fluxden = spec.fluxden[lo_index:up_index]
            spec.mask = spec.mask[lo_index:up_index]
            if trim_err:
                if spec.fluxden_err is not None:
                    spec.fluxden_err = spec.fluxden_err[lo_index:up_index]

            return spec


    def interpolate(self, new_dispersion, kind='linear', fill_value='const'):

        """Interpolate fluxden to new dispersion axis

        Parameters
        ----------
        new_dispersion : ndarray
            1D array with the new dispersion axis
        kind : str
            String that indicates the interpolation function
        fill_value : str
            A string indicating whether values outside the dispersion range
            will be extrapolated ('extrapolate') or filled with a constant
            value ('const') based on the median of the 10 values at the edge.
        """

        if fill_value=='extrapolate':
            f = sp.interpolate.interp1d(self.dispersion, self.fluxden, kind=kind, fill_value='extrapolate')
            if isinstance(self.fluxden_err, np.ndarray):
                f_err = sp.interpolate.interp1d(self.dispersion, self.fluxden_err, kind=kind, fill_value='extrapolate')
            print ('Warning: Values outside the original dispersion range will be extrapolated!')
        elif fill_value == 'const':
            fill_lo = np.median(self.fluxden[0:10])
            fill_hi = np.median(self.fluxden[-11:-1])
            f = sp.interpolate.interp1d(self.dispersion, self.fluxden, kind=kind,
                                        bounds_error= False,
                                        fill_value=(fill_lo, fill_hi))
            if isinstance(self.fluxden_err, np.ndarray):
                fill_lo_err = np.median(self.fluxden_err[0:10])
                fill_hi_err = np.median(self.fluxden_err[-11:-1])
                f_err = sp.interpolate.interp1d(self.dispersion, self.fluxden_err, kind=kind, bounds_error= False,
                                                fill_value=(fill_lo_err, fill_hi_err))
        else:
            f = sp.interpolate.interp1d(self.dispersion, self.fluxden, kind=kind)
            if isinstance(self.fluxden_err, np.ndarray):
                f_err = sp.interpolate.interp1d(self.dispersion, self.fluxden_err, kind=kind)

        self.dispersion = new_dispersion
        self.reset_mask()
        self.fluxden = f(self.dispersion)
        if isinstance(self.fluxden_err, np.ndarray):
            self.fluxden_err = f_err(self.dispersion)

    def smooth(self, width, kernel="boxcar", scale_sigma=True, inplace=False):
        """Smoothing the spectrum using a boxcar oder gaussian kernel.

        This function uses astropy.convolution to convolve the spectrum with
        the selected kernel. If scale_sigma=True, the fluxden error is scaled
        down according to sqrt(width).

        :param width: int
            Width (in pixels) of the kernel)
        :param kernel: str
            String indicating whether to use the Boxcar ("boxcar") or
            Gaussian ("gaussian") kernel.
        :param scale_sigma: bool
            Boolean to indicate whether to scale the fluxden error according to
            the width of the boxcar kernel.
        :param inplace: bool
            Boolean to indicate whether to modify the active spectrum or
            return a copy. The default is to always return a copy.
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


    def convolve_loglam(self, fwhm, method='interpolate', inplace=False,
                        force=False):
        """Convolve the spectrum in loglam space with a kernel function of specified width"""

        spec = self.copy()

        stddev = fwhm / (2 * np.sqrt(2 * np.log(2)))

        # Step 1 convert spectrum to logarithmic wavelength (velocity space)
        spec.to_log_wavelength()
        # Interpolate to linear scale in logarithmic wavelength
        new_disp = np.linspace(min(spec.dispersion), max(spec.dispersion),
                               num=len(spec.dispersion))
        if method == 'interpolate':
            spec.interpolate(new_disp)
        elif method == 'resample':
            spec.resample(new_disp, force=force, inplace=True)

        # Step 2 convolve spectrum with function
        cen = (max(new_disp)-min(new_disp))/2. + min(new_disp)
        y = gaussian(new_disp, 1.0, cen, stddev/300000, 0)
        # hacked the normalization here... did not fully understand why this happens
        conv = np.convolve(spec.fluxden, y, mode='same') / (len(new_disp)/2.)

        spec.fluxden= conv

        # Step 3 convert back to original space
        spec.to_wavelength()
        if method == 'interpolate':
            spec.interpolate(self.dispersion)
        elif method == 'resample':
            spec.resample(self.dispersion, force=force, inplace=True)

        if inplace:
            self.fluxden = spec.flux
        else:
            return SpecOneD(dispersion=spec.dispersion,
                            fluxden=spec.fluxden,
                            fluxden_err=spec.fluxden_err,
                            header=spec.header,
                            unit=spec.unit)


    # def redden(self, a_v, r_v, extinction_law='ccm89', inplace=False):
    #
    #     if self.unit != 'f_lam':
    #         raise ValueError('Dispersion units must be in wavelength (Angstroem)')
    #
    #     if extinction_law == 'ccm89':
    #         extinction = ext.ccm89(self.dispersion, a_v, r_v)
    #     elif extinction_law == 'odonnel94':
    #         extinction = ext.odonnel94(self.dispersion, a_v, r_v)
    #     elif extinction_law == 'calzetti00':
    #         extinction = ext.calzetti00(self.dispersion, a_v, r_v)
    #     elif extinction_law == 'fitzpatrick99':
    #         extinction = ext.fitzpatrick99(self.dispersion, a_v, r_v)
    #     elif extinction_law == 'fm07':
    #         print('Warning: For Fitzpatrick & Massa 2007 R_V=3.1')
    #         extinction = ext.fm07(self.dispersion, a_v)
    #     else:
    #         raise ValueError('Specified Extinction Law not recognized')
    #
    #     if inplace:
    #         self.fluxden = self.fluxden * 10.0**(-0.4*extinction)
    #     else:
    #         fluxden = self.fluxden * 10.0**(-0.4*extinction)
    #         return SpecOneD(dispersion=self.dispersion,
    #                         fluxden=fluxden,
    #                         fluxden_err=self.fluxden_err,
    #                         header=self.header,
    #                         unit = self.unit)
    #
    # def deredden(self, a_v, r_v, extinction_law='ccm89', inplace=False):
    #
    #     if self.unit != 'f_lam':
    #         raise ValueError('Dispersion units must be in wavelength (Angstroem)')
    #
    #     if extinction_law == 'ccm89':
    #         extinction = ext.ccm89(self.dispersion, a_v, r_v)
    #     elif extinction_law == 'odonnel94':
    #         extinction = ext.odonnel94(self.dispersion, a_v, r_v)
    #     elif extinction_law == 'calzetti00':
    #         extinction = ext.calzetti00(self.dispersion, a_v, r_v)
    #     elif extinction_law == 'fitzpatrick99':
    #         extinction = ext.fitzpatrick99(self.dispersion, a_v, r_v)
    #     elif extinction_law == 'fm07':
    #         print('Warning: For Fitzpatrick & Massa 2007 R_V=3.1')
    #         extinction = ext.fm07(self.dispersion, a_v)
    #     else:
    #         raise ValueError('Specified Extinction Law not recognized')
    #
    #     if inplace:
    #         self.fluxden = self.fluxden * 10.0**(0.4*extinction)
    #     else:
    #         fluxden = self.fluxden * 10.0**(0.4*extinction)
    #         return SpecOneD(dispersion=self.dispersion,
    #                         fluxden=fluxden,
    #                         fluxden_err=self.fluxden_err,
    #                         header=self.header,
    #                         unit = self.unit)


    def calculate_passband_flux(self, passband, force=False):

        spec = self.copy()

        if spec.unit == 'f_lam':
            spec.to_frequency()
        elif spec.unit != 'f_nu':
            raise ValueError('Spectrum units must be f_lam or f_nu')
        if passband.unit == 'f_lam':
            passband.to_frequency()
        elif passband.unit != 'f_nu':
            raise ValueError('PassBand units must be f_lam or f_nu')


        overlap, disp_min, disp_max = passband.check_dispersion_overlap(spec)

        if not force:
            if overlap != 'primary':
                raise ValueError('The spectrum does not fill the passband')
        else:
            print(
                "Warning: Force was set to TRUE. The spectrum might not fully fill the passband!")

        passband.match_dispersions(spec, force=force)
        spec.fluxden= passband.flux * spec.flux
        total_flux = np.trapz(spec.flux, spec.dispersion)

        if total_flux <= 0.0:
            raise ValueError('Integrated fluxden is <= 0')
        if np.isnan(total_flux):
            raise ValueError('Integrated fluxden is NaN')
        if np.isinf(total_flux):
            raise ValueError('Integrated fluxden is infinite')

        return total_flux

    def calculate_passband_magnitude(self, passband, mag_system='AB',
                                      force=False, matching='resample'):
        # This function is written for passbands in quantum efficiency
        # Therefore, the (h*nu)^-1 term is not included in the integral

        spec = self.copy()

        if mag_system == 'AB':
            if spec.unit == 'f_lam':
                spec.to_frequency()
            elif spec.unit != 'f_nu':
                raise ValueError('Spectrum units must be f_lam or f_nu')
            if passband.unit == 'f_lam':
                passband.to_frequency()
            elif passband.unit != 'f_nu':
                raise ValueError('PassBand units must be f_lam or f_nu')
        else:
            raise NotImplementedError('Only AB magnitudes are currently implemented')

        overlap, disp_min, disp_max = passband.check_dispersion_overlap(spec)

        if not force:
            if overlap != 'primary':
                raise ValueError('The spectrum does not fill the passband')
        else:
            print("Warning: Force was set to TRUE. The spectrum might not fully fill the passband!")

        passband.match_dispersions(spec, force=force, method=matching)

        spec.fluxden= passband.flux * spec.flux

        total_flux = np.trapz(spec.flux, spec.dispersion)

        if total_flux <= 0.0:
            raise ValueError('Integrated fluxden is <= 0')
        if np.isnan(total_flux):
            raise ValueError('Integrated fluxden is NaN')
        if np.isinf(total_flux):
            raise ValueError('Integrated fluxden is infinite')

        flat = FlatSpectrum(spec.dispersion, unit='f_nu')

        passband_flux = np.trapz(flat.flux * passband.flux,
                                 flat.dispersion)

        ratio = total_flux / passband_flux

        return -2.5 * np.log10(ratio)

    def renormalize_by_magnitude(self, magnitude, passband, mag_system='AB',
                                 force=False, inplace=False,
                                 matching='resample', output='spectrum'):

        spec_mag = self.calculate_passband_magnitude(passband,
                                                     mag_system=mag_system,
                                                     force=force,
                                                     matching=matching)

        dmag = magnitude - spec_mag

        if output == 'spectrum':

            if inplace:
                self.to_frequency
                self.flux_scale_factor = 10**(-0.4*dmag)
                self.fluxden = self.fluxden * 10 ** (-0.4 * dmag)
                if self.fluxden_err is not None:
                    self.fluxden_err = self.fluxden_err * 10 ** (-0.4 * dmag)
                self.to_wavelength
            else:
                spec = self.copy()
                spec.to_frequency
                spec.flux_scale_factor = 10**(-0.4*dmag)
                spec.fluxden= spec.fluxden* 10**(-0.4*dmag)
                if spec.fluxden_err is not None:
                    spec.fluxden_err = spec.fluxden_err * 10**(-0.4*dmag)
                spec.to_wavelength

                return spec

        elif output == 'flux_factor':

            return 10**(-0.4*dmag)

        else:
            raise ValueError("output mode not understood")

    def renormalize_by_spectrum(self, spectrum, dispersion='match', trim_mode='wav', inplace=False):
        """ Match the fluxden of the active spectrum to the "spectrum" given to
        the function.

        Note:
        Both spectra will be copied to dummy variables:
        spec = active spectrum
        spec2 = supplied spectrum
        """

        spec = self.copy()
        spec2 = spectrum.copy()



        if dispersion == "match":
            spec.match_dispersions(spec2, match_secondary=False,
                                  force=True, interp_kind='linear')
        elif isinstance(dispersion,(list,)):
            spec.trim_dispersion(dispersion, mode=trim_mode,inplace=True)
            spec2.trim_dispersion(dispersion, mode=trim_mode,inplace=True)
        else:
            print("Spectra will be normalized but dispersion ranges do not necessarily match!")



        average_self_flux = np.trapz(spec.flux, spec.dispersion)

        average_spec_flux = np.trapz(spec2.flux, spec2.dispersion)

        self.scale = (average_spec_flux/average_self_flux)

        if inplace:
            self.flux_scale_factor = average_spec_flux/average_self_flux
            self.fluxden = self.fluxden * (average_spec_flux / average_self_flux)
        else:
            spec = self.copy()
            spec.flux_scale_factor = average_spec_flux/average_self_flux
            flux = self.fluxden * (average_spec_flux / average_self_flux)
            spec.scale = self.scale
            spec.fluxden= flux

            return spec

    def doppler_shift(self, z, method='redshift', inplace=False):
        pass

    def redshift(self, z, inplace=False):
        # TODO Take care of fluxden conversion here as well!
        # TODO Taken care of fluxden conversion, check how IVAR behaves

        if inplace:
            self.dispersion = self.dispersion * (1.+z)
            # self.fluxden /= (1.+z)
            # self.fluxden_err /= (1.+z)
        else:
            spec = self.copy()
            spec.dispersion = spec.dispersion * (1.+z)
            # spec.fluxden /= (1.+z)
            # spec.fluxden_err /= (1.+z)

            return spec


    def medianclip_flux(self, sigma=3, binsize=11, inplace=False):
        """
        Quick hack for sigma clipping using a running median
        :param sigma:
        :param binsize:
        :param inplace:
        :return:
        """

        flux = self.fluxden.copy()
        flux_err = self.fluxden_err.copy()

        median = medfilt(flux, kernel_size=binsize)

        diff = np.abs(flux-median)

        mask = self.mask.copy()

        mask[diff > sigma * flux_err] = 0

        if inplace:
            self.mask = mask
        else:
            spec = self.copy()
            spec.mask = mask

            return spec


    def sigmaclip_flux(self, low=3, up=3, binsize=120, niter=5, inplace=False):

        hbinsize = int(binsize/2)

        flux = self.fluxden
        dispersion = self.dispersion


        mask_index = np.arange(dispersion.shape[0])

        # loop over sigma-clipping iterations
        for jdx in range(niter):

            n_mean = np.zeros(flux.shape[0])
            n_std = np.zeros(flux.shape[0])

            # calculating mean and std arrays
            for idx in range(len(flux[:-binsize])):

                # fluxden subset
                f_bin = flux[idx:idx+binsize]

                # calculate mean and std
                mean = np.median(f_bin)
                std = np.std(f_bin)

                # set array value
                n_mean[idx+hbinsize] = mean
                n_std[idx+hbinsize] = std

            # fill zeros at end and beginning
            # with first and last values
            n_mean[:hbinsize] = n_mean[hbinsize]
            n_mean[-hbinsize:] = n_mean[-hbinsize-1]
            n_std[:hbinsize] = n_std[hbinsize]
            n_std[-hbinsize:] = n_std[-hbinsize-1]

            # create index array with included pixels ("True" values)
            mask = (flux-n_mean < n_std*up) & (flux-n_mean > -n_std*low)
            mask_index = mask_index[mask]

            # mask the fluxden for the next iteration
            flux = flux[mask]

        mask = np.zeros(len(self.mask), dtype='bool')

        # mask = self.mask
        mask[:] = False
        mask[mask_index] = True
        mask = mask * np.array(self.mask, dtype=bool)

        if inplace:
            self.mask = mask
        else:
            spec = self.copy()
            spec.mask = mask

            return spec

    def fit_polynomial(self, func='legendre', order=3, inplace=False):

        if func == 'legendre':
            poly = Legendre.fit(self.dispersion[self.mask], self.fluxden[self.mask], deg=order)
        elif func == 'chebyshev':
            poly = Chebyshev.fit(self.dispersion[self.mask], self.fluxden[self.mask], deg=order)
        elif func == 'polynomial':
            poly = Polynomial.fit(self.dispersion[self.mask], self.fluxden[self.mask], deg=order)
        else:
            raise ValueError("Polynomial fitting function not specified")

        if inplace:
            self.fit_dispersion = self.dispersion
            self.fit_fluxden = poly(self.dispersion)
        else:
            spec = self.copy()
            spec.fit_dispersion = self.dispersion
            spec.fit_flux = poly(self.dispersion)

            return spec

    def mask_between(self, limits, inplace=False):

        lo_index = np.argmin(np.abs(self.dispersion - limits[0]))
        up_index = np.argmin(np.abs(self.dispersion - limits[1]))

        self.mask[lo_index:up_index] = 0

        if inplace:
            self.mask[lo_index:up_index] = 0
        else:
            spec = self.copy()
            spec.mask[lo_index:up_index] = 0

            return spec

    def create_dispersion_by_resolution(self, resolution):
        """
        This function creates a new dispersion axis in wavelength sampled by
        a fixed resolution, given in km/s
        :param resolution:
        :return:
        """

        new_dispersion = [self.dispersion[0]]
        lambda_new = 0
        while lambda_new < self.dispersion[-1]:

            d_lambda = new_dispersion[-1]/const.c.to(u.km/u.s).value * \
                       resolution
            # print(lambda_new)
            lambda_new = d_lambda + new_dispersion[-1]
            new_dispersion.append(lambda_new)

        return np.array(new_dispersion[1:-1])

    def resample_to_resolution(self, resolution, buffer=2, inplace=False):

        new_dispersion = self.create_dispersion_by_resolution(resolution)

        if inplace:
            self.resample(new_dispersion[buffer:-buffer], inplace=inplace)
        else:
            return self.resample(new_dispersion[buffer:-buffer],
                                 inplace=inplace)

    def resample(self, new_dispersion, inplace=False, force=False):
        """
        Function for resampling spectra (and optionally associated
        uncertainties) onto a new wavelength basis.

        This code is copied from
        https://github.com/ACCarnall/SpectRes
        by Adam Carnall - damc@roe.ac.uk

        and adapted by Jan-Torge Schindler for functionality within
        the SpecOneD class

        Parameters
        ----------
        new_dispersion : numpy.ndarray
            Array containing the new wavelength sampling desired for the
            spectrum or spectra.

        inplace : Boolean
            Boolean to indicate whether the results overwrite the SpecOneD
            object or a new SpecOneD object is created and returned.

        Returns
        -------
        spec/self : SpecOneD
            The function returns a SpecOneD class object with the new
            dispersion and the resampled fluxes (and fluxden errors).
        """

        # Mapping of the SpecOneD object variables to the function
        # variables

        old_spec_wavs = self.dispersion
        spec_fluxes = self.fluxden
        if self.fluxden_err is not None:
            spec_errs = self.fluxden_err
        else:
            spec_errs = None

        new_spec_wavs = new_dispersion

        if force:
            indices = np.where((new_spec_wavs < old_spec_wavs.max()) &
                               (new_spec_wavs > old_spec_wavs.min()))
            new_spec_wavs = new_spec_wavs[indices]

        # Arrays of left-hand sides and widths for the old and new bins
        spec_widths = np.zeros(old_spec_wavs.shape[0])
        spec_lhs = np.zeros(old_spec_wavs.shape[0])
        spec_lhs[0] = old_spec_wavs[0]
        spec_lhs[0] -= (old_spec_wavs[1] - old_spec_wavs[0]) / 2
        spec_widths[-1] = (old_spec_wavs[-1] - old_spec_wavs[-2])
        spec_lhs[1:] = (old_spec_wavs[1:] + old_spec_wavs[:-1]) / 2
        spec_widths[:-1] = spec_lhs[1:] - spec_lhs[:-1]

        filter_lhs = np.zeros(new_spec_wavs.shape[0] + 1)
        filter_widths = np.zeros(new_spec_wavs.shape[0])
        filter_lhs[0] = new_spec_wavs[0]
        filter_lhs[0] -= (new_spec_wavs[1] - new_spec_wavs[0]) / 2
        filter_widths[-1] = (new_spec_wavs[-1] - new_spec_wavs[-2])
        filter_lhs[-1] = new_spec_wavs[-1]
        filter_lhs[-1] += (new_spec_wavs[-1] - new_spec_wavs[-2]) / 2
        filter_lhs[1:-1] = (new_spec_wavs[1:] + new_spec_wavs[:-1]) / 2
        filter_widths[:-1] = filter_lhs[1:-1] - filter_lhs[:-2]

        if filter_lhs[0] < spec_lhs[0] or filter_lhs[-1] > spec_lhs[-1]:

            raise ValueError("spectres: The new wavelengths specified must fall"
                             "within the range of the old wavelength values:",
                             filter_lhs[0], spec_lhs[0], filter_lhs[-1],
                             spec_lhs[-1], "\n Consider setting force=True")

        # Generate output arrays to be populated
        res_fluxes = np.zeros(spec_fluxes[..., 0].shape + new_spec_wavs.shape)

        if spec_errs is not None:
            if spec_errs.shape != spec_fluxes.shape:
                raise ValueError(
                    "If specified, spec_errs must be the same shape"
                    "as spec_fluxes.")
            else:
                res_fluxerrs = np.copy(res_fluxes)

        start = 0
        stop = 0

        # Calculate new fluxden and uncertainty values, loop over new bins
        for j in range(new_spec_wavs.shape[0]):

            # Find first old bin which is partially covered by the new bin
            while spec_lhs[start + 1] <= filter_lhs[j]:
                start += 1

            # Find last old bin which is partially covered by the new bin
            while spec_lhs[stop + 1] < filter_lhs[j + 1]:
                stop += 1

            # If new bin is fully within one old bin these are the same
            if stop == start:

                res_fluxes[..., j] = spec_fluxes[..., start]
                if spec_errs is not None:
                    res_fluxerrs[..., j] = spec_errs[..., start]

            # Otherwise multiply the first and last old bin widths by P_ij
            else:

                start_factor = ((spec_lhs[start + 1] - filter_lhs[j])
                                / (spec_lhs[start + 1] - spec_lhs[start]))

                end_factor = ((filter_lhs[j + 1] - spec_lhs[stop])
                              / (spec_lhs[stop + 1] - spec_lhs[stop]))

                spec_widths[start] *= start_factor
                spec_widths[stop] *= end_factor

                # Populate res_fluxes spectrum and uncertainty arrays
                f_widths = spec_widths[start:stop + 1] * spec_fluxes[...,
                                                         start:stop + 1]
                res_fluxes[..., j] = np.sum(f_widths, axis=-1)
                res_fluxes[..., j] /= np.sum(spec_widths[start:stop + 1])

                if spec_errs is not None:
                    e_wid = spec_widths[start:stop + 1] * spec_errs[...,
                                                          start:stop + 1]

                    res_fluxerrs[..., j] = np.sqrt(np.sum(e_wid ** 2, axis=-1))
                    res_fluxerrs[..., j] /= np.sum(spec_widths[start:stop + 1])

                # Put back the old bin widths to their initial values for
                # later use
                spec_widths[start] /= start_factor
                spec_widths[stop] /= end_factor

        if inplace:

            self.dispersion = new_dispersion
            self.fluxden = res_fluxes
            if spec_errs is not None:
                self.fluxden_err = res_fluxerrs

            self.reset_mask()

        else:

            spec = self.copy()

            spec.dispersion = new_dispersion
            spec.fluxden= res_fluxes
            if spec_errs is not None:
                spec.fluxden_err = res_fluxerrs

            spec.reset_mask()

            return spec

    def bin_by_npixels(self, npix):
        """Bin npix pixels of the old spectrum to form a new spectrum. We
        assume that the bin boundaries of the old spectrum are always exactly
        in the center wavelength position between adjacent pixel wavelengths.

        :param npix:
        :return:
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

        new_wave = np.zeros(num_new_bins)
        new_flux = np.zeros(num_new_bins)
        new_flux_err = np.zeros(num_new_bins)

        for idx in range(num_new_bins):

            _new_flux = 0
            _new_flux_err = 0
            _new_dbin = 0

            for jdx in range(npix):
                _new_flux += flux[idx * npix + jdx] * dbins[idx * npix + jdx]
                _new_dbin += dbins[idx * npix + jdx]
                _new_flux_err += (flux_err[idx * npix + jdx] * dbins[
                    idx * npix + jdx]) ** 2

            rbin = rbins[npix * idx + npix - 1]
            lbin = lbins[npix * idx]
            _new_wave = (rbin - lbin) * 0.5 + lbin

            new_wave[idx] = _new_wave
            new_flux[idx] = _new_flux / _new_dbin
            new_flux_err[idx] = np.sqrt(_new_flux_err) / _new_dbin

        return SpecOneD(dispersion=new_wave, fluxden=new_flux,
                        fluxden_err=new_flux_err, unit='f_lam')


    def average_fluxden(self, disp_range=None):

        if disp_range is None:
            return np.average(self.fluxden)
        else:
            return np.average(self.trim_dispersion(disp_range).fluxden)

    def peak_fluxden(self):

        return np.max(self.fluxden)

    def peak_dispersion(self):

        return self.dispersion[np.argmax(self.fluxden)]


class FlatSpectrum(SpecOneD):

    def __init__(self, flat_dispersion, unit='f_nu'):

        try:
            flat_dispersion = np.array(flat_dispersion)
            if flat_dispersion.ndim != 1:
                raise ValueError("Flux dimension is not 1")
        except ValueError:
            print("Flux could not be converted to 1D ndarray")

        if unit == 'f_lam':
            fill_value = 3.631e-9
        if unit == 'f_nu':
            fill_value = 3.631e-20

        self.flux = np.full(flat_dispersion.shape, fill_value)
        self.raw_flux = self.flux
        self.dispersion = flat_dispersion
        self.raw_dispersion = self.dispersion

def combine_spectra(filenames, method='average', file_format='fits'):

    s_list = []

    for filename in filenames:
        spec = SpecOneD()
        if file_format == 'fits':
            spec.read_from_fits(filename)
            print(spec)
        else:
            raise NotImplementedError('File format not understood')

        s_list.append(spec)

    # TODO Test if all spectra are in same unit, if not convert
    print(s_list)
    # Resample all spectra onto slightly reduced dispersion of first spectrum
    disp = s_list[0].dispersion[5:-5]
    for spec in s_list:
        spec.resample(disp, inplace=True)

    comb_flux = np.zeros(len(disp))
    comb_fluxerr = np.zeros(len(disp))

    N = float(len(filenames))

    if method == 'average':
        for spec in s_list:
            comb_flux += spec.fluxden
            comb_fluxerr += spec.fluxden_err ** 2 / N ** 2

        comb_flux = comb_flux / N
        comb_fluxerr = np.sqrt(comb_fluxerr)

        comb_spec = SpecOneD(dispersion=disp, fluxden=comb_flux,
                             fluxden_err=comb_fluxerr, unit='f_lam',
                             )
        return comb_spec

    else:
        raise NotImplementedError('Selected method for combining spectra is '
                                  'not implemented. Implemented methods: '
                                  'average')


def pypeit_spec1d_plot(filename, show_flux_err=True, mask_values=False,
                        ex_value='OPT', show='fluxden', smooth=None):

    # plot_setup
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15, 7), dpi=140)
    fig.subplots_adjust(left=0.09, right=0.97, top=0.89, bottom=0.16)

    # Plot 0-line
    ax.axhline(y=0.0, linewidth=1.5, color='k', linestyle='--')

    # read the pypeit echelle file
    hdu = fits.open(filename)

    n_spec = hdu[0].header['NSPEC']
    target = hdu[0].header['TARGET']
    # instrument = hdu[0].header['INSTRUME']


    ylim_min = []
    ylim_max = []

    for order in range(1, n_spec+1):

        if order % 2 == 0:
            color = vermillion
        else:
            color = dblue

        wavelength = hdu[order].data['{}_WAVE'.format(ex_value)]
        if mask_values:
            mask = hdu[order].data['{}_MASK'.format(ex_value)]

        else:
            mask = np.ones_like(wavelength, dtype=bool)

        # masking the value and wavelength = 0
        wave_mask = wavelength > 1.0

        mask = np.logical_and(mask, wave_mask)


        if '{}_FLAM'.format(ex_value) in hdu[order].columns.names:
            flux = hdu[order].data['{}_FLAM'.format(ex_value)]
            flux_ivar = hdu[order].data['{}_FLAM_IVAR'.format(ex_value)]
            flux_sigma = hdu[order].data['{}_FLAM_SIG'.format(ex_value)]
        else:
            counts = hdu[order].data['{}_COUNTS'.format(ex_value)]
            counts_ivar = hdu[order].data['{}_COUNTS_IVAR'.format(ex_value)]
            counts_sigma = hdu[order].data['{}_COUNTS_SIG'.format(ex_value)]
            show = 'counts'

        if show == 'counts':
            if smooth is not None:
                counts = convolve(counts, Box1DKernel(smooth))
                counts_sigma /= np.sqrt(smooth)

            ax.plot(wavelength[mask], counts[mask], color=color)
            yy = counts[mask]
            if show_flux_err:
                ax.plot(wavelength[mask], counts_sigma[mask], color=color,
                        alpha=0.5)

        elif show == 'fluxden':
            if smooth is not None:
                flux = convolve(flux, Box1DKernel(smooth))
                flux_sigma /= np.sqrt(smooth)

            ax.plot(wavelength[mask], flux[mask], color=color, alpha=0.8)
            yy = flux[mask]
            if show_flux_err:
                ax.plot(wavelength[mask], flux_sigma[mask], color=color,
                        alpha=0.5)
        else:
            raise ValueError('Variable input show = {} not '
                             'understood'.format(show))

        percentiles = np.percentile(yy, [16, 84])
        median = np.median(yy)
        delta = np.abs(percentiles[1] - median)
        # print('delta', delta)
        # print('percentiles', percentiles)

        ylim_min.append(-0.5 * median)
        ylim_max.append(4 * percentiles[1])


    if show == 'counts':
        ax.set_ylabel(
            r'$\rm{Counts}\ [\rm{ADU}]$', fontsize=15)
    elif show == 'fluxden':
        ax.set_ylabel(
            r'$\rm{Flux}\ f_{\lambda}\ [\rm{erg}\,\rm{s}^{-1}\,\rm{cm}^{-2}\,'
            r'\rm{\AA}^{-1}]$',
            fontsize=15)
    else:
        raise ValueError('Variable input show = {} not '
                         'understood'.format(show))

    ax.set_xlabel(r'$\rm{Wavelength}\ [\rm{\AA}]$', fontsize=15)

    ax.set_ylim(min(ylim_min), max(ylim_max))

    # plt.title(r'{} {}'.format(target, instrument))

    plt.legend()

    plt.show()


def pypeit_multi_plot(filenames, show_flux_err=True, show_tellurics=False,
    mask_values=False, smooth=None, ymax=None):
    """Plot the spectrum assuming it is a pypeit spectrum

     """

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15, 7), dpi=140)
    fig.subplots_adjust(left=0.09, right=0.97, top=0.89, bottom=0.16)

    # Plot 0-line
    ax.axhline(y=0.0, linewidth=1.5, color='k', linestyle='--')

    max_limits = []

    for idx, filename in enumerate(filenames):

        color = color_list[idx]

        spec = SpecOneD()
        spec.read_pypeit_fits(filename)

        if mask_values:
            mask = spec.mask
        else:
            mask = np.ones(spec.dispersion.shape, dtype=bool)

        if smooth is not None and type(smooth) is int:
            spec.smooth(smooth, inplace=True)

        label = filename


        # Add second axis to plot telluric model
        if show_tellurics is True:
            telluric = spec.telluric[mask] / spec.telluric.max() * np.median(
                spec.fluxden[mask]) * 2.5
            ax.plot(spec.dispersion[mask], telluric[mask],
                         label='Telluric', color=color, alpha=0.5, ls='..')

        if show_flux_err:
            ax.plot(spec.dispersion[mask], spec.fluxden_err[mask], 'grey',
                    lw=1, label='Flux Error', color=color, alpha=0.5)


        ax.plot(spec.dispersion[mask], spec.fluxden[mask], 'k',
                linewidth=1, label=label, color=color)

        # # Add OBJ model if it exists
        # if hasattr(spec, 'obj_model'):
        #     ax.plot(spec.dispersion[mask], spec.obj_model, label='Obj '
        #                                                               'model')

        lim_spec = spec.copy()
        lim_spec.restore()
        lim_spec = lim_spec.mask_sn(5)
        lim_spec = lim_spec.sigmaclip_flux(3, 3)

        max_limits.append(lim_spec.flux[lim_spec.mask].max())

    if spec.unit == 'f_lam':
        ax.set_xlabel(r'$\rm{Wavelength}\ [\rm{\AA}]$', fontsize=15)
        ax.set_ylabel(
            r'$\rm{Flux}\ f_{\lambda}\ [\rm{erg}\,\rm{s}^{-1}\,\rm{cm}^{-2}\,'
            r'\rm{\AA}^{-1}]$',
            fontsize=15)

    elif spec.unit == 'f_nu':
        ax.set_xlabel(r'$\rm{Frequency}\ [\rm{Hz}]$', fontsize=15)
        ax.set_ylabel(
            r'$\rm{Flux}\ f_{\nu}\ [\rm{erg}\,\rm{s}^{-1}\,\rm{cm}^{-2}\,'
            r'\rm{Hz}^{-1}]$',
            fontsize=15)

    elif spec.unit == 'f_loglam':
        ax.set_xlabel(r'$\log\rm{Wavelength}\ [\log\rm{\AA}]$',
                           fontsize=15)
        ax.set_ylabel(
            r'$\rm{Flux}\ f_{\lambda}\ [\rm{erg}\,\rm{s}^{-1}\,\rm{cm}^{-2}\,'
            r'(\log\rm{\AA})^{-1}]$',
            fontsize=15)

    else:
        raise ValueError("Unrecognized units")


    ylim_min = 0
    if ymax == None:
        ylim_max = max(max_limits)
    else:
        ylim_max = ymax
    ax.set_ylim(ylim_min, ylim_max)
    ax.legend()
    plt.show()


def comparison_plot(spectrum_a, spectrum_b, spectrum_result,
                    show_flux_err=True):

    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(15,7), dpi=140)
    fig.subplots_adjust(left=0.09, right=0.97, top=0.89, bottom=0.16)

    ax1.plot(spectrum_a.dispersion, spectrum_a.flux, color='k')
    ax1.plot(spectrum_b.dispersion, spectrum_b.flux, color='r')

    ax2.plot(spectrum_result.dispersion, spectrum_result.flux, color='k')

    if show_flux_err:
        ax1.plot(spectrum_a.dispersion, spectrum_a.flux_err, 'grey', lw=1)
        ax1.plot(spectrum_b.dispersion, spectrum_b.flux_err, 'grey', lw=1)
        ax2.plot(spectrum_result.dispersion, spectrum_result.flux_err, 'grey', lw=1)

    if spectrum_result.unit=='f_lam':
        ax2.set_xlabel(r'$\rm{Wavelength}\ [\rm{\AA}]$', fontsize=15)
        ax1.set_ylabel(r'$\rm{Flux}\ f_{\lambda}\ [\rm{erg}\,\rm{s}^{-1}\,\rm{cm}^{2}\,\rm{\AA}^{-1}]$', fontsize=15)
        ax2.set_ylabel(r'$\rm{Flux}\ f_{\lambda}\ [\rm{erg}\,\rm{s}^{-1}\,\rm{cm}^{2}\,\rm{\AA}^{-1}]$', fontsize=15)

    elif spectrum_result.unit =='f_nu':
        ax2.set_xlabel(r'$\rm{Frequency}\ [\rm{Hz}]$', fontsize=15)
        ax1.set_ylabel(r'$\rm{Flux}\ f_{\nu}\ [\rm{erg}\,\rm{s}^{-1}\,\rm{cm}^{2}\,\rm{Hz}^{-1}]$', fontsize=15)
        ax2.set_ylabel(r'$\rm{Flux}\ f_{\nu}\ [\rm{erg}\,\rm{s}^{-1}\,\rm{cm}^{2}\,\rm{Hz}^{-1}]$', fontsize=15)

    else :
        raise ValueError("Unrecognized units")

    plt.show()
