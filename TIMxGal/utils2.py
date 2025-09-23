# BSD 3-Clause License
# 
# Copyright (c) 2024 Shubh Agrawal, Justin Bracks
# All rights reserved.

import numpy as np
import scipy
from E_lines import CII
from astropy.cosmology import Planck18 as cosmo
from astropy import constants as c, units as u
import camb
from scipy.interpolate import interp1d
from scipy.special import erf as Erf
h = cosmo.h


def erf_window(ks, kzMin):
    #erf approximation from Padmanabhan 2023
    # https://arxiv.org/abs/2212.08077
    epsilon = 1e-12
    ks[ks.value == 0] = epsilon*ks.unit
    vals = 1 - np.sqrt(np.pi)*kzMin/(2*ks) * Erf(ks/kzMin)
    return vals

def gaussian_window(k, sigma):
    """
    Returns the Gaussian window function for a given k-array and smoothing scale sigma.
    W(k) = exp(-k^2 * sigma^2 / 2)
    """
    epsilon = 1e-12
    k[k.value == 0] = epsilon*k.unit
    vals = np.exp(-k**2 * sigma**2 / 2)
    return vals

def luminosity_distance(z):
    """
    Returns the luminosity distance at redshift z using the current cosmology.

    Parameters
    ----------
    z : float or array-like
        Redshift(s) at which to compute the luminosity distance.

    Returns
    -------
    d_L : astropy.units.Quantity
        Luminosity distance in Mpc.
    """
    return cosmo.luminosity_distance(z)

def l2z(l_obs, l_emit):
    """
    Convert observed wavelength to redshift (default for CII emission line).

    Parameters:
    l_obs (float): The observed wavelength in micron.
    l_emit (float, optional): The rest-frame wavelength of the emission line in micron. 
            Defaults to lCII.

    Returns: (float) The redshift corresponding to the observed wavelength.
    """
    # if l_obs does not have units, add them
    if not hasattr(l_obs, "unit"):
        l_obs = l_obs * u.micron
    assert l_obs > l_emit, "Observed wavelength must be greater than rest-frame wavelength."
    return (l_obs - l_emit) / l_emit

def num_log_steps(x_start, x_stop, dlnx):
    """
    Calculate the number of steps needed to go from x_start to x_stop with a step size of dlnx.

    Parameters:
    x_start (float): The starting value.
    x_stop (float): The final value.
    dlnx (float): The step size in log space.
    
    Returns: (float) The number of steps needed to go from x_start to x_stop.
    """
    return np.log(x_stop / x_start) / np.log(1 + dlnx)

def transverse_scale(alpha, z_range, lilh=True):
    """
    Calculate the comoving transverse scale, using the provided 
    angular scale and redshift values

    Parameters:
    alpha (Quantity or float): Angular scale in radians.
    z_range (array-like or float): Redshift values.
    lilh (bool, optional): If True, use little-h units. Default is True.

    Returns (Quantity): Comoving transverse scale in Mpc.
    """
    if hasattr(alpha, "unit"):
        alpha = alpha.to(u.rad).value
    transLs = cosmo.comoving_transverse_distance(z_range) * alpha
    return transLs * h if lilh else transLs

def los_extent(z_min, z_max, lilh=True):
    """
    Calculate the comoving line-of-sight extent between two redshifts.

    Parameters:
    z_min (float): The lower redshift value.
    z_max (float): The upper redshift value.
    lilh (bool, optional): If True, use little-h units. Default is True.

    Returns (Quantity): Comoving line-of-sight extent in Mpc.
    """
    ext = cosmo.comoving_distance(z_max) - cosmo.comoving_distance(z_min)
    return ext * h if lilh else ext

def dnu2dr(dnu, nu_emit, z, lilh=True):
    """
    Calculate the comoving radial distance corresponding to a (observed) frequency interval.

    Parameters:
    dnu (Quantity or float): Frequency interval in Hz.
    z (float): Redshift value.
    lilh (bool, optional): If True, use little-h units. Default is True.
    nu_emit (Quantity or float, optional): Rest-frame frequency of the emission line in Hz.
            Defaults to nuCII.

    Returns (Quantity): Comoving radial distance in Mpc.
    """
    if hasattr(dnu, "unit"):
        dnu = dnu.to(u.Hz).value
    if hasattr(nu_emit, "unit"):
        nu_emit = nu_emit.to(u.Hz).value
    nu_obs = nu_emit / (1 + z)
    dR = cosmo.comoving_distance(z) - cosmo.comoving_distance(nu_emit / (nu_obs + dnu) - 1)
    return dR * h if lilh else dR

def volume_cube(z_min, z_max, d_omega, lilh=True):
    """
    Calculate the cosmological volume enclosed between z_min, z_max, for 
    a given solid angle d_omega.

    Parameters:
    z_min (float): Redshift at the near face.
    z_max (float): Redshift at the farther face.
    d_omega (float): Solid angle in steradians.
    lilh (bool, optional): If True, use little-h units. Default is True.

    Returns (Quantity): Survey volume in Mpc^3.
    """
    volume_shell = cosmo.comoving_volume(z_max) - cosmo.comoving_volume(z_min)
    sr_ratio = d_omega / (4 * np.pi * u.sr)
    return volume_shell * sr_ratio * h**3 if lilh else volume_shell * sr_ratio

def volume_scan(z_min, z_max, d_az, d_el, lilh=True):
    """
    Calculate the cosmological volume enclosed between z_min, z_max, for 
    a given angular resolution.

    Parameters:
    z_min (float): Redshift at the near face.
    z_max (float): Redshift at the farther face.
    d_az (float): Azimuthal edge in radians.
    d_el (float): Elevation edge in radians.
    lilh (bool, optional): If True, use little-h units. Default is True.

    Returns (Quantity): Survey volume in Mpc^3.
    """
    return volume_cube(z_min, z_max, d_az * d_el, lilh)

def area_scan(z, az, el, lilh=True):
    """
    Calculate the transverse area enclosed by az and el at redshift z.

    Parameters:
    z (float or array-like) : redshift value(s)
    az (float) : azimuthal scale/range in radians.
    el (float) : elevation scale/range in radians.
    lilh (bool, optional): If True, use little-h units. Default is True.

    Returns (float): The transverse area enclosed by az and el at redshift z.
    """
    return transverse_scale(az, z, lilh) * transverse_scale(el, z, lilh) # TODO: this is wrong in Juzz's code, extra h**2?

def num_modes(del_ln_k, k, volume): # TODO: unused
    """
    Calculate the number of modes in a given volume.

    Parameters:
    del_ln_k (float): The logarithmic step size in k-space.
    k (array-like or float): The wavenumber values.
    volume (Quantity): survey volume in Mpc^3.

    Returns (float): The number of modes in the given volume.
    """
    # return volume * np.log(k[-1] / k[0]) / np.log(1 + del_ln_k)
    return 4 * np.pi * volume * del_ln_k * k ** 3 / (2 * np.pi) ** 3 #TODO: Check this


def noise_per_cell(z, nei, num_spax, num_dets, time, dAz, dEl, dnu, nuRest, lilh=True):
    """
    Calculate the noise per cell in the survey.

    Parameters:
    z (float): Redshift value.
    nei (float): Noise equivalent intensity.
    num_spax (float): Number of spaxels.
    time (float): Integration time in seconds.
    dAz (float): Azimuthal resolution in radians.
    dEl (float): Elevation resolution in radians.
    dnu (float): Frequency resolution in Hz.
    lilh (bool, optional): If True, use little-h units. Default is True.

    Returns (Quantity): Noise per cell.
    """

    dR = dnu2dr(dnu, nuRest, z, lilh)
    cell_area = area_scan(z, dAz, dEl, lilh)
    cell_vol = cell_area * dR
    cell_time = time / num_spax * num_dets
    return (nei * np.sqrt(cell_vol / cell_time))** 2 #TODO should this be squared?

def MD_sfrd(z):
    """
    Calculate the star formation rate density at redshift z, using the Madau-Dickinson model.
    EQ15 from https://arxiv.org/pdf/1403.0007

    Parameters:
    z (float or array-like): Redshift value.

    Returns (float): Star formation rate density in Msun/yr/Mpc^3.
    """
    val = 0.015 * (1 + z) ** 2.7 / (1 + ((1 + z) / 2.9) ** 5.6)
    return val * u.Msun / u.yr / u.Mpc ** 3

def calc_k_modes(Da, da):
    """
    Calculate the k modes for a given extent and resolution element.
    
    Parameters:
    - Da (float): extent in a direction.
    - da (float): resolution element in the same direction.
    
    Returns:
    - k_modes (ndarray): The calculated k modes.
    """

    #units are hard coded in the return statement. TODO: Build in unit check. 
    return 2 * np.pi * np.fft.fftfreq(int(Da.value / da.value), da.value) * u.Mpc**(-1)

def I_Bracks25(z, sfrd, nu_emit, L0=10.1 * 10**6 * u.Lsun * u.yr / u.Msun, salpeterConversion = False):
    """
    Calculate the specific intensity of a source at redshift z, using Juzz's proposal formalization.

    Parameters:
    sfrd (float): Star formation rate density in Msun/yr/Mpc^3.
    z (float or array-like): Redshift value.
    L0 (float): ratio of line intensity to LIR.
    nu_emit (float, optional): Rest-frame frequency of the emission line in Hz. 
            Defaults to nuCII.

    Returns (Quantity): Specific intensity in Jy/sr.
    """
    # TODO: try Ryan's various versions for SFRD --> LCII
    # TODO: L0 definition does not make sense: taken after some computation from DeLooze?
    #nu_emit=self.EmissionLine.nu
    if not hasattr(sfrd, "unit"):
        sfrd *= u.Msun / u.yr / u.Mpc ** 3
    if not hasattr(nu_emit, "unit"):
        nu_emit *= u.Hz
    eps = L0 * sfrd 
    Ival = c.c * eps / (4 * np.pi * u.sr * cosmo.H(z) * nu_emit)

    if salpeterConversion: 
            Ival = Ival * .86 #Converting from salpeter IMF (used in MD14) to Modern IMF.
    return Ival.to(u.Jy / u.sr)


# TODO: Currently this codebase will not properly carry little h units across function calls
# This will not be an issue if the user leaves lilh as True (allows everything to be calculated
# in little h units), becuase it will default to True any way. I'll need to track all the possible 
# pathways that lilh can be passed and ensure that the decision of whether to use little h units 
# is made at a single point and carried through automatically thereafter.

class EmissionLine:
    """
    Represents an astronomical emission line with its rest-frame wavelength and name.

    Parameters
    ----------
    restWaveL : astropy.units.Quantity
        The rest-frame wavelength of the emission line (e.g., in microns).
    name : str, optional
        The name of the emission line (e.g., 'CII').

    Attributes
    ----------
    restWaveL : astropy.units.Quantity
        The rest-frame wavelength of the emission line.
    nu : astropy.units.Quantity
        The rest-frame frequency of the emission line, automatically calculated from restWaveL.
    name : str
        The name of the emission line.

    Notes
    -----
    Additional properties (such as mean intensity or redshift) can be added as needed.
    """
    def __init__(self, restWaveL, name = None):
        self.restWaveL = restWaveL #Todo: add unit check
        self.nu = self.restWaveL.to(u.Hz, equivalencies=u.spectral())
        self.name = name
        #TODO: add mean intensity/z tuple. 

class Telescope:
    """
    Represents a telescope used in a survey, including its primary mirror diameter and beam attenuation.

    Parameters
    ----------
    primary : float or astropy.units.Quantity
        Diameter of the primary mirror (in meters).
    beamAtten : callable or array-like
        Beam-dependent attenuation function or array to be applied to mode count.
    name : str, optional
        Name of the telescope.

    Attributes
    ----------
    primary : float or astropy.units.Quantity
        Diameter of the primary mirror.
    beamAtten : callable or array-like
        Beam attenuation function or array.
    name : str
        Name of the telescope.
    """
    def __init__(self, primary, beamAtten, name = None):
        self.primary = primary # diameter in m of primary mirror
        self.beamAtten = beamAtten #Beam-dependent attenuation function to be applied to mode count.
        self.name = name

class Instrument:
    """
    Represents an instrument used in a survey, including its bandwidth, noise equivalent intensity,
    frequency resolution, and number of detectors.

    Parameters
    ----------
    bandWidth : tuple or array-like
        The minimum and maximum wavelength or frequency range covered by the instrument.
    nei : float
        Noise equivalent intensity (NEI) of the instrument.
    dnu : float
        Frequency resolution in Hz.
    num_dets : int
        Number of detectors in the instrument.
    name : str, optional
        Name of the instrument.

    Attributes
    ----------
    bandWidth : tuple or array-like
        The wavelength or frequency range covered by the instrument.
    NEI : float
        Noise equivalent intensity.
    dnu : float
        Frequency resolution in Hz.
    num_dets : int
        Number of detectors.
    name : str
        Name of the instrument.

    Methods
    -------
    get_band_edges(n)
        Divide the instrument's bandwidth into n bins and return the bin edge values.
    """
    def __init__(self, bandWidth, nei, dnu, num_dets, name = None):

        self.bandWidth = bandWidth
        self.NEI = nei
        self.dnu = dnu
        self.num_dets = num_dets
        self.name = name

    def get_band_edges(self, n):
        """
        Divide the instrument's bandWidth into n bins and return the bin edge values.

        Parameters
        ----------
        n : int
            Number of bins.

        Returns
        -------
        edges : np.ndarray
            Array of bin edge values of length n+1.
        """
        # Assumes bandWidth is a tuple or array-like: (min, max)
        lmin, lmax = self.bandWidth
        return np.linspace(lmin, lmax, n+1)

class zBin: #TODO add strict unit checks
    """
    Represents a redshift bin for a survey, defined by its front and back wavelengths, and associated telescope, instrument, and emission line.

    Parameters
    ----------
    Telescope : Telescope
        The telescope object used for the survey.
    Instrument : Instrument
        The instrument object used for the survey.
    EmissionLine : EmissionLine
        The emission line object being observed.
    lamFront : float or astropy.units.Quantity
        The observed wavelength at the front edge of the bin (in microns).
    lamBack : float or astropy.units.Quantity
        The observed wavelength at the back edge of the bin (in microns).

    Attributes
    ----------
    Telescope : Telescope
        The telescope used for the bin.
    Instrument : Instrument
        The instrument used for the bin.
    EmissionLine : EmissionLine
        The emission line observed in the bin.
    zFront : float
        The redshift at the 'front' (nearest to viewer) edge of the bin.
    zBack : float
        The redshift at the 'back' (farthest from viewer) edge of the bin.
    zCenter : float
        The central redshift of the bin.
    FWHM : astropy.units.Quantity
        The full width at half maximum (FWHM) angular resolution for the bin.
    dOmega : astropy.units.Quantity
        The solid angle of the bin.

    Methods
    -------
    SFRD()
        Returns the star formation rate density at the central redshift.
    vCube(dOmega, lilh=True)
        Calculates the volume of the bin for a given solid angle.
    vScan(dAz, dEl, lilh=True)
        Calculates the volume of the bin for given angular resolutions.
    LoSmax(lilh=True)
        Calculates the maximum line-of-sight extent of the bin.
    LoSmin(dnu, lilh=True)
        Calculates the minimum line-of-sight extent (resolution) of the bin.
    transScaleFront(angle)
        Calculates the transverse scale at the front redshift.
    transScaleBack(angle)
        Calculates the transverse scale at the back redshift.
    transScale(angle)
        Calculates the transverse scale at the central redshift.
    """

    def __init__(self, Telescope, Instrument, EmissionLine, lamFront, lamBack):

        self.Telescope = Telescope
        self.Instrument = Instrument
        self.EmissionLine = EmissionLine
        self.zFront = l2z(lamFront, EmissionLine.restWaveL)
        self.zBack = l2z(lamBack, EmissionLine.restWaveL)
        self.zCenter = (self.zFront + self.zBack) / 2
        self.FWHM = (1.22 * ((lamFront+lamBack)/2)/ Telescope.primary).decompose() * u.rad
        self.dOmega = self.FWHM ** 2
        
    # TODO:all these methods take exisiting valued arguments. I should code this such that it (kwarg) defaults 
    # to the automatically calculated values but then you can override to test different values.
    def SFRD(self):
        """
        Returns the star formation rate density (SFRD) at the central redshift of the bin.

        Uses the Madau-Dickinson model to estimate the SFRD for the bin's central redshift.

        Returns
        -------
        sfrd : astropy.units.Quantity
            Star formation rate density in units of Msun/yr/Mpc^3.
        """
        sfrd = MD_sfrd(self.zCenter)
        return sfrd
    
    def vCube(self, dOmega, lilh = True):
        """
        Calculate the cosmological volume of the redshift bin for a given solid angle.

        Parameters
        ----------
        dOmega : float or astropy.units.Quantity
            Solid angle in steradians.
        lilh : bool, optional
            If True, use little-h units for the output volume. Default is True.

        Returns
        -------
        volume : astropy.units.Quantity
            The survey volume in Mpc^3 (or Mpc^3 h^-3 if lilh=True).
        """
        return volume_cube(self.front, self.back, dOmega, lilh)
    def vScan(self, dAz, dEl, lilh = True):
        """
        Calculate the volume of the redshift bin for a given angular resolution.

        Parameters
        ----------
        dAz : float or astropy.units.Quantity
            Angular resolution in the azimuthal direction.
        dEl : float or astropy.units.Quantity
            Angular resolution in the elevation direction.
        lilh : bool, optional
            If True, use little-h units for the output volume. Default is True.

        Returns
        -------
        volume : astropy.units.Quantity
            The survey volume in Mpc^3 (or Mpc^3 h^-3 if lilh=True).
        """
        return volume_scan(self.front, self.back, dAz, dEl, lilh)
    def LoSmax(self, lilh = True):
        """
        Calculate the maximum comoving line-of-sight extent of the redshift bin.

        Parameters
        ----------
        lilh : bool, optional
            If True, return the result in little-h units (Mpc h^-1). Default is True.

        Returns
        -------
        extent : astropy.units.Quantity
            The comoving line-of-sight distance between the front and back redshift edges of the bin.
        """
        return los_extent(self.zFront, self.zBack, lilh)
    def LoSmin(self, dnu, lilh = True):
        """
        Calculate the minimum comoving line-of-sight extent (resolution) of the redshift bin,
        corresponding to the instrument's frequency resolution.

        Parameters
        ----------
        dnu : float or astropy.units.Quantity
            Frequency resolution in Hz.
        lilh : bool, optional
            If True, return the result in little-h units (Mpc h^-1). Default is True.

        Returns
        -------
        extent : astropy.units.Quantity
            The minimum comoving line-of-sight distance for the bin.
        """
        return dnu2dr(dnu, self.EmissionLine.nu, self.zCenter, lilh)
    def transScaleFront(self, angle):
        """
        Calculate the comoving transverse scale corresponding to a particular angle
        at the redshift corresponding to the nearest LoS edge of the bin.

        Parameters
        ----------
        angle : float or astropy.units.Quantity
            Angular scale in radians.

        Returns
        -------
        scale : astropy.units.Quantity
            Comoving transverse scale at the front redshift, in Mpc (or Mpc h^-1 if little-h units are used).
        """
        return transverse_scale(angle, self.zFront)
    def transScaleBack(self, angle):
        """
        Calculate the comoving transverse scale corresponding to a particular angle
        at the redshift corresponding to the furthest LoS edge of the bin.

        Parameters
        ----------
        angle : float or astropy.units.Quantity
            Angular scale in radians.

        Returns
        -------
        scale : astropy.units.Quantity
            Comoving transverse scale at the back redshift, in Mpc (or Mpc h^-1 if little-h units are used).
        """
        return transverse_scale(angle, self.zBack)
    def transScale(self, angle):
        """
        Calculate the comoving transverse scale corresponding to a particular angle
        at the redshift corresponding to the center of the bin.

        Parameters
        ----------
        angle : float or astropy.units.Quantity
            Angular scale in radians.

        Returns
        -------
        scale : astropy.units.Quantity
            Comoving transverse scale at the center redshift, in Mpc (or Mpc h^-1 if little-h units are used).
        """
        return transverse_scale(angle, self.zCenter)

class LIM_survey():
    """
    Represents a line intensity mapping (LIM) survey configuration, including telescope, instrument,
    redshift bin, field dimensions, and survey time.

    Parameters
    ----------
    Telescope : Telescope
        The telescope used for the survey as a Telescope object.
    Instrument : Instrument
        The instrument used for the survey as an Instrument object.
    zBin : zBin
        The redshift bin object defining the survey's wavelength/redshift coverage.
    AZ : float or astropy.units.Quantity
        Survey field azimuthal extent (in degrees.
    EL : float or astropy.units.Quantity
        Survey field elevation extent (in degrees.
    sTime : float
        Total survey integration time (in seconds).
    lilh : bool, optional
        If True, use little-h units for calculations. Default is True.

    Attributes
    ----------
    Telescope : Telescope
        The telescope used for the survey as a Telescope object.
    Instrument : Instrument
        The instrument used for the survey as an Instrument object.
    zBin : zBin
        The redshift bin object as a zBin object.
    EmissionLine : EmissionLine
        The emission line being mapped.
    Az : float or astropy.units.Quantity
        Azimuthal field extent.
    El : float or astropy.units.Quantity
        Elevation field extent.
    sTime : float
        Survey integration time.
    lilh : bool
        Whether little-h units are used.
    num_spax : float
        Number of spatial pixels (spaxels) in the survey field.

    Methods
    -------
    cell_noise(lilh=True)
        Calculate the noise per cell in the survey, assuming uniform angular resolution.
    """    
    def __init__(self, Telescope, Instrument, zBin, AZ, EL, sTime, lilh=True):

        self.Telescope = Telescope
        self.Instrument = Instrument
        self.zBin = zBin
        self.EmissionLine = zBin.EmissionLine
        self.Az = AZ
        self.El = EL
        self.sTime = sTime
        self.lilh = lilh
        
        self.num_spax = (AZ * EL / self.zBin.FWHM ** 2).decompose() #TODO: add unit hasattr+checker

    def vVox(self):
        """
        Calculate the volume of a single voxel (3D cell) in the survey.

        Returns
        -------
        v_vox : astropy.units.Quantity
            The volume of a single voxel in Mpc^3 (or Mpc^3 h^-3 if little-h units are used).
        """
        area = area_scan(self.zBin.zCenter, self.zBin.FWHM, self.zBin.FWHM, self.lilh)
        los = self.zBin.LoSmin(self.Instrument.dnu, self.lilh)
        return (area * los).decompose()
    
    def cell_noise(self, lilh=True):
        """
        Calculate the noise per cell (voxel) in the LIM survey.

        This method computes the noise for a single spatial-spectral cell, assuming uniform angular resolution
        in both azimuth and elevation directions, and using the instrument's frequency resolution and survey parameters.

        Parameters
        ----------
        lilh : bool, optional
            If True, use little-h units for the output noise value. Default is True.

        Returns
        -------
        noise : astropy.units.Quantity
            The noise per cell in the survey, in appropriate intensity units.
        """
        return noise_per_cell(self.zBin.zCenter, self.Instrument.NEI, self.num_spax, 
                            self.Instrument.num_dets, self.sTime, self.zBin.FWHM, self.zBin.FWHM,
                            self.Instrument.dnu, self.EmissionLine.nu, lilh)


class GAL_survey():
    def __init__(self, nGals, specRatio):
        self.nGals = nGals
        self.specRatio = specRatio #This is a temporary hack and needs to be generalized


class LIMxGAL():
    """
    Represents the cross-correlation between a line intensity mapping (LIM) survey and a galaxy (GAL) survey.

    This class calculates the k-space modes, transfer functions, and provides methods for computing
    effective mode counts, auto/cross power spectra, and signal-to-noise ratios for the joint survey.

    Parameters
    ----------
    LIM_survey : LIM_survey
        The line intensity mapping survey object.
    GAL_survey : GAL_survey
        The galaxy survey object.
    dlnk : float, optional
        Logarithmic bin width in k-space. Default is 1.
    window : str or None, optional
        Type of window function to apply (e.g., 'Gaussian'). Default is None.

    Attributes
    ----------
    LIM : LIM_survey
        The LIM survey object.
    GAL : GAL_survey
        The galaxy survey object.
    window : str or None
        Window function type.
    k_xyzs : np.ndarray
        3D array of k-space vectors.
    k_mags : np.ndarray
        Magnitude of k-vectors.
    k_min : float
        Minimum k value.
    k_max : float
        Maximum k value.
    num_kbins : int
        Number of k bins.
    transferCoefs : np.ndarray
        Transfer function coefficients for the cross-correlation.
    LIMTransCoefs : np.ndarray
        Transfer function coefficients for the LIM survey only.

    Methods
    -------
    nMode_Effective(PmTuple, returnTransferAve=False, returnLIMTransferAve=False)
        Calculates the effective number of modes and optionally returns transfer function averages.
    LIM_Auto_Power(pmTuple, bLine, shotLine)
        Computes the LIM auto-power spectrum, noise, and mean intensity.
    xCorr(PmTuple, bLine, bGal, shotLine, fs, returnFull=False)
        Computes the cross-power spectrum, variance, and related quantities for the LIM-GAL survey pair.
    """
        
    def __init__(self, LIM_survey, GAL_survey, dlnk = 1, window = None):
    #This currently assumes the GAL_survey object is populated specifically ford its redshift 
    # and field overlap with the LIM survey TODO: write code to automatically generate a
    #survey object that finds overlap and constrains automatically.

        self.LIM = LIM_survey
        self.GAL = GAL_survey
        self.window = window

        """
            calculate the k-modes in the survey.
        """
    def calculate_k_xyzs(self):
        Dx = self.LIM.zBin.transScale(self.LIM.Az)
        Dy = self.LIM.zBin.transScale(self.LIM.El)
        Dz = self.LIM.zBin.LoSmax()

        dx = transverse_scale(self.LIM.zBin.FWHM, self.LIM.zBin.zCenter) #This assumes a square field.
        dy = transverse_scale(self.LIM.zBin.FWHM, self.LIM.zBin.zCenter) #This assumes a square field.
        dz = self.LIM.zBin.LoSmin(self.LIM.Instrument.dnu)

        kx, ky, kz = np.meshgrid(calc_k_modes(Dx, dx), calc_k_modes(Dy, dy), 
                                calc_k_modes(Dz, dz), indexing='ij')

        k_xyzs = np.stack((kx, ky, kz))
        return k_xyzs
    
    def setup_k_modes(self, dlnk=1):
        k_xyzs = self.calculate_k_xyzs()

        k_mags = np.sqrt(np.sum(k_xyzs**2, axis=0))
        del k_xyzs
        k_min = np.min(k_mags[k_mags != 0])
        k_max = np.max(k_mags)
        #self.k_props = np.abs(k_xyzs / self.k_mags)
        num_kbins = np.round(num_log_steps(k_min, np.max(k_max), dlnk)) + 1

        return k_mags, num_kbins

    def setup_transfer_function(self, auto = False): 
        k_xyzs = self.calculate_k_xyzs()

        kxMin = 2 * np.pi / self.LIM.zBin.transScale(self.LIM.Az)
        kyMin = 2 * np.pi / self.LIM.zBin.transScale(self.LIM.El)
        kzMin = 2 * np.pi / self.LIM.zBin.LoSmax()

        Lx = self.LIM.zBin.transScale(self.LIM.Az)
        Ly = self.LIM.zBin.transScale(self.LIM.El)
        Lz = self.LIM.zBin.LoSmax()

        kxs = k_xyzs[0][:, 0, 0]
        kys = k_xyzs[1][0, :, 0]
        kzs = k_xyzs[2][0, 0, :]
        del k_xyzs

        sigCoef = np.sqrt(8*np.log(2))
        #TODO:this currently ignores contributions from the galaxy survey to transverse sigma.
        sPar = dnu2dr(self.LIM.Instrument.dnu, self.LIM.EmissionLine.nu,
                        self.LIM.zBin.zCenter) / sigCoef
        sPerp = self.LIM.zBin.transScale(self.LIM.zBin.FWHM) / sigCoef
        galPar  = self.GAL.specRatio *  sPar

        #def erf_window(k, L):
            # Damps modes with k <~ 2pi/L
            #return scipy.special.erf(k * L / 2.0)

        if self.window == 'Padmanabhan2023':
            #print('Applying Padmanabhan 2023 windowing')
            if auto:
                LIMTransCoefs = np.einsum(
                    'i,j,k->ijk',
                    gaussian_window(kxs, sPerp) * erf_window(kxs, kzMin),
                    gaussian_window(kys, sPerp) * erf_window(kys, kzMin),
                    gaussian_window(kzs, sPar) * erf_window(kzs, kzMin) 
                    )
                return LIMTransCoefs
            else:
                transferCoefs = np.einsum(
                    'i,j,k->ijk',
                    (gaussian_window(kxs, sPerp) * erf_window(kxs, kzMin))**0.5,
                    (gaussian_window(kys, sPerp) * erf_window(kys, kzMin))**0.5,
                    (gaussian_window(kzs, sPar) * gaussian_window(kzs, galPar) * erf_window(kzs, kzMin))**0.5
                    )
                return transferCoefs
            
        elif self.window == 'GaussianOnly':
            #print('Applying Gaussian windowing')

            #print(f"sPar: {sPar}, sPerp: {sPerp}, galPar: {galPar}")
            if auto:
            #TODO: Generalize - this'll only work for the TIMxEu case or very similar.
            
                LIMTransCoefs = np.einsum('i,j,k->ijk', 
                        gaussian_window(kxs, sPerp),
                        gaussian_window(kys, sPerp),
                        gaussian_window(kzs, sPar))
                return LIMTransCoefs
            else:
                transferCoefs = np.einsum('i,j,k->ijk', 
                    gaussian_window(kxs, sPerp)**0.5,
                    gaussian_window(kys, sPerp)**0.5,
                    (gaussian_window(kzs, sPar) * gaussian_window(kzs, galPar))**0.5
                    )
                return transferCoefs

    def k_props(self):
        k_xyzs = self.calculate_k_xyzs()
        #k_xyzs = np.stack((kx, ky, kz))
        k_mags, nModes = self.setup_k_modes()
        return np.abs(k_xyzs / k_mags)

    def nMode_Effective(self, PmTuple, returnTransferAve = False, returnLIMTransferAve = False):
        #TODO: this doesn't cover the case that both return args = True. I should write that case.
        k_mags, nModes = self.setup_k_modes()
        transferCoefs = self.setup_transfer_function(auto=False)
        LIMTransCoefs = self.setup_transfer_function(auto=True)
        kflat = k_mags.value.flatten()
        wflat = transferCoefs.flatten()
        (kPms, Pm) = PmTuple
        kbin_edges = np.append([0], np.sqrt(kPms * np.append(kPms[1:], [np.max(k_mags.value)])))
        
        # get bin edges by selecting the midway between each CAMB k prediction
        # use geometric mean because bins are log-spaced

        
        nEffs = []
        for i in np.arange(len(kbin_edges)-1):
            if i+1 == len(kbin_edges):
                wSum = (np.sum(wflat[kflat>kbin_edges[i]]))
            else:
                wSum = (np.sum(wflat[(kflat>kbin_edges[i]) & (kflat<=kbin_edges[i+1])]))
            nEffs.append(wSum/2.) # accounting for double counting.

        if returnTransferAve:
            #for fig 6 - transfer function averages per bin
            transferAves = np.array([np.mean(transferCoefs[np.logical_and(k_mags.value < kbin_edges[i+1], k_mags.value >= kbin_edges[i])]) 
                for i in range(len(kbin_edges)-1)]) 
            return kbin_edges, np.asarray(nEffs), np.asarray(transferAves)

        elif returnLIMTransferAve:
            LIMTransferAves = np.array([np.mean(LIMTransCoefs[np.logical_and(k_mags.value < kbin_edges[i+1], k_mags.value >= kbin_edges[i])]) 
                for i in range(len(kbin_edges)-1)]) 
            return kbin_edges, np.asarray(nEffs), np.asarray(LIMTransferAves)

        else: return kbin_edges, np.asarray(nEffs)

    def LIM_Auto_Power(self, pmTuple, bLine, shotLine): #Assumes using Juzz's formalism for ILine
        ILine = I_Bracks25(self.LIM.zBin.zCenter, self.LIM.zBin.SFRD(), self.LIM.EmissionLine.nu)
        Pll = (bLine**2 * ILine.value**2 * pmTuple[1] + shotLine)
        NLIM = self.LIM.cell_noise()#.value
        return Pll, NLIM, ILine

    def xCorr(self, PmTuple, bLine, bGal, shotLine, fs, returnFull = False):
        Pll, NLIM, ILine = self.LIM_Auto_Power(PmTuple, bLine, shotLine)
        kEdges, nEff, LIMTransferAves = self.nMode_Effective(PmTuple, returnLIMTransferAve=True)
        nGal = self.GAL.nGals
        Pgg = 1/nGal + bGal**2 * PmTuple[1]
        x_shots = fs * ILine / self.GAL.nGals

        PxG = (bGal * bLine * ILine * PmTuple[1]).value + x_shots.value
        xVar = (PxG**2 + (Pgg + 1/nGal) * (Pll/LIMTransferAves**2 + NLIM.value))/(2*nEff)
        if returnFull:
            return PxG, xVar, x_shots, Pll, NLIM, ILine, nEff
        return PxG, xVar, x_shots
    

# ----------- Functions for multi-sub-survey work (primarily to be used with CAMB) ---------
def nEffs_multi(survList, Pms): #TODO: make return transfer aves a kwarg.
    kbin_edges_list = []
    nEffs_list = []
    transferAves_list = []

    for i in range(len(survList)):
        kbin_edges, nEffs, transferAves = survList[i].nMode_Effective(Pms[i], returnTransferAve=True)
        kbin_edges_list.append(kbin_edges)
        nEffs_list.append(nEffs)
        transferAves_list.append(transferAves)

    return kbin_edges_list, nEffs_list, transferAves_list

def kSpecs_multi(surveys, returnVal = 'magnitude'):

    zShifts = [surv.LIM.zBin.zCenter for surv in surveys]
    k_mags = [surv.setup_k_modes()[0] for surv in surveys]
    num_kbins = [surv.setup_k_modes()[1] for surv in surveys]
    k_mins = [np.min(k_mag[k_mag != 0]) for k_mag in k_mags]
    k_maxs = [np.max(k_mag) for k_mag in k_mags]

    if returnVal == 'magnitude': return k_mags
    elif returnVal == 'maxima': return k_maxs
    elif returnVal == 'minima': return k_mins
    elif returnVal == 'num_bins': return num_kbins
    elif returnVal == 'z_shifts': return zShifts
    elif returnVal == 'all':
        return k_mags, k_maxs, k_mins, num_kbins, zShifts 
    else:
        raise ValueError("Invalid returnVal specified.")
    
def kSpecs_single(survey, returnVal = 'magnitude'):

    zShift = survey.LIM.zBin.zCenter
    k_mag, num_kbin  = survey.setup_k_modes()
    k_min = np.min(k_mag[k_mag != 0])
    k_max = np.max(k_mag)

    if returnVal == 'magnitude': return k_mag
    elif returnVal == 'maxima': return k_max
    elif returnVal == 'minima': return k_min
    elif returnVal == 'num_bins': return num_kbin
    elif returnVal == 'z_shifts': return zShift
    elif returnVal == 'all':
        return k_mag, k_max, k_min, num_kbin, zShift 
    else:
        raise ValueError("Invalid returnVal specified.")

def CAMB_Pm_multi(k_mags, k_maxs, k_mins, num_kbins, zShifts, cosmology = cosmo):
    kmax=np.max(k_maxs)
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=cosmology.H0.value, ombh2=0.022, omch2=0.122, mnu=0.06, omk=0, tau=0.06)
    # TODO:Cosmo params are largely hardcoded - should be made more flexible
    pars.InitPower.set_params(ns=0.965, As=2e-9, r=0)
    pars.set_for_lmax(2500, lens_potential_accuracy=0)
    pars.set_matter_power(redshifts=zShifts, kmax=kmax * 3, nonlinear=False) #TODO: Why is this * 3 ?
    results = camb.get_results(pars)

    binned_results = [results.get_matter_power_spectrum(minkh=kmin, maxkh=np.max(km.value), npoints=int(nkb.value)) \
        for kmin, km, nkb in zip(k_mins, k_mags, num_kbins)]
    z_idxs = [np.argmin(np.abs(zs - binz)) for (_, zs, _), binz in zip(binned_results, zShifts)]
    return [(ks, Pm[z_idx]) for (ks, _, Pm), z_idx in zip(binned_results, z_idxs)]

def CAMB_Pm_single(k_mag, k_max, k_min, num_kbin, zShift, cosmology = cosmo):
    kmax=k_max
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=cosmology.H0.value, ombh2=0.022, omch2=0.122, mnu=0.06, omk=0, tau=0.06)
    # TODO:Cosmo params are largely hardcoded - should be made more flexible
    pars.InitPower.set_params(ns=0.965, As=2e-9, r=0)
    pars.set_for_lmax(2500, lens_potential_accuracy=0)
    pars.set_matter_power(redshifts=zShift, kmax=kmax * 3, nonlinear=False) #TODO: Why is this * 3 ?
    results = camb.get_results(pars)
    binned_results = results.get_matter_power_spectrum(minkh=k_min, maxkh=np.max(k_mag.value), npoints=int(num_kbin.value))
    #z_idx = np.argmin(np.abs(zShift - results.get_redshifts()))
    return (binned_results)#, z_idx)

def SNR_interp_multi(Pms, PxGs, VARs, baseIDX):
    SNRs = [PxG / np.sqrt(VAR)  for PxG, VAR in zip(PxGs, VARs)] # Calculate SNRs
    powerInterp = [interp1d(ks, PxG) for (ks, _), PxG in zip(Pms, PxGs)] #Create interpolation functions for power spectra
    SNRInterp = [interp1d(ks, SNR) for (ks, _), SNR in zip(Pms, SNRs)]# Create interpolation functions for SNRs
    noiseInterp = [interp1d(ks, np.sqrt(VAR)) for (ks, _), VAR in zip(Pms, VARs)] # Create interpolation functions for noise

    kBase = Pms[baseIDX][0] #isolate the list of k modes in bin baseIDX. We'll use that as our default k list. 
    aligned_PxGs = [interp(kBase) for interp in powerInterp] #Interpolate the forecast signals in all bins at the modes in kBase.
    aligned_SNRs = np.nan_to_num([interp(kBase) for interp in SNRInterp]) #same for SNRs - nanToNum sets NaNs = 0.
    aligned_x_noises = [interp(kBase) for interp in noiseInterp]#and x_noises.
    return aligned_PxGs, aligned_SNRs, aligned_x_noises