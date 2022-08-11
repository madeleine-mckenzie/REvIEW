#from REvIEW_imports import *
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import numpy as np
import pandas as pd

from scipy.stats import chisquare
from scipy.optimize import curve_fit
from scipy import interpolate

from NN.ffnn import FFNN
from NN.scalar import Scalar


def velocity_correction(wavelength, rv):
    ''' 
    Transforming the wavelengths in velocity space

    Lets hope my algebra is right. Should take the velocity and scale it depending on the rv_correction array that should be defined at the top of the notebook.

    Parameters
    ----------
    wavelength : array
        The array you want to scale
    rv : int
        The radial velocity correction

    Returns
    -------
    arr
        The wavelength in the rest frame
    '''

    return wavelength * (1+(-rv/300000))

def get_gaussian_area(a, c):
    '''
    Calculates the area of a gaussian given a and c

    Parameters
    ----------
    a : float
        Amplitude
    c : int
        standard deviation
    
    Returns
    -------
    float : gaussian area
    '''
    return a * np.sqrt(2 * np.pi * c**2)* 1000

#############################################
#                                           #
#  Functions for fitting the spectra        #
#                                           #
#############################################

def read_spectra(filename, rv_correction, spectra_path, ew_path):
    ''' 
    Reads in the spectra of the star.

    Doing this for each star for each plot. Trading in memory for time but could change around eventually...

    Parameters
    ----------
    filename : string
        The name of the spectra to be read in
    filename : int
        The value that the wavelength needs to be scaled by to be in the rest frame
    spectra_path : string
        The path to the spectra
    ew_path : string
        The path to where im putting the ew output files


    Returns
    -------
    pandas dataframe
        the read in pandas data frame that has been scaled in velocity space
    '''

    os.chdir(spectra_path) # navigate to the spectra path
    spec = pd.read_table(filename, sep=r"\s+", header=None, names = ['wavelength', 'flux'])
    os.chdir(ew_path) # go back to the ew directory

    # correct for radial velocity 
    spec.wavelength = velocity_correction(spec.wavelength, rv_correction)
    
    return spec

def fit_triple_gaussian(fit_spectra, centre, use_d):
    ''' 
    Fits the triple gaussian to the spectra

    Parameters
    ----------
    fit_spectra : pandas dataframe
        the dataframe of only the points you want to fit to
    use_d : bool
        Whether you want to use a fixed or unfixed continuum


    Returns
    -------
    fit : array
        the array of the gaussian fit
    residual : array
        The difference between the spectra and the fit
    rms : float
        The RMS noise of the spectrum
    chi_squared_fit : float
        The output of the stats chisquare function
    '''
    
    if use_d:
        coeff = fit_gauss3(fit_spectra, centre, True)
        fit = gauss3_plus_d(fit_spectra.wavelength, *coeff)
        
    else:
        coeff = fit_gauss3(fit_spectra, centre, False)
        fit = gauss3(fit_spectra.wavelength, *coeff)
    
    # fit parameters
    residual = fit - fit_spectra.flux
    rms = np.sqrt(np.sum(residual**2) / len(residual))
    chi_squared_fit = chisquare(fit, f_exp=fit_spectra.flux, ddof = len(coeff))

    return fit, residual, coeff, rms, chi_squared_fit

def gauss3_plus_d(x, a1, b1, c1, a2, b2, c2, a3, b3, c3, d):
    ''' A triple gaussian function with a variable continuum

    Used to fit an absorbtion feature of a spectrum. It takes the aruments of three gaussians and returns the equivelent function. The +d argument at the end allowes for the continuum to be fitted to rather than having it fixed at 1. This assumes that the normalisation process wasn't perfect.

    Parameters
    ----------
    x : int, array
            The wavelengths for the gaussian fit
    a1 : int
            The amplitude of the gaussian around the line centre
    b1 : int
            The centre wavelength of the gaussian peak
    c1 : int
            The standard deviation of the function (sigma)
    a2 : int
            The amplitude of the BLUE gaussian around the line centre
    b2 : int
            The centre BLUE wavelength of the gaussian peak
    c2 : int
            The standard deviation of the BLUE gaussian (sigma)
    a3 : int
            The amplitude of the RED gaussian around the line centre
    b3 : int
            The centre RED wavelength of the gaussian peak
    c3 : int
            The standard deviation of the RED gaussian (sigma)
    d : int
            The value of the continuum. For normalised spectra, this should be around 1

    Returns
    -------
    int, array
            (depending on what you imput as the x argument) and array of the triple gaussian curve
    '''

    return a1 * np.exp(-(x - b1)**2 / (2*c1**2)) + a2 * np.exp(-(x - b2)**2 / (2*c2**2)) + a3 * np.exp(-(x - b3)**2 / (2*c3**2)) + d

def gauss1_plus_d(x, a, b, c, d):
    ''' A single gaussian function with a variable continuum

    Used to verify the fit of a spectrum

    Parameters
    ----------
    x : int, array
            The wavelengths for the gaussian fit
    a : int
            The amplitude of the gaussian around the line centre
    b : int
            The centre wavelength of the gaussian peak
    c : int
            The standard deviation of the function (sigma)
    d : int
            The value of the continuum. For normalised spectra, this should be around 1

    Returns
    -------
    int, array
            (depending on what you imput as the x argument) and array of the single gaussian curve
    '''

    return a * np.exp(-(x - b)**2 / (2*c**2)) + d

def gauss3(x, a1, b1, c1, a2, b2, c2, a3, b3, c3):
    ''' A triple gaussian function with a fixed continuum

    Used to fit an absorbtion feature of a spectrum. It takes the aruments of three gaussians and returns the equivelent function. The continuum is fixed at 1. 

    Parameters
    ----------
    x : int, array
            The wavelengths for the gaussian fit
    a1 : int
            The amplitude of the gaussian around the line centre
    b1 : int
            The centre wavelength of the gaussian peak
    c1 : int
            The standard deviation of the function (sigma)
    a2 : int
            The amplitude of the BLUE gaussian around the line centre
    b2 : int
            The centre BLUE wavelength of the gaussian peak
    c2 : int
            The standard deviation of the BLUE gaussian (sigma)
    a3 : int
            The amplitude of the RED gaussian around the line centre
    b3 : int
            The centre RED wavelength of the gaussian peak
    c3 : int
            The standard deviation of the RED gaussian (sigma)

    Returns
    -------
    int, array
            (depending on what you imput as the x argument) and array of the triple gaussian curve
    '''

    return a1 * np.exp(-(x - b1)**2 / (2*c1**2)) + a2 * np.exp(-(x - b2)**2 / (2*c2**2)) + a3 * np.exp(-(x - b3)**2 / (2*c3**2)) + 1

def fit_gauss3(spec, centre, use_d = True):
    ''' Function which handles the fitting of the spectra

    Parameters
    ----------
    spec : pandas dataframe
            The spectra with wavelength and flux columns
    centre : int
            The line centre
    use_d : bool, optional
            Use the gaussian function with variable continuum (rather than it being fixed at 1)

    Returns
    -------
    array
            the coefficients array of the fit to the spectra
    '''
    
    # used for the bounds. How far aray from the central line can the red or blue
    # gaussians be. Won't be good for blended lines.
    pm_c = 0.3 # plus or minus centre

    if use_d:
            # guess array
            p0_array_d = [ -0.5, centre, 0.1, -0.5, centre + pm_c, 0.1, -0.5, centre - pm_c, 0.1, 1]

            #                   a1      b1        c1 a2          b2     c2  a3       b3        c3  d
            bounds_array_d = ((-1, centre - pm_c, 0, -1, centre + pm_c, 0, -1, centre - 0.9,  0, 0.95), 
                               (0, centre + pm_c, 2,  0, centre + 0.9 , 1,  0, centre - pm_c, 1, 1.05))
            # fit to the data
            coeff, var_matrix = curve_fit(gauss3_plus_d, spec.wavelength, spec.flux, 
                                            p0 = p0_array_d, bounds= bounds_array_d)
            #print(coeff)
    else:
            # guess array
            p0_array = [ -0.5, centre, 0.1, -0.5, centre + pm_c, 0.1, -0.5, centre - pm_c, 0.1]
            bounds_array = ((-1, centre - pm_c, 0, -1, centre, 0, -1, -np.inf, 0), 
                            (0, centre + pm_c, 2, 0, np.inf, 2, 0, centre, 2))

            # fit to the data
            coeff, var_matrix = curve_fit(gauss3, spec.wavelength, spec.flux, p0 = p0_array, bounds= bounds_array)

    # Return the coefficients
    return coeff

### New review functionns

def gauss_1_model(x, a, b, c, d):
    ''' A single gaussian function with a variable continuum

    Used to verify the fit of a spectrum

    Parameters
    ----------
    x : int, array
            The wavelengths for the gaussian fit
    a : int
            The amplitude of the gaussian around the line centre
    b : int
            The centre wavelength of the gaussian peak
    c : int
            The standard deviation of the function (sigma)
    d : int
            The value of the continuum. For normalised spectra, this should be around 1

    Returns
    -------
    int, array
            (depending on what you imput as the x argument) and array of the single gaussian curve
    '''

    return a * np.exp(-(x - b)**2 / (2*c**2)) + d

def gauss_2_model(x, a1, b1, c1, a2, b2, c2, d):
    ''' A double gaussian function with a variable continuum

    Used to fit an absorbtion feature of a spectrum. It takes the aruments of three gaussians and returns the equivelent function. The +d argument at the end allowes for the continuum to be fitted to rather than having it fixed at 1. This assumes that the normalisation process wasn't perfect.

    Parameters
    ----------
    x : int, array
            The wavelengths for the gaussian fit
    a1 : int
            The amplitude of the gaussian around the line centre
    b1 : int
            The centre wavelength of the gaussian peak
    c1 : int
            The standard deviation of the function (sigma)
    a2 : int
            The amplitude of the second gaussian around the line centre
    b2 : int
            The centre second wavelength of the gaussian peak
    c2 : int
            The standard deviation of the second gaussian (sigma)
    d : int
            The value of the continuum. For normalised spectra, this should be around 1

    Returns
    -------
    int, array
            (depending on what you imput as the x argument) and array of the triple gaussian curve
    '''

    return gauss_1_model(x, a1, b1, c1, 0) + gauss_1_model(x, a2, b2, c2, 0) + d

def gauss_3_model(x, a1, b1, c1, a2, b2, c2, a3, b3, c3, d):
    ''' A triple gaussian function with a variable continuum

    Used to fit an absorbtion feature of a spectrum. It takes the aruments of three gaussians and returns the equivelent function. The +d argument at the end allowes for the continuum to be fitted to rather than having it fixed at 1. This assumes that the normalisation process wasn't perfect.

    Parameters
    ----------
    x : int, array
            The wavelengths for the gaussian fit
    a1 : int
            The amplitude of the gaussian around the line centre
    b1 : int
            The centre wavelength of the gaussian peak
    c1 : int
            The standard deviation of the function (sigma)
    a2 : int
            The amplitude of the BLUE gaussian around the line centre
    b2 : int
            The centre BLUE wavelength of the gaussian peak
    c2 : int
            The standard deviation of the BLUE gaussian (sigma)
    a3 : int
            The amplitude of the RED gaussian around the line centre
    b3 : int
            The centre RED wavelength of the gaussian peak
    c3 : int
            The standard deviation of the RED gaussian (sigma)
    d : int
            The value of the continuum. For normalised spectra, this should be around 1

    Returns
    -------
    int, array
            (depending on what you imput as the x argument) and array of the triple gaussian curve
    '''

    return gauss_1_model(x, a1, b1, c1, 0) + gauss_1_model(x, a2, b2, c2, 0) + gauss_1_model(x, a3, b3, c3, 0) + d

def gauss_4_model(x, a1, b1, c1, a2, b2, c2, a3, b3, c3, a4, b4, c4, d):
    ''' A quadrupal gaussian function with a variable continuum

    Used to fit an absorbtion feature of a spectrum. It takes the aruments of three gaussians and returns the equivelent function. The +d argument at the end allowes for the continuum to be fitted to rather than having it fixed at 1. This assumes that the normalisation process wasn't perfect.

    Parameters
    ----------
    x : int, array
            The wavelengths for the gaussian fit
    a1 : int
            The amplitude of the gaussian around the line centre
    b1 : int
            The centre wavelength of the gaussian peak
    c1 : int
            The standard deviation of the function (sigma)
    a2 : int
            The amplitude of the BLUE gaussian around the line centre
    b2 : int
            The centre BLUE wavelength of the gaussian peak
    c2 : int
            The standard deviation of the BLUE gaussian (sigma)
    a3 : int
            The amplitude of the RED gaussian around the line centre
    b3 : int
            The centre RED wavelength of the gaussian peak
    c3 : int
            The standard deviation of the RED gaussian (sigma)
    d : int
            The value of the continuum. For normalised spectra, this should be around 1

    Returns
    -------
    int, array
            (depending on what you imput as the x argument) and array of the triple gaussian curve
    '''

    return gauss_1_model(x, a1, b1, c1, 0) + gauss_1_model(x, a2, b2, c2, 0) + gauss_1_model(x, a3, b3, c3, 0) + gauss_1_model(x, a4, b4, c4, 0) + d

def gauss_5_model(x, a1, b1, c1, a2, b2, c2, a3, b3, c3, a4, b4, c4, a5, b5, c5, d):
    ''' A quadrupal gaussian function with a variable continuum

    Used to fit an absorbtion feature of a spectrum. It takes the aruments of three gaussians and returns the equivelent function. The +d argument at the end allowes for the continuum to be fitted to rather than having it fixed at 1. This assumes that the normalisation process wasn't perfect.

    Parameters
    ----------
    x : int, array
            The wavelengths for the gaussian fit
    a1 : int
            The amplitude of the gaussian around the line centre
    b1 : int
            The centre wavelength of the gaussian peak
    c1 : int
            The standard deviation of the function (sigma)
    a2 : int
            The amplitude of the BLUE gaussian around the line centre
    b2 : int
            The centre BLUE wavelength of the gaussian peak
    c2 : int
            The standard deviation of the BLUE gaussian (sigma)
    a3 : int
            The amplitude of the RED gaussian around the line centre
    b3 : int
            The centre RED wavelength of the gaussian peak
    c3 : int
            The standard deviation of the RED gaussian (sigma)
    d : int
            The value of the continuum. For normalised spectra, this should be around 1

    Returns
    -------
    int, array
            (depending on what you imput as the x argument) and array of the triple gaussian curve
    '''

    return gauss_1_model(x, a1, b1, c1, 0) + gauss_1_model(x, a2, b2, c2, 0) + gauss_1_model(x, a3, b3, c3, 0) + gauss_1_model(x, a4, b4, c4, 0) + gauss_1_model(x, a5, b5, c5, 0) + d

def gauss_1_fit2(wl, n_flux, a = -0.5, b = 0.6, c = 0.1):
    '''
    fitting function for a single gaussian

    Parameters
    ----------
    wl : array
        The wavelength array
    n_flux : array
        The spectra flux array
    a : float
        Amplitude
    b : float
        Position
    c : float
        standard deviation
    
    Returns
    -------
    coeff : arr
        The coefficients of best fit
    var_matrix : arr
        Covarience matrix of the best fit
    '''

    pm_c = 0.2 # plus or minus centre
    centre = 0.6

    # guess array
    #            a  b  c, d
    p0_array = [ a, b, c, 1]
    

    #                   a1      b1      c1   d
    bounds_array = ((-1, centre - pm_c, 0, 0.95),
                    ( 0, centre + pm_c, 2, 1.05))

    p0_array = check_bounds(p0_array, bounds_array)
                    
    # fit to the data
    coeff, var_matrix = curve_fit(gauss_1_model, wl, n_flux, 
                                    p0 = p0_array, bounds= bounds_array)
    return coeff, var_matrix

def gauss_2R_fit2(wl, n_flux, a = -0.5, b = 0.6, c = 0.1, aR = -0.5, bR = 0.8, cR = 0.1):
    '''
    Fitting function for a double gaussian.
    Centre one and one to the right of it.

    Parameters
    ----------
    wl : array
        The wavelength array
    n_flux : array
        The spectra flux array
    a : float
        Amplitude
    b : float
        Position
    c : float
        standard deviation
    aR : float
        Amplitude of right gaussian
    bR : float
        Position of right gaussian
    cR : float
        standard deviation of right gaussian
    
    Returns
    -------
    coeff : arr
        The coefficients of best fit
    var_matrix : arr
        Covarience matrix of the best fit
    '''

    pm_c = 0.18 # plus or minus centre
    max_peak = 1
    centre = 0.6
    max_c = 0.2

    # guess array
    p0_array = [ a, b, c, aR, bR, cR, 1]

    #                 a      b                c   aR       bR            cR      d
    bounds_array = ((-1, centre - pm_c - 0.4, 0, -1, centre + pm_c,      0    , 0.95), 
                    ( 0, centre + pm_c + 0.4, 2,  0, centre + max_peak , max_c, 1.05))
    
    p0_array = check_bounds(p0_array, bounds_array)

    # fit to the data
    coeff, var_matrix = curve_fit(gauss_2_model, wl, n_flux, 
                                    p0 = p0_array, bounds= bounds_array)
    return coeff, var_matrix

def gauss_2L_fit2(wl, n_flux, a = -0.5, b = 0.6, c = 0.1, aL = -0.5, bL = 0.4, cL = 0.1):
    '''
    Fitting function for a double gaussian.
    Centre one and one to the left of it.

    Parameters
    ----------
    wl : array
        The wavelength array
    n_flux : array
        The spectra flux array
    a : float
        Amplitude
    b : float
        Position
    c : float
        standard deviation
    aL : float
        Amplitude of left gaussian
    bL : float
        Position of left gaussian
    cL : float
        standard deviation of left gaussian
    
    Returns
    -------
    coeff : arr
        The coefficients of best fit
    var_matrix : arr
        Covarience matrix of the best fit
    '''

    pm_c = 0.18 # plus or minus centre
    max_peak = 1
    centre = 0.6

    # guess array
    p0_array = [ a, b, c, aL, bL, cL, 1]
    

    #                 a      b                c   aL       bL       cL    d
    bounds_array = ((-1, centre - pm_c - 0.4, 0, -1, centre - max_peak,  0, 0.95), 
                    ( 0, centre + pm_c + 0.4, 2,  0, centre - pm_c, 1, 1.05))

    p0_array = check_bounds(p0_array, bounds_array)
                    
    # fit to the data
    coeff, var_matrix = curve_fit(gauss_2_model, wl, n_flux, 
                                    p0 = p0_array, bounds= bounds_array)
    return coeff, var_matrix

def gauss_3_NN(wl, n_flux, a1 = -0.5, b1 = 0.6, c1 = 0.1, a2 = -0.5, b2 = 1.1, c2 = 0.1, a3 = -0.5, b3 = 0.09, c3 = 0.1):
    '''
    Triple fit with the neural network for the central gaussian

    Parameters
    ----------
    wl : array
        The wavelength array
    n_flux : array
        The spectra flux array
    a1 : float
        Amplitude
    b1 : float
        Position
    c1 : float
        standard deviation
    a2 : float
        Amplitude of right gaussian
    b2 : float
        Position of right gaussian
    c2 : float
        standard deviation of right gaussian
    a3 : float
        Amplitude of left gaussian
    b3 : float
        Position of left gaussian
    c3 : float
        standard deviation of left gaussian
    
    Returns
    -------
    coeff : arr
        The coefficients of best fit
    var_matrix : arr
        Covarience matrix of the best fit
    '''
    
    
    pm_c = 0.18 # plus or minus centre, usual setting - 0.18
    max_peak = 1
    centre = 0.6
    a_bounds = 0.001 # plus or minus the value its allowed to be from nn guess
    c_bounds = 0.001 # same as a

    # Need to considder the bounds of the training data
    # If the network has never seen spectra in a certain region, 
    # you cant use it to fit. Should be reflected by the bad chi^2 and
    # shouldnt be selected for the final fit.
    

    # guess array
    p0_array = [ a1, b1, c1, a2, b2, c2, a3, b3, c3, 1]

    # original bounds which are defaulted to when there are errors
    #                   a1           b1        c1        a2          b2          c2    a3       b3               c3   d
    bounds_array = ((a1 - a_bounds, 0.5, c1 - c_bounds, -1, centre + pm_c,      0.05, -0.6, centre - max_peak, 0.05, 0.98), 
                    (a1 + a_bounds, 0.7, c1 + c_bounds,  0, centre + max_peak ,  0.1,    0,     centre - pm_c,  0.1, 1.02))
    p0_array = check_bounds(p0_array, bounds_array)
    
    # fit to the data
    coeff, var_matrix = curve_fit(gauss_3_model, wl, n_flux, 
                                    p0 = p0_array, bounds= bounds_array, maxfev=5000)
    return coeff, var_matrix

def gauss_3_uninformed(wl, n_flux,  a1 = -0.5, b1 = 0.6, c1 = 0.1, a2 = -0.5, b2 = 1.1, c2 = 0.1, a3 = -0.5, b3 = 0.09, c3 = 0.1):
    '''
    Uninformed triple gaussian!!!

    Parameters
    ----------
    wl : array
        The wavelength array
    n_flux : array
        The spectra flux array
    a1 : float
        Amplitude
    b1 : float
        Position
    c1 : float
        standard deviation
    a2 : float
        Amplitude of right gaussian
    b2 : float
        Position of right gaussian
    c2 : float
        standard deviation of right gaussian
    a3 : float
        Amplitude of left gaussian
    b3 : float
        Position of left gaussian
    c3 : float
        standard deviation of left gaussian
    
    Returns
    -------
    coeff : arr
        The coefficients of best fit
    var_matrix : arr
        Covarience matrix of the best fit
    '''
    
    
    pm_c = 0.18 # plus or minus centre, usual setting - 0.18
    max_peak = 1
    centre = 0.6
    

    # guess array
    p0_array = [ a1, b1, c1, a2, b2, c2, a3, b3, c3, 1]

    # original bounds which are defaulted to when there are errors
    #                   a1      b1            c1 a2          b2     c2  a3       b3        c3  d
    bounds_array = ((-1, centre - pm_c, 0, -1, centre + pm_c, 0, -1, centre - max_peak,  0, 0.90), 
                     (0, centre + pm_c, 2,  0, centre + max_peak , 1,  0, centre - pm_c, 1, 1.05))
    
    p0_array = check_bounds(p0_array, bounds_array)
    
    # fit to the data
    coeff, var_matrix = curve_fit(gauss_3_model, wl, n_flux, 
                                    p0 = p0_array, bounds= bounds_array, maxfev=5000)
    return coeff, var_matrix


def gauss_3LCR_fit2(wl, n_flux, a1 = -0.5, b1 = 0.6, c1 = 0.1, a2 = -0.5, b2 = 1.1, c2 = 0.1, a3 = -0.5, b3 = 0.09, c3 = 0.1):
    '''
    Triple gaussian with a left and right gaussian around the centre.
    Using strict bounds and informed guesses from the gradient algorithm

    Parameters
    ----------
    wl : array
        The wavelength array
    n_flux : array
        The spectra flux array
    a1 : float
        Amplitude
    b1 : float
        Position
    c1 : float
        standard deviation
    a2 : float
        Amplitude of right gaussian
    b2 : float
        Position of right gaussian
    c2 : float
        standard deviation of right gaussian
    a3 : float
        Amplitude of left gaussian
    b3 : float
        Position of left gaussian
    c3 : float
        standard deviation of left gaussian
    
    Returns
    -------
    coeff : arr
        The coefficients of best fit
    var_matrix : arr
        Covarience matrix of the best fit
    '''
    
    pm_c = 0.18 # plus or minus centre
    max_peak = 1
    centre = 0.6
    centre_upper_bound = b1 + 0.1
    centre_lower_bound = b1 - 0.1
    
    # Much tighter bounds because we should have informed guesses

    # guess array
    p0_array = [ a1, b1, c1, a2, b2, c2, a3, b3, c3, 1]

    #                   a1       b1            c1 a2          b2          c2  a3       b3            c3  d
    bounds_array = ((-1, centre_lower_bound, 0, -1, centre_upper_bound, 0, -1, centre - max_peak,  0, 0.90), 
                       (0, centre_upper_bound, 2,  0, centre + max_peak , 1,  0, centre_lower_bound, 1, 1.05))
    
    p0_array = check_bounds(p0_array, bounds_array)

    # fit to the data
    coeff, var_matrix = curve_fit(gauss_3_model, wl, n_flux, 
                                    p0 = p0_array, bounds= bounds_array, method = 'trf')
    return coeff, var_matrix

def gauss_3LLC_fit2(wl, n_flux, a1 = -0.5, b1 = 0.6, c1 = 0.1, a2 = -0.5, b2 = 0.05, c2 = 0.1, a3 = -0.5, b3 = 0.09, c3 = 0.1):
    '''
    Triple gaussian with two left gaussian and the centre.
    Using strict bounds and informed guesses from the gradient algorithm

    Parameters
    ----------
    wl : array
        The wavelength array
    n_flux : array
        The spectra flux array
    a1 : float
        Amplitude
    b1 : float
        Position
    c1 : float
        standard deviation
    a2 : float
        Amplitude of left gaussian
    b2 : float
        Position of left gaussian
    c2 : float
        standard deviation of left gaussian
    a3 : float
        Amplitude of left gaussian
    b3 : float
        Position of left gaussian
    c3 : float
        standard deviation of left gaussian
    
    Returns
    -------
    coeff : arr
        The coefficients of best fit
    var_matrix : arr
        Covarience matrix of the best fit
    '''
    
    pm_c = 0.18 # plus or minus centre
    max_peak = 1
    centre = 0.6
    centre_upper_bound = b1 + 0.1
    centre_lower_bound = b1 - 0.1
    

    # guess array
    p0_array = [ a1, b1, c1, a2, b2, c2, a3, b3, c3, 1]
    #print(p0_array_d)

    #                   a1       b1            c1 a2          b2         c2  a3           b3        c3   d
    bounds_array = ((-1, centre_lower_bound, 0, -1, centre - max_peak,  0, -1, centre - max_peak,  0, 0.90), 
                       (0, centre_upper_bound, 2,  0, centre_lower_bound, 1,  0, centre_lower_bound, 1, 1.05))

    p0_array = check_bounds(p0_array, bounds_array)

    # fit to the data
    coeff, var_matrix = curve_fit(gauss_3_model, wl, n_flux, 
                                    p0 = p0_array, bounds= bounds_array, method = 'trf')
    return coeff, var_matrix

def gauss_3CRR_fit2(wl, n_flux, a1 = -0.5, b1 = 0.6, c1 = 0.1, a2 = -0.5, b2 = 1.1, c2 = 0.1, a3 = -0.5, b3 = 1.0, c3 = 0.1):
    '''
    Triple gaussian with two right gaussian and the centre.
    Using strict bounds and informed guesses from the gradient algorithm

    Parameters
    ----------
    wl : array
        The wavelength array
    n_flux : array
        The spectra flux array
    a1 : float
        Amplitude
    b1 : float
        Position
    c1 : float
        standard deviation
    a2 : float
        Amplitude of right gaussian
    b2 : float
        Position of right gaussian
    c2 : float
        standard deviation of right gaussian
    a3 : float
        Amplitude of right gaussian
    b3 : float
        Position of right gaussian
    c3 : float
        standard deviation of right gaussian
    
    Returns
    -------
    coeff : arr
        The coefficients of best fit
    var_matrix : arr
        Covarience matrix of the best fit
    '''
    
    pm_c = 0.18 # plus or minus centre
    max_peak = 1
    centre = 0.6
    centre_upper_bound = b1 + 0.1
    centre_lower_bound = b1 - 0.1
    

    # guess array
    p0_array = [ a1, b1, c1, a2, b2, c2, a3, b3, c3, 1]

    #                   a1      b1            c1 a2          b2           c2  a3        b3            c3  d
    bounds_array = ((-1, centre_lower_bound, 0, -1, centre_upper_bound, 0, -1, centre_upper_bound,  0, 0.90), 
                       (0, centre_upper_bound, 2,  0, centre + max_peak , 1,  0, centre + max_peak,   1, 1.05))
    p0_array = check_bounds(p0_array, bounds_array)
    
    # fit to the data
    coeff, var_matrix = curve_fit(gauss_3_model, wl, n_flux, 
                                    p0 = p0_array, bounds= bounds_array, method = 'trf')
    return coeff, var_matrix

def gauss_4R_fit2(wl, n_flux, a1 = -0.5, b1 = 0.6, c1 = 0.1, a2 = -0.5, b2 = 1.1, c2 = 0.1, a3 = -0.5, b3 = 0.9, c3 = 0.1, a4 = -0.5, b4 = 0.05, c4 = 0.05):
    '''
    Quad gaussian with 1 -> central, 2 -> right, 3 -> right and 4 -> left
    Using strict bounds and informed guesses from the gradient algorithm

    Parameters
    ----------
    wl : array
        The wavelength array
    n_flux : array
        The spectra flux array
    a1 : float
        Amplitude
    b1 : float
        Position
    c1 : float
        standard deviation
    a2 : float
        Amplitude of right gaussian
    b2 : float
        Position of right gaussian
    c2 : float
        standard deviation of right gaussian
    a3 : float
        Amplitude of right gaussian
    b3 : float
        Position of right gaussian
    c3 : float
        standard deviation of right gaussian
    a4 : float
        Amplitude of left gaussian
    b4 : float
        Position of left gaussian
    c4 : float
        standard deviation of left gaussian
    
    Returns
    -------
    coeff : arr
        The coefficients of best fit
    var_matrix : arr
        Covarience matrix of the best fit
    '''

    pm_c = 0.18 # plus or minus centre
    max_peak = 1
    centre = 0.6

    # guess array
    p0_array = [ a1, b1, c1, a2, b2, c2, a3, b3, c3, a4, b4, c4, 1]

    #                   a1      b1              c1 a2          b2     c2  a3       b3      c3  a4              b4      c4   d
    bounds_array = ((-1, centre - pm_c - 0.4, 0, -1, centre + pm_c, 0, -1, centre + pm_c, 0, -1, centre - max_peak,  0, 0.90), 
                        (0, centre + pm_c + 0.4, 2,  0, centre + max_peak , 1,0, centre + max_peak , 1,  0, centre - pm_c, 1, 1.05))
    p0_array = check_bounds(p0_array, bounds_array)

    
    # fit to the data
    coeff, var_matrix = curve_fit(gauss_4_model, wl, n_flux, 
                                    p0 = p0_array, bounds= bounds_array, method = 'trf')
    return coeff, var_matrix

def gauss_4L_fit2(wl, n_flux, a1 = -0.5, b1 = 0.6, c1 = 0.1, a2 = -0.5, b2 = 1.1, c2 = 0.1, a3 = -0.5, b3 = 0.09, c3 = 0.1, a4 = -0.5, b4 = 0.05, c4 = 0.1):
    '''
    Quad gaussian with 1 -> central, 2 -> right, 3 -> left and 4 -> left
    Using strict bounds and informed guesses from the gradient algorithm

    Parameters
    ----------
    wl : array
        The wavelength array
    n_flux : array
        The spectra flux array
    a1 : float
        Amplitude
    b1 : float
        Position
    c1 : float
        standard deviation
    a2 : float
        Amplitude of right gaussian
    b2 : float
        Position of right gaussian
    c2 : float
        standard deviation of right gaussian
    a3 : float
        Amplitude of left gaussian
    b3 : float
        Position of left gaussian
    c3 : float
        standard deviation of left gaussian
    a4 : float
        Amplitude of left gaussian
    b4 : float
        Position of left gaussian
    c4 : float
        standard deviation of left gaussian
    
    Returns
    -------
    coeff : arr
        The coefficients of best fit
    var_matrix : arr
        Covarience matrix of the best fit
    '''

    pm_c = 0.18 # plus or minus centre
    max_peak = 1
    centre = 0.6
    
    # guess array
    p0_array = [ a1, b1, c1, a2, b2, c2, a3, b3, c3, a4, b4, b4, 1]

    #                   a1      b1              c1 a2          b2     c2  a3       b3      c3  a4              b4      c4   d
    bounds_array = ((-1, centre - pm_c - 0.4, 0, -1, centre + pm_c, 0, -1, centre - max_peak, 0, -1, centre - max_peak,  0, 0.90), 
                        (0, centre + pm_c + 0.4, 2,  0, centre + max_peak , 1,0, centre - pm_c, 1,  0, centre - pm_c, 1, 1.05))

    p0_array = check_bounds(p0_array, bounds_array)

    
    # fit to the data
    coeff, var_matrix = curve_fit(gauss_4_model, wl, n_flux, 
                                    p0 = p0_array, bounds= bounds_array, method = 'trf')
    return coeff, var_matrix

def check_future_points(change_array, i):
    '''
    Check that future points in the change_array are valid.
    Used so it doesnt throw an error.

    Parameters
    ----------
    change_number : 2D array
        The array of minima
    i : int
        index we are checking
    
    Returns
    -------
    bool : True if you can access that index
    '''

    # the max index which can be accessed in an array
    max_index = len(change_array)

    #look through the future gradient changes
    for j in range(4):
        # if we can access the element
        if i+j < max_index:
            if change_array[i+j][1] >= 3 and change_array[i+j][0] == 1:
                return True
        # if we cant access it, return false
        else:
            return False

def compare_triple(other_flux, coeff, wl, flux, a = 0, c = 0):
    '''
    Run fit with the old version of REvIEW: uninformed triple.
    note: here is where you set the threshold on how easily you can swap to the 
    uninformed triple version on the code. Currently if the chi-squared fit is
    2x better with the triple it will swap to that version.

    Parameters
    ----------
    other_flux : array
        Flux array of the model with the number of minima from the pre-processing
    wl : array
        The x array. Needed for curvefit to calc the uninformed triple
    flux : array
        Flux of the actual spectra. Used for curvefit
    
    Returns
    -------
    array: coeffient array from the triple fit. None if triple is not as good as the current fit
    
    '''
    if a == 0 and c == 0:
        # run fit with the conventional triple
        # do a chiquared test to see which is a better fit to the data
        coeff3, covar3 = gauss_3_uninformed(wl, flux)
    else:
        coeff3, covar3 = gauss_3_NN(wl, flux, a1 = a, c1 = c)
    
    # If the gradient method hits the bounds, swap to NN, I dont care
    if coeff[-1] > 1.028 or coeff[-1] < 0.972:
        
        return coeff3
    # look at the c parameter of the fit, if its larger than 0.2
    elif coeff[2] > 0.2:
        return coeff3

    # Looking at the +d parameter, if it does, throw it out (0.90, 1.05 bounds)
    # NN condition - has smaller bounds because of the training spectra
    #if coeff3[-1] > 1.02 or coeff3[-1] < 0.98:
    #    return None

    # look at the +d parameter if the fit, throw it out otherwise

    gauss_3_flux  = gauss_3_model(wl, *coeff3)

    # apply a penalty to the triple gaussian
    # tuned so single gaussians are always fit with a single
    # Set the threshold at which we swap to the triple
    fit_threshold = 1
    if (fit_threshold * chisquare(gauss_3_flux, flux)[0]) < chisquare(other_flux, flux)[0]:
        #print('using triple instead')
        return coeff3
    else:
        return None 

def sigma_check(change_number, minima):
    '''
    Check that the estimated sigma isnt unphysical

    Parameters
    ----------
    change_number : 2D array
        The array of minima
    minima : int
        The number of minima identified
    
    Returns
    -------
    2D array: change_number array with updated sigma
    '''

    for i in range(minima):
        if i % 3 == 2:
            if change_number[i, 1] > 24:
                change_number[i, 1] == 24
    return change_number

def check_bounds(p0, bounds):
    '''
    Makes sure that the bounds within the fitting function are all valid
    and wont throw errors.

    Parameters
    ----------
    p0 : array
        The guess array
    bounds : 2D array
        The upper and lower bounds fed into curvefit
    
    Returns
    -------
    array: p0 array returned. Will be the same as p0 passed if bounds are valid
    '''

    # bounds 0 is lower bound, bounds 1 is upper bound
    # dont need to look through the +d bound - fixed at 1
    for i in range(len(p0) -1):
        # if the guesses are outside the bounds
        if p0[i] <= bounds[0][i] or p0[i] >= bounds[1][i]:
            type = i%3
            if type == 0: # c
                # sigma doesnt impact other curves so change guess
                if p0[i] < bounds[0][i]: #really small
                    p0[i] = bounds[0][i] + 0.01 # put it inside the bounds
                else:
                    p0[i] = bounds[1][i] - 0.01
            elif type == 1: # a (currently just doing the same as c)
                if p0[i] < bounds[0][i]: #really small
                    p0[i] = bounds[0][i] + 0.01 # put it inside the bounds
                else:
                    p0[i] = bounds[1][i] - 0.01
            elif type == 2: # b, the hardest one (and most likely to mess up):
                # currently the same as all the others but might need to 
                # play around with this
                if p0[i] < bounds[0][i]: #really small
                    p0[i] = bounds[0][i] + 0.01 # put it inside the bounds
                else:
                    p0[i] = bounds[1][i] - 0.01
            else:
                print('modulo error, something out of bounds')
    return p0

def reduce_minima(change_number, minima):
    '''
    Takes the number of minima identified by the algoritm and retrieves the 4 
    (or 3 depending on if its odd or even). Maybe later it should always go to 4
    peaks regardless of whether its odd or even.

    Parameters
    ----------
    change_number : 2D array
        The minima identified by the pre-processing algorithm 
    minima : int
        The number of minima (essentially the length of change_number)
    
    Returns
    -------
    2D Array: the reduced number of minima (max of 3 or 4)
    '''

    # harsh cut for centre (seeing that there's lots of minima it should be ok)
    left_gaussian   = change_number[change_number[:,2] < 0.52]
    centre_gaussian = change_number[(change_number[:,2] > 0.52) & (change_number[:,2] < 0.68)]
    right_gaussian  = change_number[change_number[:,2] > 0.68]

    # Use a more forgiving cut if there was no centre points
    if centre_gaussian.shape[0] == 0:
        left_gaussian   = change_number[change_number[:,2] < 0.42]
        centre_gaussian = change_number[(change_number[:,2] > 0.4) & (change_number[:,2] < 0.8)]
        right_gaussian  = change_number[change_number[:,2] > 0.78]

    # remove the central gaussian
    for i in range(minima):
        # 4 was the index in the array - guarenteed to be unique
        if change_number[i, 4] == centre_gaussian[0, 4]:
            # remove the minima
            change_number = np.delete(change_number, i, 0)
            break

    # sort based on flux
    sorted_changes = change_number[change_number[:, 3].argsort()]

    # if theres an even number of points, drop to 4 gaussians
    if  minima % 2 == 0:
        change_number = sorted_changes[:3, :]
        # put the centre one back in
        change_number = np.append(change_number, centre_gaussian, axis=0)

    # theres an odd number of points, drop to 3 gaussians
    else:
        change_number = sorted_changes[:2, :]
        # put the centre one back in
        change_number = np.append(change_number, centre_gaussian, axis=0)
    
    # the rest of the alg can figure it out :) 
    return change_number

def grad_change(wl, flux):
    '''
    Pre-processing algorithm to get an estimate of the number of minima in the spectra.

    Parameters
    ----------
    wl : array
        The wavelength array. Should go from 0 to 1.2
    flux : array
        The flux values around the line we are fitting to
    
    Returns
    -------
    array : the the number and position of the minima in the spectra
    '''

    count_negative = 0
    count_positive = 0

    change_number = []

    # Smooth out the data
    #flux =  savgol_filter(rawFlux, 15, 2)

    # loop to determin the gradients
    # j -> True: negative grad
    # j -> False: positive grad

    for i in range(len(flux)-1):
        j = True if (flux[i+1] - flux[i]) < 0 else False
        if j is True and count_positive == 0:
            count_negative+= 1
            count_positive = 0

        elif j is False and count_negative == 0:
            count_positive+= 1
            count_negative = 0
        
        elif j is True and count_positive != 0: # just flipped
            change_number.append([1,count_positive, wl[i], flux[i], i])
            count_negative +=1
            count_positive = 0
        
        elif j is False and count_negative != 0: # just flipped
            change_number.append([0,count_negative, wl[i], flux[i], i])
            count_positive+=1
            count_negative =0

    # add in the last point:
    if count_negative != 0:
        # whatever j was
        change_number.append([0,count_negative, wl[i+1], flux[i+1], i+1])
    else:
        change_number.append([1,count_positive, wl[i+1], flux[i+1], i+1])

    # Going through and extracting only the suitable minima
    no_of_peaks = []
    for i in range(len(change_number)):

        # rules to identify peaks:
        # 1. change_number[i][0] == 0   means we are looking at a minima point (-ive to +ive grad)
        # 2. change_number[i][1] >3     means there had to be at least 3 points before it to classify as a peak
        #    change_number[i+1][1] >3   means there had to be at least 3 points after it to classify as a peak (noise @ bottom of the peak??)
        # 3. change_number[i][3] < 0.96 means the peak must be greater than 0.96. This doesnt work so well for poorly normalised
        #                               continuums.
        if change_number[i][0] == 0 and change_number[i][1] >= 3 and change_number[i][3] < 0.96 and check_future_points(change_number, i):
            no_of_peaks.append(change_number[i])
        
    # for the very last one (because we are iterating to len -1):
    if change_number[-1][0] == 0 and change_number[-1][1] >= 3 and change_number[-1][3] < 0.96:
            no_of_peaks.append(change_number[-1])

    # add in the last one if it isnt on the continuum
    # appending to the end so have to do the end one first
    if flux[-1] < 0.95 and no_of_peaks[-1][2] < 1:
        # adding a STD for the guess to be ~ 0.06 otherwise the guess breaks the fitting code
        no_of_peaks.append([-1, 14, wl[-1], flux[-1], -1])

    # add in the first one if it isnt on the continuum
    if flux[0] < 0.95 and no_of_peaks[0][2] > 0.2:
        no_of_peaks.append([-1, 14, 0, flux[0], 0])

    no_of_peaks = np.asarray(no_of_peaks) # to numpy
    change_number = np.asarray(change_number)

    return no_of_peaks

def make_spline(wl, spline_wl, flux):
    # option 1
    #f = interpolate.splrep(wl, flux, s=0)
    #interp_flux = interpolate.splev(spline_wl, f, der=0)

    # option 2
    #f = interpolate.interp1d(wl, flux)
    #interp_flux = f(spline_wl)

    # option 3
    #print(wl)
    f = interpolate.CubicSpline(wl, flux)
    interp_flux = f(spline_wl)

    # remove values > 1 and < 0
    interp_flux[interp_flux < 0] = 0
    interp_flux[interp_flux > 1] = 1

    # spline_wl should be the one with 86 points
    return interp_flux.reshape((1, 86))

def nn_estimate(spectra):
    '''
    Performs neural network estimate. Written by Ella
    each spectra MUST have 86 points, or eles the model won't work. Linearly interpolate. 
    '''
    # read in model and scalars
    model = FFNN(0,0,model='NN/model')
    X_scalar = Scalar()
    X_scalar.load('NN/model/X_scalar.npy')
    y_scalar = Scalar()
    y_scalar.load('NN/model/y_scalar.npy')

    X = X_scalar.transform(spectra) # transform the input
    y = model.predict(X) # predict the value
    y = y_scalar.untransform(y) # untransform the predicted value
    
    #        a        c
    return y[0, 0], y[0, 1]

def fit_spectra(wl, flux):
    '''
    Main fitting code. Contains a number of flags that will be set to true if fitting fails.
    If theres > 4 minima identified, reduce it down to the most prominant peaks.

    During the testing stage: or testing pass flux without the end two values. 
    Must be the same length as wavelength.

    Parameters
    ----------
    wl : array
        The wavelength array. Should go from 0 to 1.2
    flux : array
        The flux values around the line we are fitting to
    
    Returns
    -------
    coeff : the coefficients of the best fit
    tripe_improved_fit : bool
        Flag for whether just using the conventional triple gaussian actually improved the fit
    minima_error_to_triple : book
        Flag for whether there was an unidentified number of gaussians found in the code
    '''

    # error flags
    tripe_improved_fit = False
    minima_error_to_triple = False
    nn_warning = False

    # pass the interpolated spectra to the neural network
    # weird array parameters are just from how the spectra was set up
    interpolated_spectra = make_spline(wl, np.arange(0, 1.207+0.0142, 0.0142), flux)
    nn_a, nn_c = nn_estimate(interpolated_spectra)

    # preprocessing to identify the number of minima in the spectra
    change_number = grad_change(wl, flux)
    
    # Recovering the number of minima identified
    minima = change_number.shape[0]

    # If theres more than 4 minima, reduce it down to something that 
    # th algorithm can deal with
    if minima > 4:
        change_number = reduce_minima(change_number, minima)
        minima = change_number.shape[0]

    # might be helpful but this didnt fix the problem
    #change_number = sigma_check(change_number, minima)
    
    # a and c are now going to be estimated from the NN
    if minima == 1:

        coeff, covar = gauss_1_fit2(wl, flux, 
                                a = nn_a, #-(1-change_number[0, 3]), 
                                b = change_number[0, 2],
                                c = nn_c ) #change_number[0, 1] * 0.004) # estimated scaling factor
        # comparing it to a triple
        gauss_1_flux  = gauss_1_model(wl, *coeff)
        triple_coeff = compare_triple(gauss_1_flux, coeff, wl, flux, a = nn_a, c = nn_c)
        if triple_coeff is not None:
            coeff = triple_coeff
            tripe_improved_fit = True

    elif minima == 2:

        left_gaussian   = change_number[change_number[:,2] < 0.42]
        centre_gaussian = change_number[(change_number[:,2] > 0.4) & (change_number[:,2] < 0.8)]
        right_gaussian  = change_number[change_number[:,2] > 0.78]

        # do a harsher cut if theres multiple central gaussians
        if centre_gaussian.shape[0] > 1:
            left_gaussian   = change_number[change_number[:,2] < 0.52]
            centre_gaussian = change_number[(change_number[:,2] > 0.52) & (change_number[:,2] < 0.68)]
            right_gaussian  = change_number[change_number[:,2] > 0.68]
        
        # left - centre
        if left_gaussian.shape[0] != 0 and centre_gaussian.shape[0] != 0:
            centre_gaussian = centre_gaussian[0]
            left_gaussian = left_gaussian[0]
            coeff, covar = gauss_2L_fit2(wl, flux, 
                                        a  = nn_a, #-(1-centre_gaussian[3]),
                                        b  = centre_gaussian[2],
                                        c  = nn_c, #centre_gaussian[1] * 0.004,
                                        aL = -(1-left_gaussian[3]),
                                        bL = left_gaussian[2], 
                                        cL = left_gaussian[1] * 0.004)

        # right - centre
        elif right_gaussian.shape[0] != 0 and centre_gaussian.shape[0] != 0:
            centre_gaussian = centre_gaussian[0]
            right_gaussian = right_gaussian[0]
            coeff, covar = gauss_2R_fit2(wl, flux, 
                                        a  = nn_a, #-(1-centre_gaussian[3]),
                                        b  = centre_gaussian[2],
                                        c  = nn_c, #centre_gaussian[1] * 0.004,
                                        aR = -(1-right_gaussian[3]),
                                        bR = right_gaussian[2],
                                        cR = right_gaussian[1] * 0.004)
        
        else:
            coeff, covar = gauss_3_NN(wl, flux, a1 = nn_a, c1 = nn_c)
            minima_error_to_triple = True

        if minima_error_to_triple == False:
            # comparing it to a triple
            gauss_2_flux  = gauss_2_model(wl, *coeff)
            triple_coeff = compare_triple(gauss_2_flux, coeff, wl, flux, a = nn_a, c = nn_c)
            if triple_coeff is not None:
                coeff = triple_coeff
                tripe_improved_fit = True


    elif minima == 3:

        left_gaussian   = change_number[change_number[:,2] < 0.42]
        centre_gaussian = change_number[(change_number[:,2] > 0.4) & (change_number[:,2] < 0.8)]
        right_gaussian  = change_number[change_number[:,2] > 0.78]

        # do a harsher cut if theres multiple central gaussians
        if centre_gaussian.shape[0] > 1:
            left_gaussian   = change_number[change_number[:,2] < 0.52]
            centre_gaussian = change_number[(change_number[:,2] > 0.52) & (change_number[:,2] < 0.68)]
            right_gaussian  = change_number[change_number[:,2] > 0.68]

        # centre - right - right
        if right_gaussian.shape[0] > 1 and centre_gaussian.shape[0] == 1:
            centre_gaussian = centre_gaussian[0]
            right_gaussian1 = right_gaussian[0]
            right_gaussian2 = right_gaussian[1]
            
            coeff, covar = gauss_3CRR_fit2(wl, flux,
                                        a1 = nn_a, #-(1-centre_gaussian[3]),
                                        b1 = centre_gaussian[2],
                                        c1 = nn_c, #centre_gaussian[1] * 0.004,
                                        a2 = -(1-right_gaussian1[3]),
                                        b2 = right_gaussian1[2],
                                        c2 = right_gaussian1[1] * 0.004,
                                        a3 = -(1-right_gaussian2[3]),
                                        b3 = right_gaussian2[2],
                                        c3 = right_gaussian2[1] * 0.004)

        # left - left - centre
        elif left_gaussian.shape[0] > 1 and centre_gaussian.shape[0] == 1:
            centre_gaussian = centre_gaussian[0]
            left_gaussian1 = left_gaussian[0]
            left_gaussian2 = left_gaussian[1]
            
            coeff, covar = gauss_3LLC_fit2(wl, flux,
                                        a1 = nn_a, #-(1-centre_gaussian[3]),
                                        b1 = centre_gaussian[2],
                                        c1 = nn_c, #centre_gaussian[1] * 0.004,
                                        a2 = -(1-left_gaussian1[3]),
                                        b2 = left_gaussian1[2],
                                        c2 = left_gaussian1[1] * 0.004,
                                        a3 = -(1-left_gaussian2[3]),
                                        b3 = left_gaussian2[2],
                                        c3 = left_gaussian2[1] * 0.004)
        
        # left - centre - right
        elif left_gaussian.shape[0]  == 1 and right_gaussian.shape[0]  == 1 and centre_gaussian.shape[0] == 1:
            right_gaussian = right_gaussian[0]
            centre_gaussian = centre_gaussian[0]
            left_gaussian = left_gaussian[0]
            
            coeff, covar = gauss_3LCR_fit2(wl, flux,
                                        a1 = nn_a, #-(1-centre_gaussian[3]),
                                        b1 = centre_gaussian[2],
                                        c1 = nn_c, #centre_gaussian[1] * 0.004,
                                        a2 = -(1-right_gaussian[3]),
                                        b2 = right_gaussian[2],
                                        c2 = right_gaussian[1] * 0.004,
                                        a3 = -(1-left_gaussian[3]),
                                        b3 = left_gaussian[2],
                                        c3 = left_gaussian[1] * 0.004)
        else:
            coeff, covar = gauss_3_NN(wl, flux, a1 = nn_a, c1 = nn_c)
            minima_error_to_triple = True
        
        if minima_error_to_triple == False:
            # comparing it to a triple, but the NN version...?
            gauss_3_flux  = gauss_3_model(wl, *coeff)
            triple_coeff = compare_triple(gauss_3_flux, coeff, wl, flux, a = nn_a, c = nn_c)
            if triple_coeff is not None:
                coeff = triple_coeff
                tripe_improved_fit = True
            
        
    elif minima == 4:

        left_gaussian   = change_number[change_number[:,2] < 0.42]
        centre_gaussian = change_number[(change_number[:,2] > 0.4) & (change_number[:,2] < 0.8)]
        right_gaussian  = change_number[change_number[:,2] > 0.78]

        # left - left - centre - right
        if left_gaussian.shape[0] == 2 and centre_gaussian.shape[0] == 1 and right_gaussian.shape[0] == 1:
            right_gaussian = right_gaussian[0]
            centre_gaussian = centre_gaussian[0]
            left_gaussian1 = left_gaussian[0]
            left_gaussian2 = left_gaussian[1]

            coeff, covar = gauss_4L_fit2(wl, flux,
                                        a1 = nn_a, #-(1-centre_gaussian[3]),
                                        b1 = centre_gaussian[2],
                                        c1 = nn_c, #centre_gaussian[1] * 0.004,
                                        a2 = -(1-right_gaussian[3]),
                                        b2 = right_gaussian[2],
                                        c2 = right_gaussian[1] * 0.004,
                                        a3 = -(1-left_gaussian1[3]),
                                        b3 = left_gaussian1[2],
                                        c3 = left_gaussian1[1] * 0.004, 
                                        a4 = -(1-left_gaussian2[3]),
                                        b4 = left_gaussian2[2],
                                        c4 = left_gaussian2[1] * 0.004)
        # left - centre - right - right
        elif left_gaussian.shape[0] == 1 and centre_gaussian.shape[0] == 1 and right_gaussian.shape[0] == 2:
            right_gaussian1 = right_gaussian[0]
            right_gaussian2 = right_gaussian[1]
            centre_gaussian = centre_gaussian[0]
            left_gaussian = left_gaussian[0]

            coeff, covar = gauss_4R_fit2(wl, flux,
                            a1 = nn_a, #-(1-centre_gaussian[3]),
                            b1 = centre_gaussian[2],
                            c1 = nn_c, #centre_gaussian[1] * 0.004,
                            a2 = -(1-right_gaussian1[3]),
                            b2 = right_gaussian1[2],
                            c2 = right_gaussian1[1] * 0.004,
                            a3 = -(1-right_gaussian2[3]),
                            b3 = right_gaussian2[2],
                            c3 = right_gaussian2[1] * 0.004, 
                            a4 = -(1-left_gaussian[3]),
                            b4 = left_gaussian[2],
                            c4 = left_gaussian[1] * 0.004)
        else:
            coeff, covar = gauss_3_NN(wl, flux, a1 = nn_a, c1 = nn_c)
            minima_error_to_triple = True

        if minima_error_to_triple == False:
            # comparing it to a triple
            gauss_4_flux  = gauss_4_model(wl, *coeff)
            triple_coeff = compare_triple(gauss_4_flux, coeff, wl, flux, a = nn_a, c = nn_c)
            if triple_coeff is not None:
                coeff = triple_coeff
                tripe_improved_fit = True

    else:
        coeff, covar = gauss_3_NN(wl, flux, a1 = nn_a, c1 = nn_c)
        minima_error_to_triple = True

    
    #print('a = ', coeff[0], 'c = ', coeff[2])
    #print('curvefit ew:', get_gaussian_area(coeff[0], coeff[2]))


    return coeff, tripe_improved_fit, minima_error_to_triple

def fit_spectra_nn(wl, flux):
    '''
    Go straight to the NN fit
    
    Parameters
    ----------
    wl : array
        The wavelength array. Should go from 0 to 1.2
    flux : array
        The flux values around the line we are fitting to
    '''
    # interpolate over the spectra
    interpolated_spectra = make_spline(wl, np.arange(0, 1.207+0.0142, 0.0142), flux)
    # get the a and c from the spectra
    nn_a, nn_c = nn_estimate(interpolated_spectra)
    coeff, covar = gauss_3_NN(wl, flux, a1 = nn_a, c1=nn_c)
    return coeff

def fit_spectra_original(wl, flux):
    '''
    Main fitting code. Contains a number of flags that will be set to true if fitting fails.
    If theres > 4 minima identified, reduce it down to the most prominant peaks.

    During the testing stage: or testing pass flux without the end two values. 
    Must be the same length as wavelength.

    Parameters
    ----------
    wl : array
        The wavelength array. Should go from 0 to 1.2
    flux : array
        The flux values around the line we are fitting to
    
    Returns
    -------
    coeff : the coefficients of the best fit
    tripe_improved_fit : bool
        Flag for whether just using the conventional triple gaussian actually improved the fit
    minima_error_to_triple : book
        Flag for whether there was an unidentified number of gaussians found in the code
    '''

    # error flags
    tripe_improved_fit = False
    minima_error_to_triple = False
    nn_warning = False

    # pass the interpolated spectra to the neural network
    # weird array parameters are just from how the spectra was set up
    interpolated_spectra = make_spline(wl, np.arange(0, 1.207+0.0142, 0.0142), flux)
    nn_a, nn_c = nn_estimate(interpolated_spectra)

    # preprocessing to identify the number of minima in the spectra
    change_number = grad_change(wl, flux)
    
    # Recovering the number of minima identified
    minima = change_number.shape[0]

    # If theres more than 4 minima, reduce it down to something that 
    # th algorithm can deal with
    if minima > 4:
        change_number = reduce_minima(change_number, minima)
        minima = change_number.shape[0]

    # might be helpful but this didnt fix the problem
    #change_number = sigma_check(change_number, minima)
    
    # a and c are now going to be estimated from the NN
    if minima == 1:

        coeff, covar = gauss_1_fit2(wl, flux, 
                                a = -(1-change_number[0, 3]), 
                                b = change_number[0, 2],
                                c = change_number[0, 1] * 0.004) # estimated scaling factor
        # comparing it to a triple
        gauss_1_flux  = gauss_1_model(wl, *coeff)
        triple_coeff = compare_triple(gauss_1_flux, wl, flux)
        if triple_coeff is not None:
            coeff = triple_coeff
            tripe_improved_fit = True

    elif minima == 2:

        left_gaussian   = change_number[change_number[:,2] < 0.42]
        centre_gaussian = change_number[(change_number[:,2] > 0.4) & (change_number[:,2] < 0.8)]
        right_gaussian  = change_number[change_number[:,2] > 0.78]

        # do a harsher cut if theres multiple central gaussians
        if centre_gaussian.shape[0] > 1:
            left_gaussian   = change_number[change_number[:,2] < 0.52]
            centre_gaussian = change_number[(change_number[:,2] > 0.52) & (change_number[:,2] < 0.68)]
            right_gaussian  = change_number[change_number[:,2] > 0.68]
        
        # left - centre
        if left_gaussian.shape[0] != 0 and centre_gaussian.shape[0] != 0:
            centre_gaussian = centre_gaussian[0]
            left_gaussian = left_gaussian[0]
            coeff, covar = gauss_2L_fit2(wl, flux, 
                                        a  = -(1-centre_gaussian[3]),
                                        b  = centre_gaussian[2],
                                        c  = centre_gaussian[1] * 0.004,
                                        aL = -(1-left_gaussian[3]),
                                        bL = left_gaussian[2], 
                                        cL = left_gaussian[1] * 0.004)

        # right - centre
        elif right_gaussian.shape[0] != 0 and centre_gaussian.shape[0] != 0:
            centre_gaussian = centre_gaussian[0]
            right_gaussian = right_gaussian[0]
            coeff, covar = gauss_2R_fit2(wl, flux, 
                                        a  = -(1-centre_gaussian[3]),
                                        b  = centre_gaussian[2],
                                        c  = centre_gaussian[1] * 0.004,
                                        aR = -(1-right_gaussian[3]),
                                        bR = right_gaussian[2],
                                        cR = right_gaussian[1] * 0.004)
        
        else:
            coeff, covar = gauss_3_NN(wl, flux, a1 = nn_a, c1 = nn_c)
            minima_error_to_triple = True

        if minima_error_to_triple == False:
            # comparing it to a triple
            gauss_2_flux  = gauss_2_model(wl, *coeff)
            triple_coeff = compare_triple(gauss_2_flux, wl)
            if triple_coeff is not None:
                coeff = triple_coeff
                tripe_improved_fit = True


    elif minima == 3:

        left_gaussian   = change_number[change_number[:,2] < 0.42]
        centre_gaussian = change_number[(change_number[:,2] > 0.4) & (change_number[:,2] < 0.8)]
        right_gaussian  = change_number[change_number[:,2] > 0.78]

        # do a harsher cut if theres multiple central gaussians
        if centre_gaussian.shape[0] > 1:
            left_gaussian   = change_number[change_number[:,2] < 0.52]
            centre_gaussian = change_number[(change_number[:,2] > 0.52) & (change_number[:,2] < 0.68)]
            right_gaussian  = change_number[change_number[:,2] > 0.68]

        # centre - right - right
        if right_gaussian.shape[0] > 1 and centre_gaussian.shape[0] == 1:
            centre_gaussian = centre_gaussian[0]
            right_gaussian1 = right_gaussian[0]
            right_gaussian2 = right_gaussian[1]
            
            coeff, covar = gauss_3CRR_fit2(wl, flux,
                                        a1 = -(1-centre_gaussian[3]),
                                        b1 = centre_gaussian[2],
                                        c1 = centre_gaussian[1] * 0.004,
                                        a2 = -(1-right_gaussian1[3]),
                                        b2 = right_gaussian1[2],
                                        c2 = right_gaussian1[1] * 0.004,
                                        a3 = -(1-right_gaussian2[3]),
                                        b3 = right_gaussian2[2],
                                        c3 = right_gaussian2[1] * 0.004)

        # left - left - centre
        elif left_gaussian.shape[0] > 1 and centre_gaussian.shape[0] == 1:
            centre_gaussian = centre_gaussian[0]
            left_gaussian1 = left_gaussian[0]
            left_gaussian2 = left_gaussian[1]
            
            coeff, covar = gauss_3LLC_fit2(wl, flux,
                                        a1 = -(1-centre_gaussian[3]),
                                        b1 = centre_gaussian[2],
                                        c1 = centre_gaussian[1] * 0.004,
                                        a2 = -(1-left_gaussian1[3]),
                                        b2 = left_gaussian1[2],
                                        c2 = left_gaussian1[1] * 0.004,
                                        a3 = -(1-left_gaussian2[3]),
                                        b3 = left_gaussian2[2],
                                        c3 = left_gaussian2[1] * 0.004)
        
        # left - centre - right
        elif left_gaussian.shape[0]  == 1 and right_gaussian.shape[0]  == 1 and centre_gaussian.shape[0] == 1:
            right_gaussian = right_gaussian[0]
            centre_gaussian = centre_gaussian[0]
            left_gaussian = left_gaussian[0]
            
            coeff, covar = gauss_3LCR_fit2(wl, flux,
                                        a1 = -(1-centre_gaussian[3]),
                                        b1 = centre_gaussian[2],
                                        c1 = centre_gaussian[1] * 0.004,
                                        a2 = -(1-right_gaussian[3]),
                                        b2 = right_gaussian[2],
                                        c2 = right_gaussian[1] * 0.004,
                                        a3 = -(1-left_gaussian[3]),
                                        b3 = left_gaussian[2],
                                        c3 = left_gaussian[1] * 0.004)
        else:
            coeff, covar = gauss_3_NN(wl, flux, a1 = nn_a, c1 = nn_c)
            minima_error_to_triple = True
            
        
    elif minima == 4:

        left_gaussian   = change_number[change_number[:,2] < 0.42]
        centre_gaussian = change_number[(change_number[:,2] > 0.4) & (change_number[:,2] < 0.8)]
        right_gaussian  = change_number[change_number[:,2] > 0.78]

        # left - left - centre - right
        if left_gaussian.shape[0] == 2 and centre_gaussian.shape[0] == 1 and right_gaussian.shape[0] == 1:
            right_gaussian = right_gaussian[0]
            centre_gaussian = centre_gaussian[0]
            left_gaussian1 = left_gaussian[0]
            left_gaussian2 = left_gaussian[1]

            coeff, covar = gauss_4L_fit2(wl, flux,
                                        a1 = -(1-centre_gaussian[3]),
                                        b1 = centre_gaussian[2],
                                        c1 = centre_gaussian[1] * 0.004,
                                        a2 = -(1-right_gaussian[3]),
                                        b2 = right_gaussian[2],
                                        c2 = right_gaussian[1] * 0.004,
                                        a3 = -(1-left_gaussian1[3]),
                                        b3 = left_gaussian1[2],
                                        c3 = left_gaussian1[1] * 0.004, 
                                        a4 = -(1-left_gaussian2[3]),
                                        b4 = left_gaussian2[2],
                                        c4 = left_gaussian2[1] * 0.004)
        # left - centre - right - right
        elif left_gaussian.shape[0] == 1 and centre_gaussian.shape[0] == 1 and right_gaussian.shape[0] == 2:
            right_gaussian1 = right_gaussian[0]
            right_gaussian2 = right_gaussian[1]
            centre_gaussian = centre_gaussian[0]
            left_gaussian = left_gaussian[0]

            coeff, covar = gauss_4R_fit2(wl, flux,
                            a1 = -(1-centre_gaussian[3]),
                            b1 = centre_gaussian[2],
                            c1 = centre_gaussian[1] * 0.004,
                            a2 = -(1-right_gaussian1[3]),
                            b2 = right_gaussian1[2],
                            c2 = right_gaussian1[1] * 0.004,
                            a3 = -(1-right_gaussian2[3]),
                            b3 = right_gaussian2[2],
                            c3 = right_gaussian2[1] * 0.004, 
                            a4 = -(1-left_gaussian[3]),
                            b4 = left_gaussian[2],
                            c4 = left_gaussian[1] * 0.004)
        else:
            coeff, covar = gauss_3_NN(wl, flux, a1 = nn_a, c1 = nn_c)
            minima_error_to_triple = True

        if minima_error_to_triple == False:
            # comparing it to a triple
            gauss_4_flux  = gauss_4_model(wl, *coeff)
            triple_coeff = compare_triple(gauss_4_flux, wl, flux)
            if triple_coeff is not None:
                coeff = triple_coeff
                tripe_improved_fit = True

    else:
        coeff, covar = gauss_3_NN(wl, flux, a1 = nn_a, c1 = nn_c)
        minima_error_to_triple = True

    
    #print('a = ', coeff[0], 'c = ', coeff[2])
    #print('curvefit ew:', get_gaussian_area(coeff[0], coeff[2]))


    return coeff, tripe_improved_fit, minima_error_to_triple
