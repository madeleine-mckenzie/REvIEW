# to stop my IDE from complaining
from my_imports import *

#############################################
#                                           #
#  General helper functions                 #
#                                           #
#############################################

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


#############################################
#                                           #
#  Functions for iterating over the lines   #
#                                           #
#############################################

def line_bounds(line, bounds = 2):
    ''' Function for iterating over lines in a line list

    Parameters
    ----------
    line : int
        The line to be fitted
    bounds : int, optional
        The size of the region to be fitted. Will range from +- bounds centred on the line.
        Default range is 2.

    Returns
    -------
    list
        a list containing the line centre, the lower bound and the upper bound
    '''

    # line, lower bound, upper bound
    return [line, line - bounds/2, line + bounds/2]

def line_incrementer(linelist, lower_bound, upper_bound):
    ''' Function for incrementing over a range of lines. 

    Passing a line list and a wave length range you would like to search through, it returns an array which you can iterate through to find the lines 

    Parameters
    ----------
    linelist : array
        An array containing the wavelengths of a line list
    lower_bound : int
        The lower wavelength that you want to output equivalent widths for
    upper_bound : int
        The upper wavelength that you want to output equivalent widths for


    Returns
    -------
    2D list
        a list with rows containing the line centre and lower and upper bounds of the line.
    '''

    # lines within the given range
    line_in_bounds = linelist[(linelist > lower_bound) & (linelist < upper_bound)]
    bounds_array = []

    # iterate through all the lines within the bounds and append them to the array
    for lines in line_in_bounds:
        bounds_array.append(line_bounds(lines))
    
    return bounds_array


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
            bounds_array_d = ((-1, centre - pm_c, 0, -1, centre + pm_c, 0, -1, -np.inf, 0, 0.97), 
                            (0, centre + pm_c, 2, 0, np.inf, 2, 0, centre - pm_c, 2, 1.03))
            # fit to the data
            coeff, var_matrix = curve_fit(gauss3_plus_d, spec.wavelength, spec.flux, 
                                            p0 = p0_array_d, bounds= bounds_array_d)
    else:
            # guess array
            p0_array = [ -0.5, centre, 0.1, -0.5, centre + pm_c, 0.1, -0.5, centre - pm_c, 0.1]
            bounds_array = ((-1, centre - pm_c, 0, -1, centre, 0, -1, -np.inf, 0), 
                            (0, centre + pm_c, 2, 0, np.inf, 2, 0, centre, 2))

            # fit to the data
            coeff, var_matrix = curve_fit(gauss3, spec.wavelength, spec.flux, p0 = p0_array, bounds= bounds_array)

    # Return the coefficients
    return coeff


#############################################
#                                           #
#  Functions for plotting the spectra       #
#                                           #
#############################################

def plot_spectra_lines(pd_tab, low, upp):
    '''
    Helper function for quickly plotting lines within a given range

    Parameters
    ----------
    pd_tab : pandas dataframe
        The spectra with wavelength and flux columns
    low : int
        The lower wavelength limit
    upp : int
        The upper wavelength limit

    Returns
    -------
    plot
        a plot of the spectra within that wavelength
    '''
    plt.plot(pd_tab.wavelength, pd_tab.flux, linewidth = 3, 
    color ='teal')
    # uncomment and pass a line list
    #plt.vlines(linelist2.wavelength, 0, 1.3, alpha = 0.5, color = 'red')
    plt.ylim(0,1.2)
    plt.xlim(low, upp)
    plt.show()
    #plt.close() depending on how many you want to plot

def hight_ratio(n_spectra):
    '''
    Sets up the height ratio array for gridspec

    Each plot has the main plot which has a ratio of 1, a plot for the residuals and then a blank plot to separate all the plots
    
    Parameters
    ----------
    n_spectra : int
        The number of spectra you are plotting

    Returns
    -------
    arr
        an array of the ratio of plot hights
    '''

    arr = []
    for i in range(n_spectra):
        arr.append(1), arr.append(0.3), arr.append(0.1)
    return arr


#############################################
#                                           #
#  Main plotting functions                  #
#                                           #
#############################################

def spectra_plot(ax, filename, rv_correction, spectra_path, ew_path, centre, x_lower, x_upper, make_fit = True, use_d = True):
    ''' 
    Makes the plot of the spectra and the fits.

    Calls the read spectra function and fitting function. Then collates the information and plots it.

    Parameters
    ----------
    ax : matplotlib axis
        The axis you are plotting to
    filename : string
        The name of the spectra to analyse
    rv_correction : float
        The radial velocity correction
    centre : float
        The centre of the line to fit
    x_lower : int
        The lower bound of the wavelength
    x_upper : int
        The lower bound of the wavelength
    fit : bool, optional
        When true, fit a tripple gaussian to the spectra
    use_d : bool, optional
        When true, fit to the continuum rather than having it fixed at 1
    
    Returns
    -------
    fit_spectra.wavelength : array
        Returning it to make sure the residual array is using exactly the same wavelengths as this plot
    residual : array
        The difference between the spectra and the fit
    coeff : array
        The best fit parameters to the gaussian
    chi_squared_fit : float
        The output of the stats chisquare function
    rms : float
        The rms of the residual
    '''

    # read in the spectra
    spec = read_spectra(filename, rv_correction, spectra_path, ew_path)
    

    if make_fit:
        fit_region = 0.6 # wavelengths around the line we are fitting to

        # Assuming that the minimum value is going to be the line
        fit_spectra = spec[(spec.wavelength > centre - fit_region) & (spec.wavelength < centre + fit_region)]
        
        try:
            # move this outside of the try for testing
            fit, residual, coeff, rms, chi_squared_fit = fit_triple_gaussian(fit_spectra, centre, use_d)
            if use_d:
                fit_1_gaussian = gauss1_plus_d(fit_spectra, coeff[0], coeff[1], coeff[2], coeff[-1])
            else:
                fit_1_gaussian = gauss1_plus_d(fit_spectra, coeff[0], coeff[1], coeff[2], 1)
            # plot the output from curvefit 
            ax.plot(fit_spectra.wavelength, fit, color ='#D1495B', zorder = 3, linewidth = 3)
            # See what the gaussian actually looks like
            ax.plot(fit_spectra.wavelength, fit_1_gaussian, color ='#1D3557', zorder = 4, linewidth = 3, linestyle = '--')
        except:
            print('Could not fit')
            make_fit = False

        
    
    # the spectra points
    ax.scatter(spec.wavelength, spec.flux, marker = '+', color ='#00798C', zorder = 2)

    
    # the location of the line
    ax.vlines(centre, 0, 1.3, color = '#30638E')
    ax.set_ylim(0,1.2)
    ax.set_xlim(x_lower, x_upper)

    # for the rectangle
    rect = mpatches.Rectangle((centre - fit_region, 0.2), fit_region * 2, 0.8, color = '#EDAE49', alpha = 0.5) 
    ax.add_patch(rect)

    # stuff about the axis
    plt.setp(ax.get_xticklabels(), visible=False)
    ax.tick_params(direction='in', axis='both', which='both', bottom=True,top=True, left=True, right=True)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    
    if make_fit:
        return fit_spectra.wavelength, residual, coeff, chi_squared_fit[0], rms
    else:
        return 0,0,[0,0,0,0,0,0,0,0,0,0],0,0

def info_plot(ax, file_name, output_csv, coeff, use_d_for_gauss_fit, chi, rms):
    ''' 
    Writes the fit parameters to the plot

    Parameters
    ----------
    ax : matplotlib axis
        The axis you are plotting to
    filename : string
        The name of the spectra to analyse
    output_csv : writing to file object?
        The csv which is storing the data
    coeff : array
        The best fit parameters
    use_d_for_gauss_fit : bool
        Which version of the gaussian fit. if true - print continuum value
    chi : float
        The chi squared value. note - will =0/error if something couldn't be fit
    rms : float
        Like the chi squared option but for the rms noise
    
    Returns
    -------

    '''

    ax.text(0, 0.9, file_name , style='normal',
        bbox={'facecolor':'#D1495B', 'alpha':0.5, 'pad':10})
    try:
        ax.text(0, 0.75, 'flux ratio: ' + str(round(coeff[0], 4)))
        ax.text(0, 0.65, 'centre $\lambda$: ' + str(round(coeff[1], 4)))
        ax.text(0, 0.55, '$\sigma$: ' + str(round(coeff[2], 4)))

        d = 1 # will be updated if i'm fitting to the continuum
        if use_d_for_gauss_fit:
            ax.text(0, 0.05, 'continuum: ' + str(round(coeff[-1], 4)))
            d = coeff[-1]

        fwhm = coeff[2]*2.35482 # 2 sqrt(2 ln(2))
        ew = -1 * coeff[0]* np.sqrt(2*np.pi * coeff[2]**2) * 1000 /d # in milli angstroms 

        ax.text(0, 0.45, 'FWHM: ' + str(round(fwhm, 4)))
        ax.text(0, 0.35, 'EW: ' + str(round(ew, 4)) + r' m $\AA$')
        ax.text(0, 0.25, '$\chi^2$: ' + str(round(chi, 4)))
        ax.text(0, 0.15, 'RMS: ' + str(round(rms, 4)))
        
  
    except:
        print('not printing parameters for star ' + file_name)

    output_csv.write(str(coeff[0]) + ',' + str(coeff[1]) + ',' +str(coeff[2]) + ',' + \
                         str(fwhm) + ',' + str(ew) + ',' + str(chi) + ',' + str(rms) + ',' + str(d) + '\n')
    ax.axis('off')

def residual_plot(ax_data, fit_spectra, res, x_lower, x_upper):
    ''' 
    Plots the residual values beneeth the spectra plot. Axis must lign up with the spectra plot.

    Parameters
    ----------
    ax_data : matplotlib axis
        The axis you are plotting to
    fit_spectra : array
        The wavelength to plot
    res : array
        The residual values
    coeff : array
        The best fit parameters
    x_lower : int
        The lower bound of the wavelength
    x_upper : int
        The lower bound of the wavelength
    
    Returns
    -------

    '''

    plt.setp(ax_data.get_yticklabels(), visible=False)
    ax_data.tick_params(direction='in', axis='both', which='both', bottom=True,top=True, left=False, right=False)
    ax_data.xaxis.set_minor_locator(AutoMinorLocator())

    plt.plot(fit_spectra, res, color = '#003D5B')
    
    # make sure the data correctly aligns with the spectral plot
    ax_data.set_xlim(x_lower, x_upper)

def gen_plots(n_spectra, filename_arr, rv_correction_arr, output_csv, spectra_path, ew_path, line_centre, x_lower, x_uppper):
    ''' 
    Outputs the plot images and all the corresponding fits to the spectra.

    Calls all other functions needed to calculate and plot the EWs.


    Parameters
    ----------
    n_spectra : int
        The number of stars we are working with
    output_csv : writing to file object?
        The csv which is storing the data
    line_centre : float
        The line we want to fit
    coeff : array
        The best fit parameters
    x_lower : int
        The lower bound of the wavelength
    x_upper : int
        The lower bound of the wavelength
    
    Returns
    -------
    '''

    fig = plt.figure(constrained_layout=True)
    gs = fig.add_gridspec(n_spectra*3, 2, width_ratios=[1,0.5], height_ratios= hight_ratio(n_spectra), figure=fig)

    fig.set_size_inches(10, n_spectra*3)

    # increments the spectrum
    j = 0
    
    use_d_for_gauss_fit = True


    # iterate over the plots and calculate spectra
    for i in range(n_spectra*3):

        ax_data = fig.add_subplot(gs[i,0]) # the spectra plots
        
        if i % 3 == 0: # the info and spectra plots

            
            res_wavelength, res, coeff, chi, rms = spectra_plot(ax_data, filename_arr[j], rv_correction_arr[j], spectra_path, ew_path, line_centre, x_lower,x_uppper, True, use_d_for_gauss_fit)

            # add the line centre and name of the star
            output_csv.write(str(line_centre) + ',' + filename_arr[j] + ',')
            
            ax_info = fig.add_subplot(gs[i:i+3,1])
            info_plot(ax_info, filename_arr[j], output_csv, coeff, use_d_for_gauss_fit, chi, rms)

            j +=1

        elif i % 3 == 1: # residual plot
            residual_plot(ax_data, res_wavelength, res, x_lower,x_uppper)
            

        else: # space between plots
            ax_data.axis('off')
    
    gs.update(wspace=0, hspace=0)
    plt.savefig(str(line_centre) + '_fit.jpeg')
    plt.close()