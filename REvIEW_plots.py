from my_imports import *

 # import all the fitting functions
from REvIEW_fitting import read_spectra, gauss_1_model, gauss_2_model, gauss_3_model, gauss_4_model, gauss_3_fit2, fit_spectra


def line_bounds(line, bounds = 2):
    ''' Function for iterating over lines in a line list

    Parameters
    ----------
    line : dataframe?
        The line to be fitted and the species
    bounds : int, optional
        The size of the region to be fitted. Will range from +- bounds centred on the line.
        Default range is 2.

    Returns
    -------
    list
        a list containing the line centre, the lower bound and the upper bound and the species
    '''

    # line, lower bound, upper bound
    return [line.wavelength, line.wavelength - bounds/2, line.wavelength + bounds/2, line.species]

def line_incrementer(linelist, lower_bound, upper_bound):
    ''' Function for incrementing over a range of lines. 

    Passing a line list and a wave length range you would like to search through, it returns an array which you can iterate through to find the lines 

    Parameters
    ----------
    linelist : array
        A pandas dataframe of a line list
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
    line_in_bounds = linelist[(linelist.wavelength > lower_bound) & (linelist.wavelength < upper_bound)]
    bounds_array = []

    # iterate through all the lines within the bounds and append them to the array

    for index, line in line_in_bounds.iterrows():
        bounds_array.append(line_bounds(line))
    
    return bounds_array

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

def make_residual(flux, model):
    '''
    Calculates the residual

    Parameters
    ----------
    flux : array
        The spectra
    model : array
        The output to fitting to the flux and then samplining it the same as the flux
        
    Returns
    -------
        array: the difference between the model and the flux
    
    '''
    return model - flux

def plot_fit(ax, spectra_slice, coeff):
    '''
    Plots the spectra according to the number of elements in the coeff array

    Parameters
    ----------
    ax : plt object
        The axis we are plotting to
    spectra_slice : np array
        The x array
    coeff : np array
        The coefficients of best fit for the spectra
        
    Returns
    -------
    minima : int
        The number of gaussians received to plot
     array : the residual array, 
     array : the chi squared fit of the model
    '''
    minima = int((coeff.shape[0]-1)/3)

    if minima == 1:
        # curvefit output
        model = gauss_1_model(spectra_slice.wavelength, *coeff)
        ax.plot(spectra_slice.wavelength, model, color ='#D1495B', zorder = 3, linewidth = 3, label = 'single gaussian fit')

        # verticle line fit of the central gaussian
        ax.plot([ coeff[1],coeff[1] ], [0, 1.3], color = '#30638E', linestyle = '--', dashes=(5, 1), alpha = 0.8, zorder = 5)

    elif minima == 2:
        # curvefit output
        model = gauss_2_model(spectra_slice.wavelength, *coeff)
        ax.plot(spectra_slice.wavelength, model, color ='#D1495B', zorder = 3, linewidth = 3, label = 'double gaussian fit')

        # verticle line fit of the central gaussian
        ax.plot([ coeff[1],coeff[1] ], [0, 1.3], color = '#30638E', linestyle = '--', dashes=(5, 1), alpha = 0.8, zorder = 5)
        # second gaussian
        ax.plot([ coeff[4],coeff[4] ], [0, 1.3], color = '#30638E', linestyle = '--', dashes=(2, 1), alpha = 0.5, zorder = 6)
    
    elif minima == 3:
        # curvefit output
        model = gauss_3_model(spectra_slice.wavelength, *coeff)
        ax.plot(spectra_slice.wavelength, model, color ='#D1495B', zorder = 3, linewidth = 3, label = 'triple gaussian fit')
        
        # verticle line fit of the central gaussian
        ax.plot([ coeff[1],coeff[1] ], [0, 1.3], color = '#30638E', linestyle = '--', dashes=(5, 1), alpha = 0.8, zorder = 5)
        # second gaussian
        ax.plot([ coeff[4],coeff[4] ], [0, 1.3], color = '#30638E', linestyle = '--', dashes=(2, 1), alpha = 0.5, zorder = 6)
        # third gaussian
        ax.plot([ coeff[7],coeff[7] ], [0, 1.3], color = '#30638E', linestyle = '--', dashes=(2, 1), alpha = 0.5, zorder = 7)
    
    elif minima == 4:
        # curvefit output
        model = gauss_4_model(spectra_slice.wavelength, *coeff)
        ax.plot(spectra_slice.wavelength, model, color ='#D1495B', zorder = 3, linewidth = 3, label = 'quad gaussian fit')

        # verticle line fit of the central gaussian
        ax.plot([ coeff[1],coeff[1] ], [0, 1.3], color = '#30638E', linestyle = '--', dashes=(5, 1), alpha = 0.8, zorder = 5)
        # second gaussian
        ax.plot([ coeff[4],coeff[4] ], [0, 1.3], color = '#30638E', linestyle = '--', dashes=(2, 1), alpha = 0.5, zorder = 6)
        # third gaussian
        ax.plot([ coeff[7],coeff[7] ], [0, 1.3], color = '#30638E', linestyle = '--', dashes=(2, 1), alpha = 0.5, zorder = 7)
        # fourth gaussian
        ax.plot([ coeff[10],coeff[10] ], [0, 1.3], color = '#30638E', linestyle = '--', dashes=(2, 1), alpha = 0.5, zorder = 7)
    else:
        return 0

    # See what the gaussian actually looks like
    ax.plot(spectra_slice.wavelength, gauss_1_model(spectra_slice, coeff[0], coeff[1], coeff[2], coeff[-1]), color ='#1D3557', zorder = 4, linewidth = 2, linestyle = '--', alpha = 0.6)

    return minima, make_residual(spectra_slice.flux, model), chisquare(model, spectra_slice.flux)[0]

def shift_model(coeff, min_wl):
    '''
    Shifts all the b values so they line up with the spectra plots

    Parameters
    ----------
    coeff : list
        The output of curvefit
    min_wl : float
        The scaling factor to make it lign up with the spectra
        
    Returns
    -------
        int: the flag to put on the info plot
    
    '''
    
    for i in range(len(coeff)):
        if i%3 == 1:
            coeff[i] += min_wl

    return coeff

def flag_info(flags):
    ''' 
    Makes the plot of the spectra and the fits.

    Calls the read spectra function and fitting function. Then collates the information and plots it.

    Parameters
    ----------
    flags : dict
        The flags that were raised during the fitting procedure
        
    Returns
    -------
        int: the flag to put on the info plot
    '''

    # "uninformed triple"     : uninformed_triple,        # 1
    # "triple improved fit"   : triple_improved_fit,      # 2
    # "minima error to triple": minima_error_to_triple,   # 3
    # "continuum error"       : continuum_error           # 4
    
    for key in flags:
        if flags[key] == True:
            # continuum error most important so return that first
            if key == "continuum error":
                return 4
            elif key == "triple improved fit":
                return 2
            elif key == "minima error to triple":
                return 3
            elif key == "uninformed triple":
                return 1
    # if its fine, return 0
    return 0

def spectra_plot(ax, filename, rv_correction, spectra_path, ew_path, line_loc, x_lower, x_upper):
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
    spectra_path : str
        The string of the directory where the spectra is kept
    ew_path : str
        The string of the directory where the EW .png's will be placed
    line_loc : float
        The centre of the line to fit
    x_lower : int
        The lower bound of the wavelength
    x_upper : int
        The lower bound of the wavelength
    
    Returns
    -------
    spectra_slice.wavelength : array
        Returning it to make sure the residual array is using exactly the same wavelengths as this plot
    residual : array
        The difference between the spectra and the fit
    fit_coeff : array
        The best fit parameters to the gaussian
    plot_type : int
        Whether the spectra was fit with 1, 2, 3 or 4 gaussians.
    flags : dict
        If there was any errors which occured during the fitting process. Keys given below
    '''
    # error flags:
    
    uninformed_triple = False # fit threw an error 
    triple_improved_fit = False    # the triple gaussian with a penalty was better than original 
    minima_error_to_triple = False # Unaccounted for number of minima found
    continuum_error = False # the curvefit couldnt fit the continuum
    no_data_to_fit  = False # don't run the fitting
    
    # read in the spectra
    spec = read_spectra(filename, rv_correction, spectra_path, ew_path)
    


    
    fit_region = 0.6 # DO NOT CHANGE: wavelengths around the line we are fitting to

    # Assuming that the minimum value is going to be the line
    spectra_slice = spec[(spec.wavelength > line_loc - fit_region) & (spec.wavelength < line_loc + fit_region)]
    
    # arbirtrary number of array points
    # Just in case we are at the end of the detecter and there are missing pixles
    if spectra_slice.shape[0] < 20:
        no_data_to_fit = True
        y_lim_min = 0
        y_lim_max = 1.2

    if not no_data_to_fit:
        # make a new wavelength array that has the same increments as the spectra
        # need to do this because all the new updates are based on a wavelength 
        # array from 0 to 1.2 (fitting bounds ext..)
        wl = spectra_slice.wavelength.to_numpy() - spectra_slice.wavelength.min()

        # Try catch for error handling - just in case
        try:
            fit_coeff, triple_improved_fit, minima_error_to_triple = fit_spectra(wl, spectra_slice.flux.to_numpy())
        except:
            # something went wrong - revert to triple gaussians
            fit_coeff, covar = gauss_3_fit2(wl, spectra_slice.flux)
            uninformed_triple = True

        

        # check for continuum errors
        if fit_coeff[-1] > 1.045 or fit_coeff[-1] < 0.92:
            continuum_error = True


        ### PLOTTING SECTION ###

        # get back in terms of actual spectra wavelengths
        fit_coeff = shift_model(fit_coeff, spectra_slice.wavelength.min())

        # plot the output of the fitting
        plot_type, residual, chi_squared_fit = plot_fit(ax, spectra_slice, fit_coeff)

        # updated so the plot bounds represent the minimum value
        # only do this if you can make a fit
        y_lim_min = max(spectra_slice.flux.min()-0.05, 0)
        y_lim_max = min(spectra_slice.flux.max()+0.05, 1.2)

    # Set limits outside the plotting if statement
    ax.set_ylim(y_lim_min, y_lim_max)


    # the spectra points
    ax.scatter(spec.wavelength, spec.flux, marker = '+', color ='#00798C', label = 'spectra', zorder = 2)

    # the location of the line
    ax.vlines(line_loc, 0, 1.3, color = '#30638E')

    ax.set_xlim(x_lower, x_upper)

    # for the rectangle
    rect = mpatches.Rectangle((line_loc - fit_region, 0.2), fit_region * 2, 0.8, color = '#EDAE49', alpha = 0.5) 
    ax.add_patch(rect)

    # stuff about the axis
    ax.xaxis.set_major_locator(MultipleLocator(0.5))
    plt.setp(ax.get_xticklabels(), visible=False)
    ax.tick_params(direction='in', axis='both', which='both', bottom=True,top=True, left=True, right=True)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.set_ylabel('Norm. flux')
    

    flags = {                                           # flag numerical values
    "continuum error"       : continuum_error,          # 4 (most important)
    "uninformed triple"     : uninformed_triple,        # 1
    "triple improved fit"   : triple_improved_fit,      # 2
    "minima error to triple": minima_error_to_triple,   # 3
    "No data to fit to"     : no_data_to_fit            # 5
    }

    if not no_data_to_fit:
        return spectra_slice.wavelength, residual, fit_coeff, chi_squared_fit, plot_type, flags
    else:
        return 0,0,[0,0,0,0],0,0, flags

def info_plot(ax, file_name, output_csv, coeff, chi, minima, flags):
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
        ew = -1 * coeff[0]* np.sqrt(2*np.pi * coeff[2]**2) * 1000 # in milli angstroms

        ax.text(0, 0.75, 'flux ratio: ' + str(round(coeff[0], 4)))
        ax.text(0, 0.65, 'centre $\lambda$: ' + str(round(coeff[1], 4)))
        ax.text(0, 0.55, '$\sigma$: ' + str(round(coeff[2], 4)))
        ax.text(0, 0.45, 'continuum: ' + str(round(coeff[-1], 4)))
        ax.text(0, 0.35, 'EW: ' + str(round(ew, 4)) + r' m $\AA$')
        ax.text(0, 0.25, '$\chi^2$: ' + str(round(chi, 4)))
        ax.text(0, 0.15, 'fit: ' + str(round(minima, 1)))
        if flags: # if there are flags
            flag = '--'
            for key in flags:
                if flags[key] == True:
                    flag = key
            ax.text(0, 0.05, 'flag: ' + str(flag))
    except:
        print('not printing parameters for star ' + file_name)


    # Writing string 
    #                     a                   b                      c
    #                     EW               chi squared             d              flag
    output_csv.write(str(coeff[0]) + ',' + str(coeff[1]) + ',' +str(coeff[2]) + ',' + \
                         str(ew) + ',' + str(chi) + ','  + str(coeff[-1]) + ',' + str(flag_info(flags)) + '\n')
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
    #ax_data.set_xticks(np.arange(x_lower + 0.25, x_upper + 0.25, 0.5))
    ax_data.xaxis.set_major_locator(MultipleLocator(0.5))
    plt.setp(ax_data.get_yticklabels(), visible=False)
    ax_data.tick_params(direction='in', axis='both', which='both', bottom=True,top=True, left=False, right=False)
    ax_data.xaxis.set_minor_locator(AutoMinorLocator())
    ax_data.set_xlabel(r'wavelength ($\AA$)')

    plt.plot(fit_spectra, res, color = '#003D5B')
    
    # make sure the data correctly aligns with the spectral plot
    ax_data.set_xlim(x_lower, x_upper)
    
def gen_plots(n_spectra, filename_arr, rv_correction_arr, output_csv, spectra_path, ew_path, line_centre, x_lower, x_uppper, species):
    ''' 
    Outputs the plot images and all the corresponding fits to the spectra.
    Calls all other functions needed to calculate and plot the EWs.
    Saves the figure without returning.

    Parameters
    ----------
    n_spectra : int
        The number of stars we are working with. Passed from the notebook driver
    filename_arr : list (str)
        A list of the file names of spectra to analyse in the spectra directory
    rv_correction_arr : list (float)
        Corresponding radial velocity corrections to the filename_arr
    output_csv : writing to file object?
        The csv which is storing the data
    spectra_path : str
        The string of the directory where the spectra is kept
    ew_path : str
        The string of the directory where the EW .png's will be placed
    line_centre : float
        The line we want to fit
    x_lower : int
        The lower bound of the wavelength
    x_upper : int
        The lower bound of the wavelength
    species : float
        The atomic number of the element we are fitting the line to
    
    Returns
    -------
    '''
    
    # Setting up plot information. Variable depending on the number of spectra
    fig = plt.figure(constrained_layout = True)
    gs = fig.add_gridspec(n_spectra*3, 2, width_ratios=[1,0.5], height_ratios= hight_ratio(n_spectra), figure=fig, hspace=0.0)

    fig.set_size_inches(10, n_spectra*3)
    fig.suptitle("Species: "+ str(species))
    

    # increments the spectrum
    j = 0

    # iterate over the plots and calculate spectra
    for i in range(n_spectra*3):
        
        ax_data = fig.add_subplot(gs[i,0]) # the spectra plots
        
        if i % 3 == 0: # the info and spectra plots

            # Calculating fit and plotting spectra
            res_wavelength, res, coeff, chi, plot_type, flags = spectra_plot(ax_data, filename_arr[j], rv_correction_arr[j], spectra_path, ew_path, line_centre, x_lower,x_uppper)

            # add the line centre and name of the star
            output_csv.write(str(line_centre) + ',' + filename_arr[j] + ',')
            
            # Plotting the info on the side of the spectra
            ax_info = fig.add_subplot(gs[i:i+3,1])
            info_plot(ax_info, filename_arr[j], output_csv, coeff, chi, plot_type, flags)

            j +=1

        elif i % 3 == 1: # residual plot
            residual_plot(ax_data, res_wavelength, res, x_lower,x_uppper)
            

        else: # space between plots
            ax_data.axis('off')
    
    gs.update(wspace=0, hspace=0)
    plt.savefig(str(line_centre) + '_fit.jpeg')
    plt.close()