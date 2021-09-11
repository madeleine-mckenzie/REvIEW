# How REvIEW works:
    A reference for maddie

## Python notebook driver:
The notebook has markdown comments and should step the user through how to run review. 
Just make sure the right paths are set and variables are correct for the given situation.

## Main loop which runs REvIEW:
The code iterates through all the lines in the line list, taking a segment of the spectra and then fiting a variable number of gaussians to determin the equivelent widths (EWs). 

The function which calls all other functions to do with spectra and fitting is ```gen_plots``` in ```REvIEW_plots.py```.

Next the ```info_plot``` function is called which puta all the info to the side of the plot.

Lastly ```residual_plot``` is called to plot the difference between the spectra and the model below the spectra plot.

### Spectra_plot

Arguments in the python docs.

#### Flags:
The fitting function will output a range of flags depending on the fitting process.

1) uninformed_triple =  fit threw an error 
2) triple_improved_fit = the triple gaussian with a penalty was better than original 
3) minima_error_to_triple = Unaccounted for number of minima found
4) continuum_error = the curvefit couldnt fit the continuum
5) no_data_to_fit  = don't run the fitting

#### reading in the spectra

Both the file path and the EW path must be passed so we can cd into the right directory to import the spectra each time.

This is a potential bottleneck as the spectra isnt stored permanently each time. Plotting (probably gridspec) is still the most time consuming part of the fitting.

#### Fitting the spectra

This must always be done using a fitting region of +-0.6$\AA$ (1.2 $\AA$ in totoal). All the fitting procedures (specifically the fitting bounds) are hard coded in to work with a 1.2 $\AA$ wavelength region

Pandas array slicing is used to extract the spectra only within this wavelength range. If there are missing pixels (with less than 20 left in the spectra), do not try to fit the spectra otherwise the fitting routines will crash.

If there is data to fit, create a new wavelength array that goes from - to 1.2 by subtracting the smallest wavelength from the data.

*note* Future versions of review with ML -> might need to do a splice here to make sure the data has the correct number of points (86). Not all spectra have exactly this number of points (rounding errors).

The main call to the fitting function is enclosed in a try except block just in case the fitting throws an unexpected error. If it does, the spectra will try to be fit with the conventional "uninformed triple" which is equivelent to the original fitting routine for REvIEW. One of the flags is raised and will appear on the output csv to later remove (if you want).

Once the spectrum has been fit, check that it did not run into one of the boundaries for fitting. If it did, you can be pretty sure its a rubbish fit. These should definialy be removed later on in the fitting process (flag 4). 

Finally, shift the b term in the gaussian back in terms of the original wavelength and send it through the gaussian to get the model flux.

The rest of the function is mainly plotting things. Other functions have been documented. 

## REvIEW fitting routines 

Requires the wavelength region (should be 0 to 1.2) and flux to be passed to the function.

Two flags are included here. ```tripe_improved_fit``` and ```minima_error_to_triple = False```. These are described in the docs.

### grad_change routine

```grad_change``` is a pre-processing algorithm that gives an estimate of the number of minima in the spectra.

It iterates through the spectra from left to right and counts the number of consecutive points either going up or down. Previous versions used a ```savgol_filter``` to reduce the noise however this didn't seem to work as well in the long run.

In the loop going over the spectra, the counter j is used to determin the gradient at each point. j is True if there is a negative gradient (i.e. going into a minima) and False if j is positive (i.e. coming out of a minima).

```count_negative``` and ```count_positive``` are ints which keep track of how many previous points were either positive or negative. This is useful later for removing noisy points (points that were only positive or negative for 3 data points).

Once we have iterated through the array, only the actual minima are selected. Currently the criterea to do this is:

#1. ```change_number[i][0] == 0```: we are looking at a minima point (-ive to +ive grad)
#2. ```change_number[i][1] >3```: there had to be at least 3 points before it to classify as a peak change_number[i+1][1] >3   means there had to be at least 3 points after it to classify as a peak (noise @ bottom of the peak??)
#3. ```change_number[i][3] < 0.96```: the peak must be greater than 0.96. This doesnt work so well for poorly normalised continuums.

There also a number of conditions at the end for handling edge cases. 

The function returns only the valid/suitable minima and has the form:
    [a, b, c, d]

a) j value: 0 or a 1 corresponding to whether it was a minima or maxima turning point. As we are looking for minima, we only take ones with 0
b) ```count_negative```: represents the minima points (gradient going negative to positive from left to right). The higher the value, the longer the function has been negative for.
c) wavelength: the wavelength of where the turning point is.
d) flux: the flux of the turning point. Will be used as an estimate of the a value when we get to fitting the gaussians. 

### Dealing with more than 4 minima. 
The algorithms can only handle up to 4 gaussians at this point. Therefore if we receive more than 4, we have to decide which ones we are going to keep and which will be thrown away. This is handled by the ```reduce_minima``` funciton.

This funciton is heckin confusing actually. What was I thinking? If there is an even number of peaks, take the top 4, otherwise take the top 3.
In hind sight, this could have just been dropped to the 4 most significant peaks... Maybe something to change later.

Optional function to check wether the sigma estimates are reasonable.

### Fitting to the spectra
Next in ```fit_spectra```, the number of minima will dictate which if statment it goes into. In each case, the number of gaussians identified will be fit along with an "uininformed triple" which comes from the original REvIEW function. For >=2 minima, different options are avaliable depending on where the secondary gaussians are. 

