# Basics
import os
import numpy as np
import pandas as pd

# Scipy
from scipy.stats import chisquare
from scipy import interpolate
from scipy.optimize import curve_fit

# Plotting
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter, AutoMinorLocator, AutoLocator, MaxNLocator
import matplotlib.gridspec as gridspec
from matplotlib.colorbar import Colorbar
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from matplotlib import colors
import matplotlib.ticker as ticker
from matplotlib import rcParams

# neural network add in
from NN.ffnn import FFNN
from NN.scalar import Scalar

# RC params
rcParams['font.family'] = 'serif'
rcParams['font.size'] = 18
rcParams["axes.edgecolor"] = 'black'
rcParams["legend.edgecolor"] = '0.8'
plt.rcParams.update({'errorbar.capsize': 2})
plt.rcParams['mathtext.fontset'] = 'dejavuserif'
plt.rcParams['text.latex.preamble']=r'\usepackage{amsmath}'

def fmt(x, pos):
    ''' Used for the x10^ for colourbars'''
    if x == 0:
        return r'0'
    a, b = '{:.1e}'.format(x).split('e')
    b = int(b)
    return r'${} \times 10^{{{}}}$'.format(a, b)

def test_import():
    print('the import is working')