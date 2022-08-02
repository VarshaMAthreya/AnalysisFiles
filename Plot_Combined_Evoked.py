# -*- coding: utf-8 -*-
"""
Created on Thu Jan 27 00:20:55 2022

@author: vmysorea
"""

###Grand averaging across subjects - GDT 
import sys
import warnings
from scipy.io import savemat
sys.path.append('C:/Users/vmysorea/mne-python/')
sys.path.append('C:/Users/vmysorea/ANLffr/')
import mne 
from anlffr.helper import biosemi2mne as bs
from matplotlib import pyplot as plt
import numpy as np 
import os
bs.importbdf
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')
plt.rcParams['figure.figsize'] = [12, 8]
plt.rcParams['figure.dpi']  = 120

data_loc = ('C:/Users/vmysorea/OneDrive - purdue.edu/Desktop/PhD/GDT_Analysis/EP_GDT/ITC Measures/Above_35/')

Subjects = ['S072', 'S088']                                        ## Above 35 years
#Subjects = [ 'S273', 'S268', 'S269', 'S274', 'S282', 'S285', 'S277', 'S279','S285']  ## Below 35 years

a = mne.Evoked (data_loc + 'Evoked_1_S072')
b = mne.Evoked (data_loc + 'Evoked_1_S088')

grand_average_1 = mne.grand_average([a, b])
print (grand_average_1)
grand_average_1.plot

mne.viz.plot_compare_evokeds (evoked_1)
