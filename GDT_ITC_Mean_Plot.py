# -*- coding: utf-8 -*-
"""
Created on Thu Jan 27 01:59:03 2022

@author: vmysorea
"""

import sys
sys.path.append('C:/Users/vmysorea/mne-python/')
sys.path.append('C:/Users/vmysorea/ANLffr/')
from matplotlib import pyplot as plt
from scipy import io
import numpy as np
from scipy.stats import sem


### Loading data to plot ITC of ALL subjects - NOT ACROSS AGE

data_loc = 'C:/Users/vmysorea/OneDrive - purdue.edu/Desktop/PhD/GDT_Analysis/EP_GDT/ITC Measures/Common/1-14Hz/'
Subjects = ['S072', 'S078', 'S088', 'S308', 'S291', 'S273', 'S268', 'S269', 'S274', 'S282', 'S285', 'S277', 'S279','S285', 'S303']

### ITC 1

ITC1_Mean = np.zeros((len(Subjects),5736))
for sub in range(len(Subjects)):
    subject = Subjects[sub]
    dat = io.loadmat(data_loc + 'ITC1_' + subject, squeeze_me=True)
    
    dat.keys()
    itc1=dat['itc1']
    freqs =dat['freqs']
    #n_channels=dat['n_channels']
    t1= dat['t']
    
    itc1_ave = itc1[:,:].mean(axis=0)
    ITC1_Mean[sub,:] = itc1_ave

ITC1_Mean_total = ITC1_Mean.mean(axis=0)
#ITC1_std = np.std(ITC1_Mean, axis=0)
ITC1_sem = sem(ITC1_Mean)
#plt.errorbar(t1, ITC1_Mean_total, yerr=ITC1_std, color='red')
#figure1 = plt.plot(t1,ITC1_Mean_total)
#plt.xlim([-0.1,1.1])

tnew1 = t1[t1>0]
ITC1_Peak = ITC1_Mean_total [t1>0]
print(max(ITC1_Peak))

### ITC 2

ITC2_Mean = np.zeros((len(Subjects),5736))
for sub in range(len(Subjects)):
    subject = Subjects[sub]
    dat = io.loadmat(data_loc + 'ITC2_' + subject, squeeze_me=True)
    
    dat.keys()
    itc2=dat['itc2']
    freqs =dat['freqs']
    #n_channels=dat['n_channels']
    t2 = dat['t']
    
    itc2_ave = itc2[:,:].mean(axis=0)
    ITC2_Mean[sub,:] = itc2_ave

ITC2_Mean_total = ITC2_Mean.mean(axis=0)
#ITC2_std = np.std(ITC2_Mean, axis=0)
ITC2_sem = sem(ITC2_Mean)
#plt.errorbar(t2, ITC2_Mean_total, yerr=ITC2_sem, color='red')
#figure2 = plt.plot(t2,ITC2_Mean_total)
#plt.xlim([-0.1,1.1])

tnew2 = t2[t2>0]
ITC2_Peak = ITC2_Mean_total [t2>0]
print(max(ITC2_Peak))


### ITC 3
ITC3_Mean = np.zeros((len(Subjects),5736))
for sub in range(len(Subjects)):
    subject = Subjects[sub]
    dat = io.loadmat(data_loc + 'ITC3_' + subject, squeeze_me=True)
    
    dat.keys()
    itc3=dat['itc3']
    freqs =dat['freqs']
    #n_channels=dat['n_channels']
    t3 = dat['t']
    
    itc3_ave = itc3[:,:].mean(axis=0)
    ITC3_Mean[sub,:] = itc3_ave

ITC3_Mean_total = ITC3_Mean.mean(axis=0)
#ITC3_std = np.std(ITC3_Mean, axis=0)
ITC3_sem = sem(ITC3_Mean)
print(ITC3_sem)
plt.errorbar(t1, ITC3_Mean_total, yerr=ITC3_sem, color='red')
figure3 = plt.plot(t1,ITC3_Mean_total,label='')
plt.xlim([-0.1,1.1])

tnew3 = t3[t3>0]
ITC3_Peak = ITC3_Mean_total [t3>0]
print(max(ITC3_Peak))

### Subplots - All subjects
fig, ax = plt.subplots(3,1, sharex=True,sharey=True, constrained_layout=True)
ax[0].errorbar(t1, ITC1_Mean_total,yerr=ITC1_sem, color='darkblue', linewidth=2, ecolor='lightsteelblue')
ax[0].set_title('ITC - Gap 16 ms', loc='center', fontsize=10 )
ax[1].errorbar(t2, ITC2_Mean_total,yerr=ITC2_sem, color='purple', linewidth=2, ecolor='thistle')
ax[1].set_title('ITC - Gap 32 ms', loc='center',fontsize=10)
ax[2].errorbar(t3, ITC3_Mean_total,yerr=ITC3_sem, color='green', linewidth=2, ecolor='palegreen')
ax[2].set_title('ITC - Gap 64 ms', loc='center',fontsize=10 )
plt.xlim([-0.1,1.1])
plt.ylim([0.02,0.09])
plt.xlabel('Time (in seconds)')
#fig.text(-0.03,0.5, 'ITC Value', va='center',rotation ='vertical')
fig.suptitle('ITC for the gap durations (N=15)', x =0.55)
params = {'legend.fontsize': 'x-large',
          'figure.figsize': (7, 5),
         'ytick.labelsize':'xx-small',
         'ytick.major.pad': '6'}
plt.rcParams.update(params)
#plt.tight_layout()
fig.supylabel('ITC Value')
plt.savefig('C:/Users/vmysorea/OneDrive - purdue.edu/Desktop/PhD/ARO_Figures/ITC123_Combined_1-14Hz.png', dpi=300)




### ITC 1 - Plotting across ages

data_loc = 'C:/Users/vmysorea/OneDrive - purdue.edu/Desktop/PhD/GDT_Analysis/EP_GDT/ITC Measures/Above 35/'
Subjects_O = ['S072', 'S078', 'S088', 'S308', 'S291']                                        ## Above 35 years - ADDDD!!!!!

ITC1_Mean_O = np.zeros((len(Subjects_O),5736))
for sub in range(len(Subjects_O)):
    subject = Subjects_O[sub]
    dat = io.loadmat(data_loc + 'ITC1_' + subject, squeeze_me=True)
    
    dat.keys()
    itc1=dat['itc1']
    freqs =dat['freqs']
    n_channels=dat['n_channels']
    t1 = dat['t']
    
    itc1_ave = itc1[:,:].mean(axis=0)
    ITC1_Mean_O[sub,:] = itc1_ave

ITC1_Mean_total_O = ITC1_Mean_O.mean(axis=0)
#itc3_std = np.std(ITC3_Mean, axis=0)
#plt.errorbar(t, ITC3_Mean_total, yerr=itc3_std, color='red')
figure = plt.plot(t1,ITC1_Mean_total_O,label='Above 35 years (N = 5)')
plt.xlim([-0.1,1.1])

tnew1 = t1[t1>0]
ITC1_Peak_O = ITC1_Mean_total_O [t1>0]
print(max(ITC1_Peak_O))

###Young subjects 
data_loc = 'C:/Users/vmysorea/OneDrive - purdue.edu/Desktop/PhD/GDT_Analysis/EP_GDT/ITC Measures/Below_35/'
Subjects_Y = [ 'S273', 'S268', 'S269', 'S274', 'S282', 'S285', 'S277', 'S279','S285', 'S303']  ## Below 35 years -ADDDD!!!!!

ITC1_Mean_Y = np.zeros((len(Subjects_Y),5736))
for sub in range(len(Subjects_Y)):
    subject = Subjects_Y[sub]
    dat = io.loadmat(data_loc + 'ITC1_' + subject, squeeze_me=True)
    
    dat.keys()
    itc1=dat['itc1']
    freqs =dat['freqs']
    n_channels=dat['n_channels']
    t1 = dat['t']
    
    itc1_ave = itc1[:,:].mean(axis=0)
    ITC1_Mean_Y[sub,:] = itc1_ave

ITC1_Mean_total_Y = ITC1_Mean_Y.mean(axis=0)
#itc3_std = np.std(ITC3_Mean, axis=0)
#plt.errorbar(t, ITC3_Mean_total, yerr=itc3_std, color='red')
figure = plt.plot(t1,ITC1_Mean_total_Y, label = 'Below 35 years (N= 10)')
plt.legend()
plt.xlabel('Time (in seconds)')
plt.ylabel('ITC Value')
plt.title ('ITC Value - Gap of 16 ms')
plt.rcParams["figure.figsize"] = (5.5,5)
plt.tight_layout()
plt.savefig('C:/Users/vmysorea/OneDrive - purdue.edu/Desktop/PhD/ARO_Figures/ITC1_Combined.png', dpi=300)

tnew1 = t1[t1>0]
ITC1_Peak_Y = ITC1_Mean_total_Y [t1>0]
print(max(ITC1_Peak_Y))

###ITC 2

### Loading data to plot ITC of the subjects
data_loc = 'C:/Users/vmysorea/OneDrive - purdue.edu/Desktop/PhD/GDT_Analysis/EP_GDT/ITC Measures/Above_35/'

Subjects_O = ['S072', 'S078', 'S088', 'S308', 'S291']                                        ## Above 35 years - ADDDD!!!!!

ITC2_Mean_O = np.zeros((len(Subjects_O),5736))
for sub in range(len(Subjects_O)):
    subject = Subjects_O[sub]
    dat = io.loadmat(data_loc + 'ITC2_' + subject, squeeze_me=True)
    
    dat.keys()
    itc2=dat['itc2']
    freqs =dat['freqs']
    #n_channels=dat['n_channels']
    t1 = dat['t']
    
    ITC2_ave = itc2[:,:].mean(axis=0)
    ITC2_Mean_O[sub,:] = ITC2_ave

ITC2_Mean_total_O = ITC2_Mean_O.mean(axis=0)
#itc3_std = np.std(ITC3_Mean, axis=0)
#plt.errorbar(t, ITC3_Mean_total, yerr=itc3_std, color='red')
figure_1 = plt.plot(t1,ITC2_Mean_total_O,label='Above 35 years (N = 5')
plt.xlim([-0.1,1.1])

tnew1 = t1[t1>0]
ITC2_Peak_O = ITC2_Mean_total_O [t1>0]
print(max(ITC2_Peak_O))

###Young subjects 
data_loc = 'C:/Users/vmysorea/OneDrive - purdue.edu/Desktop/PhD/GDT_Analysis/EP_GDT/ITC Measures/Below_35/'
Subjects_Y = [ 'S273', 'S268', 'S269', 'S274', 'S282', 'S285', 'S277', 'S279','S285', 'S303']  ## Below 35 years -ADDDD!!!!!

ITC2_Mean_Y = np.zeros((len(Subjects_Y),5736))
for sub in range(len(Subjects_Y)):
    subject = Subjects_Y[sub]
    dat = io.loadmat(data_loc + 'ITC2_' + subject, squeeze_me=True)
    
    dat.keys()
    itc2=dat['itc2']
    freqs =dat['freqs']
    #n_channels=dat['n_channels']
    #t1 = dat['t']
    
    ITC2_ave = itc2[:,:].mean(axis=0)
    ITC2_Mean_Y[sub,:] = ITC2_ave

ITC2_Mean_total_Y = ITC2_Mean_Y.mean(axis=0)
#itc3_std = np.std(ITC3_Mean, axis=0)
#plt.errorbar(t, ITC3_Mean_total, yerr=itc3_std, color='red')
figure_1 = plt.plot(t1,ITC2_Mean_total_Y, label = 'Below 35 years (N= 10)')
plt.legend()
plt.xlabel('Time (in seconds)')
plt.ylabel('ITC Value')
plt.title ('ITC Value - Gap of 32 ms')
plt.rcParams["figure.figsize"] = (5.5,5)
plt.tight_layout()
plt.savefig('C:/Users/vmysorea/OneDrive - purdue.edu/Desktop/PhD/ARO_Figures/ITC2_Combined.png', dpi=300)

tnew1 = t1[t1>0]
ITC2_Peak_Y = ITC2_Mean_total_Y [t1>0]
print(max(ITC2_Peak_Y))

### ITC 3

### Loading data to plot ITC of the subjects
data_loc = 'C:/Users/vmysorea/OneDrive - purdue.edu/Desktop/PhD/GDT_Analysis/EP_GDT/ITC Measures/Above_35/'

Subjects_O = ['S072', 'S078', 'S088', 'S308', 'S291']                                        ## Above 35 years - ADDDD!!!!!

itc3_Mean_O = np.zeros((len(Subjects_O),5736))
for sub in range(len(Subjects_O)):
    subject = Subjects_O[sub]
    dat = io.loadmat(data_loc + 'itc3_' + subject, squeeze_me=True)
    
    dat.keys()
    itc3=dat['itc3']
    freqs =dat['freqs']
    n_channels=dat['n_channels']
    t1 = dat['t']
    
    itc3_ave = itc3[:,:].mean(axis=0)
    itc3_Mean_O[sub,:] = itc3_ave

itc3_Mean_total_O = itc3_Mean_O.mean(axis=0)
#itc3_std = np.std(ITC3_Mean, axis=0)
#plt.errorbar(t, ITC3_Mean_total, yerr=itc3_std, color='red')
figure_2 = plt.plot(t1,itc3_Mean_total_O,label='Above 35 years (N = 5')
plt.xlim([-0.1,1.1])

tnew1 = t1[t1>0]
itc3_Peak_O = itc3_Mean_total_O [t1>0]
print(max(itc3_Peak_O))

###Young subjects 
data_loc = 'C:/Users/vmysorea/OneDrive - purdue.edu/Desktop/PhD/GDT_Analysis/EP_GDT/ITC Measures/Below_35/'
Subjects_Y = [ 'S273', 'S268', 'S269', 'S274', 'S282', 'S285', 'S277', 'S279','S285', 'S303']  ## Below 35 years -ADDDD!!!!!

itc3_Mean_Y = np.zeros((len(Subjects_Y),5736))
for sub in range(len(Subjects_Y)):
    subject = Subjects_Y[sub]
    dat = io.loadmat(data_loc + 'itc3_' + subject, squeeze_me=True)
    
    dat.keys()
    itc3=dat['itc3']
    freqs =dat['freqs']
    n_channels=dat['n_channels']
    t1 = dat['t']
    
    itc3_ave = itc3[:,:].mean(axis=0)
    itc3_Mean_Y[sub,:] = itc3_ave

itc3_Mean_total_Y = itc3_Mean_Y.mean(axis=0)
#itc3_std = np.std(ITC3_Mean, axis=0)
#plt.errorbar(t, ITC3_Mean_total, yerr=itc3_std, color='red')
figure_2 = plt.plot(t1,itc3_Mean_total_Y, label = 'Below 35 years (N= 10)')
plt.legend()
plt.xlabel('Time (in seconds)')
plt.ylabel('ITC Value')
plt.title ('ITC Value - Gap of 64 ms')
plt.rcParams["figure.figsize"] = (5.5,5)
plt.tight_layout()
plt.savefig('C:/Users/vmysorea/OneDrive - purdue.edu/Desktop/PhD/ARO_Figures/ITC3_Combined.png', dpi=300)

tnew1 = t1[t1>0]
itc3_Peak_Y = itc3_Mean_total_Y [t1>0]
print(max(itc3_Peak_Y))

### Subplots
fig, ax = plt.subplots(3,1, sharex=True,sharey=True, constrained_layout=True)
ax[0].plot(t1, ITC1_Mean_total_O,label='Above 35 years (N= 5)')
ax[0].plot(t1, ITC1_Mean_total_Y,label='Below 35 years (N= 10)')
ax[0].set_title('ITC - Gap 16 ms', loc='center', fontsize=10 )
ax[1].plot(t1, ITC2_Mean_total_O, t1, ITC2_Mean_total_Y)
ax[1].set_title('ITC - Gap 32 ms', loc='center',fontsize=10)
ax[2].plot(t1, itc3_Mean_total_O, t1, itc3_Mean_total_Y)
ax[2].set_title('ITC - Gap 64 ms', loc='center',fontsize=10 )
plt.xlim([-0.1,1.1])
ax[0].legend()
plt.xlabel('Time (in seconds)')
fig.text(0.0001,0.5, 'ITC Value', va='center',rotation ='vertical')
plt.suptitle('ITC for the gap durations')
plt.rcParams["figure.figsize"] = (5.5,5)
#plt.tight_layout()
plt.savefig('C:/Users/vmysorea/OneDrive - purdue.edu/Desktop/PhD/ARO_Figures/ITC123_Combined.png', dpi=300)
