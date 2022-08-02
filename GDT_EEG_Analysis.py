# -*- coding: utf-8 -*-
"""
Created on Sun Nov  7 18:27:02 2021

@author: vmysorea
"""

import sys
import warnings
sys.path.append('C:/Users/vmysorea/mne-python/')
sys.path.append('C:/Users/vmysorea/ANLffr/')
import mne 
from anlffr.helper import biosemi2mne as bs
from matplotlib import pyplot as plt
import numpy as np 
import os
import fnmatch
from mne.time_frequency import tfr_multitaper
bs.importbdf
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')
plt.rcParams['figure.figsize'] = [12, 8]
plt.rcParams['figure.dpi']  = 120

froot = 'D:/PhD/Data/MTB_EP - GDT, Binding, mTRF/GDT/'              #file location
subjlist = ['S105']                                                 #Load subject folder
condlist = [1,2,3]                                                  #List of conditions- Here 3 GDs - 16, 32, 64 ms
condnames = ['0.016 ms', '0.032 ms','0.064 ms']

for subj in subjlist:
    evokeds = []
    itcs = []
    powers = []
    # Load data and read event channel
    fpath = froot + subj + '/'
    bdfs = fnmatch.filter(os.listdir(fpath), subj +
                          '_GDT*.bdf')
    
    # Load data and read event channel
    rawlist = []
    evelist = []
    
    for k, rawname in enumerate(bdfs):
        rawtemp, evestemp = bs.importbdf(fpath + rawname, verbose='DEBUG',
                                         refchans=['EXG1', 'EXG2'])
        rawlist += [rawtemp, ]
        evelist += [evestemp, ]
    raw, eves = mne.concatenate_raws(rawlist, events_list=evelist)

    #To check for bad channels
    raw.plot(duration=25.0, n_channels=32, scalings=dict(eeg=100e-6))   
    raw.info ['bads'].extend (['A1', 'A2', 'A6', 'A25', 'A24', 'A28', 'A29'])

#Filtering
raw.filter (1., 40.)
raw.info

#Blink Rejection

from anlffr.preproc import find_blinks
blinks = find_blinks(raw)
raw.plot(events=blinks, duration=25.0, n_channels=32, scalings=dict(eeg=200e-6))

from mne import compute_proj_epochs
epochs_blinks = mne.Epochs(raw, blinks, event_id=998, baseline=(-0.25, 0.25),
                           reject=dict(eeg=500e-6), tmin=-0.25, tmax=0.25)

blink_proj = compute_proj_epochs(epochs_blinks, n_eeg=1)

raw.add_proj(blink_proj)

raw.plot_projs_topomap()       #Visualizing the spatial filter

#Onset responses
for c, cond in enumerate(condlist):

        condname = condnames[c]
        epochs = mne.Epochs(
            raw, eves, cond, tmin=-0.3, proj=True,
            tmax=1.1, baseline=(-0.3, 0.0),
            reject=dict(eeg=200e-6))
        evoked = epochs.average()
        evokeds += [evoked, ]
picks=['A31']
evoked.plot(picks=picks,titles='Onset Response - Middle-Aged Adult')        

epochs = mne.Epochs(raw, eves, event_id=[1,2,3], baseline=(-0.3, 0), proj=True,
                    tmin=-0.3, tmax=1.1, reject=dict(eeg=200e-6))
evoked = epochs.average() 
picks=['A31']
evoked.plot(picks=picks,titles='Onset - Middle-Aged Adult')

##Creating manual events to add the responses of each gap size 

fs = raw.info['sfreq']
gap_durs = [.016, .032, 0.064]
eves_manual = np.zeros((3*eves.shape[0],3))
for k in range(1,eves.shape[0]):
    for m in range(3):
        current_eves = eves[k,:].copy()
        gap_samps = int(np.round(gap_durs[int(current_eves[2])-1] * fs))
        current_eves[0] = current_eves[0] + (m + 1) * int(np.round(0.5*fs)) + m*gap_samps
        eves_manual[((k-1)*2) + m, :] = current_eves

eves_manual = np.int64(eves_manual)

#Epoching

epochs_1 = mne.Epochs(raw, eves_manual[:], event_id=[1], baseline=(-0.3, 0), proj=True,
                    tmin=-0.3, tmax=1.1, reject=dict(eeg=200e-6))
epochs_2 = mne.Epochs(raw, eves_manual, event_id=[2], baseline=(-0.3, 0), proj=True,
                    tmin=-0.3, tmax=1.1, reject=dict(eeg=200e-6))
epochs_3 = mne.Epochs(raw, eves_manual, event_id=[3], baseline=(-0.3, 0), proj=True,
                    tmin=-0.3, tmax=1.1, reject=dict(eeg=200e-6))


#Onset Response- Plotting responses for all conditions for single channel
#t = evoked.times
#x = np.zeros((t.shape[0], len(condnames)))
#ch = 30
#for k in range(len(condnames)):
#    x[:, k] = evokeds[k].data[ch, :] * 1.0e6
#plt.plot(t, x)
#plt.xlabel('Time (s)')
#plt.ylabel('Evoked Response (uV)')
#plt.legend(condnames)


#Averaging

evoked_1 = epochs_1.average() 
evoked_2 = epochs_2.average() 
evoked_3 = epochs_3.average() 

picks=['A31']
evoked_1.plot(picks=picks,titles='GDT_16ms - Middle-Aged Adult')
evoked_2.plot(picks=picks,titles='GDT_32ms - Middle-Aged Adult')
evoked_3.plot(picks=picks,titles='GDT_64ms - Middle-Aged Adult')


picks=['A31', 'A32']
evokeds_1 = dict(GDT_16ms=evoked_1, GDT_32ms=evoked_2, GDT_64ms=evoked_3)
mne.viz.plot_compare_evokeds(evokeds_1, combine='mean',title='GDT - Middle-Aged Adult')

evoked_1.plot(spatial_colors=True,titles='GDT_16ms - Older Adult')
evoked_2.plot(spatial_colors=True,titles='GDT_32ms - Older Adult')
evoked_3.plot(spatial_colors=True,titles='GDT_64ms - Older Adult')

#evoked_1.save ('D:/PhD/Data/MTB_EP - GDT, Binding, mTRF/GDT/S285/S285_GDT_16ms-ave.fif(.gz)')
#evoked_2.save ('D:/PhD/Data/MTB_EP - GDT, Binding, mTRF/GDT/S285/S285_GDT_32ms-ave.fif(.gz)')
#evoked_3.save ('D:/PhD/Data/MTB_EP - GDT, Binding, mTRF/GDT/S285/S285_GDT_64ms-ave.fif(.gz)')
#mne.write_evokeds('D:/PhD/Data/MTB_EP - GDT, Binding, mTRF/GDT/S268/S268_GDT_Combined-ave.fif(.gz)', (evoked_1, evoked_2, evoked_3))

#mne.read_evokeds('D:/PhD/Data/MTB_EP - GDT, Binding, mTRF/GDT/S268/S268_GDT_16ms-ave.fif(.gz)')
#mne.read_evokeds('D:/PhD/Data/MTB_EP - GDT, Binding, mTRF/GDT/S268/S268_GDT_32ms-ave.fif(.gz)')
#mne.read_evokeds('D:/PhD/Data/MTB_EP - GDT, Binding, mTRF/GDT/S268/S268_GDT_64ms-ave.fif(.gz)')

# Compute evoked response using ITC
freqs = np.arange(1., 30., 1.)
n_cycles = freqs * 0.2
#epochs_induced = epochs.copy().subtract_evoked()
picks = [31]
power_1, itc_1 = tfr_multitaper(epochs_1, freqs, n_cycles, picks=picks,
                            time_bandwidth=4.0, n_jobs=-1)
itc_1.apply_baseline(baseline=(-0.5, 0))
power_1.apply_baseline(baseline=(-0.5, 0), mode='logratio')

freqs = np.arange(1., 30., 1.)
n_cycles = freqs * 0.2
#epochs_induced = epochs.copy().subtract_evoked()
picks = [31]
power_2, itc_2 = tfr_multitaper(epochs_2, freqs, n_cycles, picks=picks,
                            time_bandwidth=4.0, n_jobs=-1)
itc_2.apply_baseline(baseline=(-0.5, 0))
power_2.apply_baseline(baseline=(-0.5, 0), mode='logratio')

freqs = np.arange(1., 30., 1.)
n_cycles = freqs * 0.2
#epochs_induced = epochs.copy().subtract_evoked()
picks = [31]
power_3, itc_3 = tfr_multitaper(epochs_3, freqs, n_cycles, picks=picks,
                            time_bandwidth=4.0, n_jobs=-1)
itc_3.apply_baseline(baseline=(-0.5, 0))
power_3.apply_baseline(baseline=(-0.5, 0), mode='logratio')

itcs += [itc_1, itc_2, itc_3]
powers += [power_1, power_2, power_3]

#Plotting ITC and Power plots for the three conditions
power_1.plot([0], baseline=(-0.5, 0), mode='mean', title='Gap duration of 16 ms - Power')
itc_1.plot([0], title='Gap duration of 16 ms - Intertrial Coherence (Middle-Aged)')

power_2.plot([0], baseline=(-0.5, 0), mode='mean', title='Gap duration of 32 ms - Power')
itc_2.plot([0], title='Gap duration of 32 ms- Intertrial Coherence (Middle-Aged)')

power_3.plot([0], baseline=(-0.5, 0), mode='mean', title='Gap duration of 64 ms - Power')
itc_3.plot([0], title='Gap duration of 64 ms - Intertrial Coherence (Middle-Aged)')

# Plot single channel evoked responses for all conditions
#t = evoked.times
#x = np.zeros((t.shape[0], len(condnames)))
#ch = 30

#for k in range(len(condnames)):
 #       x[:, k] = evokeds[k].data[ch, :] * 1.0e6
#plt.plot(t, x)
#plt.xlabel('Time (s)')
#plt.ylabel('Evoked Response (uV)')
#plt.legend(condnames)


#Plotting itc for the three epochs
#t = itc_1.times  # Just in case
#fselect = freqs < 15.
#y = np.zeros((t.shape[0], len(condnames)))
#for k in range(len(condnames)):
 #   perChan = itcs[k].data[:, fselect, :].mean(axis=1)
 #  y[:, k] = perChan.mean(axis=0) ** 2.
 #   plt.figure()
 #   plt.plot(t, y)
 #   plt.xlabel('Time (s)')
 #   plt.ylabel('ITC (baseline subtracted)')
 #   plt.xlim((-0.5, 1.5))
 #   plt.legend(condnames)