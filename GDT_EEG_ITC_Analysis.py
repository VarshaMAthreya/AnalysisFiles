# -*- coding: utf-8 -*-
"""
Created on Mon Jan 24 16:29:40 2022

@author: vmysorea
"""
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
import fnmatch
from mne.time_frequency import tfr_multitaper
bs.importbdf
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')
plt.rcParams['figure.figsize'] = [12, 8]
plt.rcParams['figure.dpi']  = 120

froot = 'D:/PhD/Data/MTB_EP - GDT, Binding, mTRF/GDT/'
subjlist = ['S105'] #, 'S078', 'S088', 'S308', 'S291'     #Load subject file
condlist = [1,2,3]
condnames = ['0.016 ms', '0.032 ms','0.064 ms']

for subj in subjlist:
    evokeds = []
    #itcs = []
    #powers = []
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
   # raw.info ['bads'].extend (['A1', 'A2', 'A11', 'A7', 'A25', 'A28', 'A20', 'A21', 'A6', 'A30', 'A6'])

#Filtering
raw.filter (1., 40.)
#raw.info

#Blink Rejection

from anlffr.preproc import find_blinks
blinks = find_blinks(raw)
raw.plot(events=blinks, duration=25.0, n_channels=32, scalings=dict(eeg=200e-6))

from mne import compute_proj_epochs
epochs_blinks = mne.Epochs(raw, blinks, event_id=998, baseline=(-0.25, 0.25),
                           reject=dict(eeg=500e-6), tmin=-0.25, tmax=0.25)

blink_proj = compute_proj_epochs(epochs_blinks, n_eeg=1)

raw.plot_projs_topomap()       #Visualizing the spatial filter
raw.add_proj(blink_proj)

#Onset responses
for c, cond in enumerate(condlist):

        condname = condnames[c]
        epochs = mne.Epochs(raw, eves, cond, tmin=-0.3, proj=True,
            tmax=1.1, baseline=(-0.3, 0.0),
            reject=dict(eeg=200e-6))
        evoked = epochs.average()
        evokeds += [evoked, ]
#picks=['A31']
#evoked.plot(picks=picks,titles='Onset Response - S105')        

epochs = mne.Epochs(raw, eves, event_id=[1,2,3], baseline=(-0.3, 0), proj=True,
                    tmin=-0.3, tmax=1.1, reject=dict(eeg=200e-6))
evoked = epochs.average() 
picks=['A31']
evoked.plot(picks=picks,titles='Onset Response - S105')

#Saving onset response to mat file
t=epochs.times
fs = raw.info['sfreq']
a=evoked.data
#a_mean=a.mean
mat_ids = dict(OnsetResponse = a, t = t, n_channels = picks, fs=fs)
#save_loc = ('C:/Users/vmysorea/OneDrive - purdue.edu/Desktop/PhD/GDT_Analysis/EP_GDT/ITC Measures/Above_35/')
#savemat(save_loc + 'Onset Response_S105.mat', mat_ids)
#mne.write_evokeds(save_loc + 'Onset Response_S105', evoked)

##Creating manual events to add the responses of each gap size 

fs = raw.info['sfreq']
gap_durs = [.016, .032, 0.064]
eves_manual = np.zeros((3*eves.shape[0],3))
for k in range(1,eves.shape[0]):
    for m in range(3):
            current_eves = eves[k,:].copy()
            gap_samps = int(np.round(gap_durs[int(current_eves[2])-1] * fs))
            current_eves[0] = current_eves[0] + (m + 1) * int(np.round(0.5*fs)) + m*gap_samps
            current_eves = eves_manual[((k-1)*2) + m, :] 

eves_manual = np.int64(eves_manual)

#Epoching

epochs_1 = mne.Epochs(raw, eves_manual, event_id=[1], baseline=(-0.3, 0), proj=True,
                    tmin=-0.3, tmax=1.1, reject=dict(eeg=200e-6))
epochs_2 = mne.Epochs(raw, eves_manual, event_id=[2], baseline=(-0.3, 0), proj=True,
                    tmin=-0.3, tmax=1.1, reject=dict(eeg=200e-6))
epochs_3 = mne.Epochs(raw, eves_manual, event_id=[3], baseline=(-0.3, 0), proj=True,
                    tmin=-0.3, tmax=1.1, reject=dict(eeg=200e-6))

#Averaging

evoked_1 = epochs_1.average() 
evoked_2 = epochs_2.average() 
evoked_3 = epochs_3.average() 

#Plotting averaged waveforms
picks=['A31', 'A32']
evokeds_1 = dict(GDT_16ms=evoked_1, GDT_32ms=evoked_2, GDT_64ms=evoked_3)
mne.viz.plot_compare_evokeds(evokeds_1, combine='mean',title='GDT - S105')
#mne.write_evokeds (save_loc + 'Evoked_1_S105', evoked_1)
#mne.write_evokeds (save_loc + 'Evoked_2_S105', evoked_2)
#mne.write_evokeds (save_loc + 'Evoked_3_S105', evoked_3)

# Compute evoked response using ITC

### ITC1 - 16 ms

freqs = np.arange(1., 14., 1.)
n_cycles = freqs * 0.2
t= epochs_1.times
#epochs_induced = epochs.copy().subtract_evoked()
picks = [31]
power_1, itc_1 = tfr_multitaper(epochs_1, freqs, n_cycles, picks=picks,
                            time_bandwidth=4.0, n_jobs=-1, return_itc=True)
#itc_1.apply_baseline(baseline=(-0.5, 0))
power_1.apply_baseline(baseline=(-0.5, 0), mode='logratio')

#Saving ITC measures into mat file -- Taking the mean across the third row 
x=itc_1.data
mat_ids1 = dict(itc1=x,freqs=freqs,n_channels=picks, n_cycles=n_cycles, t=t)
save_loc = ('C:/Users/vmysorea/OneDrive - purdue.edu/Desktop/PhD/GDT_Analysis/EP_GDT/ITC Measures/Above_35/')
savemat(save_loc + 'ITC1_S105.mat', mat_ids1)
#savemat("C:/Users/vmysorea/OneDrive - purdue.edu/Desktop/PhD/GDT_Analysis/EP_GDT/ITC Measures/Above_35/ITC1_S105.mat", mat_ids1)

### ITC2 - 32 ms
freqs = np.arange(1., 14., 1.)
n_cycles = freqs * 0.2
T=n_cycles/freqs.size
#epochs_induced = epochs.copy().subtract_evoked()
picks = [31]
power_2, itc_2 = tfr_multitaper(epochs_2, freqs, n_cycles, picks=picks,
                            time_bandwidth=4.0, n_jobs=-1, return_itc=True)
#itc_2.apply_baseline(baseline=(-0.5, 0))
power_2.apply_baseline(baseline=(-0.5, 0), mode='logratio')

#Saving ITC measures into mat file -- Taking the mean across the third row 
y=itc_2.data
mat_ids2 = dict(itc2=y,freqs=freqs,n_cycles=n_cycles, t=t)
save_loc = ('C:/Users/vmysorea/OneDrive - purdue.edu/Desktop/PhD/GDT_Analysis/EP_GDT/ITC Measures/Above_35/')
savemat(save_loc + 'ITC2_S105.mat', mat_ids2)
#savemat("C:/Users/vmysorea/OneDrive - purdue.edu/Desktop/PhD/GDT_Analysis/EP_GDT/ITC Measures/Above_35/ITC2_S105.mat", mat_ids2)

### ITC3 - 64 ms
freqs = np.arange(1., 14., 1.)
n_cycles = freqs * 0.2
t= epochs_3.times
#epochs_induced = epochs.copy().subtract_evoked()
picks = [31]
power_3, itc_3 = tfr_multitaper(epochs_3, freqs, n_cycles, picks=picks,
                            time_bandwidth=4.0, n_jobs=-1, return_itc=True)
#itc_3.apply_baseline(baseline=(-0.5, 0))
power_3.apply_baseline(baseline=(-0.5, 0), mode='logratio')

#Saving ITC measures into mat file -- Taking the mean across the third row 
z=itc_3.data
mat_ids3 = dict(itc3=z,freqs=freqs, n_channels=picks, n_cycles=n_cycles, t=t)
save_loc = ('C:/Users/vmysorea/OneDrive - purdue.edu/Desktop/PhD/GDT_Analysis/EP_GDT/ITC Measures/Above_35/')
savemat(save_loc + 'ITC3_S105.mat', mat_ids3)
#savemat("C:/Users/vmysorea/OneDrive - purdue.edu/Desktop/PhD/GDT_Analysis/EP_GDT/ITC Measures/Above_35/ITC3_S105.mat", mat_ids3)

#itcs += [itc_1, itc_2, itc_3]
#powers += [power_1, power_2, power_3]

#Plotting ITC and Power plots for the three conditions
power_1.plot([0], baseline=(-0.5, 0), mode='mean', title='Gap duration of 16 ms - Power')
itc_1.plot([0], title='Gap duration of 16 ms - Intertrial Coherence (S105)')
plt.rcParams["figure.figsize"] = (5.5,5)
#plt.tight_layout()
#plt.savefig('C:/Users/vmysorea/OneDrive - purdue.edu/Desktop/PhD/ARO_Figures/itc1_S105.png', dpi=300)

#save_loc = ('C:/Users/vmysorea/OneDrive - purdue.edu/Desktop/PhD/GDT_Analysis/EP_GDT/ITC Measures/Above_35/')
#plt.savefig(save_loc + 'ITC1_S105.png')
#plt.savefig("C:/Users/vmysorea/OneDrive - purdue.edu/Desktop/PhD/GDT_Analysis/EP_GDT/ITC Measures/S105/ITC1_S105.png")

power_2.plot([0], baseline=(-0.5, 0), mode='mean', title='Gap duration of 32 ms - Power')
itc_2.plot([0], title='Gap duration of 32 ms- Intertrial Coherence (S105)')
#plt.savefig('C:/Users/vmysorea/OneDrive - purdue.edu/Desktop/PhD/ARO_Figures/itc2_S105.png', dpi=300)

power_3.plot([0], baseline=(-0.5, 0), mode='mean', title='Gap duration of 64 ms - Power')
itc_3.plot([0], title='Gap duration of 64 ms - Intertrial Coherence (S105)')
#plt.savefig('C:/Users/vmysorea/OneDrive - purdue.edu/Desktop/PhD/ARO_Figures/itc3_S105.png', dpi=300)

###Saving egg files:
#save_indexes = [0,1,10,11,12,13,14,15]
#epochs_save = []
#evoked_save = [] #save 32 channel evkd response
#evoked_1_save = []
#evoked_2_save = []
#evoked_3_save = []    
#t_full = epochs[-1].times
    
#for si in save_indexes:
#    evoked_save.append(evoked[si])
#    epochs_save.append(epochs[si].get_data()[:,31,:]) # Only save epochs for channel 31 
    
    
#with open(os.path.join(pickle_loc,subject+'_Binding.pickle'),'wb') as file:
#        pickle.dump([t, t_full, conds_save, epochs_save,evkd_save],file)