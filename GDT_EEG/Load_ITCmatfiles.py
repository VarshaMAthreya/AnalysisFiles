# -*- coding: utf-8 -*-
"""
Created on Mon Jan 24 19:11:20 2022

@author: vmysorea
"""
import sys
sys.path.append('C:/Users/vmysorea/mne-python/')
sys.path.append('C:/Users/vmysorea/ANLffr/')
from matplotlib import pyplot as plt
from scipy import io
import numpy as np


### Loading data to plot ITC of the subjects
data_loc = 'C:/Users/vmysorea/OneDrive - purdue.edu/Desktop/PhD/GDT_Analysis/EP_GDT/ITC Measures/Common/1-14Hz/'

Subjects = ['S072', 'S078', 'S088', 'S308', 'S291', 'S273', 'S268', 'S269', 'S274', 'S282', 'S285', 'S277', 'S279','S285', 'S303']                                        ## Above 35 years
#Subjects = [ 'S273', 'S268', 'S269', 'S274', 'S282', 'S285', 'S277', 'S279','S285']  ## Below 35 years

###itc1
itc1 = np.zeros((len(Subjects),5736))
for sub in range(len(Subjects)):

    subject = Subjects[sub]
    dat = io.loadmat(data_loc + 'ITC1_' + subject, squeeze_me=True)
    
    dat.keys()
    itc1=dat['itc1']
    freqs =dat['freqs']
    #n_channels=dat['n_channels']
    t = dat['t']
       
    ### Plotting spectrogram of ITC of the subjects 
    spectrogram = plt.pcolormesh(t, freqs, itc1.squeeze(), cmap='PRGn')
    plt.title('Gap of 16 ms')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    
plt.colorbar(spectrogram)
plt.rcParams["figure.figsize"] = (5.5,5)
plt.tight_layout()
plt.savefig('C:/Users/vmysorea/OneDrive - purdue.edu/Desktop/PhD/ARO_Figures/itc1_spectrogram.png', dpi=500)

###itc2
itc2 = np.zeros((len(Subjects),5736))
for sub in range(len(Subjects)):

    subject = Subjects[sub]
    dat = io.loadmat(data_loc + 'ITC2_' + subject, squeeze_me=True)
    
    dat.keys()
    itc2=dat['itc2']
    freqs =dat['freqs']
    #n_channels=dat['n_channels']
    t = dat['t']
       
    ### Plotting spectrogram of ITC of the subjects 
    spectrogram1 = plt.pcolormesh(t, freqs, itc2.squeeze(), cmap='PRGn')
    plt.title('Gap of 32 ms')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    
#plt.colorbar(spectrogram1)
plt.rcParams["figure.figsize"] = (5.5,5)
plt.tight_layout()
plt.savefig('C:/Users/vmysorea/OneDrive - purdue.edu/Desktop/PhD/ARO_Figures/itc2_spectrogram.png', dpi=500)

###itc3
itc3 = np.zeros((len(Subjects),5736))
for sub in range(len(Subjects)):

    subject = Subjects[sub]
    dat = io.loadmat(data_loc + 'ITC3_' + subject, squeeze_me=True)
    
    dat.keys()
    itc3=dat['itc3']
    freqs =dat['freqs']
    #n_channels=dat['n_channels']
    t = dat['t']
       
    ### Plotting spectrogram of ITC of the subjects 
    spectrogram2 = plt.pcolormesh(t, freqs, itc3.squeeze(), cmap='PRGn')
    plt.title('Gap of 64 ms')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    
#plt.colorbar(spectrogram2)
plt.rcParams["figure.figsize"] = (5.5,5)
plt.tight_layout()
plt.savefig('C:/Users/vmysorea/OneDrive - purdue.edu/Desktop/PhD/ARO_Figures/itc3_spectrogram.png', dpi=500)

###Combined Plot
fig, ax = plt.subplots(1,3, sharey=True, constrained_layout=True)
ax[0].pcolormesh(t, freqs, itc1.squeeze(), cmap='PRGn')
ax[0].set_title('ITC - Gap 16 ms', loc='center', fontsize=9)
ax[1].pcolormesh(t, freqs, itc2.squeeze(), cmap='PRGn')
ax[1].set_title('ITC - Gap 32 ms', loc='center',fontsize=9)
pcm=ax[2].pcolormesh(t, freqs, itc3.squeeze(), cmap='PRGn')
ax[2].set_title('ITC - Gap 64 ms', loc='center',fontsize=9)
plt.xlim([-0.1,1.1])
plt.ylim([0,15])
#ax[0].legend()
fig.colorbar(pcm,ax=ax[2], location = 'right', aspect=50)
#fig.text(0.0000001,0.5, 'Frequency (Hz)', va='center',rotation ='vertical',fontsize=8) #Setting Y axis label
#fig.text(0.5,0.00001, 'Time (in seconds)', va='center', fontsize=8)          #Setting X axis label
fig.suptitle('ITC for the gap durations', fontsize=12)
fig.supxlabel ('Time (s)')
fig.supylabel('Frequency (Hz)')
plt.rcParams["figure.figsize"] = (16, 4)
#plt.tight_layout()
plt.savefig('C:/Users/vmysorea/OneDrive - purdue.edu/Desktop/PhD/ARO_Figures/ITC123_Combined_Spectrogram_1-14Hz.png', dpi=300)

#dat = io.loadmat('C:/Users/vmysorea/OneDrive - purdue.edu/Desktop/PhD/GDT_Analysis/EP_GDT/ITC Measures/S268/ITC3_S268.mat', squeeze_me=True)
    

### Loading data to plot onset responses of the subjects
#Onset_Response_GDT = np.zeros((len(Subjects)))

for sub in range(len(Subjects)):

    subject = Subjects[sub]
    data = io.loadmat(data_loc + 'Onset Response_' + subject, squeeze_me=True)
    
    data.keys()
    OnsetResponse=data['OnsetResponse']
    n_channels=data['n_channels']
    t = data['t']
    fs = data['fs']
    print ('Subjects')

    
### Plotting averaged waveform of the onset response 
onset_mean = OnsetResponse[:,:].mean(axis=0)
onset_sem = OnsetResponse[:,:].std(axis=0)/np.sqrt(OnsetResponse[sub].shape[0])
plt.figure()
plt.plot(t,onset_mean.T)
plt.fill_between(t, onset_mean-onset_sem,onset_mean+onset_sem,alpha=0.5)

index = np.where(t>-0.2)
index1 = index[0]
index1 = index1[0]
index = np.where(t<1) 
index2 = index[0]
index2 = index2[-1]
print(index1, index2)

freq = np.where (freqs<30)
print(freq)
itc3_ave = itc3[:,:].mean(axis=0)
print(itc3_ave)
itc3_std = np.std(itc3)
plt.plot (t, itc3_ave)


### Subplots
