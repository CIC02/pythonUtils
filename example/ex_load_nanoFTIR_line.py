#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 21 15:09:54 2025

@author: edoardolab
"""
import sys
sys.path.append("..") #Path to the ftir library directory. Replace ".." with the correct path, or removve this line and permanently add the directory to your pythonpath
import ftir


import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

# %% Data Load and Processing

# Load data from .txt
interData = ftir.loadFullInterferogramData("./exampleFile_nanoFTIR_LineInterferograms.txt", saveInterf=False,reload=False)
    # interData: interferogram data [Row, Column, Run, Time]

# correct interferograms with balance detection
interData_BD=ftir.BalanceDetectionCorrection(interData)



# get spectra by FFT
windowF = lambda M: signal.windows.tukey(M, 0.3)  # apodisation funtion
spectra = ftir.interferogramsToSpectra(interData, windowF=windowF, paddingFactor=4, shiftMaxToZero=True)
spectra_BD = ftir.interferogramsToSpectra(interData_BD, windowF=windowF, paddingFactor=4, shiftMaxToZero=True)
    # spectra: spectral data [Row, Delay, Run, Frequency]



# %% Plot data

harmonic=2 #harmonic selector


RefMirrorPosition=interData["M"][0,0,0]*1e6  #Mirror Position in [Row=0, Delay=0, Run=0, Time=all]
O3_interf=interData[f"O{harmonic}"][0,0,0,:] #Channel O at harmonic [Row=0, Delay=0, Run=0, Time=all]

if f"A{harmonic}" in interData:
    A3_interf=interData[f"A{harmonic}"][0,0,0,:] #Channel A at harmonic  Row=0, Delay=0, Run=0, Time=all]
else:
    A3_interf=0*O3_interf #if no channel A avaiable, it assume it as 0  

#Create the window function if necessary

W=windowF(interData["M"].shape[3])

# plot interferograms
plt.figure()
plt.plot(RefMirrorPosition, np.abs(O3_interf))
plt.plot(RefMirrorPosition, np.abs(A3_interf))
plt.plot(RefMirrorPosition, W)
plt.xlabel("Mirror position [um]")
plt.ylabel("Amplitude")
plt.legend(['O3','A3','ApodisationWindow'])



# calulate SNR
SNR=np.round(np.mean(ftir.SNR(spectra)[f"O{harmonic}"]))
SNR_BD=np.round(np.mean(ftir.SNR(spectra_BD)[f"O{harmonic}"]))
print(f"original SNR = {SNR} \n with BD SNR = {SNR_BD}")
print(f"improvement  = {np.round(SNR_BD/SNR,1)}")


# Plot Spectra
wn=spectra["Wavenumber"][0,0,0,:] #Wavenumber axis  in [Row=0, Delay=0, Run=0, frequency=all]
O3_specta=spectra[f"O{harmonic}"][0,0,:,:] #Channel O at harmonic [Row=0, Delay=0, Run=0, frequency=all]
O3_specta_BD=spectra_BD[f"O{harmonic}"][0,0,:,:] #Channel O at harmonic [Row=0, Delay=0, Run=0, frequency=all]

O3_spectra_AVG=np.mean(O3_specta,0)
O3_spectra_BD_AVG=np.mean(O3_specta_BD,0)

plt.figure()
plt.subplot(2,1,1)
plt.plot(wn, np.abs(O3_spectra_AVG))
plt.plot(wn, np.abs(O3_spectra_BD_AVG))
plt.ylabel("Amplitude")
plt.xlim(0,2000)
plt.legend(['O3','O3 with BD correction'])

plt.subplot(2,1,2)
plt.plot(wn, np.angle(O3_spectra_AVG))
plt.plot(wn, np.angle(O3_spectra_BD_AVG))
plt.xlabel("Wavenumber (cm-1)")
plt.ylabel("Phase")
plt.xlim(0,2000)
plt.legend(['O3','O3 with BD correction'])



# Plot Hyperspectral  countourplots
O3_specta_BD_line=spectra_BD[f"O{harmonic}"][0,:,0,:] #Channel O at harmonic [Row=0, Delay=all, Run=0, frequency=all]
x=np.linspace(0,5, len(O3_specta_BD_line))


O3_specta_BD_line_norm=O3_specta_BD_line/np.mean(O3_specta_BD_line[:20,:],0,keepdims=True)

plt.figure()

# Amplitude countour plot
plt.subplot(1,2,1)
plt.imshow( np.abs(O3_specta_BD_line_norm).T,extent=[x[0],x[-1],wn[0],wn[-1]],aspect="auto",origin='lower')
plt.ylabel("Wavenumber (cm-1)")
plt.xlabel("x (um)")
plt.ylim(500,1300)
plt.title('O3 amplitude')
plt.clim(0.5,1.5)
plt.colorbar()

# Phase countour plot
plt.subplot(1,2,2)
plt.imshow( np.angle(O3_specta_BD_line_norm).T,extent=[x[0],x[-1],wn[0],wn[-1]],aspect="auto",origin='lower')
plt.xlabel("x (um)")
plt.ylim(500,1300)
plt.title('O3 phase')
plt.clim(-0.5,0.5)
plt.colorbar()

plt.show()


