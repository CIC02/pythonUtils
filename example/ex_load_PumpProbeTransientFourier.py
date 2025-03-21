#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 18 09:59:43 2025

@author: edoardolab

Example code to load, process and plot: Trasnfient Fourier Maps acquired with pump-probe nanoFTIR


"""



import ftir
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
# %% Data Load and Processing

# Load data from .txt
interData = ftir.loadFullInterferogramData("example/exampleFile_PumpProbe_TransientFourier.txt", saveInterf=False,reload=False)
    # interData: interferogram data [Row, Delay, Run, Time]

# correct interferograms with balance detection
interData_BD=ftir.BalanceDetectionCorrection(interData)


# get spectra by FFT
windowF = lambda M: signal.windows.tukey(M, 0.3)   # apodisation funtion
spectra = ftir.interferogramsToSpectra(interData, windowF=windowF, paddingFactor=4, shiftMaxToZero=True)
spectra_BD = ftir.interferogramsToSpectra(interData_BD, windowF=windowF, paddingFactor=4, shiftMaxToZero=True)
    # spectra: spectral data [Row, Delay, Run, Frequency]

# %% Plot data

harmonic=3 #harmonic selector



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



# Plot Spectra
wn=spectra["Wavenumber"][0,0,0] #Wavenumber axis  in [Row=0, Delay=0, Run=0, frequency=all]
O3_specta=spectra[f"O{harmonic}"][0,0,0] #Channel O at harmonic [Row=0, Delay=0, Run=0, frequency=all]
O3_specta_BD=spectra_BD[f"O{harmonic}"][0,0,0] #Channel O at harmonic [Row=0, Delay=0, Run=0, frequency=all]

plt.figure()
plt.subplot(2,1,1)
plt.plot(wn, np.abs(O3_specta))
plt.plot(wn, np.abs(O3_specta_BD))
plt.ylabel("Amplitude")
plt.xlim(0,2000)
plt.legend(['O3','O3 with BD correction'])

plt.subplot(2,1,2)
plt.plot(wn, np.angle(O3_specta))
plt.plot(wn, np.angle(O3_specta_BD))
plt.xlabel("Wavenumber (cm-1)")
plt.ylabel("Phase")
plt.xlim(0,2000)
plt.legend(['O3','O3 with BD correction'])


# Plot Transient Fourier countourplots
O3_specta_BD_TF=spectra_BD[f"O{harmonic}"][0,:,0,:] #Channel O at harmonic [Row=0, Delay=all, Run=0, frequency=all]
tau=spectra_BD["Delay"][0,:,0,0] #Delay in [Row=0, Delay=0, Run=all, frequency=0]


plt.figure()

# Amplitude countour plot
plt.subplot(2,1,1)
plt.imshow( np.abs(O3_specta_BD_TF),extent=[wn[0],wn[-1],tau[0],tau[-1]],aspect="auto")
plt.ylabel("tau")
plt.xlim(500,1300)
plt.title('O3 amplitude')
plt.clim(0,100)
plt.colorbar()

# Phase countour plot
plt.subplot(2,1,2)
plt.imshow( np.angle(O3_specta_BD_TF),extent=[wn[0],wn[-1],tau[0],tau[-1]],aspect="auto")
plt.xlabel("Wavenumber (cm-1)")
plt.ylabel("tau")
plt.xlim(500,1300)
plt.title('O3 phase')
plt.colorbar()

plt.show()
