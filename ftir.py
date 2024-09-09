# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 11:03:18 2024

@author: tnhannotte

Set of tools to read from Neaspec nano-FTIR data

"""

import csv
import re
import numpy as np
import pandas as pd



j = 1j

def loadSpectra(filename, harm):
    with open(filename,newline='') as f:
        rdr = csv.DictReader(filter(lambda row: row[0]!='#', f),delimiter='\t')
        data = []
        for row in rdr:
            data.append(row)
    data = pd.DataFrame(data).to_dict(orient="list")
    wavenumber = np.asarray(data['Wavenumber'],dtype=float)
    phase = np.asarray(data['O'+str(harm)+'P'],dtype=float)
    amp = np.asarray(data['O'+str(harm)+'A'],dtype=float)
    return wavenumber, amp * np.exp(1j*phase)

def loadInterferograms(filename, harm):
    regex = re.compile(r".*Interferometer.*\t(?P<center>(\d+\.\d+))\t(?P<distance>(\d+\.\d+)).*")
    distance = 0
    center = 0
    with open(filename,newline='') as f:
        rdr = csv.DictReader(filter(lambda row: row[0]!='#', f),delimiter='\t')
        data = []
        for row in rdr:
            data.append(row)
        f.seek(0)
        line = f.readline()
        while line != '':
            parts = re.match(regex,line)
            if parts != None:
                distance = float(parts.groupdict()["distance"])*1e-6
                center = float(parts.groupdict()["center"])*1e-6
                break
            line = f.readline()
    data = pd.DataFrame(data).to_dict(orient="list")
    data = {x.replace(' ', ''): v for x, v in data.items()}
    nbRun =int(data['Run'][-1]) + 1
    runLength = int(len(data['Row'])/nbRun)
    depth = np.asarray(data['Depth'],dtype=float)
    phase = np.asarray(data['O'+str(harm)+'P'],dtype=float)
    amp = np.asarray(data['O'+str(harm)+'A'],dtype=float)
    opticSig = amp * np.exp(1j*phase)
    
    depth = depth.reshape([nbRun,runLength])
    opticSig = opticSig.reshape([nbRun,runLength])
    
    pos = depth * distance/len(depth[0]) + center-distance/2
    
    return pos, opticSig

def interArrayToSpectra(posIn,opticSig,discardPhase = True, discardDC = True, windowF = None, paddingFactor = 1, shiftMaxToZero = False):
    pos = posIn[0,:]
    step = pos[1]-pos[0]
    av = np.mean(opticSig,0)
    if discardPhase:
        av *= np.exp(-j*(np.angle(np.mean(av))))
    if discardDC:
        av -= np.mean(av)
    if windowF != None:
        window = windowF(len(av))
    else:
        window = 1
    apodized = av*window
    apodized = np.concatenate((apodized,np.zeros(len(apodized)*(paddingFactor-1))))
    if shiftMaxToZero:
        apodized = np.roll(apodized, -np.argmax(np.abs(apodized)))
    if discardPhase:
        wavenumber = np.linspace(0,0.25e-2/step,(int(len(apodized)/2))+1)
        spectrum = np.fft.rfft(np.real(apodized))
    else:
        wavenumber = np.linspace(0,0.5e-2/step,len(apodized))
        spectrum = np.fft.fft(apodized)
    return wavenumber, spectrum


def loadSpectraFromInter(filename, harm, **kwargs):
    pos, opticSig = loadInterferograms(filename, harm)
    return interArrayToSpectra(pos,opticSig, **kwargs)