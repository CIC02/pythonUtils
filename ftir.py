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
    pos, opticSigArr = loadInterferograms2D(filename, harm)
    return pos[0,0,:,:], opticSigArr[0,0,:,:]

def interArrayToSpectra(posIn,opticSig,**kwargs):
    pos2D = np.reshape(posIn, (1,1,len(posIn),len(posIn[0])))
    opticSig2D = np.reshape(opticSig, (1,1,len(opticSig),len(opticSig[0])))
    wavenumber, spec = interArrayToSpectra2D(pos2D, opticSig2D, **kwargs)
    return wavenumber, spec[0,0]


def loadSpectraFromInter(filename, harm, **kwargs):
    pos, opticSig = loadInterferograms(filename, harm)
    return interArrayToSpectra(pos,opticSig, **kwargs)




def interArrayToSpectra2D(posIn,opticSig,discardPhase = True, discardDC = True, windowF = None, paddingFactor = 1, shiftMaxToZero = False):
    pos = posIn[0,0,0,:]
    step = pos[1]-pos[0]
    av = np.mean(opticSig,2)
    nbRow = len(av)
    nbCol = len(av[0])
    if discardPhase:
        av = np.transpose(np.transpose(av) * np.transpose(np.exp(-j*(np.angle(np.mean(av,2))))))
    if discardDC:
        av = np.transpose( np.transpose(av) - np.transpose(np.mean(av,2)))
    if windowF != None:
        window = windowF(len(av[0,0]))
    else:
        window = 1
    apodized = av*window
    apodized = np.concatenate((apodized,np.zeros((nbRow, nbCol, len(apodized)*(paddingFactor-1)))),2)
    if shiftMaxToZero:
        apodized = np.roll(apodized, -np.argmax(np.abs(apodized)),2)
    if discardPhase:
        wavenumber = np.linspace(0,0.25e-2/step,(int(len(apodized[0,0])/2))+1)
        spectrum = np.fft.rfft(np.real(apodized))
    else:
        wavenumber = np.linspace(0,0.5e-2/step,len(apodized[0,0]))
        spectrum = np.fft.fft(apodized)
    return wavenumber, spectrum

def loadInterferograms2D(filename, harm):
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
    nbRun = int(data['Run'][-1]) + 1
    nbRow = int(data['Row'][-1]) + 1
    nbCol = int(data['Column'][-1]) + 1
    runLength = int(len(data['Row'])/(nbRun*nbRow*nbCol))
    depth = np.asarray(data['Depth'],dtype=float)
    phase = np.asarray(data['O'+str(harm)+'P'],dtype=float)
    amp = np.asarray(data['O'+str(harm)+'A'],dtype=float)
    opticSig = amp * np.exp(1j*phase)
    
    depth = depth.reshape([nbRow, nbCol, nbRun, runLength])
    opticSig = opticSig.reshape([nbRow, nbCol, nbRun, runLength])
    
    pos = depth * distance/len(depth[0,0,0]) + center-distance/2
    
    return pos, opticSig

def loadSpectraFromInter2D(filename, harm, **kwargs):
    pos, opticSig = loadInterferograms2D(filename, harm)
    return interArrayToSpectra2D(pos,opticSig, **kwargs)

