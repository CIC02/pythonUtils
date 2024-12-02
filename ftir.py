# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 11:03:18 2024

@author: tnhannotte

Set of tools to read from Neaspec nano-FTIR data

"""

import re
import numpy as np
import pandas as pd
import miscUtil
import h5py

j = 1j
c = 299792458


def loadSpectra(filename, harm):
    """
    Load a spectrum txt file from a FTIR measurement at a single position
    Warning: The data processing parameters used in the neaspec software to convert the interferograms into spectra are not accessible.
    Use the interferogram data for more control over the process.

    Parameters
    ----------
    filename : str
        Path of the txt file
    harm : int
        Demodulation harmonic to extract

    Returns
    -------
    2 1D numpy arrays:
        wavenumbers, complex valued spectrum

    """
    wavenumber, field = loadSpectra2D(filename, harm)
    return wavenumber[0,0,:], field[0,0,:]

def loadSpectra2D(filename, harm):
    """
    Load a 2 dimensional array of spectra from a spectra.txt file
    Warning: The data processing parameters used in the neaspec software to convert the interferograms into spectra are not accessible.
    Use the interferogram data for more control over the process.
    

    Parameters
    ----------
    filename : str
        Path of the txt file
    harm : int
        Demodulation harmonic to extract

    Returns
    -------
    2 3D numpy arrays:
        wavenumbers, complex valued spectra

    """
    data = pd.read_csv(filename, comment = "#", delimiter='\t')
    data = data.to_dict(orient="list")
    nbRow = int(data['Row'][-1]) + 1
    nbCol = int(data['Column'][-1]) + 1
    nbWN = int(data['Omega'][-1]) + 1
    wavenumber = np.asarray(data['Wavenumber'],dtype=float)
    phase = np.asarray(data['O'+str(harm)+'P'],dtype=float)
    amp = np.asarray(data['O'+str(harm)+'A'],dtype=float)
    
    field = amp * np.exp(1j*phase)
    
    wavenumber = np.reshape(wavenumber, [nbRow, nbCol, nbWN])
    field = np.reshape(field, [nbRow, nbCol, nbWN])
    return wavenumber, field

def loadInterferograms(filename, harm):
    """
    Load interferograms from a FTIR measurement taken on a single position

    Parameters
    ----------
    filename : str
        Path of the file
    harm : int
        Demodulation harmonic to extract

    Returns
    -------
    2D numpy array
        Mirror position
    2D numpy array
        Optical signal

    """
    pos, opticSigArr = loadInterferograms2D(filename, harm)
    return pos[0,0,:,:], opticSigArr[0,0,:,:]

def interArrayToSpectra(posIn,opticSig,**kwargs):
    """
    Convert an array of interferograms in an averaged spectra (for single position measurements)

    Parameters
    ----------
    posIn : 2D numpy array
        Mirror positions, as returned from loadInterferograms()
    opticSig : 2D numpy array
        Interferograms, as returned from loadInterferograms()
    discardPhase : bool, optional
        If True, substract the average phase of the interferogram data, and perform a rfft, if False, perform a regular fft on the complex data. The default is True.
    discardDC : bool, optional
        if True, remove the DC component. The default is True.
    windowF : function(int), optional
        Windowing function, taking a integer window size as parameter. The default is None.
    paddingFactor : int, optional
        The data is zero padded to its original size multiplied by paddingFactor. The default is 1.
    shiftMaxToZero : bool, optional
        If True, the maximum value of the interferogram is shifted to position 0, to (mostly) cancel the phase slope. The default is False.

    Returns
    -------
    wavenumber : 1D numpy array
        
    spectrum : 1D numpy array
        Averaged spectrum

    """
    pos2D = np.reshape(posIn, (1,1,len(posIn),len(posIn[0])))
    opticSig2D = np.reshape(opticSig, (1,1,len(opticSig),len(opticSig[0])))
    wavenumber, spec = interArrayToSpectra2D(pos2D, opticSig2D, **kwargs)
    return wavenumber, spec[0,0]


def loadSpectraFromInter(filename, harm, **kwargs):
    """
    Load interferograms from a FTIR measurement taken on a single position, and convert them in an averaged spectrum

    Parameters
    ----------
    filename : str
        Path of the file
    harm : int
        Demodulation harmonic to extract
    discardPhase : bool, optional
        If True, substract the average phase of the interferogram data, and perform a rfft, if False, perform a regular fft on the complex data. The default is True.
    discardDC : bool, optional
        if True, remove the DC component. The default is True.
    windowF : function(int), optional
        Windowing function, taking a integer window size as parameter. The default is None.
    paddingFactor : int, optional
        The data is zero padded to its original size multiplied by paddingFactor. The default is 1.
    shiftMaxToZero : bool, optional
        If True, the maximum value of the interferogram is shifted to position 0, to (mostly) cancel the phase slope. The default is False.

    Returns
    -------
    wavenumber : 1D numpy array
        
    spectrum : 1D numpy array
        Averaged spectrum for each position on the 2D surface
    """
    pos, opticSig = loadInterferograms(filename, harm)
    return interArrayToSpectra(pos,opticSig, **kwargs)




def interArrayToSpectra2D(posIn,opticSig,discardPhase = True, discardDC = True, windowF = None, paddingFactor = 1, shiftMaxToZero = False):
    """
    Convert an array of interferograms in an array of averaged spectra

    Parameters
    ----------
    posIn : 4D numpy array
        Mirror positions, as returned from loadInterferograms2D()
    opticSig : 4D numpy array
        Interferograms, as returned from loadInterferograms2D()
    discardPhase : bool, optional
        If True, substract the average phase of the interferogram data, and perform a rfft, if False, perform a regular fft on the complex data. The default is True.
    discardDC : bool, optional
        if True, remove the DC component. The default is True.
    windowF : function(int), optional
        Windowing function, taking a integer window size as parameter. The default is None.
    paddingFactor : int, optional
        The data is zero padded to its original size multiplied by paddingFactor. The default is 1.
    shiftMaxToZero : bool, optional
        If True, the maximum value of the interferogram is shifted to position 0, to (mostly) cancel the phase slope. The default is False.

    Returns
    -------
    wavenumber : 1D numpy array
        
    spectrum : 3D numpy array
        Averaged spectrum for each position on the 2D surface

    """
    pos = np.mean(posIn,axis=(0,1,2))
    #step = (pos[-1]-pos[0])/(len(pos)-1)
    step = (pos[-1]-pos[int(len(pos)/2)])/(len(pos) - int(len(pos)/2)-1) #Average the step over the second half of data (first few points are not always equally spaced)
    av = np.mean(opticSig,2)
    nbRow = len(av)
    nbCol = len(av[0])
    if discardPhase:
        #av = np.transpose(np.transpose(av) * np.transpose(np.exp(-j*(np.angle(np.mean(av,2))))))
        for row in av:
            for spec in row:
                spec = spec * np.exp(-j*miscUtil.dataAngle(spec))
    if discardDC:
        av = np.transpose( np.transpose(av) - np.transpose(np.mean(av,2)))
    if windowF != None:
        window = windowF(len(av[0,0]))
    else:
        window = 1
    apodized = av*window
    apodized = np.concatenate((apodized,np.zeros((nbRow, nbCol, len(apodized[0,0])*(paddingFactor-1)))),2)
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
    """
    Load interferograms from a FTIR measurement taken on a 2D surface

    Parameters
    ----------
    filename : str
        Path of the file
    harm : int
        Demodulation harmonic to extract

    Returns
    -------
    4D numpy array
        Mirror position (dimensions correspond to: Row, column, run, mirror position)
    4D numpy array
        Optical signal

    """
    # regex = re.compile(r".*Interferometer.*\t(?P<center>(\d+\.\d+))\t(?P<distance>(\d+\.\d+)).*")
    # distance = 0
    # center = 0
        # f.seek(0)
        # line = f.readline()
        # while line != '':
        #     parts = re.match(regex,line)
        #     if parts != None:
        #         distance = float(parts.groupdict()["distance"])*1e-6
        #         center = float(parts.groupdict()["center"])*1e-6
        #         break
        #     line = f.readline()
    data = pd.read_csv(filename, comment = "#", delimiter='\t')
    data = data.to_dict(orient="list")
    data = {x.replace(' ', ''): v for x, v in data.items()}
    nbRun = int(data['Run'][-1]) + 1
    nbRow = int(data['Row'][-1]) + 1
    nbCol = int(data['Column'][-1]) + 1
    runLength = int(len(data['Row'])/(nbRun*nbRow*nbCol))
    depth = np.asarray(data['Depth'],dtype=float)
    pos = np.asarray(data['M'],dtype=float)
    phase = np.asarray(data['O'+str(harm)+'P'],dtype=float)
    amp = np.asarray(data['O'+str(harm)+'A'],dtype=float)
    opticSig = amp * np.exp(1j*phase)
    
    depth = depth.reshape([nbRow, nbCol, nbRun, runLength])
    opticSig = opticSig.reshape([nbRow, nbCol, nbRun, runLength])
    pos = pos.reshape([nbRow, nbCol, nbRun, runLength])
    #pos = depth * distance/len(depth[0,0,0]) + center-distance/2   
    return pos, opticSig



def loadSpectraFromInter2D(filename, harm, **kwargs):
    """
    Load interferograms from a FTIR measurement taken on a 2D surface, and convert them in averaged spectra

    Parameters
    ----------
    filename : str
        Path of the file
    harm : int
        Demodulation harmonic to extract
    discardPhase : bool, optional
        If True, substract the average phase of the interferogram data, and perform a rfft, if False, perform a regular fft on the complex data. The default is True.
    discardDC : bool, optional
        if True, remove the DC component. The default is True.
    windowF : function(int), optional
        Windowing function, taking a integer window size as parameter. The default is None.
    paddingFactor : int, optional
        The data is zero padded to its original size multiplied by paddingFactor. The default is 1.
    shiftMaxToZero : bool, optional
        If True, the maximum value of the interferogram is shifted to position 0, to (mostly) cancel the phase slope. The default is False.

    Returns
    -------
    wavenumber : 1D numpy array
        
    spectrum : 3D numpy array
        Averaged spectrum for each position on the 2D surface
    """
    pos, opticSig = loadInterferograms2D(filename, harm)
    return interArrayToSpectra2D(pos,opticSig, **kwargs)

def extractScanArea(filename):
    """
    Extract the width and height in meter of the scanning area from a interferogram file (assumes the values are stored in µm in the file)
    
    Parameters
    ----------
    filename : str
        path of the input file

    Returns
    -------
    width : float
        width in meter
    height : float
        height in meter
    """
    regex = re.compile(r".*Scan Area.*\t(?P<X>(\d+\.\d+))\t(?P<Y>(\d+\.\d+))\t\d+\.\d+.*")
    with open(filename,newline='') as f:
        line = f.readline()
        while line != '':
            parts = re.match(regex,line)
            if parts != None:
                width = float(parts.groupdict()["X"])*1e-6
                height = float(parts.groupdict()["Y"])*1e-6
                break
            line = f.readline()
    return width, height


def extractWavenumberScaling(filename):
    """
    Extract the wavenumber scaling from the header of a interferogram or spectra file
    
    Parameters
    ----------
    filename : str
        path of the input file

    Returns
    -------
    scaling: float
        wavenumber scaling
    """
    regex = re.compile(r".*Wavenumber Scaling.*\t(?P<X>(\d+\.\d+)).*")
    with open(filename,newline='') as f:
        line = f.readline()
        while line != '':
            parts = re.match(regex,line)
            if parts != None:
                scaling = float(parts.groupdict()["X"])
                break
            line = f.readline()
    return scaling

def extractInterCenterDistance(filename):
    """
    Extract the interferometer center and distance from the header of a interferogram file
    
    Parameters
    ----------
    filename : str
        path of the input file

    Returns
    -------
    center: float
        Interferometer center
    distance: float
        Interferometer distance
    """
    regex = re.compile(r".*Interferometer.*\t(?P<center>(\d+\.\d+))\t(?P<distance>(\d+\.\d+)).*")
    distance = 0
    center = 0
    with open(filename,newline='') as f:
        f.seek(0)
        line = f.readline()
        while line != '':
            parts = re.match(regex,line)
            if parts != None:
                distance = float(parts.groupdict()["distance"])*1e-6
                center = float(parts.groupdict()["center"])*1e-6
                break
            line = f.readline()
    return center, distance



def exportForCorrect(pos, inter, filename):
    """
    Export a list of interferogram in a h5 file, with time axis in picosecond.
    The output file can be open in the software Correct@TDS.

    Parameters
    ----------
    pos : 1D or 2D numpy array
        Mirror position, saves an averaged position if pos is 2D
    inter : 2D numpy array of float
        list of interferograms
    filename : str
        path of the output file

    Returns
    -------
    None.

    """
    if np.ndim(pos) == 2:
        avPos = np.mean(pos,0)
    else:
        avPos = pos
    with h5py.File(filename,"w") as hdf:
        hdf.create_dataset('timeaxis', data = 2*avPos/c*1e12)    
        for i in range(len(inter)):
            hdf.create_dataset(str(i), data = inter[i])

def importFromCorrect(filename):
    """
    Import a h5 file, as generated by Correct@TDS (time axis in ps)

    Parameters
    ----------
    filename : str
        Path of the input file

    Returns
    -------
    pos: 1D numpy array
        Mirror positions in m
    inter: 2D numpy array
        interferograms

    """
    with h5py.File(filename,"r") as f:
        t = np.array(f["timeaxis"])        
        inter = np.asarray([np.array(f[str(trace)]) for trace in range(len(f)-1)], dtype=np.complex128)
    return t*1e-12*c/2, inter
