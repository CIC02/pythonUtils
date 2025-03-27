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
import os
import gwyfile


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
    av = np.mean(opticSig,2) + 0j
    nbRow = len(av)
    nbCol = len(av[0])
    if discardPhase:
        #av = np.transpose(np.transpose(av) * np.transpose(np.exp(-j*(np.angle(np.mean(av,2))))))
        for row in av:
            for spec in row:
                spec *= np.exp(-j*miscUtil.dataAngle(spec))
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
    data = pd.read_csv(filename, comment = "#", delimiter='\t', index_col=False)
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
    Export a list of interferogram in a h5 file, with time axis in picosecond
    The output file can be open in the software Correct@TDS (https://github.com/THzbiophotonics/Correct-TDS)

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




def loadFullInterferogramData(filename,saveInterf=True,reload=False):
    """
    Load all channels from a FTIR measurement in a dictionnary of 4D arrays
    Amplitude and phase channels are combined in complex channels.
    Output dictionary is saved as .h5


    Parameters
    ----------
    filename : str
        Path of the file. .txt
    resave: bool, optional
        If True read .txt file and resaved it to h5.
        

    Returns
    -------
    out: Dictionnary of 4D numpy array  [Row, Column/Delay, Run, Time]

    """
    
    out = {}
    
    h5_path = os.path.splitext(filename)[0]+ ".h5"
    if os.path.exists(h5_path) and not(reload):
        out=LoadDataH5(h5_path)
    else:  
        data = pd.read_csv(filename, comment = "#", delimiter='\t')
        data = data.to_dict(orient="list")
        data = {x.replace(' ', ''): v for x, v in data.items()} #Remove spaces in the keys
        nbRun = int(data['Run'][-1]) + 1
        nbRow = int(data['Row'][-1]) + 1
        runLength = int(data['Depth'][-1]) + 1
        if 'Column' in data:
            nbCol = int(data['Column'][-1]) + 1
        elif 'Delay' in data:
            nbCol = int(len(data['Delay'])/(nbRun*nbRow*runLength))
        else:
            raise Exception("Unknown file format")
       
        for key, array in data.items():
            if key == 'Run' or key == 'Row' or key == 'Column'  or key =='Depth' or key.startswith('Unnamed'):
                pass
            elif key[-1] == 'P' and key[:-1]+'A' in data:
                pass
            elif key[-1] == 'A' and key[:-1]+'P' in data:
                amp = np.reshape(array,[nbRow, nbCol, nbRun, runLength])
                phase = np.reshape(data[key[:-1]+'P'],[nbRow, nbCol, nbRun, runLength])
                out[key[:-1]] = amp*np.exp(1j*phase)
            else:
                out[key] = np.reshape(array,[nbRow, nbCol, nbRun, runLength])
        if saveInterf:
            SaveDataH5(h5_path,out)
            
    return out


def loadGWYdata(filename):
    """
    Read  gwy file relative to nanoFTIR line scan. 

    Parameters
    ----------
    filename : str
        filepath of the .gwy file.

    Returns
    -------
    out : Dictionnary of 2d array
        it contain the dictionary of all the channel recorded during a nanoFTIR line scan. 
        M{n}: Mechanical signal 
        O{n}: Optical signal 
        A{n}: Optical signal
        Z:Topography 
        M: ??

    """
    
    out = {}
    # load GWY data
    gwy_path = os.path.splitext(filename)[0]+ ".gwy"
    if os.path.exists(gwy_path):
        obj = gwyfile.load(gwy_path)
        channels = gwyfile.util.get_datafields(obj)
        for key in channels.keys():

            if key == "Z":
                wx=channels[key].xreal
                wy=channels[key].yreal
            if key[-1] == 'P' and key[:-1]+'A' in channels:
                pass
            elif key[-1] == 'A' and key[:-1]+'P' in channels:
                amp =channels[key].data
                phase = channels[key[:-1]+'P'].data
                out[key[:-1]] = amp*np.exp(1j*phase)
            else:
                out[key]=channels[key].data
                
        print('GWY data loaded')
        print(f"{wx*1e6} um x {wy*1e6} um")

    else:
        print('no GWY file')
        
    return out, wx,wy

               
               
               


def interferogramsToSpectra(inter,discardPhase = True, discardDC = True, windowF = None, paddingFactor = 1, shiftMaxToZero = False):
    """
    Convert a dictionnary of interferogram data in a dictionnary of spectra
    Channel "M" is interpreted as mirror position
    Channel "Z" is ignored
    Channel "Delay" is reshaped to match the other channel dimensions
    All other channels are fourier transformed

    Parameters
    ----------
    inter:  Dictionnary of 4d arrays [Row, Column/Delay, Run, Frequency]
        interferogram data 
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
    Dictionary of 4D spectra:
        Contains one channel for every channel in the input dictionary, except for "M" and "Z"
        Plus an additionl "Wavenumber" channel

    """
    out = {}

    pos = np.mean(inter['M'],axis=(0,1,2))
    nbRow = np.shape(inter['M'])[0]
    nbCol = np.shape(inter['M'])[1]
    nbRun = np.shape(inter['M'])[2]
    nbPoint = np.shape(inter['M'])[3]
    step = (pos[int(3*len(pos)/4)]-pos[int(len(pos)/4)])/(len(pos) - int(len(pos)/2)) #Average the step between first and third quarter of data

    if discardPhase:
        wavenumber = np.linspace(0,0.25e-2/step,(int(nbPoint*paddingFactor/2))+1)
    else:
        wavenumber = np.linspace(0,0.5e-2/step,nbPoint*paddingFactor)
    wavenumber = np.repeat(wavenumber[np.newaxis,:], nbRun, axis = 0)
    wavenumber = np.repeat(wavenumber[np.newaxis,:,:], nbCol, axis = 0)
    wavenumber = np.repeat(wavenumber[np.newaxis,:,:,:], nbRow, axis = 0)
    out["Wavenumber"] = wavenumber
    
    if 'Delay' in inter:
        delay=inter["Delay"][:,:,:,0]
        delay = np.repeat(delay[:,:,:,np.newaxis], wavenumber.shape[-1], axis = 3)
        out["Delay"] = delay
        
    
    for key, array in inter.items():
        if key != "Z" and key != "M" and key != "Delay" and not(key.startswith('k')):
            processedInter = array
            if discardPhase:
                for row in processedInter:
                    for col in row:
                        for trace in col:
                            trace *= np.exp(-j*miscUtil.dataAngle(trace))
            if discardDC:
                processedInter = np.transpose( np.transpose(processedInter) - np.transpose(np.mean(processedInter,3)))
            if windowF != None:
                window = windowF(nbPoint)
            else:
                window = 1
            processedInter = processedInter*window
            processedInter = np.concatenate((processedInter,np.zeros((nbRow, nbCol, nbRun,  nbPoint*(paddingFactor-1)))),3)
            if shiftMaxToZero:
                processedInter = np.roll(processedInter, -np.argmax(np.abs(processedInter)),3)
            if discardPhase:
                spectrum = np.fft.rfft(np.real(processedInter))
            else:
                spectrum = np.fft.fft(processedInter)
            out[key] = spectrum
    
    return out


def SNR(SpectralData):
    """
    Calculate SNR of spectral data. S as maximum spectral intensity. N as mean high frequency intensity.
    

    Parameters
    ----------
    SpectralData : dictionary of multidim arrays
        

    Returns
    -------
    SNR : dictionary of multidim array 

    """
    
    SNR={}
    for key in SpectralData.keys():
        if key.startswith('O'):
            SpectraMag=np.abs(SpectralData[key])
            S=np.max(SpectraMag,-1) #Signal as maximum of the spectra
            N=np.mean(SpectraMag[...,-100:],-1) #Noise as medium of high frequency components of the spectrum
            SNR[key]= S/N
    return SNR
    

def SaveDataH5(filename,data):  
    """"
    Save data as h5 file.
    
    Parameters
    ----------
    filename: str
        filepath to save 
    data:  Dictionnary of 4D numpy array 
        Interferogram data. It needs to contain channels O and A.
    
    
    """
    if os.path.splitext(filename)[1]!='.h5':
        filename=os.path.splitext(filename)[0]+'.h5'
        
    with h5py.File(filename, 'w') as hdf:
        for key, value in data.items():
            # Create a dataset for each array in the dictionary
            hdf.create_dataset(key, data=value)
    print('Saved to h5')

def LoadDataH5(filename):
    """
    Load Data from .h5

    Parameters
    ----------
    filename: str
        filepath to save 

    Returns
    -------
    out : Dictionnary

    """
    if os.path.splitext(filename)[1]!='.h5':
        h5_path=os.path.splitext(filename)[0]+'.h5'
    else:
        h5_path=filename
        
    out={}
    with h5py.File(h5_path, 'r') as hdf:
        # Iterate over each dataset in the file
        for name in hdf.keys():
            # Load the data for each dataset into the dictionary
            out[name] = np.array(hdf[name])
    print('Loaded from h5')
    return out


def BalanceDetectionCorrection(data_in,k_mean=False):
    
    """"
    Correct optical signals O{n} with A{n} in balanced detection scheme.
    It find the scaling factork and add it to the dictionary.
    
    Parameters
    ----------
    data_in: dictionary of 4d-array 
        interferogram data at input. It needs to contain channels O and A.
    k_mean: bool
        if True it use the average value of k to correct O
    
    
    Returns
    -------
    data_out:dictionary of 4d-array
        interferogram data output. O{n} are replaced. k{n} are added.
    
    """
    
    max_index = 0
    flagA=False
    
    data_out={}
    
    # Find maximum hamonic avaiable 
    for key in data_in.keys():
        data_out[key]=data_in[key]
        if key.startswith('O'):
            index = int(key[1:])  
            if index > max_index:
                max_index = index
        if key.startswith('A'):
            flagA=True

    if not flagA: # check existance of Channel A
        print('No Channel A avaiable')         
 
    for n in range(max_index+1):  # Loop over the harmonics
        O=data_in[f"O{n}"]-np.mean(data_in[f"O{n}"],-1,keepdims=True)
        if flagA: # check existance of Channel A
            A=data_out[f"A{n}"]-np.mean(data_out[f"A{n}"],-1,keepdims=True)
            k=np.sum( O*np.conj(A), -1, keepdims=True)/np.sum( np.abs(A)**2, -1, keepdims=True)
            data_out[f"k{n}"]=k
            if k_mean:
                k1=np.mean(k)
                data_out[f"O{n}"] = O-k1*A
            else:   
                data_out[f"O{n}"] = O-k*A
        else:
            data_out[f"k{n}"]=0*O

    return data_out
    



if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from scipy import signal

    interData = loadFullInterferogramData("example/examplePumpProbe.txt")

    plt.figure()
    plt.plot(interData["M"][0,0,0], np.abs(interData["O3"][0,0,0]))
    plt.plot(interData["M"][0,0,0], np.abs(interData["A3"][0,0,0]))
    plt.xlabel("Mirror position")
    plt.ylabel("Amplitude")


    #Create the window function if necessary
    windowF = lambda M: signal.windows.tukey(M, 0.3)  

    spectra = interferogramsToSpectra(interData, windowF=windowF, paddingFactor=4, shiftMaxToZero=True)

    plt.figure()
    plt.plot(spectra["Wavenumber"][0,0,0], np.abs(spectra["O3"][0,0,0]))
    plt.plot(spectra["Wavenumber"][0,0,0], np.abs(spectra["A3"][0,0,0]))
    plt.xlabel("Wavenumber (cm-1)")
    plt.ylabel("Amplitude")

    plt.figure()
    plt.plot(spectra["Wavenumber"][0,0,0], np.angle(spectra["O3"][0,0,0]))
    plt.plot(spectra["Wavenumber"][0,0,0], np.angle(spectra["A3"][0,0,0]))
    plt.xlabel("Wavenumber (cm-1)")
    plt.ylabel("Phase")

    plt.show()


    