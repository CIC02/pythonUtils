# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 17:17:35 2024

@author: tnhannotte
"""

import numpy as np
import scipy.ndimage
import gwyfile



def extractLine(array, x1,y1,x2,y2, width = 0):
    """
    Extract a line profile in `array`, averaged over a given width around the line.
    Uses interpolation for non integer coordinates

    Parameters
    ----------
    array : 2D numpy array
        Data
    x1 : float
        x coordinate of the starting position (in px)
    y1 : float
        y coordinate of the starting position (in px)
    x2 : float
        x coordinate of the end position (in px)
    y2 : float
        y coordinate of the end position (in px)
    width : float, optional
        width of the line for averaging. The default is 0.

    Returns
    -------
    y : TYPE
        DESCRIPTION.
    avProfile : TYPE
        DESCRIPTION.

    """
    L = np.sqrt((x2-x1)**2+(y2-y1)**2)
    x = np.linspace(-width/2,width/2,int(np.ceil(width)+1))
    y = np.linspace(0,L,int(np.ceil(L)+1))
    theta = np.arctan2(y2-y1,x2-x1) - np.pi/2
    mesh = np.meshgrid(x,y)
    mesh = np.moveaxis(mesh,0,-1)
    mesh = np.reshape(mesh, (len(x)*len(y),2))
    rotmat = np.array([[np.cos(theta), -np.sin(theta)], 
                       [np.sin(theta),  np.cos(theta)]])
    mesh = (rotmat @ mesh.T).T
    mesh = mesh + [x1, y1]
    mesh = np.reshape(mesh,(len(y),len(x),2))
    mesh = np.moveaxis(mesh,-1,0)
    profile = scipy.ndimage.map_coordinates(array.T,mesh)
    # plt.figure()
    # plt.imshow(np.abs(profile))
    avProfile = np.mean(profile,1)
    return y, avProfile

def getOpticSig(obj, harm, backScan = False):
    """
    Extract a specific optical channel from a gwyfile object as a complex valued array
    (Use gwyfile.load("yourFile.gwy") to construct the object)

    Parameters
    ----------
    obj : gwyfile object
    harm : int
        demodulation harmonic
    backScan : bool, optional
        Extract the return scan if True. The default is False.

    Returns
    -------
    TYPE
        2D numpy array of complex valued signal from the gwyfile at the specified harmonic.

    """
    channels = gwyfile.util.get_datafields(obj)
    backStr = "R-" if backScan else ""
    ampChan = channels[backStr+"O"+str(harm)+"A raw"]
    amp = ampChan.data
    phase = channels[backStr+"O"+str(harm)+"P raw"].data
    return amp*np.exp(1j*phase)

def getRealDim(obj):
    """
    Extract the real physical width and height of a SNOM image from a gwyfile object

    Parameters
    ----------
    obj : gwyfile object

    Returns
    -------
    float
        width of the picture
    float
        height of the picture

    """
    channels = gwyfile.util.get_datafields(obj)
    Zchan = channels["Z C"]
    return Zchan.xreal, Zchan.yreal

def getOffset(obj):
    """
    Extract the coordinates of the top left corner of the picture

    Parameters
    ----------
    obj : gwyfile object

    Returns
    -------
    float
        x coordinate.
    float
        y coordinate.

    """
    channels = gwyfile.util.get_datafields(obj)
    Zchan = channels["Z C"]
    return Zchan.xoff, Zchan.yoff

