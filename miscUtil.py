# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 17:17:35 2024

@author: tnhannotte
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage
import gwyfile



def extractLine(array, x1,y1,x2,y2, width = 0):
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
    profile = scipy.ndimage.map_coordinates(sig.T,mesh)
    # plt.figure()
    # plt.imshow(np.abs(profile))
    avProfile = np.mean(profile,1)
    return y, avProfile

def getOpticSig(obj, harm, backScan = False):
    channels = gwyfile.util.get_datafields(obj)
    backStr = "R-" if backScan else ""
    ampChan = channels[backStr+"O"+str(harm)+"A raw"]
    amp = ampChan.data
    phase = channels[backStr+"O"+str(harm)+"P raw"].data
    return amp*np.exp(1j*phase)

def getRealDim(obj):
    channels = gwyfile.util.get_datafields(obj)
    Zchan = channels["Z C"]
    return Zchan.xreal, Zchan.yreal

def getOffset(obj):
    channels = gwyfile.util.get_datafields(obj)
    Zchan = channels["Z C"]
    return Zchan.xoff, Zchan.yoff

filename = "2024-10-22 125814 PH antennaIgor-highRes.gwy"
obj = gwyfile.load(filename)
sig = getOpticSig(obj, 3)
width, height = getRealDim(obj)

scale = height/len(sig)

plt.figure()
plt.imshow(np.abs(sig))

# x1, prof1 = extractLine(sig, 20, 465, 76, 465, width = 15)
# x2, prof2 = extractLine(sig, 76, 465, 135, 400, width = 15)
# x3, prof3 = extractLine(sig, 135, 400, 133, 110, width = 15)

# x = np.concatenate((x1,x2+x1[-1],x3+x1[-1]+x2[-1])) * scale
# profile = np.concatenate((prof1,prof2,prof3))

# plt.figure()
# plt.plot(x,np.abs(profile))

# plt.figure()
# plt.plot(x,np.unwrap(np.angle(profile)))
# plt.plot(x,3*np.ones(len(x)))
# plt.plot(x,(3+2*np.pi)*np.ones(len(x)))