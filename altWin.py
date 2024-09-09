# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 11:13:03 2024

@author: tnhannotte


Tools to create window custom window function based on existing ones
"""

import numpy as np

def widenWindow(baseWindow, flatRatio):
    def window(M):
        base = baseWindow(int(M*(1-flatRatio)))
        fullWindow = np.ones(M)
        fullWindow[0:int(len(base)/2)] = base[0:int(len(base)/2)]
        fullWindow[M-int(len(base)/2):M] = base[len(base)-int(len(base)/2):len(base)]
        return fullWindow
    return window


def assymWindow(baseWindow,posMax):
    def window(M):
        winLeft = baseWindow(2*int(M*posMax))
        winRight = baseWindow(2*M - 2*int(M*posMax))
        win=np.ones(M)
        win[0:int(M*posMax)] = winLeft[0:int(M*posMax)]
        win[int(M*posMax):] = winRight[M-int(M*posMax):]
        return win
    return window