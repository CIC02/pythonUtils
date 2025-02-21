# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 2025

@author: tnhannotte
"""

import numpy as np
from skimage.restoration import unwrap_phase

pi = np.pi
j = 1j


def vectorialUnwrap(fx2, fy2, fxfy, mask = None):
	"""
	Recover the complex field enhancement component fx and fy from the measurement of fx^2, fy^2 and fx*fy
	Can be used on a specific part of the image by specifying a mask
	The phase reconstruction is based on an unwrapping algorithm, and will not work properly on discontinuous structure

	Parameters
	----------
	fx2 : 2d numpy array
		square of the x component
	fy2 : 2d numpy array
		square of the y component
	fxfy : 2d numpy array
		product of the x and y component
	mask : 2d numpy array (optionnal)
		Used to select a specific area to recover in the image. Should only contain 0 and 1.
	Returns
	-------
	fx : 2d numpy array
		x component, multiplied by "mask" if specified
	fy : 2d numpy array
		y component, multiplied by "mask" if specified
	alpha : 2d numpy array
		Projection angles used for unwrapping. Result should not be trusted if this map has discontinuities in the area of interest.
	"""
	if mask is None:
		mask = 1

	temp = (4*(np.real(fxfy*np.conjugate(fx2)) + np.real(fxfy*np.conjugate(fy2)))) / (np.abs(fy2)**2 - np.abs(fx2)**2)
	angleProj= np.arctan(0.5*(-temp+np.sqrt(temp**2+4)))
	angleProj = unwrap_phase(np.angle(np.exp(j*angleProj*4)))/4
	a = np.cos(angleProj)
	b = np.sin(angleProj)
	c = np.cos(angleProj + pi/2)
	d = np.sin(angleProj + pi/2)
	f2Proj1 = a**2*fx2 + 2*a*b*fxfy + b**2*fy2
	f2Proj2 = c**2*fx2 + 2*c*d*fxfy + d**2*fy2

	fProj1Amp = np.sqrt(np.abs(f2Proj1))
	fProj1Phase = unwrap_phase(np.angle(f2Proj1))/2
	fProj1 = fProj1Amp*np.exp(j*fProj1Phase) * mask

	fProj2Amp = np.sqrt(np.abs(f2Proj2))
	fProj2Phase = unwrap_phase(np.angle(f2Proj2))/2
	fProj2 = fProj2Amp*np.exp(j*fProj2Phase) * mask

	#The sig after unwrap is arbitrary. We select the sign of fProj2 that is consistent with the measurement
	error1 = np.mean(np.abs(((d*fProj1 - b*fProj2) / (a*d-b*c))**2 - fx2 * mask)**2)
	error2 = np.mean(np.abs(((d*fProj1 + b*fProj2) / (a*d-b*c))**2 - fx2 * mask)**2)

	if(error1 > error2):
		fProj2 *= -1

	fx = (d*fProj1 - b*fProj2) / (a*d-b*c)
	fy = (c*fProj1 - a*fProj2) / (b*c-a*d)
	return fx,fy, angleProj*mask