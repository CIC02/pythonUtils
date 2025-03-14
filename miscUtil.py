# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 17:17:35 2024

@author: tnhannotte
"""

import numpy as np
import scipy.ndimage
import gwyfile
import cv2
from imageProcessing import alignImagesORB

pi = np.pi
j = 1j

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
	y : 1D array
		position along the line.
	avProfile : 1D array
		profile.

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

def circularProfile(array,x0,y0,r,Theta):
	"""
	Extract a circular profile from "array"

	Parameters
	----------
	mat : 2D numpy array
		dataset
	x0 : float
		x coordinate of the circle center
	y0 : float
		y coordinate of the circle center
	r : float
		radius of the circle
	Theta : 1D array of float
		List of angles to extract from the circle

	Returns
	-------
	1D numpy array
		Circular profile, same size as Theta

	"""
	x = x0 + r*np.cos(Theta)
	y = y0 + r*np.sin(Theta)
	return np.asarray(scipy.ndimage.map_coordinates(array, np.vstack((np.flipud(y),x))))

def dataAngle(x, y = None):
	"""
	Return the angle of the best fit line for the array of points (x,y), or the array x in the complex plane.

	Parameters
	----------
	x : 1D array of float or complex
		x coordinates, or complexe coordinate of the points to fit.
	y : 1D array og float, optional
		y coordinates of the points to fit. If not provided, x will be treated as a complex number array. The default is None.

	Returns
	-------
	angle : float
		Angle of the best fit line in radians.

	"""
	if y == None:
		xx = np.real(x)
		yy = np.imag(x)
	else:
		xx = x
		yy = y
	if np.min(xx) == np.max(xx):        # Data is perfectly vertical
		return np.pi/2
	if np.min(yy) == np.max(yy):        # Data is perfectly hozizontal
		return 0
	regressHor = scipy.stats.linregress(xx,yy)
	regressVer = scipy.stats.linregress(yy,xx)
	if regressHor.stderr < regressVer.stderr:
		angle = np.arctan(regressHor.slope)
	else:
		angle = np.arctan(1/regressVer.slope)
	return angle


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
	rawStr = " raw" if "O0A raw" in channels else "" 
	ampChan = channels[backStr+"O"+str(harm)+"A" + rawStr]
	amp = ampChan.data
	phase = channels[backStr+"O"+str(harm)+"P" + rawStr].data
	return amp*np.exp(1j*phase)

def getAlignedOpticSigORB(obj, objRef, harm, backScan = False,  MAX_FEATURES = 500, GOOD_MATCH_PERCENT = 0.05):
	"""
	Extract a specific optical channel from the gwyfile object "obj" as a complex valued array
	Uses the channel "M1A" to align the images from "obj" with "objRef", using ORB feature detection
	This method can give more precise result than the correlation method, but it can also incorrectly match features and give a completely wrong shift.
	Adjust "GOOD_MATCH_PERCENT" to improve the alignement quality.
	(Use gwyfile.load("yourFile.gwy") to construct the object)

	Parameters
	----------
		
	obj : gwyfile object
		Object with thte data to extract and align
	objRef: gwyFile object
		Object used as reference for the alignement
	harm : int
		demodulation harmonic
	backScan : bool, optional
		Extract the return scan if True. The default is False.
	MAX_FEATURES: int, optional
		Maximum number of features to consider for alignement. The default is 500.
	GOOD_MATCH_PERCENT: float, optional
		Proportion of matches kept to compute the homography. The default is 0.05.
	Returns
	-------
	TYPE
		2D numpy array of complex valued signal from the gwyfile at the specified harmonic.

	"""
	#get mechanical amplitudes and convert to openCV format
	channels = gwyfile.util.get_datafields(obj)
	M = np.asarray(channels["M1A raw"].data)
	M = M - np.min(M)
	M = M*255/np.max(M)
	M = np.asarray(M.astype(np.uint8))

	channels = gwyfile.util.get_datafields(objRef)
	Mref = np.asarray(channels["M1A raw"].data)
	Mref = Mref - np.min(Mref)
	Mref = Mref*255/np.max(Mref)
	Mref = np.asarray(Mref.astype(np.uint8))

	#Find the homography and keep only the translation component
	transformed_img, h  = alignImagesORB(M, Mref, MAX_FEATURES = MAX_FEATURES, GOOD_MATCH_PERCENT = GOOD_MATCH_PERCENT)
	translation = [h[1,2], h[0,2]]
	print(translation)

	#Extract and shift the optical signal
	Oraw = getOpticSig(obj, harm, backScan = backScan)
	Oshifted = np.asarray(scipy.ndimage.shift(Oraw, translation))
	return Oshifted

def getAlignedOpticSig(obj, objRef, harm, backScan = False, interpolationFactor = 1):
	"""
	Extract a specific optical channel from the gwyfile object "obj" as a complex valued array
	Uses the channel "M1A" to align the images from "obj" with "objRef", using simple correlation
	(Use gwyfile.load("yourFile.gwy") to construct the object)

	Parameters
	----------
		
	obj : gwyfile object
		Object with thte data to extract and align
	objRef: gwyFile object
		Object used as reference for the alignement
	harm : int
		demodulation harmonic
	backScan : bool, optional
		Extract the return scan if True. The default is False.
	interpolationFactor: float
		Interpolation applied to the amplitude images before cross corelation, for subpixel accuracy. Default is 1 (no interpolation)	
	Returns
	-------
		2D numpy array of complex valued signal from the gwyfile at the specified harmonic.

	"""
	M = gwyfile.util.get_datafields(obj)["M1A raw"].data
	Mref = gwyfile.util.get_datafields(objRef)["M1A raw"].data

	M -= np.mean(M)
	Mref -= np.mean(Mref)

	x = np.linspace(0,M.shape[0], M.shape[0]*interpolationFactor)
	y = np.linspace(0,M.shape[1], M.shape[1]*interpolationFactor)
	mesh = np.meshgrid(x,y)

	Mfine = scipy.ndimage.map_coordinates(M,mesh)
	MrefFine = scipy.ndimage.map_coordinates(Mref,mesh)


	corr  = scipy.signal.correlate(Mfine, MrefFine, mode = "same")

	xshift, yshift = np.unravel_index(np.argmax(corr), corr.shape)
	xshift = (xshift-corr.shape[0]/2)/interpolationFactor
	yshift = (yshift-corr.shape[1]/2)/interpolationFactor
	 #Extract and shift the optical signal
	Oraw = getOpticSig(obj, harm, backScan = backScan)
	Oshifted = np.asarray(scipy.ndimage.shift(Oraw, (-yshift, -xshift)))
	return Oshifted

def getZmasks(gwyObj, threshold_height = None,  threshold_area = 5):
	"""
	Returns a list of masks based on the Z channel of gwyObj.
	Each mask correspond to one connected elevated area on the picture.


	Parameters
	----------
		
	gwyObj : gwyfile object
		Object containing the Z channel
	threshold_height : float or None (optional)
		If None, automatically choose a threshold based on Otsu's method (see https://docs.opencv.org/4.x/d7/d4d/tutorial_py_thresholding.html)
		To specify a threshold manually, this parameter is used on a normalized Z channel, with 0 being the minimum height, and 1 the maximum height.
	threshold_area : float (optional)
		Masks with an area smaller than threshold_area will be rejected (default: 5) 
	Returns
	-------
	list of 2D numpy array
		List of masks for each detected connected structure.
	"""
	#Get Z channel
	channels = gwyfile.util.get_datafields(gwyObj)
	key = "Z C" if "Z C" in channels else "Z"
	Zchan = channels[key].data
	#Convert for openCV
	Zchan = Zchan - np.min(Zchan)
	Zchan = Zchan*255/np.max(Zchan)
	Zchan = np.asarray(Zchan.astype(np.uint8))
	if threshold_height == None:
		mode = cv2.THRESH_BINARY + cv2.THRESH_OTSU
		threshold = 0
	else:
		mode = cv2.THRESH_BINARY
		threshold = 255*threshold_height
	thresh = cv2.threshold(Zchan, threshold, 255, mode)[1]

	cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	cnts = cnts[0] if len(cnts) == 2 else cnts[1]

	masks = []

	for c in cnts:
		mask = np.zeros(np.shape(Zchan), dtype=np.uint8)
		area = cv2.contourArea(c)
		if area > threshold_area:
			cv2.drawContours(mask, [c], -1, color = 1, thickness=cv2.FILLED)
			masks.append(mask)
	return masks



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
	key = "Z C" if "Z C" in channels else "Z"
	Zchan = channels[key]
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
	key = "Z C" if "Z C" in channels else "Z"
	Zchan = channels[key]
	return Zchan.xoff, Zchan.yoff

