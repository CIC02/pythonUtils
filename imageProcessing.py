import cv2
import numpy as np

 
def alignImagesORB(im1, im2, saveMatchesImage = False, MAX_FEATURES = 500, GOOD_MATCH_PERCENT = 0.05):
	"""
	Uses ORB features and homography to align im1 with im2
	
	Parameters
	----------
	im1: openCV greyscale image (uint8 2d array)
		Image to align
	im2: openCV greyscale image (uint8 2d array)
		reference image
	saveMatchesImage : bool, optional
		If True, saves a visualisation of the matched features in "matches.jpg". The default is False.
	MAX_FEATURES: int, optional
		Maximum number of features to consider. The default is 500.
	GOOD_MATCH_PERCENT: float, optional
		Proportion of matches kept to compute the homography. The default is 0.05.
	Returns
		-------
		im1Reg: openCV greyscale image (uint8 2d array)
			Aligned image
		h: 3x3 float matrix
			Homography matrix

	"""
	# Images are already in greyscale
	im1Gray = im1
	im2Gray = im2

	# Detect ORB features and compute descriptors.
	orb = cv2.ORB_create(MAX_FEATURES)
	keypoints1, descriptors1 = orb.detectAndCompute(im1Gray, None)
	keypoints2, descriptors2 = orb.detectAndCompute(im2Gray, None)
 
	# Match features.
	matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
	matches = matcher.match(descriptors1, descriptors2, None)
 
	# Sort matches by score
	matches = sorted(matches, key = lambda x: x.distance, reverse=False)
 
	# Remove not so good matches
	numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
	matches = matches[:numGoodMatches]
 
	# Draw top matches
	if saveMatchesImage:
		imMatches = cv2.drawMatches(im1, keypoints1, im2, keypoints2, matches, None)
		cv2.imwrite("matches.jpg", imMatches)
 
	# Extract location of good matches
	points1 = np.zeros((len(matches), 2), dtype=np.float32)
	points2 = np.zeros((len(matches), 2), dtype=np.float32)
 
	for i, match in enumerate(matches):
		points1[i, :] = keypoints1[match.queryIdx].pt
		points2[i, :] = keypoints2[match.trainIdx].pt
 
	# Find homography
	h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)
	# Use homography
	height, width = im2.shape
	im1Reg = cv2.warpPerspective(im1, h, (width, height))

	return im1Reg, h