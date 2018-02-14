#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 10/29/17 4:46 PM 

@author: Hantian Liu
"""

from __future__ import print_function
from imutils.object_detection import non_max_suppression
from imutils import paths
import numpy as np
import argparse
import imutils
import cv2
from detect_p import detect
import math
from findDerivatives import findDerivatives
from utils import rgb2gray


# def Domask(path):
def Domask(I, x1, x2, y1, y2):
	xA = x1
	yA = y1
	xB = x2
	yB = y2
	image = I.copy()

	# pts = detect(path)
	# (xA, yA) = pts[0]
	# (xB, yB) = pts[1]
	h = int(math.fabs(yA - yB))
	w = int(math.fabs(xA - xB))

	#image[x1:(x1 + w),y1:(y1 + h), :] = 0
	#cv2.imshow("contour", image)
	#cv2.waitKey(0)

	# image = cv2.imread(path)
	# image = imutils.resize(image, width=min(400, image.shape[1]))
	mask = np.ones([image.shape[0], image.shape[1]])

	I_choice = image[xA:xB, yA:yB, :]
	# I_choice = image[yA:yB, xA:xB, :]
	# E = cannyEdge(I_choice)
	I_gray = rgb2gray(I_choice)
	mag = findDerivatives(I_gray)
	threshold_low = 0.65 #0.015
	threshold_low = threshold_low * mag.max()
	# for strong edge
	threshold_high = 0.8 #0.115
	threshold_high = threshold_high * mag.max()

	edges = cv2.Canny(I_choice, threshold_low, threshold_high) # 500, 800
	_, contours0, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	contours = [cv2.approxPolyDP(cnt,10, True) for cnt in contours0]
	# cv2.drawContours(mask, contours, -1, (0, 255, 0), 10, offset=(xA, yA))
	#cv2.morphologyEx(image, cv2.MORPH_CLOSE, np.ones((5, 5), dtype = 'uint8')) #filter
	cv2.drawContours(mask, contours, -1, (0, 255, 0), 6, offset = (yA, xA))
	cv2.imshow("contour", mask)
	cv2.waitKey(0)
	#contours=cv2.morphologyEx(image, cv2.MORPH_CLOSE, np.ones((5, 5), dtype = 'uint8'))
	cv2.drawContours(image, contours, -1, (0, 255, 0), 6, offset = (yA, xA))
	cv2.imshow("contour", image)
	cv2.waitKey(0)
	cv2.imshow("choice", I_choice)
	cv2.waitKey(0)

	mask[0:x1,:]=1
	mask[x2+1:,:]=1
	mask[:,0:y1]=1
	mask[:,y2+1:]=1

	return mask, h, w
