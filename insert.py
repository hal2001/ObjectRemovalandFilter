#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 10/30/17 5:14 PM 

@author: Hantian Liu
"""
import numpy as np
import math

def insertVerSeam(I, Mx, Tbx, col): #,id,frac):
	[n, m] = np.shape(Mx)

	Io=I.copy()
	for col in range(0, col):
		E = Mx.min(1)[n - 1]
		ind = np.where(Mx[n-1,:] == E)
		j = ind[0] # choose the smaller index
		j = j[0]
		j = int(j)

		Mx[n-1, j]=math.inf

		Ix = np.zeros(np.array([np.shape(Io)[0], np.shape(Io)[1] + 1, 3]))

		for i in range(1, n + 1):
			i0 = n - i
			Io = Io.astype(np.int)

			Ix[i0, 0:j, :] = Io[i0, 0:j, :]
			Ix[i0, j, :] = (Io[i0, j - 1, :] + Io[i0, j, :])/2
			Ix[i0, j + 1:, :] = Io[i0, j:, :]

			'''
			Ix[i0, 0:j, :] = I[i0, 0:j, :]
			Ix[i0, j, :] = I[i0, j - id, :] * (1 - frac) + I[i0, j, :] * frac
			Ix[i0, j + 1:m + 1, :] = I[i0, j:, :]
			'''

		j = j + Tbx[i0, j]
		j = int(j)
		Io=Ix.copy()

	Ix = Ix.astype('uint8')
	return Ix, E
