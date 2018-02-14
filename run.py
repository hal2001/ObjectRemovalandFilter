#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 10/29/17 2:33 PM

@author: Hantian Liu and Jinglei Yu
"""

from detect_p import detect
import imutils
from PIL import Image
import matplotlib.image as Img
import matplotlib.pyplot as plt
import numpy as np
from cumMinEngVer import cumMinEngVer
from cumMinEngHor import cumMinEngHor
from rmVerSeam import rmVerSeam
from genEngMap import genEngMap
from rmHorSeam import rmHorSeam
from carv import carv
import imageio
from numpy.linalg import pinv
import math
from Domask import Domask
from label import label
from insert import insertVerSeam


# load image
image_path='./images/5.jpg'

# pedestrian detection
pick=detect(image_path)
ind = input('Enter the one you want to delete: ')
ind = int(ind)
ind=pick[ind-1]
#print(ind)

# original image resizing
I = np.array(Image.open(image_path))
I=imutils.resize(I, width = min(600, I.shape[1]))
plt.figure(1)
plt.imshow(I)

# auto-refined mask
x1=ind[1]
y1=ind[0]
x2=ind[3]
y2=ind[2]
mask, h, w=Domask(I, x1, x2, y1, y2)
I_label=label(I,mask,x1,x2,y1,y2)
plt.figure(2)
plt.imshow(I_label)

# object removal with weighted seam carving
nr=0
nc=ind[3]-ind[1]
[Ic, T]=carv(I, nr, nc, I_label)
fig2=plt.figure(3)
plt.imshow(Ic)

# seam insertion
Ic_n=Ic
summ = int(nc/2)
I_label_insert = np.ones(np.shape(Ic_n[:, :, 0]))
a=genEngMap(Ic_n, I_label_insert)
[Mx, Tbx]=cumMinEngVer(a)
[Ix,Ex]=insertVerSeam(Ic_n, Mx, Tbx, summ)
plt.figure(4)
plt.imshow(Ix)
plt.show()


