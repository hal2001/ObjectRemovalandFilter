#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 10/30/17 4:06 PM 

@author: Hantian Liu
"""
import numpy as np

def label(I, mask, x1, x2,y1,y2):
    I_label = np.ones(np.shape(I[:, :, 0]))
    size=np.shape(I_label)
    I_label[mask == 0] = -10
    if x2+1>=size[0]:
        x2=x2-1
    if y2+1>=size[1]:
        y2=y2-1
    for i in range(x1, x2+1):
        c = np.where(I_label[i, y1:y2+1]==I_label[i, y1:y2+1].min())

        if I_label[i,:].__len__()<=1:
            I_label[i,:]=1
            continue
        else:
            c=c[0]
            c_start=c[0]
            c_end = c[-1]
            next=1
            if c_start==0:
                c_start=c[next]
            if c_end==size[1]-1:
                c_end=c[-next-1]

            I_label[i,y1+c_start:y1+c_end]=-10

    return I_label