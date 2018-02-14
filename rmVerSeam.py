'''
  File name: rmVerSeam.py
  Author: Hantian Liu
  Date created:
'''

'''
  File clarification:
    Removes vertical seams. You should identify the pixel from My from which 
    you should begin backtracking in order to identify pixels for removal, and 
    remove those pixels from the input image. 
    
    - INPUT I: n × m × 3 matrix representing the input image.
    - INPUT Mx: n × m matrix representing the cumulative minimum energy map along vertical direction.
    - INPUT Tbx: n × m matrix representing the backtrack table along vertical direction.
    - OUTPUT Ix: n × (m - 1) × 3 matrix representing the image with the row removed.
    - OUTPUT E: the cost of seam removal.
'''

import numpy as np


def rmVerSeam(I, Mx, Tbx, I_label):
	[n, m] = np.shape(Mx)

	Ix = np.zeros(np.shape(I))

	I_labelnew=np.zeros(np.shape(I_label))

	E = Mx.min(1)[n - 1]
	ind = np.where(Mx[n-1,:] == E)
	j = ind[0] # choose the smaller index
	j = j[0]
	j = int(j)

	for i in range(1, n + 1):
		i0 = n - i
		Ix[i0, 0:j, 0] = I[i0, 0:j, 0]
		Ix[i0, j:m - 1, 0] = I[i0, j + 1:, 0]
		Ix[i0, 0:j, 1] = I[i0, 0:j, 1]
		Ix[i0, j:m - 1, 1] = I[i0, j + 1:, 1]
		Ix[i0, 0:j, 2] = I[i0, 0:j, 2]
		Ix[i0, j:m - 1, 2] = I[i0, j + 1:, 2]

		I_labelnew[i0, 0:j] = I_label[i0, 0:j]
		I_labelnew[i0, j:m - 1] = I_label[i0, j + 1:]

		j = j + Tbx[i0, j]
		j = int(j)

	Ix = Ix[:, 0:m - 1, :]

	I_labelnew=I_labelnew[:, 0:m-1]

	Ix = Ix.astype('uint8')
	return Ix, E, I_labelnew
