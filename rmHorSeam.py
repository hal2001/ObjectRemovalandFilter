'''
  File name: rmHorSeam.py
  Author: Hantian Liu
  Date created:
'''

'''
  File clarification:
    Removes horizontal seams. You should identify the pixel from My from which 
    you should begin backtracking in order to identify pixels for removal, and 
    remove those pixels from the input image. 
    
    - INPUT I: n × m × 3 matrix representing the input image.
    - INPUT My: n × m matrix representing the cumulative minimum energy map along horizontal direction.
    - INPUT Tby: n × m matrix representing the backtrack table along horizontal direction.
    - OUTPUT Iy: (n − 1) × m × 3 matrix representing the image with the row removed.
    - OUTPUT E: the cost of seam removal.
'''
import numpy as np

def rmHorSeam(I, My, Tby):
	[n, m] = np.shape(My)

	Iy = np.zeros(np.shape(I))

	E = My.min(0)[m - 1]
	ind = np.where(My[:,m-1] == E)
	i = ind[0]
	i=i[0]
	i = int(i)

	for j in range(1, m + 1):
		j0 = m - j
		Iy[0:i, j0, 0] = I[0:i, j0, 0]
		Iy[i:n - 1, j0, 0] = I[i + 1:, j0, 0]
		Iy[0:i, j0, 1] = I[0:i, j0, 1]
		Iy[i:n - 1, j0, 1] = I[i + 1:, j0, 1]
		Iy[0:i, j0, 2] = I[0:i, j0, 2]
		Iy[i:n - 1, j0, 2] = I[i + 1:, j0, 2]
		i = i + Tby[i, j0]
		i = int(i)

	Iy = Iy[0:n - 1, :, :]
	Iy = Iy.astype('uint8')
	return Iy, E
