'''
  File name: cumMinEngVer.py
  Author: Hantian Liu
  Date created:
'''

'''
  File clarification:
    Computes the cumulative minimum energy over the vertical seam directions.
    
    - INPUT e: n × m matrix representing the energy map.
    - OUTPUT Mx: n × m matrix representing the cumulative minimum energy map along vertical direction.
    - OUTPUT Tbx: n × m matrix representing the backtrack table along vertical direction.
'''

import numpy as np

'''
def cumMinEng(e, i, j):
	if i == 1:
		return e[1, j]
	else:
		return e[i, j] + min(cumMinEng(e, i - 1, j - 1), cumMinEng(e, i - 1, j), cumMinEng(e, i - 1, j + 1))
'''

def cumMinEngVer(e):
	[n, m] = np.shape(e)
	Mx = np.zeros(np.array([n, m]))
	Tbx = np.zeros(np.array([n, m]))
	'''
    for i in range(1,n):
        for j in range(0, m):
            Mx[i,j]=cumMinEng(e, i, j)
    '''

	Mx[0, :] = e[0, :]
	for i in range(1, n):
		for j in range(0, m):
			if j - 1 < 0:
				min_last_row = min(Mx[i - 1, j], Mx[i - 1, j + 1])
				if (Mx[i - 1, j] == min_last_row):
					Tbx[i, j] = 0
				else:
					Tbx[i, j] = 1

			elif j + 1 >= m:
				min_last_row = min(Mx[i - 1, j - 1], Mx[i - 1, j])
				if (Mx[i - 1, j - 1] == min_last_row):
					Tbx[i, j] = -1
				else:
					Tbx[i, j] = 0

			else:
				min_last_row = min(Mx[i - 1, j - 1], Mx[i - 1, j], Mx[i - 1, j + 1])
				if (Mx[i - 1, j - 1] == min_last_row):
					Tbx[i, j] = -1
				elif (Mx[i - 1, j] == min_last_row):
					Tbx[i, j] = 0
				else:
					Tbx[i, j] = 1

			Mx[i, j] = e[i, j] + min_last_row

	return Mx, Tbx
