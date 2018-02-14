'''
  File name: cumMinEngHor.py
  Author: Hantian Liu
  Date created:
'''

'''
  File clarification:
    Computes the cumulative minimum energy over the horizontal seam directions.
    
    - INPUT e: n × m matrix representing the energy map.
    - OUTPUT My: n × m matrix representing the cumulative minimum energy map along horizontal direction.
    - OUTPUT Tby: n × m matrix representing the backtrack table along horizontal direction.
'''
import numpy as np

def cumMinEngHor(e):
	[n, m] = np.shape(e)
	My = np.zeros(np.array([n, m]))
	Tby = np.zeros(np.array([n, m]))

	My[:, 0] = e[:, 0]
	for j in range(1, m):
		for i in range(0, n):
			if i - 1 < 0:
				min_last_col = min(My[i, j - 1], My[i + 1, j - 1])
				if (My[i, j - 1] == min_last_col):
					Tby[i, j] = 0
				else:
					Tby[i, j] = 1

			elif i + 1 >= n:
				min_last_col = min(My[i - 1, j - 1], My[i, j - 1])
				if (My[i - 1, j - 1] == min_last_col):
					Tby[i, j] = -1
				else:
					Tby[i, j] = 0

			else:
				min_last_col = min(My[i - 1, j - 1], My[i, j - 1], My[i + 1, j - 1])
				if (My[i - 1, j - 1] == min_last_col):
					Tby[i, j] = -1
				elif (My[i, j - 1] == min_last_col):
					Tby[i, j] = 0
				else:
					Tby[i, j] = 1

			My[i, j] = e[i, j] + min_last_col

	return My, Tby
