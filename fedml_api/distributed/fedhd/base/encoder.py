import numpy as np
import sys
import time
import math
import torch
from copy import deepcopy
from torch.nn.functional import normalize


def kernel(x, y, name = 'dot'):
	dotKernel = np.dot
	gaussKernel = lambda x, y : gauss(x, y, 25)
	polyKernel  = lambda x, y : poly(x, y, 3, 5)
	cosKernel   = lambda x, y : np.dot(x,y) / (np.linalg.norm(x) * np.linalg.norm(y))
	if name == 'dot':
		k = dotKernel
	elif name == 'gauss':
		k = gaussKernel
	elif name == 'poly':
		k = polyKernel
	elif name == 'cos':
		k = cosKernel
	return k(x,y)

'''
def encoding_nonlinear(X_data, base_matrix, base_vector):
	enc_hvs = []
	n = 0
	D = len(base_vector)
	for elem in X_data:
		enc = np.empty(D)
		elem = torch.flatten(elem)
		elem = normalize(elem,p=2,dim=0)
		for i in range(0, D):
			temp = np.dot(elem, base_matrix[i])
			enc[i] = np.cos(temp + base_vector[i]) * np.sin(temp)
		enc_hvs.append(enc)
		n += 1
	return enc_hvs
'''


def encoding_nonlinear(X_data, base_matrix, base_vector):
	n = 0
	D = len(base_vector)
	enc_hvs = []
	for elem in X_data:
		elem = torch.flatten(elem)
		elem = normalize(elem,p=2,dim=0)
		enc_hvs.append(elem.tolist())
		n += 1

	enc_hvs = torch.tensor(enc_hvs)
	enc_hvs = np.matmul(base_matrix,enc_hvs.T).T
	return enc_hvs



def max_match_nonlinear(classes, enc_hv):
	index_max = -1
	score_max = -1
	for i in range(0, len(classes)):
		score = kernel(classes[i], enc_hv)
		if score > score_max:
			score_max = score
			index_max = i
	return index_max