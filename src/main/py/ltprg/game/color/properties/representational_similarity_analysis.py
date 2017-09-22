from __future__ import division
import numpy as np
from scipy.spatial import distance
from scipy.stats import pearsonr
from visdom import Visdom

viz = Visdom()

def make_dissimilarity_matrix(stim_embeddings, distance_func):
	# inputs:
	# 	stim_embeddings		(num_stims x n numpy array,
	#						 where n is the number of 
	#					 	 dimensions in each embedding)
	# 	distance_func		(distance function)
	# returns num_stims x num_stims RDM in which an entry
	#	RDM_ij is the distance between stim i and stim j
	rdm = np.zeros((stim_embeddings.shape[0], stim_embeddings.shape[0]))
	for i in range(stim_embeddings.shape[0]):
		for j in range(stim_embeddings.shape[0]):
			rdm[i, j] = distance_func(stim_embeddings[i, :], 
								  stim_embeddings[j, :])
	return rdm

def representational_similarity(rdm1, rdm2):
	assert rdm1.shape[0] == rdm1.shape[1] == rdm2.shape[0] == rdm2.shape[1]
	triu_inds = np.triu_indices(n=rdm1.shape[0], k=1, m=rdm1.shape[0])
	r, pval = pearsonr(rdm1[triu_inds], rdm2[triu_inds])
	return r, pval

def make_heatmap(rdm, ttl):
	viz.heatmap(X=rdm, opts=dict(title=ttl))

def rsa_wrapper(rdm_a, rdm_b, name_a, name_b):

	make_heatmap(rdm_a, 'RDM: ' + name_a)
	make_heatmap(rdm_b, 'RDM: ' + name_b)

	r, p = representational_similarity(rdm_a, rdm_b)
	print '\nRepresentational similarity (Pearson r) between {} and {} = {}'.format(name_a, name_b, r)
	print 'p = {}'.format(p)

def example():
	embeddings1 = np.random.rand(5, 7)
	embeddings2 = np.random.rand(5, 7)
	rsa_wrapper(rdm1, rdm2, 'rand1', 'rand2')

if __name__ =='__main__':
	example()
	