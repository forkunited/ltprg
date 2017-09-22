import numpy as np
from sklearn.manifold import TSNE
import torch
import torch.nn as nn
import visdom

def reduce_dim(X):
	vis_model = TSNE(n_components=2, random_state=0)
	np.set_printoptions(suppress=True)
	reduced = vis_model.fit_transform(X) 
	return reduced

def vis_embedding(weights,vis):
	reduced = reduce_dim(weights)
	#TODO reuse same window
	#TODO color points usefully -- maybe by true basic level label
	vis.scatter(reduced)

