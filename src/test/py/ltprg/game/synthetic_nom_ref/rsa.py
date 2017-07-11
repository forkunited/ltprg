from __future__ import division
import torch
import torch.nn as nn
from torch.autograd import Variable

def uniform_prior(len):
	# returns 1D tensor
	t = torch.ones(len)
	return Variable(t/torch.sum(t))

def normalize_tensor_rows(t):
	# t is 2D tensor
	row_sums = torch.sum(t, dim=1)
	return torch.div(t, row_sums.expand_as(t))

def model_literal_listener(learned_lexicon, world_prior):
	# multiply rows by obj_prior
	listener = world_prior.expand_as(learned_lexicon) * learned_lexicon
	return normalize_tensor_rows(listener)

def model_speaker_1(learned_lexicon, world_prior, alpha, cost_weight, costs):
	# returns logSoftmax -- not for use in RSA listener model
	t = torch.transpose(model_literal_listener(
						learned_lexicon, world_prior), 0, 1)
	utilities = torch.log(t + 10e-06)
	x = (alpha * utilities).sub_(cost_weight * costs.expand_as(utilities))
	m = nn.LogSoftmax()
	return m(x)