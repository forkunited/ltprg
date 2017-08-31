from __future__ import division
import torch
import torch.nn as nn
import torch.cuda as cuda
from torch.autograd import Variable

def uniform_prior(len):
	# returns 1D tensor
	t = torch.ones(len)
	return Variable(t/torch.sum(t))

def normalize_tensor_rows(t):
	# t is 2D tensor
	row_sums = torch.unsqueeze(torch.sum(t, dim=1), 1)
	return torch.div(t, row_sums.expand_as(t))

def model_literal_listener(learned_lexicon, world_prior):
	# multiply rows by obj_prior
	listener = world_prior.expand_as(learned_lexicon) * learned_lexicon
	return normalize_tensor_rows(listener)

def model_speaker_1_legacy(learned_lexicon, world_prior, alpha, cost_weight, costs):
	# returns logSoftmax -- not for use in RSA listener model
	t = torch.transpose(model_literal_listener(
						learned_lexicon, world_prior), 0, 1)
	utilities = torch.log(t + 10e-06)
	x = (alpha * utilities).sub_(cost_weight * costs.expand_as(utilities))
	m = nn.LogSoftmax()
	return m(x)

def model_speaker_1(learned_lexicon, rsa_params):
	# returns logSoftmax -- not for use in RSA listener model
	t = torch.transpose(model_literal_listener(
						learned_lexicon, rsa_params.world_prior), 0, 1)
	utilities = torch.log(t + 10e-06)
	x = (rsa_params.alpha * utilities).sub_(
			rsa_params.cost_weight * rsa_params.costs.expand_as(utilities)
		)
	m = nn.LogSoftmax()
	return m(x)

class RSAParams(object):
	""" RSA Parameters enapsulation.
	"""

	def __init__(self, alpha, cost_weight, cost_dict, gold_standard_lexicon,
				 world_sz=3):
		"""
		alpha				(speaker rationality param)
		cost_dict			(dict of utterance costs)
		cost_weight		(utterance cost weight in RSA model)
		gold_standard_lexicon (num utterances x num objects np array of 
							 ground-truth lexicon used to generate data)
		"""
		# dtype
		if cuda.is_available():
		    self.dtype = torch.cuda.FloatTensor
		else:
		    self.dtype = torch.FloatTensor

		self.alpha = alpha
		self.cost_weight = cost_weight
		self.world_prior = uniform_prior(world_sz).type(self.dtype)
		self.costs = Variable(torch.FloatTensor(
			[cost_dict[str(k)] for k in range(len(cost_dict))]).type(self.dtype))
		self.gold_standard_lexicon = Variable(torch.FloatTensor(
			gold_standard_lexicon).type(self.dtype))

	def to_dict(self):
		d = dict()
		d['alpha'] = self.alpha
		d['cost_weight'] = self.cost_weight
		d['costs'] = self.costs
		return d
