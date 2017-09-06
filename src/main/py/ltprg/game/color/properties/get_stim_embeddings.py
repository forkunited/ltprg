from __future__ import division
import numpy as np
import sys
sys.path.append('../../../../../../../../Packages/mungpy/src/main/py/mung/')
import torch
import torch.nn as nn
from torch.autograd import Variable
import time
from data import DataSet
from alexnet import PartialAlexnet, rgb_to_alexnet_input
from colorspace_conversions import hsls_to_rgbs

def hsl_from_datum(datum, H_fieldname, S_fieldname, L_fieldname):
	return [datum.get(H_fieldname), datum.get(S_fieldname), 
			   datum.get(L_fieldname)]

def load_stim_hsls(data_path):
	D = DataSet.load(data_path)
	hsls = []
	for i in range(len(D)):
		# target = [pull_hsls(D[i]."state.sTargetH", D[i]."state.sTargetS",
		# 			D[i]."state.sTargetL")]
		colors_this_trial = [hsl_from_datum(D[i], "state.sH_0", "state.sS_0", 
							    "state.sL_0"),
							 hsl_from_datum(D[i], "state.sH_1", "state.sS_1", 
							 	"state.sL_1"),
							 hsl_from_datum(D[i], "state.sH_2", "state.sS_2", 
							 	"state.sL_2")]
		for c in colors_this_trial:
			if not c == [None, None, None]:
				c = map(int, c)
				if c not in hsls:
					hsls.append(c)
	print '{} Colors'.format(len(hsls))
	return hsls

def get_stim_alexnet_embeddings(stim_rgbs, stop_layer):
	alexnet_to_fc6 = PartialAlexnet(stop_layer)
	print alexnet_to_fc6.model
	start_time = time.time()
	embeddings = alexnet_to_fc6.forward(rgb_to_alexnet_input(stim_rgbs[0]))
	for i in range(1, stim_rgbs.shape[0]):
		embeddings = torch.cat([embeddings, alexnet_to_fc6.forward(
								rgb_to_alexnet_input(stim_rgbs[i]))], 0)
	print 'Time for dataset = {}'.format(time.time()-start_time)
	embeddings = embeddings.data.numpy()
	return embeddings

if __name__=='__main__':
	hsls = load_stim_hsls()
	rgbs = hsls_to_rgbs(hsls)
	get_stim_alexnet_embeddings(rgbs, 'fc6')
