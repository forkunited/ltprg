from __future__ import division
import os
import math
import random
import json
import numpy as np
import torch
from fixed_alternatives_set_models import ModelTrainer, load_json

# TODO: Integrate better with rest of code; this is just the quick + dirty
def display_predictions_at_min_loss(model_name, hidden_szs, hiddens_nonlinearity,
				 train_data, validation_data, utt_set_sz,
				 obj_set_sz, obj_embedding_type, utt_dict, obj_dict,
				 weight_decay, learning_rate,
				 visualize_opt,  
				 alpha, cost_dict, cost_weight, gold_standard_lexicon,
				 save_path, 
				 obj_names_to_subs_dict, obj_names_to_basics_dict,
				 obj_names_to_supers_dict):
	trainer = ModelTrainer(model_name, hidden_szs, hiddens_nonlinearity,
				 train_data, validation_data, utt_set_sz,
				 obj_set_sz, obj_embedding_type, utt_dict, obj_dict,
				 weight_decay, learning_rate,
				 visualize_opt,  
				 alpha, cost_dict, cost_weight, gold_standard_lexicon,
				 save_path)
	
	print trainer.model
	print 'Loading checkpoint from epoch with lowest validation-set loss'
	checkpoint = torch.load(save_path + 'model_best.pth.tar')
	print 'Epoch = {}'.format(checkpoint['epoch'])
	trainer.model.load_state_dict(checkpoint['state_dict'])
	trainer.optimizer.load_state_dict(checkpoint['optimizer'])
	
	trainer.obj_names_to_subs   = obj_names_to_subs_dict # TODO: Fix me
	trainer.obj_names_to_basics = obj_names_to_basics_dict
	trainer.obj_names_to_supers = obj_names_to_supers_dict
	categories_predicted_by_condition = trainer.look_at_validation_set_predictions_given_trained_model()
	
	for k in categories_predicted_by_condition.keys():
		tot = len(categories_predicted_by_condition[k])
		perc_sub = len([item for item in categories_predicted_by_condition[k] if item == 'sub'])/tot
		perc_basic = len([item for item in categories_predicted_by_condition[k] if item == 'basic'])/tot
		perc_super = len([item for item in categories_predicted_by_condition[k] if item == 'super'])/tot
		perc_other = len([item for item in categories_predicted_by_condition[k] if item == 'other'])/tot

		print '\nCondition: {}'.format(k)
		print '	Perc Sub = {}'.format(perc_sub)
		print '	Perc Basic = {}'.format(perc_basic)
		print '	Perc Super = {}'.format(perc_super)
		print '	Perc Other (term that does not apply to target) = {}'.format(perc_other)

		print np.array([perc_sub, perc_basic, perc_super, perc_other])
		trainer.vis.bar(
			X=np.array([perc_sub, perc_basic, perc_super, perc_other]),
			opts=dict(
				rownames=['Sub', 'Basic', 'Super', 'Other'],
				title=model_name.upper() + ' : ' + k))

def wrapper():
	train_set_type = 'random_distractors' # 'random_distractors' or 'uniform_conditions'
	params_desc = 'no_hidden_layer'

	data_path = 'synthetic_data/' # temp synthetic data w/ 3300 training examples
	data_by_num_trials_path = data_path + 'datasets_by_num_trials/' + train_set_type + '/'

	train_data_fname = 'train_set14_495train_trials.JSON'
	validation_data_fname = 'validation_set14_90validation_trials.JSON'

	example_train_data 		= load_json(data_by_num_trials_path + train_data_fname) 
	example_validation_data = load_json(data_by_num_trials_path + validation_data_fname)
	d = load_json(data_path + 'true_lexicon.JSON')
	num_utts = len(d)
	num_objs = len(d['0'])

	utt_info_dict = load_json(data_path + 'utt_inds_to_names.JSON')
	obj_info_dict = load_json(data_path + 'obj_inds_to_names.JSON')
	utt_costs     = load_json(data_path + 'costs_by_utterance.JSON')
	
	# dict whose keys are utterances, vals are truth-vals by obj
	true_lexicon  = load_json(data_path + 'true_lexicon.JSON')
	# reformat to utts x objs array, add jitter
	true_lexicon = np.array([true_lexicon[str(k)] for k in range(num_utts)]) + 10e-06

	# Adam params
	decay = 0.00001
	lr = 0.0001

	model_names = ['ersa', 'nnwc', 'nnwoc']
	for m in model_names:
		results_dir = 'results/' + train_set_type + '/' + params_desc + '/' + train_data_fname.split('_')[1] + '/' + m + '/'
		display_predictions_at_min_loss(m, [], 'tanh', example_train_data, 
		 			example_validation_data, num_utts, num_objs, 
		 			'onehot', utt_info_dict, obj_info_dict, 
		 			decay, lr, True, 
		 			100, utt_costs, 0.1, true_lexicon,
					results_dir,
					load_json(data_path + 'obj_names_to_subs.JSON'), 
					load_json(data_path + 'obj_names_to_basics.JSON'), 
					load_json(data_path + 'obj_names_to_supers.JSON'))

if __name__=='__main__':
	wrapper()
