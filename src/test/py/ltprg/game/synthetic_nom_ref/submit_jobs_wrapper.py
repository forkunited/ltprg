#!/usr/bin/env python
"""
File: prep_and_submit_jobs.py
"""
import os
from subprocess import call
import fnmatch
import numpy as np
import json
from model_trainer import train_model, load_json
from itertools import product, izip

def construct_cmd(model_name, hidden_layer_szs, hiddens_nonlinearity,
				  train_data_fname, validation_data_fname, 
				  test_data_fname,
				  weight_decay, lr, 
				  alpha, cost_weight, save_path):
	hidden_layer_szs_str = '_'.join([str(x) for x in hidden_layer_szs])
	cmd = """python parse_and_run.py {} {} {} {} {} {} {} {} {} {} {}""".format(
			model_name, hidden_layer_szs_str, hiddens_nonlinearity,
		 	train_data_fname, validation_data_fname, 
		 	test_data_fname,
			weight_decay, lr, alpha, cost_weight, save_path)
	return cmd

def run_models_over_data_sets():
	train_type = 'random_distractors'
	validation_prefix = 'randomdistractors_' # stopping defined on 
	#random distractors validation set
	data_rt = 'synthetic_data/'
	synthetic_datasets_dir = data_rt + 'datasets_by_num_trials/' + train_type + '/'
	results_rt = 'results/' + train_type + '/' + 'no_hidden_layer/'

	# known generative
	rsa_alpha = 100
	rsa_cost_weight = 0.1

	hidden_layer_szs = []

	# params
	model_names = ['fasm_ersa_cts'] #['fasm_ersa_cts', 'fasm_nnwc_cts', 'fasm_nnwoc_cts']
	decays = [0.00001] # Adam params
	learning_rates = [0.0001]
	hidden_layer_szs = [[200, 200]]
	# hidden_layer_szs = [[100], [100, 100], [100, 100, 100], [200], [200, 200], [200, 200, 200]]

	set_lst = range(0, 1)
	for set_num in set_lst:
		set_name = 'set' + str(set_num)
		print '\n' + set_name + '\n'
		for root, dirs, filenames in os.walk(synthetic_datasets_dir):
		    for f in filenames:
		    	if f.endswith(".JSON"):
		    		if fnmatch.fnmatch(f, 'train_set' + str(set_num) + '_*trials*'):
			    		trainf = f.split('.')[0] # remove '.JSON'
			    	if fnmatch.fnmatch(f, validation_prefix + 'validation_set' + str(set_num) + '_*trials*'):
			    		validationf = f.split('.')[0]
			    	if fnmatch.fnmatch(f, 'validation_set' + str(set_num) + '_*trials*'):
			    		testf = f.split('.')[0]
		train_data_fname = synthetic_datasets_dir + trainf + '.JSON'
		validation_data_fname = synthetic_datasets_dir + validationf + '.JSON'
		test_data_fname = synthetic_datasets_dir + testf + '.JSON'

		print 'Train data:'
		print train_data_fname
		print 'Validation data:'
		print validation_data_fname
		print 'Test data:'
		print test_data_fname

		for m in model_names:
			# create results dir
			results_dir = results_rt + set_name + '/' + m + '/'
			if os.path.isdir(results_dir) == False:
				os.makedirs(results_dir)

			cmd = construct_cmd(m, hidden_layer_szs, 'tanh', train_data_fname, 
			 					validation_data_fname, test_data_fname,
			 					0.00001, 0.0001, rsa_alpha, rsa_cost_weight,
								results_dir)

			call(["sbatch", "--export=cmd={}".format(cmd), "submit_sbatch.sh"])


def run_grid_search(search_params):
	def gen_param_permutations(search_params):
	    return (dict(izip(search_params, x)) for x in product(*search_params.itervalues()))

	for param_permutation in gen_param_permutations(search_params):
		# Grid Search Params
		rsa_alpha = param_permutation['rsa_alpha']
		rsa_cost_weight = param_permutation['rsa_cost_weight']
		model_name = param_permutation['model_name']
		decay = param_permutation['decay']
		lr = param_permutation['lr']
		hidden_layer_szs = param_permutation['hidden_layer_szs']
		hiddens_nonlinearity = param_permutation['hiddens_nonlinearity']

		# File Paths
		train_type = 'uniform_conditions'
		validation_prefix = 'validation_set'
		data_rt = 'synthetic_data/'
		synthetic_datasets_dir = data_rt + 'datasets_by_num_trials/' + train_type + '/'
		results_dir = '/'.join(
			['results', train_type, '_'.join([str(x) for x in hidden_layer_szs]),
			 'set_99', model_name, str(rsa_alpha) + "_alpha", str(rsa_cost_weight) + "_cost_weight", 
			 str(decay) + "_decay", str(lr) + "_lr"]
		)

		# Data
		trainf = synthetic_datasets_dir + 'train_set99_3300train_trials.JSON'
		validationf = synthetic_datasets_dir + 'validation_set99_600validation_trials.JSON'
		testf = synthetic_datasets_dir + 'test_set99_600test_trials.JSON'

		# Results
		if os.path.isdir(results_dir) == False:
			os.makedirs(results_dir)		

		cmd = construct_cmd(model_name, hidden_layer_szs, hiddens_nonlinearity, trainf, 
		 					validationf, testf,
		 					lr, decay, rsa_alpha, rsa_cost_weight,
							results_dir)
		call(["sbatch", "--export=cmd={}".format(cmd), "submit_sbatch.sh"])	 	


if __name__=='__main__':
	params = {
		'rsa_alpha': [.5, 1, 10, 100],
		'rsa_cost_weight': [.1, .01],
		'model_name': [
			'fasm_ersa', 'fasm_nnwc', 'fasm_nnwoc', 'fasm_ersa_cts', 'fasm_nnwc_cts', 'fasm_nnwoc_cts'
		],
		'decay':[0.00001],
		'lr':[0.001, .0001],
		'hidden_layer_szs':[[100, 100]],
		'hiddens_nonlinearity': ['relu']
	}
        run_grid_search(params)
