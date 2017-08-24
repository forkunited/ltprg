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

def construct_cmd(model_name, hidden_layer_szs, hiddens_nonlinearity,
				  train_data_fname, validation_data_fname, 
				  test_data_fname,
				  weight_decay, lr, 
				  alpha, cost_weight, save_path):
	cmd = """python parse_and_run.py {} {} {} {} {} {} {} {} {} {} {}""".format(
			model_name, hidden_layer_szs, hiddens_nonlinearity,
		 	train_data_fname, validation_data_fname, 
		 	test_data_fname,
			weight_decay, lr, alpha, cost_weight, save_path)
	return cmd

def run_models():
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

if __name__=='__main__':
	run_models()
