from __future__ import division
import os
import fnmatch
import copy
import numpy as np
import json
import matplotlib.pyplot as plt

def load_json(filename):
	with open(filename) as json_data:
	    d = json.load(json_data)
	return d

def min_validation_loss_epoch(split_name, eval_results_dict):
	loss = eval_results_dict['Mean_'+ split_name + 'set_loss']
	ind = np.argmin(loss)
	return ind

def init_results_by_set_dict(num_sets):
	d = dict()
	d['loss'] = np.zeros(num_sets)
	d['acc_sub_nec']    = np.zeros(num_sets)
	d['acc_basic_suff'] = np.zeros(num_sets)
	d['acc_super_suff'] = np.zeros(num_sets)
	d['kl_sub_nec']    = np.zeros(num_sets)
	d['kl_basic_suff'] = np.zeros(num_sets)
	d['kl_super_suff'] = np.zeros(num_sets)
	return d

def init_results_by_split_dict(num_sets):
	d = dict()
	d['train'] = init_results_by_set_dict(num_sets)
	d['validation'] = init_results_by_set_dict(num_sets)
	d['test'] = init_results_by_set_dict(num_sets)
	return d

def load_generative_loss_by_dataset(synthetic_datasets_path, set_nums, test_split_name):
	train_loss_by_set = np.zeros(len(set_nums))
	validation_loss_by_set = np.zeros(len(set_nums))
	test_loss_by_set = np.zeros(len(set_nums))
	for i, s in enumerate(set_nums):
		set_name = 'set' + str(s)
		for root, dirs, filenames in os.walk(synthetic_datasets_path):
			for f in filenames:
				if f.endswith(".JSON"):
					if fnmatch.fnmatch(f, 'train_' + set_name + '_*LL*'):
						trainf = f.split('.')[0]
					if fnmatch.fnmatch(f, 'validation_' + set_name + '_*LL*'):
						validationf = f.split('.')[0]
					if fnmatch.fnmatch(f, test_split_name + '_' 
										+ set_name + '_*LL*'):
						testf = f.split('.')[0]
		train_loss_by_set[i] = load_json(synthetic_datasets_path 
 								 		+ trainf + '.JSON')['LL'] * -1
		validation_loss_by_set[i] = load_json(synthetic_datasets_path 
 								 		+ validationf + '.JSON')['LL'] * -1
		test_loss_by_set[i] = load_json(synthetic_datasets_path 
 								 		+ testf + '.JSON')['LL'] * -1
	return train_loss_by_set, validation_loss_by_set, test_loss_by_set

class DataPlotter(object):
	def __init__(self, results_path, model_names, set_nums, generative_losses):
		self.results_path = results_path
		self.model_names = model_names
		self.set_nums = set_nums
		self.generative_loss_by_set = generative_losses
		self.compile_results()
		self.plot_all()

	def init_results_store(self):
		self.results_by_model_by_split_by_set = dict()
		for m in self.model_names:
			self.results_by_model_by_split_by_set[m] = init_results_by_split_dict(
				len(self.set_nums))

	def load_results_by_model_set(self, model_name, set_num):
		print model_name
		set_name = 'set' + str(set_num)
		print set_name

		test_fname_prefix = self.results_path + set_name + '/' 			
		if model_name == 'nnwoc_w_rsa_on_top':
			test_fname_prefix = test_fname_prefix + 'nnwoc/rsa_added_for_test_set/'
		else:
			test_fname_prefix = test_fname_prefix + model_name + '/'
		test_results_dict = np.load(test_fname_prefix
									+ 'DatasetEvaluations' 
						   			+ '_AtPeak_TestSet.npy').item()
		self.pull_vals_at_stopping_pt(model_name, 'test', set_num, 
								  test_results_dict, 
								  'NaN')
			
		if model_name != 'nnwoc_w_rsa_on_top':
			train_validation_results_dict = np.load(self.results_path 
												+ set_name + '/' + model_name + '/'
												+ 'DatasetEvaluations' 
												+ '.npy').item()
			
			# pt of min loss on validation set
			stopping_pt = int(min_validation_loss_epoch('validation',
										train_validation_results_dict))

			self.pull_vals_at_stopping_pt(model_name, 'train', set_num, 
										  train_validation_results_dict, 
										  stopping_pt)
			self.pull_vals_at_stopping_pt(model_name, 'validation', set_num, 
										  train_validation_results_dict, 
										  stopping_pt)

		
	def pull_vals_at_stopping_pt(self, model_name, split_name, set_num, 
								results_dict, stop_pt):
		loss = results_dict['Mean_' + split_name + 'set_loss']
		acc_by_condition = results_dict['Mean_'+ split_name + 'set_acc_by_cond']
		acc_sub_nec = acc_by_condition['sub-nec']
		acc_basic_suff = acc_by_condition['basic-suff']
		acc_super_suff = acc_by_condition['super-suff']
		kl_by_condition = results_dict['Mean_'+ split_name 
										+ 'set_dist_from_goldstandard_S1_by_cond']
		kl_sub_nec = kl_by_condition['sub-nec']
		kl_basic_suff = kl_by_condition['basic-suff']
		kl_super_suff = kl_by_condition['super-suff']

		# clip to stop pt as required
		if split_name == 'train' or split_name == 'validation':
			loss = loss[stop_pt]
			acc_sub_nec    = acc_sub_nec[stop_pt]
			acc_basic_suff = acc_basic_suff[stop_pt]
			acc_super_suff = acc_super_suff[stop_pt]
			kl_sub_nec 	   = kl_sub_nec[stop_pt]
			kl_basic_suff  = kl_basic_suff[stop_pt]
			kl_super_suff  = kl_super_suff[stop_pt]

		self.results_by_model_by_split_by_set[model_name][split_name]['loss'][set_num] = loss
		self.results_by_model_by_split_by_set[model_name][split_name]['acc_sub_nec'][set_num] = acc_sub_nec
		self.results_by_model_by_split_by_set[model_name][split_name]['acc_basic_suff'][set_num] = acc_basic_suff
		self.results_by_model_by_split_by_set[model_name][split_name]['acc_super_suff'][set_num] = acc_super_suff
		self.results_by_model_by_split_by_set[model_name][split_name]['kl_sub_nec'][set_num] = kl_sub_nec
		self.results_by_model_by_split_by_set[model_name][split_name]['kl_basic_suff'][set_num] = kl_basic_suff
		self.results_by_model_by_split_by_set[model_name][split_name]['kl_super_suff'][set_num] = kl_super_suff

	def plot_performance(self, split_name, result_key, x_label, y_label, 
						title, save_name):
		plt.figure()
		lw = 2
		x = np.array(self.set_nums)

		legend_items = []
		for model_name in self.model_names:
			if (split_name != 'test' and model_name != 'nnwoc_w_rsa_on_top') or split_name == 'test':
				y = self.results_by_model_by_split_by_set[model_name][split_name][result_key]
				plt.plot(x, y, '-o', lw=lw)
				legend_items.append(model_name)

		if result_key == 'loss':
			print split_name
			plt.plot(x, self.generative_loss_by_set[split_name], '-o', lw=lw)
			legend_items.append('generative')
			print legend_items

		plt.legend(legend_items, loc=1)
		plt.xlabel(x_label)
		plt.ylabel(y_label)
		plt.title(title, wrap=True, fontsize=18, fontweight='bold')
		plt.savefig(save_name + '.png')

	def compile_results(self):
		self.init_results_store()
		for model_name in self.model_names:
			for set_num in self.set_nums:
				self.load_results_by_model_set(model_name, set_num)

	def plot_all(self):
		for split_name in self.results_by_model_by_split_by_set[self.model_names[0]].keys():
			print '\nsaving ' + split_name + ' figures'
			self.plot_performance(split_name, 'loss', 'Dataset', 
								 'Loss', split_name.upper() + ': NLLoss by Dataset',
								 self.results_path + 'Figures/' + split_name 
								 + '_loss_by_set')
			self.plot_performance(split_name, 'acc_sub_nec', 'Dataset', 
								 'Accuracy', 
								 split_name.upper() + ': Acc (Sub-Nec Cond) by Dataset',
								 self.results_path + 'Figures/' + split_name 
								 + '_acc_sub_nec_by_set')
			self.plot_performance(split_name, 'acc_basic_suff', 'Dataset', 
								 'Accuracy', 
								 split_name.upper() + ': Acc (Basic-Suff Cond) by Dataset',
								 self.results_path + 'Figures/' + split_name  
								 + '_acc_basic_suff_by_set')
			self.plot_performance(split_name, 'acc_super_suff', 'Dataset', 
								 'Accuracy', 
								 split_name.upper() + ': Acc (Super-Suff Cond) by Dataset',
								 self.results_path + 'Figures/' + split_name  
								 + '_acc_super_suff_by_set')
			self.plot_performance(split_name, 'kl_sub_nec', 'Dataset', 
								 'KL Divergence', 
								 split_name.upper() + ': KL Div (Sub-Nec Cond) from Goldstandard S1 by Dataset',
								 self.results_path + 'Figures/' + split_name  
								 + '_kl_sub_nec_by_set')
			self.plot_performance(split_name, 'kl_basic_suff', 'Dataset', 
								 'KL Divergence', 
								 split_name.upper() + ': KL Div (Basic-Suff Cond) from Goldstandard S1 by Dataset',
								 self.results_path + 'Figures/' + split_name 
								 + '_kl_basic_suff_by_set')
			self.plot_performance(split_name, 'kl_super_suff', 'Dataset', 
								 'KL Divergence', 
								 split_name.upper() + ': KL Div (Super-Suff Cond) from Goldstandard S1 by Dataset',
								 self.results_path + 'Figures/' + split_name  
								 + '_kl_super_suff_by_set')
				
def wrapper():
	model_desc = 'no_hidden_layer'
	train_set_type = 'random_distractors' # 'uniform_conditions'
	results_path = 'results/' + train_set_type + '/' + model_desc + '/'
	synthetic_data_path = 'synthetic_data/datasets_by_num_trials/' + train_set_type + '/'

	model_names = ['ersa', 'nnwc', 'nnwoc'] #, 'nnwoc_w_rsa_on_top']
	sets = range(0, 100)

	# collect generative losses by split
	train_loss_gen, validation_loss_gen, test_loss_gen = load_generative_loss_by_dataset(
		synthetic_data_path, sets, 'randomdistractors_validation')
	gen_losses = dict()
	gen_losses['train'] = train_loss_gen
	gen_losses['validation'] = validation_loss_gen
	gen_losses['test'] = test_loss_gen

	d = DataPlotter(results_path, model_names, sets, gen_losses)

if __name__=='__main__':
	wrapper()