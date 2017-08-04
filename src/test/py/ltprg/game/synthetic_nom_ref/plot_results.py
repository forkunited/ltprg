from __future__ import division
import os
import fnmatch
import numpy as np
import json
import matplotlib.pyplot as plt

def load_results(results_path):
	eval_results_dict = np.load(results_path + 'DatasetEvaluations.npy').item()
	return eval_results_dict

def load_json(filename):
	with open(filename) as json_data:
	    d = json.load(json_data)
	return d

def pull_min_vals(eval_results_dict):
	# validation set performance
	loss = eval_results_dict['Mean_validationset_loss']
	acc_by_condition = eval_results_dict['Mean_validationset_acc_by_cond']
	kl_by_condition = eval_results_dict['Mean_trainset_dist_from_goldstandard_S1_by_cond']

	ind = np.argmin(loss)
	min_loss = loss[ind]

	# acc by cond at that ind
	acc_sub_nec_at_min = acc_by_condition['sub-nec'][ind]
	acc_basic_suff_at_min = acc_by_condition['basic-suff'][ind]
	acc_super_suff_at_min = acc_by_condition['super-suff'][ind]

	# kl by cond at that ind
	kl_sub_nec_at_min = kl_by_condition['sub-nec'][ind]
	kl_basic_suff_at_min = kl_by_condition['basic-suff'][ind]
	kl_super_suff_at_min = kl_by_condition['super-suff'][ind]

	return (min_loss, acc_sub_nec_at_min, acc_basic_suff_at_min, acc_super_suff_at_min,
			kl_sub_nec_at_min, kl_basic_suff_at_min, kl_super_suff_at_min)

def pull_data_NLL(set_nums, synthetic_datasets_path, validation_prefix):
	train_NLLs_by_set = []
	validation_NLLs_by_set = []
	for s in set_nums:
		set_name = 'set' + str(s)
		for root, dirs, filenames in os.walk(synthetic_datasets_path):
			for f in filenames:
				if f.endswith(".JSON"):
					if fnmatch.fnmatch(f, 'train_' 
										+ set_name + '_*LL*'):
						trainf = = f.split('.')[0]
					if fnmatch.fnmatch(f, validation_prefix + 'validation_' 
										+ set_name + '_*LL*'):
						validationf = f.split('.')[0]
		train_NLLs_by_set.append(load_json(synthetic_datasets_path 
								 + trainf + '.JSON')['LL'] * -1)
		validation_NLLs_by_set.append(load_json(synthetic_datasets_path 
								 + validationf + '.JSON')['LL'] * -1)
	return train_NLLs_by_set, validation_NLLs_by_set


def plot_performance(model_names, train_set_szs, vals_by_model, x_label, 
					 y_label, title, save_name):
	plt.figure()
	lw=2
	for i in range(len(vals_by_model)):
		plt.plot(train_set_szs, np.array(vals_by_model[i]), '-o', lw=lw)
	plt.legend(model_names, loc=1)
	plt.xlabel(x_label)
	plt.ylabel(y_label)
	plt.title(title, wrap=True, fontsize=18, fontweight='bold')
	plt.savefig(save_name + '.png')

def pull_performance_by_set_sz(set_nums, model_name, results_rt):
	loss_by_sz = []
	sub_acc_by_sz = []
	basic_acc_by_sz = []
	super_acc_by_sz = []
	sub_kl_by_sz   = []
	basic_kl_by_sz = []
	super_kl_by_sz = []
	for i in range(len(set_nums)):
		results_path = results_rt + 'set' + str(set_nums[i]) + '/' + model_name + '/'
		print results_path
		d = load_results(results_path)
		min_loss, acc_sub, acc_basic, acc_super, kl_sub, kl_basic, kl_super = pull_min_vals(d)
		loss_by_sz.append(min_loss)
		sub_acc_by_sz.append(acc_sub)
		basic_acc_by_sz.append(acc_basic)
		super_acc_by_sz.append(acc_super)
		sub_kl_by_sz.append(kl_sub)
		basic_kl_by_sz.append(kl_basic)
		super_kl_by_sz.append(kl_super)
	return (loss_by_sz, sub_acc_by_sz, basic_acc_by_sz, super_acc_by_sz, 
		    sub_kl_by_sz, basic_kl_by_sz, super_kl_by_sz)

def wrapper():
	params_desc = 'no_hidden_layer'
	train_set_type = 'random_distractors' # 'uniform_conditions'
	validation_prefix = 'randomdistractors_'
	results_path = 'results/' + train_set_type + '/' + params_desc + '/'
	synthetic_datasets_path = 'synthetic_data/datasets_by_num_trials/' + train_set_type + '/'
	set_nums = range(0, 100)
	batch_sz = 33
	model_names = ['ersa', 'nnwc', 'nnwoc']

	# generative NLL
	trainset_generative_NLL_by_sz, validationset_generative_NLL_by_sz = pull_data_NLL(set_nums, 
													synthetic_datasets_path, validation_prefix)

	set_szs = (np.array(set_nums) + 1) * batch_sz
	print set_szs
	loss_by_model = []
	sub_acc_by_model   = []
	basic_acc_by_model = []
	super_acc_by_model = []
	sub_kl_by_model   = []
	basic_kl_by_model = []
	super_kl_by_model = []
	for i in range(len(model_names)):
		loss, sub_acc, basic_acc, super_acc, sub_kl, basic_kl, super_kl = pull_performance_by_set_sz(
			set_nums, model_names[i], results_path)
		loss_by_model.append(loss)
		sub_acc_by_model.append(sub_acc)
		basic_acc_by_model.append(basic_acc)
		super_acc_by_model.append(super_acc)
		sub_kl_by_model.append(sub_kl)
		basic_kl_by_model.append(basic_kl)
		super_kl_by_model.append(super_kl)

	loss_by_model.append(trainset_generative_NLL_by_sz, validationset_generative_NLL_by_sz)
	loss_legend_labels = model_names
	loss_legend_labels.append('train set generative', 'validation set generative')
	print loss_legend_labels
	plot_performance(loss_legend_labels, set_szs, loss_by_model,
		'Train Set Size (# Trials)', 'Min. Loss on Validation Set', 'Negative Log-Likelihood Loss by Train Set Size',
		results_path + 'loss_by_train_set_sz')

	plot_performance(model_names, set_szs, sub_acc_by_model,
		'Train Set Size (# Trials)', 'Sub-Nec Condition: Accuracy (Validation Set)', 
		'Accuracy (Sub-Nec Cond) by Train Set Size',
		results_path + 'sub_nec_acc_by_train_set_sz')
	plot_performance(model_names, set_szs, basic_acc_by_model,
		'Train Set Size (# Trials)', 'Basic-Suff Condition: Accuracy (Validation Set)', 
		'Accuracy (Basic-Suff Cond) by Train Set Size',
		results_path + 'basic_suff_acc_by_train_set_sz')
	plot_performance(model_names, set_szs, super_acc_by_model,
		'Train Set Size (# Trials)', 'Super-Suff Condition: Accuracy (Validation Set)', 
		'Accuracy (Super-Suff Cond) by Train Set Size',
		results_path + 'super_suff_acc_by_train_set_sz')

	plot_performance(model_names, set_szs, sub_kl_by_model,
		'Train Set Size (# Trials)', 'KL Div, Sub-Nec Condition (Validation Set)', 
		'Sub-Nec Condition: KL Div from Goldstandard S1 by Train Set Size',
		results_path + 'sub_nec_kl_by_train_set_sz')
	plot_performance(model_names, set_szs, basic_kl_by_model,
		'Train Set Size (# Trials)', 'KL Div, Basic-Suff Condition (Validation Set)', 
		'Basic-Suff Condition: KL Div from Goldstandard S1 by Train Set Size',
		results_path + 'basic_suff_kl_by_train_set_sz')
	plot_performance(model_names, set_szs, super_kl_by_model,
		'Train Set Size (# Trials)', 'KL Div, Super-Suff Condition (Validation Set)', 
		'Super-Suff Condition: KL Div from Goldstandard S1 by Train Set Size',
		results_path + 'super_suff_kl_by_train_set_sz')

if __name__=='__main__':
	wrapper()
	
