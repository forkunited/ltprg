from __future__ import division
import numpy as np
import matplotlib.pyplot as plt

def load_results(results_path):
	eval_results_dict = np.load(results_path + 'DatasetEvaluations.npy').item()
	return eval_results_dict

def pull_min_vals(eval_results_dict):
	# validation set performance
	loss = eval_results_dict['Mean_validationset_loss']
	acc_by_condition = eval_results_dict['Mean_validationset_acc_by_cond']
	ind = np.argmin(loss)
	min_loss = loss[ind]
	acc_sub_nec_at_min = acc_by_condition['sub-nec'][ind]
	acc_basic_suff_at_min = acc_by_condition['basic-suff'][ind]
	acc_super_suff_at_min = acc_by_condition['super-suff'][ind]
	return min_loss, acc_sub_nec_at_min, acc_basic_suff_at_min, acc_super_suff_at_min

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
	for i in range(len(set_nums)):
		results_path = results_rt + 'set' + str(set_nums[i]) + '/' + model_name + '/'
		print results_path
		d = load_results(results_path)
		min_loss, acc_sub, acc_basic, acc_super = pull_min_vals(d)
		loss_by_sz.append(min_loss)
		sub_acc_by_sz.append(acc_sub)
		basic_acc_by_sz.append(acc_basic)
		super_acc_by_sz.append(acc_super)
	return loss_by_sz, sub_acc_by_sz, basic_acc_by_sz, super_acc_by_sz

def wrapper():
	results_path = 'results/random_distractors/no_hidden_layer/'
	set_nums = range(0, 100)
	batch_sz = 33
	model_names = ['ersa', 'nnwc', 'nnwoc']

	set_szs = (np.array(set_nums) + 1) * batch_sz
	print set_szs
	loss_by_model = []
	sub_acc_by_model   = []
	basic_acc_by_model = []
	super_acc_by_model = []
	for i in range(len(model_names)):
		loss, sub_acc, basic_acc, super_acc = pull_performance_by_set_sz(set_nums, 
									model_names[i], results_path)
		loss_by_model.append(loss)
		sub_acc_by_model.append(sub_acc)
		basic_acc_by_model.append(basic_acc)
		super_acc_by_model.append(super_acc)
	plot_performance(model_names, set_szs, loss_by_model,
		'Train Set Size', 'Min. Loss on Validation Set', 'Performance by Train Set Size',
		results_path + 'loss_by_train_set_sz')
	plot_performance(model_names, set_szs, sub_acc_by_model,
		'Train Set Size', 'Accuracy, Sub-Nec Condition (Validation Set)', 
		'Accuracy (Sub-Nec Cond) by Train Set Size',
		results_path + 'sub_nec_acc_by_train_set_sz')
	plot_performance(model_names, set_szs, basic_acc_by_model,
		'Train Set Size', 'Accuracy, Basic-Suff Condition (Validation Set)', 
		'Accuracy (Basic-Suff Cond) by Train Set Size',
		results_path + 'basic_suff_acc_by_train_set_sz')
	plot_performance(model_names, set_szs, super_acc_by_model,
		'Train Set Size', 'Accuracy, Super-Suff Condition (Validation Set)', 
		'Accuracy (Super-Suff Cond) by Train Set Size',
		results_path + 'super_suff_acc_by_train_set_sz')

if __name__=='__main__':
	wrapper()
	
