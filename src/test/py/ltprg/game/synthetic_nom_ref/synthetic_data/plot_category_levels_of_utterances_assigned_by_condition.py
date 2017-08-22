from __future__ import division
import numpy as np
import json
import visdom

print '\n\nTo view plots of dataset characteristics, enter `python -m visdom.server`' 
print 'in another terminal window. Then navigate to http://localhost.com:8097'
print 'in your browser\n'
raw_input("Press Enter to continue...")
vis = visdom.Visdom()

def load_json(filename):
	with open(filename) as json_data:
	    d = json.load(json_data)
	return d

def count_occurrences_each_category(lst):
	counts_by_category = []
	counts_by_category.append(len([item for item in lst if item == 'sub']))
	counts_by_category.append(len([item for item in lst if item == 'basic']))
	counts_by_category.append(len([item for item in lst if item == 'super']))
	counts_by_category.append(len([item for item in lst if item == 'other']))
	labels = ['sub', 'basic', 'super', 'other']
	return np.array(counts_by_category), labels

def plot_category_use_by_condition(counts_by_category, category_labels, title):
	proportions_by_category = counts_by_category/counts_by_category.sum()

	vis.bar(X=proportions_by_category,
		opts=dict(
			rownames=category_labels,
			xlabel='Category Level of Assigned Utterance',
			ylabel='% Trials',
			title=title)
		)

def collect_term_type_used_by_condition():
	# load obj, utt info used to generate datasets
	obj_inds_to_names = load_json('obj_inds_to_names.JSON')
	utt_inds_to_names = load_json('utt_inds_to_names.JSON')

	obj_names_to_subs   = load_json('obj_names_to_subs.JSON')
	obj_names_to_basics = load_json('obj_names_to_basics.JSON')
	obj_names_to_supers = load_json('obj_names_to_supers.JSON')

	# TODO: loop over dir
	train_type = 'random_distractors' #'uniform_conditions'
	print 'Train Type: {}'.format(train_type)
	data_dir = 'datasets_by_num_trials/' + train_type + '/'
	# dataset_name = 'train_set99_3300train_trials'
	# dataset_name = 'validation_set99_600validation_trials'
	# dataset_name = 'validation_set14_90validation_trials'
	# dataset_name = 'train_set6_231train_trials'
	dataset_name = 'randomdistractors_validation_set6_42validation_trials'
	
	dataset = load_json(data_dir + dataset_name + '.JSON')

	# collect category levels of terms used, by condition
	sub_nec    = [] # category used for this condition
	basic_suff = []
	super_suff = []
	other 	   = [] # term doesn't apply to target
	across_dataset = []
	for trial in dataset:
		target_name    = obj_inds_to_names[str(trial['target_ind'])]
		utterance_name = utt_inds_to_names[str(trial['utterance'])]

		# determine which category level term was used
		# print '\nsub for ' + target_name + ' = ' + obj_names_to_subs[target_name]
		# print 'basic for ' + target_name + ' = ' + obj_names_to_basics[target_name]
		# print 'super for ' + target_name + ' = ' + obj_names_to_supers[target_name]
		if obj_names_to_subs[target_name] == utterance_name:
			category = 'sub'
		elif obj_names_to_basics[target_name] == utterance_name:
			category = 'basic'
		elif obj_names_to_supers[target_name] == utterance_name:
			category = 'super'
		else:
			category = 'other' 

		across_dataset.append(category)

		# add to that condition
		if trial['condition'] == 'sub-nec':
			sub_nec.append(category)
		elif trial['condition'] == 'basic-suff':
			basic_suff.append(category)
		elif trial['condition'] == 'super-suff':
			super_suff.append(category)

	# counts by category, by condition
	counts_by_cat_sub_nec, cat_labels = count_occurrences_each_category(sub_nec)
	counts_by_cat_basic_suff, _ = count_occurrences_each_category(basic_suff)
	counts_by_cat_super_suff, _ = count_occurrences_each_category(super_suff)

	counts_across_dataset, _ = count_occurrences_each_category(across_dataset)

	# plot by condition
	plot_category_use_by_condition(counts_by_cat_sub_nec, cat_labels,
									'Sub-Nec (' 
									+ dataset_name + ')')
	plot_category_use_by_condition(counts_by_cat_basic_suff, cat_labels,
									'Basic-Suff (' 
									+ dataset_name + ')')
	plot_category_use_by_condition(counts_by_cat_super_suff, cat_labels,
									'Super-Suff (' 
									+ dataset_name + ')')
	plot_category_use_by_condition(counts_across_dataset, cat_labels,
									'Whole Dataset ('
									+ dataset_name + ')')

if __name__ == '__main__':
	collect_term_type_used_by_condition()
