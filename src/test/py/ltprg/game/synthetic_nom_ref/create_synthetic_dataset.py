from __future__ import division
import copy
import numpy as np
import csv
import json
import random
import matplotlib.pyplot as plt
import visdom
import torch
import torch.nn as nn
import torch.cuda as cuda
from torch.autograd import Variable
from rsa import uniform_prior, model_literal_listener

# Creates nested synthetic datasets of increasing size. Each dataset consists of 
#	a train set, and held-out validation and test sets (trial types are unique across
#	train/valid/test sets). 
#
# Each set is saved as its own JSON file, and is a list of
#	dictionaries, where each dictionary is a trial with 
#	{"target_ind", "alt1_ind", "alt2_ind", "condition", "utterance", "log-prob of utterance"}. 
#	Utterances are sampled from an RSA level-1 speaker distribution given the 
#	ground-truth (binary) lexicon.
#	
# Two train set options: train set can contain
#	(1) uniform distribution of conditions, with each object appearing at least once
#		per batch as a distractor ('uniform-conditions')
#	(2) randomly drawn distractors (no constraints enforced; 'random-distractors')
# 
# If 'random-distractors' option is used, a second held-out validation set is also 
#	created. This set has the same random distractors as the train set, and is 
#	used for stopping.
#

random.seed(3)

# Util
def read_csv_field(filename, col_ind, fieldname):
	out = []
	with open(filename, 'rb') as csvfile:
		reader = csv.reader(csvfile, delimiter=',')
		for row in reader:
			if not(row[col_ind] == fieldname):
				out.append(row[col_ind])
	return out

def flatten_lst_of_lsts(lst_of_lsts):
	return [val for sublst in lst_of_lsts for val in sublst]

def num_unique(lst):
	return len(list(set(lst)))

def get_counts(lst):
	unique_items = sorted(list(set(lst)))
	counts = np.zeros(len(unique_items))
	for i in range(len(unique_items)):
		for j in range(len(lst)):
			if unique_items[i] == lst[j]:
				counts[i]+=1.
	return unique_items, counts

def trial_to_trialtype(trial):
	# trial[0] is target ind, trial[1] is alt1 ind, trial[2] is alt2 ind
	# we don't want to consider condition or assigned utt here
	return [trial[0]] + sorted([trial[1], trial[2]])

def get_unique_type_counts(trials):
	# trials is list of trials
	# (trial type consists
	#	of trials with same target and distractors, but distractors can 
	#	be in either position)
	unique_types = dict()
	for i, t in enumerate(trials):
		key = str(trial_to_trialtype(t))
		unique_types.setdefault(key, [])
		unique_types[key].append(i) # record inds of occurrences

	# convert to counts
	for k in unique_types.keys():
		unique_types[k] = len(unique_types[k])
	return unique_types

def create_category_lookup(obj_names, cat_names):
	# create dict whose keys are cats, vals are objs belonging to that cat,
	# e.g. 'plant': ['tulip', 'rose', 'daisy', 'sunflower']
	unique_cats = list(set(cat_names))
	d = dict()
	for cat in unique_cats:
		members = []
		for i in range(len(cat_names)):
			if cat_names[i] == cat:
				members.append(obj_names[i])
		d[cat] = members

	return d

def make_x_all_objects(subset_counts, subset_labels, all_obj_names):
	all_obj_counts = np.zeros(len(all_obj_names))
	for i in range(len(all_obj_names)):
		for j in range(len(subset_labels)):
			if all_obj_names[i] == subset_labels[j]:
				all_obj_counts[i] += subset_counts[j]
	return all_obj_names, all_obj_counts

def visualize_batch(batch, title, obj_names, utt_names, utt_inds_to_names):
	# batch is list of lists [[target, alt1, alt2, cond], [], ...]
	# (so pre-reformat); could also be whole dataset

	print 'Plotting ' + title
	print '\n# Objects = {}'.format(len(obj_names))

	# create list of objects used as targets
	targets = [t[0] for t in batch]
	targ_labels, targ_counts = get_counts(targets)
	print '# Objects used as targets = {}'.format(len(targ_labels))
	targ_labels, targ_counts = make_x_all_objects(targ_counts, targ_labels, obj_names)

	# create list of objects used as distractors
	distractors = [t[1] for t in batch] + [t[2] for t in batch]
	distractor_labels, distractor_counts = get_counts(distractors)
	print '# Objects used as distractors = {}\n'.format(len(distractor_labels))
	distractor_labels, distractor_counts = make_x_all_objects(
		                                   distractor_counts, distractor_labels, obj_names)

	# create list of conditions
	conditions = [t[3] for t in batch]
	condition_labels, condition_counts = get_counts(conditions)

	# create hist of utterances assigned
	utterances = [utt_inds_to_names[t[4]] for t in batch]
	utterance_labels, utterance_counts = get_counts(utterances)
	utterance_labels, utterance_counts = make_x_all_objects(utterance_counts, 
												utterance_labels, utt_names)

	# histogram of how often trials are used
	trial_type_dict = get_unique_type_counts(batch)
	trial_type_labels = trial_type_dict.keys()
	trial_type_counts = trial_type_dict.values()

	targ_plot_title       = title + ': Targets'
	distractor_plot_title = title + ': Distractors'
	condition_plot_title  = title + ': Conditions'
	utterance_plot_title  = title + ': Assigned Utterances'
	trial_plot_title      = title + ': Trial Types'
	hist_title 			  = title + ': Histogram of Trial Types'

	num_bins = int(np.ceil(num_unique(trial_type_counts)/2))

	vis.bar(X=targ_counts,
		opts=dict(
			rownames=targ_labels,
			title=targ_plot_title)
		)

	vis.bar(X=distractor_counts,
		opts=dict(
			rownames=distractor_labels,
			title=distractor_plot_title)
		)

	vis.bar(X=condition_counts,
		opts=dict(
			rownames=condition_labels,
			title=condition_plot_title)
		)

	vis.bar(X=utterance_counts,
		opts=dict(
			rownames=utterance_labels,
			title=utterance_plot_title)
		)

	mean_reps = np.mean(trial_type_counts)
	std_reps  = np.std(trial_type_counts)
	min_reps  = np.min(trial_type_counts)
	max_reps  = np.max(trial_type_counts)
	vis.text('\n' + title
			 + ': Reps per trial type. Mean = ' + str(mean_reps) 
			 + '(std = ' + str(std_reps) + '), '
			 + 'Range ' + str(min_reps) + '-' + str(max_reps))

def make_lexicon_heatmap(lexicon, obj_inds_to_names, utt_inds_to_names):
	print sorted(obj_inds_to_names.keys())
	print sorted(utt_inds_to_names.keys())
	obj_labels = [obj_inds_to_names[ind] for ind in sorted(
				  obj_inds_to_names.keys())]
	utt_labels = [utt_inds_to_names[ind] for ind in sorted(
				  utt_inds_to_names.keys())]
	vis.heatmap(X=lexicon,
		opts=dict(
			columnnames=obj_labels,
			rownames=utt_labels,
			xlabel='Objects',    # corresponds to colnames
			ylabel='Utterances', # corresponds to rownames
			title='Deterministic Lexicon'
		)
	)

# Create dataset
class DatasetMaker(object):
	def __init__(self, train_type):
		# INPUTS:
		#	train_type ('random_distractors' or 'uniform_conditions')
		self.train_type = train_type

		# print options
		self.visualize_train_batches = True
		self.print_trial_details     = False

		# parameters for RSA utterance assignment
		self.alpha       = 100
		self.cost_weight = 0.1
		print '\nAssigning utterances from RSA Speaker 1 distribution with'
		print 'alpha = {}, cost weight = {}'.format(self.alpha, self.cost_weight)

		synthetic_dir = 'synthetic_data/'
		self.dataset_save_path = synthetic_dir + 'datasets_by_num_trials/' + self.train_type + '/'
		self.objects_filename = synthetic_dir + 'objects.csv'
		self.num_trials_per_cond_per_batch = 11 # 33 trials/batch (11 per cond)
		self.num_conditions = 3 # sub-nec, basic-suff, super-suff
		self.batch_sz = self.num_trials_per_cond_per_batch * self.num_conditions

		# couple contstraints:
		#	- train sets must be in multiples of 
		# 	  self.num_trials_per_cond_per_batch * self.num_conditions
		#	- test, validation sets must be multiples of 3
		#	- so making test, validation sets 13.33% of total 
		#	  (closest possible to 10%)
		self.train_set_szs      = np.linspace(33, 3300, num=100)
		self.test_set_szs       = np.linspace(6, 600, num=100)
		self.validation_set_szs = np.linspace(6, 600, num=100)
	
		self.load_objects()
		self.create_object_lookups()
		self.create_utterance_set()
		self.create_lexicon()
		make_lexicon_heatmap(self.lexicon, self.obj_inds_to_obj_names, 
							 self.utt_inds_to_utt_names)
		self.save_lexicon_heatmap(synthetic_dir + 'true_lexicon.png')
		self.save_lexicon_as_json_dict('', synthetic_dir + 'true_lexicon.json')
		self.create_utterance_cost_dict()
		self.save_as_json(self.utt_inds_to_utt_names, synthetic_dir 
						  + 'utt_inds_to_names.JSON')
		self.save_as_json(self.obj_inds_to_obj_names, synthetic_dir 
						  + 'obj_inds_to_names.JSON')
		self.save_as_json(self.obj_names_to_subs, synthetic_dir 
						  + 'obj_names_to_subs.JSON')
		self.save_as_json(self.obj_names_to_basics, synthetic_dir 
						  + 'obj_names_to_basics.JSON')
		self.save_as_json(self.obj_names_to_supers, synthetic_dir 
						  + 'obj_names_to_supers.JSON')
		self.save_dataset_characteristics()
		self.generate_dataset()
		self.reformat_dataset_and_save()

        # dtype
        if cuda.is_available():
            self.dtype = torch.cuda.FloatTensor
        else
            self.dtype = torch.FloatTensor

	def save_as_json(self, d, savename):
		with open(savename, 'w') as outfile:
			json.dump(d, outfile)
		# print 'json saved'

	def save_dataset_characteristics(self):
		d = {'train_set_characteristics': self.train_type,
			 'alpha': self.alpha, 'cost_weight': self.cost_weight,
			 'num_conditions': self.num_conditions, 'batch_sz': self.batch_sz}
		self.save_as_json(d, self.dataset_save_path + 'params_used.JSON')

	def load_objects(self):
		self.obj_inds   = map(int, read_csv_field(self.objects_filename, 
			                  0, 'Obj_ind'))
		self.obj_names  = read_csv_field(self.objects_filename, 1, 'Obj_name')
		self.sub_cats   = read_csv_field(self.objects_filename, 2, 'Subordinate')
		self.basic_cats = read_csv_field(self.objects_filename, 3, 'Basic')
		self.super_cats = read_csv_field(self.objects_filename, 4, 'Super')
		self.num_objects = len(self.obj_names)

	def create_object_lookups(self):
		# package up object info
		self.subs_to_obj_names   = create_category_lookup(self.obj_names, 
														  self.sub_cats)
		self.basics_to_obj_names = create_category_lookup(self.obj_names, 
														  self.basic_cats)
		self.supers_to_obj_names = create_category_lookup(self.obj_names, 
														  self.super_cats)
		
		self.obj_inds_to_obj_names = dict(zip(self.obj_inds, self.obj_names))
		self.obj_names_to_obj_inds = dict(zip(self.obj_names, self.obj_inds))
		self.obj_names_to_subs     = dict(zip(self.obj_names, self.sub_cats))	
		self.obj_names_to_basics   = dict(zip(self.obj_names, 
											  self.basic_cats))	
		self.obj_names_to_supers   = dict(zip(self.obj_names, 
											  self.super_cats))

		print '\n# Sub categories   = {}'.format(num_unique(self.sub_cats))
		print '\n# Basic categories = {}'.format(num_unique(self.basic_cats))
		print '\n# Super categories = {}\n'.format(num_unique(self.super_cats))

	def create_utterance_set(self):
		self.utterances_as_names = sorted(list(set(self.sub_cats 
									+ self.basic_cats + self.super_cats)))
		self.utterances_as_inds  = range(len(self.utterances_as_names))
		self.utt_names_to_inds = dict(zip(self.utterances_as_names, 
									  self.utterances_as_inds))
		self.utt_inds_to_utt_names = dict(zip(self.utterances_as_inds,
									  self.utterances_as_names))
		self.num_utterances = len(self.utterances_as_names)
		print '\n# Utterances = {}'.format(self.num_utterances)
		print '\nUTTERANCES: {}\n'.format(self.utterances_as_names)

	def create_utterance_cost_dict(self):
		# keys are utt inds
		d = dict(zip(self.utterances_as_inds, [len(u) for u in self.utterances_as_names]))
		self.save_as_json(d, 'costs_by_utterance.JSON')
		# also reformat for later use in utterance assignmen
		self.costs = Variable(torch.FloatTensor(
				[d[k] for k in range(0, 
				self.num_utterances)]).type(self.dtype))

	def create_lexicon(self):
		# u x o
		print '\nGenerating lexicon: {} utterances x {} objects\n'.format(
			self.num_utterances, self.num_objects)
		self.lexicon = np.zeros((self.num_utterances, self.num_objects))
		self.lexicon_jitter = self.lexicon + 10e-06
		for i in range(self.num_utterances):
			utt_name = self.utt_inds_to_utt_names[i]
			# print '\n----------\nUtterance: {}\n'.format(utt_name)
			for j in range(self.num_objects):
				obj_name = self.obj_inds_to_obj_names[j]
				if (self.obj_names_to_subs[obj_name] == utt_name) | (
					self.obj_names_to_basics[obj_name] == utt_name) | (
					self.obj_names_to_supers[obj_name] == utt_name):
					# print obj_name
					self.lexicon[i, j] = 1.
					self.lexicon_jitter[i, j] = 1.-10e-06

	def save_lexicon_heatmap(self, savename):
		fig, ax = plt.subplots(facecolor = 'white')
		cax = ax.imshow(self.lexicon, cmap='hot', vmin=0, vmax=1, interpolation='nearest')
		cbar = fig.colorbar(cax, ticks = [0, .5, 1])
		ax.set_title('Synthetic Dataset: Deterministic Lexicon')
		ax.set_xlabel('Object')
		ax.set_ylabel('Utterance')
		plt.savefig(savename)

	def save_lexicon_as_json_dict(self, type, savename):
		if type == 'jitter':
			lexicon = self.lexicon_jitter
		else:
			lexicon = self.lexicon
		# keys are utts
		d = {i:row.tolist() for i, row in enumerate(lexicon)}
		self.save_as_json(d, savename)

	def assign_utterance_to_trial(self, target_name, alt1_name, alt2_name):
		targ_ind = self.obj_names_to_obj_inds[target_name]
		alt1_ind = self.obj_names_to_obj_inds[alt1_name]
		alt2_ind = self.obj_names_to_obj_inds[alt2_name]

		# lex is utterances x 3 torch FloatTensor wrapped in Variable

		# self.lexicon is utts x objs
		# grab lexicon cols corresponding to the 3 objs in our trial
		sub_lex = np.column_stack((self.lexicon_jitter[:, alt1_ind], 
								   self.lexicon_jitter[:, alt2_ind],
								   self.lexicon_jitter[:, targ_ind]))
		
		# get RSA speaker (level 1) distribution,
		# reformatting as appropraite for torch-based RSA code
		t = torch.transpose(model_literal_listener(
					Variable(torch.FloatTensor(sub_lex).type(self.dtype)), 
					uniform_prior(3)), 0, 1)
		utilities = torch.log(t + 10e-06)
		x = (self.alpha * utilities).sub_(self.cost_weight * self.costs.expand_as(
			utilities))
		m = nn.Softmax()
		speaker_table = m(x)
		dist_over_utts_for_target = speaker_table[2, :].unsqueeze(0)
		 # sample utterance
		utt_ind = torch.multinomial(dist_over_utts_for_target, 
									1).data.numpy()[0][0]

		log_prob_of_utt = torch.log(dist_over_utts_for_target 
									+ 10e-06).data.numpy()[0][utt_ind]

		return utt_ind, log_prob_of_utt

	def generate_trial(self, target_name, cond):
		# Generates trials, including assigning utterance from S1
		# conditions:
		# 	1. sub-necessary: 	 (1) same basic
		# 					  	 (1) same super (no reason in principle though?)
		# 	2. basic-sufficient: (1) same super
		# 						 (1) same or diff super
		# 	3. super-sufficient: (1) diff super
		# 						 (1) diff super
		# or, choose distractors randomly (not attempting a particular condition:
		#	'random-choice')
		#
		target_sub   = self.obj_names_to_subs[target_name]
		target_basic = self.obj_names_to_basics[target_name]
		target_super = self.obj_names_to_supers[target_name]

		if self.print_trial_details == True:
			print 'Target: {}'.format(target_name)
			print '		sub: {}'.format(target_sub)
			print '		basic: {}'.format(target_basic)
			print '		super: {}'.format(target_super)

		if cond == 'sub-nec':
			# same basic (but diff sub)
			options = [o for o in self.basics_to_obj_names[target_basic] if (
				       o != target_name)]
			d1 = random.sample(options, 1)[0]

			# any object that's a diff sub
			# I think experiment had same super (but diff basic) as constraint,
			# 		though not sure why
			options = [o for o in self.obj_names if (o != target_name) & (
				       o != d1)]
			d2 = random.sample(options, 1)[0]

		elif cond == 'basic-suff':
			# same super (but diff basic)
			options = [o for o in self.supers_to_obj_names[target_super] if (
				       self.obj_names_to_basics[o] != target_basic)]
			d1 = random.sample(options, 1)[0]

			# same or diff super (but diff basic)
			options = [o for o in self.obj_names if (self.obj_names_to_basics[o] 
				       != target_basic) & (o != d1)]
			d2 = random.sample(options, 1)[0]

		elif cond == 'super-suff':
			# 2x diff supers
			super_cat_options = [s for s in self.supers_to_obj_names.keys() if (
				                 s != target_super)]
			obj_options = [self.supers_to_obj_names[s] for s in super_cat_options]
			options = [val for sublist in obj_options for val in sublist]

			d1 = random.sample(options, 1)[0]
			options = [o for o in options if o != d1]
			d2 = random.sample(options, 1)[0]

		elif cond == 'random-choice':
			# draws distractors randomly, but then figures out what condition for the record
			# distractors can be any obj except another copy of the target
			options = [o for o in self.obj_names if (self.obj_names_to_subs[o] != target_sub)]
			ds = random.sample(options, 2)
			d1 = ds[0]
			d2 = ds[1]
			assert d1 != d2

			# figure out what condition was created
			d1_basic = self.obj_names_to_basics[d1]
			d1_super = self.obj_names_to_supers[d1]
			d2_basic = self.obj_names_to_basics[d2]
			d2_super = self.obj_names_to_supers[d2]
			if (target_basic == d1_basic) or (target_basic == d2_basic):
				cond = 'sub-nec' # at least 1 distractor has same basic
			elif (target_super == d1_super) or (target_super == d2_super):
				cond = 'basic-suff' # at least 1 distractor has super
			else:
				cond = 'super-suff' # both distractors are different super

		distractor_names = [d1, d2]
		random.shuffle(distractor_names)

		# Assign utterance
		utt_ind, log_prob_of_utt = self.assign_utterance_to_trial(target_name, 
												 distractor_names[0], 
								   				 distractor_names[1])

		trial = [target_name, distractor_names[0], distractor_names[1], 
				 cond, utt_ind, log_prob_of_utt]

		if self.print_trial_details:
			print '\n---------------------------------'
			print '\nCondition: {}\n'.format(cond)
			print 'Target: {}		({}		({}))'.format(target_name, 
					self.obj_names_to_basics[target_name], 
				    self.obj_names_to_supers[target_name])
			print 'Alt 1: {}		({}		({}))'.format(distractor_names[0], 
				 	self.obj_names_to_basics[distractor_names[0]], 
					self.obj_names_to_supers[distractor_names[0]])
			print 'Alt 2: {}		({}		({}))'.format(distractor_names[1], 
					self.obj_names_to_basics[distractor_names[1]], 
					self.obj_names_to_supers[distractor_names[1]])
			print 'Utterance: {}'.format(self.utt_inds_to_utt_names[utt_ind])
			print trial
			print '\n---------------------------------\n'

		assert(len(list(set(trial))) == 6)
		return trial

	def mean_log_prob_utterances_in_set(self, trials):
		return np.mean(np.array([t[5] for t in trials]))

	def reformat_trials(self, trials):
		trials_as_inds = []
		for t in trials:
			d = dict()
			d['target_ind'] = self.obj_names_to_obj_inds[t[0]]
			d['alt1_ind']   = self.obj_names_to_obj_inds[t[1]]
			d['alt2_ind']   = self.obj_names_to_obj_inds[t[2]]
			d['condition']	= t[3]
			d['utterance']  = t[4]
			trials_as_inds.append(d)
		return trials_as_inds

	def check_unique_across_splits(self, trial, other_set_trials):
		# checks if trial type is unique across splits 
		# curr_set (the set to which we're adding this trial), train_set, 
		# and other_heldoutset are lists of lists of object inds
		existing_trial_types = get_unique_type_counts(other_set_trials)
		if str(trial_to_trialtype(trial)) in existing_trial_types.keys():
			return False # not unique
		else:
			return True

	def make_trial_with_checking(self, condition, other_set_trials):
		check = 0
		while check == 0:
			candidate = self.generate_trial(random.choice(self.obj_names), 
				                            condition)

			# check that candidate is not 
			if self.check_unique_across_splits(candidate, other_set_trials):
				check = 1
				return candidate

	def create_holdout_set(self, trial_type, targ_sz, holdout_set, other_set_trials):
		# Creates a validation or test split given a target size, and 
		# 	checks that trials are not duplicative with trials in the 
		#	train set and other heldout set 
		# 	(validation or test). Nests.
		#
		# INPUTS: trial_type 				('uniform_conditions' or 'random_distractors')
		#		  targ_sz 		    (must be a multiple of self.num_conditions,
		#							 so that trials are balanced across 
		#							 conditions)
		#		  holdout_set       (either empty, or contains some trials 
		#							 already; trials are lst of obj inds)
		#		  other_set_trials  (lst of trials seen in other sets; may be 
		#							 empty if other sets not yet created)
		# OUTPUTS: holdout_set 		(list of lst of object inds -- not 
		#							 reformatted yet)

		# generate trials
		curr_sz = len(holdout_set) # num trials already in set (in case of nesting)
		num_trials_to_add = targ_sz - curr_sz
		assert (num_trials_to_add % self.num_conditions) == 0

		# top up
		if trial_type == 'random_distractors':
			for i in range(num_trials_to_add):
				holdout_set.append(self.make_trial_with_checking('random-choice',
									other_set_trials))
		elif trial_type == 'uniform_conditions':
			num_trials_per_condition = int(num_trials_to_add/self.num_conditions)
			for i in range(num_trials_per_condition):
				holdout_set.append(self.make_trial_with_checking('sub-nec', 
										other_set_trials))
				holdout_set.append(self.make_trial_with_checking('basic-suff', 
					                    other_set_trials))
				holdout_set.append(self.make_trial_with_checking('super-suff', 
					                    other_set_trials))
		return holdout_set

	def create_train_set_batch(self):
		# Creates train set batches
		# TODO: Could be more efficient
		check = 0
		attempts = 0
		while check == 0:
			# each batch uses each obj as a target at least once
			objs_to_use_as_targets = copy.deepcopy(self.obj_names)

			# 33 targets per batch (11 per cond, 'uniform_conditions' case), 
			#	or total (L case)
			# 1 obj must appear as target 2x for conditions to be balanced
			objs_to_use_as_targets.append(random.sample(objs_to_use_as_targets, 1)[0])
			random.shuffle(objs_to_use_as_targets)

			if self.train_type == 'uniform_conditions':
				# each batch uses each obj as a distractor at least once
				objs_to_use_as_distractors = copy.deepcopy(self.obj_names)

				# partition targets
				cond1_targs = objs_to_use_as_targets[0 : self.num_trials_per_cond_per_batch]
				cond2_targs = (objs_to_use_as_targets[self.num_trials_per_cond_per_batch :
					                                  2*self.num_trials_per_cond_per_batch])
				cond3_targs = (objs_to_use_as_targets[2*self.num_trials_per_cond_per_batch :
					                                  3*self.num_trials_per_cond_per_batch])

				# generate trials
				trials = []
				for i in range(self.num_trials_per_cond_per_batch):
					trials.append(self.generate_trial(cond1_targs[i], 'sub-nec'))
					trials.append(self.generate_trial(cond2_targs[i], 'basic-suff'))
					trials.append(self.generate_trial(cond3_targs[i], 'super-suff'))

				# check that each distractor has been used at least 1x
				distractors = ([sublst[1] for sublst in trials] 
								+ [sublst[2] for sublst in trials])
				not_used_in_trials = ([o for o in objs_to_use_as_distractors 
					                   if o not in list(set(distractors))])
				attempts += 1
				if len(not_used_in_trials) == 0:
					check = 1

			elif self.train_type == 'random_distractors':
				# generate trials
				trials = []
				for i in range(len(objs_to_use_as_targets)):
					t = self.generate_trial(objs_to_use_as_targets[i], 'random-choice')
					trials.append(t)
				check = 1

		# print '{} attempts'.format(attempts)
		return trials

	def generate_dataset(self):
		# train, validation, test sets contain distinct sets of trials

		assert (len(self.train_set_szs) == len(self.test_set_szs) & 
				len(self.train_set_szs) == len(self.validation_set_szs))

		self.train_sets_by_set_sz      = []
		self.validation_sets_by_set_sz = []
		self.test_sets_by_set_sz       = []

		if self.train_type == 'random_distractors':
			print '\nRandom distractors training set'
			print 'Creating a second validation set, with random distractors'
			self.randomdistractor_validation_sets_by_set_sz = []

		for i in range(len(self.train_set_szs)):
			print '\nMaking set # {}'.format(i)

			# init sets
			train_set      = []
			validation_set = []
			test_set       = []
			if self.train_type == 'random_distractors':
				randomdistractor_validation_set = []
			if i != 0:
				# nest: build on previous (smaller) datasets
				train_set      = copy.copy(self.train_sets_by_set_sz[i-1])
				validation_set = copy.copy(self.validation_sets_by_set_sz[i-1])
				test_set       = copy.copy(self.test_sets_by_set_sz[i-1])
				if self.train_type == 'random_distractors':
					randomdistractor_validation_set = copy.copy(
						self.randomdistractor_validation_sets_by_set_sz[i-1])
			# create train set
			train_set += self.create_train_set_batch()
			print 'Train set made'

			# TODO: Clean up
			if self.train_type == 'random_distractors':
				randomdistractor_validation_set = self.create_holdout_set(
					'random_distractors', int(self.validation_set_szs[i]),
					randomdistractor_validation_set,
					train_set + test_set + validation_set)

			# create test, validation
			non_testset_trials = train_set + validation_set
			if self.train_type == 'random_distractors':
				non_testset_trials = non_testset_trials + randomdistractor_validation_set
			test_set = self.create_holdout_set('uniform_conditions',
												int(self.test_set_szs[i]),
												test_set, non_testset_trials)

			non_validset_trials = train_set + test_set
			if self.train_type == 'random_distractors':
				non_validset_trials = non_validset_trials + randomdistractor_validation_set
			validation_set = self.create_holdout_set('uniform_conditions',
													int(self.validation_set_szs[i]),
													validation_set, non_validset_trials)

			assert len(train_set) == self.train_set_szs[i] # added correct num trials
			assert len(validation_set) == self.validation_set_szs[i]
			assert len(test_set) == self.test_set_szs[i]
			if self.train_type == 'random_distractors':
				assert len(randomdistractor_validation_set) == self.validation_set_szs[i]

			if i%20 == 0:
				if self.visualize_train_batches == True:
					visualize_batch(train_set, 'Set ' + str(i) 
										 + ' - Train Set (' + self.train_type + '; ' 
										 + str(int(self.train_set_szs[i])) 
										 + ' trials)',
										 self.obj_names, self.utterances_as_names,
										 self.utt_inds_to_utt_names)
					visualize_batch(validation_set, 'Set ' + str(i) 
										 + ' - Validation Set (' 
										 + str(int(self.validation_set_szs[i])) 
										 + ' trials)', 
										 self.obj_names, self.utterances_as_names,
										 self.utt_inds_to_utt_names)
					visualize_batch(test_set, 'Set ' + str(i) 
										 + ' - Test Set (' 
										 + str(int(self.test_set_szs[i])) 
										 + ' trials)', 
										 self.obj_names, self.utterances_as_names,
										 self.utt_inds_to_utt_names)
					if self.train_type == 'random_distractors':
						visualize_batch(randomdistractor_validation_set, 'Set ' + str(i) 
										 + ' - Random Distractors Validation Set (' 
										 + str(int(self.validation_set_szs[i])) 
										 + ' trials)', 
										 self.obj_names, self.utterances_as_names,
										 self.utt_inds_to_utt_names)

			self.train_sets_by_set_sz.append(train_set)
			self.validation_sets_by_set_sz.append(validation_set)
			self.test_sets_by_set_sz.append(test_set)
			if self.train_type == 'random_distractors':
				self.randomdistractor_validation_sets_by_set_sz.append(
					randomdistractor_validation_set)

	def reformat_dataset_and_save(self):
		# could be more efficient given nested reformats
		num_datasets = len(self.train_sets_by_set_sz)
		for i in range(num_datasets):
			# reformats to lst of dicts
			print '\nReformatting Set # {}'.format(i)

			train_set      = self.reformat_trials(self.train_sets_by_set_sz[i])
			validation_set = self.reformat_trials(self.validation_sets_by_set_sz[i])
			test_set       = self.reformat_trials(self.test_sets_by_set_sz[i])

			train_set_LL = self.mean_log_prob_utterances_in_set(
								self.train_sets_by_set_sz[i])
			validation_set_LL = self.mean_log_prob_utterances_in_set(
								self.validation_sets_by_set_sz[i])
			test_set_LL = self.mean_log_prob_utterances_in_set(
								self.test_sets_by_set_sz[i])

			print 'Train set sz = {}, Valid set sz = {}, Test set sz = {}'.format(
				len(train_set), len(validation_set), len(test_set))

			print 'Train Mean Log-Prob of Utterance = {}'.format(train_set_LL)
			print 'Validation Mean Log-Prob of Utterance = {}'.format(
				validation_set_LL)
			print 'Test Mean Log-Prob of Utterance = {}'.format(test_set_LL)

			if self.train_type == 'random_distractors':
				randomdistractor_validation_set = self.reformat_trials(
					self.randomdistractor_validation_sets_by_set_sz[i])
				randomdistractor_validation_set_LL = self.mean_log_prob_utterances_in_set(
					self.randomdistractor_validation_sets_by_set_sz[i])
				print 'Random Distractors Valid set sz = {}, (LL = {})'.format(
					len(randomdistractor_validation_set), randomdistractor_validation_set_LL)
				rd_validation_name = (self.dataset_save_path + 'randomdistractors_validation_set' 
							+ str(i) + '_'
							+ str(int(self.validation_set_szs[i])) 
							+ 'validation_')
				self.save_as_json(randomdistractor_validation_set, rd_validation_name + 'trials.JSON')
				self.save_as_json({'LL': float(randomdistractor_validation_set_LL)}, rd_validation_name + 'LL.JSON')

			# save sets as json
			train_name = (self.dataset_save_path + 'train_set' 
							+ str(i) + '_'
							+ str(int(self.train_set_szs[i])) 
							+ 'train_')
			self.save_as_json(train_set, train_name + 'trials.JSON')
			self.save_as_json({'LL': float(train_set_LL)}, train_name + 'LL.JSON')

			validation_name = (self.dataset_save_path + 'validation_set' 
							+ str(i) + '_'
							+ str(int(self.validation_set_szs[i])) 
							+ 'validation_')
			self.save_as_json(validation_set, validation_name + 'trials.JSON')
			self.save_as_json({'LL': float(validation_set_LL)}, validation_name + 'LL.JSON')

			test_name = (self.dataset_save_path + 'test_set' 
							+ str(i) + '_'
							+ str(int(self.test_set_szs[i])) 
							+ 'test_')
			self.save_as_json(test_set, test_name + 'trials.JSON')
			self.save_as_json({'LL': float(test_set_LL)}, test_name + 'LL.JSON')

def wrapper():
	object_info = DatasetMaker('random_distractors') #'uniform_conditions')

if __name__ == '__main__':
	print '\n\nTo view plots of dataset characteristics, enter `python -m visdom.server`' 
	print 'in another terminal window. Then navigate to http://localhost.com:8097'
	print 'in your browser\n'
	raw_input("Press Enter to continue...")
	vis = visdom.Visdom()

	wrapper()
