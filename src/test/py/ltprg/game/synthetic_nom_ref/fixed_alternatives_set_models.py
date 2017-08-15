from __future__ import division
import math
import time
import random
import json
import shutil
import os
import numpy as np 
import visdom
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
from network_components import MLP
from rsa import uniform_prior, model_speaker_1
from vis_embedding import vis_embedding

# Trains model. Also assesses mean loss, accuracy, and KL-divergence from 
#		gold-standard S1 distribution (S1 produced by performing RSA using 
#		ground-truth lexicon) on full train and held-out validation sets
#		every n epochs.
#
# Model descriptions
# - EXPLICIT RSA MODEL ('ersa'): given an object embedding, neural network
#				produces truthiness vals between 0 and 1 for each 
#				utterance in the alternatives set. Each object in a trial 
#				is fed through the network, producing a lexicon that is 
#				then fed to RSA. RSA returns a level-1 speaker distribution, 
#				P(u | target, context, L)
# - NEURAL NETWORK WITH CONTEXT MODEL ('nnwc'): produces distribution over 
#				utterances in the fixed alternatives set given a trial's 
#				concatenated object embeddings, with target object in final 
#				position
# - NEURAL NETWORK WITHOUT CONTEXT MODEL ('nnwoc') produces distribution 
#				over utterances given target object emebdding only
#
# TODO: Add commandline args
#		Switch to inplace where possible
#		Add cuda support (though not really necessary yet given training speed)
# 		Troubleshoot ReLU nan gradient issue
#		Add mini-batches (currently sz 1) (lower priority)
#		Add image embedding options (lower priority)

random.seed(3)

def load_json(filename):
	with open(filename) as json_data:
	    d = json.load(json_data)
	return d

def one_hot(ind, sz):
	# 2D tensor one-hot
	out = torch.FloatTensor(1, sz).zero_()
	out[0, ind] = 1
	return out

def init_cond_dict(conditions):
	d = dict()
	for k in conditions:
		d[k] = []
	return d

class ModelTrainer(object):
	def __init__(self, model_name, hidden_szs, hiddens_nonlinearity,
				 train_data, validation_data, utt_set_sz,
				 obj_set_sz, obj_embedding_type, utt_dict, obj_dict,
				 weight_decay, learning_rate,
				 visualize_opt,  
				 alpha, cost_dict, cost_weight, gold_standard_lexicon,
				 save_path):
		# model_name		('ersa', 'nnwc', 'nnwoc':
		# hidden_szs		(lst of hidden layer szs; specifies both the
		#					 number of hidden layers and their sizes)
		# hiddens_nonlinearity ('relu', 'tanh')
		# train_data		(lst of dictionaries, e.g.
		#			 		 {'target_ind': 1, 'alt1_ind': 5,
		#				 	 'alt2_ind': 18, 'utterance': 4,
		#				 	 'condition': 'sub-nec'})
		# validation_data 	(held-out validation set whose trial types
		#					 are distinct from those in train_data;
		#					 same format as train_data)
		# utt_set_sz    	(num utterances in fixed alternatives set)
		# obj_set_sz		(num objects in dataset)
		# obj_embedding_type ('onehot')
		# utt_dict			(dict whose keys are utterance inds (as strings),
		#					 and whose vals are utterance names, for
		#					 trial printouts)
		# obj_dict			(dict whose keys are object inds, and 
		#					 whose vals are object names (as strings), 
		#					 for trial printouts)
		# weight_decay		(weight decay (l2 penalty))
		# learning_rate		(initial learning rate in Adam optimization)
		# visualize_opt		(plot learning curves in Visdom; True/False)
		#
		# >> Parameters used for comparison between model prediction and 
		#	 goldstandard(RSA) S1 distribution (all models); and by ersa model:
		# alpha				(speaker rationality param)
		# cost_dict			(dict of utterance costs)
		# cost_weight		(utterance cost weight in RSA model)
		# gold_stardard_lexicon (num utterances x num objects np array of 
		#					 ground-truth lexicon used to generate data)
		# save_path			(where to save results)

		assert model_name in ['ersa', 'nnwc', 'nnwoc']
		assert obj_embedding_type in ['onehot']

		self.utt_inds_to_names = utt_dict
		self.obj_inds_to_names = obj_dict

		self.visualize_opt = visualize_opt
		self.prep_visualize()
		self.display_predictions_opt = False # print only validation set preds

		self.model_name = model_name
		self.train_data = train_data
		self.validation_data = validation_data
		self.utt_set_sz = utt_set_sz
		self.obj_set_sz = obj_set_sz
		self.obj_embedding_type = obj_embedding_type
		self.hidden_szs = hidden_szs
		self.hiddens_nonlinearity = hiddens_nonlinearity
		self.weight_decay = weight_decay
		self.learning_rate = learning_rate

		# RSA params (for ersa model, and gold-standard S1 comparison)
		self.alpha         = alpha
		self.cost_weight   = cost_weight
		self.world_prior = uniform_prior(3) #objs within trials are equally salient
		self.costs = Variable(torch.FloatTensor(
			[cost_dict[str(k)] for k in range(0, self.utt_set_sz)]))
		self.use_gold_standard_lexicon = False # do not change
		self.gold_standard_lexicon = Variable(torch.FloatTensor(
										gold_standard_lexicon))

		self.conditions = list(set([trial['condition'] for trial in self.train_data]))
		self.save_path = save_path
		
		# create model
		if self.model_name == 'ersa':
			if self.obj_embedding_type == 'onehot':
				in_sz = self.obj_set_sz
			self.model = MLP(in_sz, self.hidden_szs, self.utt_set_sz, 
							 self.hiddens_nonlinearity, 'sigmoid')
		elif model_name == 'nnwc':
			if self.obj_embedding_type == 'onehot':
				in_sz = self.obj_set_sz * 3
			self.model = MLP(in_sz, self.hidden_szs, self.utt_set_sz, 
							 self.hiddens_nonlinearity, 'logSoftmax')
		elif model_name == 'nnwoc':
			if self.obj_embedding_type == 'onehot':
				in_sz = self.obj_set_sz
			self.model = MLP(in_sz, self.hidden_szs, self.utt_set_sz, 
							 self.hiddens_nonlinearity, 'logSoftmax')

		# loss function, optimization
		self.criterion = nn.NLLLoss() # neg log-like loss, operates on log probs
		self.optimizer = optim.Adam(self.model.parameters(), 
									weight_decay=self.weight_decay, 
									lr=self.learning_rate)

	def format_inputs(self, target_obj_ind, alt1_obj_ind, alt2_obj_ind, 
					  format_type):
		# returns 2D tensor input to MLP
		# TODO: Support other embedding types
		if self.obj_embedding_type == 'onehot':
			if format_type == 'ersa' or format_type == 'nnwoc_w_rsa_on_top':
				return Variable(torch.cat(
							[one_hot(alt1_obj_ind, self.obj_set_sz),
							one_hot(alt2_obj_ind, self.obj_set_sz),
							one_hot(target_obj_ind, self.obj_set_sz)], 
							0))
			elif format_type == 'nnwc':
				return Variable(torch.cat(
							[one_hot(alt1_obj_ind, self.obj_set_sz),
							one_hot(alt2_obj_ind, self.obj_set_sz),
							one_hot(target_obj_ind, self.obj_set_sz)],  
							1))
			elif format_type == 'nnwoc':
				return Variable(one_hot(target_obj_ind, 
							self.obj_set_sz))

	def compare_S1s_from_learned_versus_goldstandard_lexicons(self, trial, 
															  prediction):
		# returns scalar KL-divergence(S1 given gold-standard lexicon,
		# S1 given learn lexicon)
   
		# S1 on gold-standard lexicon
		self.use_gold_standard_lexicon = True
		gold_stardard_S1_dist, label = self.predict(trial)
		self.use_gold_standard_lexicon = False

		# print '\n\n'
		# np.set_printoptions(suppress=True)
		# print 'Pred 	Goldstandard S1 	Label'
		# print np.column_stack((
		# 	torch.exp(prediction).data.numpy()[0], 
		# 	torch.exp(gold_stardard_S1_dist).data.numpy()[0],
		# 	one_hot(label.data.numpy()[0], self.utt_set_sz).numpy()[0]))

		# print '\n KL from label:'
		# print nn.KLDivLoss()(prediction, Variable(one_hot(label.data.numpy()[0], self.utt_set_sz))).data.numpy()[0]

		# compute KL-divergence
		# 	KLDivLoss takes in x, targets, where x is log-probs
		#	and targets is probs (not log)
		kl_div = nn.KLDivLoss()(prediction, torch.exp(
								gold_stardard_S1_dist)).data.numpy()[0]
		return kl_div

	def kl_baseline(self, prediction):
		# returns scalar KL-divergence of prediction from uniform dist
		kl_div = nn.KLDivLoss()(prediction, uniform_prior(self.utt_set_sz)
								).data.numpy()[0]
		return kl_div

	def mean_performance_dataset(self, data_set, set_name):
		loss_by_trial = []
		acc_by_trial  = []
		acc_by_trial_by_condition = init_cond_dict(self.conditions)
		S1_dist_goldstandard_learned = []
		S1_dist_goldstandard_learned_by_condition = init_cond_dict(self.conditions)
		baseline_kl_from_uniform     = []
		for trial in data_set:
			if set_name == 'validation' or set_name == 'test':
				self.display_predictions_opt = True
			prediction, label = self.predict(trial)
			self.display_predictions_opt = False

			loss, accuracy = self.evaluate(prediction, label)

			loss_by_trial.append(loss.data.numpy()[0])
			acc_by_trial.append(accuracy.data.numpy()[0])
			acc_by_trial_by_condition[trial['condition']].append(
							accuracy.data.numpy()[0][0])

			# assess KL-divergence(S1 from gold-standard lexicon, S1 from 
			# learned lexicon)
			kl_from_S1 = self.compare_S1s_from_learned_versus_goldstandard_lexicons(
					trial, prediction)
			S1_dist_goldstandard_learned.append(kl_from_S1)
			S1_dist_goldstandard_learned_by_condition[trial['condition']].append(
					kl_from_S1)
			baseline_kl_from_uniform.append(self.kl_baseline(prediction))

		mean_acc_by_cond = dict()
		mean_dist_from_goldstandard_by_cond = dict()
		for k in self.conditions:
			mean_acc_by_cond[k] = np.mean(acc_by_trial_by_condition[k])
			mean_dist_from_goldstandard_by_cond[k] = np.mean(
				S1_dist_goldstandard_learned_by_condition[k])

		mean_loss = np.mean(loss_by_trial)
		mean_acc = np.mean(acc_by_trial)
		mean_dist_from_goldstandard = np.mean(S1_dist_goldstandard_learned)
		mean_baseline_kl = np.mean(baseline_kl_from_uniform)

		return mean_loss, mean_acc, mean_acc_by_cond, mean_dist_from_goldstandard, mean_dist_from_goldstandard_by_cond, mean_baseline_kl

	def evaluate_datasets(self, epoch):
		# mean NLL, acc for each dataset
		(train_loss, train_acc, train_acc_by_cond, 
			train_dist_from_goldstandard,
			train_dist_from_goldstandard_by_cond, 
			train_baseline_kl) = self.mean_performance_dataset(self.train_data,
											'train')

		(validation_loss, validation_acc, val_acc_by_cond,
			validation_dist_from_goldstandard, 
			validation_dist_from_goldstandard_by_cond,
			validation_baseline_kl) = self.mean_performance_dataset(
											self.validation_data, 'validation')

		# collect
		self.mean_trainset_loss.append(train_loss)
		self.mean_trainset_acc.append(train_acc)

		self.mean_validationset_loss.append(validation_loss)
		self.mean_validationset_acc.append(validation_acc)
		
		for k in self.conditions:
			self.mean_trainset_acc_by_cond[k].append(train_acc_by_cond[k])
			self.mean_validationset_acc_by_cond[k].append(val_acc_by_cond[k])
			self.mean_trainset_dist_from_goldstandard_S1_by_cond[k].append(
				train_dist_from_goldstandard_by_cond[k])
			self.mean_validationset_dist_from_goldstandard_S1_by_cond[k].append(
				validation_dist_from_goldstandard_by_cond[k])

		self.dataset_eval_epoch.append(epoch)

		self.mean_trainset_dist_from_goldstandard_S1.append(
				train_dist_from_goldstandard)
		self.mean_trainset_kl_from_uniform.append(
				train_baseline_kl)
		self.mean_validationset_dist_from_goldstandard_S1.append(
				validation_dist_from_goldstandard)
		self.mean_validationset_kl_from_uniform.append(
				validation_baseline_kl)

		# display performance info
		print '\nMean train set loss = {}'
		print self.mean_trainset_loss
		print 'Mean validation set loss = '
		print self.mean_validationset_loss
		print 'Mean train set acc = '
		print self.mean_trainset_acc
		print 'Mean validation set acc = '
		print self.mean_validationset_acc
		print 'Mean train / validation set accuracy by trial = '
		print train_acc_by_cond
		print val_acc_by_cond
		print 'Mean train set KL-div from goldstandard S1 = '
		print self.mean_trainset_dist_from_goldstandard_S1
		print '(Baseline) Mean train set KL-div from uniform distribution = '
		print self.mean_trainset_kl_from_uniform
		print 'Mean validation set KL-div from goldstandard S1 = '
		print self.mean_validationset_dist_from_goldstandard_S1
		print '(Baseline) Mean validation set KL-div from uniform distribution = '
		print self.mean_validationset_kl_from_uniform

		# plot
		self.plot_mean_dataset_results(epoch)
		self.plot_evaluations_by_cond(epoch)
		# vis_embedding(self.model.get_embedding(), self.vis)

	def prep_visualize(self):
		if self.visualize_opt == True:
			print '\n\nTo view live performance plots, enter `python -m visdom.server`' 
			print 'in another terminal window. Then navigate to http://localhost.com:8097'
			print 'in your browser\n'
			raw_input("Press Enter to continue...")
			self.vis = visdom.Visdom() 

	def plot_learning_curve(self, epoch):
		if self.visualize_opt == True:
			x = np.array([epoch])
			y_loss = np.array([self.train_loss_by_epoch[-1]])
			y_acc = np.array([self.train_acc_by_epoch[-1]])
			if epoch == 1:
				self.loss_win = self.vis.line(X=x, Y=y_loss,
					opts=dict(
						title=self.model_name.upper() + ': NLLLoss Over Training')
					)
				self.acc_win = self.vis.line(X=x, Y=y_acc,
					opts=dict(
						title=self.model_name.upper() + ': Accuracy Over Training')
					)
			else:
				self.vis.updateTrace(X=x, Y=y_loss, win=self.loss_win)
				self.vis.updateTrace(X=x, Y=y_acc, win=self.acc_win)

	def plot_mean_dataset_results(self, epoch):
		if self.visualize_opt == True:
			x = np.array(np.column_stack(([epoch], [epoch])))
			y_loss = np.array(
						np.column_stack(
							([self.mean_trainset_loss[-1]],
							[self.mean_validationset_loss[-1]])))
			y_acc = np.array(
						np.column_stack(
							([self.mean_trainset_acc[-1]],
							[self.mean_validationset_acc[-1]])))
			y_kl = np.array(
						np.column_stack(
							([self.mean_trainset_dist_from_goldstandard_S1[-1]],
							 [self.mean_validationset_dist_from_goldstandard_S1[-1]],
							 [self.mean_trainset_kl_from_uniform[-1]],
							 [self.mean_validationset_kl_from_uniform[-1]])))
			if epoch == 0:
				self.dataset_eval_loss_win = self.vis.line(X=x, Y=y_loss,
					opts=dict(
						legend=['Train Set', 'Validation Set'],
						title=self.model_name.upper() + ': Mean NLLLoss of Datasets')
					)
				self.dataset_eval_acc_win = self.vis.line(
					X=x,
					Y=y_acc,
					opts=dict(
						legend=['Train Set', 'Validation Set'],
						title=self.model_name.upper() + ': Mean Accuracy of Datasets')
					)
				self.dataset_eval_dist_from_goldstandard_win = self.vis.line(
					X=np.array(np.column_stack(([epoch], [epoch], [epoch], [epoch]))),
					Y=y_kl,
					opts=dict(
						legend=['Train Set', 'Validation Set', 'Train Baseline (KL Div from Uniform)', 'Validation Baseline'],
						title=self.model_name.upper() + ': Mean KL-Div from Goldstandard S1 of Datasets')
					)
			else:
				self.vis.updateTrace(X=x, Y=y_loss, win=self.dataset_eval_loss_win)
				self.vis.updateTrace(X=x, Y=y_acc, win=self.dataset_eval_acc_win)
				self.vis.updateTrace(X=np.array(np.column_stack(([epoch], [epoch], [epoch], [epoch]))),
					Y=y_kl, win=self.dataset_eval_dist_from_goldstandard_win)

	def plot_evaluations_by_cond(self, epoch):
		if self.visualize_opt == True:
			x = np.array(
				np.column_stack(
					tuple([epoch] * len(self.conditions))))
			y_train_acc = np.array(
				np.column_stack(
					tuple([self.mean_trainset_acc_by_cond[k][-1] for k in self.conditions])))
			y_validation_acc = np.array(
				np.column_stack(
					tuple([self.mean_validationset_acc_by_cond[k][-1] for k in self.conditions])))
			y_train_kl = np.array(
				np.column_stack(
					tuple([self.mean_trainset_dist_from_goldstandard_S1_by_cond[k][-1] for k in self.conditions])))
			y_validation_kl = np.array(
				np.column_stack(
					tuple([self.mean_validationset_dist_from_goldstandard_S1_by_cond[k][-1] for k in self.conditions])))
			if epoch == 0:
				self.trainset_eval_by_cond_acc_win = self.vis.line(
					X=x, Y=y_train_acc, opts=dict(
						legend=self.conditions, 
						title=self.model_name.upper() 
						+ ' : Mean Train Set Acc by Condition'))
				self.validationset_eval_by_cond_acc_win = self.vis.line(
					X=x, Y=y_validation_acc, opts=dict(
						legend=self.conditions, 
						title=self.model_name.upper() 
						+ ' : Mean Validation Set Acc by Condition'))
				self.trainset_eval_by_cond_kl_win = self.vis.line(
					X=x, Y=y_train_kl, opts=dict(
						legend=self.conditions, 
						title=self.model_name.upper() 
						+ ' : Mean Train Set KL-Div from Goldstandard S1'))
				self.validationset_eval_by_cond_kl_win = self.vis.line(
					X=x, Y=y_validation_kl, opts=dict(
						legend=self.conditions, 
						title=self.model_name.upper() 
						+ ' : Mean Validation Set KL-Div from Goldstandard S1'))
			else:
				self.vis.updateTrace(X=x, Y=y_train_acc, 
									 win=self.trainset_eval_by_cond_acc_win)
				self.vis.updateTrace(X=x, Y=y_validation_acc, 
									 win=self.validationset_eval_by_cond_acc_win)
				self.vis.updateTrace(X=x, Y=y_train_kl, 
									 win=self.trainset_eval_by_cond_kl_win)
				self.vis.updateTrace(X=x, Y=y_validation_kl, 
									 win=self.validationset_eval_by_cond_kl_win)

	def save_model_details(self):
		d = dict()
		d['model_name'] = self.model_name
		d['alpha']      = self.alpha
		d['obj_embedding_type'] = self.obj_embedding_type
		d['hiddens_szs'] = self.hidden_szs
		d['hiddens_nonlinearity'] = self.hiddens_nonlinearity
		d['weight_decay'] = self.weight_decay
		d['learning_rate'] = self.learning_rate
		d['cost_weight'] = self.cost_weight
		d['costs'] = self.costs
		np.save(self.save_path + 'model_details.npy', d)

	def save_checkpoint(self, epoch, is_best):
		filename = self.save_path + 'checkpoint.pth.tar'
		d = {'epoch': epoch, 'state_dict': self.model.state_dict(), 
			 'optimizer': self.optimizer.state_dict()}
		torch.save(d, filename)
		if is_best == True:
			shutil.copyfile(filename, self.save_path + 'model_best.pth.tar')

	def save_results(self):
		# save results dictionaries as npy files
		learning_curves = dict()
		learning_curves['Train_loss_by_epoch'] = self.train_loss_by_epoch
		learning_curves['Train_acc_by_epoch'] = self.train_acc_by_epoch
		np.save(self.save_path + 'LearningCurves.npy', learning_curves)

		dataset_evals = dict()
		dataset_evals['Dataset_eval_epochs_collected'] = self.dataset_eval_epoch
		dataset_evals['Mean_trainset_loss'] = self.mean_trainset_loss
		dataset_evals['Mean_trainset_acc'] = self.mean_trainset_acc
		dataset_evals['Mean_trainset_acc_by_cond'] = self.mean_trainset_acc_by_cond
		dataset_evals['Mean_trainset_dist_from_goldstandard_S1'] = self.mean_trainset_dist_from_goldstandard_S1
		dataset_evals['Mean_trainset_dist_from_goldstandard_S1_by_cond'] = self.mean_trainset_dist_from_goldstandard_S1_by_cond
		dataset_evals['Mean_validationset_loss'] = self.mean_validationset_loss
		dataset_evals['Mean_validationset_acc'] = self.mean_validationset_acc
		dataset_evals['Mean_validationset_acc_by_cond'] = self.mean_validationset_acc_by_cond
		dataset_evals['Mean_validationset_dist_from_goldstandard_S1'] = self.mean_validationset_dist_from_goldstandard_S1
		dataset_evals['Mean_validationset_dist_from_goldstandard_S1_by_cond'] = self.mean_validationset_dist_from_goldstandard_S1_by_cond
		np.save(self.save_path + 'DatasetEvaluations.npy', dataset_evals)

		# save checkpoint(s)
		is_best = False
		if self.mean_validationset_loss[-1] < self.best_validationset_loss:
			self.best_validationset_loss = self.mean_validationset_loss[-1]
			is_best = True
		self.save_checkpoint(self.dataset_eval_epoch[-1], is_best)

	def retrieve_category_predicted(self, target_name, predicted_utt_name):
		if predicted_utt_name == self.obj_names_to_subs[target_name]:
			return 'sub'
		elif predicted_utt_name == self.obj_names_to_basics[target_name]:
			return 'basic'
		elif predicted_utt_name == self.obj_names_to_supers[target_name]:
			return 'super'
		else:
			return 'other'

	def look_at_validation_set_predictions_given_trained_model(self):
		# prints predictions over dataset
		categories_predicted_by_condition = init_cond_dict(self.conditions)
		self.display_predictions_opt = False # TODO: simplify
		for trial in self.validation_data:
			prediction, _ = self.predict(trial)

			target_obj_ind = trial['target_ind']
			alt1_obj_ind   = trial['alt1_ind']
			alt2_obj_ind   = trial['alt2_ind']
			utt_ind = trial['utterance']
			condition = trial['condition']

			self.display_predictions_opt = True
			target_name, predicted_utt_name, label_utt_name = self.display_prediction(
				target_obj_ind, alt1_obj_ind, alt2_obj_ind, condition, prediction, utt_ind)
			categories_predicted_by_condition[condition].append(
				self.retrieve_category_predicted(target_name, predicted_utt_name))
			self.display_predictions_opt = False
		return categories_predicted_by_condition

	def display_prediction(self, target_obj_ind, alt1_obj_ind, alt2_obj_ind,
						   condition, prediction_dist, label_utt_ind):
		if self.display_predictions_opt == True:
			_, predicted_utt_ind = torch.max(prediction_dist, 1)
			predicted_utt_ind = predicted_utt_ind.data.numpy()[0][0] # extract from tensor

			target_name = self.obj_inds_to_names[str(target_obj_ind)]
			alt1_name   = self.obj_inds_to_names[str(alt1_obj_ind)]
			alt2_name   = self.obj_inds_to_names[str(alt2_obj_ind)]
			predicted_utt_name = self.utt_inds_to_names[str(predicted_utt_ind)]
			label_utt_name     = self.utt_inds_to_names[str(label_utt_ind)]

			print '\nCondition: {}'.format(condition)
			print '	Target: {}'.format(target_name)
			print '	Alt 1: {}'.format(alt1_name)
			print '	Alt 2: {}'.format(alt2_name)
			print 'Label: {}'.format(label_utt_name)
			print 'Prediction: {}'.format(predicted_utt_name)
			print 'Correct? {}'.format(predicted_utt_ind==label_utt_ind)

			return target_name, predicted_utt_name, label_utt_name

	def predict(self, trial):
		target_obj_ind = trial['target_ind']
		alt1_obj_ind   = trial['alt1_ind']
		alt2_obj_ind   = trial['alt2_ind']
		utt_ind = trial['utterance']
		condition = trial['condition']

		# inputs are 2D tensors
		inputs = self.format_inputs(target_obj_ind, alt1_obj_ind, alt2_obj_ind, 
									self.model_name)

		# forward pass
		outputs = self.model.forward(inputs) # MLP forward

		# if ersa model (or goldstandard S1), feed MLP output into RSA
		if self.model_name == 'ersa' or self.model_name == 'nnwoc_w_rsa_on_top' or self.use_gold_standard_lexicon == True:
			if self.use_gold_standard_lexicon == True:
				# uses ground-truth lexicon (for comparison w/ 
				# model predictions); grab objects for this trial
				inds = Variable(torch.LongTensor([alt1_obj_ind, alt2_obj_ind, 
												 target_obj_ind]))
				lexicon = torch.index_select(self.gold_standard_lexicon, 
											 1, inds)
			else:
				# uses learned params
				lexicon = torch.transpose(outputs, 0, 1)
			# pass through RSA
			speaker_table = model_speaker_1(lexicon, self.world_prior, 
											self.alpha, self.cost_weight, 
											self.costs)
			# pull dist over utterances for target obj
			outputs = speaker_table[2, :].unsqueeze(0)

		# format label
		label = Variable(torch.LongTensor([utt_ind]))

		self.display_prediction(target_obj_ind, alt1_obj_ind, alt2_obj_ind,
						   		condition, outputs, utt_ind)
		return outputs, label

	def evaluate(self, prediction, label):
		loss = self.criterion(prediction, label)
		_, ind = torch.max(prediction, 1)
		accuracy = ind==label	
		return loss, accuracy

	def update(self, loss, max_norm):
		loss.backward() # backprop
		norm = torch.nn.utils.clip_grad_norm(
			self.model.parameters(), max_norm) # clip gradient
		# print ' {}'.format(norm)
		self.optimizer.step() # update
		self.optimizer.zero_grad() # zero gradient buffers

	def train(self):
		start_time = time.time()
		max_norm = 1 # grad norm
		num_epochs = 1000 # epochs to train

		self.train_loss_by_epoch = [] # learning curve
		self.train_acc_by_epoch  = []
		
		dataset_eval_freq = 1#5 # every n epochs
		self.dataset_eval_epoch      = [] # epoch evaluated
		self.mean_trainset_loss   = [] # mean of dataset
		self.mean_trainset_acc    = []
		self.mean_trainset_acc_by_cond = init_cond_dict(self.conditions)
		self.mean_validationset_loss = []
		self.mean_validationset_acc  = []
		self.mean_validationset_acc_by_cond = init_cond_dict(self.conditions)

		# KL-div between S1 distribution on gold-standard lexicon,
		# and on learned lexicon (MLP output)
		self.mean_trainset_dist_from_goldstandard_S1      = []
		self.mean_trainset_dist_from_goldstandard_S1_by_cond = init_cond_dict(
			self.conditions)
		self.mean_validationset_dist_from_goldstandard_S1 = []
		self.mean_validationset_dist_from_goldstandard_S1_by_cond = init_cond_dict(
			self.conditions)
		self.mean_trainset_kl_from_uniform      = []
		self.mean_validationset_kl_from_uniform = []

		epoch = 0
		self.evaluate_datasets(epoch) # establish baseline
		self.best_validationset_loss = self.mean_validationset_loss[-1]
		while epoch < num_epochs:
			epoch += 1
			print '\nEpoch {}'.format(epoch)

			train_loss_this_epoch = []
			train_acc_this_epoch  = []

			random.shuffle(self.train_data)
			for trial in self.train_data:
				prediction, label = self.predict(trial)
				loss, accuracy = self.evaluate(prediction, label)
				self.update(loss, max_norm)

				train_loss_this_epoch.append(loss.data.numpy()[0])
				train_acc_this_epoch.append(accuracy.data.numpy()[0])

			self.train_loss_by_epoch.append(np.mean(train_loss_this_epoch))
			self.train_acc_by_epoch.append(np.mean(train_acc_this_epoch))
			print 'Loss = {}'.format(self.train_loss_by_epoch)
			print 'Accuracy = {}'.format(self.train_acc_by_epoch)
			self.plot_learning_curve(epoch)

			print 'Epoch runtime = {}'.format(time.time() - start_time)

			if epoch % dataset_eval_freq == 0:
				self.save_model_details()
				self.evaluate_datasets(epoch)
				self.save_results()

		print 'Train time = {}'.format(time.time() - start_time)
		self.save_results()

	def evaluate_test_set_at_peak(self, test_set):
		# set model to stopping pt
		print 'Loading checkpoint from epoch with lowest validation-set loss'
		checkpoint = torch.load(self.save_path + 'model_best.pth.tar')
		print 'Epoch = {}'.format(checkpoint['epoch'])
		self.model.load_state_dict(checkpoint['state_dict'])

		# assess performance
		(loss, acc, acc_by_cond,dist_from_goldstandard, 
			dist_from_goldstandard_by_cond, _) = self.mean_performance_dataset(test_set, 'test')

		dataset_evals = dict()
		dataset_evals['Mean_testset_loss'] = loss
		dataset_evals['Mean_testset_acc'] = acc
		dataset_evals['Mean_testset_acc_by_cond'] = acc_by_cond
		dataset_evals['Mean_testset_dist_from_goldstandard_S1'] = dist_from_goldstandard
		dataset_evals['Mean_testset_dist_from_goldstandard_S1_by_cond'] = dist_from_goldstandard_by_cond
		
		if self.model_name == 'nnwoc_w_rsa_on_top':
			self.save_path = self.save_path + 'rsa_added_for_test_set/'
			if os.path.isdir(self.save_path) == False:
				os.makedirs(self.save_path)
		np.save(self.save_path + 'DatasetEvaluations_AtPeak_TestSet.npy', dataset_evals)

def train_model(model_name, hidden_szs, hiddens_nonlinearity,
				 train_data, validation_data, test_data,
				 utt_set_sz, obj_set_sz, 
				 obj_embedding_type, utt_dict, obj_dict,
				 weight_decay, learning_rate,
				 visualize_opt,
				 alpha, cost_dict, cost_weight, gold_standard_lexicon,
				 save_path):
	trainer = ModelTrainer(model_name, hidden_szs, hiddens_nonlinearity,
				 train_data, validation_data, utt_set_sz,
				 obj_set_sz, obj_embedding_type, utt_dict, obj_dict,
				 weight_decay, learning_rate,
				 visualize_opt,  
				 alpha, cost_dict, cost_weight, gold_standard_lexicon,
				 save_path)
	trainer.train()
	trainer.evaluate_test_set_at_peak(test_data)
	# How well does model do if trained acontextually, but RSA added on top?
	if trainer.model_name == 'nnwoc':
		trainer.model_name = 'nnwoc_w_rsa_on_top'
		print trainer.model_name
		trainer.evaluate_test_set_at_peak(test_data)

def run_example():
	train_set_type = 'random_distractors' # 'random_distractors' or 'uniform_conditions'

	data_path = 'synthetic_data/' # temp synthetic data w/ 3300 training examples
	data_by_num_trials_path = data_path + 'datasets_by_num_trials/' + train_set_type + '/'

	# train_data_fname      = 'train_set99_3300train_trials.JSON'
	# validation_data_fname = 'validation_set99_600validation_trials.JSON'

	train_data_fname = 'train_set14_495train_trials.JSON'
	validation_data_fname = 'randomdistractors_validation_set14_90validation_trials.JSON'
	test_data_fname = 'validation_set14_90validation_trials.JSON' # uniform conds

	example_train_data 		= load_json(data_by_num_trials_path + train_data_fname) 
	example_validation_data = load_json(data_by_num_trials_path + validation_data_fname)
	example_test_data 		= load_json(data_by_num_trials_path + test_data_fname)

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

	# Train model
	# model_name = 'ersa'
	# model_name = 'nnwc'
	model_name = 'nnwoc'

	results_dir = 'results/' + train_set_type + '/no_hidden_layer/' + train_data_fname.split('_')[1] + '/' + model_name + '/'
	# results_dir = 'results/' + train_set_type + '/' + train_data_fname.split('_')[1] + '/' + model_name + '/'
	if os.path.isdir(results_dir) == False:
		os.makedirs(results_dir)

	train_model(model_name, [], 'tanh', example_train_data, 
	 			example_validation_data, example_test_data,
	 			num_utts, num_objs, 
	 			'onehot', utt_info_dict, obj_info_dict, 
	 			decay, lr, True, 
	 			100, utt_costs, 0.1, true_lexicon,
				results_dir)

if __name__=='__main__':
	run_example()
