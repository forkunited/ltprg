from __future__ import division
import math
import time
import random
import json
import shutil
import numpy as np 
import visdom
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
from network_components import MLP
from rsa import uniform_prior, model_speaker_1

# TODO: Add checkpoints
#		Add cuda support
#		Switch to inplace where possible
#		Add commandline args
#		Add image embedding options
# 		Troubleshoot ReLU nan gradient issue

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

class ModelTrainer(object):
	def __init__(self, model_name, hidden_szs, hiddens_nonlinearity,
				 batch_sz, train_data, validation_data, utt_set_sz,
				 obj_set_sz, obj_embedding_type, visualize_opt,
				 display_validationset_predictions_opt, **kwargs):
		# model_name		('ersa', 'nnwc', 'nnwoc')
		# hidden_szs		(lst of hidden layer szs)
		# batch_sz			(int)
		# train_data		(lst of dictionaries, e.g.
		#			 		 {'target_ind': 1, 'alt1_ind': 5,
		#				 	 'alt2_ind': 18, 'utterance': 4,
		#				 	 'condition': 'sub-nec'})
		# validation_data 	(same format as train_data)
		# utt_set_sz    	(num utterances in fixed alternatives set)
		# obj_embedding_type ('onehot') # TODO: Add image types
		# visualize_opt		(plot learning curves in Visdom; True/False)
		# display_validationset_predictions_opt (print model predictions; True/False)
		# rsa_level			(optional; level of recursion, from 1)
		# alpha				(optional; speaker rationality param)
		# cost_dict			(optional; dict of utterance costs)
		# cost_weight		(optional; utterance cost weight in RSA model)
		# utt_dict			(optional; dict of utterance inds to names)
		# obj_dict			(optional; dict of object inds to names)

		assert model_name in ['ersa', 'nnwc', 'nnwoc']
		assert obj_embedding_type in ['onehot']

		if display_validationset_predictions_opt == True:
			self.utt_inds_to_names = kwargs['utt_dict']
			self.obj_inds_to_names = kwargs['obj_dict']

		self.visualize_opt = visualize_opt
		self.prep_visualize()

		self.model_name = model_name
		self.batch_sz   = batch_sz
		self.train_data = train_data
		self.validation_data = validation_data
		self.utt_set_sz = utt_set_sz
		self.obj_set_sz = obj_set_sz
		self.obj_embedding_type = obj_embedding_type

		if self.model_name == 'ersa':
			# set RSA params
			self.rsa_dist_type = 'speaker'
			self.rsa_level     = kwargs['rsa_level']
			self.alpha         = kwargs['alpha']
			self.cost_weight   = kwargs['cost_weight']
			cost_dict = kwargs['cost_dict']

			# (using always uniform for now: objs within 
			# trials are equally salient)
			self.world_prior = uniform_prior(3)
			# utterance costs
			self.costs = Variable(torch.FloatTensor(
				[cost_dict[str(k)] for k in range(0, self.utt_set_sz)]))

			if self.obj_embedding_type == 'onehot':
				in_sz = self.obj_set_sz

			# set model
			self.model = MLP(in_sz, hidden_szs, self.utt_set_sz, hiddens_nonlinearity, 
							 'sigmoid')

		elif model_name == 'nnwc':
			if self.obj_embedding_type == 'onehot':
				in_sz = self.obj_set_sz * 3

			self.model = MLP(in_sz, hidden_szs, self.utt_set_sz, hiddens_nonlinearity,
							 'logSoftmax')

		elif model_name == 'nnwoc':
			if self.obj_embedding_type == 'onehot':
				in_sz = self.obj_set_sz

			self.model = MLP(in_sz, hidden_szs, self.utt_set_sz, hiddens_nonlinearity,
							 'logSoftmax')

		self.criterion = nn.NLLLoss() # neg log-like loss, operates on log probs
		self.optimizer = optim.Adam(self.model.parameters(), weight_decay=0.00001)

	def format_inputs(self, target_obj_ind, alt1_obj_ind, alt2_obj_ind):
		# returns 2D tensor input to MLP
		if self.obj_embedding_type == 'onehot':
			if self.model_name == 'ersa':
				inputs = Variable(torch.cat(
							[one_hot(alt1_obj_ind, self.obj_set_sz),
							one_hot(alt2_obj_ind, self.obj_set_sz),
							one_hot(target_obj_ind, self.obj_set_sz)], 
							0))
			elif self.model_name == 'nnwc':
				inputs = Variable(torch.cat(
							[one_hot(alt1_obj_ind, self.obj_set_sz),
							one_hot(alt2_obj_ind, self.obj_set_sz),
							one_hot(target_obj_ind, self.obj_set_sz)],  
							1))
			elif self.model_name == 'nnwoc':
				inputs = Variable(one_hot(target_obj_ind, 
							self.obj_set_sz))
		# TODO: Support other embedding types
		return inputs

	def mean_performance_dataset(self, data_set):
		loss_by_trial = []
		acc_by_trial  = []
		for trial in data_set:
			prediction, label = self.predict(trial)
			loss, accuracy = self.evaluate(prediction, label)
			loss_by_trial.append(loss.data.numpy()[0])
			acc_by_trial.append(accuracy.data.numpy()[0])
		return np.mean(loss_by_trial), np.mean(acc_by_trial)

	def evaluate_datasets(self, epoch):
		# mean NLL, acc for each dataset
		train_loss, train_acc = self.mean_performance_dataset(self.train_data)

		self.display_predictions_opt = True
		validation_loss, validation_acc = self.mean_performance_dataset(
											self.validation_data)
		self.display_predictions_opt = False

		self.mean_trainset_loss.append(train_loss)
		self.mean_trainset_acc.append(train_acc)
		self.mean_validationset_loss.append(validation_loss)
		self.mean_validationset_acc.append(validation_acc)
		self.dataset_eval_epoch.append(epoch)
		print '\nMean train set loss = '
		print self.mean_trainset_loss
		print 'Mean validation set loss = '
		print self.mean_validationset_loss
		print 'Mean train set acc = '
		print self.mean_trainset_acc
		print 'Mean validation set acc = '
		print self.mean_validationset_acc
		self.plot_mean_dataset_results(epoch)

	def prep_visualize(self):
		if self.visualize_opt == True:
			print '\n\nTo view live performance plots, enter `python -m visdom.server`' 
			print 'in another terminal window. Then navigate to http://localhost.com:8097'
			print 'in your browser\n'
			raw_input("Press Enter to continue...")
			self.vis = visdom.Visdom()
			self.display_predictions_opt = False # print only validation set preds

	def plot_learning_curve(self, epoch):
		if self.visualize_opt == True:
			if epoch == 1:
				self.loss_win = self.vis.line(
					X=np.array([epoch]),
					Y=np.array([self.train_loss_by_epoch[-1]]),
					opts=dict(
						title=self.model_name.upper() + ': NLLLoss Over Training')
					)

				self.acc_win = self.vis.line(
					X=np.array([epoch]),
					Y=np.array([self.train_acc_by_epoch[-1]]),
					opts=dict(
						title=self.model_name.upper() + ': Accuracy Over Training')
					)

			else:
				self.vis.updateTrace(
					X=np.array([epoch]),
					Y=np.array([self.train_loss_by_epoch[-1]]),
					win=self.loss_win)

				self.vis.updateTrace(
					X=np.array([epoch]),
					Y=np.array([self.train_acc_by_epoch[-1]]),
					win=self.acc_win)

	def plot_mean_dataset_results(self, epoch):
		if self.visualize_opt == True:
			x = np.array(np.column_stack(([epoch], [epoch])))
			
			if epoch == 0:
				self.dataset_eval_loss_win = self.vis.line(
					X=x,
					Y=np.array(
						np.column_stack(
							([self.mean_trainset_loss[-1]],
							[self.mean_validationset_loss[-1]]))),
					opts=dict(
						legend=['Train Set', 'Validation Set'],
						title=self.model_name.upper() + ': Mean NLLLoss of Datasets')
					)

				self.dataset_eval_acc_win = self.vis.line(
					X=x,
					Y=np.array(
						np.column_stack(
							([self.mean_trainset_acc[-1]],
							[self.mean_validationset_acc[-1]]))),
					opts=dict(
						legend=['Train Set', 'Validation Set'],
						title=self.model_name.upper() + ': Mean Accuracy of Datasets')
					)

			else:
				self.vis.updateTrace(
					X=x,
					Y=np.array(
						np.column_stack(
							([self.mean_trainset_loss[-1]],
							[self.mean_validationset_loss[-1]]))),
					win=self.dataset_eval_loss_win)

				self.vis.updateTrace(
					X=x,
					Y=np.array(
						np.column_stack(
							([self.mean_trainset_acc[-1]],
							[self.mean_validationset_acc[-1]]))),
					win=self.dataset_eval_acc_win)

	def display_prediction(self, target_obj_ind, alt1_obj_ind, alt2_obj_ind,
						   condition, prediction_dist, label_utt_ind):
		if self.display_predictions_opt == True:
			_, predicted_utt_ind = torch.max(prediction_dist, 1)
			predicted_utt_ind = predicted_utt_ind.data.numpy()[0][0] # extract from tensor

			print '\nCondition: {}'.format(condition)
			print '	Target: {}'.format(self.obj_inds_to_names[str(
										target_obj_ind)])
			print '	Alt 1: {}'.format(self.obj_inds_to_names[str(
										alt1_obj_ind)])
			print '	Alt 2: {}'.format(self.obj_inds_to_names[str(
										alt2_obj_ind)])
			print 'Label: {}'.format(self.utt_inds_to_names[str(
										label_utt_ind)])
			print 'Prediction: {}'.format(self.utt_inds_to_names[str(
										predicted_utt_ind)])
			print 'Correct? {}'.format(predicted_utt_ind==label_utt_ind)

	def predict(self, trial):
		target_obj_ind = trial['target_ind']
		alt1_obj_ind   = trial['alt1_ind']
		alt2_obj_ind   = trial['alt2_ind']
		utt_ind = trial['utterance']
		condition = trial['condition']

		# inputs are 2D tensors
		inputs = self.format_inputs(target_obj_ind, 
									alt1_obj_ind, alt2_obj_ind)

		# forward pass
		outputs = self.model.forward(inputs) # MLP forward
		if self.model_name == 'ersa':
			# pass through RSA
			speaker_table = model_speaker_1(
								torch.transpose(outputs, 0, 1), 
								self.world_prior, self.alpha, 
								self.cost_weight, self.costs)
			# dist over utterances for target obj
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

		self.train_loss_by_epoch = [] # learning curve
		self.train_acc_by_epoch  = []
		
		dataset_eval_freq = 10 # every n epochs
		self.mean_trainset_loss   = [] # mean of dataset
		self.mean_trainset_acc    = []
		self.mean_validationset_loss = []
		self.mean_validationset_acc  = []
		self.dataset_eval_epoch      = [] # epoch evaluated

		epoch = 0
		self.evaluate_datasets(epoch) # establish baseline
		while True:
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

			if epoch % dataset_eval_freq == 0:
				self.evaluate_datasets(epoch)

if __name__=='__main__':
	data_path = 'example_data/' # temp synthetic data w/ 3300 training examples
	train_data_fname      = data_path + 'train_set99_3300train_trials.JSON'
	validation_data_fname = data_path + 'validation_set99_600validation_trials.JSON'
	example_train_data 		= load_json(train_data_fname) 
	example_validation_data = load_json(validation_data_fname)
	d = load_json(data_path + 'true_lexicon.JSON')
	num_utts = len(d)
	num_objs = len(d['0'])

	utt_info_dict = load_json(data_path + 'utt_inds_to_names.JSON')
	obj_info_dict = load_json(data_path + 'obj_inds_to_names.JSON')
	utt_costs     = load_json(data_path + 'costs_by_utterance.JSON')

	# Train model
	trainer = ModelTrainer('ersa', [100], 'tanh', 1, example_train_data, 
				 			example_validation_data, num_utts, num_objs, 
				 			'onehot', True, True,
				 			utt_dict=utt_info_dict, obj_dict=obj_info_dict,
				 			rsa_level=1, alpha=100, cost_dict=utt_costs,
				 			cost_weight=0.1)
	trainer.train()
