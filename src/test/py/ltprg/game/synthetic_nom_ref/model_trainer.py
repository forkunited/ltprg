from fixed_alternatives_set_models_refactored import FASM_ERSA, FASM_NNWC, FASM_NNWOC
from basic_model import ModelType, EmbeddingType
import json
import math
import numpy as np 
import random
from rsa import uniform_prior, RSAParams
import shutil
import time
import torch
import torch.nn as nn
from torch.autograd import Variable
from unbounded_alternatives_set_models import UASM_ERSA, UASM_NNWC, UASM_NNWOC
import visdom
from vis_embedding import vis_embedding

"""
Framework for training models.
"""

# ----------------
# HELPER FUNCTIONS
# ----------------

def load_json(filename):
    with open(filename) as json_data:
        d = json.load(json_data)
    return d


def init_cond_dict(conditions):
    d = dict()
    for k in conditions:
        d[k] = []
    return d


def establish_seed(seed=3):
    random.seed(seed)


class ModelTrainer(object):
    def __init__(self,  model, train_data, validation_data,
                 should_visualize, save_path):
        """
        model             BasicModel object to train & evaluate
        train_data        (lst of dictionaries, e.g.
                           {'target_ind': 1, 'alt1_ind': 5,
                           'alt2_ind': 18, 'utterance': 4,
                           'condition': 'sub-nec'})
        validation_data   (held-out validation set whose trial types
                           are distinct from those in train_data;
                           same format as train_data)
        should_visualize     (plot learning curves in Visdom; True/False)
        save_path         (where to save results)
        """
        # Initialize model training params
        establish_seed()
        self.model = model
        self.train_data = train_data
        self.validation_data = validation_data
        self.should_visualize = should_visualize
        self.save_path = save_path
        self.conditions = list(set([trial['condition'] for trial in self.train_data]))

        # Visualization Tools        
        self.should_visualize = should_visualize
        if self.should_visualize:
            print '\n\nTo view live performance plots, enter `python -m visdom.server`' 
            print 'in another terminal window. Then navigate to http://localhost.com:8097'
            print 'in your browser\n'
            raw_input("Press Enter to continue...")
            self.vis = visdom.Visdom() 


    def train(self):
        # start_time = time.time()
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
            start_time = time.time()
            epoch += 1
            print '\nEpoch {}'.format(epoch)

            train_loss_this_epoch = []
            train_acc_this_epoch  = []

            random.shuffle(self.train_data)
            for trial in self.train_data:
                prediction, label = self.model.predict(trial)
                loss, accuracy = self.model.evaluate(prediction, label)
                self.model.update(loss, max_norm)

                train_loss_this_epoch.append(loss.data.numpy()[0])
                train_acc_this_epoch.append(accuracy.data.numpy()[0])

            self.train_loss_by_epoch.append(np.mean(train_loss_this_epoch))
            self.train_acc_by_epoch.append(np.mean(train_acc_this_epoch))
            print 'Loss = {}'.format(self.train_loss_by_epoch)
            print 'Accuracy = {}'.format(self.train_acc_by_epoch)
            if self.should_visualize:
                self.plot_learning_curve(epoch)

            print 'Epoch runtime = {}'.format(time.time() - start_time)

            if epoch % dataset_eval_freq == 0:
                self.model.save_details()
                self.evaluate_datasets(epoch)
                self.save_results()

        print 'Train time = {}'.format(time.time() - start_time)
        self.save_results()


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
        if self.should_visualize:
            self.plot_mean_dataset_results(epoch)
            self.plot_evaluations_by_cond(epoch)
            # vis_embedding(self.model.get_embedding(), self.vis)


    def mean_performance_dataset(self, data_set, set_name):
        """ Compute mean performance of a model on a given dataset.
        """
        loss_by_trial = []
        acc_by_trial  = []
        acc_by_trial_by_condition = init_cond_dict(self.conditions)
        S1_dist_goldstandard_learned = []
        S1_dist_goldstandard_learned_by_condition = init_cond_dict(self.conditions)
        baseline_kl_from_uniform     = []
        for trial in data_set:
            show_pred = (set_name == 'validation' or set_name == 'test')
            prediction, label = self.model.predict(trial, show_pred)
            loss, accuracy = self.model.evaluate(prediction, label)

            loss_by_trial.append(loss.data.numpy()[0])
            acc_by_trial.append(accuracy.data.numpy()[0])
            acc_by_trial_by_condition[trial['condition']].append(
                            accuracy.data.numpy()[0])

            # assess KL-divergence(S1 from gold-standard lexicon, S1 from 
            # learned lexicon)
            kl_from_S1 = self.model.learned_versus_gold_standard_S1(trial, prediction)
            S1_dist_goldstandard_learned.append(kl_from_S1)
            S1_dist_goldstandard_learned_by_condition[trial['condition']].append(
                    kl_from_S1)
            baseline_kl_from_uniform.append(self.model.kl_baseline(prediction))

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


    def save_results(self):
        """ Save results dictionaries as npy files.
        """
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


    def save_checkpoint(self, epoch, is_best):
        """ Save model weights at a checkpoint.
        """
        filename = self.save_path + 'checkpoint.pth.tar'
        d = {'epoch': epoch, 'state_dict': self.model.model.state_dict(), 
             'optimizer': self.model.optimizer.state_dict()}
        torch.save(d, filename)
        if is_best == True:
            shutil.copyfile(filename, self.save_path + 'model_best.pth.tar')


    def look_at_validation_set_predictions_given_trained_model(self):
        """ Prints predictions over dataset.
        """
        categories_predicted_by_condition = init_cond_dict(self.conditions)
        for trial in self.validation_data:
            prediction_distr, _ = self.model.predict(trial)
            target_name, predicted_utt_name, label_utt_name = \
                self.display_prediction(trial, prediction_distr)
            categories_predicted_by_condition[trial['condition']].append(
                self.retrieve_category_predicted(target_name, predicted_utt_name))
        return categories_predicted_by_condition


    def retrieve_category_predicted(self, target_name, predicted_utt_name):
        """ Retrieve category of predicted utterance.
        """
        if predicted_utt_name == self.obj_names_to_subs[target_name]:
            return 'sub'
        elif predicted_utt_name == self.obj_names_to_basics[target_name]:
            return 'basic'
        elif predicted_utt_name == self.obj_names_to_supers[target_name]:
            return 'super'
        else:
            return 'other'


    def plot_learning_curve(self, epoch):
        """ Plot learning curve according to given epoch.
        """
        x = np.array([epoch])
        y_loss = np.array([self.train_loss_by_epoch[-1]])
        y_acc = np.array([self.train_acc_by_epoch[-1]])
        if epoch == 1:
            self.loss_win = self.vis.line(X=x, Y=y_loss,
                opts=dict(
                    title=self.model.get_model_name().upper() + ': NLLLoss Over Training')
                )
            self.acc_win = self.vis.line(X=x, Y=y_acc,
                opts=dict(
                    title=self.model.get_model_name().upper() + ': Accuracy Over Training')
                )
        else:
            self.vis.updateTrace(X=x, Y=y_loss, win=self.loss_win)
            self.vis.updateTrace(X=x, Y=y_acc, win=self.acc_win)


    def plot_mean_dataset_results(self, epoch):
        """ Plot mean dataset results.
        """
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
                    title=self.model.get_model_name().upper() + ': Mean NLLLoss of Datasets')
                )
            self.dataset_eval_acc_win = self.vis.line(
                X=x,
                Y=y_acc,
                opts=dict(
                    legend=['Train Set', 'Validation Set'],
                    title=self.model.get_model_name().upper() + ': Mean Accuracy of Datasets')
                )
            self.dataset_eval_dist_from_goldstandard_win = self.vis.line(
                X=np.array(np.column_stack(([epoch], [epoch], [epoch], [epoch]))),
                Y=y_kl,
                opts=dict(
                    legend=['Train Set', 'Validation Set', 'Train Baseline (KL Div from Uniform)', 'Validation Baseline'],
                    title=self.model.get_model_name().upper() + ': Mean KL-Div from Goldstandard S1 of Datasets')
                )
        else:
            self.vis.updateTrace(X=x, Y=y_loss, win=self.dataset_eval_loss_win)
            self.vis.updateTrace(X=x, Y=y_acc, win=self.dataset_eval_acc_win)
            self.vis.updateTrace(X=np.array(np.column_stack(([epoch], [epoch], [epoch], [epoch]))),
                Y=y_kl, win=self.dataset_eval_dist_from_goldstandard_win)


    def plot_evaluations_by_cond(self, epoch):
        """ Plot evaluations according to the trial condition.
        """
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
                    title=self.model.get_model_name().upper() 
                    + ' : Mean Train Set Acc by Condition'))
            self.validationset_eval_by_cond_acc_win = self.vis.line(
                X=x, Y=y_validation_acc, opts=dict(
                    legend=self.conditions, 
                    title=self.model.get_model_name().upper() 
                    + ' : Mean Validation Set Acc by Condition'))
            self.trainset_eval_by_cond_kl_win = self.vis.line(
                X=x, Y=y_train_kl, opts=dict(
                    legend=self.conditions, 
                    title=self.model.get_model_name().upper() 
                    + ' : Mean Train Set KL-Div from Goldstandard S1'))
            self.validationset_eval_by_cond_kl_win = self.vis.line(
                X=x, Y=y_validation_kl, opts=dict(
                    legend=self.conditions, 
                    title=self.model.get_model_name().upper() 
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


def train_model(model, train_data, validation_data,
                 should_visualize, save_path):
    trainer = ModelTrainer(model, train_data, validation_data,
                 should_visualize, save_path)
    trainer.train()


def run_example():
    train_set_type = 'random_distractors' # 'random_distractors' or 'uniform_conditions'

    data_path = 'synthetic_data/' # temp synthetic data w/ 3300 training examples
    data_by_num_trials_path = data_path + 'datasets_by_num_trials/' + train_set_type + '/'

    # train_data_fname      = 'train_set99_3300train_trials.JSON'
    # validation_data_fname = 'validation_set99_600validation_trials.JSON'

    train_data_fname = 'train_set14_495train_trials.JSON'
    validation_data_fname = 'validation_set14_90validation_trials.JSON'

    example_train_data      = load_json(data_by_num_trials_path + train_data_fname) 
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

    # RSA params
    rsa_params = RSAParams(
        alpha=0.1,
        cost_weight=100,
        cost_dict=utt_costs,
        gold_standard_lexicon=true_lexicon
    )

    def create_save_path(model_name, train_set_type, train_data_fname):
        return 'results/{}/local_runs_for_viewing/no_hidden_layer/{}/{}/'.format(
            train_set_type,
            train_data_fname.split('_')[1],
            model_name
        )

    # Various Models:
    # ---------------
    # Model Type 1 -- Fixed Alternative Set Models
    fasm_ersa = FASM_ERSA(
        model_name='fasm_ersa',
        model_type=ModelType.to_string(ModelType.ERSA),
        hidden_szs=[],
        hiddens_nonlinearity='tanh',
        utt_set_sz=num_utts,
        obj_set_sz=num_objs,
        obj_embedding_type=EmbeddingType.ONE_HOT,
        utt_dict=utt_info_dict,
        obj_dict=obj_info_dict,
        weight_decay=decay,
        learning_rate=lr,
        rsa_params=rsa_params,
        save_path=create_save_path('fasm_ersa', train_set_type, train_data_fname)
    )

    fasm_nnwc = FASM_NNWC(
        model_name='fasm_nnwc',
        model_type=ModelType.to_string(ModelType.NNWC),
        hidden_szs=[],
        hiddens_nonlinearity='tanh',
        utt_set_sz=num_utts,
        obj_set_sz=num_objs,
        obj_embedding_type=EmbeddingType.ONE_HOT,
        utt_dict=utt_info_dict,
        obj_dict=obj_info_dict,
        weight_decay=decay,
        learning_rate=lr,
        rsa_params=rsa_params,
        save_path=create_save_path('fasm_nnwc', train_set_type, train_data_fname)
    )

    fasm_nnwoc = FASM_NNWOC(
        model_name='fasm_nnwoc',
        model_type=ModelType.to_string(ModelType.NNWOC),
        hidden_szs=[],
        hiddens_nonlinearity='tanh',
        utt_set_sz=num_utts,
        obj_set_sz=num_objs,
        obj_embedding_type=EmbeddingType.ONE_HOT,
        utt_dict=utt_info_dict,
        obj_dict=obj_info_dict,
        weight_decay=decay,
        learning_rate=lr,
        rsa_params=rsa_params,
        save_path=create_save_path('fasm_nnwoc', train_set_type, train_data_fname)
    )

    # Model Type 2 -- Unbounded Alternative Set Models
    # ------------------------------------------------
    uasm_ersa = UASM_ERSA(
        model_name='uasm_ersa',
        model_type=ModelType.to_string(ModelType.ERSA),
        hidden_szs=[],
        hiddens_nonlinearity='tanh',
        utt_set_sz=num_utts,
        obj_set_sz=num_objs,
        obj_embedding_type=EmbeddingType.ONE_HOT,
        utt_dict=utt_info_dict,
        obj_dict=obj_info_dict,
        weight_decay=decay,
        learning_rate=lr,
        rsa_params=rsa_params,
        save_path=create_save_path('uasm_ersa', train_set_type, train_data_fname)
    )


    uasm_nnwc = UASM_NNWC(
        model_name='uasm_nnwc',
        model_type=ModelType.to_string(ModelType.NNWC),
        hidden_szs=[50, 100, 200],
        hiddens_nonlinearity='relu',
        utt_set_sz=num_utts,
        obj_set_sz=num_objs,
        obj_embedding_type=EmbeddingType.ONE_HOT,
        utt_dict=utt_info_dict,
        obj_dict=obj_info_dict,
        weight_decay=decay,
        learning_rate=lr,
        rsa_params=rsa_params,
        save_path=create_save_path('uasm_nnwc', train_set_type, train_data_fname)
    )

    uasm_nwwoc = FASM_NNWOC(
        model_name='uasm_nnwoc',
        model_type=ModelType.to_string(ModelType.NNWOC),
        hidden_szs=[],
        hiddens_nonlinearity='tanh',
        utt_set_sz=num_utts,
        obj_set_sz=num_objs,
        obj_embedding_type=EmbeddingType.ONE_HOT,
        utt_dict=utt_info_dict,
        obj_dict=obj_info_dict,
        weight_decay=decay,
        learning_rate=lr,
        rsa_params=rsa_params,
        save_path=create_save_path('uasm_nnwoc', train_set_type, train_data_fname)
    )

    # Example
    train_model(
        model=uasm_nnwc,
        train_data=example_train_data,
        validation_data=example_validation_data,
        should_visualize=True,
        save_path=uasm_ersa.save_path
    )
  

if __name__=='__main__':
    run_example()
