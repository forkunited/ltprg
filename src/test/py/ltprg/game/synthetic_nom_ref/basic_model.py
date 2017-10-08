import abc
import numpy as np 
import os
from rsa import uniform_prior
import torch
import torch.nn as nn
import torch.cuda as cuda
from torch import optim
from torch.autograd import Variable

class BasicModel(object):
    """ Base class fo experimental models. 
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self, model_name, model_type, hidden_szs, hiddens_nonlinearity,
                 utt_set_sz, obj_set_sz, obj_embedding_type, utt_dict, obj_dict,
                 weight_decay, learning_rate, rsa_params,
                 save_path):
        """
        model_type        defines the model type, see ModelType class
        hidden_szs        (lst of hidden layer szs; specifies both the
                           number of hidden layers and their sizes)
        hiddens_nonlinearity ('relu', 'tanh')
        utt_set_sz        (num utterances in fixed alternatives set)
        obj_set_sz        (num objects in dataset)
        obj_embedding_type defines the embedding type, see EmbeddingType class
        utt_dict          (dict whose keys are utterance inds (as strings),
                           and whose vals are utterance names, for
                           trial printouts)
        obj_dict          (dict whose keys are object inds, and 
                           whose vals are object names (as strings), 
                           for trial printouts)
        weight_decay      (weight decay (l2 penalty))
        learning_rate     (initial learning rate in Adam optimization)
        rsa_params        see RSAParams Class for details
        save_path         (where to save results)
        """
        # Initialize model training params
        self.model_name = model_name
        self.model_type = model_type
        self.hidden_szs = hidden_szs
        self.hiddens_nonlinearity = hiddens_nonlinearity
        self.utt_set_sz = utt_set_sz
        self.obj_set_sz = obj_set_sz
        self.obj_embedding_type = obj_embedding_type
        self.utt_inds_to_names = utt_dict
        self.obj_inds_to_names = obj_dict
        self.weight_decay = weight_decay
        self.learning_rate = learning_rate
        self.rsa_params = rsa_params

        # Construct model, loss formulation, and optimizer.
        self.model = self.create_model()
        self.criterion = nn.NLLLoss()  # neg log-like loss, operates on log probs
        self.optimizer = optim.Adam(self.model.parameters(), 
                                    weight_decay=self.weight_decay, 
                                    lr=self.learning_rate)
        
        # Creates Save Directory
        self.save_path = save_path
        if not os.path.isdir(self.save_path):
            os.makedirs(self.save_path)

        # dtype
        if cuda.is_available():
            self.dtype = torch.cuda.FloatTensor
            self.label_dtype = torch.cuda.LongTensor
        else:
            self.dtype = torch.FloatTensor
            self.label_dtype = torch.LongTensor


    def evaluate(self, prediction, label):
        """ Apply crtierion function eval model's prediction.
        """
        loss = self.criterion(prediction.type(self.dtype), label.type(self.label_dtype))
        _, ind = torch.max(prediction, 1)
        accuracy = ind.type(self.label_dtype) == label
        return loss, accuracy


    def learned_versus_gold_standard_S1(self, trial, prediction):
        """returns scalar KL-divergence(S1 given gold-standard lexicon,
            S1 given learn lexicon)
        """
   
        # S1 on gold-standard lexicon
        gold_stardard_S1_dist, _ = self.predict(trial,
            use_gold_standard_lexicon=True)

        # compute KL-divergence
        #   KLDivLoss takes in x, targets, where x is log-probs
        #   and targets is probs (not log)
        return nn.KLDivLoss()(prediction, torch.exp(
                                gold_stardard_S1_dist)).data[0]


    def kl_baseline(self, prediction):
        """ Compute scalar KL-divergence of prediction from uniform dist.
        """
        kl_div = nn.KLDivLoss()(prediction, uniform_prior(self.utt_set_sz, self.dtype)
                                ).data[0]
        return kl_div


    @abc.abstractmethod
    def predict(self, trial, display_prediction=False,
                use_gold_standard_lexicon=False):
        """ Make prediction for specified trial.
        """
        return


    def update(self, loss, max_norm):
        """ Compute backprop on loss, clip gradients, and apply update.
        """
        loss.backward() # backprop
        norm = torch.nn.utils.clip_grad_norm(
            self.model.parameters(), max_norm) # clip gradient
        self.optimizer.step() # update
        self.optimizer.zero_grad() # zero gradient buffers


    def get_model_name(self):
        """ Get model name.
        """
        return self.model_name


    def get_model_type(self):
        """ Get model type.
        """
        return self.model_type


    @abc.abstractmethod
    def create_model(self):
        """ Create underlying model.
        """
        return


    @abc.abstractmethod
    def format_inputs(trial, rsa_on_top=False):
        """ Format inputs for model.
        """
        return


    def display_prediction(self, trial, prediction_dist):
        """ Print and return informaion pertaining to a prediction.
        """
        _, predicted_utt_ind = torch.max(prediction_dist, 1)
        predicted_utt_ind = predicted_utt_ind.data[0] # extract from tensor

        target_name = self.obj_inds_to_names[str(trial['target_ind'])]
        alt1_name   = self.obj_inds_to_names[str(trial['alt1_ind'])]
        alt2_name   = self.obj_inds_to_names[str(trial['alt2_ind'])]
        predicted_utt_name = self.utt_inds_to_names[str(predicted_utt_ind)]
        label_utt_name     = self.utt_inds_to_names[str(trial['utterance'])]

        print '\nCondition: {}'.format(trial['condition'])
        print ' Target: {}'.format(target_name)
        print ' Alt 1: {}'.format(alt1_name)
        print ' Alt 2: {}'.format(alt2_name)
        print 'Label: {}'.format(label_utt_name)
        print 'Prediction: {}'.format(predicted_utt_name)
        print 'Correct? {}'.format(predicted_utt_ind==trial['utterance'])

        return target_name, predicted_utt_name, label_utt_name


    def save_details(self):
        """ Save model details to disk.
        """
        d = dict()
        d['model_type'] = ModelType.to_string(self.model_type)
        d['model_name'] = self.model_name
        d['obj_embedding_type'] = self.obj_embedding_type
        d['hiddens_szs'] = self.hidden_szs
        d['hiddens_nonlinearity'] = self.hiddens_nonlinearity
        d['weight_decay'] = self.weight_decay
        d['learning_rate'] = self.learning_rate
        d.update(self.rsa_params.to_dict())
        np.save(self.save_path + 'model_details.npy', d)


class ModelType(object):
    """ Enumeration of possible model types. 
    """
    ERSA = 0  # Explict RSA (Rational Speech Act)
    NNWC = 1  # Neural Network with Context
    NNWOC = 2  # Neural Network without Context

    @staticmethod
    def to_string(mt):
        if mt == ModelType.ERSA:
            return 'ersa'
        elif mt == ModelType.NNWC:
            return 'nnwc'
        elif mt == ModelType.NNWOC:
            return 'nnwoc'
        else:
            return 'NA - Model Type Does Not Exist'


class EmbeddingType(object):
    """ Enumeration of possible embedding types.
    """
    ONE_HOT = 0  # One Hot Vector 


def one_hot(ind, sz):
    # 2D tensor one-hot
    if cuda.is_available():
        dtype = torch.cuda.FloatTensor
    else:
        dtype = torch.FloatTensor
    out = torch.FloatTensor(1, sz).zero_().type(dtype)
    out[0, ind] = 1
    return out
