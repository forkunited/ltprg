import abc
import json
import random
from rsa import uniform_prior, RSAParams
import torch
import torch.nn as nn
from torch.autograd import Variable

"""
Framework for training models.
"""

class ModelTrainer(object):
    def __init__(self, model_name, model_type, hidden_szs, hiddens_nonlinearity,
                 train_data, validation_data, utt_set_sz,
                 obj_set_sz, obj_embedding_type, utt_dict, obj_dict,
                 weight_decay, learning_rate, should_visualize,
                 rsa_params, save_path):
        """
        model_type        defines the model type, see ModelType class
        hidden_szs        (lst of hidden layer szs; specifies both the
                           number of hidden layers and their sizes)
        hiddens_nonlinearity ('relu', 'tanh')
        train_data        (lst of dictionaries, e.g.
                           {'target_ind': 1, 'alt1_ind': 5,
                           'alt2_ind': 18, 'utterance': 4,
                           'condition': 'sub-nec'})
        validation_data   (held-out validation set whose trial types
                           are distinct from those in train_data;
                           same format as train_data)
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
        visualize_opt     (plot learning curves in Visdom; True/False)
        
        rsa_params        see RSAParams Class for details
        save_path         (where to save results)
        """
        assert model_type in [ModelType.ERSA, ModelType.NNWC, ModelType.NNWOC]

        # Initialize model training params
        establish_seed()
        self.model_name = model_name
        self.model_type = model_type
        self.hidden_szs = hidden_szs
        self.hiddens_nonlinearity = hiddens_nonlinearity
        self.train_data = train_data
        self.validation_data = validation_data
        self.utt_set_sz = utt_set_sz
        self.obj_set_sz = obj_set_sz
        self.obj_embedding_type = obj_embedding_type
        self.utt_inds_to_names = utt_dict
        self.obj_inds_to_names = obj_dict
        self.weight_decay = weight_decay
        self.learning_rate = learning_rate
        self.should_visualize = should_visualize
        self.rsa_params = rsa_params
        self.save_path = save_path
        self.conditions = list(set([trial['condition'] for trial in self.train_data]))
        self.model = self.create_model()


    @abc.abstractmethod
    def train(self):
        return


    @abc.abstractmethod
    def train_model(self):
        return


    @abc.abstractmethod
    def run_example(self):
        return


class BasicModel(object):
    """ Base class fo experimental models. 
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self, model_name, model_type, hidden_szs, hiddens_nonlinearity,
                 utt_set_sz, obj_set_sz, obj_embedding_type, utt_dict, obj_dict,
                 weight_decay, learning_rate, rsa_params, save_path):
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
        
        # Construct model, loss formulation, and optimizer
        self.model = self.create_model()
        self.criterion = nn.NLLLoss()  # neg log-like loss, operates on log probs
        self.optimizer = optim.Adam(self.model.parameters(), 
                                    weight_decay=self.weight_decay, 
                                    lr=self.learning_rate)

    def evaluate(self, prediction, label):
        """ Apply crtierion function eval model's prediction.
        """
        loss = self.criterion(prediction, label)
        _, ind = torch.max(prediction, 1)
        accuracy = ind==label   
        return loss, accuracy


    @abc.abstractmethod
    def predict(self, trial, display_prediction=False):
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


    @abc.abstractmethod:
    def create_model(self):
        """ Create underlying model.
        """
        return


    @abc.abstractmethod
    def format_inputs(trial):
        """ Format inputs for model.
        """
        return


    def display_prediction(self, trial, prediction_dist):
        """ Print and return informaion pertaining to a prediction.
        """
        _, predicted_utt_ind = torch.max(prediction_dist, 1)
        predicted_utt_ind = predicted_utt_ind.data.numpy()[0][0] # extract from tensor

        target_name = self.obj_inds_to_names[str(trial['target_ind'])]
        alt1_name   = self.obj_inds_to_names[str(trial['alt1_ind'])]
        alt2_name   = self.obj_inds_to_names[str(trial['alt2_ind'])]
        predicted_utt_name = self.utt_inds_to_names[str(predicted_utt_ind)]
        label_utt_name     = self.utt_inds_to_names[str(trial['utterance'])]

        print '\nCondition: {}'.format(condition)
        print ' Target: {}'.format(target_name)
        print ' Alt 1: {}'.format(alt1_name)
        print ' Alt 2: {}'.format(alt2_name)
        print 'Label: {}'.format(label_utt_name)
        print 'Prediction: {}'.format(predicted_utt_name)
        print 'Correct? {}'.format(predicted_utt_ind==label_utt_ind)

        return target_name, predicted_utt_name, label_utt_name


    def save_model_details(self):
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


class EmbbedingType(object):
    """ Enumeration of possible embedding types.
    """
    ONE_HOT = 0  # One Hot Vector 


# ----------------
# HELPER FUNCTIONS
# ----------------


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


def establish_seed(seed=3):
    random.seed(seed)
