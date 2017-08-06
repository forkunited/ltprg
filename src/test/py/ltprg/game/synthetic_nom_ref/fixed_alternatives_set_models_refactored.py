from __future__ import division

import abc
from basic_model import BasicModel, ModelType, EmbeddingType, one_hot
from network_components import MLP
from RSA import model_speaker_1
import torch
import torch.nn as nn
from torch.autograd import Variable


""" Models that leverage a fixed alternative set.
"""

class FixedAlternativeSetModel(BasicModel):
    """ Superclass reserved for shared components needed for models with 
      fixed alternative sets. Also known as FASM.
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self, model_name, model_type, hidden_szs, hiddens_nonlinearity,
                 utt_set_sz, obj_set_sz, obj_embedding_type, utt_dict, obj_dict,
                 weight_decay, learning_rate, rsa_params,
                 save_path):
        super(FixedAlternativeSetModel, self).__init__(
            model_name, model_type, hidden_szs, hiddens_nonlinearity,
            utt_set_sz, obj_set_sz, obj_embedding_type, utt_dict, obj_dict,
            weight_decay, learning_rate, rsa_params,
            save_path)
        assert self.obj_embedding_type == embbedingType.ONE_HOT

class FASM_ERSA(FixedAlternativeSetModel):
    """ EXPLICIT RSA MODEL ('ersa'): given an object embedding, neural network
        produces truthiness vals between 0 and 1 for each 
        utterance in the alternatives set. Each object in a trial 
        is fed through the network, producing a lexicon that is 
        then fed to RSA. RSA returns a level-1 speaker distribution, 
        P(u | target, context, L)
    """

    def create_model(self):
        """ Create underlying model.
        """
        in_sz = self.obj_set_sz
        return MLP(in_sz, self.hidden_szs, self.utt_set_sz, 
             self.hiddens_nonlinearity, 'sigmoid')


    def format_inputs(trial):
        """ Format inputs for model.
            trial (dict) {
                          'alt1_ind': a,
                          'alt2_ind': b,
                          'target_ind': c
                          }
        """
        return Variable(torch.cat(
                    [one_hot(trial['alt1_ind'], self.obj_set_sz),
                    one_hot(trial['alt1_ind'], self.obj_set_sz),
                    one_hot(trial['target_ind'] ,self.obj_set_sz)], 0))


    def predict(self, trial, display_prediction=False,
                use_gold_standard_lexicon=False):
        """ Make prediction for specified trial.
        """
        # inputs are 2D tensors
        inputs = self.format_inputs(trial)

        # forward pass
        outputs = self.model.forward(inputs) # MLP forward

        # Gold standard comparison
        if use_gold_standard_lexicon:
            # uses ground-truth lexicon (for comparison w/ 
            # model predictions); grab objects for this trial
            inds = Variable(torch.LongTensor(
                [trial['alt1_ind'], trial['alt2_ind'], trial['target_ind']]))
            lexicon = torch.index_select(self.rsa_params.gold_standard_lexicon, 1, inds)
        else:
            # uses learned params
            lexicon = torch.transpose(outputs, 0, 1)

        # feed MLP output into RSA, using learned params
        speaker_table = model_speaker_1(lexicon, self.rsa_params)

        # pull dist over utterances for target obj
        outputs = speaker_table[2, :].unsqueeze(0)

        # format label
        label = Variable(torch.LongTensor([trial['utterance']]))

        # display, if necessary
        if display_prediction:
          self.display_prediction(trial, outputs)
        
        return outputs, label


class FASM_NNWC(FixedAlternativeSetModel):
    """ NEURAL NETWORK WITH CONTEXT MODEL ('nnwc'): produces distribution over 
        utterances in the fixed alternatives set given a trial's 
        concatenated object embeddings, with target object in final 
        position
    """

    def create_model(self):
        """ Create underlying model.
        """
        in_sz = self.obj_set_sz * 3
        return MLP(in_sz, self.hidden_szs, self.utt_set_sz, 
                     self.hiddens_nonlinearity, 'logSoftmax')


    def format_inputs(tiral):
        """ Format inputs for model.
            trial (dict) {
                          'alt1_ind': a,
                          'alt2_ind': b,
                          'target_ind': c
                          }
        """
        return Variable(torch.cat(
                    [one_hot(trial['alt1_ind'], self.obj_set_sz),
                    one_hot(trial['alt1_ind'], self.obj_set_sz),
                    one_hot(trial['target_ind'] ,self.obj_set_sz)], 0))


    def predict(self, trial, display_prediction=False,
                use_gold_standard_lexicon=False):
        """ Make prediction for specified trial.
        """
        # inputs are 2D tensors
        inputs = self.format_inputs(trial)

        # forward pass
        outputs = self.model.forward(inputs) # MLP forward

        # Gold standard comparison
        if use_gold_standard_lexicon:
            # uses ground-truth lexicon (for comparison w/ 
            # model predictions); grab objects for this trial
            inds = Variable(torch.LongTensor(
                [trial['alt1_ind'], trial['alt2_ind'], trial['target_ind']]))
            lexicon = torch.index_select(self.rsa_params.gold_standard_lexicon, 1, inds)

            # pass through RSA
            speaker_table = model_speaker_1(lexicon, self.rsa_params)

            # pull dist over utterances for target obj
            outputs = speaker_table[2, :].unsqueeze(0)

        # format label
        label = Variable(torch.LongTensor([trial['utterance']]))

        # display, if necessary
        if display_prediction:
          self.display_prediction(trial, outputs)
        
        return outputs, label


class FASM_NNWOC(FixedAlternativeSetModel):
    """ NEURAL NETWORK WITHOUT CONTEXT MODEL ('nnwoc') produces distribution 
        over utterances given target object emebdding only
    """

    def create_model(self):
        """ Create underlying model.
        """
        in_sz = self.obj_set_sz
        return MLP(in_sz, self.hidden_szs, self.utt_set_sz, 
                     self.hiddens_nonlinearity, 'logSoftmax')


    def format_inputs(trial):
        """ Format inputs for model.
            trial (dict) {
                          'target_ind': c
                          }
        """
        return Variable(one_hot(trial['target_ind'], self.obj_set_sz))


    def predict(self, trial, display_prediction=False,
                use_gold_standard_lexicon=False):
        """ Make prediction for specified trial.
        """
        # inputs are 2D tensors
        inputs = self.format_inputs(trial)

        # forward pass
        outputs = self.model.forward(inputs) # MLP forward

        # Gold standard comparison
        if use_gold_standard_lexicon:
            # uses ground-truth lexicon (for comparison w/ 
            # model predictions); grab objects for this trial
            inds = Variable(torch.LongTensor(
                [trial['alt1_ind'], trial['alt2_ind'], trial['target_ind']]))
            lexicon = torch.index_select(self.rsa_params.gold_standard_lexicon, 1, inds)

            # pass through RSA
            speaker_table = model_speaker_1(lexicon, self.rsa_params)
            
            # pull dist over utterances for target obj
            outputs = speaker_table[2, :].unsqueeze(0)

        # format label
        label = Variable(torch.LongTensor([trial['utterance']]))

        # display, if necessary
        if display_prediction:
          self.display_prediction(trial, outputs)
        
        return outputs, label