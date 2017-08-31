from __future__ import division

import abc
from basic_model import BasicModel, ModelType, EmbeddingType, one_hot
from network_components import MLP
from rsa import model_speaker_1
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
        assert self.obj_embedding_type == EmbeddingType.ONE_HOT

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
        return MLP(self.obj_set_sz, self.hidden_szs, self.utt_set_sz, 
             self.hiddens_nonlinearity, 'sigmoid')


    def format_inputs(self, trial):
        """ Format inputs for model.
            trial (dict) {
                          'alt1_ind': a,
                          'alt2_ind': b,
                          'target_ind': c
                          }
        """
        return Variable(torch.cat(
                    [one_hot(trial['alt1_ind'], self.obj_set_sz),
                    one_hot(trial['alt2_ind'], self.obj_set_sz),
                    one_hot(trial['target_ind'], self.obj_set_sz)], 0))


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
                [trial['alt1_ind'], trial['alt2_ind'], trial['target_ind']]).type(self.dtype))
            lexicon = torch.index_select(self.rsa_params.gold_standard_lexicon, 1, inds)
        else:
            # uses learned params
            lexicon = torch.transpose(outputs, 0, 1)

        # feed MLP output into RSA, using learned params
        speaker_table = model_speaker_1(lexicon, self.rsa_params)

        # pull dist over utterances for target obj
        outputs = speaker_table[2, :].unsqueeze(0)

        # format label
        label = Variable(torch.LongTensor([trial['utterance']]).type(self.dtype))

        # display, if necessary
        if display_prediction:
          self.display_prediction(trial, outputs)
        
        return outputs, label


class FASM_NN(FixedAlternativeSetModel):
    """ Neural Network Model. Prediction is the same for 
        FASM_NNWC and FASM_NNWOC. So this has been factored out here.
    """
    def predict(self, trial, display_prediction=False,
                use_gold_standard_lexicon=False):
        """ Make prediction for specified trial.
        """
        # inputs are 2D tensors
        inputs = self.format_inputs(trial)

        # forward pass
        pred = self.model.forward(inputs) # MLP forward

        # Gold standard comparison
        if use_gold_standard_lexicon:
            # uses ground-truth lexicon (for comparison w/ 
            # model predictions); grab objects for this trial
            inds = Variable(torch.LongTensor(
                [trial['alt1_ind'], trial['alt2_ind'], trial['target_ind']]).type(self.dtype))
            lexicon = torch.index_select(self.rsa_params.gold_standard_lexicon, 1, inds)

            # pass through RSA
            speaker_table = model_speaker_1(lexicon, self.rsa_params)

            # pull dist over utterances for target obj
            pred = speaker_table[2, :].unsqueeze(0)

        # format label
        label = Variable(torch.LongTensor([trial['utterance']]).type(self.dtype))

        # display, if necessary
        if display_prediction:
          self.display_prediction(trial, pred)
        
        return pred, label


class FASM_NNWC(FASM_NN):
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


    def format_inputs(self, trial):
        """ Format inputs for model.
            trial (dict) {
                          'alt1_ind': a,
                          'alt2_ind': b,
                          'target_ind': c
                          }
        """
        return Variable(torch.cat(
                    [one_hot(trial['alt1_ind'], self.obj_set_sz),
                    one_hot(trial['alt2_ind'], self.obj_set_sz),
                    one_hot(trial['target_ind'] ,self.obj_set_sz)], 1))


class FASM_NNWOC(FASM_NN):
    """ NEURAL NETWORK WITHOUT CONTEXT MODEL ('nnwoc') produces distribution 
        over utterances given target object emebdding only
    """

    def create_model(self):
        """ Create underlying model.
        """
        return MLP(self.obj_set_sz, self.hidden_szs, self.utt_set_sz, 
                     self.hiddens_nonlinearity, 'logSoftmax')


    def format_inputs(self, trial):
        """ Format inputs for model.
            trial (dict) {
                          'target_ind': c
                          }
        """
        return Variable(one_hot(trial['target_ind'], self.obj_set_sz))


class FASM_ERSA_CTS(FixedAlternativeSetModel):
    """ EXPLICIT RSA MODEL ('ersa'): given an object embedding, neural network
        produces truthiness vals between 0 and 1 for each 
        utterance in the alternatives set. Each object in a trial 
        is fed through the network, producing a lexicon that is 
        then fed to RSA. RSA returns a level-1 speaker distribution, 
        P(u | target, context, L). This model differs from FASM_ERSA in
        that it uses the CTS (concat-to-single) architecture, where
        the object representation is concatenated with a potential
        utterance (embedding) to be applied to the given object.
    """

    def create_model(self):
        """ Create underlying model.
        """
        in_sz = self.obj_set_sz + self.utt_set_sz
        return MLP(in_sz, self.hidden_szs, 1, 
             self.hiddens_nonlinearity, 'sigmoid')


    def format_inputs(self, trial):
        """ Format inputs for model.
            trial (dict) {
                          'alt1_ind': a,
                          'alt2_ind': b,
                          'target_ind': c
                          }
        """
        alt1_embedding = torch.cat([
            one_hot(trial['alt1_ind'], self.obj_set_sz),
            one_hot(trial['utterance'], self.utt_set_sz)
        ], 1)
        alt2_embedding = torch.cat([
            one_hot(trial['alt2_ind'], self.obj_set_sz),
            one_hot(trial['utterance'], self.utt_set_sz)
        ], 1)
        target_embedding = torch.cat([
            one_hot(trial['target_ind'], self.obj_set_sz),
            one_hot(trial['utterance'], self.utt_set_sz)
        ], 1)
        return Variable(torch.cat([
                    alt1_embedding,
                    alt2_embedding,
                    target_embedding
                ], 0))


    def predict(self, trial, display_prediction=False,
                use_gold_standard_lexicon=False):
        """ Make prediction for specified trial.
        """
        # Gold standard comparison
        if use_gold_standard_lexicon:
            # uses ground-truth lexicon (for comparison w/ 
            # model predictions); grab objects for this trial
            inds = Variable(torch.LongTensor(
                [trial['alt1_ind'], trial['alt2_ind'], trial['target_ind']]).type(self.dtype))
            lexicon = torch.index_select(self.rsa_params.gold_standard_lexicon, 1, inds).type(self.dtype)
        else:
            # For CTS models, the output is a single probability for
            # the object-utterance pairing. We need to compute these 
            # probabilities for each obj-utt pairings.
            lexicon = None
            for utterance in range(self.utt_set_sz):
                trial_with_new_utt = trial.copy()
                trial_with_new_utt['utterance'] = utterance
                inputs = self.format_inputs(trial_with_new_utt)
                truthiness_values = torch.transpose(
                    self.model.forward(inputs.type(self.dtype)),
                    0,
                    1
                )
                if lexicon is None:
                    # First utterance, truthiness values
                    lexicon = truthiness_values
                else:
                    lexicon = torch.cat([lexicon, truthiness_values])

        # feed MLP output into RSA, using learned params
        speaker_table = model_speaker_1(lexicon.type(self.dtype), self.rsa_params)

        # pull dist over utterances for target obj
        pred = speaker_table[2, :].unsqueeze(0)

        # format label
        label = Variable(torch.LongTensor([trial['utterance']]))

        # display, if necessary
        if display_prediction:
            self.display_prediction(trial, pred)

        return pred.type(self.dtype), label.type(self.dtype)


class FASM_NN_CTS(FixedAlternativeSetModel):
    """ Neural network model for FASM_NNWC_CTS and FASM_NNWOC_CTS.
        Both share the same prediction function so it's been
        factored out here.
    """
    def predict(self, trial, display_prediction=False,
                use_gold_standard_lexicon=False):
        """ Make prediction for specified trial.
        """
        if use_gold_standard_lexicon:
            # uses ground-truth lexicon (for comparison w/ 
            # model predictions); grab objects for this trial
            inds = Variable(torch.LongTensor(
                [trial['alt1_ind'], trial['alt2_ind'], trial['target_ind']]).type(self.dtype))
            lexicon = torch.index_select(self.rsa_params.gold_standard_lexicon, 1, inds)

            # pass through RSA
            speaker_table = model_speaker_1(lexicon, self.rsa_params)

            # pull dist over utterances for target obj
            pred = speaker_table[2, :].unsqueeze(0)
        else:
            # Concat-To-Single (CTS) models provide a single
            # truthiness value for applying a given utterance
            # to a given trial. Here we find this 
            # probability across all possible utterances
            # apply a final softmax layer and then 
            # return the resultant label.
            utt_scores = None
            for utterance in range(self.utt_set_sz):
                trial_with_new_utt = trial.copy()
                trial_with_new_utt['utterance'] = utterance
                inputs = self.format_inputs(trial_with_new_utt)
                utt_score = self.model.forward(inputs)
                if utt_scores is None:
                    utt_scores = utt_score
                else:
                    utt_scores = torch.cat([utt_scores, utt_score], 1)
            m = nn.LogSoftmax()
            pred = m(utt_scores)

        # format label
        label = Variable(torch.LongTensor([trial['utterance']]).type(self.dtype))

        # display, if necessary
        if display_prediction:
          self.display_prediction(trial, pred)

        return pred, label

class FASM_NNWC_CTS(FASM_NN_CTS):
    """ NEURAL NETWORK WITH CONTEXT MODEL ('nnwc'): produces distribution over 
        utterances in the fixed alternatives set given a trial's 
        concatenated object embeddings, with target object in final 
        position. Additionally it utilizes a CTS (concat-to-single)
        model where this input is further concatenated with a potential
        utterance embedding.
    """

    def create_model(self):
        """ Create underlying model.
        """
        in_sz = self.obj_set_sz * 3 + self.utt_set_sz
        return MLP(in_sz, self.hidden_szs, 1, 
                     self.hiddens_nonlinearity, 'sigmoid')


    def format_inputs(self, trial):
        """ Format inputs for model.
            trial (dict) {
                          'alt1_ind': a,
                          'alt2_ind': b,
                          'target_ind': c
                          }
        """
        return Variable(torch.cat([
                    one_hot(trial['alt1_ind'], self.obj_set_sz),
                    one_hot(trial['alt2_ind'], self.obj_set_sz),
                    one_hot(trial['target_ind'], self.obj_set_sz),
                    one_hot(trial['utterance'], self.utt_set_sz)
                ], 1))


class FASM_NNWOC_CTS(FASM_NN_CTS):
    """ NEURAL NETWORK WITHOUT CONTEXT MODEL ('nnwoc') produces 
        a "truthiness" scalar given target object embedding concatenated
        with the desired utterance embedding.
    """

    def create_model(self):
        """ Create underlying model.
        """
        in_sz = self.obj_set_sz + self.utt_set_sz
        return MLP(in_sz, self.hidden_szs, 1, 
                     self.hiddens_nonlinearity, 'sigmoid')


    def format_inputs(self, trial):
        """ Format inputs for model.
            trial (dict) {
                          'target_ind': c
                          }
        """
        return Variable(torch.cat([
            one_hot(trial['target_ind'], self.obj_set_sz),
            one_hot(trial['utterance'], self.utt_set_sz)
            ], 1))
