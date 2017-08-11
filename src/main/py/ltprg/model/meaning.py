import torch
import torch.nn as nn
from ltprg.model.seq import sort_seq_tensors, unsort_seq_tensors

class MeaningModel(nn.Module):

    def __init__(self):
        super(MeaningModel, self).__init__()

    def forward(self, utterance, world, observation):
        """ Computes batch of meaning matrices """
        pass

class MeaningModelIndexedWorld(MeaningModel):

    def __init__(self, world_input_size):
        super(MeaningModelIndexedWorld, self).__init__()
        self._world_input_size = world_input_size

    def _meaning(self, utterance, input):
        """ Computes batch of meanings from batches of utterances and inputs """
        pass

    def forward(self, utterance, world, observation):
        inputs_per_observation = observation.size(1)/self._world_input_size
        observation = observation.view(observation.size(0), inputs_per_observation, self._world_input_size)
        input = torch.gather(observation, 1, world.long().unsqueeze(2).repeat(1,1,observation.size(2)))
        return self._construct_meaning(utterance, input)

    # utt is Batch x utterance prior size x utt length
    # input is Batch x world prior size x input size
    def _construct_meaning(self, utt, input):
        utt_batch = None
        utt_prior_size = None
        if not isinstance(utt, tuple):
            utt_exp =  utt.unsqueeze(1).expand(utt.size(0),input.size(1),utt.size(1),utt.size(2)).transpose(1,2)
            if not utt_exp.is_contiguous():
                utt_exp = utt_exp.contiguous()
            utt_batch = utt_exp.view(-1,utt.size(2))
            utt_prior_size = utt.size(1)
        else:
            # Handle utt represnted as tuple of tensors (like in case of sequences with lengths)
            utt_parts = []
            for i in range(len(utt)):
                utt_part = None
                if len(utt[i].size()) == 3:
                    utt_exp =  utt[i].unsqueeze(1).expand(utt[i].size(0),input.size(1),utt[i].size(1),utt[i].size(2)).transpose(1,2)
                    if not utt_exp.is_contiguous():
                        utt_exp = utt_exp.contiguous()
                    utt_part = utt_exp.view(-1,utt[i].size(2))
                else:
                    utt_exp =  utt[i].unsqueeze(1).expand(utt[i].size(0),input.size(1),utt[i].size(1)).transpose(1,2)
                    if not utt_exp.is_contiguous():
                        utt_exp = utt_exp.contiguous()
                    utt_part = utt_exp.view(-1)
                utt_parts.append(utt_part)
            utt_batch = tuple(utt_parts)
            utt_prior_size = utt[0].size(1)

        input_exp = input.unsqueeze(1).expand(input.size(0),utt_prior_size,input.size(1),input.size(2))
        if not input_exp.is_contiguous():
            input_exp = input_exp.contiguous()
        input_batch = input_exp.view(-1,input.size(2))

        meaning = self._meaning(utt_batch, input_batch)

        return meaning.view(input.size(0), utt_prior_size, input.size(1))

class MeaningModelIndexedWorldSequentialUtterance(MeaningModelIndexedWorld):
    def __init__(self, world_input_size, seq_model):
        super(MeaningModelIndexedWorldSequentialUtterance, self).__init__(world_input_size)
        self._seq_model = seq_model
        self._decoder = nn.Linear(seq_model.get_hidden_size(), 1)
        self._decoder_nl = nn.Sigmoid()

    def _meaning(self, utterance, input):
        seq = utterance[0].transpose(0,1)
        seq_length = utterance[1]
        sorted_seq, sorted_length, sorted_inputs, sorted_indices = sort_seq_tensors(seq, seq_length, inputs=[input])
        output, hidden = self._seq_model(seq_part=sorted_seq, seq_length=sorted_length, input=sorted_inputs[0])
        decoded = self._decoder(hidden.view(-1, hidden.size(0)*hidden.size(2)))
        output = self._decoder_nl(decoded)
        return unsort_seq_tensors(sorted_indices, [output])[0]
