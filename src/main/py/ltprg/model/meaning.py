import torch
import torch.nn as nn
from ltprg.model.seq import sort_seq_tensors, unsort_seq_tensors, SequenceModel
from torch.autograd import Variable

class MeaningModel(nn.Module):

    def __init__(self):
        super(MeaningModel, self).__init__()

    def forward(self, utterance, world, observation):
        """ Computes batch of meaning matrices """
        pass

    def on_gpu(self):
        return next(self.parameters()).is_cuda

    def save(self, model_path):
        init_params = self._get_init_params()
        model_obj = dict()
        model_obj["init_params"] = init_params
        model_obj["state_dict"] = self.state_dict()
        model_obj["meaning_type"] = type(self).__name__
        torch.save(model_obj, model_path)

    @staticmethod
    def load(model_path):
        model_obj = torch.load(model_path)
        init_params = model_obj["init_params"]
        state_dict = model_obj["state_dict"]
        meaning_type = model_obj["meaning_type"]

        model = None
        if meaning_type == "MeaningModelIndexedWorldSequentialUtterance":
            model = MeaningModelIndexedWorldSequentialUtterance.make(init_params)
        model.load_state_dict(state_dict)

        return model

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
        world_indices = world.long().unsqueeze(2).repeat(1,1,observation.size(2))
        if self.on_gpu():
            world_indices = world_indices.cuda()
        input = torch.gather(observation, 1, world_indices)
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

        #return meaning.view(input.size(0), input.size(1), utt_prior_size).transpose(1,2)
        return meaning.view(input.size(0), utt_prior_size, input.size(1))

class SequentialUtteranceInputType:
    IN_SEQ = "IN_SEQ"
    OUT_SEQ = "OUT_SEQ"

class MeaningModelIndexedWorldSequentialUtterance(MeaningModelIndexedWorld):
    def __init__(self, world_input_size, seq_model, input_type=SequentialUtteranceInputType.IN_SEQ, encode_input=False, enc_size=0):
        super(MeaningModelIndexedWorldSequentialUtterance, self).__init__(world_input_size)

        self._init_params = dict()
        self._init_params["world_input_size"] = world_input_size
        self._init_params["seq_model"] = seq_model._get_init_params()
        self._init_params["input_type"] = input_type
        self._init_params["encode_input"] = encode_input
        self._init_params["enc_size"] = enc_size

        self._seq_model = seq_model

        if input_type == SequentialUtteranceInputType.IN_SEQ:
            self._decoder = nn.Linear(seq_model.get_hidden_size()*seq_model.get_directions(), 1)
        else:
            self._encode_input = encode_input

            if encode_input:
                self._input_enc = nn.Linear(self._world_input_size, enc_size)
                self._input_nl = nn.Tanh()
                self._enc_size = enc_size
            else:
                self._enc_size = self._world_input_size

            self._decoder_mu = nn.Linear(seq_model.get_hidden_size()*seq_model.get_directions(), self._enc_size)
            self._decoder_Sigma = nn.Linear(seq_model.get_hidden_size()*seq_model.get_directions(), self._enc_size * self._enc_size)
            self._decoder_Sigma.bias = nn.Parameter(torch.eye(self._enc_size).view(self._enc_size * self._enc_size))
            self._mse = nn.MSELoss()

        self._decoder_nl = nn.Sigmoid()
        self._input_type = input_type

    def _get_init_params(self):
        return self._init_params

    @staticmethod
    def make(init_params):
        world_input_size = init_params["world_input_size"]
        seq_model = SequenceModel.make(init_params["seq_model"])
        input_type = init_params["input_type"]
        encode_input = init_params["encode_input"]
        enc_size = init_params["enc_size"]
        return MeaningModelIndexedWorldSequentialUtterance(world_input_size, seq_model, input_type=input_type, encode_input=encode_input, enc_size=enc_size)

    def get_seq_model(self):
        return self._seq_model

    def _meaning(self, utterance, input):
        seq = utterance[0].transpose(0,1)
        seq_length = utterance[1]
        sorted_seq, sorted_length, sorted_inputs, sorted_indices = sort_seq_tensors(seq, seq_length, inputs=[input], on_gpu=self.on_gpu())

        output = None
        if self._input_type == SequentialUtteranceInputType.IN_SEQ:
            output, hidden = self._seq_model(seq_part=sorted_seq, seq_length=sorted_length, input=sorted_inputs[0])
            if isinstance(hidden, tuple): # Handle LSTM
                hidden = hidden[0]
            decoded = self._decoder(hidden.transpose(0,1).contiguous().view(-1, hidden.size(0)*hidden.size(2)))
            output = self._decoder_nl(decoded)
        else:
            output, hidden = self._seq_model(seq_part=sorted_seq, seq_length=sorted_length, input=None)
            if isinstance(hidden, tuple): # Handle LSTM
                hidden = hidden[0]
            mu = self._decoder_mu(hidden.transpose(0,1).contiguous().view(-1, hidden.size(0)*hidden.size(2)))

            #score = Variable(torch.zeros(mu.size(0)))
            #for i in range(mu.size(0)):
            #    score[i] = -self._mse(mu[i], sorted_inputs[0][i])
            #output = score

            #score = - self._mse(mu, sorted_inputs[0])
            #output = self._decoder_nl(score)
            inp = sorted_inputs[0]
            if self._encode_input:
                #inp = self._input_enc(sorted_inputs[0])
                inp = self._input_nl(self._input_enc(sorted_inputs[0]))

            Sigma_flat = self._decoder_Sigma(hidden.transpose(0,1).contiguous().view(-1, hidden.size(0)*hidden.size(2)))
            Delta = inp - mu
            Sigma = Sigma_flat.view(-1, self._enc_size, self._enc_size)
            score = - Delta.unsqueeze(1).bmm(Sigma).bmm(Delta.unsqueeze(1).transpose(1,2)).squeeze()
            output = score

        return unsort_seq_tensors(sorted_indices, [output])[0]
