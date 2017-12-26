import sys
import time
import torch.nn as nn
import torch.nn.init as init
import numpy as np
import torch
import abc
import copy
import mung.torch_ext.eval
from torch.autograd import Variable
from mung.feature import Symbol

def sort_seq_tensors(seq, length, inputs=None, on_gpu=False):
    sorted_length, sorted_indices = torch.sort(length, 0, True)
    if on_gpu:
        sorted_indices = sorted_indices.cuda()
    sorted_seq = seq.transpose(0,1)[sorted_indices].transpose(0,1)

    if inputs is not None:
        sorted_inputs = [input[sorted_indices] for input in inputs]
        return sorted_seq, sorted_length, sorted_inputs, sorted_indices
    else:
        return sorted_seq, sorted_length, sorted_indices

def unsort_seq_tensors(sorted_indices, tensors):
    _, unsorted_indices = torch.sort(sorted_indices, 0, False)
    return [tensor[unsorted_indices] for tensor in tensors]

class RNNType:
    LSTM = "LSTM"
    GRU = "GRU"

class DataParameter:
    SEQ = "seq"
    INPUT  = "input"

    @staticmethod
    def make(seq="seq", input="input"):
        data_parameters = dict()
        data_parameters[DataParameter.SEQ] = seq
        data_parameters[DataParameter.INPUT] = input
        return data_parameters

class SamplingMode:
    FORWARD = "FORWARD"
    BEAM = "BEAM"

class VariableLengthNLLLoss(nn.Module):
    def __init__(self, norm_dim=False):
        """
        Constructs NLLLoss for variable length sequences.

        Borrowed from
        https://github.com/ruotianluo/neuraltalk2.pytorch/blob/master/misc/utils.py
        """
        super(VariableLengthNLLLoss, self).__init__()
        self._norm_dim = norm_dim

    def _to_contiguous(self, tensor):
        if tensor.is_contiguous():
            return tensor
        else:
            return tensor.contiguous()

    def forward(self, input, target, mask):
        # truncate to the same size
        #target = target[:, :input.size(1)]
        #mask =  mask[:, :input.size(1)]
        input = self._to_contiguous(input).view(-1, input.size(2))
        target = self._to_contiguous(target).view(-1, 1)
        mask = self._to_contiguous(mask).view(-1, 1)
        output = - input.gather(1, target)
        output = output *  mask

        if not self._norm_dim:
            return torch.sum(output) / torch.sum(mask)
        else:
            return (torch.sum(output), torch.sum(mask))


class SequenceModel(nn.Module):
    def __init__(self, name, hidden_size, bidir):
        super(SequenceModel, self).__init__()
        self._hidden_size = hidden_size
        self._name = name

        self._bidir = bidir
        self._directions = 1
        if self._bidir:
            self._directions = 2

    # @abc.abstractmethod
    def _init_hidden(self, batch_size, input=None):
        """ Initializes hidden state, possibly given some input """
        pass

    # @abc.abstractmethod
    def _forward_from_hidden(self, hidden, seq_part, seq_length, input=None):
        """ Runs the model forward from a given hidden state """
        pass

    def get_hidden_size(self):
        return self._hidden_size

    def get_directions(self):
        return self._directions

    def get_name(self):
        return self._name

    def on_gpu(self):
        return next(self.parameters()).is_cuda

    def forward(self, seq_part=None, seq_length=None, input=None):
        if seq_part is None:
            n = 1
            if input is not None:
                n = input.size(0)
            seq_part = torch.Tensor([Symbol.index(Symbol.SEQ_START)]) \
                .repeat(n).long().view(1, n)
            seq_length = torch.ones(n)

        hidden = self._init_hidden(seq_length.size(0), input=input)
        return self._forward_from_hidden(hidden, seq_part, seq_length, input=input)

    def forward_batch(self, batch, data_parameters):
        input = None
        if DataParameter.INPUT in data_parameters and data_parameters[DataParameter.INPUT] in batch:
            input = Variable(batch[data_parameters[DataParameter.INPUT]])

        seq, length, mask = batch[data_parameters[DataParameter.SEQ]]
        length = length - 1
        seq_in = Variable(seq[:seq.size(0)-1]).long() # Input remove final token

        if self.on_gpu():
            seq_in = seq_in.cuda()
            if input is not None:
                input = input.cuda()

        model_out, hidden = self(seq_part=seq_in, seq_length=length, input=input)
        return model_out, hidden

    def loss(self, batch, data_parameters, loss_criterion):
        model_out, hidden = self.forward_batch(batch, data_parameters)
        seq, length, mask = batch[data_parameters[DataParameter.SEQ]]
        target_out = Variable(seq[1:seq.size(0)]).long() # Output (remove start token)

        if self.on_gpu():
            target_out = target_out.cuda()
            mask = mask.cuda()

        loss = loss_criterion(model_out, target_out[:model_out.size(0)], Variable(mask[:,1:(model_out.size(0)+1)]))
        return loss

    # NOTE: Assumes seq_part does not contain end tokens
    def sample(self, n_per_input=1, seq_part=None, max_length=15, input=None, heuristic=None, context=None, n_before_heuristic=100):
        n = 1
        input_count = 1
        samples_per_input = 1
        if heuristic is None:
            samples_per_input = n_per_input
        else:
            samples_per_input = n_before_heuristic

        if input is not None:
            if isinstance(input, Variable):
                input = input.data
            input_count = input.size(0)
            n = input.size(0) * samples_per_input
            input = input.repeat(1, samples_per_input).view(n, input.size(1))
            if self.on_gpu():
                input = input.cuda()

        if seq_part is not None:
            input_count = seq_part.size(1)
            n = seq_part.size(1) * samples_per_input
            seq_part = seq_part.repeat(samples_per_input, 1).view(seq_part.size(0), n)
            if isinstance(seq_part, Variable):
                seq_part = seq_part.data
        else:
            if input is None:
                n = samples_per_input
            seq_part = torch.Tensor([Symbol.index(Symbol.SEQ_START)]) \
                .repeat(n).long().view(1,n)

        if heuristic is not None:
            context = (context[0].unsqueeze(0).expand(samples_per_input, context[0].size(0), context[0].size(1)).contiguous().view(n, context[0].size(1)),
                       context[1].unsqueeze(0).expand(samples_per_input, context[1].size(0)).contiguous().view(n, 1))

        if self.on_gpu():
            seq_part = seq_part.cuda()

        end_idx = Symbol.index(Symbol.SEQ_END)
        ended = torch.zeros(n).long()
        ended_count = 0
        unit_length = torch.ones(n).long()
        seq_length = unit_length*seq_part.size(0)
        sample = copy.deepcopy(seq_part)

        output, hidden = self(seq_part=Variable(seq_part), seq_length=seq_length, input=Variable(input))
        for i in range(seq_part.size(0), max_length):
            output_dist = output[output.size(0)-1].exp()
            next_token = torch.multinomial(output_dist).data
            sample = torch.cat((sample, next_token.transpose(1,0)), dim=0)
            output, hidden = self._forward_from_hidden(hidden,
                                                       Variable(next_token.view(1, next_token.size(0))),
                                                       unit_length,
                                                       input=Variable(input))

            for j in range(next_token.size(0)):
                seq_length[j] += 1 - ended[j]
                if next_token[j][0] == end_idx:
                    ended[j] = 1
                    ended_count += 1

            if ended_count == n:
                break

        # Return a list... like beam search...
        ret_samples = []
        for i in range(input_count):
            input_in = None
            if input is not None:
                input_in = input[(i*samples_per_input):((i+1)*samples_per_input)]
            sample_in = sample[:,(i*samples_per_input):((i+1)*samples_per_input)]
            seq_length_in = seq_length[(i*samples_per_input):((i+1)*samples_per_input)]

            if heuristic is not None:
                context_in = (context[0][(i*samples_per_input):((i+1)*samples_per_input)], context[1][(i*samples_per_input):((i+1)*samples_per_input)])
                heuristic_output, _ = heuristic((sample_in, seq_length_in), Variable(input_in), None, context=context_in)
                top_indices = heuristic_output.topk(n_per_input)[1]
                sample_in = sample_in.transpose(0,1)[top_indices].transpose(0,1)
                seq_length_in = seq_length_in[top_indices.cpu()]

            # FIXME Add score at some point
            ret_samples.append((sample_in, seq_length_in, 0.0))
        return ret_samples

    # NOTE: Input is a batch of inputs
    def beam_search(self, beam_size=5, max_length=15, seq_part=None, input=None, heuristic=None, context=None):
        beams = []
        if seq_part is not None:
            seq_part = seq_part.transpose(1,0)

        if input is not None:
            if self.on_gpu():
                input = input.cuda()

            for i in range(input.size(0)):
                seq_part_i = None
                if seq_part is not None:
                    seq_part_i = seq_part[i].transpose(1,0)
                input_i = input[i]

                context_i = None
                input_index_i = None
                if context is not None:
                    context_i = context[0][i]
                    input_index_i = context[1][i]*torch.ones(1).long()
                    if self.on_gpu():
                        input_index_i = input_index_i.cuda()
                beams.append(self._beam_search_single(beam_size, max_length, seq_part=seq_part_i, input=input_i, heuristic=heuristic, context=(context_i, input_index_i)))
        else:
            beams.append(self._beam_search_single(beam_size, max_length, heuristic=heuristic))

        return beams

    def _beam_search_single(self, beam_size, max_length, seq_part=None, input=None, heuristic=None, context=None):
        if seq_part is None:
            seq_part = torch.Tensor([Symbol.index(Symbol.SEQ_START)]).long().view(1,1)
        else:
            if isinstance(seq_part, Variable):
                seq_part = seq_part.data
            seq_part = seq_part.view(seq_part.size(0), 1)

        if input is not None:
            if isinstance(input, Variable):
                input = input.data
            input = input.view(1, input.size(0))
            if context is not None and context[0] is not None:
                context = (context[0].view(1, context[0].size(0)), context[1].view(1, context[1].size(0)))

        end_idx = Symbol.index(Symbol.SEQ_END)
        ended = torch.zeros(1).long()
        unit_length = torch.ones(beam_size).long()
        seq_length = torch.ones(1).long()*seq_part.size(0)

        if self.on_gpu():
            seq_part = seq_part.cuda()
            ended = ended.cuda()

        output, hidden = self(seq_part=Variable(seq_part), seq_length=seq_length, input=Variable(input))
        if isinstance(hidden, tuple):
            hidden = tuple([h.repeat(1,1,beam_size).view(1,beam_size, h.size(2)) for h in hidden])
        else:
            hidden = hidden.repeat(1,1, beam_size).view(1, beam_size, hidden.size(2))

        beam = seq_part.repeat(1,beam_size).view(1,beam_size)
        scores = torch.zeros(1) #beam_size)

        # Output is len x batch x vocab
        vocab_size = output.size(2)

        # This mask is for ignoring all vocabulary extention scores except the
        # first on ended sequences
        ended_ignore_mask = torch.ones(vocab_size)
        ended_ignore_mask[0] = 0.0

        vocab = None
        vocab_rep = None
        heuristic_state = None
        heuristic_lengths = None
        beam_heuristic_lengths = None
        if heuristic is not None:
            vocab = torch.arange(0, vocab_size).long()
            vocab_rep = vocab.repeat(beam_size).unsqueeze(0)
            vocab = vocab.unsqueeze(0)
            heuristic_lengths = torch.zeros(vocab_size).long()
            beam_heuristic_lengths = torch.zeros(vocab_size*beam_size).long()
            if self.on_gpu():
                vocab = vocab.cuda()
                vocab_rep = vocab_rep.cuda()

        input_single = None
        if input is not None:
            input_single = input
            input = input.repeat(beam_size, 1)

        context_single = None
        if context is not None:
            context_single = context

        if self.on_gpu():
            beam = beam.cuda()
            scores = scores.cuda()
            ended = ended.cuda()
            ended_ignore_mask = ended_ignore_mask.cuda()

        for i in range(seq_part.size(0), max_length):
            output_dist = output[output.size(0)-1]

            # When a sequence ends, it needs to not be extended with multiple scores
            # So:
            # Ignores all extensions of ended sequences except the first by adding -Inf
            # before taking the top k scores
            ended_mat = ended.unsqueeze(1).expand_as(output_dist).float()
            ignore_mask = ended_ignore_mask.unsqueeze(0).expand_as(ended_mat)*ended_mat*float('-inf')
            ignore_mask[ignore_mask != ignore_mask] = 0.0 # Send nans to 0 (0*-inf = nan)

            next_scores = scores.unsqueeze(1).expand_as(output_dist) + (1.0-ended_mat)*output_dist.data + ignore_mask

            if heuristic is not None:
                seq_len = beam.size(0)
                # Sequence length x (vocab_size * beam_size tensor)
                # Beam sequences repeated in congtiguous blocks of vocab size...
                # to be extended with each element of vocab
                #expanded_beam = beam.unsqueeze(0).expand((vocab_size,seq_len,beam_size)) \
                #    .transpose(0,2).contiguous() \
                #    .view(seq_len,vocab_size*beam_size)
                expanded_beam = None
                lens = None
                if output_dist.size(0) > 1:
                    expanded_beam = beam.unsqueeze(0).expand((vocab_size,seq_len,beam_size)) \
                        .transpose(0,2).contiguous() \
                        .view(seq_len,vocab_size*beam_size)
                    expanded_beam = torch.cat((expanded_beam, vocab_rep), dim=0)
                    beam_heuristic_lengths[:] = seq_len + 1
                    lens = beam_heuristic_lengths
                else:
                    expanded_beam = seq_part.view(1,1).unsqueeze(0).expand((vocab_size,seq_len,1)) \
                        .transpose(0,2).contiguous() \
                        .view(seq_len, vocab_size)
                    expanded_beam = torch.cat((expanded_beam, vocab), dim=0)
                    heuristic_lengths[:] = seq_len + 1
                    lens = heuristic_lengths

                expanded_input = None
                if input is not None:
                    expanded_input = input_single.expand(expanded_beam.size(1), input_single.size(1))

                expanded_context = None
                if context is not None and context[0] is not None:
                    expanded_context = (context_single[0].expand(expanded_beam.size(1), context_single[0].size(1)).contiguous(), context_single[1].expand(expanded_beam.size(1), context_single[1].size(1)).contiguous())
                heuristic_output, heuristic_state = heuristic((expanded_beam, lens), expanded_input, heuristic_state, context=expanded_context)
                # Output is vector of scores (beam_0.v_0, beam_0.v_1,..., beam_1.v_1...)
                heuristic_output = heuristic_output.view(output_dist.size())
                next_scores += heuristic_output

            top_indices = next_scores.view(next_scores.size(0)*next_scores.size(1)).topk(beam_size)[1]
            top_seqs = top_indices / vocab_size
            top_exts = top_indices % vocab_size

            next_beam = torch.zeros(beam.size(0) + 1, beam_size).long()
            next_hidden = None
            if isinstance(hidden, tuple):
                next_hidden = tuple([Variable(torch.zeros(1, beam_size, h.size(2))) for h in hidden])
            else:
                next_hidden = Variable(torch.zeros(1, beam_size, hidden.size(2)))
            next_seq_length = torch.ones(beam_size).long()
            next_ended = torch.zeros(beam_size).long()
            scores = torch.zeros(beam_size)

            if self.on_gpu():
                next_beam = next_beam.cuda()
                if isinstance(hidden, tuple):
                    next_hidden = tuple([h.cuda() for h in next_hidden])
                else:
                    next_hidden = next_hidden.cuda()
                next_ended = next_ended.cuda()
                scores = scores.cuda()

            for j in range(beam_size):
                scores[j] = next_scores[top_seqs[j], top_exts[j]]
                next_beam[0:i,j] = beam[:,top_seqs[j]]
                next_beam[i,j] = top_exts[j]
                if isinstance(hidden, tuple):
                    for k in range(len(hidden)):
                        next_hidden[k][0,j] = hidden[k][0,top_seqs[j]]
                else:
                    next_hidden[0,j] = hidden[0,top_seqs[j]]
                next_seq_length[j] = seq_length[top_seqs[j]] + (1 - ended[top_seqs[j]])
                next_ended[j] = ended[top_seqs[j]]
                if top_exts[j] == end_idx and next_ended[j] != 1:
                    next_ended[j] = 1
            beam = next_beam
            hidden = next_hidden
            seq_length = next_seq_length
            ended = next_ended

            if sum(ended) == beam_size:
                break

            output, hidden = self._forward_from_hidden(hidden,
                                                       Variable(beam[i].view(1,beam[i].size(0))),
                                                       unit_length,
                                                       input=Variable(input))

        return beam, seq_length, scores


    def save(self, model_path):
        init_params = self._get_init_params()
        model_obj = dict()
        model_obj["init_params"] = init_params
        model_obj["state_dict"] = self.state_dict()
        model_obj["arch_type"] = type(self).__name__
        torch.save(model_obj, model_path)

    @staticmethod
    def load(model_path):
        model_obj = torch.load(model_path)
        init_params = model_obj["init_params"]
        state_dict = model_obj["state_dict"]
        arch_type = model_obj["arch_type"]

        model = SequenceModel.make(init_params)
        model.load_state_dict(state_dict)

        return model

    @staticmethod
    def make(init_params):
        model = None
        if arch_type == "SequenceModelInputEmbedded":
            model = SequenceModelInputEmbedded.make(init_params)
        elif arch_type == "SequenceModelInputToHidden":
            model = SequenceModelInputToHidden.make(init_params)
        elif arch_type == "SequenceModelNoInput":
            model = SequenceModelNoInput.make(init_params)
        return model

""" FIXME Put his back later maybe
class EvaluationSequential(ltprg.model.eval.Evaluation):
    __metaclass__ = abc.ABCMeta

    def __init__(self, name, data, data_parameters):
        super(EvaluationSequential, self).__init__(name, data, data_parameters)

        # Loading all the stuff on construction will be faster but hog
        # memory.  If it's a problem, then move this into the run method.
        batch = data.get_batch(0, data.get_size())
        seq, length, mask = batch[seq_view_name]

        self._name = name
        self._input_view_name = data_parameters[DataParameter.INPUT]
        self._seq_view_name = data_parameters[DataParameter.SEQ]
        self._data_input = Variable(batch[data_parameters[DataParameter.INPUT]])
        self._seq_length = length - 1
        self._seq_in = Variable(seq[:seq.size(0)-1]).long() # Input remove final token
        self._target_out = Variable(seq[1:seq.size(0)]).long() # Output (remove start token)
        self._mask = mask

    @abc.abstractmethod
    def run_helper(self, model, model_out, hidden):
        # Evaluates the model according to its output

    def run(self, model):
        model.eval()
        model_out, hidden = self(seq_part=self._seq_in,
                                 seq_length=self._seq_length,
                                 input=self._data_input)

        result = self._run_helper(model, model_out, hidden)

        model.train()
        return result
"""

class SequenceModelInputToHidden(SequenceModel):
    def __init__(self, name, seq_size, input_size, embedding_size, rnn_size,
                 rnn_layers, rnn_type=RNNType.GRU, dropout=0.5, bidir=False, input_layers=1):
        super(SequenceModelInputToHidden, self).__init__(name, rnn_size, bidir)

        self._init_params = dict()
        self._init_params["name"] = name
        self._init_params["seq_size"] = seq_size
        self._init_params["input_size"] = input_size
        self._init_params["embedding_size"] = embedding_size
        self._init_params["rnn_size"] = rnn_size
        self._init_params["rnn_layers"] = rnn_layers
        self._init_params["rnn_type"] = rnn_type
        self._init_params["dropout"] = dropout
        self._init_params["bidir"] = bidir
        self._init_params["input_layers"] = input_layers

        self._rnn_layers = rnn_layers
        self._rnn_type = rnn_type
        self._seq_size = seq_size

        self._input_layers = input_layers

        self._encoder = nn.Linear(input_size, rnn_size*rnn_layers*self._directions/(4**(input_layers-1)))
        self._encoder_nl = nn.Tanh()
        if self._input_layers == 2:
            self._encoder_0 = nn.Linear(rnn_size*rnn_layers*self._directions/(4**(input_layers-1)), rnn_size*rnn_layers*self._directions)
            self._encoder_0_nl = nn.Tanh()
        elif self._input_layers != 1:
            raise ValueError("Can only have 1 or 2 input layers")

        self._drop = nn.Dropout(dropout)
        self._emb = nn.Embedding(seq_size, embedding_size)
        self._rnn = getattr(nn, rnn_type)(embedding_size, rnn_size, rnn_layers, dropout=dropout, bidirectional=bidir)
        self._decoder = nn.Linear(rnn_size*self._directions, seq_size)
        self._softmax = nn.LogSoftmax()

        self._init_weights()

    def _get_init_params(self):
        return self._init_params

    def _init_hidden(self, batch_size, input=None):
        weight = next(self.parameters()).data

        hidden = self._encoder_nl(self._encoder(input))
        if self._input_layers > 1:
            hidden = self._encoder_0_nl(self._encoder_0(hidden))

        hidden = hidden.view(hidden.size()[0], self._rnn_layers*self._directions, self.get_hidden_size()).transpose(0,1).contiguous()

        if self._rnn_type == RNNType.GRU:
            return hidden
        else:
            return (hidden, \
                    Variable(weight.new(self._rnn_layers*self._directions, batch_size, self._hidden_size).zero_()))

    def _forward_from_hidden(self, hidden, seq_part, seq_length, input=None):
        emb_pad = self._drop(self._emb(seq_part))

        emb = nn.utils.rnn.pack_padded_sequence(emb_pad, seq_length.numpy(), batch_first=False)

        output, hidden = self._rnn(emb, hidden)

        output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=False)
        rnn_out_size = output.size()

        output = self._softmax(self._decoder(output.view(-1, rnn_out_size[2])))
        output = output.view(rnn_out_size[0], rnn_out_size[1], output.size(1))

        return output, hidden

    def _init_weights(self):
        init_range = 0.01

        #self._emb.weight.data.uniform_(-initrange, initrange)
        init.normal(self._emb.weight.data, mean=0.0, std=init_range)

        #self._encoder.bias.data.fill_(0)
        #self._encoder.weight.data.uniform_(-initrange, initrange)
        #init.normal(self._encoder.weight.data, mean=0.0, std=init_range)

        #self._decoder.bias.data.fill_(0)
        #self._decoder.weight.data.uniform_(-initrange, initrange)
        #init.normal(self._decoder.weight.data, mean=0.0, std=init_range)

    @staticmethod
    def make(init_params):
        name = init_params["name"]
        seq_size = init_params["seq_size"]
        input_size = init_params["input_size"]
        embedding_size = init_params["embedding_size"]
        rnn_size = init_params["rnn_size"]
        rnn_layers = init_params["rnn_layers"]
        rnn_type = init_params["rnn_type"]
        dropout = init_params["dropout"]

        bidir = False
        if "bidir" in init_params:
            bidir = init_params["bidir"]

        input_layers = 1
        if "input_layers" in init_params:
            input_layers = init_params["input_layers"]

        return SequenceModelInputToHidden(name, seq_size, input_size, embedding_size, rnn_size, rnn_layers, rnn_type=rnn_type, dropout=dropout, bidir=bidir, input_layers=input_layers)


class SequenceModelInputEmbedded(SequenceModel):
    def __init__(self, name, seq_size, input_size, embedding_size, rnn_size,
                 rnn_layers, rnn_type=RNNType.GRU, dropout=0.5, bidir=False):
        super(SequenceModelInputEmbedded, self).__init__(name, rnn_size, bidir)

        self._init_params = dict()
        self._init_params["name"] = name
        self._init_params["seq_size"] = seq_size
        self._init_params["input_size"] = input_size
        self._init_params["embedding_size"] = embedding_size
        self._init_params["rnn_size"] = rnn_size
        self._init_params["rnn_layers"] = rnn_layers
        self._init_params["rnn_type"] = rnn_type
        self._init_params["dropout"] = dropout
        self._init_params["bidir"] = bidir

        self._rnn_layers = rnn_layers
        self._rnn_type = rnn_type
        self._drop = nn.Dropout(dropout)
        self._emb = nn.Embedding(seq_size, embedding_size)
        self._rnn = getattr(nn, rnn_type)(embedding_size + input_size, rnn_size, rnn_layers, dropout=dropout, bidirectional=bidir)
        self._decoder = nn.Linear(rnn_size*self._directions, seq_size)
        self._softmax = nn.LogSoftmax()

        self._init_weights()

    def _get_init_params(self):
        return self._init_params

    def _init_hidden(self, batch_size, input=None):
        weight = next(self.parameters()).data
        if self._rnn_type == RNNType.GRU:
            return Variable(weight.new(self._rnn_layers*self._directions, batch_size, self._hidden_size).zero_())
        else:
            return (Variable(weight.new(self._rnn_layers*self._directions, batch_size, self._hidden_size).zero_()), \
                    Variable(weight.new(self._rnn_layers*self._directions, batch_size, self._hidden_size).zero_()))

    def _forward_from_hidden(self, hidden, seq_part, seq_length, input=None):
        emb_pad = self._drop(self._emb(seq_part))

        input_seq = input.unsqueeze(0).expand(emb_pad.size(0),emb_pad.size(1),input.size(1))
        emb_pad = torch.cat((emb_pad, input_seq), 2) # FIXME Is this right?

        emb = nn.utils.rnn.pack_padded_sequence(emb_pad, seq_length.numpy(), batch_first=False)

        output, hidden = self._rnn(emb, hidden)

        output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=False)
        rnn_out_size = output.size()

        output = self._softmax(self._decoder(output.view(-1, rnn_out_size[2])))
        output = output.view(rnn_out_size[0], rnn_out_size[1], output.size(1))

        return output, hidden

    def _init_weights(self):
        init_range = 0.01

        #self._emb.weight.data.uniform_(-initrange, initrange)
        init.normal(self._emb.weight.data, mean=0.0, std=init_range)

        #self._encoder.bias.data.fill_(0)
        #self._encoder.weight.data.uniform_(-initrange, initrange)
        #init.normal(self._encoder.weight.data, mean=0.0, std=init_range)

        #self._decoder.bias.data.fill_(0)
        #self._decoder.weight.data.uniform_(-initrange, initrange)
        #init.normal(self._decoder.weight.data, mean=0.0, std=init_range)

    @staticmethod
    def make(init_params):
        name = init_params["name"]
        seq_size = init_params["seq_size"]
        input_size = init_params["input_size"]
        embedding_size = init_params["embedding_size"]
        rnn_size = init_params["rnn_size"]
        rnn_layers = init_params["rnn_layers"]
        rnn_type = init_params["rnn_type"]
        dropout = init_params["dropout"]
        bidir = init_params["bidir"]
        return SequenceModelInputEmbedded(name, seq_size, input_size, embedding_size, rnn_size, rnn_layers, rnn_type=rnn_type, dropout=dropout, bidir=bidir)


class SequenceModelNoInput(SequenceModel):
    def __init__(self, name, seq_size, embedding_size, rnn_size,
                 rnn_layers, rnn_type=RNNType.GRU, dropout=0.5, bidir=False):
        super(SequenceModelNoInput, self).__init__(name, rnn_size, bidir)

        self._init_params = dict()
        self._init_params["name"] = name
        self._init_params["seq_size"] = seq_size
        self._init_params["embedding_size"] = embedding_size
        self._init_params["rnn_size"] = rnn_size
        self._init_params["rnn_layers"] = rnn_layers
        self._init_params["rnn_type"] = rnn_type
        self._init_params["dropout"] = dropout
        self._init_params["bidir"] = bidir

        self._rnn_layers = rnn_layers
        self._rnn_type = rnn_type
        self._drop = nn.Dropout(dropout)
        self._emb = nn.Embedding(seq_size, embedding_size)
        self._rnn = getattr(nn, rnn_type)(embedding_size, rnn_size, rnn_layers, dropout=dropout, bidirectional=bidir)
        self._decoder = nn.Linear(rnn_size*self._directions, seq_size)
        self._softmax = nn.LogSoftmax()

        self._init_weights

    def _get_init_params(self):
        return self._init_params

    def _init_hidden(self, batch_size, input=None):
        weight = next(self.parameters()).data
        if self._rnn_type == RNNType.GRU:
            return Variable(weight.new(self._rnn_layers*self._directions, batch_size, self._hidden_size).zero_())
        else:
            return (Variable(weight.new(self._rnn_layers*self._directions, batch_size, self._hidden_size).zero_()), \
                    Variable(weight.new(self._rnn_layers*self._directions, batch_size, self._hidden_size).zero_()))

    def _forward_from_hidden(self, hidden, seq_part, seq_length, input=None):
        emb_pad = self._drop(self._emb(seq_part))
        emb = nn.utils.rnn.pack_padded_sequence(emb_pad, seq_length.numpy(), batch_first=False)

        output, hidden = self._rnn(emb, hidden)

        output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=False)
        rnn_out_size = output.size()
        output = self._softmax(self._decoder(output.view(-1, rnn_out_size[2])))
        output = output.view(rnn_out_size[0], rnn_out_size[1], output.size(1))

        return output, hidden

    def _init_weights(self):
        init_range = 0.01

        #self._emb.weight.data.uniform_(-initrange, initrange)
        init.normal(self._emb.weight.data, mean=0.0, std=init_range)

        #self._encoder.bias.data.fill_(0)
        #self._encoder.weight.data.uniform_(-initrange, initrange)
        #init.normal(self._encoder.weight.data, mean=0.0, std=init_range)

        #self._decoder.bias.data.fill_(0)
        #self._decoder.weight.data.uniform_(-initrange, initrange)
        #init.normal(self._decoder.weight.data, mean=0.0, std=init_range)

    @staticmethod
    def make(init_params):
        name = init_params["name"]
        seq_size = init_params["seq_size"]
        embedding_size = init_params["embedding_size"]
        rnn_size = init_params["rnn_size"]
        rnn_layers = init_params["rnn_layers"]
        dropout = init_params["dropout"]
        rnn_type = init_params["rnn_type"]

        bidir = False
        if "bidir" in init_params:
            bidir = init_params["bidir"]
        return SequenceModelNoInput(name, seq_size, embedding_size, rnn_size, rnn_layers, rnn_type=rnn_type, dropout=dropout, bidir=bidir)
