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
        sorted_indices = sorted_indices.cuda(seq.get_device())
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
    SMC = "SMC"
    BEAM_SAMPLE = "BEAM_SAMPLE"

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
            if isinstance(batch[data_parameters[DataParameter.INPUT]], tuple):
                input = (Variable(batch[data_parameters[DataParameter.INPUT]][0]), batch[data_parameters[DataParameter.INPUT]][1])
            else:
                input = Variable(batch[data_parameters[DataParameter.INPUT]])

        seq, length, mask = batch[data_parameters[DataParameter.SEQ]]
        length = length - 1
        seq_in = Variable(seq[:seq.size(0)-1]).long() # Input remove final token

        if self.on_gpu():
            seq_in = seq_in.cuda()
            if input is not None:
                if isinstance(input, tuple):
                    input[0] = input[0].cuda()
                else:
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
            # FIXME Fix to match smc if used later
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

        output, hidden = self(seq_part=Variable(seq_part, requires_grad=False), seq_length=seq_length, input=Variable(input, requires_grad=False))
        for i in range(seq_part.size(0), max_length):
            output_dist = output[output.size(0)-1].exp()
            next_token = torch.multinomial(output_dist).data
            sample = torch.cat((sample, next_token.transpose(1,0)), dim=0)
            output, hidden = self._forward_from_hidden(hidden,
                                                       Variable(next_token.view(1, next_token.size(0)), requires_grad=False),
                                                       unit_length,
                                                       input=Variable(input))

            for j in range(next_token.size(0)):
                seq_length[j] += 1 - ended[j]
                if next_token[j][0] == end_idx and ended[j] != 1:
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
                heuristic_output, _ = heuristic((sample_in, seq_length_in), Variable(input_in, requires_grad=False), None, context=context_in)
                top_indices = heuristic_output.topk(n_per_input)[1]
                sample_in = sample_in.transpose(0,1)[top_indices].transpose(0,1)
                seq_length_in = seq_length_in[top_indices.cpu()]

            # FIXME Add score at some point
            ret_samples.append((sample_in, seq_length_in, 0.0))
        return ret_samples

    def _rearrange_sample(self, sample, seq_length, ended, next_token, hidden, range_index, indices):
        range_size = indices.size(0)
        range_start = range_index*range_size
        range_end = (range_index+1)*range_size
        sample_indices = (range_start + indices).data
        
        sample[:,range_start:range_end] = sample.transpose(0,1)[sample_indices].transpose(0,1)
        seq_length[range_start:range_end] = seq_length[sample_indices.cpu()]
        ended[range_start:range_end] = ended[sample_indices.cpu()]
        next_token[range_start:range_end] = next_token[sample_indices]

        # FIXME Clean up this slop
        if isinstance(hidden, tuple):
            if isinstance(hidden[0], tuple):
                hidden[0][0][:,range_start:range_end] = hidden[0][0][:,sample_indices]
                hidden[0][1][:,range_start:range_end] = hidden[0][1][:,sample_indices]
                hidden[1][range_start:range_end] = hidden[1][sample_indices]
            else:
                hidden[0][:,range_start:range_end] = hidden[0][:,sample_indices]
                hidden[1][:,range_start:range_end] = hidden[1][:,sample_indices]
        else:
            hidden[:,range_start:range_end] = hidden[:,sample_indices]

    # NOTE: Assumes seq_part does not contain end tokens
    def smc(self, n_per_input=1, seq_part=None, max_length=15, input=None, heuristic=None, context=None):
        n = 1
        input_count = 1
        samples_per_input = n_per_input

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
            context = (context[0].unsqueeze(0).expand(samples_per_input, context[0].size(0), context[0].size(1)).transpose(0,1).contiguous().view(n, context[0].size(1)),
                       context[1].unsqueeze(0).expand(samples_per_input, context[1].size(0)).transpose(0,1).contiguous().view(n, 1))

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
            next_token = torch.multinomial(output_dist, num_samples=1).data
            sample = torch.cat((sample, next_token.transpose(1,0)), dim=0)

            for j in range(next_token.size(0)):
                 seq_length[j] += 1 - ended[j]
                 if next_token[j][0] == end_idx and ended[j] != 1:
                     ended[j] = 1
                     ended_count += 1

            if ended_count == n:
                break

            if heuristic is not None:
                heuristic_output, _ = heuristic((sample, seq_length), Variable(input, requires_grad=False), None, context=context)
                for j in range(input_count):
                    # Move ended samples to front
                    indices = Variable(torch.arange(0, samples_per_input), requires_grad=False).long()
                    if self.on_gpu():
                        indices = indices.cuda()
                    input_ended_count = 0
                    for k in range(j*samples_per_input, (j+1)*samples_per_input):
                        if ended[k] == 1:
                            indices[input_ended_count] = k - j*samples_per_input
                            indices[k- j*samples_per_input] = input_ended_count
                            input_ended_count += 1

                    self._rearrange_sample(sample, seq_length, ended, next_token, hidden, j, indices)

                    # Resample based on heuristic amongst non-ended samples if there is more than one non-ended
                    first_non_ended = j*samples_per_input + input_ended_count
                    if first_non_ended >= (j+1)*samples_per_input-1: # At most one non-ended input, so don't bother resampling
                        continue
 
                    indices = Variable(torch.arange(0, samples_per_input), requires_grad=False).long()
                    if self.on_gpu():
                        indices= indices.cuda()
                    w_normalized = nn.functional.softmax(Variable(heuristic_output[first_non_ended:((j+1)*samples_per_input)], requires_grad=False))
                    
                    input_ended_count = first_non_ended-j*samples_per_input
                    indices[input_ended_count:samples_per_input] = input_ended_count + torch.multinomial(w_normalized, num_samples=samples_per_input-input_ended_count,replacement=True)
                    self._rearrange_sample(sample, seq_length, ended, next_token, hidden, j, indices)

            output, hidden = self._forward_from_hidden(hidden,
                                                       Variable(next_token.view(1, next_token.size(0)), requires_grad=False),
                                                       unit_length,
                                                       input=Variable(input, requires_grad=False))

        # Return a list... like beam search...
        ret_samples = []
        for i in range(input_count):
            sample_in = sample[:,(i*samples_per_input):((i+1)*samples_per_input)]
            seq_length_in = seq_length[(i*samples_per_input):((i+1)*samples_per_input)]
            # FIXME Add score at some point
            ret_samples.append((sample_in, seq_length_in, 0.0))
        return ret_samples

    # NOTE: Assumes seq_part does not contain end tokens
    def beam_sample(self, n_per_input=1, seq_part=None, max_length=15, input=None, heuristic=None, context=None, n_before_heuristic=10):
        n = 1
        input_count = 1
        samples_per_input = n_per_input

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
            context = (context[0].unsqueeze(0).expand(samples_per_input*n_before_heuristic, context[0].size(0), context[0].size(1)).transpose(0,1).contiguous().view(-1, context[0].size(1)),
                       context[1].unsqueeze(0).expand(samples_per_input*n_before_heuristic, context[1].size(0)).transpose(0,1).contiguous().view(-1, 1))

        if self.on_gpu():
            seq_part = seq_part.cuda()

        end_idx = Symbol.index(Symbol.SEQ_END)
        ended = torch.zeros(n).long()
        ended_count = 0
        unit_length = torch.ones(n).long()
        seq_length = unit_length*seq_part.size(0)
        sample = copy.deepcopy(seq_part)
        seq_sample_count = sample.size(1)
        output, hidden = self(seq_part=Variable(seq_part), seq_length=seq_length, input=Variable(input))
        for i in range(seq_part.size(0), max_length):
            output_dist = output[output.size(0)-1].exp()
            next_token = torch.multinomial(output_dist, num_samples=n_before_heuristic, replacement=True).data
            
            # Extend sample to contain n_before_heuristic*seq_sample_count samples extended with one token 
            # to be evaluated by heuristic
            next_token = next_token.view(seq_sample_count*n_before_heuristic, 1)
            sample = sample.unsqueeze(2).expand(sample.size(0), sample.size(1), n_before_heuristic).contiguous().view(sample.size(0), -1)
            if isinstance(hidden, tuple):
                if isinstance(hidden[0], tuple):
                    hidden_0 = hidden[0][0].unsqueeze(2).expand(hidden[0][0].size(0),hidden[0][0].size(1), n_before_heuristic, hidden[0][0].size(2)).contiguous().view(hidden[0][0].size(0),hidden[0][0].size(1)*n_before_heuristic,hidden[0][0].size(2))
                    hidden_1 = hidden[0][1].unsqueeze(2).expand(hidden[0][1].size(0),hidden[0][1].size(1), n_before_heuristic, hidden[0][1].size(2)).contiguous().view(hidden[0][1].size(0),hidden[0][1].size(1)*n_before_heuristic,hidden[0][1].size(2))
                    hidden_1_0 = hidden[1].unsqueeze(1).expand(hidden[1].size(0),n_before_heuristic,hidden[1].size(1), hidden[1].size(2)).contiguous().view(hidden[1].size(0)*n_before_heuristic, hidden[1].size(1),hidden[1].size(2))
                    hidden = ((hidden_0, hidden_1), hidden_1_0)
                else:
                    hidden_0 = hidden[0].unsqueeze(2).expand(hidden[0].size(0),hidden[0].size(1), n_before_heuristic, hidden[0].size(2)).contiguous().view(hidden[0].size(0),hidden[0].size(1)*n_before_heuristic,hidden[0].size(2))
                    hidden_1 = hidden[1].unsqueeze(2).expand(hidden[1].size(0),hidden[1].size(1), n_before_heuristic, hidden[0].size(2)).contiguous().view(hidden[1].size(0),hidden[1].size(1)*n_before_heuristic,hidden[1].size(2))
                    hidden = (hidden_0, hidden_1)
            else:
                hidden = hidden.unsqueeze(2).expand(hidden.size(0),hidden.size(1), n_before_heuristic, hidden.size(2)).contiguous().view(hidden.size(0),hidden.size(1)*n_before_heuristic,hidden.size(2)) 
            ended = ended.unsqueeze(1).expand(ended.size(0), n_before_heuristic).contiguous().view(-1)
            seq_length = seq_length.unsqueeze(1).expand(seq_length.size(0), n_before_heuristic).contiguous().view(-1)
            
            sample = torch.cat((sample, next_token.transpose(1,0)), dim=0)
            next_per_input = samples_per_input*n_before_heuristic

            for j in range(next_token.size(0)):
                seq_length[j] += 1 - ended[j]
            if heuristic is not None:
                heuristic_output, _ = heuristic((sample, seq_length), Variable(input, requires_grad=False), None, context=context)
                for j in range(input_count):
                    # Sort the sample based on the heuristic                    
                    _, indices = torch.sort(Variable(heuristic_output[(j*next_per_input):((j+1)*next_per_input)], requires_grad=False),0, True)
                    self._rearrange_sample(sample, seq_length, ended, next_token, hidden, j, indices)

            # Cut the sample back down so there are just n_samples_per_input samples per input
            # from all the possible sampled extensions
            next_token = next_token.contiguous().view(input_count, -1)[:,0:samples_per_input].contiguous().view(-1)
            sample = sample.contiguous().view(-1, input_count, next_per_input)[:,:,0:samples_per_input].contiguous().view(-1,input_count*samples_per_input)
            if isinstance(hidden, tuple):
                if isinstance(hidden[0], tuple):
                    hidden_0 = hidden[0][0].contiguous().view(-1,input_count, next_per_input,hidden[0][0].size(2))[:,:,0:samples_per_input].contiguous().view(-1,input_count*samples_per_input,hidden[0][0].size(2))
                    hidden_1 = hidden[0][1].contiguous().view(-1,input_count, next_per_input,hidden[0][1].size(2))[:,:,0:samples_per_input].contiguous().view(-1,input_count*samples_per_input,hidden[0][1].size(2))
                    hidden_1_0 = hidden[1].contiguous().view(input_count, next_per_input, hidden[1].size(1), hidden[1].size(2))[:,0:samples_per_input].contiguous().view(input_count*samples_per_input, hidden[1].size(1),hidden[1].size(2))
                    hidden = ((hidden_0, hidden_1), hidden_1_0)
                else:
                    hidden_0 = hidden[0].contiguous().view(-1,input_count, next_per_input,hidden[0].size(2))[:,:,0:samples_per_input].contiguous().view(-1,input_count*samples_per_input,hidden[0].size(2))
                    hidden_1 = hidden[1].contiguous().view(-1,input_count, next_per_input,hidden[1].size(2))[:,:,0:samples_per_input].contiguous().view(-1,input_count*samples_per_input,hidden[1].size(2))
                    hidden = (hidden_0, hidden_1)
            else:
                hidden = hidden.contiguous().view(-1,input_count, next_per_input,hidden.size(2))[:,:,0:samples_per_input].contiguous().view(-1,input_count*samples_per_input,hidden.size(2))
            ended = ended.contiguous().view(input_count, -1)[:,0:samples_per_input].contiguous().view(-1)
            seq_length = seq_length.contiguous().view(input_count, -1)[:,0:samples_per_input].contiguous().view(-1)
            
            for j in range(next_token.size(0)):
                if next_token[j] == end_idx and ended[j] != 1:
                    ended[j] = 1
                    ended_count += 1

            if ended_count == n:
                break

            output, hidden = self._forward_from_hidden(hidden,
                                                       Variable(next_token.view(1, next_token.size(0)), requires_grad=False),
                                                       unit_length,
                                                       input=Variable(input, requires_grad=False))

        # Return a list... like beam search...
        ret_samples = []
        for i in range(input_count):
            sample_in = sample[:,(i*samples_per_input):((i+1)*samples_per_input)]
            seq_length_in = seq_length[(i*samples_per_input):((i+1)*samples_per_input)]
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

        model = SequenceModel.make(init_params, arch_type)
        model.load_state_dict(state_dict)

        return model

    @staticmethod
    def make(init_params, arch_type):
        model = None
        if arch_type == "SequenceModelInputEmbedded":
            model = SequenceModelInputEmbedded.make(init_params)
        elif arch_type == "SequenceModelInputToHidden":
            model = SequenceModelInputToHidden.make(init_params)
        elif arch_type == "SequenceModelNoInput":
            model = SequenceModelNoInput.make(init_params)
        elif arch_type == "SequenceModelAttendedInput":
            model = SequenceModelAttendedInput.make(init_params)
        return model

class SequenceModelInputToHidden(SequenceModel):
    def __init__(self, name, seq_size, input_size, embedding_size, rnn_size,
                 rnn_layers, rnn_type=RNNType.GRU, dropout=0.5, bidir=False,
                 input_layers=1, embedding_init=None, freeze_embedding=False,
                 conv_input=False, conv_kernel=1, conv_stride=1):
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
        self._init_params["freeze_embedding"] = freeze_embedding
        self._init_params["conv_input"] = conv_input
        self._init_params["conv_kernel"] = conv_kernel
        self._init_params["conv_stride"] = conv_stride

        self._rnn_layers = rnn_layers
        self._rnn_type = rnn_type
        self._seq_size = seq_size

        self._input_layers = input_layers
        self._freeze_embedding = freeze_embedding

        self._conv_input = conv_input
        encoded_size = rnn_size*rnn_layers*self._directions/(4**(input_layers-1))
        if not self._conv_input:
            self._encoder = nn.Linear(input_size, encoded_size)
            self._encoder_nl = nn.Tanh()
            if self._input_layers == 2:
                self._encoder_0 = nn.Linear(encoded_size, rnn_size*rnn_layers*self._directions)
                self._encoder_0_nl = nn.Tanh()
            elif self._input_layers != 1:
                raise ValueError("Can only have 1 or 2 input layers")
        else:
            if self._input_layers != 1:
                raise ValueError("Input layers must be 1 when convolving input")
            self._encoder = nn.Conv1d(1, encoded_size, conv_kernel, stride=conv_stride)
            self._encoder_nl = nn.LeakyReLU()
            self._encoder_pool = nn.AvgPool1d(input_size/conv_kernel)

        self._drop = nn.Dropout(dropout)
        self._emb = nn.Embedding(seq_size, embedding_size)
        self._rnn = getattr(nn, rnn_type)(embedding_size, rnn_size, rnn_layers, dropout=dropout, bidirectional=bidir)
        self._decoder = nn.Linear(rnn_size*self._directions, seq_size)
        self._softmax = nn.LogSoftmax()

        self._init_weights(embedding_init=embedding_init, freeze_embedding=freeze_embedding)

    def _get_init_params(self):
        return self._init_params

    def _init_hidden(self, batch_size, input=None):
        weight = next(self.parameters()).data

        hidden = None
        if not self._conv_input:
            hidden = self._encoder_nl(self._encoder(input))
            if self._input_layers > 1:
                hidden = self._encoder_0_nl(self._encoder_0(hidden))
        else:
            input = input.unsqueeze(1)
            hidden = self._encoder_pool(self._encoder_nl(self._encoder(input)))
      
        hidden = hidden.view(hidden.size()[0], self._rnn_layers*self._directions, self.get_hidden_size()).transpose(0,1).contiguous()

        if self._rnn_type == RNNType.GRU:
            return hidden
        else:
            return (hidden, \
                    Variable(weight.new(self._rnn_layers*self._directions, batch_size, self._hidden_size).zero_()))

    def _forward_from_hidden(self, hidden, seq_part, seq_length, input=None):
        emb_pad = self._drop(self._emb(seq_part))

        emb = nn.utils.rnn.pack_padded_sequence(emb_pad, seq_length.numpy(), batch_first=False)

        self._rnn.flatten_parameters()
        output, hidden = self._rnn(emb, hidden)

        output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=False)
        rnn_out_size = output.size()

        output = self._softmax(self._decoder(output.view(-1, rnn_out_size[2])))
        output = output.view(rnn_out_size[0], rnn_out_size[1], output.size(1))

        return output, hidden

    def _init_weights(self, embedding_init=None, freeze_embedding=False):
        if embedding_init is None:
            init_range = 0.01
            init.normal(self._emb.weight.data, mean=0.0, std=init_range)
        else:
            self._emb.weight.data = embedding_init
            if freeze_embedding:
                self._emb.weight.requires_grad = False

        #self._emb.weight.data.uniform_(-initrange, initrange)

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

        freeze_embedding = False
        if "freeze_embedding" in init_params:
            freeze_embedding = init_params["freeze_embedding"]

        conv_input = False
        conv_kernel = 1
        conv_stride = 1
        if "conv_input" in init_params:
            conv_input = init_params["conv_input"]
            conv_kernel = init_params["conv_kernel"]
            conv_stride = init_params["conv_stride"]

        return SequenceModelInputToHidden(name, seq_size, input_size, embedding_size, \
            rnn_size, rnn_layers, rnn_type=rnn_type, dropout=dropout, bidir=bidir, \
            input_layers=input_layers, freeze_embedding=freeze_embedding,\
            conv_input=conv_input, conv_kernel=conv_kernel, conv_stride=conv_stride)


class SequenceModelInputEmbedded(SequenceModel):
    def __init__(self, name, seq_size, input_size, embedding_size, rnn_size,
                 rnn_layers, rnn_type=RNNType.GRU, dropout=0.5, bidir=False, embedding_init=None,
                 freeze_embedding=False, non_emb=False):
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
        self._init_params["freeze_embedding"] = freeze_embedding
        self._init_params["non_emb"] = non_emb

        self._freeze_embedding = freeze_embedding

        self._rnn_layers = rnn_layers
        self._rnn_type = rnn_type
        self._drop = nn.Dropout(dropout)

        self._non_emb = non_emb

        if non_emb:
            self._emb = nn.Linear(seq_size, embedding_size)
            self._tanh = nn.Tanh()
        else:
            self._emb = nn.Embedding(seq_size, embedding_size)

        self._rnn = getattr(nn, rnn_type)(embedding_size + input_size, rnn_size, rnn_layers, dropout=dropout, bidirectional=bidir)
        self._decoder = nn.Linear(rnn_size*self._directions, seq_size)
        self._softmax = nn.LogSoftmax()

        self._init_weights(embedding_init=embedding_init, freeze_embedding=freeze_embedding)

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
        emb_pad = None
        if self._non_emb:
            emb_pad = self._drop(self._tanh(self._emb(seq_part)))
        else:
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

    def _init_weights(self, embedding_init=None, freeze_embedding=False):
        if embedding_init is None:
            init_range = 0.01
            init.normal(self._emb.weight.data, mean=0.0, std=init_range)
        else:
            self._emb.weight.data = embedding_init
            if freeze_embedding:
                self._emb.weight.requires_grad = False

        #self._emb.weight.data.uniform_(-initrange, initrange)
        #init.normal(self._emb.weight.data, mean=0.0, std=init_range)

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

        non_emb = False
        if "non_emb" in init_params:
            non_emb = init_params["non_emb"]

        freeze_embedding = False
        if "freeze_embedding" in init_params:
            freeze_embedding = init_params["freeze_embedding"]

        return SequenceModelInputEmbedded(name, seq_size, input_size, embedding_size, rnn_size, rnn_layers, rnn_type=rnn_type, dropout=dropout, bidir=bidir, freeze_embedding=freeze_embedding, non_emb=non_emb)


class SequenceModelNoInput(SequenceModel):
    def __init__(self, name, seq_size, embedding_size, rnn_size,
                 rnn_layers, rnn_type=RNNType.GRU, dropout=0.5, bidir=False,
                 embedding_init=None, freeze_embedding=False, non_emb=False):
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
        self._init_params["freeze_embedding"] = freeze_embedding
        self._init_params["non_emb"] = non_emb

        self._freeze_embedding = freeze_embedding

        if non_emb:
            # FIXME This was for earlier version... can remove
            self._emb = nn.Linear(seq_size, embedding_size)
            self._tanh = nn.Tanh()
        else:
            self._emb = nn.Embedding(seq_size, embedding_size)

        self._non_emb = non_emb
        self._rnn_layers = rnn_layers
        self._rnn_type = rnn_type
        self._drop = nn.Dropout(dropout)
        if non_emb:
            self._rnn = getattr(nn, rnn_type)(seq_size, rnn_size, rnn_layers, dropout=dropout, bidirectional=bidir) # FIXME embedding to seq_size
        else:
            self._rnn = getattr(nn, rnn_type)(embedding_size, rnn_size, rnn_layers, dropout=dropout, bidirectional=bidir)
        self._decoder = nn.Linear(rnn_size*self._directions, seq_size)
        self._softmax = nn.LogSoftmax()

        self._init_weights(embedding_init=embedding_init, freeze_embedding=freeze_embedding)

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
        if self._non_emb:
            emb_pad = seq_part
        else:
            emb_pad = self._drop(self._emb(seq_part)) 

        emb = nn.utils.rnn.pack_padded_sequence(emb_pad, seq_length.numpy(), batch_first=False)

        output, hidden = self._rnn(emb, hidden)

        output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=False)
        rnn_out_size = output.size()
        output = self._softmax(self._decoder(output.view(-1, rnn_out_size[2])))
        output = output.view(rnn_out_size[0], rnn_out_size[1], output.size(1))

        return output, hidden

    def _init_weights(self, embedding_init=None, freeze_embedding=False):
        if embedding_init is None:
            init_range = 0.01
            init.normal(self._emb.weight.data, mean=0.0, std=init_range)
        else:
            self._emb.weight.data = embedding_init
            if freeze_embedding:
                self._emb.weight.requires_grad = False

    @staticmethod
    def make(init_params):
        name = init_params["name"]
        seq_size = init_params["seq_size"]
        embedding_size = init_params["embedding_size"]
        rnn_size = init_params["rnn_size"]
        rnn_layers = init_params["rnn_layers"]
        dropout = init_params["dropout"]
        rnn_type = init_params["rnn_type"]
        non_emb = False
        if "non_emb" in init_params:
            non_emb = init_params["non_emb"]

        bidir = False
        if "bidir" in init_params:
            bidir = init_params["bidir"]
        return SequenceModelNoInput(name, seq_size, embedding_size, rnn_size, rnn_layers, rnn_type=rnn_type, dropout=dropout, bidir=bidir, non_emb=non_emb)


class SequenceModelPair(SequenceModel):
    def __init__(self, name, in_model, out_model, hidden_size):
        super(SequenceModelPair, self).__init__(name, hidden_size, False)

        self._init_params = dict()
        self._init_params["name"] = name
        self._init_params["hidden_size"] = hidden_size
        self._init_params["in_model"] = in_model._get_init_params()
        self._init_params["out_model"] = out_model._get_init_params()
        self._init_params["in_model_arch"] = type(in_model).__name__
        self._init_params["out_model_arch"] = type(out_model).__name__

        self._in_model = in_model
        self._out_model = out_model
        self._hidden_size = hidden_size

        self._hidden = nn.Linear(in_model.get_hidden_size()*in_model.get_directions(), hidden_size)
        self._hidden_nl = nn.Tanh()

    def _get_init_params(self):
        return self._init_params

    def forward(self, seq_part=None, seq_length=None, input=None):
        output, hidden = self._in_model(seq_part=input[0], seq_length=input[1])
        if isinstance(hidden, tuple): # Handle LSTM
            hidden = hidden[0]
        hidden = self._hidden(hidden.transpose(0,1).contiguous().view(-1, hidden.size(0)*hidden.size(2)))
        hidden = self._hidden_nl(hidden)

        return self._out_model(seq_part=seq_part, seq_length=seq_length, input=hidden)

    @staticmethod
    def make(init_params):
        name = init_params["name"]
        hidden_size = init_params["hidden_size"]
        in_model_params = init_params["in_model"]
        out_model_params = init_params["out_model"]
        in_model_arch = init_params["in_model_arch"]
        out_model_arch = init_params["out_model_arch"]

        in_model = SequenceModel.make(in_model_params, in_model_arch)
        out_model = SequenceModel.make(out_model_params, out_model_arch)
        return SequenceModelPair(name, in_model, out_model, hidden_size)


# FIXME: This is a temporary helper method to deal
# with the current messy representation of utterances.
# The utterances should be refactored to render this method unnecessary
def strs_for_scored_samples(samples, data):
    strs = []
    for i in range(len(samples)):
        seqs, seq_lengths, _ = samples[i]
        strs_for_i = [" ".join([data.get_feature_token(seqs[k][j]).get_value() \
            for k in range(seq_lengths[j])]) for j in range(seqs.size(1))]
        strs.append(strs_for_i)
    return strs

class SequenceModelAttendedInput(SequenceModel):
    def __init__(self, name, seq_size, input_size, embedding_size, rnn_size,
                 rnn_layers, rnn_type=RNNType.GRU, dropout=0.5, bidir=False,
                 embedding_init=None, freeze_embedding=False, conv_kernel=1, conv_stride=1, attn_type="EMBEDDING"):
        super(SequenceModelAttendedInput, self).__init__(name, rnn_size, bidir)

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
        self._init_params["freeze_embedding"] = freeze_embedding
        self._init_params["conv_kernel"] = conv_kernel
        self._init_params["conv_stride"] = conv_stride
        self._init_params["attn_type"] = attn_type

        #if rnn_size != embedding_size:
        #    raise ValueError("Currently rnn and embedding sizes must be equal for embedding attn")

        self._rnn_layers = rnn_layers
        self._rnn_type = rnn_type
        self._seq_size = seq_size

        self._freeze_embedding = freeze_embedding

        encoded_size = rnn_size*rnn_layers*self._directions
        self._encoder = nn.Conv1d(1, encoded_size, conv_kernel, stride=conv_stride)
        self._encoder_nl = nn.LeakyReLU()
        self._encoder_pool = nn.MaxPool1d(input_size/conv_kernel)

        self._drop = nn.Dropout(dropout)
        if attn_type == "EMBEDDING":
            self._emb = nn.Embedding(seq_size, embedding_size*self._directions)
            self._rnn = getattr(nn, rnn_type)(embedding_size*self._directions+rnn_size*self._directions, rnn_size, rnn_layers, dropout=dropout, bidirectional=bidir)
            self._decoder = nn.Linear(rnn_size*self._directions, seq_size)
            self._attn_w = nn.Parameter(torch.FloatTensor(rnn_size, embedding_size))
        else:
            self._emb = nn.Embedding(seq_size, embedding_size)
            self._rnn = getattr(nn, rnn_type)(embedding_size, rnn_size, rnn_layers, dropout=dropout, bidirectional=bidir)
            self._decoder = nn.Linear(rnn_size*self._directions*2, seq_size)
            self._attn_w = nn.Parameter(torch.FloatTensor(rnn_size, rnn_size))
        self._softmax1 = nn.LogSoftmax(dim=1)
        self._softmax2 = nn.LogSoftmax(dim=2)

        self._attn_type = attn_type

        self._init_weights(embedding_init=embedding_init, freeze_embedding=freeze_embedding)

    def _get_init_params(self):
        return self._init_params

    def _init_hidden(self, batch_size, input=None):
        weight = next(self.parameters()).data

        input = input.unsqueeze(1)
        encoded = self._encoder_nl(self._encoder(input))
        hidden = self._encoder_pool(encoded)
      
        hidden = hidden.view(hidden.size()[0], self._rnn_layers*self._directions, self.get_hidden_size()).transpose(0,1).contiguous()

        if self._rnn_type == RNNType.GRU:
            return (hidden, encoded)
        else:
            return ((hidden, \
                    Variable(weight.new(self._rnn_layers*self._directions, batch_size, self._hidden_size).zero_())), \
                    encoded)

    def _forward_from_hidden(self, hidden, seq_part, seq_length, input=None):
        hidden, encoded_input = hidden
        emb_pad = self._drop(self._emb(seq_part))

        if self._attn_type == "EMBEDDING":
            emb_pad = emb_pad.transpose(0,1)
            alpha = torch.exp(self._softmax(torch.bmm(\
                                            torch.bmm(emb_pad,self._attn_w.unsqueeze(0).expand(emb_pad.size(0), self._attn_w.size(0), self._attn_w.size(1)))\
                                            , encoded_input)))
 
            #alpha = torch.exp(self._softmax2(torch.bmm(emb_pad, encoded_input)))
            attn = torch.bmm(alpha, encoded_input.transpose(1, 2))
            emb_pad = torch.cat((emb_pad, attn), dim=2)
            emb_pad = emb_pad.transpose(0,1)

        emb = nn.utils.rnn.pack_padded_sequence(emb_pad, seq_length.numpy(), batch_first=False)

        self._rnn.flatten_parameters()
        output, hidden = self._rnn(emb, hidden)

        output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=False)
        rnn_out_size = output.size()

        if self._attn_type != "EMBEDDING":
            output = output.transpose(0, 1)
            alpha = torch.exp(self._softmax2(torch.bmm(\
                                            torch.bmm(output,self._attn_w.unsqueeze(0).expand(output.size(0), self._attn_w.size(0), self._attn_w.size(1)))\
                                            , encoded_input)))

            #alpha = torch.exp(self._softmax2(torch.bmm(output, encoded_input)))
            attn = torch.bmm(alpha, encoded_input.transpose(1,2))
            output = torch.cat((output, attn), dim=2)
            output = output.transpose(0,1).contiguous()

        output = self._softmax1(self._decoder(output.view(-1, output.size(2))))
        output = output.view(rnn_out_size[0], rnn_out_size[1], output.size(1))

        return output, (hidden, encoded_input)

    def _init_weights(self, embedding_init=None, freeze_embedding=False):
        if embedding_init is None:
            init_range = 0.01
            init.normal(self._emb.weight.data, mean=0.0, std=init_range)
        else:
            self._emb.weight.data = embedding_init
            if freeze_embedding:
                self._emb.weight.requires_grad = False

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
        freeze_embedding = init_params["freeze_embedding"]
        conv_kernel = init_params["conv_kernel"]
        conv_stride = init_params["conv_stride"]

        attn_type = "EMBEDDING"
        if "attn_type" in init_params:
            attn_type = init_params["attn_type"]

        return SequenceModelAttendedInput(name, seq_size, input_size, embedding_size, \
            rnn_size, rnn_layers, rnn_type=rnn_type, dropout=dropout, bidir=bidir, \
            freeze_embedding=freeze_embedding,\
            conv_kernel=conv_kernel, conv_stride=conv_stride, attn_type=attn_type)
