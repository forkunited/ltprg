import torch
import torch.nn as nn
from torch.autograd import Variable
from ltprg.model.dist import Categorical
from ltprg.model.seq import SamplingMode, SequenceModel
from ltprg.model.rsa import DistributionType, DataParameter

class UniformIndexPriorFn(nn.Module):
    def __init__(self, size):
        super(UniformIndexPriorFn, self).__init__()
        self._size = size

    def forward(self, observation):
        vs = torch.arange(0,self._size).unsqueeze(0).repeat(observation.size(0),1)
        return Categorical(Variable(vs))

    # NOTE: This assumes that all values in vs are indices that fall within
    # the range of the support
    def get_index(self, vs, observation, support, preset_batch=False):
        return vs.data.long(), False, None

    def set_data_batch(self, batch, data_parameters):
        pass

class SequenceSamplingPriorFn(nn.Module):
    def __init__(self, model, input_size, mode=SamplingMode.FORWARD, samples_per_input=1, uniform=True, seq_length=15, dist_type=DistributionType.S):
        super(SequenceSamplingPriorFn, self).__init__()
        self._model = model
        self._input_size = input_size
        self._mode = mode
        self._samples_per_input = samples_per_input
        self._uniform = uniform
        self._seq_length = seq_length

        self._fixed_input = None
        self._fixed_seq = None
        self._ignored_input = None
        self._dist_type = dist_type

        if not uniform:
            raise ValueError("Non-uniform sequence prior not implemented")

    def set_ignored_input(self, ignored_input):
        self._fixed_input = None
        self._ignored_input = ignored_input

    def set_fixed_input(self, fixed_input):
        self._ignored_input = None
        self._fixed_input = fixed_input

    def set_fixed_seq(self, seq=None, length=None):
        if seq is None or length is None:
            self._fixed_seq = None
        else:
            self._fixed_seq = (seq.transpose(0,1), length)

    def set_samples_per_input(self, samples_per_input):
        self._samples_per_input = samples_per_input

    def forward(self, observation):
        batch_size = observation.size(0)
        inputs_per_observation = observation.size(1)/self._input_size
        all_inputs = None
        if self._fixed_input is not None:
            all_input = observation.view(batch_size*inputs_per_observation, self._input_size)
            fixed_input_offset = torch.arange(0, batch_size).long()*inputs_per_observation + self._fixed_input.long()
            all_input = all_input[fixed_input_offset]
            inputs_per_observation = 1
        elif self._ignored_input is not None:
            all_inputs = Variable(torch.zeros((inputs_per_observation - 1)*batch_size, self._input_size))
            obs_inputs = observation.view(batch_size, inputs_per_observation, self._input_size)
            all_index = 0
            for i in range(batch_size):
                ignored_i = self._ignored_input[i]
                for j in range(inputs_per_observation):
                    if j != ignored_i:
                        all_inputs[all_index] = obs_inputs[i,j]
                        all_index += 1

            inputs_per_observation = observation.size(1)/self._input_size - 1
        else:
            all_inputs = observation.view(batch_size*inputs_per_observation, self._input_size)

        samples = None
        if self._mode == SamplingMode.FORWARD:
            samples = self._model.sample(n_per_input=self._samples_per_input, max_length=self._seq_length, input=all_inputs)
        elif self._mode == SamplingMode.BEAM:
            samples = self._model.beam_search(beam_size=self._samples_per_input, max_length=self._seq_length, input=all_inputs)

        has_fixed = 0
        if self._fixed_seq is not None:
            has_fixed = 1

        seq_supp_batch = Variable(torch.zeros(batch_size, self._samples_per_input * inputs_per_observation + has_fixed, self._seq_length).long())
        length_supp_batch = torch.zeros(batch_size, self._samples_per_input * inputs_per_observation + has_fixed).long()
        for i in range(batch_size):
            if self._fixed_seq is not None:
                seq_supp_batch[i,0,:] = self._fixed_seq[0][i]
                length_supp_batch[i,0] = self._fixed_seq[1][i]

            for j in range(inputs_per_observation):
                seqs, lengths, scores = samples[i*inputs_per_observation+j]
                seqs = Variable(seqs)
                seq_supp_batch[i, (has_fixed+j*self._samples_per_input):(has_fixed+(j+1)*self._samples_per_input), 0:seqs.size(0)] = seqs.transpose(0,1)
                length_supp_batch[i, (has_fixed+j*self._samples_per_input):(has_fixed+(j+1)*self._samples_per_input)] = lengths

        return Categorical((seq_supp_batch, length_supp_batch))

    def get_index(self, seq_with_len, observation, support, preset_batch=False):
        if preset_batch:
            return torch.zeros(seq_with_len[0].size(0)).long(), False, None
        else:
            return Categorical.get_support_index(seq_with_len, support)

    def set_data_batch(self, batch, data_parameters):
        seqType = DataParameter.UTTERANCE
        inputType = DataParameter.WORLD
        if self._dist_type == DistributionType.L:
            seqType == DataParameter.WORLD
            inputType = DataParameter.UTTERANCE

        # NOTE: If dist type != mode, this means that 
        # for example, the L model is running with an utterance prior
        # that should include the observed utterance
        #
        # Shouldn't ignore the target input in this case, because the 
        # listener doesn't have access to this.
        if self.training or self._dist_type != data_parameters.get_mode():
            seq, length, mask = batch[data_parameters[seqType]]
            self.set_fixed_seq(seq=Variable(seq), length=length)
            if self._dist_type == data_parameters.get_mode():
                self.set_ignored_input(batch[data_parameters[inputType]].squeeze())
        else:
            self.set_fixed_seq(seq=None, length=None)
            self.set_ignored_input(None)
            # This is broken in several ways...
            #self.set_fixed_input(batch[data_parameters[inputType]].squeeze())
