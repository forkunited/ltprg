import torch
from ltprg.model.dist import Categorical
from ltprg.model.seq import SamplingMode
from ltprg.model.seq import SequenceModel

class UniformIndexPriorFn(nn.Module):
    def __init__(self, size):
        super(UnfiormVectorPriorFn, self).__init__()

    def forward(self, observation):
        vs = torch.arange(0,size).unsqueeze(0).repeat(observation.size(0),1)
        return Categorical(Variable(vs))

    def get_index(self, vs, observation, support):
        return vs.data, False, None

class SequenceSamplingPriorFn(nn.Module):
    def __init__(self, model, input_size, mode=SamplingMode.FORWARD, samples_per_input=1, uniform=True, seq_length=15):
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
            self._fixed_seq = (seq, length)

    def set_samples_per_input(self, samples_per_input):
        self._samples_per_input

    def forward(self, observation):
        batch_size = observation.size(0)
        inputs_per_observation = observation.size(1)/self._input_size
        all_input = None
        if self._fixed_input is not None:
            all_input = observation.view(batch_size*inputs_per_observation, self._input_size)
            fixed_input_offset = torch.arange(0, batch_size).long()*inputs_per_observation + self._fixed_input
            all_input = all_input.view(fixed_input_offset)
            inputs_per_observation = 1
        elif self._ignored_input is not None:
            all_input = torch.zeros((inputs_per_observation - 1)*batch_size, self._input_size)
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
            all_input = observation.view(batch_size*inputs_per_observation, self._input_size)

        samples = None
        if self._mode == SamplingModel.FORWARD:
            samples = self._model.sample(n_per_input=self._samples_per_input, max_length=self._seq_length, input=all_input)
        elif self._mode == SamplingModel.BEAM:
            samples = self._beam_search(beam_size=self._samples_per_input, max_length=self._seq_length, input=all_input)

        has_fixed = 0
        if self._fixed_seq is not None:
            has_fixed = 1

        seq_supp_batch = torch.zeros(batch_size, self._samples_per_input * inputs_per_observation + has_fixed, self._seq_length).long()
        length_supp_batch = torch.zeros(self._samples_per_input * inputs_per_observation + has_fixed).long()
        for i in range(batch_size):
            if self._fixed_seq is not None:
                seq_supp_batch[i,0] = self._fixed_seq[0]
                length_supp_batch[i,0] = self._fixed_seq[1]

            for j in range(inputs_per_observation):
                seqs, lengths, scores = samples[i*inputs_per_observation+j]
                seq_supp_batch[i, (has_fixed+j*self._samples_per_input):(has_fixed+(j+1)*self._samples_per_input)] = seqs.transpose(0,1)
                length_supp_batch[i, (has_fixed+j*self._samples_per_input):(has_fixed+(j+1)*self._samples_per_input)] = lengths

        return Categorical((Variable(seq_supp_batch), length_supp_batch))

    def get_index(self, seq_with_len, observation, support):
        seq, length = seq_with_len
        seq_support, length_support = support

        index = torch.zeros(seq.size(0)).long()
        has_missing = False
        mask = toch.ones(seq.size(0)).long()

        for i in range(seq.size(0)):
            found = False
            for s in range(seq_support.size(1)):
                if length[i] == length_support[i,s] and torch.equal(seq[i], seq_support[i,s]):
                    index[i] = s
                    found = True
                    break
            if not found:
                has_missing = True
                mask[i] = 0

        return index, has_missing, mask
