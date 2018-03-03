import torch
import torch.nn as nn
from ltprg.model.seq import sort_seq_tensors, unsort_seq_tensors, SequenceModel
from torch.autograd import Variable

class ObservationModel(nn.Module):

    def __init__(self):
        super(ObservationModel, self).__init__()

    def forward(self, observation):
        """ Computes batch of transformed observations """
        pass

    def on_gpu(self):
        return next(self.parameters()).is_cuda

    def save(self, model_path):
        init_params = self._get_init_params()
        model_obj = dict()
        model_obj["init_params"] = init_params
        model_obj["state_dict"] = self.state_dict()
        model_obj["obs_type"] = type(self).__name__
        torch.save(model_obj, model_path)

    @staticmethod
    def load(model_path):
        model_obj = torch.load(model_path)
        init_params = model_obj["init_params"]
        state_dict = model_obj["state_dict"]
        meaning_type = model_obj["obs_type"]

        model = None
        if meaning_type == "ObservationModelIndexedSequential":
            model = ObservationModelIndexedSequential.make(init_params)
        model.load_state_dict(state_dict)

        return model


class ObservationModelIndexed(ObservationModel):

    def __init__(self, indexed_obs_size, num_indices):
        super(ObservationModelIndexed, self).__init__()
        self._indexed_obs_size = indexed_obs_size
        self._num_indices = num_indices

    # observations: batch x input observation
    # indices (one-hots): batch x (num_indices) x (num_indices)
    # return batch x num_indices x indexd_obs_size
    def _forward_for_indices(self, observation, indices):
        """
        Computes batch of transformed observations from input observations
        and indexed index indicators
        """
        pass

    def forward(self, observation):
        indices = torch.eye(self._num_indices).unsqueeze(0).expand(observation[0].size(0), self._num_indices, self._num_indices)

        if self.on_gpu():
            indices = indices.cuda()
        
        indices = Variable(indices)

        transformed = self._forward_for_indices(observation, indices)
        return transformed.view(indices.size(0), self._num_indices*self._indexed_obs_size)


class ObservationModelIndexedSequential(ObservationModelIndexed):
    def __init__(self, indexed_obs_size, num_indices, seq_model):
        super(ObservationModelIndexedSequential, self).__init__(indexed_obs_size, num_indices)

        self._init_params = dict()
        self._init_params["indexed_obs_size"] = indexed_obs_size
        self._init_params["num_indices"] = num_indices
        self._init_params["arch_type"] = type(seq_model).__name__
        self._init_params["seq_model"] = seq_model._get_init_params()

        self._seq_model = seq_model

        self._decoder = nn.Linear(seq_model.get_hidden_size()*seq_model.get_directions(), indexed_obs_size)
        self._decoder_nl = nn.Tanh()

    def _get_init_params(self):
        return self._init_params

    @staticmethod
    def make(init_params):
        indexed_obs_size = init_params["indexed_obs_size"]
        num_indices = init_hidden["num_indices"]
        seq_model = SequenceModel.make(init_params["seq_model"], init_params["arch_type"])
        return ObservationModelIndexedSequential(indexed_obs_size, num_indices, seq_model)

    def get_seq_model(self):
        return self._seq_model

    # observations: batch x input observation
    # indices (one-hots): batch x (num_indices) x (num_indices)
    # return batch x num_indices x indexed_obs_size
    def _forward_for_indices(self, observation, indices):
        num_indices = indices.size(2)
        batch_size = indices.size(0)
        max_len = observation[0].size(1)
        seq = observation[0].transpose(0,1) # After transpose: Length x batch
        seq_length = observation[1] # Batch

        # length, indices*batch
        seq = seq.unsqueeze(1).expand(max_len, num_indices, batch_size).contiguous().view(-1, num_indices*batch_size)
        seq_length = seq_length.unsqueeze(1).expand(batch_size, num_indices).contiguous().view(-1, num_indices*batch_size).squeeze()
        indices = indices.contiguous().view(-1, num_indices)

        sorted_seq, sorted_length, sorted_inputs, sorted_indices = sort_seq_tensors(seq, seq_length, inputs=[indices], on_gpu=self.on_gpu())

        output, hidden = self._seq_model(seq_part=sorted_seq, seq_length=sorted_length, input=sorted_inputs[0])
        if isinstance(hidden, tuple): # Handle LSTM
            hidden = hidden[0]
        decoded = self._decoder(hidden.transpose(0,1).contiguous().view(-1, hidden.size(0)*hidden.size(2)))
        output = self._decoder_nl(decoded)

        unsorted_output = unsort_seq_tensors(sorted_indices, [output])[0]
        return unsorted_output.view(batch_size, num_indices, self._indexed_obs_size)

