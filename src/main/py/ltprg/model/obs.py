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
            device = 0
            if isinstance(observation, tuple):
                device = observation[0].get_device()
            else:
                device = observation.get_device()
            indices = indices.cuda(device)

        indices = Variable(indices, requires_grad=False)

        transformed = self._forward_for_indices(observation, indices)
        transformed = torch.cat((transformed, indices), 2)
        return transformed.view(indices.size(0), self._num_indices*(self._num_indices+self._indexed_obs_size))


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
        if len(seq.size()) == 2:
            seq = seq.unsqueeze(1).expand(max_len, num_indices, batch_size).contiguous().view(-1, num_indices*batch_size)
        else:
            seq = seq.unsqueeze(1).expand(max_len, num_indices, batch_size, seq.size(2)).contiguous().view(-1, num_indices*batch_size, seq.size(2)).float()
        seq_length = seq_length.unsqueeze(1).expand(batch_size, num_indices).contiguous().view(-1, num_indices*batch_size).squeeze()
        indices = indices.contiguous().view(-1, num_indices)
        sorted_seq, sorted_length, sorted_inputs, sorted_indices = sort_seq_tensors(seq, seq_length, inputs=[indices], on_gpu=self.on_gpu())

        output, hidden = self._seq_model(seq_part=sorted_seq, seq_length=sorted_length, input=sorted_inputs[0])
        if isinstance(hidden, tuple): # Handle LSTM
            hidden = hidden[0]
        decoded = self._decoder(hidden.transpose(0,1).contiguous().view(-1, hidden.size(0)*hidden.size(2)))
        output = self._decoder_nl(decoded)

        unsorted_output = unsort_seq_tensors(sorted_indices, [output])[0]
        unsorted_output = unsorted_output.view(batch_size, num_indices, self._indexed_obs_size)

        return unsorted_output


class ObservationModelReordered(ObservationModel):

    def __init__(self, indexed_obs_size, num_indices):
        super(ObservationModelReordered, self).__init__()
        self._indexed_obs_size = indexed_obs_size
        self._num_indices = num_indices

    # observations: batch x input observation
    # indices (one-hots): batch x num_indices
    # return batch x num_indices x indexed_obs_size
    def _forward_for_indices(self, observation, indices):
        """
        Computes batch of transformed observations from input observations
        and indexed index indicators
        """
        pass

    def forward(self, observation):
        indices = torch.arange(0, self._num_indices).unsqueeze(0).expand(observation[0].size(0), self._num_indices)

        if self.on_gpu():
            device = 0
            if isinstance(observation, tuple):
                device = observation[0].get_device()
            else:
                device = observation.get_device()
            indices = indices.cuda(device)

        indices = Variable(indices, requires_grad=False)

        transformed = self._forward_for_indices(observation, indices)
        return transformed.view(indices.size(0), self._num_indices*self._indexed_obs_size)

class ObservationModelReorderedSequential(ObservationModelReordered):
    def __init__(self, indexed_obs_size, num_indices, seq_model):
        super(ObservationModelReorderedSequential, self).__init__(indexed_obs_size, num_indices)

        self._init_params = dict()
        self._init_params["indexed_obs_size"] = indexed_obs_size
        self._init_params["num_indices"] = num_indices
        self._init_params["arch_type"] = type(seq_model).__name__
        self._init_params["seq_model"] = seq_model._get_init_params()

        self._seq_model = seq_model

        if indexed_obs_size != seq_model.get_hidden_size()*seq_model.get_directions():
            raise ValueError("indxed_obs_size must be the same as the seq_model hidden size")

    def _get_init_params(self):
        return self._init_params

    @staticmethod
    def make(init_params):
        indexed_obs_size = init_params["indexed_obs_size"]
        num_indices = init_hidden["num_indices"]
        seq_model = SequenceModel.make(init_params["seq_model"], init_params["arch_type"])
        return ObservationModelReorderedSequential(indexed_obs_size, num_indices, seq_model)

    def get_seq_model(self):
        return self._seq_model

    # observations: batch x input observation
    # indices (one-hots): batch x num_indices
    # return batch x num_indices x indexed_obs_size
    def _forward_for_indices(self, observation, indices):
        num_indices = indices.size(2)
        batch_size = indices.size(0)
        max_len = observation[0].size(1)
        obj_size = observation[0].size(2)

        reordered_obs = self._make_obs_reorderings(indices, observation)
        seq = reordered_obs[0].view(batch_size*num_indices, max_len, obj_size).transpose(0,1)
        seq_length = reordered_obs[1].view(batch_size*num_indices)

        sorted_seq, sorted_length, sorted_inputs, sorted_indices = sort_seq_tensors(seq, seq_length, inputs=None, on_gpu=self.on_gpu())

        output, hidden = self._seq_model(seq_part=sorted_seq, seq_length=sorted_length, input=sorted_inputs[0])
        if isinstance(hidden, tuple): # Handle LSTM
            hidden = hidden[0]
        output = hidden.transpose(0,1).contiguous().view(-1, hidden.size(0)*hidden.size(2))

        unsorted_output = unsort_seq_tensors(sorted_indices, [output])[0]
        unsorted_output = unsorted_output.view(batch_size, num_indices, self._indexed_obs_size)

        return unsorted_output

    # observation: batch x length x obj size
    # indices: batch x num_indices
    # return: batch x num_indices x length x obj size (reorderings)
    def _make_obs_reorderings(self, indices, observation):
        batch_size = observation[0].size(0)
        num_indices = indices.size(1)
        max_len = observation[0].size(1)
        obj_size = observation[0].size(2)
        len = observation[1]

        indexed_objs = self._get_indexed_obs_obj(indices, observation), offset_indices
        last_objs = self._get_last_obs_obj(num_indices, observation), offset_last_indices

        seq = observation[0].unsqueeze(1).expand(batch_size, num_indices, max_len, obj_size).view(batch_size*num_indices, max_len, obj_size).transpose(0,1)
        seq_clone = seq.clone()

        seq_clone[offset_indices] = seq[last_indices]
        seq_clone[last_indices] = seq[offset_indices]

        len = len.unsqueeze(1).expand(batch_size, num_indices)

        return (seq_clone, len)


    # observation: batch x length x obj size
    # indices: batch x num_indices
    # return: batch x num_indices x obj size, num_indices*batch_size (indices)
    def _get_indexed_obs_obj(self, indices, observation):
        batch_size = observation[0].size(0)
        seq_length = observation[0].size(1)
        obj_size = observation[0].size(2)
        num_indices = indices.size(1)

        offset = torch.arange(0,batch_size).unsqueeze(0).expand(num_indices, batch_size).transpose(0,1).contiguous().view(num_indices*batch_size)
        offset_indices = offset+indices.view(batch_size*num_indices)

        indexed_obs = observation[0].view(batch_size*seq_length, obj_size)[offset_indices]
        return indexed_obs.view(batch_size, num_indices, obj_size), offset_indices

    # observation: batch x length x obj size
    # return: batch x num_indices x obj size, num_indices*batch_size (indices)
    def _get_last_obs_obj(self, num_indices, observation):
        indices = torch.ones(observation.size(0), num_indices).long()*observation[1]
        return self._get_indexed_obs_obj(indices, observation)
