import torch
import numpy as np
import torch.nn as nn
import torch.nn.init as init
from ltprg.model.seq import VariableLengthNLLLoss, RNNType, DataParameter
from torch.autograd import Variable

class EditType:
    DELETE = "DELETE"
    REPLACE = "REPLACE"

class EditModel(nn.Module):

    def __init__(self, name):
        super(EditModel, self).__init__()
        self._name = name

    def get_name(self):
        return self._name

    def sample(self, seq, seq_length, n_per_input=10, input=None, heuristic=None, context=None, n_before_heuristic=100):
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

        input_count = seq.size(1)
        n = seq.size(1) * samples_per_input

        seq = seq.unsqueeze(2).repeat(1,1,samples_per_input).view(seq.size(0), n).long()
        seq_length = seq_length.unsqueeze(1).repeat(1,samples_per_input).view(n)
        
        if isinstance(seq, Variable):
            seq = seq.data
        
        if heuristic is not None:
            context = (context[0].unsqueeze(0).expand(samples_per_input, context[0].size(0), context[0].size(1)).transpose(0,1).contiguous().view(n, context[0].size(1)),
                       context[1].unsqueeze(0).expand(samples_per_input, context[1].size(0)).transpose(0,1).contiguous().view(n, 1))

        if self.on_gpu():
            seq = seq.cuda()

        dist = self.forward(Variable(seq), seq_length, edit_type=EditType.REPLACE, input=input)
        
        # Add zeros so that tokens in seq align with dists
        zs = torch.zeros(1,dist.size(1), dist.size(2))
        if self.on_gpu():
            zs = zs.cuda()
        dist = torch.cat((zs, dist.data, zs), dim=0) 

        L, B, V = dist.size()
        dist = dist.view(L*B, V).exp()

        # Zero input tokens out of distributions (so they aren't sampled)
        zero_indices = seq[:L].view(L*B)
        for i in range(L*B):
            dist[i, zero_indices[i]] = 0.0

        # Draw sample tokens
        sample_tokens = torch.multinomial(dist, num_samples=1)
        sample_tokens = sample_tokens.view(L, B)

        # Build resulting sequences (delete UNCs and replace others)
        for b in range(0, B):
            edit_index = np.random.randint(1, high=seq_length[b]-1) # Sample index between 1 and seq_length[b]
            edit_token = sample_tokens[edit_index,b]
            if edit_token == 0: # Deletion
                for i in range(edit_index, seq_length[b]):
                    if i == seq.size(0) - 1:
                        seq[i,b] = 0
                    else:
                        seq[i,b] = seq[i+1, b]
                seq_length[b] -= 1
            else: # Replacement
                seq[edit_index, b] = edit_token 

        # Return a list... like beam search...
        ret_samples = []
        for i in range(input_count):
            input_in = None
            if input is not None:
                input_in = input[(i*samples_per_input):((i+1)*samples_per_input)]
            seq_in = seq[:,(i*samples_per_input):((i+1)*samples_per_input)]
            seq_length_in = seq_length[(i*samples_per_input):((i+1)*samples_per_input)]

            if heuristic is not None:
                context_in = (context[0][(i*samples_per_input):((i+1)*samples_per_input)], context[1][(i*samples_per_input):((i+1)*samples_per_input)])
                heuristic_output, _ = heuristic((seq_in, seq_length_in), Variable(input_in, requires_grad=False), None, context=context_in)
                top_indices = heuristic_output.topk(n_per_input)[1]
                seq_in = seq_in.transpose(0,1)[top_indices].transpose(0,1)
                seq_length_in = seq_length_in[top_indices.cpu()]

            ret_samples.append((seq_in, seq_length_in, 0.0))            
        return ret_samples

    def forward(self, seq, seq_length, edit_type=EditType.REPLACE, input=None):
        """ Computes distributions over tokens for some edit
        type at every index within a sequence
        """
        pass

    def forward_batch(self, batch, data_parameters):
        input = None
        if DataParameter.INPUT in data_parameters and data_parameters[DataParameter.INPUT] in batch:
            if isinstance(batch[data_parameters[DataParameter.INPUT]], tuple):
                input = (Variable(batch[data_parameters[DataParameter.INPUT]][0]), batch[data_parameters[DataParameter.INPUT]][1])
            else:
                input = Variable(batch[data_parameters[DataParameter.INPUT]])

        seq, length, mask = batch[data_parameters[DataParameter.SEQ]]
        seq_in = Variable(seq[:seq.size(0)]).long()

        if self.on_gpu():
            seq_in = seq_in.cuda()
            if input is not None:
                if isinstance(input, tuple):
                    input[0] = input[0].cuda()
                else:
                    input = input.cuda()

        replace_out = self(seq_in, length, edit_type=EditType.REPLACE, input=input)
        delete_out = self(seq_in, length, edit_type=EditType.DELETE, input=input)
        return replace_out, delete_out

    def loss(self, batch, data_parameters, loss_criterion):
        replace_out, delete_out = self.forward_batch(batch, data_parameters)
        seq, length, mask = batch[data_parameters[DataParameter.SEQ]]
        
        replace_target_out = Variable(seq[1:(seq.size(0)-1)]).long() # Output (remove start and end token)
        replace_mask = mask[:, 1:(replace_out.size(0)+1)]
        replace_length = length - 2
        
        delete_target_out = Variable(torch.zeros(seq.size(0)-1, seq.size(1))).long()
        delete_mask = mask[:, 1:(delete_out.size(0)+1)]
        delete_length = length - 1

        if self.on_gpu():
            delete_target_out = delete_target_out.cuda()
            delete_mask = delete_mask.cuda()
            replace_target_out = replace_target_out.cuda()
            replace_mask = replace_mask.cuda()

        replace_loss = loss_criterion(replace_out, replace_target_out[:replace_out.size(0)], Variable(replace_mask))
        delete_loss = loss_criterion(delete_out, delete_target_out[:delete_out.size(0)], Variable(delete_mask))
        return replace_loss + delete_loss

    def on_gpu(self):
        return next(self.parameters()).is_cuda

    def save(self, model_path):
        init_params = self._get_init_params()
        model_obj = dict()
        model_obj["init_params"] = init_params
        model_obj["state_dict"] = self.state_dict()
        model_obj["edit_model_type"] = type(self).__name__
        torch.save(model_obj, model_path)

    @staticmethod
    def load(model_path):
        model_obj = torch.load(model_path)
        init_params = model_obj["init_params"]
        state_dict = model_obj["state_dict"]
        edit_model_type = model_obj["edit_model_type"]

        model = None
        if edit_model_type == "EditModelSequentialNoInput":
            model = EditModelSequentialNoInput.make(init_params)
        model.load_state_dict(state_dict)

        return model

class EditModelSequentialNoInput(EditModel):
    def __init__(self, name, seq_size, embedding_size, rnn_size, rnn_type=RNNType.GRU, dropout=0.5):
        super(EditModelSequentialNoInput, self).__init__(name)

        self._init_params = dict()
        self._init_params["name"] = name
        self._init_params["seq_size"] = seq_size
        self._init_params["embedding_size"] = embedding_size
        self._init_params["rnn_size"] = rnn_size
        self._init_params["rnn_type"] = rnn_type
        self._init_params["dropout"] = dropout

        self._seq_size = seq_size
        self._embedding_size = embedding_size
        self._rnn_size = rnn_size
        self._rnn_type = rnn_type
        self._dropout = dropout

        self._emb = nn.Embedding(seq_size, embedding_size)
        self._rnn = getattr(nn, rnn_type)(embedding_size, rnn_size, 1, dropout=dropout, bidirectional=True)
        self._decoder = nn.Linear(rnn_size*2, seq_size)
        self._softmax = nn.LogSoftmax()
        self._drop = nn.Dropout(dropout)

        self._init_weights()

    def _get_init_params(self):
        return self._init_params

    def forward(self, seq, seq_length, edit_type=EditType.REPLACE, input=None):
        hidden = self._init_hidden(seq_length.size(0))
        return self._forward_from_hidden(hidden, seq, seq_length, edit_type)

    def _init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        if self._rnn_type == RNNType.GRU:
            return Variable(weight.new(2, batch_size, self._rnn_size).zero_())
        else:
            return (Variable(weight.new(2, batch_size, self._rnn_size).zero_()), \
                    Variable(weight.new(2, batch_size, self._rnn_size).zero_()))

    def _forward_from_hidden(self, hidden, seq, seq_length, edit_type):
        emb_pad = self._drop(self._emb(seq))
        emb = nn.utils.rnn.pack_padded_sequence(emb_pad, seq_length.numpy(), batch_first=False)
        output, hidden = self._rnn(emb, hidden)
        # output : L x B x H*2
        output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=False)
        L, B, H2 = output.size()
        H = H2/2

        if edit_type == EditType.DELETE:
            out_forward = output[0:(output.size(0)-1),:,0:H]
            out_backward = output[1:output.size(0),:,H:H2]
            out = torch.cat((out_forward, out_backward), dim=2)
            output_dist = self._softmax(self._decoder(out.view(-1, H2)))
            return output_dist.view(L-1, B, output_dist.size(1))
        elif edit_type == EditType.REPLACE: 
            out_forward = output[0:(output.size(0)-2),:,0:H]
            out_backward = output[2:output.size(0),:,H:H2]
            out = torch.cat((out_forward, out_backward), dim=2)
            output_dist = self._softmax(self._decoder(out.view(-1, H2)))
            return  output_dist.view(L-2, B, output_dist.size(1))

    def _init_weights(self):
        init_range = 0.01
        init.normal(self._emb.weight.data, mean=0.0, std=init_range)

    @staticmethod
    def make(init_params):
        name = init_params["name"]
        seq_size = init_params["seq_size"]
        embedding_size = init_params["embedding_size"]
        rnn_size = init_params["rnn_size"]
        rnn_type = init_params["rnn_type"]
        dropout = init_params["dropout"]
        
        return EditModelSequentialNoInput(name, seq_size, embedding_size, rnn_size, rnn_type=rnn_type, dropout=dropout)
