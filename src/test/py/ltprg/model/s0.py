import sys
import torch.nn as nn
import time
import numpy as np
import torch
from torch.autograd import Variable
from mung.feature import MultiviewDataSet, Symbol
from mung.data import Partition

RNN_TYPE = "GRU" # LSTM currently broken... need to make cell state
EMBEDDING_SIZE = 100
RNN_SIZE = 100
RNN_LAYERS = 1
TRAINING_ITERATIONS=100
TRAINING_BATCH_SIZE=100
DROP_OUT = 0.5
LOG_INTERVAL = 10

torch.manual_seed(1)
np.random.seed(1)

# Loss function borrowed from
# https://github.com/ruotianluo/neuraltalk2.pytorch/blob/master/misc/utils.py
class VariableLengthNLLLoss(nn.Module):
    def __init__(self):
        super(VariableLengthNLLLoss, self).__init__()

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
        output = torch.sum(output) / torch.sum(mask)

        return output

class S0(nn.Module):
    def __init__(self, rnn_type, world_size, embedding_size, rnn_size,
                 rnn_layers, utterance_size, dropout=0.5):
        super(S0, self).__init__()

        print "Construction S0"
        print "Type: " + rnn_type
        print "World size: " + str(world_size)
        print "Embedding size: " + str(embedding_size)
        print "RNN size: " + str(rnn_size)
        print "RNN layers: " + str(rnn_layers)
        print "Utterance size: " + str(utterance_size)
        print "Drop out: " + str(dropout)

        self._rnn_layers = rnn_layers

        self._encoder = nn.Linear(world_size, rnn_size)
        self._encoder_nl = nn.Tanh()
        self._drop = nn.Dropout(dropout)
        self._emb = nn.Embedding(utterance_size, embedding_size)

        if rnn_type in ['LSTM', 'GRU']:
            self._rnn = getattr(nn, rnn_type)(embedding_size, rnn_size,
                                              rnn_layers, dropout=dropout)
        else:
            try:
                nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[rnn_type]
            except KeyError:
                raise ValueError( """An invalid option for `--model` was supplied,
                                 options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']""")
            self_.rnn = nn.RNN(embedding_size, rnn_size, rnn_layers,
                               nonlinearity=nonlinearity, dropout=dropout)
        self._decoder = nn.Linear(rnn_size, utterance_size)
        self._softmax = nn.LogSoftmax()
        self._criterion = VariableLengthNLLLoss()

    def forward(self, world, utterance_part, utterance_length):
        hidden = self._encoder_nl(self._encoder(world))
        return self._forward_from_hidden(hidden, utterance_part, utterance_length)

    def _forward_from_hidden(self, hidden, utterance_part, utterance_length):
        emb_pad = self._drop(self._emb(utterance_part))

        hidden = hidden.view(self._rnn_layers, hidden.size()[0], hidden.size()[1])
        emb = nn.utils.rnn.pack_padded_sequence(emb_pad, utterance_length, batch_first=False)
        
        output, hidden = self._rnn(emb, hidden)
        
        output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=False)
        rnn_out_size = output.size()

        output = self._softmax(self._decoder(output.view(-1, rnn_out_size[2])))
        output = output.view(rnn_out_size[0], rnn_out_size[1], output.size(1))

        return output, hidden

    def sample(self, world, utterance_part, utterance_length):
        # Get output and hidden state after running utterance prefixes
        # through the model
        output, hidden = self(world, utterance_part, utterance_length)
        print str(output.size()) + " " + str(hidden.size())        

        # Get final output and hidden
        # Sample next tokens (for batch) from output
        # Feed token into emb
        # Feed emb and hidden into rnn
        # Get next output


    def init_weights(self):
        initrange = 0.1
        self._emb.weight.data.uniform_(-initrange, initrange)
        self._encoder.bias.data.fill_(0)
        self._encoder.weight.data.uniform_(-initrange, initrange)
        self._decoder.bias.data.fill_(0)
        self._decoder.weight.data.uniform_(-initrange, initrange)

    def learn(self, data, iterations, batch_size, eval_data=None):
        # 'train' enables dropout
        self.train()
        total_loss = 0
        start_time = time.time()
        for i in range(iterations):
            batch = data.get_random_batch(batch_size)
            world = Variable(batch["world"])
            utterance, length, mask = batch["utterance"]
            length = length - 1

            utt_in = Variable(utterance[:utterance.size(0)-1]).long() # Input remove final token
            target_out = Variable(utterance[1:utterance.size(0)]).long() # Output (remove start token)

            self.zero_grad()
            model_out, hidden = self(world, utt_in, length)
            loss = self._criterion(model_out, target_out[:model_out.size(0)], Variable(mask[:,1:(model_out.size(0)+1)]))
            loss.backward()

            lr = 0.001
            for p in self.parameters():
                p.data.add_(-lr, p.grad.data)
            total_loss += loss.data

            if i % LOG_INTERVAL == 0 and i > 0:
                cur_loss = total_loss[0] / LOG_INTERVAL
                elapsed = time.time() - start_time
                eval_loss = 0.0
                if eval_data is not None:
                    eval_loss = self.evaluate(eval_data)
                print('| {:5d}/{:d} batches |  ms/batch {:5.2f} | '
                        'train loss {:5.7f} | eval loss {:5.7f}'.format(
                    i, iterations, elapsed * 1000 / LOG_INTERVAL, cur_loss, eval_loss))
                total_loss = 0
                start_time = time.time()
                
        # 'eval' disables dropout
        self.eval()

    def evaluate(self, data):
        # Turn on evaluation mode which disables dropout.
        self.eval()

        batch = data.get_batch(0, data.get_size())
        world = Variable(batch["world"])
        utterance, length, mask = batch["utterance"]
        length = length - 1

        utt_in = Variable(utterance[:utterance.size(0)-1]).long() # Input remove final token
        target_out = Variable(utterance[1:utterance.size(0)]).long() # Output (remove start token)

        model_out, hidden = self(world, utt_in, length)
        loss = self._criterion(model_out, target_out[:model_out.size(0)], Variable(mask[:,1:(model_out.size(0)+1)]))
 
        self.train()
        return loss.data[0]
        

data_dir = sys.argv[1]
partition_file = sys.argv[2]
utterance_dir = sys.argv[3]
world_dir = sys.argv[4]

D = MultiviewDataSet.load(data_dir,
                          dfmat_paths={ "world" : world_dir },
                          dfmatseq_paths={ "utterance" : utterance_dir })
partition = Partition.load(partition_file)
D_parts = D.partition(partition, lambda d : d.get("gameid"))
D_train = D_parts["train"]
D_dev = D_parts["dev"]

world_size = D_train["world"].get_feature_set().get_size()
utterance_size = D_train["utterance"].get_matrix(0).get_feature_set().get_token_count()

model = S0(RNN_TYPE, world_size, EMBEDDING_SIZE, RNN_SIZE, RNN_LAYERS,
           utterance_size, dropout=DROP_OUT)
model.learn(D_train, TRAINING_ITERATIONS, TRAINING_BATCH_SIZE, eval_data=D_dev)

# FIXME Get start symbol index
# FIXME Get end symbol index
# FIXME Get sample of worlds
start_idx = Symbol.index(Symbol.SEQ_START)
end_idx = Symbol.index(Symbol.SEQ_END)
D_train["utterance"].get_feature_token(index).get_value()
D_dev_close = D_dev.filter(lambda d : d.get("state.condition") == "close")
D_dev_split = D_dev.filter(lambda d : d.get("state.condition") == "split")
D_dev_far = D_dev.filter(lambda d : d.get("state.condition") == "far")

model.sample()


