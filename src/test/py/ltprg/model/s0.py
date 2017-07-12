import sys
import torch.nn as nn
import time
from torch.autograd import Variable
from mung.feature import MultiviewDataSet
from mung.data import Partition

RNN_TYPE = "LSTM"
EMBEDDING_SIZE = 100
RNN_SIZE = 100
RNN_LAYERS = 1
TRAINING_ITERATIONS=100
TRAINING_BATCH_SIZE=100
DROP_OUT = 0.5
LOG_INTERVAL = 100

np.random.seed(1)

# Loss function borrowed from
# https://github.com/ruotianluo/neuraltalk2.pytorch/blob/master/misc/utils.py
class VariableLengthNLLLoss(nn.Module):
    def __init__(self):
        super(LanguageModelCriterion, self).__init__()

    def _to_contiguous(self, tensor):
        if tensor.is_contiguous():
            return tensor
        else:
            return tensor.contiguous()

    def forward(self, input, target, mask):
        # truncate to the same size
        target = target[:, :input.size(1)]
        mask =  mask[:, :input.size(1)]
        input = self._to_contiguous(input).view(-1, input.size(2))
        target = self._to_contiguous(target).view(-1, 1)
        mask = self._to_contiguous(mask).view(-1, 1)
        output = - input.gather(1, target) * mask
        output = torch.sum(output) / torch.sum(mask)

        return output

class S0(nn.Module):
    def __init__(self, rnn_type, world_size, embedding_size, rnn_size,
                 rnn_layers, utterance_size, dropout=0.5):
        super(S0_decoder, self).__init__()

        self._encoder = nn.Linear(world_size, rnn_size)
        self._encoder_nl = nn.Tanh()
        self._drop = nn.Dropout(dropout)
        self._emb = nn.Embedding(world_size, embedding_size)

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
        self._decoder = nn.Linear(rnn_hidden, utterance_size)
        self._softmax = nn.LogSoftmax()

    def forward(self, world, utterance_parts, utterance_lengths):
        hidden = self._encoder_nl(self._encoder(world))
        emb_pad = self._drop(self._emb(utterance_parts))

        emb = nn.utils.rnn.pack_padded_sequence(emb_pad, utterance_lengths, batch_first=False)
        output, hidden = self._rnn(emb, hidden)
        output = nn.utils.rnn.pas_packed_sequence(output, batch_first=False)

        output = self._decoder(output)
        return self._softmax(output), hidden

    def init_weights(self):
        initrange = 0.1
        self._emb.weight.data.uniform_(-initrange, initrange)
        self._encoder.bias.data.fill_(0)
        self._encoder.weight.data.uniform_(-initrange, initrange)
        self._decoder.bias.data.fill_(0)
        self._decoder.weight.data.uniform_(-initrange, initrange)

    def learn(self, data, iterations, batch_size):
        # 'train' enables dropout
        self.train()
        total_loss = 0
        start_time = time.time()
        criterion = VariableLengthNLLLoss()
        for i in range(iterations):
            batch = data.get_random_batch()
            world = batch["world"]
            utterance, length, mask = batch["utterance"]
            utt_in = utterance[:utterance.shape[0]-1] # Input remove final token
            target_out = utterance[1:utterance.shape[0]] # Output (remove start token)

            model_out, hidden = self(world, utt_in, length)
            loss = criterion(model_out, target_out, mask)
            loss.backward()

            total_loss += loss.data

            if i % LOG_INTERVAL == 0 and i > 0:
                cur_loss = total_loss[0] / LOG_INTERVAL
                elapsed = time.time() - start_time
                print('| {:5d}/{:5d} batches |  ms/batch {:5.2f} | '
                        'loss {:5.2f} | ppl {:8.2f}'.format(
                    i, iterations, elapsed * 1000 / LOG_INTERNAL, cur_loss, math.exp(cur_loss)))
                total_loss = 0
                start_time = time.time()

        # 'eval' disables dropout
        self.eval()


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

world_size = D["world"].get_feature_set().get_size()
utterance_size = D["utterance"].get_matrix(0).get_feature_set().get_token_count()

model = S0(RNN_TYPE, world_size, EMBEDDING_SIZE, RNN_SIZE, RNN_LAYERS,
           utterance_size, dropout=DROP_OUT)
model.learn(D_train, TRAINING_ITERATIONS, TRAINING_BATCH_SIZE)
