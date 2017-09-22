import sys
import torch.nn as nn
import time
import numpy as np
import torch
import copy
import ltprg.data.feature
from torch.autograd import Variable
from torch.optim import Adam
from mung.feature import MultiviewDataSet, Symbol
from mung.data import Partition

from ltprg.model.eval import Loss
from ltprg.model.seq import RNNType, SequenceModelNoInput, SequenceModelInputToHidden, SequenceModelInputEmbedded, VariableLengthNLLLoss, DataParameter
from ltprg.model.learn import OptimizerType, Trainer
from ltprg.util.log import Logger

RNN_TYPE = RNNType.LSTM
EMBEDDING_SIZE = 100
RNN_SIZE = 100
RNN_LAYERS = 1
TRAINING_ITERATIONS=10000 # 10000 #4000 #1000 #00
TRAINING_BATCH_SIZE=128
DROP_OUT = 0.5
OPTIMIZER_TYPE = OptimizerType.ADAM
LEARNING_RATE = 0.004 #0.05 #0.001
LOG_INTERVAL = 500 #500
DEV_SAMPLE_SIZE = None # None is full

torch.manual_seed(1)
torch.cuda.manual_seed(1)
np.random.seed(1)

def output_model_samples(model, D, utterance_length):
    samples = model.sample(n_per_input=20, max_length=utterance_length)
    beams = model.beam_search(beam_size=20, max_length=utterance_length)

    for i in range(len(samples)):
        sampled_utt =  " ".join([D["utterance"].get_feature_token(samples[i][0][j,0]).get_value()
                        for j in range(samples[i][1][0])])
        print "Sampled " + str(i) + ": " + sampled_utt

    beam, utterance_length, scores = beams[0]
    top_utts = [" ".join([D["utterance"].get_feature_token(beam[k][j]).get_value() for k in range(utterance_length[j])]) for j in range(beam.size(1))]

    print " "
    print "Top " + str(beam.size(1)) + " utterances"
    for j in range(len(top_utts)):
        print str(scores[j]) + ": " + top_utts[j]
    print "\n"

gpu = bool(sys.argv[1])
data_dir = sys.argv[2]
partition_file = sys.argv[3]
utterance_dir = sys.argv[4]
output_results_path = sys.argv[5]
output_model_path = sys.argv[6]

D = MultiviewDataSet.load(data_dir,
                          dfmat_paths={ },
                          dfmatseq_paths={ "utterance" : utterance_dir })
partition = Partition.load(partition_file)
D_parts = D.partition(partition, lambda d : d.get("gameid"))
D_train = D_parts["train"]
D_dev = D_parts["dev"].get_random_subset(DEV_SAMPLE_SIZE)
D_dev_close = D_dev.filter(lambda d : d.get("state.condition") == "close")
D_dev_split = D_dev.filter(lambda d : d.get("state.condition") == "split")
D_dev_far = D_dev.filter(lambda d : d.get("state.condition") == "far")

utterance_size = D_train["utterance"].get_matrix(0).get_feature_set().get_token_count()
utterance_length = D_train["utterance"].get_feature_seq_set().get_size()

logger = Logger()
data_parameters = DataParameter.make(seq="utterance", input="world")
loss_criterion = VariableLengthNLLLoss()
model = SequenceModelNoInput("S", utterance_size, \
        EMBEDDING_SIZE, RNN_SIZE, RNN_LAYERS, dropout=DROP_OUT, rnn_type=RNN_TYPE)

loss_criterion_unnorm = VariableLengthNLLLoss(norm_dim=True)

if gpu:
    model = model.cuda()
    loss_criterion = loss_criterion.cuda()
    loss_criterion_unnorm = loss_criterion_unnorm.cuda()

dev_loss = Loss("Dev Loss", D_dev, data_parameters, loss_criterion_unnorm, norm_dim=True)
dev_close_loss = Loss("Dev Close Loss", D_dev_close, data_parameters, loss_criterion_unnorm, norm_dim=True)
dev_split_loss = Loss("Dev Split Loss", D_dev_split, data_parameters, loss_criterion_unnorm, norm_dim=True)
dev_far_loss = Loss("Dev Far Loss", D_dev_far, data_parameters, loss_criterion_unnorm, norm_dim=True)

evaluation = dev_loss
other_evaluations = [dev_close_loss, dev_split_loss, dev_far_loss]

trainer = Trainer(data_parameters, loss_criterion, logger, \
            evaluation, other_evaluations=other_evaluations)
model, best_model = trainer.train(model, D_train, TRAINING_ITERATIONS, \
            batch_size=TRAINING_BATCH_SIZE, optimizer_type=OPTIMIZER_TYPE, lr=LEARNING_RATE, \
            log_interval=LOG_INTERVAL)

output_model_samples(best_model, D_dev_close, utterance_length)

logger.dump(output_results_path)
best_model.save(output_model_path)
