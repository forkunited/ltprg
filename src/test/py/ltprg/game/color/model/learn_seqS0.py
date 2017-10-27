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
from mung.torch_ext.eval import Loss
from mung.torch_ext.learn import Traniner, OptimizerType
from mung.util.log import Logger

from ltprg.model.seq import RNNType, SequenceModelInputToHidden, SequenceModelInputEmbedded, VariableLengthNLLLoss, DataParameter

RNN_TYPE = RNNType.LSTM
INPUT_TYPE = "NOT_EMBEDDED" #"EMBEDDED"
EMBEDDING_SIZE = 100
RNN_SIZE = 100
RNN_LAYERS = 1
TRAINING_ITERATIONS=10000 #10000 #1000 #00
TRAINING_BATCH_SIZE=128
DROP_OUT = 0.5
OPTIMIZER_TYPE = OptimizerType.ADAM
LEARNING_RATE = 0.005 #0.05 #0.001
LOG_INTERVAL = 500
DEV_SAMPLE_SIZE = None # None is full

def output_model_samples(model, D, max_length, batch_size=20):
    data = D.get_data()
    batch, batch_indices = D.get_random_batch(batch_size, return_indices=True)
    samples = model.sample(input=batch["world"], max_length=max_length)
    beams = model.beam_search(input=batch["world"], max_length=max_length)

    for i in range(len(batch_indices)):
        index = batch_indices[i]
        H = data.get(index).get("state.sTargetH")
        S = data.get(index).get("state.sTargetS")
        L = data.get(index).get("state.sTargetL")

        utterance_lists = data.get(index).get("utterances[*].nlp.lemmas.lemmas", first=False)
        observed_utt = " # ".join([" ".join(utterance) for utterance in utterance_lists])
        sampled_utt =  " ".join([D["utterance"].get_feature_token(samples[i][0][j,0]).get_value()
                        for j in range(samples[i][1][0])])

        beam, utterance_length, scores = beams[i]
        top_utts = [" ".join([D["utterance"].get_feature_token(beam[k][j]).get_value() for k in range(utterance_length[j])]) for j in range(beam.size(1))]

        print "Condition: " + data.get(index).get("state.condition")
        print "ID: " + data.get(index).get("id")
        print "H: " + str(H) + ", S: " + str(S) + ", L: " + str(L)
        print "True utterance: " + observed_utt
        print "Sampled utterance: " + sampled_utt
        print " "
        print "Top " + str(beam.size(1)) + " utterances"
        for j in range(len(top_utts)):
            print str(scores[j]) + ": " + top_utts[j]
        print "\n"

gpu = bool(int(sys.argv[1]))
data_dir = sys.argv[2]
partition_file = sys.argv[3]
utterance_dir = sys.argv[4]
world_dir = sys.argv[5]
output_results_path = sys.argv[6]
output_model_path = sys.argv[7]

torch.manual_seed(1)
if gpu:
    torch.cuda.manual_seed(1)
np.random.seed(1)

D = MultiviewDataSet.load(data_dir,
                          dfmat_paths={ "world" : world_dir },
                          dfmatseq_paths={ "utterance" : utterance_dir })
partition = Partition.load(partition_file)
D_parts = D.partition(partition, lambda d : d.get("gameid"))
D_train = D_parts["train"]
D_dev = D_parts["dev"].get_random_subset(DEV_SAMPLE_SIZE)
D_dev_close = D_dev.filter(lambda d : d.get("state.condition") == "close")
D_dev_split = D_dev.filter(lambda d : d.get("state.condition") == "split")
D_dev_far = D_dev.filter(lambda d : d.get("state.condition") == "far")

world_size = D_train["world"].get_feature_set().get_size()
utterance_size = D_train["utterance"].get_matrix(0).get_feature_set().get_token_count()
utterance_length = D_train["utterance"].get_feature_seq_set().get_size()

logger = Logger()
data_parameters = DataParameter.make(seq="utterance", input="world")
loss_criterion = VariableLengthNLLLoss()
model=None
if INPUT_TYPE == "EMBEDDED":
    model = SequenceModelInputEmbedded("S0", utterance_size, world_size, \
        EMBEDDING_SIZE, RNN_SIZE, RNN_LAYERS, dropout=DROP_OUT, rnn_type=RNN_TYPE)
else:
    model = SequenceModelInputToHidden("S0", utterance_size, world_size, \
        EMBEDDING_SIZE, RNN_SIZE, RNN_LAYERS, dropout=DROP_OUT, rnn_type=RNN_TYPE)

loss_criterion_unnorm = VariableLengthNLLLoss(norm_dim=True)

if gpu:
    loss_criterion = loss_criterion.cuda()
    model = model.cuda()
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
output_model_samples(best_model, D_dev_split, utterance_length)
output_model_samples(best_model, D_dev_far, utterance_length)

logger.dump(file_path=output_results_path)
best_model.save(output_model_path)
