import sys
import torch.nn as nn
import time
import numpy as np
import torch
import copy
from torch.autograd import Variable
from torch.optim import Adam
from mung.feature import MultiviewDataSet, Symbol
from mung.data import Partition

from ltprg.model.eval import Loss
from ltprg.model.seq import SequenceModelInputToHidden, VariableLengthNLLLoss, DataParameter
from ltprg.model.learn import Trainer
from ltprg.util.log import Logger

RNN_TYPE = "GRU" # LSTM currently broken... need to make cell state
EMBEDDING_SIZE = 100
RNN_SIZE = 100
RNN_LAYERS = 1
TRAINING_ITERATIONS=200#1000 #00
TRAINING_BATCH_SIZE=100
DROP_OUT = 0.5
LEARNING_RATE = 0.005 #0.05 #0.001
LOG_INTERVAL = 200

torch.manual_seed(1)
np.random.seed(1)

def output_model_samples(model, D, batch_size=20):
    data = D.get_data()
    batch, batch_indices = D.get_random_batch(batch_size, return_indices=True)
    samples = model.sample(input=batch["world"])
    beams = model.beam_search(input=batch["world"])

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

data_dir = sys.argv[1]
partition_file = sys.argv[2]
utterance_dir = sys.argv[3]
world_dir = sys.argv[4]
output_results_path = sys.argv[5]
output_model_path = sys.argv[6]

D = MultiviewDataSet.load(data_dir,
                          dfmat_paths={ "world" : world_dir },
                          dfmatseq_paths={ "utterance" : utterance_dir })
partition = Partition.load(partition_file)
D_parts = D.partition(partition, lambda d : d.get("gameid"))
D_train = D_parts["train"]
D_dev = D_parts["dev"]
D_dev_close = D_dev.filter(lambda d : d.get("state.condition") == "close")
D_dev_split = D_dev.filter(lambda d : d.get("state.condition") == "split")
D_dev_far = D_dev.filter(lambda d : d.get("state.condition") == "far")

world_size = D_train["world"].get_feature_set().get_size()
utterance_size = D_train["utterance"].get_matrix(0).get_feature_set().get_token_count()

logger = Logger()
data_parameters = DataParameter.make(seq="utterance", input="world")
loss_criterion = VariableLengthNLLLoss()
model = SequenceModelInputToHidden("S0", utterance_size, world_size, \
    EMBEDDING_SIZE, RNN_SIZE, RNN_LAYERS, dropout=DROP_OUT)

dev_loss = Loss("Dev Loss", D_dev, data_parameters, loss_criterion)
dev_close_loss = Loss("Dev Close Loss", D_dev_close, data_parameters, loss_criterion)
dev_split_loss = Loss("Dev Split Loss", D_dev_split, data_parameters, loss_criterion)
dev_far_loss = Loss("Dev Far Loss", D_dev_far, data_parameters, loss_criterion)

evaluation = dev_loss
other_evaluations = [dev_close_loss, dev_split_loss, dev_far_loss]

trainer = Trainer(data_parameters, loss_criterion, logger, \
            evaluation, other_evaluations=other_evaluations)
model, best_model = trainer.train(model, D_train, TRAINING_ITERATIONS, \
            batch_size=TRAINING_BATCH_SIZE, lr=LEARNING_RATE, \
            log_interval=LOG_INTERVAL)

output_model_samples(best_model, D_dev_close)
output_model_samples(best_model, D_dev_split)
output_model_samples(best_model, D_dev_far)

logger.dump(output_results_path)
torch.save(best_model.state_dict(), output_model_path)

