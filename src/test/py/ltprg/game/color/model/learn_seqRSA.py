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
from torch.nn import NLLLoss

from ltprg.model.seq import SequenceModel, SamplingMode, SequenceModelInputEmbedded
from ltprg.model.eval import Loss
from ltprg.model.meaning import MeaningModelIndexedWorldSequentialUtterance
from ltprg.model.prior import UniformIndexPriorFn, SequenceSamplingPriorFn
from ltprg.model.rsa import DataParameter, DistributionType, RSA, RSADistributionAccuracy
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
LOG_INTERVAL = 5
DEV_SAMPLE_SIZE = 500 # None (none means full)

torch.manual_seed(1)
np.random.seed(1)

def output_model_samples(model, data_parameters, D, batch_size=20):
    data = D.get_data()
    batch, batch_indices = D.get_random_batch(batch_size, return_indices=True)
    model_S = model.to_level(DistributionType.S, 1)
    dp_S = data_parameters.to_mode(DistributionType.S)
    S_dist = model_S.forward_batch(batch, dp_S)
    S_dist_support_utts = S_dist.support()[0]
    S_dist_support_lens = S_dist.support()[1]
    S_dist_ps = S_dist.ps()

    for i in range(len(batch_indices)):
        index = batch_indices[i]
        H = data.get(index).get("state.sTargetH")
        S = data.get(index).get("state.sTargetS")
        L = data.get(index).get("state.sTargetL")

        utterance_lists = data.get(index).get("utterances[*].nlp.lemmas.lemmas", first=False)
        observed_utt = " # ".join([" ".join(utterance) for utterance in utterance_lists])



        support_utts_i = [" ".join([D["utterance"].get_feature_token(S_dist_support_utts[i,j,k]).get_value() \
                                    for k in range(S_dist_support_lens[i,j])]) for j in range(S_dist_ps.size(1))]
        ps_i = S_dist_ps[i]

        print "Condition: " + data.get(index).get("state.condition")
        print "ID: " + data.get(index).get("id")
        print "H: " + str(H) + ", S: " + str(S) + ", L: " + str(L)
        print "True utterance: " + observed_utt
        print " "
        print "Support utterances"
        for j in range(len(support_utts_i)):
            print str(ps_i[j]) + ": " + support_utts_i[j]
        print "\n"


training_dist = sys.argv[1]
training_level = int(sys.argv[2])
data_dir = sys.argv[3]
partition_file = sys.argv[4]
utterance_dir = sys.argv[5]
L_world_dir = sys.argv[6]
L_observation_dir = sys.argv[7]
S_world_dir = sys.argv[8]
S_observation_dir = sys.argv[9]
seq_model_path = sys.argv[10]
output_results_path = sys.argv[11]

D = MultiviewDataSet.load(data_dir,
                          dfmat_paths={ "L_world" : L_world_dir, \
                                        "L_observation" : L_observation_dir, \
                                        "S_world" : S_world_dir, \
                                        "S_observation" : S_observation_dir \
                           },
                          dfmatseq_paths={ "utterance" : utterance_dir })
partition = Partition.load(partition_file)

D_parts = D.partition(partition, lambda d : d.get("gameid"))
D_train = D_parts["train"]
D_dev = D_parts["dev"].get_random_sample(DEV_SAMPLE_SIZE)
D_dev_close = D_dev.filter(lambda d : d.get("state.condition") == "close")
D_dev_split = D_dev.filter(lambda d : d.get("state.condition") == "split")
D_dev_far = D_dev.filter(lambda d : d.get("state.condition") == "far")

world_input_size = 3
utterance_size = D_train["utterance"].get_matrix(0).get_feature_set().get_token_count()

logger = Logger()
data_parameters = DataParameter.make(utterance="utterance", L_world="L_world", L_observation="L_observation",
                                     S_world="S_world", S_observation="S_observation",
                                     mode=training_dist, utterance_seq=True)
loss_criterion = NLLLoss()

seq_prior_model = SequenceModel.load(seq_model_path)
world_prior_fn = UniformIndexPriorFn(3) # 3 colors per observation
utterance_prior_fn = SequenceSamplingPriorFn(seq_prior_model, world_input_size, \
                                             mode=SamplingMode.FORWARD,
                                             samples_per_input=1,
                                             uniform=True,
                                             seq_length=15) # 3 is color dimension
seq_meaning_model = SequenceModelInputEmbedded("Meaning", utterance_size, world_input_size, \
    EMBEDDING_SIZE, RNN_SIZE, RNN_LAYERS, dropout=DROP_OUT)
meaning_fn = MeaningModelIndexedWorldSequentialUtterance(world_input_size, seq_meaning_model)
rsa_model = RSA.make(training_dist, training_level, meaning_fn, world_prior_fn, utterance_prior_fn, L_bottom=True)

dev_loss = Loss("Dev Loss", D_dev, data_parameters, loss_criterion)

dev_l0_acc = RSADistributionAccuracy("Dev L0 Accuracy", 0, DistributionType.L, D_dev, data_parameters)
#dev_close_l0_acc = RSADistributionAccuracy("Dev Close L0 Accuracy", 0, DistributionType.L, D_dev_close, data_parameters)
#dev_split_l0_acc = RSADistributionAccuracy("Dev Split L0 Accuracy", 0, DistributionType.L, D_dev_split, data_parameters)
#dev_far_l0_acc = RSADistributionAccuracy("Dev Far L0 Accuracy", 0, DistributionType.L, D_dev_far, data_parameters)

dev_l1_acc = RSADistributionAccuracy("Dev L1 Accuracy", 1, DistributionType.L, D_dev, data_parameters)
#dev_close_l1_acc = RSADistributionAccuracy("Dev Close L1 Accuracy", 1, DistributionType.L, D_dev_close, data_parameters)
#dev_split_l1_acc = RSADistributionAccuracy("Dev Split L1 Accuracy", 1, DistributionType.L, D_dev_split, data_parameters)
#dev_far_l1_acc = RSADistributionAccuracy("Dev Far L1 Accuracy", 1, DistributionType.L, D_dev_far, data_parameters)

evaluation = dev_loss
other_evaluations = [dev_l0_acc, dev_l1_acc]
                     # dev_close_l0_acc, dev_split_l0_acc, dev_far_l0_acc, \
                     #, dev_close_l1_acc, dev_split_l1_acc, dev_far_l1_acc]

trainer = Trainer(data_parameters, loss_criterion, logger, \
            evaluation, other_evaluations=other_evaluations)
rsa_model, best_model = trainer.train(rsa_model, D_train, TRAINING_ITERATIONS, \
            batch_size=TRAINING_BATCH_SIZE, lr=LEARNING_RATE, \
            log_interval=LOG_INTERVAL)

output_model_samples(best_model, data_parameters, D_dev_close)
output_model_samples(best_model, data_parameters, D_dev_split)
output_model_samples(best_model, data_parameters, D_dev_far)

logger.dump(output_results_path)
