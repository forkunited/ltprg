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

from ltprg.model.seq import RNNType, SequenceModel, SamplingMode, SequenceModelInputEmbedded, SequenceModelInputToHidden, SequenceModelNoInput
from ltprg.model.seq_heuristic import HeuristicL0
from ltprg.model.eval import Loss
from ltprg.model.meaning import MeaningModelIndexedWorldSequentialUtterance, SequentialUtteranceInputType
from ltprg.model.prior import UniformIndexPriorFn, SequenceSamplingPriorFn
from ltprg.model.rsa import DataParameter, DistributionType, RSA, RSADistributionAccuracy
from ltprg.model.learn import OptimizerType, Trainer
from ltprg.util.log import Logger

RNN_TYPE = RNNType.LSTM
EMBEDDING_SIZE = 100
RNN_SIZE = 100
RNN_LAYERS = 1
TRAINING_ITERATIONS= 10000 #30000 #1000 #00
TRAINING_BATCH_SIZE=128 # 50 #100 #10
DROP_OUT = 0.0
OPTIMIZER_TYPE = OptimizerType.ADADELTA
LEARNING_RATE = 0.5 # 0.0005 #0.05 #BEST 0.001
LOG_INTERVAL = 5000
DEV_SAMPLE_SIZE = 400 # None (none means full)
SAMPLES_PER_INPUT= 4 # 4
SAMPLING_MODE = SamplingMode.BEAM # BEAM FORWARD

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
    S_dist_ps = S_dist.p()

    for i in range(len(batch_indices)):
        index = batch_indices[i]
        H = data.get(index).get("state.sTargetH")
        S = data.get(index).get("state.sTargetS")
        L = data.get(index).get("state.sTargetL")

        utterance_lists = data.get(index).get("utterances[*].nlp.lemmas.lemmas", first=False)
        observed_utt = " # ".join([" ".join(utterance) for utterance in utterance_lists])



        support_utts_i = [" ".join([D["utterance"].get_feature_token(S_dist_support_utts.data[i,j,k]).get_value() \
                                    for k in range(S_dist_support_lens[i,j])]) for j in range(S_dist_ps.size(1))]
        ps_i = S_dist_ps[i]

        print "Condition: " + data.get(index).get("state.condition")
        print "ID: " + data.get(index).get("id")
        print "H: " + str(H) + ", S: " + str(S) + ", L: " + str(L)
        print "True utterance: " + observed_utt
        print " "
        print "Support utterances"
        for j in range(len(support_utts_i)):
            print str(ps_i.data[j]) + ": " + support_utts_i[j]
        print "\n"

gpu = bool(sys.argv[1])
training_dist = sys.argv[2]
training_level = int(sys.argv[3])
data_dir = sys.argv[4]
partition_file = sys.argv[5]
utterance_dir = sys.argv[6]
L_world_dir = sys.argv[7]
L_observation_dir = sys.argv[8]
S_world_dir = sys.argv[9]
S_observation_dir = sys.argv[10]
seq_model_path = sys.argv[11]
output_results_path = sys.argv[12]
meaning_fn_input_type = sys.argv[13]
training_input_mode = sys.argv[14]
prior_beam_heuristic = sys.argv[15]

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
D_dev = D_parts["dev"].get_random_subset(DEV_SAMPLE_SIZE)
D_dev_close = D_dev.filter(lambda d : d.get("state.condition") == "close")
D_dev_split = D_dev.filter(lambda d : d.get("state.condition") == "split")
D_dev_far = D_dev.filter(lambda d : d.get("state.condition") == "far")

world_input_size = D_train["L_observation"].get_feature_set().get_token_count() / 3
utterance_size = D_train["utterance"].get_matrix(0).get_feature_set().get_token_count()

logger = Logger()
data_parameters = DataParameter.make(utterance="utterance", L_world="L_world", L_observation="L_observation",
                                     S_world="S_world", S_observation="S_observation",
                                     mode=training_dist, utterance_seq=True)
loss_criterion = NLLLoss()
if gpu:
    loss_criterion = loss_criterion.cuda()

seq_prior_model = SequenceModel.load(seq_model_path)
world_prior_fn = UniformIndexPriorFn(3, on_gpu=gpu) # 3 colors per observation

seq_meaning_model = None
soft_bottom = None
if meaning_fn_input_type == SequentialUtteranceInputType.IN_SEQ:
    seq_meaning_model = SequenceModelInputEmbedded("Meaning", utterance_size, world_input_size, \
        EMBEDDING_SIZE, RNN_SIZE, RNN_LAYERS, dropout=DROP_OUT, rnn_type=RNN_TYPE)
    soft_bottom = False
else:
    seq_meaning_model = SequenceModelNoInput("Meaning", utterance_size, \
        EMBEDDING_SIZE, RNN_SIZE, RNN_LAYERS, dropout=DROP_OUT, rnn_type=RNN_TYPE)
    soft_bottom = True

meaning_fn = MeaningModelIndexedWorldSequentialUtterance(world_input_size, seq_meaning_model, input_type=meaning_fn_input_type)

beam_heuristic = None
if prior_beam_heuristic == "L0":
    beam_heuristic = HeuristicL0(world_prior_fn, meaning_fn, soft_bottom=soft_bottom)

utterance_prior_fn = SequenceSamplingPriorFn(seq_prior_model, world_input_size, \
                                             mode=SAMPLING_MODE,
                                             samples_per_input=SAMPLES_PER_INPUT,
                                             uniform=True,
                                             seq_length=D["utterance"].get_feature_seq_set().get_size(),
                                             heuristic=beam_heuristic,
                                             training_input_mode=training_input_mode)

rsa_model = RSA.make(training_dist + "_" + str(training_level), training_dist, training_level, meaning_fn, world_prior_fn, utterance_prior_fn, L_bottom=True, soft_bottom=soft_bottom)
if gpu:
    rsa_model = rsa_model.cuda()

loss_criterion_unnorm = NLLLoss(size_average=False)
if gpu:
    loss_criterion_unnorm = loss_criterion_unnorm.cuda()

dev_loss = Loss("Dev Loss", D_dev, data_parameters, loss_criterion_unnorm)

dev_l0_acc = RSADistributionAccuracy("Dev L0 Accuracy", 0, DistributionType.L, D_dev, data_parameters)
dev_close_l0_acc = RSADistributionAccuracy("Dev Close L0 Accuracy", 0, DistributionType.L, D_dev_close, data_parameters)
dev_split_l0_acc = RSADistributionAccuracy("Dev Split L0 Accuracy", 0, DistributionType.L, D_dev_split, data_parameters)
dev_far_l0_acc = RSADistributionAccuracy("Dev Far L0 Accuracy", 0, DistributionType.L, D_dev_far, data_parameters)

dev_l1_acc = RSADistributionAccuracy("Dev L1 Accuracy", 1, DistributionType.L, D_dev, data_parameters)
dev_close_l1_acc = RSADistributionAccuracy("Dev Close L1 Accuracy", 1, DistributionType.L, D_dev_close, data_parameters)
dev_split_l1_acc = RSADistributionAccuracy("Dev Split L1 Accuracy", 1, DistributionType.L, D_dev_split, data_parameters)
dev_far_l1_acc = RSADistributionAccuracy("Dev Far L1 Accuracy", 1, DistributionType.L, D_dev_far, data_parameters)

evaluation = dev_loss
other_evaluations = [dev_l0_acc, dev_l1_acc, \
                      dev_close_l0_acc, dev_split_l0_acc, dev_far_l0_acc, \
                      dev_close_l1_acc, dev_split_l1_acc, dev_far_l1_acc]

trainer = Trainer(data_parameters, loss_criterion, logger, \
            evaluation, other_evaluations=other_evaluations)
rsa_model, best_meaning = trainer.train(rsa_model, D_train, TRAINING_ITERATIONS, \
            batch_size=TRAINING_BATCH_SIZE, optimizer_type=OPTIMIZER_TYPE, lr=LEARNING_RATE, \
            log_interval=LOG_INTERVAL, best_part_fn=lambda m : m.get_meaning_fn())

best_model = RSA.make(training_dist + "_" + str(training_level), training_dist, training_level, best_meaning, world_prior_fn, utterance_prior_fn, L_bottom=True, soft_bottom=soft_bottom)

output_model_samples(best_model, data_parameters, D_dev_close)
output_model_samples(best_model, data_parameters, D_dev_split)
output_model_samples(best_model, data_parameters, D_dev_far)

logger.dump(output_results_path)
