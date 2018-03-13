import sys
import torch.nn as nn
import time
import numpy as np
import torch
import copy
import torch.cuda
import ltprg.data.feature
from torch.autograd import Variable
from torch.optim import Adam
from mung.feature import MultiviewDataSet, Symbol
from mung.data import Partition
from mung.torch_ext.eval import Loss, Evaluation
from mung.torch_ext.learn import OptimizerType, Trainer
from mung.util.log import Logger

from torch.nn import NLLLoss

from ltprg.model.seq import RNNType, SequenceModel, SamplingMode, SequenceModelInputEmbedded, SequenceModelInputToHidden, SequenceModelNoInput
from ltprg.model.seq_heuristic import HeuristicL0
from ltprg.model.meaning import MeaningModelIndexedWorldSequentialUtterance, SequentialUtteranceInputType
from ltprg.model.prior import UniformIndexPriorFn, SequenceSamplingPriorFn, MultiLayerIndexPriorFn
from ltprg.model.rsa import DataParameter, DistributionType, RSA, RSADistributionAccuracy
from ltprg.game.color.eval import ColorMeaningPlot

RNN_TYPE = RNNType.LSTM
BIDIRECTIONAL=True
INPUT_LAYERS = 1
RNN_LAYERS = 1
TRAINING_ITERATIONS=20000 #7000 #9000 #10000 #30000 #1000 #00
DROP_OUT = 0.0 # BEST 0.5
OPTIMIZER_TYPE = OptimizerType.ADAM #ADADELTA # BEST ADAM
LOG_INTERVAL = 100
N_BEFORE_HEURISTIC=100
SAMPLE_LENGTH = 8
GRADIENT_CLIPPING = 5.0 #5.0 # 5.0
WEIGHT_DECAY=0.0 # 1e-6

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

        utterance_lists = data.get(index).get("utterances[*].nlp.lemmas.lemmas", first=False)
        observed_utt = " # ".join([" ".join(utterance) for utterance in utterance_lists])



        support_utts_i = [" ".join([D["utterance"].get_feature_token(S_dist_support_utts.data[i,j,k]).get_value() \
                                    for k in range(S_dist_support_lens[i,j])]) for j in range(S_dist_ps.size(1))]
        ps_i = S_dist_ps[i]

        print "Diffs: " + data.get(index).get("state.diffs")
        print "ID: " + data.get(index).get("id")
        print "True utterance: " + observed_utt
        print " "
        print "Support utterances"
        for j in range(len(support_utts_i)):
            print str(ps_i.data[j]) + ": " + support_utts_i[j]
        print "\n"

gpu = bool(int(sys.argv[1]))
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
learning_rate = float(sys.argv[16])
batch_size = int(sys.argv[17])
input_rep_name = sys.argv[18]
utterance_prior_name= sys.argv[19]
samples_per_input = int(sys.argv[20])
embedding_size = int(sys.argv[21])
rnn_size = int(sys.argv[22])
training_data_size = sys.argv[23]
seed = int(sys.argv[24])
alpha = float(sys.argv[25])
output_meaning_prefix = sys.argv[26]
final_output_results_path = sys.argv[27]
training_sampling_mode = sys.argv[28]
eval_sampling_mode = sys.argv[29]
small_sample_size = sys.argv[30]
selection_eval_trials = int(sys.argv[31])
selection_model_type = sys.argv[32]
world_prior_depth = int(sys.argv[33])
output_meaning_model_path = sys.argv[34]
training_condition = sys.argv[35]
data_condition = sys.argv[36]

if training_data_size == "None":
    training_data_size = None
else:
    training_data_size = int(training_data_size)

if small_sample_size == "None":
    small_sample_size = None
else:
    small_sample_size = int(small_sample_size)

if gpu:
    torch.cuda.manual_seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)

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
D_dev = D_parts["dev"]

grid_size = (D_train["L_observation"].get_feature_set().get_token_count() / 2) / 3 # FIXME Note this is broken... assumes cielab for 3

close_far_split_field = None
close_far_split = None
if data_condition == "SKEWED":
    close_far_split_field = "state.skewProb"
    close_far_split = 0.5
elif data_condition == "COLORDIFFS":
    close_far_split_field = "state.colorDiffMin"
    close_far_split = 12.5
else:
    close_far_split_field = "state.diffs"
    close_far_split = np.ceil(grid_size/2.0)

if grid_size == 1 and data_condition != "SKEWED" and data_condition != "COLORDIFFS":
    D_train_close = D_train 
    D_train_far = D_train
    D_dev_close = D_dev
    D_dev_far = D_dev
elif grid_size % 2 == 0 and data_condition != "SKEWED" and data_condition != "COLORDIFFS":
    D_train_close = D_train.filter(lambda d : int(d.get(close_far_split_field)) <= close_far_split)
    D_train_far = D_train.filter(lambda d : int(d.get(close_far_split_field)) > close_far_split)
    D_dev_close = D_dev.filter(lambda d : int(d.get(close_far_split_field)) <= close_far_split)
    D_dev_far = D_dev.filter(lambda d : int(d.get(close_far_split_field)) > close_far_split)
else:
    D_train_close = D_train.filter(lambda d : float(d.get(close_far_split_field)) < close_far_split)
    D_train_far = D_train.filter(lambda d : float(d.get(close_far_split_field)) > close_far_split)
    D_dev_close = D_dev.filter(lambda d : float(d.get(close_far_split_field)) < close_far_split)
    D_dev_far = D_dev.filter(lambda d : float(d.get(close_far_split_field)) > close_far_split)

# FIXME Put this back later
#if D_train_close.get_size() > D_train_far.get_size():
#    D_train_close = D_train_close.get_random_subset(D_train_far.get_size())
#    D_dev_close = D_dev_close.get_random_subset(D_dev_far.get_size())
#elif D_train_close.get_size() < D_train_far.get_size():
#    D_train_far = D_train_far.get_random_subset(D_train_close.get_size())
#    D_dev_far = D_dev_far.get_random_subset(D_dev_close.get_size())

# FIXME Temporary hack to make all data conditions the same size
D_train_far = D_train_far.get_random_subset(6000)
D_dev_far = D_dev_far.get_random_subset(2000)
D_train_close = D_train_close.get_random_subset(6000)
D_dev_close = D_dev_close.get_random_subset(2000)

print "Split train into " + str(D_train_close.get_size()) + " close and " + str(D_train_far.get_size()) + " far"
print "Split dev into " + str(D_dev_close.get_size()) + " close and " + str(D_dev_far.get_size()) + " far"

if training_condition == "close":
    D_train = D_train_close
elif training_condition == "far":
    D_train = D_train_far

if training_data_size is not None:
    D_train.shuffle()
    D_train = D_train.get_subset(0, training_data_size)

D_dev_sample = D_dev
if small_sample_size is not None:
    D_dev_sample = D_dev.get_random_subset(small_sample_size)

world_input_size = D_train["L_observation"].get_feature_set().get_token_count() / 2
utterance_size = D_train["utterance"].get_matrix(0).get_feature_set().get_token_count()

data_parameters = DataParameter.make(utterance="utterance", L_world="L_world", L_observation="L_observation",
                                     S_world="S_world", S_observation="S_observation",
                                     mode=training_dist, utterance_seq=True)
loss_criterion = NLLLoss()
if gpu:
    loss_criterion = loss_criterion.cuda()

seq_prior_model = SequenceModel.load(seq_model_path)

seq_meaning_model = None
soft_bottom = None
if meaning_fn_input_type == SequentialUtteranceInputType.IN_SEQ:
    seq_meaning_model = SequenceModelInputToHidden("Meaning", utterance_size, world_input_size, \
        embedding_size, rnn_size, RNN_LAYERS, dropout=DROP_OUT, rnn_type=RNN_TYPE, bidir=BIDIRECTIONAL, input_layers=INPUT_LAYERS)
    soft_bottom = False
else:
    seq_meaning_model = SequenceModelNoInput("Meaning", utterance_size, \
        embedding_size, rnn_size, RNN_LAYERS, dropout=DROP_OUT, rnn_type=RNN_TYPE, bidir=BIDIRECTIONAL)
    soft_bottom = True

meaning_fn = MeaningModelIndexedWorldSequentialUtterance(world_input_size, seq_meaning_model, input_type=meaning_fn_input_type)#, encode_input=True, enc_size=100)

world_prior_fn = UniformIndexPriorFn(2, on_gpu=gpu, unnorm=soft_bottom) # 2 objs per observation
if world_prior_depth > 0:
    world_prior_fn = MultiLayerIndexPriorFn(2, world_input_size*2, world_prior_depth, on_gpu=gpu, unnorm=soft_bottom)

beam_heuristic = None
if prior_beam_heuristic == "L0":
    beam_heuristic = HeuristicL0(world_prior_fn, meaning_fn, soft_bottom=soft_bottom)

utterance_prior_fn = SequenceSamplingPriorFn(seq_prior_model, world_input_size, \
                                             training_mode=training_sampling_mode,
                                             eval_mode=eval_sampling_mode,
                                             samples_per_input=samples_per_input,
                                             uniform=True,
                                             seq_length=D["utterance"].get_feature_seq_set().get_size(),
                                             heuristic=beam_heuristic,
                                             training_input_mode=training_input_mode,
                                             sample_length=SAMPLE_LENGTH,
                                             n_before_heuristic=N_BEFORE_HEURISTIC)

rsa_model = RSA.make(training_dist + "_" + str(training_level), training_dist, training_level, meaning_fn, world_prior_fn, utterance_prior_fn, L_bottom=True, soft_bottom=soft_bottom, alpha=alpha)
if gpu:
    rsa_model = rsa_model.cuda()

loss_criterion_unnorm = NLLLoss(size_average=False)
if gpu:
    loss_criterion_unnorm = loss_criterion_unnorm.cuda()

dev_l0_sample_acc = RSADistributionAccuracy("Dev Sample L0 Accuracy", 0, DistributionType.L, D_dev_sample, data_parameters)
dev_l1_sample_acc = RSADistributionAccuracy("Dev Sample L1 Accuracy", 1, DistributionType.L, D_dev_sample, data_parameters, trials=selection_eval_trials)

evaluation = dev_l1_sample_acc
other_evaluations = [dev_l0_sample_acc]
if selection_model_type == "L_0":
    evaluation = dev_l0_sample_acc
    other_evaluations = [dev_l1_sample_acc]

logger = Logger()
final_logger = Logger()

record_prefix = dict()
record_prefix["seed"] = seed
record_prefix["arch"] = meaning_fn_input_type
record_prefix["lr"] = learning_rate
record_prefix["bsz"] = batch_size
record_prefix["input_rep"] = input_rep_name
record_prefix["utt_prior"] = utterance_prior_name
record_prefix["samples_per_input"] = samples_per_input
record_prefix["rnn_size"] = rnn_size
record_prefix["embedding_size"] = embedding_size
record_prefix["training_size"] = training_data_size
record_prefix["alpha"] = alpha
record_prefix["selection_model_type"] = selection_model_type
record_prefix["training_condition"] = training_condition
logger.set_record_prefix(record_prefix)
logger.set_file_path(output_results_path)
final_logger.set_record_prefix(record_prefix)
final_logger.set_file_path(final_output_results_path)

trainer = Trainer(data_parameters, loss_criterion, logger, \
            evaluation, other_evaluations=other_evaluations, max_evaluation=True)
rsa_model, best_meaning, best_iteration = trainer.train(rsa_model, D_train, TRAINING_ITERATIONS, \
            batch_size=batch_size, optimizer_type=OPTIMIZER_TYPE, lr=learning_rate, weight_decay=WEIGHT_DECAY, \
            grad_clip=GRADIENT_CLIPPING, log_interval=LOG_INTERVAL, best_part_fn=lambda m : m.get_meaning_fn())

best_model = RSA.make(training_dist + "_" + str(training_level), training_dist, training_level, best_meaning, world_prior_fn, utterance_prior_fn, L_bottom=True, soft_bottom=soft_bottom, alpha=alpha)

#output_model_samples(best_model, data_parameters, D_dev_sample_close)
#output_model_samples(best_model, data_parameters, D_dev_sample_split)
#output_model_samples(best_model, data_parameters, D_dev_sample_far)

logger.dump(file_path=output_results_path, record_prefix=record_prefix)

train_loss =  Loss("Train Loss", D_train, data_parameters, loss_criterion_unnorm)
dev_loss = Loss("Dev Loss", D_dev, data_parameters, loss_criterion_unnorm)
dev_l0_acc = RSADistributionAccuracy("Dev L0 Accuracy", 0, DistributionType.L, D_dev, data_parameters)
dev_l1_acc = RSADistributionAccuracy("Dev L1 Accuracy", 1, DistributionType.L, D_dev, data_parameters)
dev_close_l0_acc = RSADistributionAccuracy("Dev Close L0 Accuracy", 0, DistributionType.L, D_dev_close, data_parameters)
dev_close_l1_acc = RSADistributionAccuracy("Dev Close L1 Accuracy", 1, DistributionType.L, D_dev_close, data_parameters)
dev_far_l0_acc = RSADistributionAccuracy("Dev Far L0 Accuracy", 0, DistributionType.L, D_dev_far, data_parameters)
dev_far_l1_acc = RSADistributionAccuracy("Dev Far L1 Accuracy", 1, DistributionType.L, D_dev_far, data_parameters)


final_evals = [train_loss, dev_loss, \
               dev_l0_acc, dev_l1_acc, \
               dev_close_l0_acc, dev_close_l1_acc, \
               dev_far_l0_acc, dev_far_l1_acc]

results = Evaluation.run_all(final_evals, best_model)
results["Model"] = best_model.get_name()
results["Iteration"] = best_iteration
final_logger.log(results)
final_logger.dump(file_path=final_output_results_path, record_prefix=record_prefix)

best_meaning.save(output_meaning_model_path)
