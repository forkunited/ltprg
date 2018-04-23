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
from ltprg.model.obs import ObservationModel, ObservationModelReorderedSequential
from ltprg.model.seq_heuristic import HeuristicL0
from ltprg.model.meaning import MeaningModel, MeaningModelIndexedWorldSequentialUtterance, SequentialUtteranceInputType
from ltprg.model.prior import UniformIndexPriorFn, SequenceSamplingPriorFn, MultiLayerIndexPriorFn
from ltprg.model.rsa import DataParameter, DistributionType, RSA, RSADistributionAccuracy
from ltprg.game.color.eval import ColorMeaningPlot

RNN_TYPE = RNNType.LSTM
BIDIRECTIONAL=True
INPUT_LAYERS = 1
RNN_LAYERS = 1
TRAINING_ITERATIONS=7000 #7000 #9000 #10000 #30000 #1000 #00
DROP_OUT = 0.5 #0.5 # BEST 0.5
OPTIMIZER_TYPE = OptimizerType.ADAM #ADADELTA # BEST ADAM
LOG_INTERVAL = 100 #100
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
target_dir = sys.argv[7]
observation_dir = sys.argv[8]
seq_model_path = sys.argv[9]
output_results_path = sys.argv[10]
meaning_fn_input_type = sys.argv[11]
training_input_mode = sys.argv[12]
prior_beam_heuristic = sys.argv[13]
learning_rate = float(sys.argv[14])
batch_size = int(sys.argv[15])
input_rep_name = sys.argv[16]
utterance_prior_name= sys.argv[17]
samples_per_input = int(sys.argv[18])
embedding_size = int(sys.argv[19])
rnn_size = int(sys.argv[20])
training_data_size = sys.argv[21]
seed = int(sys.argv[22])
alpha = float(sys.argv[23])
output_meaning_prefix = sys.argv[24]
final_output_results_path = sys.argv[25]
training_sampling_mode = sys.argv[26]
eval_sampling_mode = sys.argv[27]
small_sample_size = sys.argv[28]
selection_eval_trials = int(sys.argv[29])
selection_model_type = sys.argv[30]
world_prior_depth = int(sys.argv[31])
output_meaning_model_path = sys.argv[32]
training_condition = sys.argv[33]
obs_seq = bool(int(sys.argv[34]))
existing_meaning_model_path = sys.argv[35]
existing_obs_model_path = sys.argv[36]
training_source = sys.argv[37]

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

D = None
if obs_seq:
    D = MultiviewDataSet.load(data_dir,
                          dfmat_paths={ "target" : target_dir },
                          dfmatseq_paths={ "observation" : observation_dir, "utterance" : utterance_dir },
                          ordering_seq="observation")
else:
    D = MultiviewDataSet.load(data_dir,
                          dfmat_paths={ "target" : target_dir, "observation" : observation_dir },
                          dfmatseq_paths={ "utterance" : utterance_dir },
                          ordering_seq="utterance")

partition = Partition.load(partition_file)

D_parts = D.partition(partition, lambda d : d.get("gameid"))
D_train = D_parts["train"]

if training_data_size is not None:
    D_train.shuffle()
    D_train = D_train.get_subset(0, training_data_size)

D_dev = D_parts["dev"]

D_dev_sample = D_dev
if small_sample_size is not None:
    D_dev_sample = D_dev.get_random_subset(small_sample_size)

D_dev_close = D_dev.filter(lambda d : d.get("state.state.condition.name") == "CLOSE")
D_dev_split = D_dev.filter(lambda d : d.get("state.state.condition.name") == "SPLIT")
D_dev_far = D_dev.filter(lambda d : d.get("state.state.condition.name") == "FAR")

utterance_size = D_train["utterance"].get_matrix(0).get_feature_set().get_token_count()

data_parameters = DataParameter.make(utterance="utterance", L_world="target", L_observation="observation",
                                     S_world="target", S_observation="observation",
                                     mode=training_dist, utterance_seq=True)
loss_criterion = NLLLoss()
if gpu:
    loss_criterion = loss_criterion.cuda()

seq_prior_model = SequenceModel.load(seq_model_path)

observation_fn = None
world_input_size = None
if obs_seq:
    # 3 Is for one-hot indicating the target  #Old InputEmbedded
    world_input_size = rnn_size/2 # + 3 in indexed model
    if BIDIRECTIONAL:
        world_input_size *= 2

    if existing_obs_model_path == "None":
        observation_size = D_train["observation"].get_matrix(0).get_feature_set().get_token_count()
        seq_observation_model = SequenceModelNoInput("Observation", observation_size, #3, \
            embedding_size, rnn_size/2, RNN_LAYERS, dropout=DROP_OUT, rnn_type=RNN_TYPE, bidir=BIDIRECTIONAL, non_emb=True)
        observation_fn = ObservationModelReorderedSequential(world_input_size, 3, seq_observation_model)
    else:
        observation_fn = ObservationModel.load(existing_obs_model_path)
else:
    world_input_size = D_train["observation"].get_feature_set().get_token_count() / 3

soft_bottom = (meaning_fn_input_type != SequentialUtteranceInputType.IN_SEQ)

meaning_fn = None
if existing_meaning_model_path == "None:"
    seq_meaning_model = None
    if meaning_fn_input_type == SequentialUtteranceInputType.IN_SEQ:
        seq_meaning_model = SequenceModelInputToHidden("Meaning", utterance_size, world_input_size, \
            embedding_size, rnn_size, RNN_LAYERS, dropout=DROP_OUT, rnn_type=RNN_TYPE, bidir=BIDIRECTIONAL, input_layers=INPUT_LAYERS)
    else:
        seq_meaning_model = SequenceModelNoInput("Meaning", utterance_size, \
            embedding_size, rnn_size, RNN_LAYERS, dropout=DROP_OUT, rnn_type=RNN_TYPE, bidir=BIDIRECTIONAL)
    meaning_fn = MeaningModelIndexedWorldSequentialUtterance(world_input_size, seq_meaning_model, input_type=meaning_fn_input_type)
else:
    meaning_fn = MeaningModel.load(existing_meaning_model_path)

world_prior_fn = UniformIndexPriorFn(3, on_gpu=gpu, unnorm=soft_bottom) # 3 objs per observation
if world_prior_depth > 0:
    world_prior_fn = MultiLayerIndexPriorFn(3, world_input_size*3, world_prior_depth, on_gpu=gpu, unnorm=soft_bottom) # FIXME Currently broken

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

rsa_model = RSA.make(training_dist + "_" + str(training_level), training_dist, training_level, meaning_fn, world_prior_fn, utterance_prior_fn, L_bottom=True, soft_bottom=soft_bottom, alpha=alpha, observation_fn=observation_fn)
if gpu:
    rsa_model = rsa_model.cuda()

loss_criterion_unnorm = NLLLoss(size_average=False)
if gpu:
    loss_criterion_unnorm = loss_criterion_unnorm.cuda()

dev_l0_sample_acc = RSADistributionAccuracy("Dev Sample L0 Accuracy", 0, DistributionType.L, D_dev_sample, data_parameters)
dev_l1_sample_acc = RSADistributionAccuracy("Dev Sample L1 Accuracy", 1, DistributionType.L, D_dev_sample, data_parameters, trials=selection_eval_trials)

evaluation = dev_l1_sample_acc # l1
other_evaluations = [dev_l0_sample_acc]
#if selection_model_type == "L_0":
#    evaluation = dev_l0_sample_acc
#    other_evaluations = [dev_l1_sample_acc]

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

best_model = RSA.make(training_dist + "_" + str(training_level), training_dist, training_level, best_meaning, world_prior_fn, utterance_prior_fn, L_bottom=True, soft_bottom=soft_bottom, alpha=alpha, observation_fn=observation_fn)

#output_model_samples(best_model, data_parameters, D_dev_close)
#output_model_samples(best_model, data_parameters, D_dev_split)
#output_model_samples(best_model, data_parameters, D_dev_far)

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
