import os.path as path
import datetime
import argparse
import torch
import json
import numpy as np
import ltprg.data.feature
import mung.config.feature as cfeature
import mung.config.torch_ext.learn as clearn
import ltprg.config.rsa as crsa
import ltprg.config.seq as cseq
from mung.util.config import Config
from mung.util.log import Logger
from mung.torch_ext.eval import Evaluation
from ltprg.util.file import make_indexed_dir
from ltprg.model.rsa import RSA
from ltprg.model.seq import VariableLengthNLLLoss, DataParameter
from ltprg.data.curriculum import make_sua_datum_token_frequency_fn

parser = argparse.ArgumentParser()
parser.add_argument('id', action="store")
parser.add_argument('env', action="store")
parser.add_argument('data', action="store")
parser.add_argument('model', action="store")
parser.add_argument('learn', action="store")
parser.add_argument('rsa_train_evals', action="store")
parser.add_argument('rsa_dev_evals', action="store")
parser.add_argument('rsa_test_evals', action="store")
parser.add_argument('s_train_evals', action="store")
parser.add_argument('s_dev_evals', action="store")
parser.add_argument('s_test_evals', action="store")
parser.add_argument('output_dir', action="store")
parser.add_argument('--seed', action='store', dest='seed', type=int, default=1)
parser.add_argument('--gpu', action='store', dest='gpu', type=int, default=1)
parser.add_argument('--eval_test', action='store', dest='eval_test', type=int, default=0)
parser.add_argument('--clean_length', action='store', dest='clean_length', type=int, default=12)
parser.add_argument('--only_correct_clean_train', action='store', dest='only_correct_clean_train', type=bool, default=True)
args, extra_env_args = parser.parse_known_args()

# Initalize current run parameters
id = args.id
gpu = bool(args.gpu)
seed = args.seed
eval_test = bool(args.eval_test)
output_dir = args.output_dir
output_path = make_indexed_dir(path.join(output_dir, str(id) + "_seed" + str(seed)))

extra_env = Config.load_from_dict({ "output_path" : output_path })
extra_env = Config.load_from_list(extra_env_args).merge(extra_env)

# Load configuration files
env = Config.load(args.env).merge(extra_env)
data_config = Config.load(args.data, environment=env)
model_config = Config.load(args.model, environment=env)
learn_config = Config.load(args.learn, environment=env)
rsa_train_evals_config = Config.load(args.rsa_train_evals, environment=env)
rsa_dev_evals_config = Config.load(args.rsa_dev_evals, environment=env)
rsa_test_evals_config = Config.load(args.rsa_test_evals, environment=env)
s_train_evals_config = Config.load(args.s_train_evals, environment=env)
s_dev_evals_config = Config.load(args.s_dev_evals, environment=env)
s_test_evals_config = Config.load(args.s_test_evals, environment=env)

# Random seed
if gpu:
    torch.cuda.manual_seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)

# Setup data, model, and evaluations
_, data_sets = cfeature.load_mvdata(data_config)

keys = set(data_sets.keys())
for key in keys:
    if key.startswith("train") or key.startswith("dev") or key.startswith("test"):
        new_key = key + "_cleanutts"
        data_sets[new_key] = data_sets[key].filter(lambda d : len(d.get("utterances")) == 1 and len(d.get("utterances[0].nlp.clean_strs.strs")) <= args.clean_length)
        print "Created extra data set " + new_key + " of size " + str(data_sets[new_key].get_size()) + " from " + str(data_sets[key].get_size())

        if args.only_correct_clean_train and key.startswith("train"):
            data_sets[new_key] = data_sets[new_key].filter(lambda d : d.get("state.state.target") == d.get("state.state.listenerOrder")[d.get("action.action.lClicked")] )
            print "Filtered clean train data to only listener correct examples (" + str(data_sets[new_key].get_size()) + ")"


rsa_data_parameter, rsa_model = crsa.load_rsa_model(model_config, data_sets["train"], gpu=gpu)
seq_data_parameter = DataParameter.make(**model_config["seq_data_parameter"])
rsa_train_evals = crsa.load_evaluations(rsa_train_evals_config, data_sets, gpu=gpu)
rsa_dev_evals = crsa.load_evaluations(rsa_dev_evals_config, data_sets, gpu=gpu)
rsa_test_evals = crsa.load_evaluations(rsa_test_evals_config, data_sets, gpu=gpu)
s_train_evals = cseq.load_evaluations(s_train_evals_config, data_sets, gpu=gpu)
s_dev_evals = cseq.load_evaluations(s_dev_evals_config, data_sets, gpu=gpu)
s_test_evals = cseq.load_evaluations(s_test_evals_config, data_sets, gpu=gpu)

# Setup output files
log_path = path.join(output_path, "log")
results_path = path.join(output_path, "results")
test_results_path = path.join(output_path, "test_results")
meaning_model_path = path.join(output_path, "model_meaning")
observation_model_path = path.join(output_path, "model_observation")
s_model_path = path.join(output_path, "model_s")
config_output_path = path.join(output_path, "config.json")

# Output config
full_config = dict()
full_config["id"] = id
full_config["gpu"] = gpu
full_config["seed"] = seed
full_config["time"] = str(datetime.datetime.now())
full_config["eval_test"] = eval_test
full_config["output_dir"] = output_dir
full_config["env"] = env.get_dict()
full_config["data"] = data_config.get_dict()
full_config["model"] = model_config.get_dict()
full_config["learn"] = learn_config.get_dict()
full_config["rsa_train_evals"] = rsa_train_evals_config.get_dict()
full_config["rsa_dev_evals"] = rsa_dev_evals_config.get_dict()
full_config["rsa_test_evals"] = rsa_test_evals_config.get_dict()
full_config["s_train_evals"] = s_train_evals_config.get_dict()
full_config["s_dev_evals"] = s_dev_evals_config.get_dict()
full_config["s_test_evals"] = s_test_evals_config.get_dict()
with open(config_output_path, 'w') as fp:
    json.dump(full_config, fp)

rsa_logger = Logger()
rsa_logger.set_file_path(log_path + "_rsa")
s_logger = Logger()
s_logger.set_file_path(log_path + "_s")

# Run training 
rsa_loss_criterion = torch.nn.NLLLoss()
s_loss_criterion = VariableLengthNLLLoss()
best_part_fn = lambda m : (m.get_meaning_fn(), m.get_observation_fn())
last_model, best_part, best_iteration = clearn.cotrain_from_config(learn_config, \
    [seq_data_parameter, rsa_data_parameter], [s_loss_criterion, rsa_loss_criterion], [s_logger, rsa_logger], [s_train_evals, rsa_train_evals], \
    [rsa_model.get_meaning_fn().get_seq_model(),rsa_model], data_sets, best_part_fns=[(lambda m : m), best_part_fn], \
    curriculum_key_fn_constructor=make_sua_datum_token_frequency_fn)

best_meaning_fn = best_part[0]
best_observation_fn = best_part[1]
level = last_model.get_level()
dist_type = last_model.get_distribution_type()
world_prior_fn = last_model.get_world_prior_fn()
utterance_prior_fn = last_model.get_utterance_prior_fn()
alpha = last_model.get_alpha()

best_model = RSA.make(dist_type + "_" + str(level), dist_type, level, \
                      best_meaning_fn, world_prior_fn, utterance_prior_fn, \
                      L_bottom=True, soft_bottom=False, alpha=alpha, \
                      observation_fn=best_observation_fn)

# Output logs
rsa_logger.dump(file_path=log_path + "_rsa")
s_logger.dump(file_path=log_path + "_s")

# Output results
rsa_results_logger = Logger()
rsa_results = Evaluation.run_all(rsa_dev_evals, best_model)
rsa_results["Iteration"] = best_iteration
rsa_results_logger.log(rsa_results)
rsa_results_logger.dump(file_path=results_path + "_rsa")

s_results_logger = Logger()
s_results = Evaluation.run_all(s_dev_evals, best_model.get_meaning_fn().get_seq_model())
s_results["Iteration"] = best_iteration
s_results_logger.log(s_results)
s_results_logger.dump(file_path=results_path + "_s")

# Output test results
if eval_test:
    rsa_test_results_logger = Logger()
    rsa_results = Evaluation.run_all(rsa_test_evals, best_model)
    rsa_results["Iteration"] = best_iteration
    rsa_test_results_logger.log(rsa_results)
    rsa_test_results_logger.dump(file_path=test_results_path + "_rsa")

    s_test_results_logger = Logger()
    s_results = Evaluation.run_all(s_test_evals, best_model.get_meaning_fn().get_seq_model())
    s_results["Iteration"] = best_iteration
    s_test_results_logger.log(s_results)
    s_test_results_logger.dump(file_path=test_results_path + "_s")

# Output model
best_meaning_fn.save(meaning_model_path)
if best_observation_fn is not None:
    best_observation_fn.save(observation_model_path)
best_meaning_fn.get_seq_model().save(s_model_path)
