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
from mung.util.config import Config
from mung.util.log import Logger
from mung.torch_ext.eval import Evaluation
from ltprg.util.file import make_indexed_dir
from ltprg.model.rsa import RSA

parser = argparse.ArgumentParser()
parser.add_argument('id', action="store")
parser.add_argument('env', action="store")
parser.add_argument('data', action="store")
parser.add_argument('model', action="store")
parser.add_argument('learn', action="store")
parser.add_argument('train_evals', action="store")
parser.add_argument('dev_evals', action="store")
parser.add_argument('test_evals', action="store")
parser.add_argument('output_dir', action="store")
parser.add_argument('--seed', action='store', dest='seed', type=int, default=1)
parser.add_argument('--gpu', action='store', dest='gpu', type=int, default=1)
parser.add_argument('--eval_test', action='store', dest='eval_test', type=int, default=0)
args, extra_env_args = parser.parse_known_args()

# Initalize current run parameters
id = args.id
gpu = bool(args.gpu)
seed = args.seed
eval_test = bool(args.eval_test)
output_dir = args.output_dir

extra_env = Config.load_from_dict({ "arg_output_dir" : output_dir })
extra_env = Config.load_from_list(extra_env_args).merge(extra_env)

# Load configuration files
env = Config.load(args.env).merge(extra_env)
data_config = Config.load(args.data, environment=env)
model_config = Config.load(args.model, environment=env)
learn_config = Config.load(args.learn, environment=env)
train_evals_config = Config.load(args.train_evals, environment=env)
dev_evals_config = Config.load(args.dev_evals, environment=env)
test_evals_config = Config.load(args.test_evals, environment=env)

# Random seed
if gpu:
    torch.cuda.manual_seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)

# Setup data, model, and evaluations
_, data_sets = cfeature.load_mvdata(data_config)
data_parameter, rsa_model = crsa.load_rsa_model(model_config, data_sets["train"], gpu=gpu)
train_evals = crsa.load_evaluations(train_evals_config, data_sets, gpu=gpu)
dev_evals = crsa.load_evaluations(dev_evals_config, data_sets, gpu=gpu)
test_evals = crsa.load_evaluations(test_evals_config, data_sets, gpu=gpu)

# Setup output files
output_path = make_indexed_dir(path.join(output_dir, str(id) + "_seed" + str(seed)))
log_path = path.join(output_path, "log")
results_path = path.join(output_path, "results")
test_results_path = path.join(output_path, "test_results")
meaning_model_path = path.join(output_path, "model_meaning")
observation_model_path = path.join(output_path, "model_observation")
config_output_path = path.join(output_path, "config.json")

logger = Logger()
logger.set_file_path(log_path)

# Run training 
loss_criterion = torch.nn.NLLLoss()
best_part_fn = lambda m : (m.get_meaning_fn(), m.get_observation_fn())
last_model, best_part, best_iteration = clearn.train_from_config(learn_config, \
    data_parameter, loss_criterion, logger, train_evals, rsa_model, data_sets, best_part_fn=best_part_fn)

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
logger.dump(file_path=log_path)

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
full_config["train_evals"] = train_evals_config.get_dict()
full_config["dev_evals"] = dev_evals_config.get_dict()
full_config["test_evals"] = test_evals_config.get_dict()
with open(config_output_path, 'w') as fp:
    json.dump(full_config, fp)

# Output results
results_logger = Logger()
results = Evaluation.run_all(dev_evals, best_model)
results["Iteration"] = best_iteration
results_logger.log(results)
results_logger.dump(file_path=results_path)

# Output test results
if eval_test:
    test_results_logger = Logger()
    results = Evaluation.run_all(test_evals, best_model)
    results["Iteration"] = best_iteration
    test_results_logger.log(results)
    test_results_logger.dump(file_path=test_results_path)

# Output model
best_meaning_fn.save(meaning_model_path)
if best_observation_fn is not None:
    best_observation_fn.save(observation_model_path)





