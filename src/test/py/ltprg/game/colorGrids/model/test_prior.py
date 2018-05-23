import os.path as path
import datetime
import argparse
import torch
import json
import numpy as np
import time
import ltprg.config.rsa as crsa
import mung.config.feature as cfeature
from mung.util.config import Config
from ltprg.util.file import make_indexed_dir
from ltprg.model.rsa import PriorView

parser = argparse.ArgumentParser()
parser.add_argument('id', action="store")
parser.add_argument('env', action="store")
parser.add_argument('data', action="store")
parser.add_argument('model', action="store")
parser.add_argument('output_dir', action="store")
parser.add_argument('--seed', action='store', dest='seed', type=int, default=1)
parser.add_argument('--gpu', action='store', dest='gpu', type=int, default=1)
args, extra_env_args = parser.parse_known_args()

# Initalize current run parameters
id = args.id
gpu = bool(args.gpu)
seed = args.seed
output_dir = args.output_dir
output_path = make_indexed_dir(path.join(output_dir, str(id) + "_seed" + str(seed)))

extra_env = Config.load_from_dict({ "output_path" : output_path })
extra_env = Config.load_from_list(extra_env_args).merge(extra_env)

# Load configuration files
env = Config.load(args.env).merge(extra_env)
data_config = Config.load(args.data, environment=env)
model_config = Config.load(args.model, environment=env)

# Random seed
if gpu:
    torch.cuda.manual_seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)

# Setup data, model, and evaluations
_, data_sets = cfeature.load_mvdata(data_config)

data_parameter, rsa_model = crsa.load_rsa_model(model_config, data_sets["train"], gpu=gpu)
priorv = PriorView("Prior", data_sets["train"].get_random_subset(0,30), data_parameter, output_path)
start_time = time.time()
result = priorv.run(rsa_model)
duration = time.time() - start_time

print "Entropy: " + str(result) + " Time: " + str(duration) + " seconds"
