import argparse
import torch
import numpy as np
import ltprg.data.feature

from mung.util.config import Config
from mung.feature import MultiviewDataSet, Symbol

from ltprg.model.rsa import RSA, load_evaluations_from_config

parser = argparse.ArgumentParser()
parser.add_argument('env', action="store")
parser.add_argument('data', action="store")
parser.add_argument('model', action="store")
parser.add_argument('train_evals', action="store")
parser.add_argument('dev_evals', action="store")
parser.add_argument('test_evals', action="store")
parser.add_argument('--seed', action='store', dest='seed', type=int, default=1)
parser.add_argument('--gpu', action='store', dest='gpu', type=bool, default=True)
parser.add_argument('--eval_test', action='store', dest='eval_test', type=bool, default=False)
args = parser.parse_args()

gpu = args.gpu
seed = args.seed
env = Config.load(args.env)
data_config = Config.load(args.data, environment=env)
model_config = Config.load(args.model, environment=env)
train_evals_config = Config.load(args.train_evals, environment=env)
dev_evals_config = Config.load(args.dev_evals, environment=env)
test_evals_config = Config.load(args.test_evals, environment=env)

if gpu:
    torch.cuda.manual_seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)

_, data_sets = MultiviewDataSet.load_from_config(data_config)
rsa = RSA.load_from_config(model_config, data_sets["train"], gpu=gpu)

train_evals = load_evaluations_from_config(train_evals_config, data_sets, gpu=False)
dev_evals = load_evaluations_from_config(dev_evals_config, data_sets, gpu=False)
test_evals = load_evaluations_from_config(test_evals_config, data_sets, gpu=False)


