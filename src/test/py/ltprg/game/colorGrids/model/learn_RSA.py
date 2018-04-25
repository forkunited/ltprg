import argparse
import torch
import numpy as np
import ltprg.data.feature

from mung.util.config import Config
from mung.feature import MultiviewDataSet, Symbol

parser = argparse.ArgumentParser()
parser.add_argument('env', action="store")
parser.add_argument('data', action="store")
parser.add_argument('--seed', action='store', dest='seed', type=int, default=1)
parser.add_argument('--gpu', action='store', dest='gpu', type=bool, default=True)
args = parser.parse_args()

gpu = args.gpu
seed = args.seed
env = Config.load(args.env)
data_config = Config.load(args.data, environment=env)

if gpu:
    torch.cuda.manual_seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)

_, S = MultiviewDataSet.load_from_config(data_config)

print "Loaded data..."
for key in S.keys():
    print key + ": " + str(S[key].get_size())
