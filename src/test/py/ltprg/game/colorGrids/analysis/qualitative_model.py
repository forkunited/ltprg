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
from ltprg.model.rsa import RSA, PriorView
from torch.autograd import Variable
from ltprg.game.color.properties.colorspace_conversions import hsls_to_rgbs, rgbs_to_labs
from ltprg.game.color.eval import ColorMeaningPlot
from ltprg.util.img import make_rgb_img, make_gray_img
from ltprg.game.color.util import construct_color_space

parser = argparse.ArgumentParser()
parser.add_argument('id', action="store")
parser.add_argument('env', action="store")
parser.add_argument('data', action="store")
parser.add_argument('model', action="store")
parser.add_argument('other_model', action="store")
parser.add_argument('subset', action="store")
parser.add_argument('output_dir', action="store")
parser.add_argument('--seed', action='store', dest='seed', type=int, default=1)
parser.add_argument('--gpu', action='store', dest='gpu', type=int, default=1)
parser.add_argument('--grid', action='store', dest='grid', type=bool, default=False)
parser.add_argument('--only_model_diffs', action='store', dest='only_model_diffs', type=bool, default=True)
parser.add_argument('--max_examples', action='store', dest='max_examples', type=int, default=300)
parser.add_argument('--model_name', action='store', dest='model_name', type=str, default="Default")
parser.add_argument('--other_model_name', action='store', dest='other_model_name', type=str, default="Other")

args, extra_env_args = parser.parse_known_args()

# Initalize current run parameters
id = args.id
gpu = bool(args.gpu)
seed = args.seed
output_dir = args.output_dir
output_path = make_indexed_dir(path.join(output_dir, str(id) + "_seed" + str(seed)))

only_model_diffs = args.only_model_diffs
max_examples = args.max_examples

extra_env = Config.load_from_dict({ "output_path" : output_path })
extra_env = Config.load_from_list(extra_env_args).merge(extra_env)

# Load configuration files
env = Config.load(args.env).merge(extra_env)
data_config = Config.load(args.data, environment=env)
model_config = Config.load(args.model, environment=env)
other_model_config = Config.load(args.other_model, environment=env)


def set_seed(s):
    # Random seed
    if gpu:
        torch.cuda.manual_seed(s)
    torch.manual_seed(s)
    np.random.seed(s)

set_seed(seed)

# Setup data, model, and evaluations
_, data_sets = cfeature.load_mvdata(data_config)
data_parameter, rsa_model = crsa.load_rsa_model(model_config, data_sets["train"], gpu=gpu)
other_data_parameter, other_rsa_model = crsa.load_rsa_model(other_model_config, data_sets["train"], gpu=gpu)

D = data_sets[args.subset]

set_seed(seed)

if only_model_diffs:
    batch_size = 128
    dis_ids = set()   
    for i in range(D.get_num_batches(batch_size)):
        batch,indices = D.get_batch(i, batch_size,return_indices=True)
        dist = rsa_model.forward_batch(batch, data_parameter)
        other_dist = other_rsa_model.forward_batch(batch, other_data_parameter)
        _, pred = torch.max(dist.p(), 1)
        _, other_pred = torch.max(other_dist.p(), 1)
        disagreements = pred != other_pred
        for j in range(disagreements.size(0)):
            if disagreements[j].data[0]:
               dis_ids.add(D.get_data().get(indices[j]).get_id())
    D = D.filter(lambda d : d.get_id() in dis_ids) 

if D.get_size() > max_examples:
    D = D.get_subset(0,max_examples)

print "Data size: " + str(D.get_size())

# Run prior evaluations
set_seed(seed)
evaluations = []
priorv = PriorView(args.model_name + "Prior", D, data_parameter, output_path)
evaluations.append(priorv)
Evaluation.run_all(evaluations, rsa_model) # Returns results, but don't need

set_seed(seed)
evaluations = []
priorv = PriorView(args.other_model_name + "Prior", D, other_data_parameter, output_path)
evaluations.append(priorv)
Evaluation.run_all(evaluations, other_rsa_model) # Returns results, but don't need

