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
parser.add_argument('subset', action="store")
parser.add_argument('output_dir', action="store")
parser.add_argument('--seed', action='store', dest='seed', type=int, default=1)
parser.add_argument('--gpu', action='store', dest='gpu', type=int, default=1)
parser.add_argument('--grid', action='store', dest='grid', type=bool, default=False)
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

D = data_sets[args.subset]

# Run prior evaluations
evaluations = []
priorv = PriorView("Prior", D, data_parameter, output_path)
evaluations.append(priorv)
Evaluation.run_all(evaluations, rsa_model) # Returns results, but don't need

if args.grid:
    exit()

# Compute color meaning function visualizations
COLORS_PER_DIM=50 # Number of values per dimensions H x S of color to compute meanings over
COLOR_IMG_WIDTH=140
COLOR_IMG_HEIGHT=140

# Make color space over which to compute meanings
# This consists of colors with varying H and S dimensions of HSL
colors = construct_color_space(n_per_dim=COLORS_PER_DIM)
world_idx = torch.arange(0, colors.size(0)).long().unsqueeze(0)
if gpu:
    world_idx = world_idx.cuda()
    colors = colors.cuda()

meaning_fn = rsa_model.get_meaning_fn()

# Compute meanings over color space
# Gives tensor of dim utterance count x color count
meanings = torch.zeros(D.get_size(), world_idx.size(1))
for i in range(D.get_size()/10):
    batch = D.get_batch(i, 10)
    seq,seq_len,_ = batch["utterance"]
    if gpu:
        seq = seq.cuda()
    meanings_i = meaning_fn((Variable(seq.transpose(0,1).long().unsqueeze(0)), seq_len.unsqueeze(0)), \
                    Variable(world_idx), \
                    Variable(colors.view(1,colors.size(0)*colors.size(1)))).squeeze().data
    meanings[i*10:((i+1)*10)] = meanings_i

def make_img(mng):
    # mng = normalize(mng, dim=0) # meaning is dim num_colors
    meaning_reshaped = mng.contiguous().view(COLORS_PER_DIM,COLORS_PER_DIM).cpu().numpy()
    return make_gray_img(meaning_reshaped, width=COLOR_IMG_WIDTH,height=COLOR_IMG_HEIGHT)

# Make image representations of meanings
meaning_imgs = [make_img(meanings[u]) for u in range(meanings.size(0))]

# Save meaning imgs
for u, img in enumerate(meaning_imgs):
    img.save(output_path + "/" + D.get_data().get(u).get("id") + ".png")
