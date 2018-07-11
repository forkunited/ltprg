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

COLORS_PER_DIM=50 # Number of values per dimensions H x S of color to compute meanings over
COLOR_IMG_WIDTH=140
COLOR_IMG_HEIGHT=140

parser = argparse.ArgumentParser()
parser.add_argument('id', action="store")
parser.add_argument('env', action="store")
parser.add_argument('data', action="store")
parser.add_argument('model', action="store")
parser.add_argument('other_model', action="store")
parser.add_argument('output_dir', action="store")
parser.add_argument('--seed', action='store', dest='seed', type=int, default=1)
parser.add_argument('--gpu', action='store', dest='gpu', type=int, default=1)
parser.add_argument('--model_name', action='store', dest='model_name', type=str, default="Default")
parser.add_argument('--other_model_name', action='store', dest='other_model_name', type=str, default="Other")

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

D = data_sets["train"]
# Construct some utterances by hand
reverse_lookup = D["utterance"].get_feature_seq_set().get_feature_set().make_token_lookup()
max_sample_length = 10
utt_tokens = [["#start#", "blue", "#end#"], \
              ["#start#", "purple", "#end#"], \
              ["#start#", "green", "#end#"], \
              ["#start#", "orange", "#end#"], \
              ["#start#", "yellow", "#end#"], \
              ["#start#", "red", "#end#"], \
              ["#start#", "gray", "#end#"], \
              ["#start#", "bright", "green", "#end#"], \
              ["#start#", "dark", "green", "#end#"], \
              ["#start#", "dark", "-er", "green", "#end#"], \
              ["#start#", "bright", "-est", "#end#"], \
              ["#start#", "orange", "-ish", "red", "#end#"], \
              ["#start#", "not", "blue", "#end#"], \
              ["#start#", "not", "red", "#end#"], \
              ["#start#", "green", "-ish", "#end#"], \
              ["#start#", "red", "-ish", "#end#"], \
              ["#start#", "purple", "-ish", "#end#"], \
              ["#start#", "yellow", "-ish", "#end#"], \
              ["#start#", "dark", "red", "#end#"], \
              ["#start#", "neon", "#end#"], \
              ["#start#", "neon", "green", "#end#"], \
              ["#start#", "neon", "blue", "#end#"], \
              ["#start#", "neon", "red", "#end#"], \
              ["#start#", "blue", "yellow", "#end#"], \
              ["#start#", "green", "like", "grass", "less", "bright", "#end#"],
              ["#start#", "purple", "blue", "that", "is", "dark", "-er", "#end#"],
              ["#start#", "light", "-est", "blue", "less", "gray", "#end#"],
              ["#start#", "the", "blu", "-est", "of", "the", "purples", "#end#"]]
utt_sample = torch.zeros((len(utt_tokens), max_sample_length)).long()
utt_lens = torch.zeros(len(utt_tokens)).long()
for u in range(len(utt_tokens)):
    for i in range(len(utt_tokens[u])):
        if utt_tokens[u][i] not in reverse_lookup:
            raise ValueError(utt_tokens[u][i] + " not found in feature vocabulary.")
        utt_sample[u,i] = reverse_lookup[utt_tokens[u][i]]
        utt_lens[u] = len(utt_tokens[u])
utt_sample_strs = ["_".join(utt[1:(len(utt)-1)]) for utt in utt_tokens]

set_seed(seed)

# Make color space over which to compute meanings
# This consists of colors with varying H and S dimensions of HSL
colors = construct_color_space(n_per_dim=COLORS_PER_DIM, standardized=True)
world_idx = torch.arange(0, colors.size(0)).long().unsqueeze(0)
if gpu:
    world_idx = world_idx.cuda()
    colors = colors.cuda()
    utt_sample = utt_sample.cuda()

meaning_fn = rsa_model.get_meaning_fn()
other_meaning_fn = other_rsa_model.get_meaning_fn()

# Compute meanings over color space
# Gives tensor of dim utterance count x color count
meanings = meaning_fn((Variable(utt_sample.unsqueeze(0)), utt_lens.unsqueeze(0)), \
                    Variable(world_idx), \
                    Variable(colors.view(1,colors.size(0)*colors.size(1)))).squeeze().data

other_meanings = other_meaning_fn((Variable(utt_sample.unsqueeze(0)), utt_lens.unsqueeze(0)), \
                    Variable(world_idx), \
                    Variable(colors.view(1,colors.size(0)*colors.size(1)))).squeeze().data

def make_img(mng):
    meaning_reshaped = mng.contiguous().view(COLORS_PER_DIM,COLORS_PER_DIM).cpu().numpy()
    
    meaning_reshaped_lin = np.log(meaning_reshaped/(1.0-meaning_reshaped))
    max_val = np.max(meaning_reshaped_lin)
    min_val = np.min(meaning_reshaped_lin)
    meaning_reshaped_lin = (meaning_reshaped_lin - min_val)/(max_val - min_val)
    
    meaning_img = make_gray_img(meaning_reshaped, width=COLOR_IMG_WIDTH,height=COLOR_IMG_HEIGHT)
    meaning_lin_img = make_gray_img(meaning_reshaped_lin, width=COLOR_IMG_WIDTH,height=COLOR_IMG_HEIGHT)

    return meaning_img, meaning_lin_img

# Make image representations of meanings
meaning_imgs = [make_img(meanings[u]) for u in range(meanings.size(0))]
other_meaning_imgs = [make_img(other_meanings[u]) for u in range(other_meanings.size(0))]

# Save meaning imgs
for u, img in enumerate(meaning_imgs):
    img[0].save(output_path + "/" + args.model_name + "_" + utt_sample_strs[u] + ".png")
    img[1].save(output_path + "/" + args.model_name + "_" + utt_sample_strs[u] + "_linearized.png")
for u, img in enumerate(other_meaning_imgs):
    img[0].save(output_path + "/" + args.other_model_name + "_" + utt_sample_strs[u] + ".png")
    img[1].save(output_path + "/" + args.other_model_name + "_" + utt_sample_strs[u] + "_linearized.png")

