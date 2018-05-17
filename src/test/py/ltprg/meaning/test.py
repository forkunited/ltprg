import sys
import numpy as np
import json
import torch
from os.path import join
from torch.autograd import Variable
from mung.data import DataSet
from mung.feature import MultiviewDataSet
from ltprg.model.meaning import MeaningModel
from ltprg.model.seq import SequenceModel, strs_for_scored_samples
from ltprg.game.color.properties.colorspace_conversions import hsls_to_rgbs, rgbs_to_labs
from ltprg.game.color.eval import ColorMeaningPlot
from ltprg.util.img import make_rgb_img, make_gray_img
from ltprg.game.color.util import construct_color_space
from util import *

COLORS_PER_DIM=50 # Number of values per dimensions H x S of color to compute meanings over
COLOR_IMG_WIDTH=140
COLOR_IMG_HEIGHT=140

gpu = False
seed = 1
data_color_dir = sys.argv[1]
data_utterance_dir = sys.argv[2]
model_s_path = sys.argv[3]
model_meaning_path = sys.argv[4]
output_path = sys.argv[5]
n_samples = int(sys.argv[6])
max_sample_length = int(sys.argv[7])

if gpu:
    torch.cuda.manual_seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)

# Load data and models
D_color = MultiviewDataSet.load(data_color_dir, dfmatseq_paths={ "utterance" : data_utterance_dir })
s = SequenceModel.load(model_s_path) # Language model for sampling utterances
meaning_fn = MeaningModel.load(model_meaning_path) # Meaning function
if not gpu:
    s = s.cpu()
    meaning_fn.cpu()

# n_samples = 10
if n_samples:
    # Sample utterances
    s_sample = s.sample(n_per_input=n_samples, max_length=max_sample_length)
    utt_sample, utt_lens, _ = s_sample[0]
    utt_sample = utt_sample.transpose(0,1) # Gives batch x seq length of utterances
    utt_sample_strs = strs_for_scored_samples(s_sample, D_color["utterance"])[0]
else:
    # Construct some utterances by hand
    reverse_lookup = D_color["utterance"].get_feature_seq_set().get_feature_set().make_token_lookup()
    utt_tokens = [["green", "-ish"], \
                    ["dark", "red"], \
                    ["blue", "yellow"]] # Banana not in vocabulary
    constructed_utts, compose_idx = make_constructed_utts(utt_tokens)
    print constructed_utts, compose_idx
    utt_sample = torch.zeros((len(constructed_utts), max_sample_length)).long()
    utt_lens = torch.zeros(len(constructed_utts)).long()
    for u in range(len(constructed_utts)):
        for i in range(len(constructed_utts[u])):
            if constructed_utts[u][i] not in reverse_lookup:
                raise ValueError(constructed_utts[u][i] + " not found in feature vocabulary.")
            utt_sample[u,i] = reverse_lookup[constructed_utts[u][i]]
            utt_lens[u] = len(constructed_utts[u])
    utt_sample_strs = [" ".join(utt) for utt in constructed_utts]
    utt_sample_strs += ["composition: " + " + ".join(utt_toks) for utt_toks in utt_tokens]
    utt_sample_strs += ["subtraction: " + " - ".join([" ".join(utt_toks), tok]) for utt_toks in utt_tokens for tok in utt_toks]

# utt_sample = torch.cat((utt_sample, constructed_indices))
# utt_lens = torch.cat((utt_lens, constructed_lens))
# utt_sample_strs.extend([" ".join(constructed_utts[u]) for u in range(len(constructed_utts))])

# Make color space over which to compute meanings
# This consists of colors with varying H and S dimensions of HSL
colors = construct_color_space(n_per_dim=COLORS_PER_DIM)
world_idx = torch.arange(0, colors.size(0)).long().unsqueeze(0)
if gpu:
    world_idx = world_idx.cuda()
    colors = colors.cuda()
    utt_sample = utt_sample.cuda()
    meaning_fn = meaning_fn.cuda()

# Compute meanings over color space
# Gives tensor of dim utterance count x color count
meanings = meaning_fn((Variable(utt_sample.unsqueeze(0)), utt_lens.unsqueeze(0)), \
                    Variable(world_idx), \
                    Variable(colors.view(1,colors.size(0)*colors.size(1)))).squeeze().data

meaning_ops = []
for idx in compose_idx:
    meaning_ops.append(meaning_pointwise_product(meanings[idx[0]:idx[1]]).unsqueeze(0))

for idx in compose_idx:
    meaning_ops.extend(meaning_pointwise_difference(
            meanings[idx[1]], 
            meanings[idx[0]:idx[1]]
        ))

def make_img(mng):
    # mng = normalize(mng, dim=0) # meaning is dim num_colors
    meaning_reshaped = mng.contiguous().view(COLORS_PER_DIM,COLORS_PER_DIM).cpu().numpy()
    return make_gray_img(meaning_reshaped, width=COLOR_IMG_WIDTH,height=COLOR_IMG_HEIGHT)

# Make image representations of meanings
meaning_imgs = [make_img(meanings[u]) for u in range(meanings.size(0))] + [make_img(m) for m in meaning_ops]

# Save meaning imgs
for u, img in enumerate(meaning_imgs):
    img.save(output_path + "_" + str(u) + ".png")

# Save utterances
full_str = ""
for u,utt_str in enumerate(utt_sample_strs):
    full_str += "(" + str(u) + ") " + utt_str + "\n"
with open(output_path + "_str.txt", "w") as text_file:
    text_file.write(full_str)

# Make and save image representation of color space
colors_rgb = construct_color_space(n_per_dim=COLORS_PER_DIM, rgb=True)
color_img = make_rgb_img(colors_rgb.view(COLORS_PER_DIM, COLORS_PER_DIM, 3).numpy(), width=COLOR_IMG_WIDTH,height=COLOR_IMG_HEIGHT)
color_img.save(output_path + "_colors.png")
