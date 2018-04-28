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

COLORS_PER_DIM=50 # Number of values per dimensions H x S of color to compute meanings over
COLOR_IMG_WIDTH=140
COLOR_IMG_HEIGHT=140

gpu = True
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

# Sample utterances
s_sample = s.sample(n_per_input=n_samples, max_length=max_sample_length)
utt_sample, utt_lens, _ = s_sample[0]
utt_sample = utt_sample.transpose(0,1) # Gives batch x seq length of utterances
utt_sample_strs = strs_for_scored_samples(s_sample, D_color["utterance"])[0]

# Construct some utterances by hand
reverse_lookup = D_color["utterance"].get_feature_seq_set().get_feature_set().make_token_lookup()
constructed_utts = [["#start#", "green", "-ish", "#end#"], \
                    ["#start#", "blue", "banana", "#end"]]
constructed_indices = torch.zeros((len(constructed_utts), utt_sample.size(1))).long()
constructed_lens = torch.zeros(len(constructed_utts)).long()
for u in range(len(constructed_utts)):
    for i in range(len(constructed_utts[u])):
        if constructed_utts[u][i] not in reverse_lookup:
            raise ValueError(constructed_utts[u][i] + " not found in feature vocabulary.")
        constructed_indices[u,i] = reverse_lookup[constructed_utts[u][i]]
        constructed_lens[u] = len(constructed_utts[u])

utt_sample = torch.cat((utt_sample, constructed_indices))
utt_lens = torch.cat((utt_lens, constructed_lens))
utt_sample_strs.extend(" ".join(constructed_utts[u] for u in range(len(constructed_utts))))

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

# Make image representations of meanings
meaning_imgs = []
for u in range(meanings.size(0)): # For each utterance
    meaning_reshaped = meanings[u].contiguous().view(COLORS_PER_DIM,COLORS_PER_DIM).cpu().numpy()
    meaning_imgs.append(make_gray_img(meaning_reshaped, width=COLOR_IMG_WIDTH,height=COLOR_IMG_HEIGHT))

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
