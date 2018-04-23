import torch
import colorsys
import numpy as np
import math
from torch.autograd import Variable
from mung.torch_ext.eval import Evaluation
from ltprg.model.seq import SamplingMode
from skimage.color import rgb2lab, lab2rgb
from PIL import Image, ImageDraw, ImageFont

HSL_H_MAX = 360.0
HSL_S_MAX = 100.0
HSL_L_MAX = 100.0
HSL_L_CONSTANT = 50.0

# H: [0-360]
# S: [0-100]
def construct_color_space(n_per_dim=50, rgb=False):
    colors = torch.zeros(n_per_dim*n_per_dim, 3)
    color_idx = 0
    for h_i in range(n_per_dim):
        h = h_i*(HSL_H_MAX / n_per_dim)
        for s_i in range(n_per_dim):
            s = s_i*(HSL_S_MAX / n_per_dim)
            l = HSL_L_CONSTANT
            rgb_values = colorsys.hls_to_rgb(h/HSL_H_MAX,l/HSL_L_MAX,s/HSL_S_MAX)
            if rgb:
                colors[color_idx,0] = rgb_values[0]
                colors[color_idx,1] = rgb_values[1]
                colors[color_idx,2] = rgb_values[2]
            else:
                cielab = rgb2lab([[[rgb_values]]])[0][0][0]
                colors[color_idx,0] = cielab[0]
                colors[color_idx,1] = cielab[1]
                colors[color_idx,2] = cielab[2]
            color_idx += 1
    return colors
