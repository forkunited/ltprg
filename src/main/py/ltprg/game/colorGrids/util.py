import torch
import colorsys
import numpy as np
import math
from ltprg.game.color.util import construct_color_space

def construct_grid_space(n_per_dim=50, standardized=False):
    colors = construct_color_space(n_per_dim=n_per_dim, rgb=False, standardized=standardized)
    color_pos = torch.zeros(n_per_dim*n_per_dim*9, 12)
    num_colors = n_per_dim*n_per_dim

    for pos in range(9):
        color_pos[pos*num_colors:(pos+1)*num_colors,0:3] = colors
        color_pos[pos*num_colors:(pos+1)*num_colors,3+pos] = 1.0

    return color_pos 

