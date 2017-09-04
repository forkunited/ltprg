from __future__ import division
import numpy as np
import colorsys
import skimage
from skimage.color import rgb2lab, rgb2luv

def hsls_to_rgbs(hsls):
	# hsls is list of [H, S, L]
	return [colorsys.hls_to_rgb(
			color[0]/360, color[2]/100, color[1]/100) 
			for color in hsls] 

def rgbs_to_labs(rgbs):
	# rgbs is list of [R, G, B]
	return [rgb2lab([[[color]]])[0][0][0] for color in rgbs]


def rgbs_to_luvs(rgbs):
	# rgbs is list of [R, G, B]
	return [rgb2luv([[[color]]])[0][0][0] for color in rgbs]

def fourier_transform(hsl):
    # Convert to hsv using https://gist.github.com/xpansive/1337890
    h = float(hsl[0]) / 360.0
    s = float(hsl[1]) / 100.0
    l = float(hsl[2]) / 100.0

    if l < .5:
        s = s * l
    else:
        s = s * (1.0 - l)

    s = 2.0*s/(l+s)
    v = l + s

    vec = []
    for i in range(3):
        for j in range(3):
            for k in range(3):
                c = j*h + k*s + l*v
                re = np.cos(-2.0*np.pi*c)
                im = np.sin(-2.0*np.pi*c)
                vec.append(re)
                vec.append(im)
    return vec

def color_paper_space(hsls):
	return [fourier_transform(color) for color in hsls]