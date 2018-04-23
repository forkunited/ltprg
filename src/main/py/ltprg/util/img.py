import numpy as np
from PIL import Image

# From https://stackoverflow.com/questions/44297679/converting-an-array-of-floating-point-pixels-in-0-1-to-grayscale-image-in-pyth
def make_gray_img(arr, width=140, height=140):
    pixels = 255 * (1.0 - arr)
    img = Image.fromarray(pixels.astype(np.uint8), mode='L')
    img = img.resize((width, height))
    return img.convert('RGB')

def make_rgb_img(arr, width=140, height=140):
    arr = 255 * arr
    img = Image.fromarray(arr.astype(np.uint8), mode='RGB')
    img = img.resize((width, height))
    return img
