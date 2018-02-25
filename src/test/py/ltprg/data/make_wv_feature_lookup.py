import sys
import numpy as np
from mung.feature import FeatureSequenceSet

feature_seq_dir= sys.argv[1]
wv_file = sys.argv[2]
output_file = sys.argv[3]

def load_wv():
    token_to_wv = dict()
    wv_size = 0
    with open(wv_file, 'r') as fp:
        for line in fp:
            line_parts = line.split(' ')
            token = line_parts[0]
            wv = np.array([float(line_parts[i]) for i in range(1, len(line_parts))])
            token_to_wv[token] = wv
            wv_size = wv.shape[0]
    return token_to_wv, wv_size

token_to_wv, wv_size = load_wv()
feature_seq = FeatureSequenceSet.load(feature_seq_dir)
vocab_size = feature_seq.get_feature_set_size()
wv_mat = np.zeros(shape=(vocab_size, wv_size))

for i in range(vocab_size):
    token = feature_seq.get_feature_token(i).get_value()
    wv = None
    if token in token_to_wv:
        wv = token_to_wv[token]
    else:
        wv = np.random.normal(loc=0.0, scale=1.0, size=(wv_size))
    wv_mat[i] = wv

np.save(output_file, wv_mat)
