import sys
import numpy as np
from mung.feature import FeatureSequenceSet, Symbol

NUM_SYMBOLS = 4

feature_seq_dir= sys.argv[1]
wv_file = sys.argv[2]
output_file = sys.argv[3]
missing_random = bool(int(sys.argv[4]))

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

            if not missing_random:
                wv_size += NUM_SYMBOLS
                token_to_wv[token] = np.concatenate((token_to_wv[token], np.zeros(NUM_SYMBOLS)))

    return token_to_wv, wv_size

token_to_wv, wv_size = load_wv()
feature_seq = FeatureSequenceSet.load(feature_seq_dir)
vocab_size = feature_seq.get_feature_set(0).get_token_count()
wv_mat = np.zeros(shape=(vocab_size, wv_size))

for i in range(vocab_size):
    token = feature_seq.get_feature_token(i).get_value()
    wv = None
    if token in token_to_wv:
        wv = token_to_wv[token]
    elif missing_random:
        wv = np.random.normal(loc=0.0, scale=0.01, size=(wv_size))
    else:
       symbol_idx = Symbol.index(token)        
       if symbol_idx is None:
           symbol_idx = Symbol.index(Symbol.SEQ_UNC)
       wv = np.zeros(wv_size)
       wv[wv_size - NUM_SYMBOLS + symbol_idx] = 1.0
    wv_mat[i] = wv

np.save(output_file, wv_mat)
