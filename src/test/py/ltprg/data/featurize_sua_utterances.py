import numpy as np
import sys
from mung.data import DataSet, Datum
from mung.feature import FeaturePathSequence, FeatureSequenceSet, DataFeatureMatrixSequence

MAX_UTTERANCE_LENGTH = 15

input_data_dir = sys.argv[1]
output_feature_dir = sys.argv[2]

np.random.seed(1)

data = DataSet.load(input_data_dir)
feat_seq = FeaturePathSequence("utterance_lemmas", ["utterances[*].lemmas.lemmas"], MAX_UTTERANCE_LENGTH, min_occur=2)
feat_seq_set = FeatureSequenceSet(feature_seqs=[feat_seq])
mat = DataFeatureMatrixSequence(data, feat_seq_set)
mat.save(output_feature_dir)

