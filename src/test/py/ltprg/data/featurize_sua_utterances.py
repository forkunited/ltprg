import numpy as np
import sys
from mung.data import DataSet, Datum, Partition
from mung.feature import FeaturePathSequence, FeatureSequenceSet, DataFeatureMatrixSequence

MAX_UTTERANCE_LENGTH = 15

input_data_dir = sys.argv[1]
output_feature_dir = sys.argv[2]
partition_file = sys.argv[3]

np.random.seed(1)

partition = Partition.load(partition_file)
data_full = DataSet.load(input_data_dir)
data_parts = data_full.partition(partition, lambda d : d.get("gameid"))

feat_seq = FeaturePathSequence("utterance_lemmas", ["utterances[*].nlp.lemmas.lemmas"], MAX_UTTERANCE_LENGTH, min_occur=2, token_fn=lambda x : x.lower())
feat_seq_set = FeatureSequenceSet(feature_seqs=[feat_seq])
feat_seq_set.init(data_parts["train"])

mat = DataFeatureMatrixSequence(data_full, feat_seq_set, init_features=False)
mat.save(output_feature_dir)

