# import unittest
import sys
import numpy as np
from os.path import join
from mung.data import DataSet, Datum, Partition
from mung.feature import FeatureSet, DataFeatureMatrix
from feature import FeatureCielabEmbeddingType #, FeatureVisualEmbeddingType

def featurize_embeddings(input_data_dir, output_feature_dir, partition_file,
                         partition_fn, feature_name, paths, embedding_type,
                         init_data="train", include_positions=False, position_count=1, row_count=1, standardize=False):
	assert embedding_type in ['fc-6', 'cielab']
	partition = Partition.load(partition_file)
	data_full = DataSet.load(input_data_dir)
	data_parts = data_full.partition(partition, partition_fn)
	#if embedding_type == 'fc-6':
	#	feat = FeatureVisualEmbeddingType(feature_name, paths)
	#elif 
        if embedding_type == 'cielab':
		feat = FeatureCielabEmbeddingType(feature_name, paths, \
			include_positions=include_positions, position_count=position_count, row_count=row_count, standardize=standardize)
	feat_set = FeatureSet(feature_types=[feat])
	feat_set.init(data_parts[init_data])
	mat = DataFeatureMatrix(data_full, feat_set, init_features=False)
	mat.save(output_feature_dir)
