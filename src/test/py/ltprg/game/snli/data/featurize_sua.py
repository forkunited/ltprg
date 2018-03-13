import unittest
import numpy as np
import sys
import mung.feature_helpers
import ltprg.data.feature_helpers
from os.path import join

input_data_dir = sys.argv[1]
output_feature_dir = sys.argv[2]
partition_file = sys.argv[3]

# Necessary to allow unittest.main() to work
del sys.argv[3]
del sys.argv[2]
del sys.argv[1]

np.random.seed(1)

class FeaturizeSUA(unittest.TestCase):

    # Constructs sequences of integer indices representing utterances
    def test_utterance_indices(self):
        print "Featurizing utterance indices"
        mung.feature_helpers.featurize_path_enum_seqs(
            input_data_dir, # Source data set
            join(output_feature_dir, "utterance_idx"), # Output directory
            partition_file, # Data partition
            lambda d : d.get("id"), # Function that partitions the data
            "utterance_idx", # Name of the feature
            ["utterance.nlp.token_strs.strs"], # JSON path into data examples
            30, # Maximum utterance length
            token_fn=lambda x : x, # Function applied to tokens to construct the vocabulary
            indices=True) # Indicates that indices will be computed instead of one-hot vectors

    # Constructs sequences of integer indices representing utterances
    def test_premise_indices(self):
        print "Featurizing premise indices"
        mung.feature_helpers.featurize_path_enum_seqs(
            input_data_dir, # Source data set
            join(output_feature_dir, "premise_idx"), # Output directory
            partition_file, # Data partition
            lambda d : d.get("id"), # Function that partitions the data
            "premise_idx", # Name of the feature
            ["state.nlp.token_strs.strs"], # JSON path into data examples
            30, # Maximum utterance length
            token_fn=lambda x : x, # Function applied to tokens to construct the vocabulary
            indices=True) # Indicates that indices will be computed instead of one-hot vectors

    # Constructs an index of the target color index (0, 1, or 2) that
    # the speaker observed
    def test_speaker_target_indices(self):
        print "Featurizing speaker target indices"
        mung.feature_helpers.featurize_path_scalars(
            input_data_dir,
            join(output_feature_dir, "speaker_target_idx"),
            partition_file,
            lambda d : d.get("id"),
            "speaker_target_idx",
            ["state.sTarget"])

if __name__ == '__main__':
    unittest.main()
