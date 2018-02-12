import unittest
import numpy as np
import sys
import mung.feature_helpers
import ltprg.data.feature_helpers
from os.path import join

input_data_dir = sys.argv[1]
output_feature_dir = sys.argv[2]
partition_file = sys.argv[3]
grid_dim = int(sys.argv[4])

# Necessary to allow unittest.main() to work
del sys.argv[4]
del sys.argv[3]
del sys.argv[2]
del sys.argv[1]

np.random.seed(1)

class FeaturizeSUA(unittest.TestCase):

    # Constructs sequences of integer indices representing cleaned utterances
    def test_utterance_clean_indices(self):
        print "Featurizing utterance clean indices"
        mung.feature_helpers.featurize_path_enum_seqs(
            input_data_dir, # Source data set
            join(output_feature_dir, "utterance_clean_idx"), # Output directory
            partition_file, # Data partition
            lambda d : d.get("gameid"), # Function that partitions the data
            "utterance_clean_idx", # Name of the feature
            ["utterances[*].nlp.clean_strs.strs"], # JSON path into data examples
            30, # Maximum utterance length
            token_fn=lambda x : x, # Function applied to tokens to construct the vocabulary
            indices=True) # Indicates that indices will be computed instead of one-hot vectors

    # Constructs sequences of integer indices representing last cleaned utterances
    #def test_utterance_clean_last_indices(self):
    #    print "Featurizing utterance clean last indices"
    #    mung.feature_helpers.featurize_path_enum_seqs(
    #        input_last_data_dir, # Source data set
    #        join(output_feature_dir, "utterance_clean_last_idx"), # Output directory
    #        partition_file, # Data partition
    #        lambda d : d.get("gameid"), # Function that partitions the data
    #        "utterance_clean_last_idx", # Name of the feature
    #        ["utterances[*].nlp.clean_strs.strs"], # JSON path into data examples
    #        24, # Maximum utterance length
    #        token_fn=lambda x : x, # Function applied to tokens to construct the vocabulary
    #        indices=True) # Indicates that indices will be computed instead of one-hot vectors

    # Constructs indicators of which color (0, 1, or 2) that the listener clicked
    #def test_listener_clicked(self):
    #    print "Featurizing listener clicks"
    #    mung.feature_helpers.featurize_path_scalars(
    #        input_data_dir, # Source data set
    #        join(output_feature_dir, "listener_clicked"), # Output directory
    #        partition_file, # Data partition
    #        lambda d : d.get("gameid"), # Function that partitions the data
    #        "listener_clicked", # Name of the feature
    #        ["action.lClicked_0", "action.lClicked_1", "action.lClicked_2"]) # JSON paths to feature values

    # Constructs indices of which color (0, 1, or 2) that the listener clicked
    #def test_listener_clicked_indices(self):
    #    print "Featurizing listener click indices"
    #    mung.feature_helpers.featurize_path_scalars(
    #        input_data_dir, # Source data set
    #        join(output_feature_dir, "listener_clicked_idx"), # Output directory
    #        partition_file, # Data partition
    #        lambda d : d.get("gameid"), # Function that partitions the data
    #        "listener_clicked_idx", # Name of the feature
    #        ["action.lClickedIndex"]) # JSON paths to feature values

    # Constructs vectors of the colors that the speaker observed
    def test_speaker_colors(self):
        print "Featurizing speaker colors"
        
        dims = []
        for i in range(grid_dim*grid_dim):
            dims.append("state.sObj0_Shp" + str(i) + "_ClrH")
            dims.append("state.sObj0_Shp" + str(i) + "_ClrS")
        for i in range(grid_dim*grid_dim):
            dims.append("state.sObj1_Shp" + str(i) + "_ClrH")
            dims.append("state.sObj1_Shp" + str(i) + "_ClrS")

        mung.feature_helpers.featurize_path_scalars(
            input_data_dir,
            join(output_feature_dir, "speaker_colors"),
            partition_file,
            lambda d : d.get("gameid"),
            "speaker_colors",
            dims)

    def test_speaker_colors_cielab(self):
        print "Featurizing speaker colors (cielab)"
        
        dims = []
        for i in range(grid_dim*grid_dim):
            dims.append(["state.sObj0_Shp" + str(i) + "_ClrH","state.sObj0_Shp" + str(i) + "_ClrS","state.sObj0_Shp" + str(i) + "_ClrL"])
        for i in range(grid_dim*grid_dim):
            dims.append(["state.sObj1_Shp" + str(i) + "_ClrH","state.sObj0_Shp" + str(i) + "_ClrS","state.sObj0_Shp" + str(i) + "_ClrL"])

        ltprg.data.feature_helpers.featurize_embeddings(
            input_data_dir,
            join(output_feature_dir, "speaker_colors_cielab"),
            partition_file,
            lambda d : d.get("gameid"),
            # "context_fc_embedding",
            "speaker_colors_cielab",
            dims,
            # "fc-6",
            "cielab")

    # Constructs an index of the target color index (0, 1, or 2) that
    # the speaker observed
    def test_speaker_target_indices(self):
        print "Featurizing speaker target indices"
        mung.feature_helpers.featurize_path_scalars(
            input_data_dir,
            join(output_feature_dir, "speaker_target_idx"),
            partition_file,
            lambda d : d.get("gameid"),
            "speaker_target_idx",
            ["state.sTarget"])

if __name__ == '__main__':
    unittest.main()
