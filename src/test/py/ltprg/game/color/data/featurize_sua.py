import unittest
import numpy as np
import sys
import mung.feature_helpers
import ltprg.data.feature_helpers
from os.path import join

input_data_dir = sys.argv[1]
input_last_data_dir = sys.argv[2]
output_feature_dir = sys.argv[3]
partition_file = sys.argv[4]

# Necessary to allow unittest.main() to work
del sys.argv[4]
del sys.argv[3]
del sys.argv[2]
del sys.argv[1]

np.random.seed(1)

class FeaturizeSUA(unittest.TestCase):

    # Constructs sequences of one-hot vectors representing lemmatized utterances
    def test_utterance_lemmas(self):
        print "Featurizing utterances"
        mung.feature_helpers.featurize_path_enum_seqs(
            input_data_dir, # Source data set
            join(output_feature_dir, "utterance_lemmas"), # Output directory
            partition_file, # Data partition
            lambda d : d.get("gameid"), # Function that partitions the data
            "utterance_lemmas", # Name of the feature
            ["utterances[*].nlp.lemmas.lemmas"], # JSON path into data examples
            15, # Maximum utterance length
            token_fn=lambda x : x.lower()) # Function applied to tokens to construct the vocabulary

    # Constructs sequences of integer indices representing lemmatized utterances
    def test_utterance_lemma_indices(self):
        print "Featurizing utterance indices"
        mung.feature_helpers.featurize_path_enum_seqs(
            input_data_dir, # Source data set
            join(output_feature_dir, "utterance_lemma_idx"), # Output directory
            partition_file, # Data partition
            lambda d : d.get("gameid"), # Function that partitions the data
            "utterance_lemmas_idx", # Name of the feature
            ["utterances[*].nlp.lemmas.lemmas"], # JSON path into data examples
            15, # Maximum utterance length
            token_fn=lambda x : x.lower(), # Function applied to tokens to construct the vocabulary
            indices=True) # Indicates that indices will be computed instead of one-hot vectors

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

    # Constructs sequences of integer indices representing short cleaned utterances
    def test_utterance_clean_full_indices(self):
        print "Featurizing utterance clean full indices"
        mung.feature_helpers.featurize_path_enum_seqs(
            input_data_dir, # Source data set
            join(output_feature_dir, "utterance_clean_full_idx"), # Output directory
            partition_file, # Data partition
            lambda d : d.get("gameid"), # Function that partitions the data
            "utterance_clean_full_idx", # Name of the feature
            ["utterances[*].nlp.clean_strs.strs"], # JSON path into data examples
            72, # Maximum utterance length
            token_fn=lambda x : x, # Function applied to tokens to construct the vocabulary
            indices=True) # Indicates that indices will be computed instead of one-hot vectors

    # Constructs sequences of integer indices representing short cleaned utterances
    def test_utterance_clean_long_indices(self):
        print "Featurizing utterance clean long indices"
        mung.feature_helpers.featurize_path_enum_seqs(
            input_data_dir, # Source data set
            join(output_feature_dir, "utterance_clean_long_idx"), # Output directory
            partition_file, # Data partition
            lambda d : d.get("gameid"), # Function that partitions the data
            "utterance_clean_long_idx", # Name of the feature
            ["utterances[*].nlp.clean_strs.strs"], # JSON path into data examples
            50, # Maximum utterance length
            token_fn=lambda x : x, # Function applied to tokens to construct the vocabulary
            indices=True) # Indicates that indices will be computed instead of one-hot vectors

    # Constructs sequences of integer indices representing short cleaned utterances
    def test_utterance_clean_short_indices(self):
        print "Featurizing utterance clean short indices"
        mung.feature_helpers.featurize_path_enum_seqs(
            input_data_dir, # Source data set
            join(output_feature_dir, "utterance_clean_short_idx"), # Output directory
            partition_file, # Data partition
            lambda d : d.get("gameid"), # Function that partitions the data
            "utterance_clean_short_idx", # Name of the feature
            ["utterances[*].nlp.clean_strs.strs"], # JSON path into data examples
            15, # Maximum utterance length
            token_fn=lambda x : x, # Function applied to tokens to construct the vocabulary
            indices=True) # Indicates that indices will be computed instead of one-hot vectors

    # Constructs sequences of integer indices representing last cleaned utterances
    def test_utterance_clean_last_indices(self):
        print "Featurizing utterance clean last indices"
        mung.feature_helpers.featurize_path_enum_seqs(
            input_last_data_dir, # Source data set
            join(output_feature_dir, "utterance_clean_last_idx"), # Output directory
            partition_file, # Data partition
            lambda d : d.get("gameid"), # Function that partitions the data
            "utterance_clean_last_idx", # Name of the feature
            ["utterances[*].nlp.clean_strs.strs"], # JSON path into data examples
            24, # Maximum utterance length
            token_fn=lambda x : x, # Function applied to tokens to construct the vocabulary
            indices=True) # Indicates that indices will be computed instead of one-hot vectors

    # Constructs indicators of which color (0, 1, or 2) that the listener clicked
    def test_listener_clicked(self):
        print "Featurizing listener clicks"
        mung.feature_helpers.featurize_path_scalars(
            input_data_dir, # Source data set
            join(output_feature_dir, "listener_clicked"), # Output directory
            partition_file, # Data partition
            lambda d : d.get("gameid"), # Function that partitions the data
            "listener_clicked", # Name of the feature
            ["action.lClicked_0", "action.lClicked_1", "action.lClicked_2"]) # JSON paths to feature values

    # Constructs indices of which color (0, 1, or 2) that the listener clicked
    def test_listener_clicked_indices(self):
        print "Featurizing listener click indices"
        mung.feature_helpers.featurize_path_scalars(
            input_data_dir, # Source data set
            join(output_feature_dir, "listener_clicked_idx"), # Output directory
            partition_file, # Data partition
            lambda d : d.get("gameid"), # Function that partitions the data
            "listener_clicked_idx", # Name of the feature
            ["action.lClickedIndex"]) # JSON paths to feature values

    # Constructs vectors of the colors that the listener observed
    def test_listener_colors(self):
        print "Featurizing listener colors"
        mung.feature_helpers.featurize_path_scalars(
            input_data_dir,
            join(output_feature_dir, "listener_colors"),
            partition_file,
            lambda d : d.get("gameid"),
            "listener_colors",
            ["state.lH_0","state.lS_0","state.lL_0",
             "state.lH_1","state.lS_1","state.lL_1",
             "state.lH_2","state.lS_2","state.lL_2"])

    def test_listener_colors_ft(self):
        print "Featurizing listener colors (ft)"
        mung.feature_helpers.featurize_path_scalars(
            input_data_dir,
            join(output_feature_dir, "listener_colors_ft"),
            partition_file,
            lambda d : d.get("gameid"),
            "listener_colors_ft",
            ["state.lFT_0", "state.lFT_1", "state.lFT_2"])

    # Constructs vectors of the colors that the speaker observed
    def test_speaker_colors(self):
        print "Featurizing speaker colors"
        mung.feature_helpers.featurize_path_scalars(
            input_data_dir,
            join(output_feature_dir, "speaker_colors"),
            partition_file,
            lambda d : d.get("gameid"),
            "speaker_colors",
            ["state.sH_0","state.sS_0","state.sL_0",
             "state.sH_1","state.sS_1","state.sL_1",
             "state.sH_2","state.sS_2","state.sL_2"])

    def test_speaker_colors_ft(self):
        print "Featurizing speaker colors (ft)"
        mung.feature_helpers.featurize_path_scalars(
            input_data_dir,
            join(output_feature_dir, "speaker_colors_ft"),
            partition_file,
            lambda d : d.get("gameid"),
            "speaker_colors_ft",
            ["state.sFT_0", "state.sFT_1", "state.sFT_2"])

    # Constructs vectors of colors, and an indicator of the target that the
    # speaker observed
    def test_speaker_observed(self):
        print "Featurizing speaker observations"
        mung.feature_helpers.featurize_path_scalars(
            input_data_dir,
            join(output_feature_dir, "speaker_observed"),
            partition_file,
            lambda d : d.get("gameid"),
            "speaker_observed",
            ["state.sH_0","state.sS_0","state.sL_0",
             "state.sH_1","state.sS_1","state.sL_1",
             "state.sH_2","state.sS_2","state.sL_2",
             "state.sTarget_0","state.sTarget_1","state.sTarget_2"])

    def test_speaker_observed_ft(self):
        print "Featurizing speaker observations (ft)"
        mung.feature_helpers.featurize_path_scalars(
            input_data_dir,
            join(output_feature_dir, "speaker_observed_ft"),
            partition_file,
            lambda d : d.get("gameid"),
            "speaker_observed_ft",
            ["state.sFT_0", "state.sFT_1", "state.sFT_2",
             "state.sTarget_0","state.sTarget_1","state.sTarget_2"])

    # Constructs the target color that the speaker observed
    def test_speaker_target_color(self):
        print "Featurizing speaker target colors"
        mung.feature_helpers.featurize_path_scalars(
            input_data_dir,
            join(output_feature_dir, "speaker_target_color"),
            partition_file,
            lambda d : d.get("gameid"),
            "speaker_target_color",
            ["state.sTargetH","state.sTargetS","state.sTargetL"])

    def test_speaker_target_color_ft(self):
        print "Featurizing speaker target colors (ft)"
        mung.feature_helpers.featurize_path_scalars(
            input_data_dir,
            join(output_feature_dir, "speaker_target_color_ft"),
            partition_file,
            lambda d : d.get("gameid"),
            "speaker_target_color_ft",
            ["state.sTargetFT"])

    # Constructs an indicator of the target color index (0, 1, or 2) that
    # the speaker observed
    def test_speaker_target(self):
        print "Featurizing speaker targets"
        mung.feature_helpers.featurize_path_scalars(
            input_data_dir,
            join(output_feature_dir, "speaker_target"),
            partition_file,
            lambda d : d.get("gameid"),
            "speaker_target",
            ["state.sTarget_0","state.sTarget_1","state.sTarget_2"])

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
            ["state.sTargetIndex"])

    def test_speaker_target_color_cielab(self):
        print "Featurizing speaker target color (cielab)"
        ltprg.data.feature_helpers.featurize_embeddings(
            input_data_dir,
            join(output_feature_dir, "speaker_target_color_cielab"),
            partition_file,
            lambda d : d.get("gameid"),
            # "target_fc_embedding",
            "speaker_target_color_cielab",
            [["state.sTargetH", "state.sTargetS", "state.sTargetL"]],
            # "fc-6",
             "cielab")

    def test_speaker_colors_cielab(self):
        print "Featurizing speaker colors (cielab)"
        ltprg.data.feature_helpers.featurize_embeddings(
            input_data_dir,
            join(output_feature_dir, "speaker_colors_cielab"),
            partition_file,
            lambda d : d.get("gameid"),
            # "context_fc_embedding",
            "speaker_colors_cielab",
            [["state.lH_0", "state.lS_0", "state.lL_0"],
             ["state.lH_1", "state.lS_1", "state.lL_1"],
             ["state.lH_2", "state.lS_2", "state.lL_2"]],
             # "fc-6",
             "cielab")

if __name__ == '__main__':
    unittest.main()
