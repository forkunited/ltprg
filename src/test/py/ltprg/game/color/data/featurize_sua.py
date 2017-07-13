import unittest
import numpy as np
import sys
import mung.feature_helpers
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

    def test_utterance_lemmas(self):
        print "Featurizing utterances"
        mung.feature_helpers.featurize_path_enum_seqs(
            input_data_dir,
            join(output_feature_dir, "utterance_lemmas"),
            partition_file,
            lambda d : d.get("gameid"),
            "utterance_lemmas",
            ["utterances[*].nlp.lemmas.lemmas"],
            15, # Maximum utterance length
            token_fn=lambda x : x.lower())

    
    def test_utterance_lemma_indices(self):
        print "Featurizing utterance indices"
        mung.feature_helpers.featurize_path_enum_seqs(
            input_data_dir,
            join(output_feature_dir, "utterance_lemma_idx"),
            partition_file,
            lambda d : d.get("gameid"),
            "utterance_lemmas_idx",
            ["utterances[*].nlp.lemmas.lemmas"],
            15, # Maximum utterance length
            token_fn=lambda x : x.lower(),
            indices=True)


    def test_listener_clicked(self):
        print "Featurizing listener clicks"
        mung.feature_helpers.featurize_path_scalars(
            input_data_dir,
            join(output_feature_dir, "listener_clicked"),
            partition_file,
            lambda d : d.get("gameid"),
            "listener_clicked",
            ["action.lClicked_0", "action.lClicked_1", "action.lClicked_2"])
    

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

    def test_speaker_target_color(self):
        print "Featurizing speaker target colors"
        mung.feature_helpers.featurize_path_scalars(
            input_data_dir,
            join(output_feature_dir, "speaker_target_color"),
            partition_file,
            lambda d : d.get("gameid"),
            "speaker_target_color",
            ["state.sTarget_H","state.sTarget_S","state.sTarget_L"])

    def test_speaker_target(self):
        print "Featurizing speaker targets"
        mung.feature_helpers.featurize_path_scalars(
            input_data_dir,
            join(output_feature_dir, "speaker_target"),
            partition_file,
            lambda d : d.get("gameid"),
            "speaker_target",
            ["state.sTarget_0","state.sTarget_1","state.sTarget_2"])


if __name__ == '__main__':
    unittest.main()
