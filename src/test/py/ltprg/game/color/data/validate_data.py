import unittest
import numpy as np
import sys
import mung.feature_helpers
from os.path import join
from mung.feature import MultiviewDataSet
from mung.data import Partition

sua_data_dir = sys.argv[1]
features_dir = sys.argv[2]
partition_file = sys.argv[3]

# Necessary to allow unittest.main() to work
del sys.argv[3]
del sys.argv[2]
del sys.argv[1]

np.random.seed(1)

class ValidateColorData(unittest.TestCase):    

    def test_validate_data(self):
        D = MultiviewDataSet.load(
            sua_data_dir,
            dfmat_paths={ 
                "listener_clicked" : join(features_dir, "listener_clicked"),
                "listener_colors" : join(features_dir, "listener_colors"),
                "speaker_colors" : join(features_dir, "speaker_colors"),
                "speaker_observed" : join(features_dir, "speaker_observed"),
                "speaker_target_color" : join(features_dir, "speaker_target_color"),
                "speaker_target" : join(features_dir, "speaker_target")
            },
            dfmatseq_paths={ 
                "utterance_lemma_idx" : join(features_dir, "utterance_lemma_idx")
            })
        partition = Partition.load(partition_file)
        D_parts = D.partition(partition, lambda d : d.get("gameid"))
        D_train = D_parts["train"]

        batch, indices = D_train.get_random_batch(5, return_indices=True)

        self._check_ids(D_train, batch, indices)
        self._check_scalar_view("listener_clicked", D_train, batch, indices)
        self._check_scalar_view("listener_colors", D_train, batch, indices)
        self._check_scalar_view("speaker_colors", D_train, batch, indices)
        self._check_scalar_view("speaker_observed", D_train, batch, indices)
        self._check_scalar_view("speaker_target_color", D_train, batch, indices)
        self._check_scalar_view("speaker_target", D_train, batch, indices)

        self._check_utterances(D_train, batch, indices)    


    def _check_utterances(self, D, batch, indices):
        feat = D["utterance_lemma_idx"].get_feature_seq_set().get_feature_seq(0)
        utt_batch, utt_length, utt_mask = batch["utterance_lemma_idx"]
        utt_batch = utt_batch.transpose(1,0,2)
        path_len = len("utterances[*].nlp.lemmas.lemmas_")
        for i in range(len(indices)):
            lemmas = D.get_data().get(indices[i]).get("utterances[*].nlp.lemmas.lemmas", first=False)
            feat_strs = [feat.get_type(0).get_token(int(idx[0])).get()[path_len:] for idx in utt_batch[i]]
            seq_index = 1
            for lemma_utterance in lemmas:
                for lemma in lemma_utterance:
                    if feat_strs[seq_index] != 'SYM_UNC':
                        self.assertEqual(feat_strs[seq_index], lemma)
                    seq_index += 1
                seq_index += 1

    def _check_scalar_view(self, view, D, batch, indices):
        feats = D[view].get_feature_set()
        for i in range(len(indices)):
            for j in range(feats.get_size()):
                feat_name = feats.get_feature_token(j).get()[:-2] # -2 chops off the list index added by path features
                self.assertEqual(batch[view][i][j], float(D.get_data().get(indices[i]).get(feat_name)))


    # Ensures determinism in data loading
    def _check_ids(self, D, batch, indices):
        #[u'6421-c_36_0', u'6035-4_18_0', u'8343-3_32_0', u'5834-1_4_0', u'1971-c_4_0']
        ids = [D.get_data().get(i).get("id") for i in indices]
        self.assertEqual(ids[0], '6421-c_36_0')
        self.assertEqual(ids[1], '6035-4_18_0')
        self.assertEqual(ids[2], '8343-3_32_0')
        self.assertEqual(ids[3], '5834-1_4_0')
        self.assertEqual(ids[4], '1971-c_4_0')

if __name__ == '__main__':
    unittest.main()
