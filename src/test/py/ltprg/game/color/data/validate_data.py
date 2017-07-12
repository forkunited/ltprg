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


if __name__ == '__main__':
    unittest.main()
