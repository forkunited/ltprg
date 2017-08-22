import numpy as np
import unittest
import sys
import mung.feature_helpers
from os.path import join
from mung.feature import MultiviewDataSet
from mung.data import Partition
from ltprg.model.seq import SequenceModel, SamplingMode
from ltprg.model.prior import SequenceSamplingPriorFn

data_dir = sys.argv[1]
partition_file = sys.argv[2]
utterance_dir = sys.argv[3]
L_world_dir = sys.argv[4]
L_observation_dir = sys.argv[5]
S_world_dir = sys.argv[6]
S_observation_dir = sys.argv[7]
seq_model_path = sys.argv[8]

# Necessary to allow unittest.main() to work
del sys.argv[8]
del sys.argv[7]
del sys.argv[6]
del sys.argv[5]
del sys.argv[4]
del sys.argv[3]
del sys.argv[2]
del sys.argv[1]

np.random.seed(1)

class TestPrior(unittest.TestCase):

    # See if 0856-3_17_0 is sensible
    def test_seq_utterance_prior(self):
        D = MultiviewDataSet.load(data_dir,
                                  dfmat_paths={ "L_world" : L_world_dir, \
                                                "L_observation" : L_observation_dir, \
                                                "S_world" : S_world_dir, \
                                                "S_observation" : S_observation_dir \
                                   },
                                  dfmatseq_paths={ "utterance" : utterance_dir })
        partition = Partition.load(partition_file)

        D_parts = D.partition(partition, lambda d : d.get("gameid"))
        D_dev_sub = D_parts["dev"].filter(lambda d : d.get("gameid").startswith("085"))

        world_input_size = 3
        utterance_size = D_dev_sub["utterance"].get_matrix(0).get_feature_set().get_token_count()

        seq_prior_model = SequenceModel.load(seq_model_path)
        utterance_prior_fn = SequenceSamplingPriorFn(seq_prior_model, world_input_size, \
                                                     mode=SamplingMode.BEAM, #FORWARD, # BEAM
                                                     samples_per_input=4,
                                                     uniform=True,
                                                     seq_length=D_dev_sub["utterance"].get_feature_seq_set().get_size()) # 3 is color dimension

        print "Running prior on data size " + str(D_dev_sub.get_data().get_size())

        prior = utterance_prior_fn(D_dev_sub["S_observation"].get_batch(0, D_dev_sub.get_data().get_size()))

        for index in range(D_dev_sub.get_data().get_size()):
            if D_dev_sub.get_data().get(index).get("id") != "0856-3_17_0":
                continue

            H_0 = D_dev_sub.get_data().get(index).get("state.sH_0")
            S_0 = D_dev_sub.get_data().get(index).get("state.sS_0")
            L_0 = D_dev_sub.get_data().get(index).get("state.sL_0")
            H_1 = D_dev_sub.get_data().get(index).get("state.sH_1")
            S_1 = D_dev_sub.get_data().get(index).get("state.sS_1")
            L_1 = D_dev_sub.get_data().get(index).get("state.sL_1")
            H_2 = D_dev_sub.get_data().get(index).get("state.sH_2")
            S_2 = D_dev_sub.get_data().get(index).get("state.sS_2")
            L_2 = D_dev_sub.get_data().get(index).get("state.sL_2")

            utterance_lists = D_dev_sub.get_data().get(index).get("utterances[*].nlp.lemmas.lemmas", first=False)
            observed_utt = " # ".join([" ".join(utterance) for utterance in utterance_lists])

            support_utts = self._get_support_utts(D_dev_sub, prior, index)

            prior_index = utterance_prior_fn(D_dev_sub["S_observation"].get_batch(index, 1))
            single_support_utts_0 = self._get_support_utts(D_dev_sub, prior_index, 0)

            prior_index = utterance_prior_fn(D_dev_sub["S_observation"].get_batch(index, 1))
            single_support_utts_1 = self._get_support_utts(D_dev_sub, prior_index, 0)

            print "ID: " + D_dev_sub.get_data().get(index).get("id")
            print "H0: " + str(H_0) + ", S0: " + str(S_0) + ", L0: " + str(L_0)
            print "H1: " + str(H_1) + ", S1: " + str(S_1) + ", L1: " + str(L_1)
            print "H2: " + str(H_2) + ", S2: " + str(S_2) + ", L2: " + str(L_2)
            print "True utterance: " + observed_utt
            print " "
            print "Batch sampled utterances and single sampled utterances"
            for j in range(len(support_utts)):
                print support_utts[j] + "\t" + single_support_utts_0[j] + "\t" + single_support_utts_1[j]
            print "\n"

    def _get_support_utts(self, D, prior, index=0):
        prior_utts = prior.support()[0]
        prior_lens = prior.support()[1]
        support_utts = [" ".join([D["utterance"].get_feature_token(prior_utts.data[index,j,k]).get_value() \
                                        for k in range(prior_lens[index,j])]) for j in range(prior.p().size(1))]
        return support_utts


if __name__ == '__main__':
    unittest.main()
