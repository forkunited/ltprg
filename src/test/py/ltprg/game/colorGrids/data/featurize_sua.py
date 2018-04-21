import unittest
import numpy as np
import sys
import mung.feature_helpers
import ltprg.data.feature_helpers
from os.path import join
from ltprg.game.color.properties.colorspace_conversions import hsls_to_rgbs, rgbs_to_labs

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
            50, # Maximum utterance length
            indices=True, # Indicates that indices will be computed instead of one-hot vectors
            token_fn=lambda x : x) # Function applied to tokens to construct the vocabulary

    def test_colors_seq(self):
        print "Featurizing color sequence"
        num_objs = 3
        seq_length = grid_dim*grid_dim*num_objs+num_objs
        color_dim = 3
        def matrix_fn(datum):
            seq = np.zeros(shape=(seq_length, color_dim))
            for i in range(num_objs):
                obj_colors = np.array(datum.get("state.state.objs[" + str(i) + "].shapes[*].color", first=False))
                for j in range(obj_colors.shape[0]): # Convert to cielab
                    color_lst = [obj_colors[j,0], obj_colors[j,1], obj_colors[j,2]]
                    rgb = np.array(hsls_to_rgbs([map(int, color_lst)]))[0]
                    lab = np.array(rgbs_to_labs([rgb]))[0]
                    obj_colors[j] = lab
                seq[(i+i*grid_dim*grid_dim):(i+(i+1)*grid_dim*grid_dim),:] = obj_colors
                seq[i+(i+1)*grid_dim*grid_dim,:] = 0.0

            return seq

        def length_fn(datum):
            return seq_length

        mung.feature_helpers.featurize_matrix_seq(input_data_dir,
                             join(output_feature_dir, "colors_seq"),
                             partition_file,
                             lambda d : d.get("gameid"),
                             "colors_seq",
                             matrix_fn,
                             length_fn,
                             seq_length,
                             color_dim,
                             init_data="train")


    def test_objs_seq(self):
        print "Featurizing obj sequence"
        num_objs = 3
        seq_length = num_objs
        color_dim = 3
        def matrix_fn(datum):
            seq = np.zeros(shape=(seq_length, color_dim*grid_dim*grid_dim))
            for i in range(num_objs):
                obj_colors = np.array(datum.get("state.state.objs[" + str(i) + "].shapes[*].color", first=False))
                for j in range(obj_colors.shape[0]): # Convert to cielab
                    color_lst = [obj_colors[j,0], obj_colors[j,1], obj_colors[j,2]]
                    rgb = np.array(hsls_to_rgbs([map(int, color_lst)]))[0]
                    lab = np.array(rgbs_to_labs([rgb]))[0]
                    obj_colors[j] = lab
                seq[i] = obj_colors.reshape((color_dim*grid_dim*grid_dim)) 
            return seq

        def length_fn(datum):
            return seq_length

        mung.feature_helpers.featurize_matrix_seq(input_data_dir,
                             join(output_feature_dir, "objs_seq"),
                             partition_file,
                             lambda d : d.get("gameid"),
                             "objs_seq",
                             matrix_fn,
                             length_fn,
                             seq_length,
                             color_dim*grid_dim*grid_dim,
                             init_data="train")
 

    def test_colors(self):
        print "Featurizing speaker objs colors (cielab)"
        num_objs = 3
        dims = []
        for i in range(num_objs):
            for j in range(grid_dim*grid_dim):
                dims.append(["state.state.objs[" + str(i) + "].shapes[" + str(j) + "].color[0]", \
                             "state.state.objs[" + str(i) + "].shapes[" + str(j) + "].color[1]", \
                             "state.state.objs[" + str(i) + "].shapes[" + str(j) + "].color[2]"])

        ltprg.data.feature_helpers.featurize_embeddings(
            input_data_dir,
            join(output_feature_dir, "colors"),
            partition_file,
            lambda d : d.get("gameid"),
            # "context_fc_embedding",
            "colors",
            dims,
            # "fc-6",
            "cielab")

    
    # Constructs an index of the target color index (0, 1, or 2)
    def test_target_indices(self):
        print "Featurizing target indices"
        mung.feature_helpers.featurize_path_scalars(
            input_data_dir,
            join(output_feature_dir, "target_idx"),
            partition_file,
            lambda d : d.get("gameid"),
            "target_idx",
            ["state.state.target"])

if __name__ == '__main__':
    unittest.main()
