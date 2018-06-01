import sys
import numpy as np
from mung.data import DataSet, Partition

np.random.seed(1)

SPLIT_SIZES = [0.8,0.1,0.1]
PART_NAMES = ["train", "dev", "test"]

data_dir = sys.argv[1]
split_output_file = sys.argv[2]

D_all = DataSet.load(data_dir, id_key="gameid")
partition = Partition.make(D_all, SPLIT_SIZES, PART_NAMES, lambda d : d.get_id())
partition.save(split_output_file)
