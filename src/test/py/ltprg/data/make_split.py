import sys
import numpy as np
import argparse
from mung.data import DataSet, Partition

PART_NAMES = ["train", "dev", "test"]

parser = argparse.ArgumentParser()
parser.add_argument('data_dir', action="store")
parser.add_argument('split_output_file', action="store")
parser.add_argument('train_size', action="store", type=float)
parser.add_argument('dev_size', action="store", type=float)
parser.add_argument('test_size', action="store", type=float)
parser.add_argument('--seed', action='store', dest='seed', type=int, default=1)
parser.add_argument('--maintain_partition_file', action='store', dest='maintain_partition_file', default=None)
args = parser.parse_args()

np.random.seed(args.seed)

data_dir = sys.argv[1]
split_output_file = sys.argv[2]

part_sizes = [args.train_size, args.dev_size, args.test_size]

maintain_part = None
if args.maintain_partition_file is not None:
    maintain_part = Partition.load(args.maintain_partition_file)

D_all = DataSet.load(data_dir, id_key="gameid")
partition = Partition.make(D_all, part_sizes, PART_NAMES, lambda d : d.get_id(), maintain_part=maintain_part)
partition.save(split_output_file)
