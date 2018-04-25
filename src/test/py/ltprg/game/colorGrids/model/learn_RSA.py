import argparse

from mung.util.config import Config
from mung.feature import MultiviewDataSet, Symbol

parser = argparse.ArgumentParser()
parser.add_argument('env', action="store")
parser.add_argument('data', action="store")
parser.add_argument('--seed', action='store', dest='seed', type=int, default=1)
args = parser.parse_args()

seed = args["seed"]
env = Config.load(args["env"])
data_config = Config.load(args["data"], environment=env)

gpu = env["gpu"]

if gpu:
    torch.cuda.manual_seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)

_, S = MultiviewDataSet.load_from_config(data_config)

print "Loaded data..."
for key in S.keys():
    print key + ": " + len(S[key])
