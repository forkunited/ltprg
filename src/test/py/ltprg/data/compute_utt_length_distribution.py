import sys
from mung.data import DataSet

data_dir = sys.argv[1]

D = DataSet.load(data_dir)

histogram = dict()
for i in range(len(D)):
    strs = D[i].get("utterance.clean_strs.strs")
    if len(strs) not in histogram:
        histogram[len(strs)] = 0
    histogram[len(strs)] += 1

print "Length\tCount"
for length in histogram[length]:
    print str(length) + ": " + str(histogram[length])
