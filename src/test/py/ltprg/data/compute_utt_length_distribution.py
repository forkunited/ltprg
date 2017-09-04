import sys
from mung.data import DataSet

data_dir = sys.argv[1]

D = DataSet.load(data_dir)

histogram = dict()
for i in range(len(D)):
    strs = D[i].get("utterances[*].nlp.clean_strs.strs", first=False)
    
    length = 2 + max(0, len(strs) - 1)
    for s in strs:
        length += len(s)

    if length not in histogram:
        histogram[length] = 0
    histogram[length] += 1

print "Length\tCount"
for length in histogram.keys():
    print str(length) + ": " + str(histogram[length])
