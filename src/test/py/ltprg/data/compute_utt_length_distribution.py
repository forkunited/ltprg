import sys
from mung.data import DataSet

data_dir = sys.argv[1]

D = DataSet.load(data_dir)

empty_utt_count = 0
length_histogram = dict()
utt_count_histogram = dict()
highest_length = 0
highest_length_index = 0
for i in range(len(D)):
    strs = D[i].get("utterances[*].nlp.clean_strs.strs", first=False)
    
    length = 2 + max(0, len(strs) - 1)
    for s in strs:
        length += len(s)

    if len(strs) > 0 and length == 2:
        empty_utt_count += 1

    if length not in length_histogram:
        length_histogram[length] = 0
    length_histogram[length] += 1

    if length > highest_length:
        highest_length = length
        highest_length_index = i

    if len(strs) not in utt_count_histogram:
        utt_count_histogram[len(strs)] = 0
    utt_count_histogram[len(strs)] += 1

print "Empty utterances: " + str(empty_utt_count)

print "Lengths distribution"
print "Length\tCount"
for length in length_histogram.keys():
    print str(length) + ": " + str(length_histogram[length])

print "Utterance count distribution"
print "Utterances\tCount"
for count in utt_count_histogram.keys():
    print str(count) + ": " + str(utt_count_histogram[count])

print "Longest length datum: "
strs = D[highest_length_index].get("utterances[*].nlp.clean_strs.strs", first=False)
long_utt = ""
for utt in strs:
    for s in utt:
        long_utt += s + " "
    long_utt += "# "
print long_utt

