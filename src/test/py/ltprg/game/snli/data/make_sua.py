import sys
import json
from mung.data import Datum, DataSet

raw_train_dir = sys.argv[1]
raw_dev_dir = sys.argv[2]
raw_test_dir = sys.argv[3]
output_data_dir = sys.argv[4]
output_split = sys.argv[5]

def make_sua(raw_data, split, part_name):
    sua_datums = []
    sua_id = 0
    for datum in raw_data:
        label = None
        if datum["gold_label"] == "-":
            continue
        elif datum["gold_label"] == "contradiction":
            label = 0
        elif datum["gold_label"] == "entailment":
            label = 1
        elif datum["gold_label"] == "neutral":
            label = 2

        split[part_name][str(sua_id)] = 1

        sua_properties = dict()
        sua_properties["id"] = str(sua_id)
        sua_properties["game_id"] = datum["pairID"]
        sua_properties["sua"] = 0
        sua_properties["roundNum"] = 0
        sua_properties["state"] = {
            "type" : "StateSNLI",
            "sTarget" : label,
            "contents" : datum["sentence1"]
        }
        sua_properties["utterances"] = {
            "sender": "speaker",
            "eventType": "utterance",
            "time": 1,
            "type": "Utterance",
            "contents": datum["sentence2"]
        }
        sua_properties["action"] = None
        sua_datums.append(Datum(properties=sua_properties))
        sua_id += 1

    return sua_datums

raw_train = DataSet.load(raw_train_dir, id_key="pairID")
raw_dev = DataSet.load(raw_train_dir, id_key="pairID")
raw_test = DataSet.load(raw_train_dir, id_key="pairID")

sua_datums = []
partition_split = { "train" : dict(), "dev" : dict(), "test" : dict()}
sua_datums.extend(make_sua(raw_train, partition_split, "train"))
sua_datums.extend(make_sua(raw_dev, partition_split, "dev"))
sua_datums.extend(make_sua(raw_test, partition_split, "test"))

sua_data.save(output_data_dir)
sua_data = DataSet(data=sua_datums)

partition = dict()
partition["keep_data"] = False
partition["size"] = len(sua_datums)
partition["parts"] = partition_split
with open(output_split, 'w') as fp:
    json.dump(partition, fp)
