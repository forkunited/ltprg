import sys
import numpy as np
from mung.data import DataSet, Datum, Partition

data_dir_1 = sys.argv[1]
data_dir_2 = sys.argv[2]
part_file_1 = sys.argv[3]
part_file_2 = sys.argv[4]
data_name_1 = sys.argv[5]
data_name_2 = sys.argv[6]
output_dir = sys.argv[7]
output_part_file = sys.argv[8]

def add_datum(datums_list, d, d_source_name):
    d_props = d.to_dict()
    for record in d_props["records"]:
        for event in record["events"]:
            for key in event:
                if isinstance(event[key], dict) and "condition" in event[key]:
                    event[key]["condition"]["source"] = d_source_name
    datums_list.append(Datum(properties=d_props, id_key="gameid"))

D_1 = DataSet.load(data_dir_1, id_key="gameid")
D_2 = DataSet.load(data_dir_2, id_key="gameid")
P_1 = Partition.load(part_file_1)
P_2 = Partition.load(part_file_2)

datums_out = []
for d in D_1:
    add_datum(datums_out, d, data_name_1)
for d in D_2:
    add_datum(datums_out, d, data_name_2)
D_out = DataSet(data=datums_out, id_key="gameid")
P_out = P_1.union(P_2)

D_out.save(output_dir)
P_out.save(output_part_file)
