import sys
import numpy as np
from mung.data import DataSet, Datum, Partition

color_data_dir = sys.argv[1]
grid_data_dir_1 = sys.argv[2]
grid_data_dir_2 = sys.argv[3]
color_part_file = sys.argv[4]
grid_part_file_1 = sys.argv[5]
grid_part_file_2 = sys.argv[6]
color_data_name = sys.argv[7]
grid_data_name_1 = sys.argv[8]
grid_data_name_2 = sys.argv[9]
output_dir = sys.argv[10]
output_part_file = sys.argv[11]

def add_datum(datums_list, d, d_source_name):
    d_props = d.to_dict()
    for record in d_props["records"]:
        for event in record["events"]:
            for key in event:
                if isinstance(event[key], dict) and "condition" in event[key]:
                    event[key]["condition"]["source"] = d_source_name
                    if "numDiffs" not in event[key]["condition"]:
                        event[key]["condition"]["numDiffs"] = 9
    datums_list.append(Datum(properties=d_props, id_key="gameid"))

D_color = DataSet.load(color_data_dir, id_key="gameid")
D_grid_1 = DataSet.load(grid_data_dir_1, id_key="gameid")
D_grid_2 = DataSet.load(grid_data_dir_2, id_key="gameid")
P_color = Partition.load(color_part_file)
P_grid_1 = Partition.load(grid_part_file_1)
P_grid_2 = Partition.load(grid_part_file_2)

datums_out = []
for d in D_color:
    add_datum(datums_out, d, color_data_name)
for d in D_grid_1:
    add_datum(datums_out, d, grid_data_name_1)
for d in D_grid_2:
    add_datum(datums_out, d, grid_data_name_2)
D_out = DataSet(data=datums_out, id_key="gameid")

P_grid_1.merge_parts(["train", "dev"], "train")
P_grid_1.split_part("train", ["train", "dev"], [0.8, 0.1])

P_out = P_color.union(P_grid_1).union(P_grid_2)


D_out.save(output_dir)
P_out.save(output_part_file)
