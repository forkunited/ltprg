import sys
import numpy as np
from mung.data import DataSet, Datum, Partition

color_data_dir = sys.argv[1]
grid_data_dir = sys.argv[2]
color_data_name = sys.argv[3]
grid_data_name = sys.argv[4]
output_dir = sys.argv[5]

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
D_grid = DataSet.load(grid_data_dir, id_key="gameid")

datums_out = []
for d in D_color:
    add_datum(datums_out, d, color_data_name)
for d in D_grid:
    add_datum(datums_out, d, grid_data_name)
D_out = DataSet(data=datums_out, id_key="gameid")

D_out.save(output_dir)
