import sys
from mung.data import DataSet, Datum

input_data_dir = sys.argv[1]
output_data_dir = sys.argv[2]

D = DataSet.load(input_data_dir, id_key="gameid")
datums_out = []

for d in D:
    d_props = d.to_dict()
    target_index = d.get("state.state.target")
    target_obj = d.get("state.state.objs[" + target_index + "]")

    d_props["state"]["state"]["targetObj"] = target_obj

    datums_out.append(Datum(properties=d_props, id_key="gameid"))

D_out = DataSet(data=datums_out, id_key="gameid")
D_out.save(output_data_dir)