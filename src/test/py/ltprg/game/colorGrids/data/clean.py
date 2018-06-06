import sys
from mung.data import DataSet, Datum

def filter_rounds(datum, to_remove):
    rounds = datum.get("records")
    remove_rounds = set()
    if datum.get("gameid") in to_remove:
        remove_rounds = to_remove[datum.get("gameid")]

    remove_indices = []
    for i in range(len(rounds)):
        has_speaker = False
        has_listener = False
        for event in rounds[i]["events"]:
            if event["eventType"] == "utterance" and event["sender"] == "speaker":
                has_speaker = True        
            if event["eventType"] == "action":
                has_listener = True

        if (not has_speaker) or (not has_listener) or rounds[i]["roundNum"] in remove_rounds:
            remove_indices.append(i)

    for index in remove_indices:
        del rounds[index]

    return datum

def has_rounds(datum):
    return len(datum.get("records")) > 0

cleaning_file = sys.argv[1]
grid_data_dir_1 = sys.argv[2]
grid_data_dir_2 = sys.argv[3]
grid_data_dir_3 = sys.argv[4]
clean_data_dir = sys.argv[5]

D_grid_1 = DataSet.load(grid_data_dir_1, id_key="gameid")
D_grid_2 = DataSet.load(grid_data_dir_2, id_key="gameid")
D_grid_3 = DataSet.load(grid_data_dir_3, id_key="gameid")

cleaning_content = None
with open(cleaning_file, "r") as fp:
    cleaning_content = fp.read()

rounds_to_remove = dict()
lines = cleaning_content.split("\n")
for line in lines:
    line_parts = line.strip().split(" ")
    game_id = line_parts[1]
    to_remove = []
    if len(line_parts) > 2:
        to_remove = line_parts[2]
        if to_remove == "*":
            to_remove = set(range(1,61))
        else:
            to_remove = set([int(round) for round in to_remove.split(",")])
    rounds_to_remove[game_id] = to_remove

datums_out = []

for d in D_grid_1:
    filtered_d = filter_rounds(d, rounds_to_remove)
    if has_rounds(filtered_d):
        datums_out.append(filtered_d)

for d in D_grid_2:
    filtered_d = filter_rounds(d, rounds_to_remove)
    if has_rounds(filtered_d):
        datums_out.append(filtered_d)

for d in D_grid_3:
    filtered_d = filter_rounds(d, rounds_to_remove)
    if has_rounds(filtered_d):
        datums_out.append(filtered_d)

D_out = DataSet(data=datums_out, id_key="gameid")
D_out.save(clean_data_dir)