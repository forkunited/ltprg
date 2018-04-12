#!/usr/bin/python

"""
This script convert reference game data from CSV to JSON format.  In the CSV
format, there should be several directories containing CSV files.  The CSV files
of a particular directory should describe reference game events of particular
type.  For example, one directory could contain CSV files describing
'utterances', another directory could contain CSV files describing 'states',
and a third could contain CSV files containing 'actions'.  Each column of one
of these CSV files represents some dimension of an event, and each row represents
a particular game event.

In the "examples/games/csv/color" directory, there is an example of the color
reference game data set in this CSV format.  This data set consists of state,
action, and utterance events represented by CSV files in the corresponding
directories (these directories were built from the source "filteredCorpus.csv"
file containing all the data).  Each CSV file in these directories contains
events from a single game (but this isn't strictly necessary for running
this script... a directory could just contain a single csv file with all events
of a particular type.)

Running this script on the CSV data will produce files containing game JSON
objects of the form:

{
"gameid" : "*unique game identifier string*",
"records": [{ "roundNum": 1,
              "events": [
               { "type": "*StateSubtype*", "time": 1476996301986, "..." : "..." },
               { "type": "Utterance", "time": 1476996265147, "sender": "speaker", "contents": "*Stuff said by speaker*"},
               { "type": "Utterance", "time": 1476996265180, "sender": "listener", "contents": "*Stuff said by listener*"},
               { "type": "Utterance", "time": 1476996265190, "sender": "speaker", "contents": "*More stuff said by speaker*"},
               { "...", "..."},
               { "type": "*ActionSubtype*", "time": 1476996267239, "..." : "..." }
              ]
            },
            { "roundNum": 2, "events": [ { "...": "..." } ]},
            { "roundNum": 3, "events": [ { "...": "..." } ]},
            { "..." : "..."}
           ]
}

Note that in the above schema, place-holder values are given between the
asterisks, and the "..." fields indicate that the object could contain more
fields.

In this format, each reference game is represented by a single JSON object
containing a *"gameid"* unique identifier field for the game, and a *"records"*
field that contains a list of numbered game round objects.  Each round object
consists of a *"roundNum"* (round number) and a list of events.  Each event
is either a state, an utterance, or an action.  Each of these events can
have several game-specific dimensions with arbitrarily complicated substructure
determined by the dimensions of the events of a particular game.

For example, running this script with the following arguments will
reproduce the color data set in JSON format as it currently resides in
examples/games/json/color.

    game_groupby: gameid
    record_groupby: roundNum
    input_dirs: examples/games/csv/color/state/,examples/games/csv/color/action/,examples/games/csv/color/utterance/
    input_file_types: StateColor,ActionColor,Utterance
    output_dir: examples/games/json/color

The first two arguments specify which CSV columns to use to group the CSV lines
into game and record (round) JSON objects.   The third argument lists the
directories containing CSV files representing different event types.  The fourth
argument specifies the names of the types for the events in these directories.
The final argument gives path to the output directory in which to output the
JSON game objects.

Args:
    game_groupby (:obj:`str`): Name of the column on which to group csv lines
        into game JSON objects.  This column
    record_groupby (:obj:`str`): Name of the column on which to group csv lines
        into round records (In the example given above, this will be
        "roundNum")
    input_dirs (:obj:`str`): Comma-separated list of directories containing
        CSV files describing game events of different types
    input_file_types (:obj:`str`): Comma-separated list of types of events in
        the CSV files in the directories given by "input_dirs"
    output_dir (:obj:`str`): Directory in which to store the output JSON data

"""

import csv
import sys
import json
from os import listdir
from os.path import isfile, join

game_groupby = sys.argv[1]
record_groupby = sys.argv[2]
input_dirs = sys.argv[3].split(",")
input_file_types = sys.argv[4].split(",")
output_dir = sys.argv[5]
delimiter = ","
if len(sys.argv) > 6:
    delimiter = sys.argv[6]
    if delimiter == "[TAB]":
        delimiter = "\t"

def process_csv_files(input_dirs, input_file_types):
    D = dict()
    for i in range(len(input_dirs)):
        input_files = [join(input_dirs[i], f) for f in listdir(input_dirs[i]) if isfile(join(input_dirs[i], f))]
        input_file_type = input_file_types[i]
        for input_file in input_files:
            process_csv_file(input_file, D, input_file_type)
    return D

def process_csv_file(file_path, D, file_type):
    f = open(file_path, 'rt')
    try:
        reader = csv.DictReader(f, delimiter=delimiter)
        for record in reader:
            process_record(record, D, file_type)
    finally:
        f.close()
    return D

def process_record(record, D, record_type):
    D_sub = D
    if record[game_groupby] not in D_sub:
        D_sub[record[game_groupby]] = dict()
    D_sub = D_sub[record[game_groupby]]

    if record[record_groupby] not in D_sub:
        D_sub[record[record_groupby]] = []
    D_sub = D_sub[record[record_groupby]]

    sub_record = dict()
    for key in record:
        if key != game_groupby and key != record_groupby:
            if key == "time":
                sub_record[key] = int(record[key])
            elif (record[key].startswith("[") and record[key].endswith("]")) or (record[key].startswith("{") and record[key].endswith("}")):
                obj = json.loads(record[key])
                if key == "action" or key == "utterance" or key == "obj":
                    for k in obj:
                        sub_record[k] = obj[k]
            else:
                sub_record[key] = record[key]
    sub_record["type"] = record_type

    D_sub.append(sub_record)

def output_obj(file_path, obj):
    f = open(file_path, 'w')
    try:
        f.write(json.dumps(obj))
    finally:
        f.close()

def output_record_files(D):
    for key in D:
        records = D[key]
        record_keys = [str(ikey) for ikey in sorted([int(rkey) for rkey in records.keys()])]
        records_list = []
        for record_key in record_keys: # A record key is usually the number for a round
            record = dict()
            record[record_groupby] = int(record_key)
            record["events"] = sorted(records[record_key], key=lambda x: x["time"])
            records_list.append(record)
        document_obj = dict()
        document_obj[game_groupby] = key
        document_obj["records"] = records_list
        output_obj(output_dir + "/" + key, document_obj)

output_record_files(process_csv_files(input_dirs, input_file_types))
