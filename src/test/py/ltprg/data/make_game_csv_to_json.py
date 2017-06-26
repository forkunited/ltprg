#!/usr/bin/python

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
        reader = csv.DictReader(f, delimiter=',')
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
