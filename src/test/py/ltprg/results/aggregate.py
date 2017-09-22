#!/usr/bin/python

import csv
import sys
from os import listdir
from os.path import isfile, join
from collections import OrderedDict

input_file_dir = sys.argv[1]
output_file_path = sys.argv[2]
tsv_line_start = None
tsv_line_end = None

def process_tsv_file(file_path):
    f = open(file_path, 'rt')
    rows = []
    try:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            rows.append(row)
    finally:
        f.close()
    return rows

def output_tsv(file_path, rows):
    keys = set(rows[0].keys())
    for row in rows:
        keys = keys & set(row.keys())

    for row in rows:
        to_rem = []
        for row_key in row.keys():
            if row_key not in keys:
                to_rem.append(row_key)
        for rem in to_rem:
            del row[rem]


    fields = OrderedDict([(k, None) for k in rows[0].keys()])
    f = open(file_path, 'wb')
    try:
        writer = csv.DictWriter(f, delimiter='\t', fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    finally:
        f.close()

def aggregate_directory(file_dir, file_type):
    files = [f for f in listdir(file_dir) if isfile(join(file_dir, f))]
    rows = []
    for file in files:
        rows.extend(process_tsv_file(join(file_dir, file)))
    return rows

output_tsv(output_file_path, aggregate_directory(input_file_dir, input_file_type))
