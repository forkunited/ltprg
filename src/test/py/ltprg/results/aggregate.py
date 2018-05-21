#!/usr/bin/python

import csv
import sys
import argparse
import json
import numpy as np
from scipy import stats
from jsonpath_ng import jsonpath
from jsonpath_ng.ext import parse
from os import listdir
from os.path import isfile, join
from collections import OrderedDict

parser = argparse.ArgumentParser()
parser.add_argument('input_dir', action="store")
parser.add_argument('result_file_prefix', action="store")
parser.add_argument('config_file_prefix', action="store")
parser.add_argument('output_file_path', action="store")
parser.add_argument('--config_fields', nargs='+')
parser.add_argument('--results_fields', nargs='+')
args = parser.parse_args()

def process_result(result_file_path, config_file_path):
    config = None
    with open(config_file_path, 'rb') as fp:
        config = json.load(fp)

    config_field_values = dict()
    for field in args.config_fields:
        path_values = [(field, match.value) for match in parse(field).find(config)]
        for key, value in path_values:
            config_field_values[key] = value

    f = open(result_file_path, 'rt')
    rows = []
    try:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            row.update(config_field_values)
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

def group_rows(rows):
    agg = dict()  # Maps config field values to sets of results
    for row in rows:
        cur_agg = agg
        for field in args.config_fields:
            if field not in cur_agg:
                cur_agg[field] = dict()
            cur_agg = cur_agg[field]

        for field in args.results_fields:
            if field not in cur_agg:
                cur_agg[field] = []
            cur_agg[field].append(row[field])

    return make_grouped_rows(agg, 0, dict(), [])

def make_grouped_rows(agg, field_index, cur_row, grouped_rows):
    if field_index == len(args.config_fields):
        for field in args.results_fields:
            grouped_results = agg[field]
            grouped_results = np.array([float(grouped_results[i]) for i in range(len(grouped_results))])
            cur_row[field + " Mean"] = np.mean(grouped_results)
            cur_row[field + " SE"] = stats.sem(grouped_results)
            cur_row[field + " Max"] = np.max(grouped_results)
            grouped_rows.append(cur_row)
    else:
        for key in agg.keys():
            cur_row = dict(cur_row)
            cur_row[args.config_fields[field_index]] = key
            make_grouped_rows(agg[key], field_index+1, cur_row, grouped_rows)
    return grouped_rows

def aggregate_directory(dir_path):
    dirs = [join(dir_path, f) for f in listdir(dir_path) if not isfile(join(dir_path, f))]
    result_files = []
    config_files = []
    for d in dirs:
        for f in listdir(d):
            file_path = join(d, f)
            if isfile(file_path) and f.startswith(args.result_file_prefix):
                result_files.append(file_path)
            elif isfile(file_path) and f.startswith(args.config_file_prefix):
                config_files.append(file_path)

    rows = []
    for i in range(len(result_files)):
        rows.extend(process_result(result_files[i], config_files[i]))
    return rows

output_tsv(args.output_file_path, group_rows(aggregate_directory(args.input_dir)))
