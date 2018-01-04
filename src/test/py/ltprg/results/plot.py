#!/usr/bin/python
#
# Usage: plot.py [input_file] [xlabel] [ylabel] [x] [y] [where] [where_values] [groupby]
#
# input_file: Input tsv file where the first row contains column names
# xlabel: Label for plot horizontal axis
# ylabel: Label for plot vertical axis
# x: Name of column to plot on horizontal axis
# y: Name of column to plot on vertical axis
# where: Comma-separated list of columns for which to constrain the values contained in the plot
# where_values: Comma-separated list of values by which to constrain the columns given in [where]
# groupby: Comma-separated list of columns on which to group the data into separate curves
#
# The script will generate a 2-dimensional plot containing a set of curves.  Values are averaged
# across rows of data that fit the constraints given in [where] and [where_values].  The averages
# are computed for separate curves determined by the [groupby]
#

import csv
import sys
import numpy as np
from scipy import stats
from random import randint

input_file = sys.argv[1]
xlabel = sys.argv[2]
ylabel = sys.argv[3]
x = sys.argv[4]
y = sys.argv[5]

where = None
where_values = None
if len(sys.argv) > 6 and sys.argv[6] != 'None' and sys.argv[7] != 'None':
    where = sys.argv[6].split(",")
    where_values = sys.argv[7].split(",")

groupby = None
if len(sys.argv) > 8:
    groupby = sys.argv[8].split(",")

make_table = False
if len(sys.argv) > 9:
    make_table = (sys.argv[9] == "True")

def read_tsv_file(file_path):
    f = open(file_path, 'rt')
    rows = []
    try:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            rows.append(row)
    finally:
        f.close()
    return rows


def row_match(row, where, where_values):
    if where is None:
        return True

    for i in range(len(where)):
        if row[where[i]] != where_values[i]:
            return False
    return True


# Map [groupby],x -> y value list filtered by 'where'
def aggregate(rows, x, y, where, where_values, groupby):
    agg = dict()
    for row in rows:
        if not row_match(row, where, where_values):
            continue
        cur_agg = agg

        if groupby is not None:
            for key in groupby:
                if row[key] not in cur_agg:
                    cur_agg[row[key]] = dict()
                cur_agg = cur_agg[row[key]]

        x_value = row[x]
        y_value = row[y]
        if x_value not in cur_agg:
            cur_agg[x_value] = []
        cur_agg[x_value].append(float(y_value))
    return agg


def compute_statistics_helper(agg, agg_depth, keys, statistics, overall_statistics):
    if agg_depth == 0:
        cur_stats = statistics
        for key in keys:
            if key not in cur_stats:
                cur_stats[key] = dict()
            cur_stats = cur_stats[key]
        cur_stats["mu"] = np.mean(agg)
        cur_stats["stderr"] = stats.sem(agg)
        cur_stats["max"] = max(agg)
        overall_statistics["y_max"] = max(overall_statistics["y_max"], cur_stats["mu"])
        if len(keys[len(keys) - 1]) != 0:
            overall_statistics["x_max"] = max(overall_statistics["x_max"], float(keys[len(keys) - 1]))
    else:
        for key in agg:
            keys.append(key)
            compute_statistics_helper(agg[key], agg_depth - 1, keys, statistics, overall_statistics)
            keys.pop()
    return statistics, overall_statistics


def compute_statistics(agg, groupby):
    overall_statistics = dict()
    overall_statistics["y_max"] = 1.0
    overall_statistics["x_max"] = 0
    statistics = dict()
    depth = 1
    if groupby is not None:
        depth = len(groupby) + 1
    return compute_statistics_helper(agg, depth, [], statistics, overall_statistics)


def make_latex_plot_helper(statistics, groupby, depth, keys, s):
    if depth == 0:
        plot_str = "\\addplot[color=black!" + str(randint(30,100)) + ",dash pattern=on " + str(randint(1,3)) + "pt off " + str(randint(1,2)) + "pt,error bars/.cd, y dir=both,y explicit] coordinates {\n"

	x_values = [float(x_value) for x_value in statistics.keys()]
	x_values.sort()
        for x_value in x_values:
	    x_str = str(int(x_value))
            plot_str = plot_str + "(" + x_str + "," + str(statistics[x_str]["mu"]) + ")+-(0.0," + str(statistics[x_str]["stderr"]) + ")\n"

        plot_str = plot_str + "};\n"
        plot_str = plot_str + "\\addlegendentry{\\tiny{"
        if groupby is not None:
            for i in range(len(groupby)):
                plot_str = plot_str + groupby[i] + "=" + keys[i] + " "
	plot_str = plot_str.strip()
        plot_str = plot_str + "}};\n\n"

        return s + plot_str
    else:
        for key in statistics:
            keys.append(key)
            s = make_latex_plot_helper(statistics[key], groupby, depth - 1, keys, s)
            keys.pop()
    return s


def make_latex_plot(statistics, overall_statistics, xlabel, ylabel, groupby):
    s = ("\\begin{figure*}[ht]\n"
           "\\begin{center}\n"
           "\\begin{tikzpicture}\n"
           "\\begin{axis}[%\n"
           "width=.5\\textwidth,height=.5\\textwidth,\n"
           "anchor=origin, % Shift the axis so its origin is at (0,0)\n"
           "ymin=0,ymax=" + str(overall_statistics["y_max"]) + ",xmin=0,xmax=" + str(overall_statistics["x_max"]) + ",%\n"
           "xlabel=" + xlabel + ",\n"
           "ylabel=" + ylabel + ",\n"
           "legend pos=outer north east\n"
           "]\n"
    )

    depth = 0
    if groupby is not None:
        depth = len(groupby)

    s = s + make_latex_plot_helper(statistics, groupby, depth, [], "")

    s = s + ("\\end{axis}\n"
                 "\\end{tikzpicture}\n"
                 "\\end{center}\n"
                 "\\end{figure*}\n"
    )

    return s

def make_aggregate_table_helper(statistics, groupby, depth, keys, s):
    if depth == 0:
        x_values = [float(x_value) if len(x_value) != 0 else "" for x_value in statistics.keys()]
        x_values.sort()
        for x_value in x_values:
            x_str = str(x_value)
            for key in keys:
                s += key + "\t"
            if x_str not in statistics:
                x_str = str(int(float(x_str))) # FIXME Stupid hack for now
            s += x_str + "\t" + str(statistics[x_str]["mu"]) + "\t" + str(statistics[x_str]["stderr"]) + "\t" + str(statistics[x_str]["max"]) + "\n"
        return s
    else:
        for key in statistics:
            keys.append(key)
            s = make_aggregate_table_helper(statistics[key], groupby, depth - 1, keys, s)
            keys.pop()
        return s

def make_aggregate_table(statistics, overall_statistics, xlabel, ylabel, groupby):
    s = "\t".join(groupby) + "\t" + xlabel + "\t" + ylabel + "\t" + ylabel + " (stderr)" + "\t" + ylabel + " (max)\n"

    depth = 0
    if groupby is not None:
        depth = len(groupby)

    s = s + make_aggregate_table_helper(statistics, groupby, depth, [], "")

    return s

rows = read_tsv_file(input_file)
agg = aggregate(rows, x, y, where, where_values, groupby)
statistics, overall_statistics = compute_statistics(agg, groupby)
if make_table:
    print(make_aggregate_table(statistics, overall_statistics, xlabel, ylabel, groupby))
else:
    print(make_latex_plot(statistics, overall_statistics, xlabel, ylabel, groupby))

