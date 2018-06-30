import sys
from mung.data import DataSet, Partition

input_sua_dir = sys.argv[1]
partition_file = sys.argv[2]

S = dict()
P = Partition.load(partition_file)
D = DataSet.load(input_sua_dir)
D_parts = D.partition(P, lambda d: d.get("gameid"))

def compute_stats(D_full, data_name):
    S = dict()
    S["full"] = D_full

    S["color"] = D_full.filter(lambda d : d.get("state.state.condition.source") == "color3")
    S["color_close"] = S["color"].filter(lambda d : d.get("state.state.condition.name") == "CLOSE")
    S["color_split"] = S["color"].filter(lambda d : d.get("state.state.condition.name") == "SPLIT")
    S["color_far"] = S["color"].filter(lambda d : d.get("state.state.condition.name") == "FAR")

    S["grid"] = D_full.filter(lambda d : d.get("state.state.condition.source") == "grid3")
    S["grid_close"] = S["grid"].filter(lambda d : d.get("state.state.condition.name") == "CLOSE")
    S["grid_split"] = S["grid"].filter(lambda d : d.get("state.state.condition.name") == "SPLIT")
    S["grid_far"] = S["grid"].filter(lambda d : d.get("state.state.condition.name") == "FAR")

    S["grid_alld"] = S["grid"].filter(lambda d : d.get("state.state.condition.numDiffs") == 9)
    S["grid_larged"] = S["grid"].filter(lambda d : d.get("state.state.condition.numDiffs") >= 7 and d.get("state.state.condition.numDiffs") <= 8)
    S["grid_mediumd"] = S["grid"].filter(lambda d : d.get("state.state.condition.numDiffs") >= 3 and d.get("state.state.condition.numDiffs") <= 6)
    S["grid_smalld"] = S["grid"].filter(lambda d : d.get("state.state.condition.numDiffs") >= 1 and d.get("state.state.condition.numDiffs") <= 2)

    S["grid_close_alld"] = S["grid_close"].filter(lambda d : d.get("state.state.condition.numDiffs") == 9)
    S["grid_close_larged"] = S["grid_close"].filter(lambda d : d.get("state.state.condition.numDiffs") >= 7 and d.get("state.state.condition.numDiffs") <= 8)
    S["grid_close_mediumd"] = S["grid_close"].filter(lambda d : d.get("state.state.condition.numDiffs") >= 3 and d.get("state.state.condition.numDiffs") <= 6)
    S["grid_close_smalld"] = S["grid_close"].filter(lambda d : d.get("state.state.condition.numDiffs") >= 1 and d.get("state.state.condition.numDiffs") <= 2)

    S["grid_split_alld"] = S["grid_split"].filter(lambda d : d.get("state.state.condition.numDiffs") == 9)
    S["grid_split_larged"] = S["grid_split"].filter(lambda d : d.get("state.state.condition.numDiffs") >= 7 and d.get("state.state.condition.numDiffs") <= 8)
    S["grid_split_mediumd"] = S["grid_split"].filter(lambda d : d.get("state.state.condition.numDiffs") >= 3 and d.get("state.state.condition.numDiffs") <= 6)
    S["grid_split_smalld"] = S["grid_split"].filter(lambda d : d.get("state.state.condition.numDiffs") >= 1 and d.get("state.state.condition.numDiffs") <= 2)

    S["grid_far_alld"] = S["grid_far"].filter(lambda d : d.get("state.state.condition.numDiffs") == 9)
    S["grid_far_larged"] = S["grid_far"].filter(lambda d : d.get("state.state.condition.numDiffs") >= 7 and d.get("state.state.condition.numDiffs") <= 8)
    S["grid_far_mediumd"] = S["grid_far"].filter(lambda d : d.get("state.state.condition.numDiffs") >= 3 and d.get("state.state.condition.numDiffs") <= 6)
    S["grid_far_smalld"] = S["grid_far"].filter(lambda d : d.get("state.state.condition.numDiffs") >= 1 and d.get("state.state.condition.numDiffs") <= 2)

    print data_name + " data stats"
    print "---------------"

    for key, D in S.iteritems():
        total_count = 0.0
        total_correct = 0.0
        missing_action = 0.0
        for d in D:
            lClicked = d.get("action.action.lClicked")
            if lClicked is None:
                missing_action += 1.0
                continue

            target = d.get("state.state.target")
            selected = d.get("state.state.listenerOrder")[lClicked]

            correct = 0.0
            if target == selected:
                correct = 1.0

            total_correct += correct
            total_count += 1.0
        print key + ": " + str(total_correct/total_count) + "(" + str(total_correct) + "/" + str(total_count) + ") [" + str(missing_action) + "]"
    print "\n\n\n"

compute_stats(D, "Full")
compute_stats(D_parts["train"], "Train")
compute_stats(D_parts["dev"], "Dev")
compute_stats(D_parts["test"], "Test")

