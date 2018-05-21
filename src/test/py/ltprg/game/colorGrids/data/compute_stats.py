import sys
from mung.data import DataSet

input_sua_dir = sys.argv[1]

S = dict()
D_full = DataSet.load(input_sua_dir)

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
S["grid_mediumed"] = S["grid"].filter(lambda d : d.get("state.state.condition.numDiffs") >= 3 and d.get("state.state.condition.numDiffs") <= 6)
S["grid_smalld"] = S["grid"].filter(lambda d : d.get("state.state.condition.numDiffs") >= 1 and d.get("state.state.condition.numDiffs") <= 2)

S["grid_close_alld"] = S["grid_close"].filter(lambda d : d.get("state.state.condition.numDiffs") == 9)
S["grid_close_larged"] = S["grid_close"].filter(lambda d : d.get("state.state.condition.numDiffs") >= 7 and d.get("state.state.condition.numDiffs") <= 8)
S["grid_close_mediumed"] = S["grid_close"].filter(lambda d : d.get("state.state.condition.numDiffs") >= 3 and d.get("state.state.condition.numDiffs") <= 6)
S["grid_close_smalld"] = S["grid_close"].filter(lambda d : d.get("state.state.condition.numDiffs") >= 1 and d.get("state.state.condition.numDiffs") <= 2)

S["grid_split_alld"] = S["grid_split"].filter(lambda d : d.get("state.state.condition.numDiffs") == 9)
S["grid_split_larged"] = S["grid_split"].filter(lambda d : d.get("state.state.condition.numDiffs") >= 7 and d.get("state.state.condition.numDiffs") <= 8)
S["grid_split_mediumed"] = S["grid_split"].filter(lambda d : d.get("state.state.condition.numDiffs") >= 3 and d.get("state.state.condition.numDiffs") <= 6)
S["grid_split_smalld"] = S["grid_split"].filter(lambda d : d.get("state.state.condition.numDiffs") >= 1 and d.get("state.state.condition.numDiffs") <= 2)

S["grid_far_alld"] = S["grid_far"].filter(lambda d : d.get("state.state.condition.numDiffs") == 9)
S["grid_far_larged"] = S["grid_far"].filter(lambda d : d.get("state.state.condition.numDiffs") >= 7 and d.get("state.state.condition.numDiffs") <= 8)
S["grid_far_mediumed"] = S["grid_far"].filter(lambda d : d.get("state.state.condition.numDiffs") >= 3 and d.get("state.state.condition.numDiffs") <= 6)
S["grid_far_smalld"] = S["grid_far"].filter(lambda d : d.get("state.state.condition.numDiffs") >= 1 and d.get("state.state.condition.numDiffs") <= 2)

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
