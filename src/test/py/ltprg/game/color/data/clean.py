#!/usr/bin/python

import csv
import sys
import os
from collections import OrderedDict

input_file_path = sys.argv[1]
output_state_dir = sys.argv[2]
output_action_dir = sys.argv[3]
output_utterance_dir = sys.argv[4]

def read_messy_file(file_path):
    f = open(file_path, 'rt')
    D = dict()
    try:
        reader = csv.DictReader(f, delimiter=',')
        for record in reader:
            if record["gameid"] not in D:
                D[record["gameid"]] = dict()
            if record["roundNum"] not in D[record["gameid"]]:
                D[record["gameid"]][record["roundNum"]] = []
            D[record["gameid"]][record["roundNum"]].append(record)
    finally:
        f.close()
    return D

def make_state_record(record):
    listenerObjs = [dict(), dict(), dict()]
    speakerObjs = [dict(), dict(), dict()]

    clickedLisIndex = int(record["clickLocL"]) - 1
    clickedSpIndex = int(record["clickLocS"]) - 1
    alt1LisIndex = int(record["alt1LocL"]) - 1
    alt1SpIndex = int(record["alt1LocS"]) - 1
    alt2LisIndex = int(record["alt2LocL"]) - 1
    alt2SpIndex = int(record["alt2LocS"]) - 1

    target = 0
    if record["clickStatus"] == "target":
        target = 1
    listenerObjs[clickedLisIndex]["Target"] = target
    listenerObjs[clickedLisIndex]["Status"] = record["clickStatus"]
    listenerObjs[clickedLisIndex]["H"] = record["clickColH"]
    listenerObjs[clickedLisIndex]["S"] = record["clickColS"]
    listenerObjs[clickedLisIndex]["L"] = record["clickColL"]
    speakerObjs[clickedSpIndex]["Target"] = target
    speakerObjs[clickedSpIndex]["Status"] = record["clickStatus"]
    speakerObjs[clickedSpIndex]["H"] = record["clickColH"]
    speakerObjs[clickedSpIndex]["S"] = record["clickColS"]
    speakerObjs[clickedSpIndex]["L"] = record["clickColL"]

    target = 0
    if record["alt1Status"] == "target":
        target = 1
    listenerObjs[alt1LisIndex]["Target"] = target
    listenerObjs[alt1LisIndex]["Status"] = record["alt1Status"]
    listenerObjs[alt1LisIndex]["H"] = record["alt1ColH"]
    listenerObjs[alt1LisIndex]["S"] = record["alt1ColS"]
    listenerObjs[alt1LisIndex]["L"] = record["alt1ColL"]
    speakerObjs[alt1SpIndex]["Target"] = target
    speakerObjs[alt1SpIndex]["Status"] = record["alt1Status"]
    speakerObjs[alt1SpIndex]["H"] = record["alt1ColH"]
    speakerObjs[alt1SpIndex]["S"] = record["alt1ColS"]
    speakerObjs[alt1SpIndex]["L"] = record["alt1ColL"]

    target = 0
    if record["alt2Status"] == "target":
        target = 1
    listenerObjs[alt2LisIndex]["Target"] = target
    listenerObjs[alt2LisIndex]["Status"] = record["alt2Status"]
    listenerObjs[alt2LisIndex]["H"] = record["alt2ColH"]
    listenerObjs[alt2LisIndex]["S"] = record["alt2ColS"]
    listenerObjs[alt2LisIndex]["L"] = record["alt2ColL"]
    speakerObjs[alt2SpIndex]["Target"] = target
    speakerObjs[alt2SpIndex]["Status"] = record["alt2Status"]
    speakerObjs[alt2SpIndex]["H"] = record["alt2ColH"]
    speakerObjs[alt2SpIndex]["S"] = record["alt2ColS"]
    speakerObjs[alt2SpIndex]["L"] = record["alt2ColL"]

    state = dict()
    state["gameid"] = record["gameid"]
    state["roundNum"] = int(record["roundNum"])
    state["time"] = int(float(record["msgTime"])) - 1 # Hacky solution
    state["condition"] = record["condition"]
    state["targetD1Diff"] = int(record["targetD1Diff"])
    state["targetD2Diff"] = int(record["targetD2Diff"])
    state["D1D2Diff"] = int(record["D1D2Diff"])

    for i in range(len(listenerObjs)):
        for key in listenerObjs[i]:
            state["l" + key + "_" + str(i)] = listenerObjs[i][key]
            state["s" + key + "_" + str(i)] = speakerObjs[i][key]

        if listenerObjs[i]["Target"] == 1:
            for key in listenerObjs[i]:
                if key != "Target" and key != "Status":
                    state["lTarget" + key] = listenerObjs[i][key]
            state["lTargetIndex"] = i;

        if speakerObjs[i]["Target"] == 1:
            for key in speakerObjs[i]:
                if key != "Target" and key != "Status":
                    state["sTarget" + key] = speakerObjs[i][key]
            state["sTargetIndex"] = i;

    return state


def make_action_record(record):
    listenerObjs = [dict(), dict(), dict()]
    speakerObjs = [dict(), dict(), dict()]

    clickedLisIndex = int(record["clickLocL"]) - 1
    clickedSpIndex = int(record["clickLocS"]) - 1
    alt1LisIndex = int(record["alt1LocL"]) - 1
    alt1SpIndex = int(record["alt1LocS"]) - 1
    alt2LisIndex = int(record["alt2LocL"]) - 1
    alt2SpIndex = int(record["alt2LocS"]) - 1

    listenerObjs[clickedLisIndex]["Clicked"] = 1
    speakerObjs[clickedSpIndex]["Clicked"] = 1

    listenerObjs[alt1LisIndex]["Clicked"] = 0
    speakerObjs[alt1SpIndex]["Clicked"] = 0

    listenerObjs[alt2LisIndex]["Clicked"] = 0
    speakerObjs[alt2SpIndex]["Clicked"] = 0

    action = dict()
    action["gameid"] = record["gameid"]
    action["roundNum"] = int(record["roundNum"])
    action["time"] = int(float(record["clkTime"]))
    action["condition"] = record["condition"]
    action["lClickedIndex"] = clickedLisIndex
    action["sClickedIndex"] = clickedSpIndex
    action["outcome"] = record["outcome"]

    for i in range(len(listenerObjs)):
        for key in listenerObjs[i]:
            action["l" + key + "_" + str(i)] = listenerObjs[i][key]
            action["s" + key + "_" + str(i)] = speakerObjs[i][key]
        
    return action


def make_utterance_records(round_records):
    utterances = []

    for record in round_records:
        utterance = dict()
        utterance["gameid"] = record["gameid"]
        utterance["roundNum"] = int(record["roundNum"])
        utterance["time"] = int(float(record["msgTime"])) # Data has .0 after all times...
        utterance["sender"] = record["role"]
        utterance["contents"] = record["contents"]
        utterances.append(utterance)

    utterances.sort(key=lambda u: u["time"])
    return utterances

def process_game(game_records):
    states = []
    actions = []
    utterances = []
    for key, round_records in game_records.items():
        first_record = round_records[0]
        states.append(make_state_record(first_record))
        actions.append(make_action_record(first_record))
        utterances.extend(make_utterance_records(round_records))
    return states, actions, utterances


def process_games(game_record_dict):
    processed_games = dict()
    for key, value in game_record_dict.items():
        processed_games[key] = process_game(value)
    return processed_games


def output_csv(file_path, rows):
    fields = OrderedDict([(k, None) for k in rows[0].keys()])
    f = open(file_path, 'wb')
    try:
        writer = csv.DictWriter(f, delimiter=',', fieldnames=fields, quoting=csv.QUOTE_NONNUMERIC)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    finally:
        f.close()


def output_games(state_dir, action_dir, utterance_dir, games_to_state_action_utterances):
    for game, sau in games_to_state_action_utterances.items():
        print "Outputting " + game
        output_csv(os.path.join(state_dir, game), sau[0])
        output_csv(os.path.join(action_dir, game), sau[1])
        output_csv(os.path.join(utterance_dir, game), sau[2])


processed_games = process_games(read_messy_file(input_file_path))
output_games(output_state_dir, output_action_dir, output_utterance_dir, processed_games)
