"""
Constructs a data set of state-utterance-action examples from reference game
data.  Each example consists of a single game state, an action that occurred
in that state, and a sequence of utterances that occurred before the action.

Args:
    utterance_mode (:obj:`str`): Value determing which utterances are included
        in each state-utterance-action example.  Possible values are:
            UTTERANCES_ALL : Keep all utterances
            UTTERANCES_SPEAKER : Include only speaker utterances
            UTTERANCES_LISTENER : Include only listener utterances
            UTTERANCES_SPEAKER_LAST : Include the final speaker utterance
                before an action
            UTTERANCES_LISTENER_LAST : Include the final listener utterance
                before an action
    input_data_dir (:obj:`str`): Input directory containing JSON game data
    output_data_dir (:obj:`str`): Output directory in which to store
        state-utterance-action data
"""

import sys
import time
import numpy as np
from mung.data import DataSet, Datum

UTTERANCES_ALL = "UTTERANCES_ALL"
UTTERANCES_SPEAKER = "UTTERANCES_SPEAKER"
UTTERANCES_LISTENER = "UTTERANCES_LISTENER"
UTTERANCES_SPEAKER_LAST = "UTTERANCES_SPEAKER_LAST"
UTTERANCES_LISTENER_LAST = "UTTERANCES_LISTENER_LAST"

utterance_mode = sys.argv[1]
input_data_dir = sys.argv[2]
output_data_dir = sys.argv[3]
exclude_missing_utts = bool(sys.argv[4])

np.random.seed(1) # Ensures data loaded in same order each time

game_data = DataSet.load(input_data_dir, id_key="gameid")
sua_datums = []
hanging_speaker_utts = 0
hanging_listener_utts = 0
for i in range(game_data.get_size()):
    datum = game_data.get(i)
    game_id = datum.get("gameid")
    for record in datum.get("records"):
        roundNum = record["roundNum"]
        sua_index = 0
        cur_utts = []
        cur_state = None
        has_speaker = False
        has_listener = False
        prev_utt_count = 0
        for event in record["events"]:
            if event["type"].startswith("State"):
                cur_state = event
            elif event["type"].startswith("Utterance"):
                cur_utts.append(event)
                if event["sender"] == "speaker":
                    has_speaker = True
                elif event["sender"] == "listener":
                    has_listener = True
            else:
                cur_action = event

                sua_properties = dict()
                sua_properties["id"] = game_id + "_" + str(roundNum) + "_" + str(sua_index)
                sua_properties["gameid"] = game_id
                sua_properties["roundNum"] = roundNum
                sua_properties["sua"] = sua_index
                sua_properties["state"] = cur_state

                if utterance_mode == UTTERANCES_ALL:
                    sua_properties["utterances"] = cur_utts
                elif utterance_mode == UTTERANCES_SPEAKER:
                    sua_properties["utterances"] = [cur_utt for cur_utt in cur_utts if cur_utt["sender"] == "speaker"]
                elif utterance_mode == UTTERANCES_LISTENER:
                    sua_properties["utterances"] = [cur_utt for cur_utt in cur_utts if cur_utt["sender"] == "listener"]
                elif utterance_mode == UTTERANCES_SPEAKER_LAST and has_speaker:
                    sua_properties["utterances"] = [next(cur_utt for cur_utt in reversed(cur_utts) if cur_utt["sender"] == "speaker")]
                elif utterance_mode == UTTERANCES_LISTENER_LAST and has_listener:
                    sua_properties["utterances"] = [next(cur_utt for cur_utt in reversed(cur_utts) if cur_utt["sender"] == "listener")]
                else:
                    sua_properties["utterances"] = []

                sua_properties["action"] = cur_action

                prev_utt_count = len(sua_properties["utterances"])

                if (not exclude_missing_utts) or len(sua_properties["utterances"]) > 0:
                    sua_datums.append(Datum(properties=sua_properties))

                sua_index += 1
                cur_state = None
                cur_utts = []
                has_speaker = False
                has_listener = False

        if len(cur_utts) > 0 and prev_utt_count == 0:
            hanging_speaker_utts += len([cur_utt for cur_utt in cur_utts if cur_utt["sender"] == "speaker"])
            hanging_listener_utts += len([cur_utt for cur_utt in cur_utts if cur_utt["sender"] == "listener"])

print "NOTE: " + str(hanging_listener_utts) + " hanging listener utterances and " + str(hanging_speaker_utts) + " hanging speaker utterances"

sua_data = DataSet(data=sua_datums)
sua_data.save(output_data_dir)
