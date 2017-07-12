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

np.random.seed(1) # Ensures data loaded in same order each time

game_data = DataSet.load(input_data_dir, id_key="gameid")
sua_datums = []
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
                
                sua_datums.append(Datum(properties=sua_properties))
                
                sua_index += 1
                cur_state = None
                cur_utts = []
                has_speaker = False
                has_listener = False

sua_data = DataSet(data=sua_datums)
sua_data.save(output_data_dir)

