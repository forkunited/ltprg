import sys
import numpy as np
from mung.data import DataSet, Datum

data_dir = sys.argv[1]
output_dir = sys.argv[2]
grid_dimension = int(sys.arg[3])

D = DataSet.load(data_dir, id_key="gameid")
grid_data = []
for datum in D:
    color_dict = datum.to_dict()
    grid_dict = dict()
    grid_dict["gameid"] = color_dict["gameid"]
    grid_dict["records"] = []
    for record in color_dict["records"]:
        grid_record = dict()
        grid_record["roundNum"] = record["roundNum"]
        grid_record["events"] = []
        for event in record["events"]:
            grid_event = dict()
            if event["type"] == "StateColor":
                grid_event["type"] = "StateColorGrid"
                grid_event["eventType"] = "state"
                grid_event["time"] = event["time"]

                obj0 = { "gridDimension" : grid_dimension, "cellLength" : -1, \
                         "shapes" : \
                            [ \
                                { "color" : [ int(event["sH_0"]), int(event["sS_0"]), int(event["sL_0"])]} \
                                for i in range(grid_dimension*grid_dimension) \
                            ] \
                        }
                obj1 = { "gridDimension" : grid_dimension, "cellLength" : -1, \
                         "shapes" : \
                            [ \
                                { "color" : [ int(event["sH_1"]), int(event["sS_1"]), int(event["sL_1"])]} \
                                for i in range(grid_dimension*grid_dimension) \
                            ] \
                        }
                obj2 = { "gridDimension" : grid_dimension, "cellLength" : -1, \
                         "shapes" : \
                            [ \
                                { "color" : [ int(event["sH_2"]), int(event["sS_2"]), int(event["sL_2"])]} \
                                for i in range(grid_dimension*grid_dimension) \
                            ] \
                        }

                grid_event["state"] = dict()
                grid_event["state"]["listenerOrder"] = [0,1,2]
                grid_event["state"]["speakerOrder"] = [0,1,2]
                grid_event["state"]["target"] = int(event["sTargetIndex"])
                grid_event["state"]["objs"] = [obj0,obj1,obj2]
                grid_event["state"]["condition"] = { "name" : event["condition"].upper(), "gridDimension" : grid_dimension }

                if event["lStatus_0"] == event["sStatus_0"]:
                    grid_event["state"]["listenerOrder"][0] = 0
                elif event["lStatus_0"] == event["sStatus_1"]:
                    grid_event["state"]["listenerOrder"][0] = 1
                else:
                    grid_event["state"]["listenerOrder"][0] = 2

                if event["lStatus_1"] == event["sStatus_0"]:
                    grid_event["state"]["listenerOrder"][1] = 0
                elif event["lStatus_1"] == event["sStatus_1"]:
                    grid_event["state"]["listenerOrder"][1] = 1
                else:
                    grid_event["state"]["listenerOrder"][1] = 2

                if event["lStatus_2"] == event["sStatus_0"]:
                    grid_event["state"]["listenerOrder"][2] = 0
                elif event["lStatus_2"] == event["sStatus_1"]:
                    grid_event["state"]["listenerOrder"][2] = 1
                else:
                    grid_event["state"]["listenerOrder"][2] = 2
            elif event["type"] == "ActionColor":
                grid_event["type"] = "ActionColorGrid"
                grid_event["eventType"] = "action"
                grid_event["time"] = event["time"]

                grid_event["action"] = dict()
                grid_event["action"]["lClicked"] = int(event["lClickedIndex"])
                grid_event["action"]["mouseX"] = -1
                grid_event["action"]["mouseY"] = -1
                grid_event["action"]["condition"] = { "name" : event["condition"].upper() }
            elif event["type"] == "Utterance":
                grid_event = event
                grid_event["eventType"] = "utterance"
            grid_record["events"].append(grid_event)
        grid_dict["records"].append(grid_record)

    grid_datum = Datum(properties=grid_dict, id_key="gameid")
    grid_data.append(grid_datum)

D_grid = DataSet(data=grid_data, id_key="gameid")
D_grid.save(output_dir)
