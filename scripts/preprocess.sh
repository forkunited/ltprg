#!/bin/bash

ROOT_DIR=[PATH_TO_LTPRG_ROOT]

# e.g. src/test/py/ltprg/game/colorGrids/data/featurize_sua.py
FEATURIZE_SUA_SCRIPT=[PATH_TO_GAME_SPECIFIC_SUA_FEATURIZATION_SCRIPT]

# e.g. examples/games/csv/colorGrids/1
CSV_DATA_DIR=[PATH_TO_INPUT_CSV_GAME_DATA]

# e.g. examples/games/json/colorGrids/clean
JSON_DATA_DIR=[PATH_TO_OUTPUT_JSON_GAME_DATA]

# e.g. examples/games/json/colorGrids/merged/nlp
JSON_WITH_NLP_DATA_DIR=[PATH_TO_OUTPUT_JSON_NLP_GAME_DATA]

# e.g. examples/games/json/colorGrids/merged/sua_speaker
SUA_DATA_DIR=[PATH_TO_OUTPUT_JSON_STATE_UTTERANCE_ACTION_DATA]

# e.g. examples/games/splits/colorGrids_merged
PARTITION_FILE=[PATH_TO_OUTPUT_PARTITION_FILE]

FEATURE_DIR=[PATH_TO_OUTPUT_FEATURIZED_DATA_DIRECTORY]

# Make JSON data
python ${ROOT_DIR}/src/test/py/ltprg/data/make_game_csv_to_json.py gameid roundNum ${CSV_DATA_DIR}/state,${CSV_DATA_DIR}/action,${CSV_DATA_DIR}/utterance StateColorGrid,ActionColorGrid,Utterance ${JSON_DATA_DIR} [TAB]

# Annotate JSON with NLP annotations (tokenize, etc)
python ${ROOT_DIR}/src/test/py/ltprg/data/annotate_json_nlp.py ${JSON_DATA_DIR} ${JSON_WITH_NLP_DATA_DIR}

# Make state-action-utterance format of the data (one JSON object per round)
python ${ROOT_DIR}/src/test/py/ltprg/data/make_sua.py UTTERANCES_SPEAKER ${JSON_DATA_DIR} ${SUA_DATA_DIR} 1 0

# Make data split 
python ${ROOT_DIR}/src/test/py/ltprg/data/make_split.py ${JSON_DATA_DIR} ${PARTITION_FILE}

# Featurize
python ${FEATURIZE_SUA_SCRIPT} ${SUA_DATA_DIR} ${FEATURE_DIR} ${PARTITION_FILE}
