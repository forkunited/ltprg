#!/bin/bash

ROOT_DIR=[PATH_TO_LTPRG_ROOT]

# e.g. src/test/py/ltprg/game/colorGrids/data/featurize_sua.py
FEATURIZE_SUA_SCRIPT=${ROOT_DIR}/src/test/games/colorGrids/data/featurize_sua.py

# e.g. examples/games/csv/colorGrids/raw/3
CSV_DATA_DIR_1=examples/games/csv/colorGrids/raw/1
CSV_DATA_DIR_2=examples/games/csv/colorGrids/raw/2
CSV_DATA_DIR_3=examples/games/csv/colorGrids/raw/3

# e.g. examples/games/json/colorGrids/raw/3
JSON_DATA_DIR_1=examples/games/json/colorGrids/raw/1
JSON_DATA_DIR_2=examples/games/json/colorGrids/raw/2
JSON_DATA_DIR_3=examples/games/json/colorGrids/raw/3

# e.g. examples/games/json/colorGrids/cleaning.txt
CLEANING_FILE=examples/games/json/colorGrids/raw/cleaning.txt

# e.g. examples/games/json/colorGrids/clean/
JSON_DATA_DIR_CLEAN=examples/games/json/colorGrids/clean
JSON_DATA_DIR_COLOR=examples/games/json/colorGrids/color/color3

# e.g. examples/games/json/colorGrids/clean_nlp
JSON_WITH_NLP_DATA_DIR=examples/games/json/colorGrids/clean_nlp
JSON_COLOR_WITH_NLP_DATA_DIR=examples/games/json/colorGrids/color/color3_nlp
JSON_MERGED_NLP_DATA_DIR=examples/games/json/colorGrids/merged/nlp

# e.g. examples/games/json/colorGrids/merged/sua_speaker_notarget
SUA_DATA_DIR_NO_TARGET=examples/games/json/colorGrids/merged/sua_speaker_notarget
SUA_DATA_DIR=examples/games/json/colorGrids/merged/sua_speaker

# e.g. examples/games/splits/colorGrids_merged
OLD_PARTITION_FILE=examples/games/splits/colorGrids_12_color_merged_34_80_33_10_33_10
PARTITION_FILE=examples/games/splits/colorGrids_merged

FEATURE_DIR=[PATH_TO_OUTPUT_FEATURIZED_DATA_DIRECTORY]

cd ${ROOT_DIR}

# Make JSON data
python ${ROOT_DIR}/src/test/py/ltprg/data/make_game_csv_to_json.py gameid roundNum ${CSV_DATA_DIR_1}/state,${CSV_DATA_DIR_1}/action,${CSV_DATA_DIR_1}/utterance StateColorGrid,ActionColorGrid,Utterance ${JSON_DATA_DIR_1} [TAB]
python ${ROOT_DIR}/src/test/py/ltprg/data/make_game_csv_to_json.py gameid roundNum ${CSV_DATA_DIR_2}/state,${CSV_DATA_DIR_2}/action,${CSV_DATA_DIR_2}/utterance StateColorGrid,ActionColorGrid,Utterance ${JSON_DATA_DIR_2} [TAB]
python ${ROOT_DIR}/src/test/py/ltprg/data/make_game_csv_to_json.py gameid roundNum ${CSV_DATA_DIR_3}/state,${CSV_DATA_DIR_3}/action,${CSV_DATA_DIR_3}/utterance StateColorGrid,ActionColorGrid,Utterance ${JSON_DATA_DIR_3} [TAB]

# Clean JSON data
python ${ROOT_DIR}/src/test/py/ltprg/game/colorGrids/data/clean.py ${CLEANING_FILE} ${JSON_DATA_DIR_1} ${JSON_DATA_DIR_2} ${JSON_DATA_DIR_3} ${JSON_DATA_DIR_CLEAN}

# Annotate JSON with NLP annotations (tokenize, etc)
python ${ROOT_DIR}/src/test/py/ltprg/data/annotate_json_nlp.py ${JSON_DATA_DIR_CLEAN} ${JSON_WITH_NLP_DATA_DIR}
python ${ROOT_DIR}/src/test/py/ltprg/data/annotate_json_nlp.py ${JSON_DATA_DIR_COLOR} ${JSON_COLOR_WITH_NLP_DATA_DIR}

# Merge grid and color data
python ${ROOT_DIR}/src/test/py/ltprg/game/colorGrids/data/merge_data.py ${JSON_COLOR_WITH_NLP_DATA_DIR} ${JSON_WITH_NLP_DATA_DIR} color3 grid3 ${JSON_MERGED_NLP_DATA_DIR}

# Make state-action-utterance format of the data (one JSON object per round)
python ${ROOT_DIR}/src/test/py/ltprg/data/make_sua.py UTTERANCES_SPEAKER ${JSON_MERGED_NLP_DATA_DIR} ${SUA_DATA_DIR_NO_TARGET} 1 0
python ${ROOT_DIR}/src/test/py/ltprg/game/colorGrids/data/annotate_with_targets.py ${SUA_DATA_DIR_NO_TARGET} ${SUA_DATA_DIR}

# Make data split 
python ${ROOT_DIR}/src/test/py/ltprg/data/make_split.py ${JSON_MERGED_NLP_DATA_DIR} ${PARTITION_FILE} 0.8 0.1 0.1 --maintain_partition_file ${OLD_PARTITION_FILE}

# Featurize
python ${FEATURIZE_SUA_SCRIPT} ${SUA_DATA_DIR} ${FEATURE_DIR} ${PARTITION_FILE} 3

