#!/bin/bash

export CUDA_VISIBLE_DEVICES=${1}
export GPU=1
export SEED=${1}
export EVAL_TEST=0
export ROOT_DIR=[PATH_TO_LTPRG_DIRECTORY]
export CONFIG_DIR=${ROOT_DIR}/config/game/colorGrids
export SCRIPT=${ROOT_DIR}/src/test/py/ltprg/game/colorGrids/model/learn_RSA.py
export ENV_FILE=${ROOT_DIR}/env_local.json
export DATA_CONFIG_FILE=${CONFIG_DIR}/data/cgmerged_cfirst_clean.json
export LEARN_CONFIG_FILE=${CONFIG_DIR}/learn/rsa/cgmerged_src_color3_data.json
export TRAIN_EVALS_CONFIG_FILE=${CONFIG_DIR}/eval/rsa/cgmerged_train_src_color3.json
export DEV_EVALS_CONFIG_FILE=${CONFIG_DIR}/eval/rsa/cgmerged_dev_src_color3.json
export TEST_EVALS_CONFIG_FILE=${CONFIG_DIR}/eval/rsa/cgmerged_test_src_color3.json
export OUTPUT_DIR=[PATH_TO_OUTPUT_DIRECTORY]

cd ${ROOT_DIR}

sizes=( 250 500 1000 2500 5000 )
for sz in "${sizes[@]}"
do
    export TRAIN_DATA=train_src_color3
    export TRAIN_SIZE=${sz}
    export S0_MODEL=[PATH_TO_S0_MODEL]

    export JOB_ID=l0_onlycolor_${TRAIN_SIZE}
    export LEARN_CONFIG_FILE=${CONFIG_DIR}/learn/rsa/cgmerged_src_grid3_data.json
    export MODEL_CONFIG_FILE=${CONFIG_DIR}/model/rsa/onlycolor/l0.json
    python ${SCRIPT} ${JOB_ID} ${ENV_FILE} ${DATA_CONFIG_FILE} ${MODEL_CONFIG_FILE} ${LEARN_CONFIG_FILE} ${TRAIN_EVALS_CONFIG_FILE} ${DEV_EVALS_CONFIG_FILE} ${TEST_EVALS_CONFIG_FILE} ${OUTPUT_DIR} --gpu ${GPU} --seed ${SEED} --eval_test ${EVAL_TEST} --train_data ${TRAIN_DATA} --train_data_size ${TRAIN_SIZE} --s0_model ${S0_MODEL}

    export JOB_ID=l1_onlycolor_${TRAIN_SIZE}
    export LEARN_CONFIG_FILE=${CONFIG_DIR}/learn/rsa/cgmerged_src_grid3_data.json
    export MODEL_CONFIG_FILE=${CONFIG_DIR}/model/rsa/onlycolor/l1.json
    python ${SCRIPT} ${JOB_ID} ${ENV_FILE} ${DATA_CONFIG_FILE} ${MODEL_CONFIG_FILE} ${LEARN_CONFIG_FILE} ${TRAIN_EVALS_CONFIG_FILE} ${DEV_EVALS_CONFIG_FILE} ${TEST_EVALS_CONFIG_FILE} ${OUTPUT_DIR} --gpu ${GPU} --seed ${SEED} --eval_test ${EVAL_TEST} --train_data ${TRAIN_DATA} --train_data_size ${TRAIN_SIZE} --s0_model ${S0_MODEL}

done

export S0_MODEL=[PATH_TO_S0_MODEL]

export JOB_ID=l0_onlycolor_full
export LEARN_CONFIG_FILE=${CONFIG_DIR}/learn/rsa/cgmerged_src_grid3.json
export MODEL_CONFIG_FILE=${CONFIG_DIR}/model/rsa/onlycolor/conv_l0.json
python ${SCRIPT} ${JOB_ID} ${ENV_FILE} ${DATA_CONFIG_FILE} ${MODEL_CONFIG_FILE} ${LEARN_CONFIG_FILE} ${TRAIN_EVALS_CONFIG_FILE} ${DEV_EVALS_CONFIG_FILE} ${TEST_EVALS_CONFIG_FILE} ${OUTPUT_DIR} --gpu ${GPU} --seed ${SEED} --eval_test ${EVAL_TEST} --s0_model ${S0_MODEL}

export JOB_ID=l1_onlycolor_full
export LEARN_CONFIG_FILE=${CONFIG_DIR}/learn/rsa/cgmerged_src_grid3.json
export MODEL_CONFIG_FILE=${CONFIG_DIR}/model/rsa/onlycolor/conv_l1.json
python ${SCRIPT} ${JOB_ID} ${ENV_FILE} ${DATA_CONFIG_FILE} ${MODEL_CONFIG_FILE} ${LEARN_CONFIG_FILE} ${TRAIN_EVALS_CONFIG_FILE} ${DEV_EVALS_CONFIG_FILE} ${TEST_EVALS_CONFIG_FILE} ${OUTPUT_DIR} --gpu ${GPU} --seed ${SEED} --eval_test ${EVAL_TEST} --s0_model ${S0_MODEL}

