import sys
import ast
import numpy as np
from fixed_alternatives_set_models import  (
    train_model,
    load_json,
    FASM_ERSA_CTS,
    FASM_NNWC_CTS,
    FASM_NNWOC_CTS
)
from basic_model import ModelType, EmbeddingType

# args should be in order:
#   model_name
#   hidden_szs
#   hiddens_nonlinearity
#   train_data_fname
#   validation_data_fname
#   test_data_fname
#   weight_decay
#   learning_rate
#   alpha
#   cost_weight
#   save_path

def parse_args_and_run():
    print sys.argv
    assert len(sys.argv) == 12

    model_name = sys.argv[1]
    hidden_szs = ast.literal_eval(sys.argv[2])
    hiddens_nonlinearity = sys.argv[3]
    train_data_fname = sys.argv[4]
    validation_data_fname = sys.argv[5]
    test_data_fname = sys.argv[6]
    weight_decay = ast.literal_eval(sys.argv[7])
    learning_rate = ast.literal_eval(sys.argv[8])
    alpha = ast.literal_eval(sys.argv[9])
    cost_weight = ast.literal_eval(sys.argv[10])
    save_path = sys.argv[11]

    # load stuff
    train_data = load_json(train_data_fname)
    validation_data = load_json(validation_data_fname)
    test_data = load_json(test_data_fname)

    # fixed params
    data_rt = 'synthetic_data/'
    d = load_json(data_rt + 'true_lexicon.json')
    utt_set_sz = len(d)
    obj_set_sz = len(d['0'])
    utt_dict = load_json(data_rt + 'utt_inds_to_names.JSON')
    obj_dict = load_json(data_rt + 'obj_inds_to_names.JSON')
    cost_dict = load_json(data_rt + 'costs_by_utterance.JSON')
    true_lexicon  = load_json(data_rt + 'true_lexicon.JSON')
    true_lexicon = np.array([true_lexicon[str(k)] for k in range(utt_set_sz)]) + 10e-06
    obj_embedding_type = EmbeddingType.ONE_HOT
    should_visualize = False

    rsa_params = RSAParams(
        alpha=alpha,
        cost_weight=cost_weight,
        cost_dict=cost_dict,
        gold_standard_lexicon=true_lexicon
    )

    save_path_full = '{}/{}'.format(save_path, '/'.join(hidden_szs))

    if m == 'fasm_ersa':
        model = FASM_ERSA(
            model_name='fasm_ersa',
            model_type=ModelType.to_string(ModelType.ERSA),
            hidden_szs=hidden_szs,
            hiddens_nonlinearity='relu',
            utt_set_sz=utt_set_sz,
            obj_set_sz=obj_set_sz,
            obj_embedding_type=obj_embedding_type,
            utt_dict=utt_dict,
            obj_dict=obj_dict,
            weight_decay=weight_decay,
            learning_rate=learning_rate,
            rsa_params=rsa_params,
            save_path=save_path_full
        )
    elif m == 'fasm_nnwc':
        model = FASM_NNWC(
            model_name=m,
            model_type=ModelType.to_string(ModelType.NNWC),
            hidden_szs=hidden_szs,
            hiddens_nonlinearity='relu',
            utt_set_sz=utt_set_sz,
            obj_set_sz=obj_set_sz,
            obj_embedding_type=obj_embedding_type,
            utt_dict=utt_dict,
            obj_dict=obj_dict,
            weight_decay=weight_decay,
            learning_rate=learning_rate,
            rsa_params=rsa_params,
            save_path=save_path_full
        )
    elif m == 'fasm_nnwoc':
        model = FASM_NNWOC(
            model_name=m,
            model_type=ModelType.to_string(ModelType.NNWOC),
            hidden_szs=hidden_szs,
            hiddens_nonlinearity='relu',
            utt_set_sz=utt_set_sz,
            obj_set_sz=obj_set_sz,
            obj_embedding_type=obj_embedding_type,
            utt_dict=utt_dict,
            obj_dict=obj_dict,
            weight_decay=weight_decay,
            learning_rate=learning_rate,
            rsa_params=rsa_params,
            save_path=save_path_full
        )
    elif m == 'fasm_ersa_cts':
        model = FASM_ERSA_CTS(
            model_name=m,
            model_type=ModelType.to_string(ModelType.ERSA),
            hidden_szs=hidden_szs,
            hiddens_nonlinearity='relu',
            utt_set_sz=utt_set_sz,
            obj_set_sz=obj_set_sz,
            obj_embedding_type=obj_embedding_type,
            utt_dict=utt_dict,
            obj_dict=obj_dict,
            weight_decay=weight_decay,
            learning_rate=learning_rate,
            rsa_params=rsa_params,
            save_path=save_path_full
        )
    elif m == 'fasm_nnwc_cts':
        model = FASM_NNWC_CTS(
            model_name=m,
            model_type=ModelType.to_string(ModelType.NNWC),
            hidden_szs=hidden_szs,
            hiddens_nonlinearity='relu',
            utt_set_sz=utt_set_sz,
            obj_set_sz=obj_set_sz,
            obj_embedding_type=obj_embedding_type,
            utt_dict=utt_dict,
            obj_dict=obj_dict,
            weight_decay=weight_decay,
            learning_rate=learning_rate,
            rsa_params=rsa_params,
            save_path=save_path_full
        )
    elif m == 'fasm_nnwoc_cts':
        model = FASM_NNWOC_CTS(
            model_name=m,
            model_type=ModelType.to_string(ModelType.NNWOC),
            hidden_szs=hidden_szs,
            hiddens_nonlinearity='relu',
            utt_set_sz=utt_set_sz,
            obj_set_sz=obj_set_sz,
            obj_embedding_type=obj_embedding_type,
            utt_dict=utt_dict,
            obj_dict=obj_dict,
            weight_decay=weight_decay,
            learning_rate=learning_rate,
            rsa_params=rsa_params,
            save_path=save_path_full
        )
    else:
        raise Exception("Unrecognized Model Requested.")

    # Example
    train_model(
        model=model,
        train_data=train_data,
        validation_data=validation_data,
        should_visualize=should_visualize,
        save_path=save_path_full
    )

if __name__=='__main__':
    parse_args_and_run()