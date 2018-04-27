from mung.torch_ext.eval import Loss
from ltprg.model.seq import DataParameter, SequenceModelNoInput
from ltprg.model.seq import VariableLengthNLLLoss

# Expects config of the form:
# {
#   data_parameter : {
#     seq : [SEQUENCE PARAMETER NAME]
#     input : [INPUT PARAMETER NAME]
#   }
#   name : [ID FOR MODEL]
#   arch_type : [SequenceModelNoInput]
#   dropout : [DROPOUT]
#   rnn_layers : [RNN_LAYERS]
#   rnn_size : [SIZE OF RNN HIDDEN LAYER]
#   embedding_size : [EMBEDDING_SIZE]
#   rnn_type : [RNN TYPE]
# }
def load_seq_model(config, D, gpu=False):
    data_parameter = DataParameter.make(**config["data_parameter"])
    seq_field = data_parameter["seq"]
    utterance_size = D[seq_field].get_matrix(0).get_feature_set().get_token_count()
  
    dropout = float(config["dropout"])
    rnn_layers = int(config["rnn_layers"])
    rnn_size = int(config["rnn_size"])
    embedding_size = int(config["embedding_size"])
    rnn_type = config["rnn_type"]

    model = SequenceModelNoInput(config["name"], utterance_size, \
        embedding_size, rnn_size, rnn_layers, dropout=dropout, rnn_type=rnn_type)

    return data_parameter, model

# Expects config of the form:
# {
#    data_parameter : {
#     seq : [SEQUENCE PARAMETER NAME]
#     input : [INPUT PARAMETER NAME]
#   },
#   evaluations : [
#    name : [NAME FOR EVALUATION]
#    type : (VariableLengthNLLLoss)
#    data : [NAME OF DATA SUBSET]
#    (Optional) data_size : [SIZE OF RANDOM SUBET OF DATA TO TAKE]
#   ]
# }
def load_evaluations(config, D, gpu=False):
    data_parameter = DataParameter.make(**config["data_parameter"])
    evaluations = []

    loss_criterion = VariableLengthNLLLoss(norm_dim=True)
    if gpu:
        loss_criterion = loss_criterion.cuda()

    for eval_config in config["evaluations"]:
        data = D[eval_config["data"]]
        if "data_size" in config:
            data = data.get_random_subset(int(eval_config["data_size"]))

        if eval_config["type"] == "VariableLengthNLLLoss":
            loss = Loss(eval_config["name"], data, data_parameter, loss_criterion, norm_dim=True)
            evaluations.append(loss)
        else:
            raise ValueError("Invalid seq evaluation type in config (" + str(eval_config["type"]))
    return evaluations
