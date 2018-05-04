from torch.nn import NLLLoss
from mung.torch_ext.eval import Loss
from ltprg.model.rsa import DataParameter, DistributionType, RSA, RSADistributionAccuracy, PriorView
from ltprg.model.prior import UniformIndexPriorFn, SequenceSamplingPriorFn
from ltprg.model.meaning import MeaningModel, MeaningModelIndexedWorldSequentialUtterance, SequentialUtteranceInputType
from ltprg.model.obs import ObservationModel, ObservationModelReorderedSequential
from ltprg.model.seq import SequenceModel, SequenceModelNoInput, SequenceModelInputToHidden
from ltprg.model.seq_heuristic import HeuristicL0


# Expects config of the form:
# {
#   data_parameter : {
#     utterance : [UTTERANCE DATA PARAMETER NAME]
#     L_world : [L_WORLD DATA PARAMETER NAME]
#     L_observation : [L_OBSERVATION DATA PARAMETER NAME]
#     S_world : [S_WORLD DATA PARAMETER NAME]
#     S_observation : [S_OBSERVATION DATA PARAMETER NAME]
#     mode : [S OR L DISTRIBUTION]
#     utterance_seq :[INDICATOR OF WHETHER UTTERANCE IS SEQUENTIAL]
#   },
#   utterance_prior : {
#     seq_model_path : [PATH TO STORED UTTERANCE PRIOR SEQUENCE MODEL]
#     heuristic : [NAME OF HEURISTIC FOR GUIDING UTTERANCE SAMPLING (L0|None)]
#     parameters : {
#       training_mode : [SAMPLING MODE DURING TRAINING (FORWARD|BEAM|SMC)] 
#       eval_mode : [SAMPLING MODE DURING EVALUATION (FORWARD|BEAM|SMC)]
#       samples_per_input : [SAMPLES PER INPUT WITHIN OBSERVATION]
#       uniform : [INDICATOR OF WHETHER SAMPLES ARE UNIFORM OR WEIGHTED]
#       training_input_mode : [Mode for sampling during training (IGNORE_TRUE_WORLD|ONLY_TRUE_WORLD|None)]                                  
#       sample_length : [Length of samples to take]
#       n_before_heuristic : [Samples prior to applying heuristic in forward sampling]
#     }
#   },
#   world_prior : {
#     support_size : [NUMBER OF WORLDS IN SUPPORT]
#   },
#   meaning_fn : {
#     seq_model : {
#       (Optional) model_path : [PATH TO EXISTING MEANING MODEL]
#       bidirectional : [INDICATES WHETHER SEQUENCE MDOEL IS BIDIRECTIONAL]
#       dropout : [DROPOUT]
#       rnn_layers : [RNN_LAYERS]
#       rnn_size : [SIZE OF RNN HIDDEN LAYER]
#       embedding_size : [EMBEDDING_SIZE]
#       rnn_type : [RNN TYPE],
#       conv_input : [INDICATOR OF WHETHER OR NOT TO CONVOLVE THE INPUT]
#       conv_kernel : [KERNEL SIZE FOR CONVOLUTION]
#       conv_stride : [STRIDE LENGTH FOR CONVOLUTION]
#     }
#   },
#   (Optional) observation_fn : {
#     seq_model : {
#       (Optional) model_path : [PATH TO EXISTING OBSERVATION MODEL]
#       bidirectional : [INDICATES WHETHER SEQUENCE MDOEL IS BIDIRECTIONAL]
#       dropout : [DROPOUT]
#       rnn_layers : [RNN_LAYERS]
#       rnn_size : [SIZE OF RNN HIDDEN LAYER]
#       embedding_size : [EMBEDDING_SIZE]
#       rnn_type : [RNN TYPE]
#       non_emb : [INDICATES WHETHER INPUTS SHOULD BE EMBEDDED PRIOR TO RNN]
#     }
#   },
#   training_level : [RSA LEVEL AT WHICH TO TRAIN]
#   alpha : [ALPHA RATIONALITY PARAMETER]
# }
def load_rsa_model(config, D, gpu=False):
    data_parameter = DataParameter.make(**config["data_parameter"])
    utterance_field = config["data_parameter"]["utterance"]
    observation_field = config["data_parameter"]["L_observation"]
    if config["data_parameter"]["mode"] == DistributionType.S:
        observation_field = config["data_parameter"]["S_observation"]
    world_support_size = int(config["world_prior"]["support_size"])

    # Optionally setup observation function if observation is sequential
    observation_fn = None
    world_input_size = None
    if "observation_fn" in config:
        obs_config = config["observation_fn"]["seq_model"]
        world_input_size = int(obs_config["rnn_size"]) * int(obs_config["rnn_layers"])
        if bool(int(obs_config["bidirectional"])):
            world_input_size *= 2

        if "model_path" in obs_config:
            observation_fn = ObservationModel.load(obs_config["model_path"])
        else:
            observation_size = D[observation_field].get_matrix(0).get_feature_set().get_token_count()
            seq_observation_model = SequenceModelNoInput("Observation", observation_size,
                int(obs_config["embedding_size"]), int(obs_config["rnn_size"]), int(obs_config["rnn_layers"]), 
                dropout=float(obs_config["dropout"]), rnn_type=obs_config["rnn_type"], bidir=bool(int(obs_config["bidirectional"])), 
                non_emb=bool(int(obs_config["non_emb"])))
            observation_fn = ObservationModelReorderedSequential(world_input_size, world_support_size, seq_observation_model)
    else:
        world_input_size = D[observation_field].get_feature_set().get_token_count() / world_support_size

    # Setup meaning function
    meaning_config = config["meaning_fn"]["seq_model"]
    meaning_fn = None
    if "model_path" in meaning_config:
        meaning_fn = MeaningModel.load(meaning_config["model_path"])
    else:
        utterance_size = D[utterance_field].get_matrix(0).get_feature_set().get_token_count()
        conv_input = False
        conv_kernel = 1
        conv_stride = 1
        if "conv_input" in meaning_config:
            conv_input = bool(int(meaning_config["conv_input"]))
            conv_kernel = int(meaning_config["conv_kernel"])
            conv_stride = int(meaning_config["conv_stride"])

        seq_meaning_model = SequenceModelInputToHidden("Meaning", utterance_size, world_input_size, \
        int(meaning_config["embedding_size"]), int(meaning_config["rnn_size"]), int(meaning_config["rnn_layers"]), \
        dropout=float(meaning_config["dropout"]), rnn_type=meaning_config["rnn_type"], 
        bidir=bool(int(meaning_config["bidirectional"])), input_layers=1,\
        conv_input=conv_input, conv_kernel=conv_kernel,conv_stride=conv_stride)
        meaning_fn = MeaningModelIndexedWorldSequentialUtterance(world_input_size, seq_meaning_model, \
            input_type=SequentialUtteranceInputType.IN_SEQ)

    # Setup world prior
    world_prior_fn = UniformIndexPriorFn(world_support_size, on_gpu=gpu, unnorm=False)

    # Setup utterance prior
    heuristic = None
    if config["utterance_prior"]["heuristic"] == "L0":
        heuristic = HeuristicL0(world_prior_fn, meaning_fn, soft_bottom=False)

    seq_prior_model = SequenceModel.load(config["utterance_prior"]["seq_model_path"])

    utterance_prior_params = dict(config["utterance_prior"]["parameters"])
    utterance_prior_params["seq_length"] = D[utterance_field].get_feature_seq_set().get_size()
    utterance_prior_params["heuristic"] = heuristic
    utterance_prior_fn = SequenceSamplingPriorFn(seq_prior_model, world_input_size, **utterance_prior_params)
    
    rsa_model = RSA.make(data_parameter.get_mode() + "_" + str(config["training_level"]), \
                data_parameter.get_mode(), config["training_level"], meaning_fn, world_prior_fn, 
                utterance_prior_fn, L_bottom=True, soft_bottom=False, 
                alpha=config["alpha"], observation_fn=observation_fn)
    if gpu:
        rsa_model = rsa_model.cuda()

    return data_parameter, rsa_model

# Expects config of the form:
# {
#    data_parameter : {
#     utterance : [UTTERANCE DATA PARAMETER NAME]
#     L_world : [L_WORLD DATA PARAMETER NAME]
#     L_observation : [L_OBSERVATION DATA PARAMETER NAME]
#     S_world : [S_WORLD DATA PARAMETER NAME]
#     S_observation : [S_OBSERVATION DATA PARAMETER NAME]
#     mode : [S OR L DISTRIBUTION]
#     utterance_seq :[INDICATOR OF WHETHER UTTERANCE IS SEQUENTIAL]
#   },
#   evaluations : [
#    name : [NAME FOR EVALUATION]
#    type : (NLLLoss, RSADistributionAccuracy, PriorView)
#    data : [NAME OF DATA SUBSET]
#    (Optional) data_size : [SIZE OF RANDOM SUBET OF DATA TO TAKE]
#    parameters : {
#      (RSADistributionAccuracy) dist_level : [RSA DISTRIBUTION LEVEL TO EVALUATE]
#      (RSADistributionAccuracy) dist_type : [TYPE OF RSA DISTRIBUTION (L|S)]
#
#      (PriorView) output_dir : [DIRECTORY IN WHICH OUTPUT FOR CURRENT JOB IS STORED]
#    }
#   ]
# }
def load_evaluations(config, D, gpu=False):
    data_parameter = DataParameter.make(**config["data_parameter"])
    evaluations = []

    loss_criterion = NLLLoss(size_average=False)
    if gpu:
        loss_criterion = loss_criterion.cuda()

    for eval_config in config["evaluations"]:
        data = D[eval_config["data"]]
        if "data_size" in eval_config:
            data = data.get_random_subset(int(eval_config["data_size"]))

        if eval_config["type"] == "NLLLoss":
            loss = Loss(eval_config["name"], data, data_parameter, loss_criterion)
            evaluations.append(loss)
        elif eval_config["type"] == "RSADistributionAccuracy":
            acc = RSADistributionAccuracy(eval_config["name"], int(eval_config["parameters"]["dist_level"]), \
                eval_config["parameters"]["dist_type"], data, data_parameter)
            evaluations.append(acc)
        elif eval_config["type"] == "PriorView":
            priorv = PriorView(eval_config["name"], data, data_parameter, eval_config["parameters"]["output_dir"])
            evaluations.append(priorv)
        else:
            raise ValueError("Invalid RSA evaluation type in config (" + str(eval_config["type"]))
    return evaluations
