import torch
import torch.nn as nn
import mung.torch_ext.eval
from torch.autograd import Variable
from torch.nn import NLLLoss
from ltprg.model.dist import Categorical
from ltprg.model.prior import UniformIndexPriorFn, SequenceSamplingPriorFn
from ltprg.model.meaning import MeaningModel, MeaningModelIndexedWorldSequentialUtterance, SequentialUtteranceInputType
from ltprg.model.obs import ObservationModel, ObservationModelReorderedSequential
from ltprg.model.seq import SequenceModel, SequenceModelNoInput, SequenceModelInputToHidden
from ltprg.model.seq_heuristic import HeuristicL0
from mung.torch_ext.eval import DistributionAccuracy, Loss

EPSILON=1e-9 # Note, was 1e-6 on color, and not in normalize_rows back then

class DistributionType:
    L = "L"
    S = "S"

class DataParameter:
    UTTERANCE = "utterance"
    WORLD = "world"
    OBSERVATION = "observation"

    def __init__(self, utterance, L_world, L_observation, S_world, S_observation, mode=DistributionType.L, utterance_seq=False):
        self._utterance = utterance
        self._L_world = L_world
        self._L_observation = L_observation
        self._S_world = S_world
        self._S_observation = S_observation
        self._mode = mode
        self._utterance_seq = utterance_seq

    def __getitem__(self, key):
        if self._mode == DistributionType.L:
            if key == DataParameter.UTTERANCE:
                return self._utterance
            elif key == DataParameter.WORLD:
                return self._L_world
            elif key == DataParameter.OBSERVATION:
                return self._L_observation
            elif key == mung.torch_ext.eval.DataParameter.TARGET:
                return self._L_world
        else:
            if key == DataParameter.UTTERANCE:
                return self._utterance
            elif key == DataParameter.WORLD:
                return self._S_world
            elif key == DataParameter.OBSERVATION:
                return self._S_observation
            elif key == mung.torch_ext.eval.DataParameter.TARGET:
                return self._utterance

    def is_utterance_seq(self):
        return self._utterance_seq

    def get_mode(self):
        return self._mode

    def to_mode(self, mode):
        return DataParameter(self._utterance, self._L_world, self._L_observation, self._S_world, self._S_observation, mode=mode, utterance_seq=self._utterance_seq)

    @staticmethod
    def make(utterance="utterance", L_world="world", L_observation="observation",
        S_world="world", S_observation="observation", mode=DistributionType.L, utterance_seq=False):
        return DataParameter(utterance, L_world, L_observation, S_world, S_observation, mode=mode, utterance_seq=utterance_seq)

def _normalize_rows(t, softmax=False):
    if not softmax:
        row_sums = torch.sum(t, len(t.size())-1, keepdim=True) + EPSILON
        #return torch.exp(torch.log(t)-torch.log(row_sums+EPSILON).expand_as(t))
        return torch.div(t, row_sums.expand_as(t))
    else:
        s = nn.Softmax()
        return s(t.view(-1, t.size(len(t.size())-1))).view(t.size())


def _size_up_tensor(t):
    """
    Reshapes the tensor to have an extra dimension.  This is necessary for
    reshaping input batches of utterances/worlds into batches of utterance/world
    distributions used by the RSA modules

    Args:
        t (:obj:`batch_like`)
    Returns:
        t with an added distributional dimension
    """
    return torch.unsqueeze(t, 1)

def _size_down_tensor(t):
    """
    Inverts the operation performed by _size_up_tensor

    Args:
        t (:obj:`batch_like`)
    Returns:
        t with the distributional dimension removed
    """
    return torch.squeeze(t, 1)

class RSA(nn.Module):
    def __init__(self, name, level, meaning_fn, world_prior_fn, utterance_prior_fn, L_bottom=True, soft_bottom=False, alpha=1.0, observation_fn=None):
        super(RSA, self).__init__()
        self._name = name
        self._level = level
        self._meaning_fn = meaning_fn
        self._world_prior_fn = world_prior_fn
        self._utterance_prior_fn = utterance_prior_fn
        self._L_bottom = L_bottom
        self._soft_bottom = soft_bottom
        self._alpha = alpha
        self._observation_fn =observation_fn

    def get_name(self):
        return self._name

    def get_meaning_fn(self):
        return self._meaning_fn

    def get_world_prior_fn(self):
        return self._world_prior_fn

    def get_utterance_prior_fn(self):
        return self._utterance_prior_fn

    def get_observation_fn(self):
        return self._observation_fn

    def to_level(self, dist_type, level, L_bottom=True, soft_bottom=False):
        model = None
        if dist_type == DistributionType.L:
            model = L(self._name, level, self._meaning_fn, self._world_prior_fn, self._utterance_prior_fn, L_bottom=L_bottom, soft_bottom=soft_bottom, alpha=self._alpha, observation_fn=self._observation_fn)
        else:
            model = S(self._name, level, self._meaning_fn, self._world_prior_fn, self._utterance_prior_fn, L_bottom=L_bottom, soft_bottom=soft_bottom, alpha=self._alpha, observation_fn=self._observation_fn)
        if self.on_gpu():
            model = model.cuda()
        return model

    def on_gpu(self):
        return next(self.parameters()).is_cuda

    def cuda(self):
        ret = super(RSA, self).cuda()

        if torch.cuda.device_count() > 1:
            if self._observation_fn is not None:
                self._observation_fn = self._observation_fn.cuda(1)

        return ret

    @staticmethod
    def make(name, dist_type, level, meaning_fn, world_prior_fn, utterance_prior_fn, L_bottom=True, soft_bottom=False, alpha=1.0, observation_fn=None):
        if dist_type == DistributionType.L:
            return L(name, level, meaning_fn, world_prior_fn, utterance_prior_fn, L_bottom=L_bottom, soft_bottom=soft_bottom, alpha=alpha, observation_fn=observation_fn)
        else:
            return S(name, level, meaning_fn, world_prior_fn, utterance_prior_fn, L_bottom=L_bottom, soft_bottom=soft_bottom, alpha=alpha, observation_fn=observation_fn)

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
    #       rnn_type : [RNN TYPE]
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
    @staticmethod
    def load_from_config(config, D, gpu=False):
        data_parameter = DataParameter.make(**config["data_parameter"])
        utterance_field = config["data_parameter"]["utterance"]
        observation_field = config["data_parameter"]["L_observation"]
        if config["data_parameter"]["mode"] == DistributionType.S:
            observation_field = config["data_parameter"]["S_observation"]
        world_support_size = config["world_prior"]["support_size"]

        # Optionally setup observation function if observation is sequential
        observation_fn = None
        world_input_size = None
        if "observation_fn" in config:
            obs_config = config["observation_fn"]["seq_model"]
            world_input_size = obs_config["rnn_size"] * obs_config["rnn_layers"]
            if obs_config["bidirectional"]:
                world_input_size *= 2

            if "model_path" in obs_config:
                observation_fn = ObservationModel.load(obs_config["model_path"])
            else:
                observation_size = D[observation_field].get_matrix(0).get_feature_set().get_token_count()
                seq_observation_model = SequenceModelNoInput("Observation", observation_size,
                    obs_config["embedding_size"], obs_config["rnn_size"], obs_config["rnn_layers"], 
                    dropout=obs_config["dropout"], rnn_type=obs_config["rnn_type"], bidir=obs_config["bidirectional"], 
                    non_emb=obs_config["non_emb"])
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
            seq_meaning_model = SequenceModelInputToHidden("Meaning", utterance_size, world_input_size, \
            meaning_config["embedding_size"], meaning_config["rnn_size"], meaning_config["rnn_layers"], \
            dropout=meaning_config["dropout"], rnn_type=meaning_config["rnn_type"], 
            bidir=meaning_config["bidirectional"], input_layers=1)
            meaning_fn = MeaningModelIndexedWorldSequentialUtterance(world_input_size, seq_meaning_model, \
                input_type=SequentialUtteranceInputType.IN_SEQ)

        # Setup world prior
        world_prior_fn = UniformIndexPriorFn(world_support_size, on_gpu=gpu, unnorm=False)

        # Setup utterance prior
        heuristic = None
        if config["utterance_prior"]["heuristic"] == "L0":
            heuristic = HeuristicL0(world_prior_fn, meaning_fn, soft_bottom=False)

        seq_prior_model = SequenceModel.load(config["seq_model_path"])

        utterance_prior_params = config["utterance_prior"]
        utterance_prior_params["seq_length"] = D[utterance_field].get_feature_seq_set.get_size()
        utterance_prior_params["heuristic"] = heuristic
        utterance_prior_fn = SequenceSamplingPriorFn(seq_prior_model, world_input_size, **utterance_prior_params)
        
        rsa_model = RSA.make(data_parameter.get_mode() + "_" + str(config["training_level"]), \
                    data_parameter.get_mode(), config["training_level"], meaning_fn, world_prior_fn, 
                    utterance_prior_fn, L_bottom=True, soft_bottom=False, 
                    alpha=config["alpha"], observation_fn=observation_fn)
        if gpu:
            rsa_model = rsa_model.cuda()

        return rsa_model

class S(RSA):
    def __init__(self, name, level, meaning_fn, world_prior_fn, utterance_prior_fn, L_bottom=True, soft_bottom=False, alpha=1.0, observation_fn=None):
        """
        Constructs an RSA speaker module.  This module assumes world priors and
        utterance priors have finite discrete support.

        The speaker computes utterance distributions given worlds and
        observations.  Observations are aspects of the world that have no
        uncertainty from the perspective from the perspectives of speakers
        or listeners.

        Args:
            level (int): Level of the distribution

            meaning_fn (:obj:`utterance batch_like`, :obj:`world batch_like`, :obj:`observation batch_like` -> :obj:`tensor_like`):
                Args:
                    (Batch size) x (Utterance prior size) x (Utterance) array
                        of utterances
                    (Batch size) x (World prior size) x (World) array of worlds
                    (Batch size) x (Observation) array of observations
                Returns:
                    (Batch size) x (Utterance prior size) x (World prior size)
                        tensor of meaning values

            world_prior_fn (:obj:`observation batch_like` -> :obj:`dist.Distribution`):
                Args:
                    (Batch size) x (Observation) array of observations
                Returns:
                    A vectorized set of world distributions from a set of
                        observations.

            utterance_prior_fn (:obj:`observation batch_like` -> :obj:`dist.Distribution`):
                Args:
                    (Batch size) x (World) array of worlds
                Returns:
                    A vectorized set of utterance distributions from a set of
                        observations

            L_bottom (bool, optional): Indicates whether the model bottoms out at
                a literal listener or a literal speaker.  Defaults to True.
        """

        super(S, self).__init__(name, level, meaning_fn, world_prior_fn, utterance_prior_fn, L_bottom=L_bottom, soft_bottom=soft_bottom, alpha=alpha, observation_fn=observation_fn)

        self._L = None
        if self._level != 0:
            next_level = self._level
            if L_bottom:
                next_level = self._level - 1
            self._L = L(name, next_level, meaning_fn, world_prior_fn, utterance_prior_fn, L_bottom=L_bottom, soft_bottom=soft_bottom, alpha=alpha)

    def forward(self, world, observation=None, world_dist=False):
        """
        Computes the speaker distributions for the given worlds and
        observations

        Args:
            world (:obj:`world batch_like`):
                (Batch size) x (World prior size) x (World) batch of worlds.
                If world_dist is false, then there is no world prior dimension
            observation (:obj:`observation batch_like`, optional):
                (Batch size) x (Observation size) batch of observations

            world_dist (bool, optional):  Indicates whether world is a batch of
                single worlds or world prior supports.  Defaults to False.

        Returns:
            ltprg.dist.Categorical : (Batch size) x (World prior size) categorical
                speaker utterance distributions
        """

        if not world_dist:
            world = _size_up_tensor(world)

        if observation is None:
            observation = torch.zeros(world.size(0))

        if self._observation_fn is not None:
            if torch.cuda.device_count() > 1:
                if isinstance(observation, tuple):
                    observation = (observation[0].cuda(1),observation[1])
                else:
                    observation = observation.cuda(1)
            observation = self._observation_fn(observation)
            if self.on_gpu():
                if isinstance(observation, tuple):
                    observation = (observation[0].cuda(),observation[1])
                else:
                    observation = observation.cuda()

        utterance_prior = self._utterance_prior_fn(observation)
        world_support = None
        ps = None
        if self._level == 0:
            meaning = self._meaning_fn(utterance_prior.support(), world, observation).transpose(2,1)
            ps = _normalize_rows(utterance_prior.p().unsqueeze(1).expand_as(meaning) * meaning, softmax=self._soft_bottom)
        else:
            l = self._L(utterance_prior.support(), observation, utterance_dist=True)
            world_support = l.support()
            l_dist = l.p().transpose(2,1)
            ps = _normalize_rows(utterance_prior.p().unsqueeze(1).expand_as(l_dist) * (l_dist ** self._alpha))

        if not world_dist:
            if self._level > 0:
                world = _size_down_tensor(world)
                world_index, has_missing, mask  = self._world_prior_fn.get_index(world, observation, world_support)

                # world_index contains a list of world indices into worldxutterance
                # matrices.  So the offsets below will take the ith world index into
                # the ith matrix.
                # Note that there is a separate worldxutterance matrix for each observation.
                world_index_offset = torch.arange(0, ps.size(0)).long()*ps.size(1)
                if self.on_gpu():
                    world_index_offset = world_index_offset.cuda() + world_index
                else:
                    world_index_offset += world_index
                ps = ps.view(ps.size(0)*ps.size(1), ps.size(2))[world_index_offset]
                if has_missing:
                    raise ValueError("Prior missing input world") #ps = ps * mask.expand_as(ps) # FIXME Broken
            else:
                ps = _size_down_tensor(ps)

        return Categorical(utterance_prior.support(), ps=ps)

    def forward_batch(self, batch, data_parameters):
        world = Variable(batch[data_parameters[DataParameter.WORLD]], requires_grad=False)
        observation = Variable(batch[data_parameters[DataParameter.OBSERVATION]], requires_grad=False)
        if self.on_gpu():
            world = world.cuda()
            observation = observation.cuda()

        self._utterance_prior_fn.set_data_batch(batch, data_parameters)
        self._world_prior_fn.set_data_batch(batch, data_parameters)

        return self(world, observation=observation)

    def loss(self, batch, data_parameters, loss_criterion):
        utterance = None
        if data_parameters.is_utterance_seq():
            seq, length, _ = batch[data_parameters[DataParameter.UTTERANCE]]
            seq = Variable(seq, requires_grad=False)
            length = Variable(length, requires_grad=False)

            if self.on_gpu():
               seq = seq.cuda()

            utterance = (seq.transpose(0,1), length)
        else:
            utterance = Variable(batch[data_parameters[DataParameter.UTTERANCE]], requires_grad=False)

            if self.on_gpu():
                utterance = utterance.cuda()

        observation = Variable(batch[data_parameters[DataParameter.OBSERVATION]], requires_grad=False)

        model_dist = self.forward_batch(batch, data_parameters)
        index, _, _ = self._utterance_prior_fn.get_index(utterance, observation, model_dist.support(), preset_batch=True)
        return loss_criterion(torch.log(model_dist.p() + EPSILON), Variable(index, requires_grad=False))

class L(RSA):
    def __init__(self, name, level, meaning_fn, world_prior_fn, utterance_prior_fn, L_bottom=True, soft_bottom=False, alpha=1.0, observation_fn=None):
        """
        Constructs an RSA listener module.  This module assumes world priors and
        utterance priors have finite discrete support.

        The listener computes world distributions given utterances and
        observations.  Observations are aspects of the world that have no
        uncertainty from the perspective from the perspectives of speakers
        or listeners.

        Args:
            level (int): Level of the distribution

            meaning_fn (:obj:`utterance batch_like`, :obj:`world batch_like`, :obj:`observation batch_like` -> :obj:`tensor_like`):
                Args:
                    (Batch size) x (Utterance prior size) x (Utterance) array
                        of utterances
                    (Batch size) x (World prior size) x (World) array of worlds
                    (Batch size) x (Observation) array of observations
                Returns:
                    (Batch size) x (Utterance prior size) x (World prior size)
                        tensor of meaning values

            world_prior_fn (:obj:`observation batch_like` -> :obj:`dist.Distribution`):
                Args:
                    (Batch size) x (Observation) array of observations
                Returns:
                    A vectorized set of world distributions from a set of
                        observations.

            utterance_prior_fn (:obj:`observation batch_like` -> :obj:`dist.Distribution`):
                Args:
                    (Batch size) x (World) array of worlds
                Returns:
                    A vectorized set of utterance distributions from a set of
                        observations

            L_bottom (bool, optional): Indicates whether the model bottoms out at
                a literal listener or a literal speaker.  Defaults to True.
        """
        super(L, self).__init__(name, level, meaning_fn, world_prior_fn, utterance_prior_fn, L_bottom=L_bottom, soft_bottom=soft_bottom, alpha=alpha, observation_fn=observation_fn)

        self._S = None
        if self._level != 0:
            next_level = self._level
            if not L_bottom:
                next_level = self._level - 1
            self._S = S(name, next_level, meaning_fn, world_prior_fn, utterance_prior_fn, L_bottom=L_bottom, soft_bottom=soft_bottom, alpha=alpha)

    def forward(self, utterance, observation=None, utterance_dist=False):
        """
        Computes the listener distributions for the given meanings and
        observations

        Args:
            utterance (:obj:`utterance batch_like`):
                (Batch size) x (Utterance prior size) x (Utterance) batch of
                utterances. If utterance_dist is false, then there is no
                utterance prior dimension
            observation (:obj:`observation batch_like`, optional):
                (Batch size) x (Observation size) batch of observations

            utterance_dist (bool, optional): Indicates whether utterance is
                a batch of single utterances or utterance prior supports.
                Defaults to False.

        Returns:
            ltprg.dist.Categorical : (Batch size) x (Utterance prior size)
            categorical listener world distributions
        """
        if not utterance_dist:
            if isinstance(utterance, tuple):
                utterance = (_size_up_tensor(utterance[0]), _size_up_tensor(utterance[1]))
            else:
                utterance = _size_up_tensor(utterance)

        if observation is None:
            if isinstance(utterance, tuple):
                observation = torch.zeros(utterance[0].size(0))
            else:
                observation = torch.zeros(utterance.size(0))

        if self._observation_fn is not None:
            if torch.cuda.device_count() > 1:
                if isinstance(observation, tuple):
                    observation = (observation[0].cuda(1),observation[1])
                else:
                    observation = observation.cuda(1)
            observation = self._observation_fn(observation)
            if self.on_gpu():
                if isinstance(observation, tuple):
                    observation = (observation[0].cuda(),observation[1])
                else:
                    observation = observation.cuda()

        world_prior = self._world_prior_fn(observation)
        utterance_support = None
        ps = None
        if self._level == 0:
            meaning = self._meaning_fn(utterance, world_prior.support(), observation)
            ps = _normalize_rows(world_prior.p().unsqueeze(1).expand_as(meaning) * meaning, softmax=self._soft_bottom)
        else:
            s = self._S(world_prior.support(), observation=observation, world_dist=True)
            utterance_support = s.support()
            s_dist = s.p().transpose(2,1)  # Batch size x world size x utterance size
            ps = _normalize_rows(world_prior.p().unsqueeze(1).expand_as(s_dist) * s_dist)

        if not utterance_dist:
            if self._level > 0:
                if isinstance(utterance, tuple):
                    utterance = (_size_down_tensor(utterance[0]), _size_down_tensor(utterance[1]))
                else:
                    utterance = _size_down_tensor(utterance)
                utt_index, has_missing, mask = self._utterance_prior_fn.get_index(utterance, observation, utterance_support)
                utt_index_offset = torch.arange(0, ps.size(0)).long()*ps.size(1) + utt_index
                if self.on_gpu():
                    utt_index_offset = utt_index_offset.cuda()
                ps = ps.view(ps.size(0)*ps.size(1), ps.size(2))[utt_index_offset]
                if has_missing:
                    raise ValueError("Utterance prior missing input utterance") #ps = ps * mask.expand_as(ps) # FIXME Broken
            else:
                ps = _size_down_tensor(ps)

        return Categorical(world_prior.support(), ps=ps)

    def forward_batch(self, batch, data_parameters):
        utterance = None
        if data_parameters.is_utterance_seq():
            seq, length, _ = batch[data_parameters[DataParameter.UTTERANCE]]
            seq = Variable(seq.long(), requires_grad=False)
            if self.on_gpu():
                seq = seq.cuda()
            utterance = (seq.transpose(0,1), length)
        else:
            utterance = Variable(batch[data_parameters[DataParameter.WORLD]], requires_grad=False)
            if self.on_gpu():
                utterance = utterance.cuda()

        # FIXME This should be checked in a nicer way
        # Also need to add it to the speaker
        if self._observation_fn is not None:
            seq, length, _ = batch[data_parameters[DataParameter.OBSERVATION]]
            seq = Variable(seq.long())
            if self.on_gpu():
                seq = seq.cuda()
            observation = (seq.transpose(0,1), length)
        else:
            observation = Variable(batch[data_parameters[DataParameter.OBSERVATION]], requires_grad=False)
            if self.on_gpu():
                observation = observation.cuda()

        self._utterance_prior_fn.set_data_batch(batch, data_parameters)
        self._world_prior_fn.set_data_batch(batch, data_parameters)

        return self(utterance, observation=observation)

    def loss(self, batch, data_parameters, loss_criterion):
        world = Variable(batch[data_parameters[DataParameter.WORLD]]).squeeze()
        if self.on_gpu():
            world = world.cuda()

        # FIXME This should be checked in a nicer way
        # Also need to add it to the speaker
        if self._observation_fn is not None:
            seq, length, _ = batch[data_parameters[DataParameter.OBSERVATION]]
            seq = Variable(seq.long(), requires_grad=False)
            if self.on_gpu():
                seq = seq.cuda()
            observation = (seq.transpose(0,1), length)
        else:
            observation = Variable(batch[data_parameters[DataParameter.OBSERVATION]], requires_grad=False)
            if self.on_gpu():
                observation = observation.cuda()

        model_dist = self.forward_batch(batch, data_parameters)
        index, _, _ = self._world_prior_fn.get_index(world, observation, model_dist.support(), preset_batch=True)
        return loss_criterion(torch.log(model_dist.p() + EPSILON), Variable(index, requires_grad=False))


class RSADistributionAccuracy(DistributionAccuracy):
    def __init__(self, name, level, distribution_type, data, data_parameters, target_indexed = False, L_bottom = True, trials=1):
        super(RSADistributionAccuracy, self).__init__(name, data, data_parameters, model_fn=None, target_indexed = target_indexed, trials=trials)

        def _mfn(batch, model, data_parameters):
            model = model.to_level(self._distribution_type, self._level, L_bottom=L_bottom, soft_bottom=model._soft_bottom)
            return model.forward_batch(batch, data_parameters.to_mode(self._distribution_type))
        self._model_fn = _mfn

        self._level = level
        self._distribution_type = distribution_type
        self._L_bottom = L_bottom
        self._data_parameters = data_parameters.to_mode(self._distribution_type)

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
#    type : (NLLLoss, RSADistributionAccuracy)
#    data : [NAME OF DATA SUBSET]
#    (Optional) data_size : [SIZE OF RANDOM SUBET OF DATA TO TAKE]
#    parameters : {
#      (RSADistributionAccuracy) dist_level : [RSA DISTRIBUTION LEVEL TO EVALUATE]
#      (RSADistributionAccuracy) dist_type : [TYPE OF RSA DISTRIBUTION (L|S)]
#    }
#   ]
# }
def load_evaluations_from_config(config, D, gpu=False):
    data_parameter = DataParameter.make(**config["data_parameter"])
    evaluations = []

    loss_criterion = NLLLoss(size_average=False)
    if gpu:
        loss_criterion = loss_criterion.cuda()

    for eval_config in config["evaluations"]:
        data = D[eval_config["data"]]
        if "data_size" in config:
            data = data.get_random_subset(eval_config["data_size"])

        if eval_config["type"] == "NLLLoss":
            loss = Loss(eval_config["name"], data, data_parameter, loss_criterion)
            evaluations.append(loss)
        elif eval_config["type"] == "RSADistributionAccuracy":
            acc = RSADistributionAccuracy(eval_config["name"], eval_config["parameters"]["dist_level"], \
                eval_config["parameters"]["dist_type"], data, data_parameter)
            evaluations.append(acc)
        else:
            raise ValueError("Invalid RSA evaluation type in config (" + str(eval_config["type"]))
    return evaluations

