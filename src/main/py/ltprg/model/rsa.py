import torch
import torch.nn as nn
import mung.torch_ext.eval
import os.path
import json
from torch.autograd import Variable
from ltprg.model.dist import Categorical
from mung.torch_ext.eval import Evaluation, DistributionAccuracy

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

    def get_level(self):
        return self._level

    def get_alpha(self):
        return self._alpha

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

    def get_distribution_type(self):
        return DistributionType.S

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

    def get_distribution_type(self):
        return DistributionType.L

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



# At each evaluation, outputs a json file containing priors for each round
# represented by a data point
class PriorView(Evaluation):
    def __init__(self, name, data, data_parameters, output_dir):
        super(PriorView, self).__init__(name, data, data_parameters)

        self._directory_path = os.path.join(output_dir, name)
        if not os.path.exists(self._directory_path):
            os.makedirs(self._directory_path)

        self._iteration = 0

    def _run_batch(self, model, batch):
        batch_result = []

        utterance_prior_fn = model.get_utterance_prior_fn()
        observation = Variable(batch[self._data_parameters[DataParameter.OBSERVATION]], requires_grad=False)
        if model.on_gpu():
            observation = observation.cuda()
        utterance_prior_fn.set_data_batch(batch, self._data_parameters)
        prior_utts, prior_lens = utterance_prior_fn(observation).support()

        # Run l0 overall all utterance prior utterances paired with contexts
        obs = batch[self._data_parameters[DataParameter.OBSERVATION]]
        batch_size = obs.size(0)
        support_size = prior_utts.size(1)
        l0_batch = dict()
        l0_batch[self._data_parameters[DataParameter.OBSERVATION]] = obs.unsqueeze(1).expand(batch_size, support_size, obs.size(1)).contiguous().view(batch_size*support_size, obs.size(1))
        l0_batch[self._data_parameters[DataParameter.UTTERANCE]] = (prior_utts.contiguous().view(batch_size*support_size,prior_utts.size(2)).data.transpose(0,1), \
                                                                   prior_lens.contiguous().view(batch_size*support_size), None)
        l0 = model.to_level(DistributionType.L, 0, L_bottom=True, soft_bottom=model._soft_bottom)
        p_0 = l0.forward_batch(l0_batch, self._data_parameters.to_mode(DistributionType.L)).p().data
        p_0 = p_0.contiguous().view(batch_size, support_size, p_0.size(1))

        H = torch.sum(-p_0*torch.log(p_0))

        t = 0
        for i in range(observation.size(0)):
            round_i = { "roundNum" : i, "events" : [] }
            support_i = prior_utts[i]
            support_lens_i = prior_lens[i]
            for j in range(support_i.size(0)):
                dist_str = "(" + " ".join([str(round(val, 2)) for val in p_0[i,j]])  + ") "
                utt_tokens = [self._data[self._data_parameters[DataParameter.UTTERANCE]].get_feature_token(support_i.data[j,k]).get_value() \
                    for k in range(support_lens_i[j])]
                utt_str = dist_str + " ".join(utt_tokens)
                utt_event = { "eventType": "utterance", "type": "Utterance",
                    "sender": "speaker", "contents": utt_str,"time": t }
                round_i["events"].append(utt_event)
                t += 1

            batch_result.append(round_i)

        return (batch_result, H)

    def _aggregate_batch(self, agg, batch_result):
        data_round_index = len(agg[0]["records"])
        for i in range(len(batch_result[0])):
            # For each new round

            round_events = batch_result[0][i]["events"]
            t = round_events[len(round_events) - 1]["time"] + 1

            state = self._data.get_data().get(data_round_index + i).get("state.state")
            round_events.append({ "eventType" : "state", "state" : state , "type" : "State", "time" : t })
            t += 1

            round_events.append({ "eventType": "action", "time": t, "mouseY": -1,
                                    "mouseX": -1, "lClicked": state["listenerOrder"].index(state["target"]), "type": "Action",
                                    "condition": state["condition"] })
        
        agg[0]["records"].extend(batch_result[0])
        return (agg[0], agg[1] + batch_result[1])

    def _initialize_result(self):
        agg = ({ "records" : [], "gameid" : self._name + " (" + str(self._iteration) + ")" }, 0.0)
        return agg

    def _finalize_result(self, result):
        output_file = os.path.join(self._directory_path, \
            str(self._iteration) + ".json")
 
        with open(output_file, mode="w") as fp:
            json.dump(result[0], fp) 

        self._iteration += 1
        return result[1]/self._data.get_size()

