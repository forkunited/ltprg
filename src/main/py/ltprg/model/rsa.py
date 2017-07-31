import torch
import torch.nn as nn
from ltprg.model.dist import Categorical

def _normalize_rows(t):
	row_sums = torch.sum(t, dim=len(t.size())-1)
	return torch.div(t, row_sums.expand_as(t))


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


class S(nn.Module):
    def __init__(self, level, meaning_fn, world_prior_fn, utterance_prior_fn, L_bottom=True):
        """
        Constructs an RSA speaker module.  This module assumes world priors and
        utterance priors have finite discrete support.

        The speaker computes utterance distributions given worlds and
        observations.  Observations are aspects of the world that have no
        uncertainty from the perspective from the perspectives of speakers
        or listeners.

        Args:
            level (int): Level of the distribution

            meaning_fn (:obj:`utterance batch_like`, :obj:`world batch_like` -> :obj:`tensor_like`):
                Args:
                    (Batch size) x (Utterance prior size) x (Utterance) array
                        of utterances
                    (Batch size) x (World prior size) x (World) array of worlds
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

        super(S, self).__init__()
        self._level = level
        self._meaning_fn = meaning_fn
        self._world_prior_fn = world_prior_fn
        self._utterance_prior_fn = utterance_prior_fn

        self._L = None
        if self._level != 0:
            next_level = self._level
            if L_bottom:
                next_level = self._level - 1
            self._L = L(next_level, meaning_fn, world_prior_fn, utterance_prior_fn, L_bottom=L_bottom)

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

        utterance_prior = self._utterance_prior_fn(observation)
        ps = None
        if self._level == 0:
            meaning = self._meaning_fn(utterance, world).transpose(2,1)
            ps = _normalize_rows(utterance_prior.p().unsqueeze(1).expand_as(meaning) * meaning)
        else:
            l_dist = self._L(utterance_prior.support(), observation, utterance_dist=True).p().transpose(2,1)
            ps = _normalize_rows(utterance_prior.p().unsqueeze(1).expand_as(l_dist) * l_dist)

        if not world_dist:
            if self._level > 0:
                world = _size_down_tensor(world)
                world_index = self._world_prior_fn.get_index(world, observation)
                    
                # world_index contains a list of world indices into worldxutterance
                # matrices.  So the offsets below will take the ith world index into
                # the ith matrix.
                # Note that there is a separate worldxutterance matrix for each observation.
                world_index_offset = torch.arange(0, ps.size(0)).long()*ps.size(1) + world_index
                ps = ps.view(ps.size(0)*ps.size(1), ps.size(2))[world_index_offset]
            else:
                ps = _size_down_tensor(ps)

        return Categorical(utterance_prior.support(), ps=ps)


class L(nn.Module):
    def __init__(self, level, meaning_fn, world_prior_fn, utterance_prior_fn, L_bottom=True):
        """
        Constructs an RSA listener module.  This module assumes world priors and
        utterance priors have finite discrete support.

        The listener computes world distributions given utterances and
        observations.  Observations are aspects of the world that have no
        uncertainty from the perspective from the perspectives of speakers
        or listeners.

        Args:
            level (int): Level of the distribution

            meaning_fn (:obj:`utterance batch_like`, :obj:`world batch_like` -> :obj:`tensor_like`):
                Args:
                    (Batch size) x (Utterance prior size) x (Utterance) array
                        of utterances
                    (Batch size) x (World prior size) x (World) array of worlds
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

        super(L, self).__init__()
        self._level = level
        self._meaning_fn = meaning_fn
        self._world_prior_fn = world_prior_fn
        self._utterance_prior_fn = utterance_prior_fn

        self._S = None
        if self._level != 0:
            next_level = self._level
            if not L_bottom:
                next_level = self._level - 1
            self._S = S(next_level, meaning_fn, world_prior_fn, utterance_prior_fn, L_bottom=L_bottom)

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
            utterance = _size_up_tensor(utterance)

        if observation is None:
            observation = torch.zeros(utterance.size(0))

        world_prior = self._world_prior_fn(observation)    

        ps = None
        if self._level == 0:
            meaning = self._meaning_fn(utterance, world_prior.support())
            ps = _normalize_rows(world_prior.p().unsqueeze(1).expand_as(meaning) * meaning)
        else:
            s_dist = self._S(world_prior.support(), world_dist=True).p().transpose(2,1)  # Batch size x world size x utterance size
            ps = _normalize_rows(world_prior.p().unsqueeze(1).expand_as(s_dist) * s_dist)

        if not utterance_dist:
            if self._level > 0:
                utterance = _size_down_tensor(utterance)
                utt_index = self._utterance_prior_fn.get_index(utterance, observation)
                utt_index_offset = torch.arange(0, ps.size(0)).long()*ps.size(1) + utt_index
                ps = ps.view(ps.size(0)*ps.size(1), ps.size(2))[utt_index_offset]
            else:
                ps = _size_down_tensor(ps)

        return Categorical(world_prior.support(), ps=ps)


