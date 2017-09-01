import torch.nn as nn
import torch
from ltprg.model.rsa import DistributionType, RSA, DataParameter

class HeuristicL0(nn.Module):
    def __init__(self, world_prior_fn, meaning_fn, soft_bottom=False):
        super(HeuristicL0, self).__init__()
        self._world_prior_fn = world_prior_fn
        self._meaning_fn = meaning_fn
        self._L_0 = RSA.make("L_0_heuristic", DistributionType.L, 0, meaning_fn, world_prior_fn, None, L_bottom=True, soft_bottom=soft_bottom)
        self._observation = None

    def set_data_batch(self, batch, data_parameters):
        self._observation = batch(data_parameters[DataParameter.OBSERVATION])

    def forward(self, seq, input, heuristic_state):
        L_dist = self._L_0(utterance, self._observation)
        input_index, has_missing, mask = L_dist.get_index(input)
        return torch.log(torch.gather(L_dist.p(), 1, input_index.unsqueeze(1)).squeeze())
