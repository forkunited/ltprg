import torch.nn as nn
import torch
from ltprg.model.rsa import DistributionType, RSA, DataParameter, _normalize_rows

class HeuristicL0(nn.Module):
    def __init__(self, world_prior_fn, meaning_fn, soft_bottom=False):
        super(HeuristicL0, self).__init__()
        self._world_prior_fn = world_prior_fn
        self._meaning_fn = meaning_fn
        self._L_0 = RSA.make("L_0_heuristic", DistributionType.L, 0, meaning_fn, world_prior_fn, None, L_bottom=True, soft_bottom=soft_bottom)

    def forward(self, seq, input, heuristic_state, context=None):
        L_dist = self._L_0((seq[0].transpose(0,1), seq[1]), context[0])
        #input_index, has_missing, mask = L_dist.get_index(context[1])
        return torch.log(torch.gather(L_dist.p().data, 1, context[1]).squeeze()), heuristic_state # Switched from input_index.unsqueeze(1) to context[1]

class HeuristicL0H(nn.Module):
    def __init__(self, world_prior_fn, meaning_fn):
        super(HeuristicL0H, self).__init__()
        self._world_prior_fn = world_prior_fn
        self._meaning_fn = meaning_fn

    def forward(self, seq, input, heuristic_state, context=None):
        utterance = torch.unsqueeze(seq[0].transpose(0, 1),1)
        world_prior = self._world_prior_fn(context[0])
        meaning = self._meaning_fn(utterance, world_prior.support(), context[0])
        ps = _normalize_rows(self._world_prior.p().unsqueeze(1).expand_as(meaning) * meaning, softmax=False)
        ps = torch.squeeze(ps, 1)
        neg_H = torch.sum(ps*torch.log(ps), 1)
        return neg_H, heuristic_state