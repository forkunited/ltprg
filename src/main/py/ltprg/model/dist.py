import abc
import torch
from torch.autograd import Variable

class Distribution(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self):
        pass

    @abc.abstractmethod
    def sample(self, n=1):
        """ Returns a sample from the distribution """

    @abc.abstractmethod
    def support(self):
        """ Returns the support of the distribution """

    @abc.abstractmethod
    def p(self):
        """ Returns pmf or pdf """

class Categorical(Distribution):
    def __init__(self, vs, ps=None):
        """
            Constructs a vectorized set of categorical distributions

            Args:
                vs (:obj:`batch_like`): (Batch size) x (Support size) array of
		    		supports
                ps (:obj:`batch_like`, optional): (Batch size) x
                    (Support size) array of probabilities.  Defaults to a batch of
                    uniform distributions
        """
        Distribution.__init__(self)

        self._vs = vs
        self._ps = ps
        if self._ps is None:
            vs_for_size = self._vs
            if isinstance(self._vs, tuple):
                vs_for_size = self._vs[0]
            self._ps = torch.ones(vs_for_size.size(0), vs_for_size.size(1))
            self._ps = Variable(self._ps/torch.sum(self._ps, dim=1).repeat(1,self._ps.size(1)))

    def __getitem__(self, key):
        if key == 0:
            return self._ps

    def sample(self, n=1):
        indices = torch.multinomial(self._ps, n, True).data
        if isinstance(self._vs, tuple):
            vs_parts = []
            for i in range(len(self._vs)):
                vs_parts.append(torch.gather(self._vs[i], len(self._vs[i].size())-1, indices))
            return tuple(vs_parts)
        else:
            return torch.gather(self._vs, len(self._vs.size())-1, indices)

    def support(self):
        return self._vs

    def p(self):
        return self._ps

    def get_index(self, value):
        return Categorical.get_support_index(value, self._vs)

    @staticmethod
    def get_support_index(value, support):
        index = None
        has_missing = False
        mask = None

        if isinstance(support, tuple):
            index = torch.zeros(value[0].size(0)).long()
            mask = torch.ones(value[0].size(0)).long()

            if isinstance(value[0], Variable):
                value = (value[0].data, value[1])

            if isinstance(support[0], Variable):
                support = (support[0].data, support[1])
        
            for b in range(support[0].size(0)): # Over batch
                found = False
                for s in range(support[0].size(1)): # Over samples in support
                    match = True
                    for i in range(len(support)): # Over values in tuple
                        if (len(value[i].size()) > 1 and not torch.equal(support[i][b,s], value[i][b])) \
                           or (len(value[i].size()) == 1 and support[i][b,s] != value[i][b]):
                            match = False
                            break
                    if match:
                        index[b] = s
                        found = True
                        break
                if not found:
                    has_missing = True
                    mask[b] = 0
        else:
            index = torch.zeros(value.size(0)).long()
            mask = torch.ones(value.size(0)).long()

            if isinstance(value, Variable):
                value = value.data

            if isinstance(support, Variable):
                support = support.data

            for b in range(support.size(0)): # Over batch
                found = False
                for s in range(support.size(1)): # Over samples in support
                    if (len(value.size()) > 1 and torch.equal(support[b,s], value[b])) \
                       or (len(value.size()) == 1 and support[b,s] == value[b]):
                        index[b] = s
                        found = True
                        break
                if not found:
                    has_missing = True
                    mask[b] = 0
        return index, has_missing, mask
