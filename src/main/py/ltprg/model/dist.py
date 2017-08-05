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
