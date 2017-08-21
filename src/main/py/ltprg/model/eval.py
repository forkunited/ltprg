import abc
import torch
from torch.autograd import Variable

class DataParameter:
    TARGET = "target"

    @staticmethod
    def make(target="target"):
        data_parameters = dict()
        data_parameters[DataParameter.TARGET] = target
        return data_parameters

class Evaluation(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, name):
        super(Evaluation, self).__init__()
        self._name = name

    @abc.abstractmethod
    def run(self, model):
        """ Evaluates the model """

    def get_name(self):
        return self._name

    @staticmethod
    def run_all(evaluations, model):
        results = dict()
        for evaluation in evaluations:
            results[evaluation.get_name()] = evaluation.run(model)
        return results

class Loss(Evaluation):
    def __init__(self, name, data, data_parameters, loss_criterion):
        super(Loss, self).__init__(name)
        self._data = data
        self._data_parameters = data_parameters
        self._loss_criterion = loss_criterion

    def run(self, model):
        model.eval() 
        batch = self._data.get_batch(0, self._data.get_size())
        loss = model.loss(batch, self._data_parameters, self._loss_criterion)
        model.train()
        return loss.data[0]

class DistributionAccuracy(Evaluation):
    def __init__(self, name, data, data_parameters, model_fn=None, target_indexed = False):
        super(DistributionAccuracy, self).__init__(name)
        self._data = data
        self._data_parameters = data_parameters
        self._model_fn = model_fn
        self._target_indexed = target_indexed

    def run(self, model):
        model.eval()

        batch = self._data.get_batch(0, self._data.get_size())
        dist = None
        if self._model_fn is None:
            dist = model.forward_batch(batch, self._data_parameters)
        else:
            dist = self._model_fn(batch, model, self._data_parameters)

        target = batch[self._data_parameters[DataParameter.TARGET]].squeeze()

        model_ps = dist.p().data
        max_ps, max_index = torch.max(model_ps, 1 )

        # Indicators of whether maxima are unique
        max_unique = (torch.sum(max_ps.expand_as(model_ps) == model_ps, 1) == 1).long()

        total_correct = None
        if self._target_indexed:
            total_correct = torch.sum(max_unique*((target == max_index).long()))
        else:
            target_index, has_missing, mask = dist.get_index(target)
            # Count of where model max is same as target
            total_correct = torch.sum(mask*max_unique*((target_index == max_index).long()))

        model.train()

        return float(total_correct) / target.size(0)
