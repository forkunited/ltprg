import abc
import torch
from torch.autograd import Variable

EVALUATION_BATCH_SIZE = 200

class DataParameter:
    TARGET = "target"

    @staticmethod
    def make(target="target"):
        data_parameters = dict()
        data_parameters[DataParameter.TARGET] = target
        return data_parameters

class Evaluation(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, name, data, data_parameters):
        super(Evaluation, self).__init__()
        self._name = name
        self._data = data
        self._data_parameters = data_parameters

    def run(self, model):
        model.eval()

        batch_size = EVALUATION_BATCH_SIZE
        if batch_size > self._data.get_size():
            batch_size = self._data.get_size()

        result = self._initialize_result()

        for i in range(self._data.get_num_batches(batch_size)):
            result = self._aggregate_batch(result, self._run_batch(model, self._data.get_batch(i, batch_size)))

        final_batch = self._data.get_final_batch(batch_size)
        if final_batch is not None:
            result = self._aggregate_batch(result, self._run_batch(model, final_batch))

        model.train()

        return self._finalize_result(result)

    @abc.abstractmethod
    def _run_batch(self, model, batch):
        """ Evaluates the model on a given batch of data """

    @abc.abstractmethod
    def _aggregate_batch(self, agg, batch_result):
        """ Aggregates batch result into total """

    @abc.abstractmethod
    def _initialize_result(self):
        """ Initializes the aggregate result """

    @abc.abstractmethod
    def _finalize_result(self, result):
        """ Returns the final value given the aggregate result """

    def get_name(self):
        return self._name

    @staticmethod
    def run_all(evaluations, model):
        results = dict()
        for evaluation in evaluations:
            results[evaluation.get_name()] = evaluation.run(model)
        return results

class Loss(Evaluation):
    def __init__(self, name, data, data_parameters, loss_criterion, norm_dim=False):
        super(Loss, self).__init__(name, data, data_parameters)
        self._loss_criterion = loss_criterion
        self._norm_dim = norm_dim

    def _run_batch(self, model, batch):
        loss = model.loss(batch, self._data_parameters, self._loss_criterion)
        if self._norm_dim:
            return (loss[0].data[0], loss[1].data[0])
        else:
            return loss.data[0]

    def _aggregate_batch(self, agg, batch_result):
        if self._norm_dim:
            return (agg[0] + batch_result[0], agg[1] + batch_result[1])
        else:
            return agg + batch_result

    def _initialize_result(self):
        if self._norm_dim:
            return (0.0, 0.0)
        else:
            return 0.0

    def _finalize_result(self, result):
        if self._norm_dim:
            return result[0] / result[1]
        else:
            return result / self._data.get_size()

class DistributionAccuracy(Evaluation):
    def __init__(self, name, data, data_parameters, model_fn=None, target_indexed = False):
        super(DistributionAccuracy, self).__init__(name, data, data_parameters)
        self._model_fn = model_fn
        self._target_indexed = target_indexed

    def _run_batch(self, model, batch):
        dist = None
        if self._model_fn is None:
            dist = model.forward_batch(batch, self._data_parameters)
        else:
            dist = self._model_fn(batch, model, self._data_parameters)

        target = batch[self._data_parameters[DataParameter.TARGET]].squeeze()

        model_ps = dist.p().data.cpu()
        max_ps, max_index = torch.max(model_ps, 1, keepdim=True)

        # Indicators of whether maxima are unique
        max_unique = (torch.sum(max_ps.expand_as(model_ps) == model_ps, 1, keepdim=True) == 1).long()

        total_correct = None
        if self._target_indexed:
            total_correct = torch.sum(max_unique.squeeze()*((target == max_index.squeeze()).long()))
        else:
            target_index, has_missing, mask = dist.get_index(target)
            # Count of where model max is same as target
            total_correct = torch.sum(mask*max_unique.squeeze()*((target_index == max_index.squeeze()).long()))

        return float(total_correct)

    def _aggregate_batch(self, agg, batch_result):
        return agg + batch_result

    def _initialize_result(self):
        return 0.0

    def _finalize_result(self, result):
        return result / self._data.get_size()
