import torch
from torch.autograd import Variable

class Evaluation(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, name):
        super(Evaluation, self).__init__()
        self._name = name

    @abc.abstractmethod
    def run(self, model):
        """ Evaluates the model """"

    def get_name(self):
        return self._name

    @staticmethod
    def run_all(evaluations, model):
        results = dict()
        for evaluation in evaluations:
            results[evaluation.get_name()] = evaluation.evaluate(model)
        return results

class EvaluationSequential(Evaluation):
    __metaclass__ = abc.ABCMeta

    def __init__(self, name, data, input_view_name, seq_view_name):
        super(EvaluationSequential, self).__init__(name)

        # Loading all the stuff on construction will be faster but hog
        # memory.  If it's a prooblem, then move this into the run method.
        batch = data.get_batch(0, data.get_size())
        seq, length, mask = batch[seq_view_name]

        self._name = name
        self._data = data
        self._input_view_name = input_view_name
        self._seq_view_name = seq_view_name
        self._data_input = Variable(batch[input_view_name])
        self._seq_length = length - 1
        self._seq_in = Variable(seq[:seq.size(0)-1]).long() # Input remove final token
        self._target_out = Variable(seq[1:seq.size(0)]).long() # Output (remove start token)
        self._mask = mask

    @abc.abstractmethod
    def run_helper(self, model, model_out, hidden):
        """ Evaluates the model according to its output """"

    def run(self, model):
        model.eval()
        model_out, hidden = self(seq_part=self._seq_in,
                                 seq_length=self._seq_length,
                                 input=self._data_input)

        result = self._run_helper(model, model_out, hidden)

        model.train()
        return result

class EvaluationSequentialLoss(EvaluationSequential):
    __metaclass__ = abc.ABCMeta

    def __init__(self, name, data, input_view_name, seq_view_name, loss_criterion):
        super(EvaluationSequential, self).__init__(name, data, input_view_name, seq_view_name)
        self._loss_criterion = loss_criterion

    def run_helper(self, model, model_out, hidden):
        loss = self._loss_criterion(model_out, self._target_out[:model_out.size(0)], Variable(self._mask[:,1:(model_out.size(0)+1)]))
        return loss.data[0]

# FIXME
class Accuracy(Evaluation):
    def __init__(self, name, data):
        super(Accuracy, self).__init__(name)
        self._data = data

    def run(self, model):
