import abc

class ModelTrainer(object):
    """ Abstract class for all models used for synthethic data. """
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def format_inputs(self):
        return

    @abc.abstractmethod
    def update(self):
        return

    @abc.abstractmethod
    def predict(self):
        return

    @abc.abstractmethod
    def evaluate(self):
        return

    @abc.abstractmethod
    def train(self):
        return

    @abc.abstractmethod
    def train_model(self):
        return

    @abc.abstractmethod
    def run_example()
        return




