import time
import copy
from torch.optim import Adam
from ltprg.model.eval import Evaluation

class Trainer:
    def __init__(self, data_parameters, loss_criterion, logger, evaluation, other_evaluations=None, max_evaluation=False):
        self._data_parameters = data_parameters
        self._loss_criterion = loss_criterion
        self._logger = logger
        self._all_evaluations = [evaluation]
        self._max_evaluation = max_evaluation

        if other_evaluations is not None:
            self._all_evaluations.extend(other_evaluations)

    def train(self, model, data, iterations, batch_size=100, lr=0.001, log_interval=100):
        model.train()
        start_time = time.time()
        optimizer = Adam(model.parameters(), lr=lr)
        total_loss = 0.0

        self._logger.set_key_order(["Model", "Iteration", "Avg batch ms", "Avg batch loss"])

        best_model = copy.deepcopy(model)
        best_result = float("inf")
        best_iteration = 0
        if self._max_evaluation:
            best_result = - best_result

        for i in range(1, iterations + 1):
            batch = data.get_random_batch(batch_size)
            loss = model.loss(batch, self._data_parameters, self._loss_criterion)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.data

            if i % log_interval == 0:
                avg_loss = total_loss[0] / log_interval
                
                avg_batch_ms = (time.time() - start_time)/log_interval
                eval_start_time = time.time()
                results = Evaluation.run_all(self._all_evaluations, model)
                results["Model"] = model.get_name()
                results["Iteration"] = i
                results["Avg batch time"] = avg_batch_ms
                results["Evaluation time"] = time.time()-eval_start_time
                results["Avg batch loss"] = avg_loss
                self._logger.log(results)

                main_result = results[self._all_evaluations[0].get_name()]
                if (self._max_evaluation and main_result > best_result) or \
                    ((not self._max_evaluation) and main_result < best_result):
                    best_result = main_result
                    best_model = copy.deepcopy(model)
                    best_iteration = i

                total_loss = 0.0
                start_time = time.time()

        model.eval()

        return model, best_model
