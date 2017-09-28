import time
import copy
import torch.nn.utils
from torch.optim import Adam, Adadelta
from ltprg.model.eval import Evaluation

class OptimizerType:
    ADAM = "ADAM"
    ADADELTA = "ADADELTA"

class Trainer:
    def __init__(self, data_parameters, loss_criterion, logger, evaluation, other_evaluations=None, max_evaluation=False, sample_with_replacement=False):
        self._data_parameters = data_parameters
        self._loss_criterion = loss_criterion
        self._logger = logger
        self._all_evaluations = [evaluation]
        self._max_evaluation = max_evaluation
        self._sample_with_replacement = sample_with_replacement

        if other_evaluations is not None:
            self._all_evaluations.extend(other_evaluations)

    def train(self, model, data, iterations, batch_size=100, optimizer_type=OptimizerType.ADAM, lr=0.001, grad_clip=None, weight_decay=0.0, log_interval=100, best_part_fn=None):
        model.train()
        start_time = time.time()

        optimizer = None
        if optimizer_type == OptimizerType.ADAM:
            optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            optimizer = Adadelta(model.parameters(), rho=0.95, lr=lr, weight_decay=weight_decay)

        total_loss = 0.0

        self._logger.set_key_order(["Model", "Iteration", "Avg batch time", "Evaluation time", "Avg batch loss"])

        best_part = None
        if best_part_fn is None:
            best_part = copy.deepcopy(model)
        else:
            best_part = copy.deepcopy(best_part_fn(model))

        best_result = float("inf")
        if self._max_evaluation:
            best_result = float("-inf")

        best_iteration = 0
        if self._max_evaluation:
            best_result = - best_result

        b = 0
        for i in range(1, iterations + 1):
            batch = None
            if self._sample_with_replacement:
                batch = data.get_random_batch(batch_size)
            else:
                batch = data.get_batch(b, batch_size)

            loss = model.loss(batch, self._data_parameters, self._loss_criterion)

            optimizer.zero_grad()
            loss.backward()

            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm(model.parameters(), grad_clip)
            
            optimizer.step()

            total_loss += loss.data

            b += 1

            if (not self._sample_with_replacement) and  b == data.get_num_batches(batch_size):
                b = 0
                data.shuffle()

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
                self._logger.save()

                main_result = results[self._all_evaluations[0].get_name()]
                if (self._max_evaluation and main_result > best_result) or \
                    ((not self._max_evaluation) and main_result < best_result):
                    best_result = main_result
                    if best_part_fn is None:
                        best_part = copy.deepcopy(model)
                    else:
                        best_part = copy.deepcopy(best_part_fn(model))
                    best_iteration = i

                total_loss = 0.0
                start_time = time.time()

        model.eval()

        return model, best_part
