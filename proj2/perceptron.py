import logging
import numpy as np
from .core import Classifier

logger = logging.getLogger(__name__)


def fixed_increment_rule(rate: int = 1):
    # if the learning rate is too small, convergence is slow.
    # if the learning rate is too large, can possible diverge.
    return lambda k: rate


FIXED_INCREMENT_RULE = "fixed"
BATCH_RELAXATION_RULE = "relax"
RULES = [FIXED_INCREMENT_RULE, BATCH_RELAXATION_RULE]


class Perceptron(Classifier):
    def __init__(self, rule: str = 'fixed', **kwargs):
        super().__init__(**kwargs)

        if rule == "relax":
            self.method = "relax"
            self.learning_rate = fixed_increment_rule()
            self.training_rule = batch_relaxation
        elif rule == "fixed":
            self.method = "fixed"
            self.learning_rate = fixed_increment_rule()
            self.training_rule = fixed_increment
        else:
            raise KeyError("Unrecognized training rule %s" % rule)

    def train(self, samples: np.ndarray, labels: np.ndarray, **kwargs):
        if not hasattr(self, "labels"): self.labels = np.unique(labels)
        if not hasattr(self, "weights"): self.weights = np.ones((samples.shape[-1] + 1))

        # Begin training
        self.training_rule(self, samples, labels)

    def test(self, samples: np.ndarray, labels: np.ndarray, **kwargs):
        pass

    def classify(self, data: np.ndarray):
        assert hasattr(self, "weights"), "Perceptron must be trained before classification."

        for i in range(data.shape[0]):
            sample = data[i]
            net = np.dot(self.weights[1:, :].T, sample) + self.weights[0]
            return net
        # net_j = np.dot(w_ji[1:,:].T, x) + w_ji[0]
        # y = yfunc(net_j)


    def iter_classify(self, data: np.ndarray):
        pass


def fixed_increment(p: Perceptron, samples: np.ndarray, labels: np.ndarray):
    """
    Fixed-Increment Single-Sample Perceptron rule (Algorithm 5.4 from the Duda book).
    """
    learning = True
    weights = np.ones((samples.shape[-1] + 1))

    print(weights)
    print(samples[0])

    while not learning:
        pass

    return weights


def batch_relaxation(p: Perceptron, samples: np.ndarray, labels: np.ndarray):
    """
    Batch Relaxation with Margin Perceptron rule (Algorithm 5.8 from the Duda book).
    """
    pass
