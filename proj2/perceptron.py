import logging
from enum import Enum

import numpy as np
from typing import Callable, Iterable

from .core import Classifier

logger = logging.getLogger(__name__)


class Strategy(str, Enum):
    OneAgainstRest = "rest"
    OneAgainstOther = "other"


class Rules(str, Enum):
    FixedIncrement = "fixed"
    BatchRelaxation = "relax"

    @property
    def method(self):
        return Rules.get_method(self)

    @classmethod
    def get_method(cls, rule):
        if rule is cls.FixedIncrement:
            return fixed_increment
        if rule is cls.BatchRelaxation:
            return batch_relaxation

        raise KeyError("Unrecognized training rule %s" % rule)


class Perceptron(Classifier):
    def __init__(self, rule: Rules = "fixed", strategy: Strategy = "rest", **kwargs):
        super().__init__(**kwargs)
        self.strategy = Strategy(strategy)
        self.rule = Rules(rule)

        # temporary stuff
        self.learning_rate = make_learning_rate()

    def train(self, samples: np.array, labels: np.array, **kwargs):
        """
        Train this Perceptron.
        :param samples:
        :param labels:
        :param kwargs:
        :return:
        """
        if not hasattr(self, "labels"):
            self.labels = np.unique(labels)
        if not hasattr(self, "weights"):
            self.weights = np.ones((samples.shape[-1] + 1))

        # Begin training
        self.rule.method(self, samples, labels)

    def test(self, samples: np.array, labels: np.array, **kwargs):
        """
        Test this Perceptron.
        :param samples:
        :param labels:
        :param kwargs:
        :return:
        """
        pass

    def classify(self, data: np.array):
        assert hasattr(self, "weights"), "Perceptron must be trained before classification."

        for i in range(data.shape[0]):
            sample = data[i]
            net = np.dot(self.weights[1:, :].T, sample) + self.weights[0]
            return net
            # net_j = np.dot(w_ji[1:,:].T, x) + w_ji[0]
            # y = yfunc(net_j)

    def iter_classify(self, data: np.array):
        pass


def criterion(weights: np.array, errors: Iterable, b: float = 0) -> float:
    dot_weights = lambda y: np.dot(weights[1:].T, y) + weights[0]

    result = np.sum([((dot_weights(y) - b) ** 2) / np.sum(y ** 2) for y in errors])
    return 0.5 * result


def criterion_gradient(weights: np.array, errors: Iterable, margin: float = 0) -> float:
    pass


def make_learning_rate(rate: float = 1) -> float:
    """
    If the learning rate is too small, convergence occurs slowly.
    If it's too large,  learning can possible diverge.
    :param rate:
    """
    return lambda k: rate


def fixed_increment(p: Perceptron, samples: np.array, labels: np.array):
    """
    Fixed-Increment Single-Sample Perceptron rule (Algorithm 5.4 from the Duda book).
    :param p: The perceptron to train.
    :param samples: Training samples.
    :param labels: Training labels.
    :return: The training rates.
    """
    learning = True
    (n, feats) = samples.shape
    weights = np.ones((feats + 1,))
    k = 0

    # while not learning:
    #     pass

    return weights


def batch_relaxation(p: Perceptron, samples: np.array, labels: np.array, rate: Callable[[float], float],
                     margin: float = 0):
    """
    Batch Relaxation with Margin Perceptron rule (Algorithm 5.8 from the Duda book).
    :param p: The perceptron to train.
    :param samples: Training samples.
    :param labels: Training labels.
    :param rate: Training rate.
    :param b: threshold

    :return: The weights from training.
    """
    learning = True
    (n, feats) = samples.shape
    weights = np.ones((feats + 1,))
    k = -1

    while True:
        k = (k + 1) % n
        errors = []
        j = 0
        for j in range(n):
            y = samples[j, :]
            net = np.dot(weights[1:].T, y) + weights[0]
            if net <= margin:
                errors.append(y)
        # update weights
        weights = weights + rate(k)
    pass
