import logging
from enum import Enum
import numpy as np
from typing import Callable, Iterable, Optional
from .core import Classifier

logger = logging.getLogger(__name__)

LearningRate = Callable[[float], float]


class Strategy(str, Enum):
    OneAgainstRest = "rest"
    OneAgainstOther = "other"


class TrainingRule(str, Enum):
    FixedIncrement = "fixed"
    BatchRelaxation = "relax"

    @property
    def method(self):
        return TrainingRule.get_method(self)

    @classmethod
    def get_method(cls, rule):
        if rule is cls.FixedIncrement:
            return fixed_increment
        if rule is cls.BatchRelaxation:
            return batch_relaxation

        raise KeyError("Unrecognized training rule %s" % rule)


class Perceptron(Classifier):
    def __init__(self, rule: TrainingRule = "fixed", strategy: Strategy = "rest", **kwargs):
        super().__init__(**kwargs)
        self.strategy = Strategy(strategy)
        self.rule = TrainingRule(rule)

        # Training data placeholders
        self.labels = None
        self.weights = None
        self.learning_rate = make_learning_rate()

    def train(self, samples: np.array, labels: np.array, **kwargs):
        """
        Train this Perceptron.
        :param samples:
        :param labels:
        :param kwargs:
        :return:
        """
        self.labels = np.unique(labels)

        # Begin training
        self.weights = self.rule.method(samples, labels, self.learning_rate)

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
        return

    def iter_classify(self, data: np.array):
        pass


def criterion(weights: np.ndarray, errors: Iterable[np.ndarray], b: float = 0) -> float:
    values = [(np.dot(weights.T, y) - b) ** 2 / np.sum(y ** 2) for y in errors]
    return 0.5 * np.sum(values)


def criterion_gradient(weights: np.ndarray, errors: Iterable[np.ndarray], b: float = 0) -> float:
    return np.array([y * (np.dot(weights.T, y) - b) / np.sum(y ** 2) for y in errors])


def make_learning_rate(rate: float = 1) -> float:
    """
    If the learning rate is too small, convergence occurs slowly.
    If it's too large,  learning can possible diverge.
    :param rate:
    """
    return lambda k: rate


def fixed_increment(samples: np.array, labels: np.array, rate: LearningRate, weights: np.array = None) -> np.array:
    """
    Fixed-Increment Single-Sample Perceptron rule (Algorithm 5.4 from the Duda book).
    :param samples: Training samples.
    :param labels: Training labels.
    :param rate: Training rate.
    :param weights:
    :return:
    """
    (n, feats) = samples.shape
    if weights is None:
        weights = np.ones((feats,))

    learning = True
    k = 0

    while learning:
        pass

    return weights


def batch_relaxation(samples: np.array, labels: np.array, rate: LearningRate, b: float = 0,
                     weights: np.array = None) -> np.array:
    """
    Batch Relaxation with Margin Perceptron rule (Algorithm 5.8 from the Duda book).
    :param samples: Samples to use for training.
    :param labels: The corresponding labels for samples.
    :param rate: The learning rate.
    :param b:
    :param weights: Initial set of weights to use (default = identity matrix)
    :return: The weights after training.
    """
    if weights is None:
        weights = np.ones((samples.shape[-1],))

    n = samples.shape[0]
    k = 0
    iterations = 0

    while True:
        iterations += 1
        mistakes = []

        # Attempt classification
        for y in samples:
            net = np.dot(weights.T, y)
            if net <= b:
                mistakes.append(y)

        # Update weights
        error = np.sum([y * (b - np.dot(weights.T, y)) / np.sum(y ** 2) for y in mistakes])
        weights = weights + rate(k) * error

        # Terminate on convergence
        if len(mistakes) == 0:
            break
        k = (k + 1) % n

    return weights
