import logging
from enum import Enum

import numpy as np
from typing import Callable, Iterable, Union

from .core import Classifier

logger = logging.getLogger(__name__)

LearningRate = Callable[[float], float]
narray = Union[np.ndarray]


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
        self.discriminants = {}
        self.learning_rate = make_learning_rate()

    def train(self, samples: narray, labels: narray, **kwargs):
        for i, label in enumerate(np.unique(labels)):
            self.discriminants[label] = fixed_increment(normalize_features(label, samples, labels))
            logger.info("Weights after training for class %s: %r" % (label, self.discriminants[label]))

    def fixed(self, samples: narray, labels: narray) -> np.array:
        self.weights = []

        for i in range(self.labels.shape[0]):
            print(i)

    def test(self, samples: narray, labels: narray, **kwargs):
        """
        Test this Perceptron.
        :param samples:
        :param labels:
        :param kwargs:
        :return:
        """
        pass

    def classify(self, data: narray):
        assert hasattr(self, "weights"), "Perceptron must be trained before classification."
        return

    def iter_classify(self, data: narray):
        pass

    @property
    def labels(self):
        return self.discriminants.keys()


def criterion(weights: narray, errors: Iterable[narray], b: float = 0) -> float:
    values = [(np.dot(weights.T, y) - b) ** 2 / np.sum(y ** 2) for y in errors]
    return 0.5 * np.sum(values)


def criterion_gradient(weights: narray, errors: Iterable[narray], b: float = 0) -> float:
    return np.array([y * (np.dot(weights.T, y) - b) / np.sum(y ** 2) for y in errors])


def make_learning_rate(rate: float = 1) -> float:
    """
    If the learning rate is too small, convergence occurs slowly.
    If it's too large,  learning can possible diverge.
    :param rate:
    """
    return lambda k: rate


def normalize_features(cls: int, samples: narray, labels: narray, weight: float = 1, negative_weight: float = -1):
    (n, features) = samples.shape
    result = np.ones((n, features + 1))

    for i in range(n):
        result[i][1:] = samples[i]
        result[i][0] = weight if labels[i] == cls else negative_weight

    return result


def fixed_increment(samples: narray, rate: float = 1) -> np.array:
    """
    Fixed-Increment Single-Sample Perceptron rule (Algorithm 5.4 from the Duda book).
    :param samples: Normalized augmented features for training
    :param rate: The learning rate
    :return: The trained weights.
    """
    (n, features) = samples.shape
    weights = np.ones((features,))
    old_weights = np.copy(weights)
    trial = 0

    while True:
        trial += 1
        errors = 0

        for y in samples:
            net = np.dot(weights.T, y)
            correct = (y[0] > 0 and net >= 0) or (y[0] < 0 and net < 0)

            if not correct:
                # print("ERROR %i ==========================" % errors)
                # print("\tnet is %f and label is %f. error? %s" % (net, y[0], correct == False))
                # print("\tgot weights: %s" % weights)
                # print("\tgot samp: %s" % y)
                # print("\tw + y == %s" % (weights + y))
                weights += y
                # print("\tnow weights: %s" % weights)
                # print("\tnow samp: %s" % y)
                errors += 1

        if errors == 0:
            break

        if True:
            # if trial == 1 or trial % 500 == 0:
            print("Trial %i (%i misclassifications)\n\told: %s\n\tnew: %s\n\tdelta: %s" % (
            trial, errors, old_weights, weights, (old_weights - weights)))
            old_weights = np.copy(weights)

            # print("Trial %i: Errors: %i weights: %r" % (trials, errors, weights))

    print("Completed training after %i trials." % trial)
    return weights


def batch_relaxation(samples: narray, labels: narray, rate: LearningRate, b: float = 0,
                     weights: narray = None) -> np.array:
    """
    Batch Relaxation with Margin Perceptron rule (Algorithm 5.8 from the Duda book).
    :param samples: Samples to use for training.
    :param labels: The corresponding labels for samples.
    :param rate: The learning rate.
    :param b:
    :param weights: Initial set of weights to use (default = identity matrix)
    :return: The weights after training.
    """
    pass
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
        k = (k + 1) % n

        # Terminate on convergence
        if len(mistakes) == 0:
            break

    return weights
