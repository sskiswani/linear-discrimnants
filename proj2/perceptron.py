import logging
from itertools import cycle
from enum import Enum
from typing import Callable, Iterable, Union
import numpy as np
from .core import Classifier
from .util import approx

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
    def __init__(self, rule: TrainingRule = "fixed",
                 strategy: Strategy = "rest",
                 learn_rate: LearningRate = None,
                 **kwargs):
        super().__init__(**kwargs)
        self.strategy = Strategy(strategy)
        self.rule = TrainingRule(rule)
        # self.learning_rate = lambda x: 0.01
        # self.learning_rate = make_learning_rate() if learn_rate is None else learn_rate

        # Training data placeholders
        self.discriminants = {}

    def train(self, samples: narray, labels: narray, **kwargs):
        logger.info("Training using rule %s" % self.rule)

        for i, label in enumerate(np.unique(labels)):
            normalized = normalize_features(label, samples, labels)
            self.discriminants[label] = self.rule.method(normalized)
            logger.info("Weights after training for class %s: %r" % (label, self.discriminants[label]))

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


def fixed_increment(samples: narray, rate: float = 0.01, **kwargs) -> np.array:
    """
    Fixed-Increment Single-Sample Perceptron rule (Algorithm 5.4 from the Duda book).
    :param samples: Normalized augmented features for training
    :param rate: The learning rate.
    :param kwargs:
    :return: The trained weights.
    """
    signum = lambda x: np.sign(x) >= 0
    (n, features) = samples.shape
    weights = np.random.rand(features, 1).reshape((features,))
    trial = 0
    accuracy = []

    while True:
        trial += 1
        errors = 0

        for k, y in enumerate(samples):
            net = np.dot(weights.T, y)

            if signum(y[0]) != signum(net):
                # print("missclassified (net: %s, y[0]: %f)\n" % (net,y[0]), y)
                weights = weights + np.sign(y[0]) * rate * y
                # weights = weights + rate(k) * y
                # weights = weights + y
                errors += 1

        accuracy.append(errors)
        if errors == 0:
            print("Trial %i: %i errors" % (trial, errors))
            break

        if trial == 1 or trial % 500 == 0:
            print("Trial %i: %i errors" % (trial, errors))

    logger.info("Completed training after %i trials." % trial)
    return weights


def batch_relaxation(samples: narray, rate: LearningRate, margin: float = 0.01, **kwargs) -> np.array:
    """
    Batch Relaxation with Margin Perceptron rule (Algorithm 5.8 from the Duda book).
    :param samples: Samples to use for training.
    :param rate: The learning rate.
    :param margin: Margin specifying plane such that: dot(weights.T, y) >= margin
    :param kwargs:
    :return: The weights after training.
    """
    if not (0 < margin < 2):
        raise ValueError("Margin must be in the range (0,2), got %f" % margin)
    (n, features) = samples.shape
    weights = np.random.rand(features, 1).reshape((features,))
    trial = 0

    for k in cycle(range(n)):
        errors = []

        # Attempt classification
        for y in samples:
            net = np.dot(weights.T, y)

            if net <= margin or np.allclose(net, margin):
                # print("got error for value %f (label: %f)" % (net, y[0]))
                errors.append(np.copy(y))

        # Terminate on convergence
        if len(errors) == 0:
            break

        # Update weights
        correction = np.sum([x * (margin - np.dot(weights.T, x)) / np.linalg.norm(x) ** 2 for x in errors], axis=0)
        weights = weights + rate(k) * correction

        logger.info(
            "Finished trial %i with %i errors (weights: %s) correction: %s" % (trial, len(errors), weights, correction))
        trial += 1

    return weights
