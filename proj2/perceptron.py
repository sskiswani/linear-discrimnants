import logging
import time
from enum import Enum
from itertools import cycle
from typing import Callable, Iterable, Union

import numpy as np

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
    def __init__(self, rule: TrainingRule = "fixed",
                 strategy: Strategy = "rest",
                 learn_rate: LearningRate = None,
                 **kwargs):
        super().__init__(**kwargs)
        self.strategy = Strategy(strategy)
        self.rule = TrainingRule(rule)
        self.weights = {}
        self.learning_rate = make_learning_rate(0.01) if learn_rate is None else learn_rate

    def train(self, samples: narray, labels: narray, **kwargs):
        logger.info("Training using rule %s" % self.rule)
        start = time.time()
        for i, label in enumerate(np.unique(labels)):
            begin = time.time()
            normalized = normalize_features(label, samples, labels)
            self.weights[label] = self.rule.method(normalized, rate=self.learning_rate, debug=True)
            end = time.time()
            logger.info("Training for class %s took %.3fs:\n%r" % (label, (end - begin), self.weights[label]))
        end = time.time()
        logger.info("Total training time: %.3f" % (end - start))

    def test(self, samples: narray, labels: narray, **kwargs):
        """
        Test this Perceptron.
        :param samples: Testing samples.
        :param labels: Labels corresponding to testing samples.
        :param kwargs:
        :return:
        """
        correct = 0
        for trial, (true, classf) in enumerate(zip(labels, self.iter_classify(samples))):
            if true == classf:
                correct += 1

        # Output statistics
        total = samples.shape[0]
        acc = (correct / total) * 100.0
        print("Got %i correct out of %i (%.2f accuracy)" % (correct, total, acc))

        for label, weight in self.weights.items():
            print("w[%i] == %s" % (label, weight))

    def classify(self, data: narray, debug: bool = False) -> narray:
        return np.array(list(self.iter_classify(data)))

    def iter_classify(self, data: narray, debug: bool = False):
        assert hasattr(self, "weights"), "Perceptron must be trained before classification."
        for x in data:
            values = np.array([[label, np.dot(a[1:].T, x) + a[0]] for label, a in self.weights.items()])
            best = np.argmax(values[:, 1])

            if debug:
                yield values[best][0], values
            else:
                yield values[best][0]

    @property
    def labels(self):
        return self.weights.keys()


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


def fixed_increment(samples: narray, debug: bool = True, **kwargs) -> np.array:
    """
    Fixed-Increment Single-Sample Perceptron rule (Algorithm 5.4 from the Duda book).
    :param samples: Normalized augmented features for training
    :param debug: Whether or not to output debugging information.
    :param kwargs:
    :return: The trained weights.
    """
    signum = lambda x: np.sign(x) >= 0
    (n, features) = samples.shape
    weights = np.random.rand(features, 1).reshape((features,))
    trial = 0

    while True:
        trial += 1
        errors = []

        for y in samples:
            net = np.dot(weights.T, y)
            if signum(y[0]) != signum(net):
                weights = weights + np.sign(y[0]) * y
                errors.append(y)

        # Terminate on convergence
        if len(errors) == 0:
            break

        if debug and trial % 5000 == 0:
            print("Trial %i: %i errors" % (trial, len(errors)))

    logger.info("Completed training after %i trials." % trial)
    return weights


def batch_relaxation(samples: narray,
                     rate: LearningRate,
                     margin: float = 1,
                     debug: bool = True,
                     **kwargs) -> np.array:
    """
    Batch Relaxation with Margin Perceptron rule (Algorithm 5.8 from the Duda book).
    :param samples: Samples to use for training.
    :param rate: The learning rate.
    :param margin: Margin specifying plane such that: dot(weights.T, y) >= margin
    :param debug: Whether or not to output debugging information.
    :param kwargs:
    :return: The weights after training.
    """
    if not (0 < margin < 2):
        raise ValueError("Margin must be in the range (0,2), got %f" % margin)
    (n, features) = samples.shape
    weights = np.random.rand(features, 1).reshape((features,))
    trial = 0

    # TODO: Remove hardcoded rate
    rate = lambda x: 0.01

    for k in cycle(range(n)):
        errors = []

        # Attempt classification
        for y in samples:
            net = np.dot(weights.T, y)
            if (net <= margin and y[0] > 0) or (net > margin and y[0] < 0):
                # print("got %f for %f" % (net, y[0]))
                errors.append(np.copy(y))

        # Terminate on convergence
        if len(errors) == 0:
            break
        if debug and trial % 5000 == 0:
            print("Trial %i: %i errors" % (trial, len(errors)))

        # Update weights
        correction = np.sum(
                [np.sign(x[0]) * x * (margin - np.dot(weights.T, x)) / np.linalg.norm(x) ** 2 for x in errors],
                axis=0)
        weights = weights + rate(k) * correction
        trial += 1

    logger.info("Completed training after %i trials." % trial)
    return weights
