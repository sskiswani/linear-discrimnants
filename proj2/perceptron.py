import logging
import time
from enum import Enum
from itertools import cycle
from typing import Callable, Iterable
import numpy as np
from .core import Classifier
from .util import narray, dist, approx

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
    def __init__(self, rule: TrainingRule = "fixed",
                 strategy: Strategy = "rest",
                 learn_rate: float = 0.05,
                 **kwargs):
        super().__init__(**kwargs)
        self.strategy = Strategy(strategy)
        self.rule = TrainingRule(rule)
        self.weights = {}
        self.learning_rate = learn_rate

    def train(self, samples: narray, labels: narray, **kwargs):
        logger.info("Training using rule %s" % self.rule)
        start = time.time()

        for label in np.unique(labels):
            begin = time.time()
            normalized = normalize_features(label, samples, labels)
            self.weights[label] = self.rule.method(normalized, rate=self.learning_rate, debug=True)
            end = time.time()
            logger.info("Training for class %s took %.3fs" % (label, (end - begin)))

        end = time.time()
        logger.info("Total training time: %.3f" % (end - start))

    def test(self, samples: narray, labels: narray, **kwargs):
        """
        Test this Perceptron.
        :param samples: Testing samples.
        :param labels: Labels corresponding to testing samples.
        :param kwargs: extra arguments
        """
        correct = 0
        labs = [0 for _ in range(len(self.weights))]
        for trial, (true, (classf, vals)) in enumerate(zip(labels, self.iter_classify(samples, debug=True))):
            if self.strategy == Strategy.OneAgainstRest:
                for i in range(len(vals)):
                    if (true == vals[i][0] and vals[i][1] > 0) or (true != vals[i][0] and vals[i][1] <= 0):
                        labs[i] += 1

            if true == classf:
                correct += 1

        total = samples.shape[0]
        if self.strategy == Strategy.OneAgainstRest:
            for i, label in enumerate(self.weights.keys()):
                acc = (labs[i] / total) * 100.0
                print("Class %i vs rest: %i correct out of %i (%.2f accuracy)" % (label, labs[i], total, acc))

        # Output statistics
        acc = (correct / total) * 100.0
        print("Got %i correct out of %i (%.2f accuracy)" % (correct, total, acc))

        # Output the weights used
        for label, weight in self.weights.items():
            print("w[%i] ==\n%s" % (label, weight))

    def classify(self, data: narray, debug: bool = False) -> narray:
        return np.array(list(self.iter_classify(data)))

    def iter_classify(self, data: narray, debug: bool = False):
        assert hasattr(self, "weights"), "Perceptron must be trained before classification."
        for x in data:
            values = np.array([[label, np.dot(a[1:], x) + a[0]] for label, a in self.weights.items()])
            best = np.argmax(values[:, 1])

            if debug:
                yield values[best][0], values
            else:
                yield values[best][0]

    @property
    def labels(self):
        return self.weights.keys()


class MulticlassPerceptron(Classifier):
    def __init__(self, rule: TrainingRule = "fixed", strategy: Strategy = "rest", learn_rate: float = 0.05, **kwargs):
        super().__init__(**kwargs)
        self.margin = kwargs.get('margin', 1.0)
        self.learning_rate = learn_rate
        self.strategy = Strategy(strategy)
        self.rule = TrainingRule(rule)
        self.weights = {}

    def train(self, samples: narray, labels: narray, **kwargs):
        logger.info("Training using rule %s" % self.rule)
        start = time.time()

        if self.strategy == Strategy.OneAgainstRest:
            self.weights = self.relax_training(samples, labels, rate=self.learning_rate, margin=100)

            # if self.rule == TrainingRule.BatchRelaxation:
            #     self.weights = self.relax_training(samples, labels, rate=self.learning_rate)
            # else:
            #     self.weights = self.fixed_increment(samples, labels)

        end = time.time()
        logger.info("Total training time: %.3f" % (end - start))

    def fixed_increment(self, samples: narray, labels: narray, **kwargs):
        max_trials = kwargs.get('max_trials', 100000)
        (n, features) = samples.shape
        classes = np.unique(labels)
        weights = {c: np.zeros((features,)) for c in classes}

        trial = -1
        while True:
            trial += 1
            if trial > max_trials:
                break

            errors = []

            # Attempt classification
            for x, l in zip(samples, labels):
                net_max = -np.inf
                pred = None

                for (c, w) in weights.items():
                    net = np.sum(np.dot(w, x))
                    if net > net_max:
                        net_max = net
                        pred = (c, w)

                if pred[0] != l:
                    errors.append(x)
                    weights[l] += x
                    weights[pred[0]] -= x

            if trial % 1000 == 0:
                logger.debug("Trial %i: %i errors" % (trial, len(errors)))

            if len(errors) == 0:
                break

        logger.info("Completed training after %i trials." % trial)
        return weights

    def relax_training(self, samples: narray, labels: narray, rate: float = 1.5, margin: float = 3.0, **kwargs):
        max_trials = kwargs.get('max_trials', 100000)
        tolerance = kwargs.get('tolerance', 0.0001)
        (n, features) = samples.shape
        classes = np.unique(labels)
        weights = {c: np.zeros((features,)) for c in classes}

        better, worse = 0, 0
        trial = -1

        old_delta = np.inf
        while trial < max_trials:
            trial += 1
            errors = {c: [] for c in classes}
            total_errors = 0

            # Attempt classification
            for x, l in zip(samples, labels):
                values = np.array([[label, np.dot(weight, x)] for label, weight in weights.items()])
                best = np.argsort(values[:, 1])[::-1]
                pred, net = values[best][0]
                
                if pred != l:
                    errors[pred].append(-x)
                    errors[l].append(x)
                    total_errors += 1

            # Terminate if no errors are encountered
            if total_errors == 0:
                break

            # Update weights based on training results
            delta = 0
            for c in classes:
                update = rate * np.sum([x * (margin - np.dot(weights[c], x)) / np.sum(x ** 2) for x in errors[c]],
                                       axis=0)
                delta += np.abs(np.sum(update))
                weights[c] += update

            if delta < old_delta:
                better += 1
            else:
                worse += 1

            if delta <= tolerance:
                logger.debug("Early termination due to small update.")
                break

            old_delta = delta

            if trial % 1000 == 0:
                logger.debug("Trial %i: %i errors (delta: %f) better: %i worse: %i (rate %f)" % (
                    trial, total_errors, delta, better, worse, rate))
                better = worse = 0

        logger.info("Completed training after %i trials." % trial)
        return weights

    def test(self, samples: narray, labels: narray, **kwargs):
        correct, total = 0, samples.shape[0]

        for pred, actual in zip(self.iter_classify(samples), labels):
            if pred == actual:
                correct += 1

        accuracy = 100. * (correct / total)
        print("Multiclass Perceptron: %i correct out of %i (%.2f accuracy)" % (correct, total, accuracy))

        # Output the weights used
        for label, weight in self.weights.items():
            print("w[%i] ==\n%s" % (label, weight))

    def iter_classify(self, data: narray, **kwargs):
        for x in data:
            values = np.array([[label, np.dot(weight, x)] for label, weight in self.weights.items()])
            best = np.argmax(values[:, 1])
            yield values[best][0]
            # if debug:
            #     yield values[best][0], values
            # else:
            #     yield values[best][0]


def perceptron_criterion(weights: narray, errors: Iterable[narray]) -> float:
    return np.sum([-1 * np.vdot(weights, y) for y in errors])


def criterion(weights: narray, errors: Iterable[narray], margin: float = 0) -> float:
    values = np.array([((np.dot(weights, y) - margin) / np.linalg.norm(y)) ** 2 for y in errors])
    return 0.5 * np.sum(values)


def criterion_gradient(weights: narray, errors: Iterable[narray], b: float = 0) -> float:
    return np.array([y * (np.dot(weights, y) - b) / np.sum(y ** 2) for y in errors])


def normalize_features(cls: int, samples: narray, labels: narray, weight: float = 1, negative_weight: float = -1):
    (n, features) = samples.shape
    result = np.ones((n, features + 1))
    for i in range(n):
        result[i][1:] = samples[i]
        if labels[i] != cls:
            result[i] *= -1
    return result


def fixed_increment(samples: narray, theta: float = 0.000003, debug: bool = True,
                    **kwargs) -> np.array:
    """
    Fixed-Increment Single-Sample Perceptron rule (Algorithm 5.4 from the Duda book).
    :param samples: Normalized augmented features for training
    :param theta: threshold
    :param debug: Whether or not to output debugging information.
    :param kwargs:
    :return: The trained weights.
    """
    (n, features) = samples.shape
    weights = np.zeros((features,))
    trial = -1
    best = np.inf
    bestd = np.inf
    bw = weights

    while True:
        trial += 1
        errors = []
        old_weights = weights.copy()

        for k, y in enumerate(samples):
            net = np.sum(weights * y)
            if net <= 0:
                weights = weights + y
                errors.append(y)

        # Terminate on convergence
        if len(errors) == 0:
            break

        # Grab debug values
        if len(errors) < best:
            best = len(errors)
            bw = old_weights

        if trial > 100000:
            weights = bw
            break

        delta = dist(weights, old_weights)
        if delta < bestd:
            bestd = delta
        if delta <= theta:
            break
        if debug and trial % 5000 == 0:
            logger.debug("Trial %i: %i errors (best %i) (delta: %f)" % (trial, len(errors), best, delta))

    logger.info("Completed training after %i trials." % trial)
    return weights


def batch_relaxation(samples: narray,
                     rate: float = 0.02,
                     margin: float = 2,
                     theta: float = 0.001,
                     debug: bool = True,
                     **kwargs) -> np.array:
    """
    Batch Relaxation with Margin Perceptron rule (Algorithm 5.8 from the Duda book).
    :param samples: Samples to use for training.
    :param rate: The learning rate
    :param margin: Margin specifying plane such that: dot(weights, y) >= margin
    :param theta: Stopping threshold
    :param debug: Whether or not to output debugging information
    :param kwargs: Any other arguments
    :return: The weights after training.
    """
    if not (0 < rate < 2):
        raise ValueError("Rate must be in the range (0,2), got %f" % rate)
    (n, features) = samples.shape
    weights = np.random.rand(features, 1).reshape((features,))
    trial = 0
    best = np.inf
    bd = np.inf

    for _ in cycle(range(n)):
        errors = []

        # Attempt classification
        for y in samples:
            net = np.sum(weights * y)
            if net <= margin:
                errors.append(np.copy(y))

        # Terminate on convergence
        if len(errors) == 0:
            break

        # Update weights
        update = rate * np.sum([x * (margin - np.dot(weights, x)) / np.sum(x ** 2) for x in errors], axis=0)
        old_weights = np.copy(weights)
        weights = weights + update

        # Rate limiting
        if trial > 100000:
            break

        # Grab debug values
        if len(errors) < best:
            best = len(errors)
        delta = dist(old_weights, weights)
        if delta < bd:
            bd = delta

        # Terminate on small update intervals
        if approx(delta, 0, 0.00001) and approx(criterion(old_weights, errors, margin), 0, 0.00001):
            break

        # Output debug information
        if trial % 5000 == 0:
            if debug:
                crit = criterion(weights, errors, margin)
                logger.debug("Trial %i: %i errors (best %i, best delta: %f) rate %.2f crit: %f" % (
                    trial, len(errors), best, bd, rate, crit))

        trial += 1

    logger.info("Completed training after %i trials." % trial)
    return weights
