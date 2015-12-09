import logging
import random
import typing
from collections import Counter
from enum import Enum
from itertools import combinations
from time import time

import numpy as np

from .core import Classifier
from .util import narray, get, count_trials

logger = logging.getLogger(__name__)

LearnRateType = typing.Optional[typing.Callable[[int], float]]


class Criterion(str, Enum):
    Jp = "jp"
    Jr = "jr"

    @property
    def method(self):
        if self is Criterion.Jr:
            return Criterion.jr_func
        if self is Criterion.Jp:
            return Criterion.jp_func

    @classmethod
    def jr_func(cls, bias: float, a: narray, errors: narray) -> float:
        result = 0
        for y in errors:
            top = (np.dot(a.T, y) - bias)
            bot = np.linalg.norm(y)
            if np.abs(top) > 1000: top = 1
            result += (top / bot) ** 2
        return 0.5 * result

    @classmethod
    def jp_func(cls, a: narray, errors: narray) -> float:
        return np.sum([np.dot(-a.T, y) for y in errors])

    @classmethod
    def get_method(cls, rule):
        if cls == Criterion.Jr: return Criterion.jr_func
        if cls == Criterion.Jp: return Criterion.jp_func
        raise KeyError("Unrecognized training rule %s" % rule)


class WeakClassifier(object):
    logged = False
    def __init__(self, data: narray, bias=15.0, rate: LearnRateType = None, name: str = None, silent: bool = False,
                 **kwargs):
        self.name = get(name, "WeakClassifier")
        self.criterion = Criterion.Jr
        self.bias = bias
        lrate = 0.03
        self.learn_rate = get(rate, lambda k: lrate)
        self.silent = silent
        if not WeakClassifier.logged:
            WeakClassifier.logged = True
            print('rate: ', lrate, ' bias: ', bias)
        self.weights = self.batch_relaxation(data, **kwargs)

    def log(self, msg):
        if self.silent: return
        logger.info(msg)

    def batch_relaxation(self, data: narray, **kwargs) -> narray:
        is_correct = lambda h, l: (h > self.bias and l > 0) or (h < self.bias and l <= 0)
        max_trials = kwargs.get('max_trials', 100000)

        (n, features) = data.shape
        weights = np.random.random((features,))
        numTrials = 0

        for k in count_trials(n, np.inf):
            numTrials += 1
            errors = []

            # Apply on data set
            for j, x in enumerate(data):
                net = np.dot(weights, x)
                if net <= self.bias:
                    errors.append(x)
                # if not is_correct(net, x[0]): errors.append(x)
            done = (len(errors) == 0)

            crit = Criterion.jr_func(self.bias, weights, errors)
            done = done or np.isclose(crit, 0)

            if (numTrials % 5000) == 0 or numTrials == 1 or done:
                self.log("\tT%i -- errors: %i crit: %f" % (numTrials, len(errors), crit))

            # Calculate the update
            update = 0
            for y in errors:
                top = (self.bias - np.dot(weights.T, y))
                bot = np.linalg.norm(y) ** 2
                update += np.dot(top / bot, y)
            weights += self.learn_rate(k) * update

            if done or numTrials > max_trials:
                break

        self.log("Completed training after %i trials." % numTrials)
        return weights

    def test(self, samples: narray, labels: narray, **kwargs):
        correct = 0
        total = samples.shape[0]
        for (label, pred) in zip(labels, self.classify(samples)):
            if label > 0 and pred > 0: correct += 1
            if label <= 0 and pred <= 0: correct += 1
        self.log("\t%s: %i correct, %i wrong (%.2f%%)" % (self.name, correct, total - correct, 100 * (correct / total)))
        return 1 - (correct / total)

    def classify(self, data: narray, **kwargs):
        return list(self.iter_classify(data))

    def iter_classify(self, data: narray, **kwargs):
        for x in data:
            net = np.dot(self.weights[1:].T, x) + self.weights[0]
            yield (-1 if net <= self.bias else 1)
            # yield (-1 if net <= 0 else 1)

class AdaBoost(object):
    def __init__(self, labels: tuple = None, size: int = 10, silent: bool = False, **kwargs):
        self.size = size
        self.label_set = get(labels, (-1, -1))
        self.classifiers = []
        self.silent = silent

    def log(self, msg):
        if self.silent: return
        logger.info(msg)

    def train(self, samples: narray, labels: narray, label_set=None):
        log_fmt = "WC%i -- train_size: %i test_size: %i miss_classf: %i (err: %f err2: %f alpha=%r)"
        is_correct = lambda h, l: (h > 0 and l > 0) or (h <= 0 and l <= 0)
        classes = np.unique(labels)
        self.label_set = get(label_set, (classes[0], classes[1]))
        weights = []

        # Grab values we're interested in
        def filtered(normalize=True):
            neg_label = self.label_set[1]
            for (label, x) in zip(labels, samples):
                if label not in self.label_set:
                    continue

                val = np.array([label] + x.tolist())
                if normalize:
                    val[0] = 1
                    if label == neg_label:
                        val *= -1
                yield val

        set_data = np.array(list(filtered()))
        (n, features) = set_data.shape
        distro = np.array([1 / n for _ in range(n)])

        for k in range(self.size):
            # Train weak leaner Ck
            training_samples = sample_from(set_data, distro)
            weak = WeakClassifier(set_data, name="WC%i" % k, silent=self.silent)

            # Get training error of Ck
            # testing_samples = set_data
            testing_samples = sample_from(set_data, distro)
            hk = np.array(weak.classify(np.array([x[0] * x[1:] for x in testing_samples])))
            wrong = np.sum(0 if is_correct(h, y) else 1 for (h, y) in zip(hk, testing_samples[:, 0]))

            # Get indices for testing samples
            consolidated = []
            for h, y in zip(hk, testing_samples):
                for i, x in enumerate(set_data):
                    if not np.all(x == y): continue
                    consolidated.append((i, h, y))

            # Update distribution
            err = wrong / testing_samples.shape[0]
            err2 = np.sum([distro[i] * np.exp(-y[0] * h) for i, h, y in consolidated])
            # err = err2
            # if np.isclose(err, 0): err = 1e-6
            # coeff = ((1 - err) / err)
            # alpha = 0 if coeff == 0 else 0.5 * np.log(coeff)
            alpha = 0.5 * np.log((1-err2)/err2)

            # Update distribution
            for i, h, y in consolidated:
                distro[i] *= np.exp(-alpha * y[0] * h)

            self.log(log_fmt % (k, training_samples.shape[0], testing_samples.shape[0], wrong, err, err2, alpha))

            if not np.isclose(np.sum(distro), 1.):
                distro /= np.sum(distro)

            weights.append(alpha)
            self.classifiers.append(weak)

        self.weights = np.array(weights)
        return self

    def test(self, samples: narray, labels: narray, **kwargs):
        filtered = np.array([[label] + x.tolist() for (label, x) in zip(labels, samples) if label in self.label_set])
        correct = 0
        total = filtered.shape[0]
        for (label, h) in zip(filtered[:, 0], self.classify(filtered[:, 1:])):
            pred = self.label_set[0] if h > 0 else self.label_set[1]
            if label > 0 and pred > 0: correct += 1
            if label <= 0 and pred <= 0: correct += 1
        acc = 100 * (correct / total)
        self.log("AdaBoost(%r) %i correct of %i (%.2f%%)" % (self.label_set, correct, total, acc))

    def classify(self, data: narray, **kwargs):
        return list(self.iter_classify(data))

    def iter_classify(self, data: narray, **kwargs):
        wc_hk = np.array([wc.classify(data) for wc in self.classifiers])
        for hk in wc_hk.T:
            net = np.dot(self.weights, hk)
            yield 1 if net > 0 else -1

    def __str__(self):
        return "AdaBoost[%i,%i]" % self.label_set


class MulticlassAdaBoost(Classifier):
    def __init__(self, size: int = 10, **kwargs):
        super().__init__(**kwargs)
        self.silenceChildren = kwargs.get('silenceChildren', True)
        self.size = size
        self.classifiers = {}

    def train(self, samples: narray, labels: narray, **kwargs):
        begin = time()
        classes = np.unique(labels)
        self.sets = list(combinations(classes, 2))
        for lset in list(combinations(classes, 2)):
            start = time()
            ada = AdaBoost(lset, self.size, silent=self.silenceChildren)
            self.classifiers[lset] = ada.train(samples, labels, label_set=lset)
            end = time()
            logger.info("Trained %s in %.2fs" % (ada, (end - start)))
        delta = time() - begin
        logger.info("Trained all AdaBoost classifiers in %.2f seconds." % (delta))

    def test(self, samples: narray, labels: narray, **kwargs):
        # Test ensemble
        correct = 0
        total = samples.shape[0]

        for label, pred in zip(labels, self.classify(samples)):
            if label == pred:
                correct += 1

        acc = 100 * (correct / total)
        logger.info("MCAdaBoost: %i correct of %i (%.2f%%)" % (correct, total, acc))

        # Test individuals
        for lset in self.sets:
            ada = self.classifiers[lset]
            ada.silent = False
            ada.test(samples, labels)

    def classify(self, data: narray, **kwargs):
        return list(self.iter_classify(data))

    def iter_classify(self, data: narray, **kwargs):
        predictions = np.array(
                [[lset[0 if p > 0 else 1] for p in ada.classify(data)]
                 for lset, ada in self.classifiers.items()])

        for pred in predictions.T:
            yield Counter(pred).most_common(1)[0][0]


def sample_from(data, weights: narray, min_size=1, cap=100000) -> narray:
    min_size = max(min_size, 1)
    vals, choices = [], []
    attempt = 0
    while len(vals) < min_size:
        prob = random.random() * np.sum(weights)
        attempt += 1
        if attempt >= cap: break
        while prob > 0:
            c = np.random.choice(data.shape[0], p=weights)
            choices.append(c)
            vals.append(data[c])
            prob -= weights[c]
    return np.array(vals)


def run(train_data: narray, test_data: narray, ada_size: int = 10):
    mc = MulticlassAdaBoost(size=ada_size, silenceChildren=False)
    mc.train(train_data[:, 1:], train_data[:, 0])
    mc.test(test_data[:, 1:], test_data[:, 0])
    return mc

    # result = []
    # for lset in [(8, 9), (1, 2), (1, 3), (2, 3)]:
    #     ada = AdaBoost(lset, ada_size)
    #     ada.train(train_data[:, 1:], train_data[:, 0], label_set=lset)
    #     ada.test(test_data[:, 1:], test_data[:, 0])
    #     exit()
    #     result.append(ada)
    # return result


def run_individual(train_data: narray, test_data: narray, ada_size: int = 10):
    result = []
    for lset in [(0, 3), (1, 2), (1, 3), (2, 3)]:
        ada = AdaBoost(lset, ada_size)
        ada.train(train_data[:, 1:], train_data[:, 0], label_set=lset)
        ada.test(test_data[:, 1:], test_data[:, 0])
        result.append(ada)
    return result
