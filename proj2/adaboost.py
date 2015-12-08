import logging
import numpy as np
import random
from .core import Classifier
from .perceptron import batch_relaxation, normalize_features
from .util import narray

logger = logging.getLogger(__name__)


class WeakClassifier(Classifier):
    def __init__(self, classes, **kwargs):
        super().__init__(**kwargs)
        self.weights = np.array([])

    def train(self, samples: narray, labels: narray, margin: float = 1.0, max_trials: int = 100000, **kwargs) -> narray:
        # Batch Relaxation with Margin Perceptron rule (Algorithm 5.8 from the Duda book).
        pass

    def test(self, samples: narray, labels: narray, **kwargs):
        pass

    def iter_classify(self, data: narray, **kwargs):
        yield 0


class AdaBoost(Classifier):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.classes = []
        self.classifiers = []
        self.weights = []

    def train(self, samples: narray, labels: narray, components: int = 10, **kwargs):
        self.classes = [1, 2]
        combined = np.zeros((samples.shape[0], samples.shape[1] + 1))
        combined[:, 0] = np.array(labels)
        combined[:, 1:] = np.array(samples)

        cls_samples = np.array([x for x in combined if x[0] in self.classes])
        (n, features) = cls_samples.shape

        weights = np.zeros((features,))
        distro = np.array([1 / n for _ in range(n)])

        for k in range(components):
            training_samples = sample(cls_samples, distro)
            logger.info("C%i -- training samples: %i" % (k, training_samples.shape[0]))
            weights = batch_relaxation(training_samples[:, 1:], plot=False)

            # Test the weak classifier
            testing_samples = sample(cls_samples, distro)
            for x in testing_samples:
                net = np.dot(weights, x[:, 1:])
                # if net <= 0 and:

            a = 0.5 * np.log((1 - err) / err)

            # a = 0.5 * np.log((1 - err) / err)
            self.classifiers.append(weights)

    def test(self, samples: narray, labels: narray, **kwargs):
        pass

    def classify(self, data: narray, **kwargs) -> narray:
        return np.array(list(self.iter_classify(data)))

    def iter_classify(self, data: narray, **kwargs):
        yield 0


def sample(data, weights: narray) -> narray:
    p = random.random()
    vals = []
    while p > 0:
        c = np.random.choice(data.shape[0], p=weights)
        vals.append(data[c])
        p -= weights[c]
    return np.array(vals)
