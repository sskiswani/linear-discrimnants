import logging
import numpy as np
from typing import Callable
from .core import Classifier
from .perceptron import Perceptron, TrainingRule, Strategy
from .util import narray

logger = logging.getLogger(__name__)


def error():
    pass


class AdaBoost(Classifier):
    def __init__(self, components: int = 5, rule: TrainingRule = "fixed", strategy: Strategy = "rest", **kwargs):
        super().__init__(**kwargs)
        self.size = components
        self.classifiers = [Perceptron(TrainingRule(rule), Strategy(strategy)) for _ in range(components)]
        self.weights = np.array([1 / components for _ in range(components)])

    def train(self, samples: narray, labels: narray, stages: int = 10, **kwargs):
        alpha = lambda err: 0.5 * np.log((1 - err) / err)
        for k in range(stages):
            pass

    def test(self, samples: narray, labels: narray, **kwargs):
        pass

    def classify(self, data: narray):
        return np.array(list(self.iter_classify(data)))

    def iter_classify(self, data: narray):
        yield 0
