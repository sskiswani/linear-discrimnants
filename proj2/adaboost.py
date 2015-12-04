import logging
import numpy as np
from .core import Classifier
from .perceptron import Perceptron, TrainingRule, Strategy
from .util import narray

logger = logging.getLogger(__name__)


class AdaBoost(Classifier):
    def __init__(self, size: int = 5, rule: TrainingRule = "fixed", strategy: Strategy = "rest", **kwargs):
        super().__init__(**kwargs)
        self.learners = [Perceptron(TrainingRule(rule), Strategy(strategy)) for _ in range(size)]
        self.W = {}

    def train(self, samples: narray, labels: narray, **kwargs):
        pass

    def test(self, samples: narray, labels: narray, **kwargs):
        pass

    def classify(self, data: narray):
        return np.array(list(self.iter_classify(data)))

    def iter_classify(self, data: narray):
        yield 0
