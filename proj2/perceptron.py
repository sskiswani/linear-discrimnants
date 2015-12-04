import logging
import numpy as np
from .core import Classifier

logger = logging.getLogger(__name__)


class Perceptron(Classifier):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def test(self, samples, labels):
        pass

    def classify(self, data):
        pass

    def iter_classify(self, data):
        pass