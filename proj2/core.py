import logging
import os
import pickle
import numpy as np
from .util import narray

__all__ = [
    'Classifier',
]

logger = logging.getLogger(__name__)


class Classifier(object):
    """
    Base class for all classifiers, defines the expected functions every classifier should have.
    """

    def __init__(self, **kwargs):
        # self.training_data = training_data
        # classes = training_data[:, 0]
        # n = training_data.shape[0]
        # self.priors = {a: (np.sum(classes == a) / n) for a in np.unique(classes)}
        pass

    def train(self, samples: narray, labels: narray, **kwargs):
        assert not hasattr(super(), 'train')

    def test(self, samples: narray, labels: narray, **kwargs):
        assert not hasattr(super(), 'test')

    def classify(self, data: narray, **kwargs) -> narray:
        return np.array(list(self.iter_classify(data)))

    def iter_classify(self, data: np.ndarray, **kwargs):
        assert not hasattr(super(), 'iter_classify')

    def save(self, location: str, **kwargs):
        fpath = os.path.abspath(location)
        with open(fpath, 'wb') as f:
            pickle.dump(self.__dict__, f)

    @classmethod
    def load(cls, location: str, **kwargs):
        fname = os.path.abspath(location)
        obj = cls.__new__(cls)

        with open(fname, 'rb') as f:
            attributes = pickle.load(f)
            obj.__dict__.update(attributes)

        return obj
