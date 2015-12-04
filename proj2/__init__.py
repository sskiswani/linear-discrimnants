import logging
import os

import numpy as np

from . import perceptron, adaboost
from . import util, core

logger = logging.getLogger()
np.set_printoptions(precision=3, suppress=True, linewidth=180)
# np.set_printoptions(linewidth=180)

_all_ = [
    'adaboost',
    'core',
    'perceptron',
    'util'
]

DEFAULTS = {
    "fixed": {
        "wine": {},
        "digits": {"rate": 1}
    },
    "relax": {
        "wine": {"rate": 1},
        "digits": {"rate": 1}
    },
    "rest": {
        "wine": {},
        "digits": {}
    },
    "other": {
        "wine": {},
        "digits": {}
    },
    "ada": {
        "wine": {},
        "digits": {}
    },
    "svm": {
        "wine": {},
        "digits": {}
    },
    "kern": {
        "wine": {},
        "digits": {}
    },
    "samme": {
        "wine": {},
        "digits": {}
    }
}
METHODS = DEFAULTS.keys()


def set_logger(verbosity: int = 5):
    level = list(range(6))[5 - verbosity % 6] * 10
    logging.basicConfig(level=level, format="[%(asctime)s.%(msecs)03d][%(levelname)s][%(name)s] %(message)s",
                        datefmt='%H:%M:%S')
    logging.info('Set logging level to %s (%i)' % (logging.getLevelName(level), level))


def run(method, training_file, testing_file, verbose: int = 0, cache: bool = False, **kwargs):
    set_logger(verbose)
    logging.info('Loading files <%s> and <%s>' % (training_file, testing_file))

    # Parse files.
    train_data = np.genfromtxt(training_file)
    test_data = np.genfromtxt(testing_file)

    if method == "fixed" or method == "relax":
        cpath = 'bin/classf_cache/ptron_%s_%s.picl' % (method, os.path.basename(training_file).split('_')[0])

        if cache and os.path.exists(cpath):
            logging.info("Loading classifer at path <%s>" % (cpath))
            classf = perceptron.Perceptron.load(cpath)
        else:
            if cache:
                logging.info("Coulnd't find classifier in cache, training a new one.")
            classf = perceptron.Perceptron(method)
            classf.train(train_data[:, 1:], train_data[:, 0])
            if cache:
                logging.info("Caching Perceptron to <%s>" % (cpath))
                classf.save(cpath)

        classf.test(test_data[:, 1:], test_data[:, 0])
