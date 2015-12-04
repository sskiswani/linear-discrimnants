import logging

import numpy as np

from . import core
from . import perceptron, adaboost

logger = logging.getLogger()
np.set_printoptions(precision=3, suppress=True, linewidth=80)

_all_ = [
    'adaboost',
    'core',
    'perceptron'
]

METHODS = [
    "fixed",
    "relax",
    "rest",
    "other",
    "ada",
    "svm",
    "kern",
    "samme"
]


def set_logger(verbosity: int = 5):
    level = list(range(6))[5 - verbosity % 6] * 10
    logging.basicConfig(level=level, format="[%(asctime)s.%(msecs)03d][%(levelname)s][%(name)s] %(message)s",
                        datefmt='%H:%M:%S')
    logging.info('Set logging level to %s (%i)' % (logging.getLevelName(level), level))


def run(method, training_file, testing_file, verbose: int = 0, **kwargs):
    set_logger(verbose)
    logging.info('Loading training file "%s" and testing file "%s"' % (training_file, testing_file))

    # Parse files.
    train_data = np.genfromtxt(training_file)
    test_data = np.genfromtxt(testing_file)

    debug_data = np.array([
        [1, 0.028, 1.31, -6.2],
        [1, 0.07, 0.58, -0.78],
        [1, 1.54, 2.01, -1.63],
        [1, -0.44, 1.18, -4.32],
        [1, -0.81, 0.21, 5.73],
        [1, -1.52, 3.16, 2.77],
        [1, 2.20, 2.42, -0.19],
        [1, 0.91, 1.94, 6.21],
        [2, 0.011, 1.03, -0.21],
        [2, 1.27, 1.28, 0.08],
        [2, 0.13, 3.12, 0.16],
        [2, -0.21, 1.23, -0.11],
        [2, -2.18, 1.39, -0.19],
        [2, 0.34, 1.96, -0.16],
        [2, -1.38, 0.94, 0.45],
        [2, -0.12, 0.82, 0.17],
    ])

    if method == "fixed" or method == "relax":
        classf = perceptron.Perceptron(method)
        classf.train(debug_data[:,1:], debug_data[:,0])
        # classf.train(train_data[:, 1:], train_data[:, 0])
        # classf.test(test_data[:, 1:], train_data[:, 0])
