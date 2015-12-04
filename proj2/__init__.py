import logging
import numpy as np
from . import util, core
from . import perceptron, adaboost

logger = logging.getLogger()
# np.set_printoptions(precision=3, suppress=True, linewidth=80)
# np.set_printoptions(linewidth=180)
np.set_printoptions(suppress=True, linewidth=180)

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


def run(method, training_file, testing_file, verbose: int = 0, **kwargs):
    set_logger(verbose)
    logging.info('Loading training file "%s" and testing file "%s"' % (training_file, testing_file))

    # Parse files.
    train_data = np.genfromtxt(training_file)
    test_data = np.genfromtxt(testing_file)

    if method == "fixed" or method == "relax":
        classf = perceptron.Perceptron(method)
        classf.train(train_data[:, 1:], train_data[:, 0])
        # classf.test(test_data[:, 1:], train_data[:, 0])
