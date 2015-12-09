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

LOG_LEVELS = [50, 40, 30, 20, 10, 0]


def set_logger(verbosity: int = 5):
    level = LOG_LEVELS[verbosity]
    logging.basicConfig(level=level, format="[%(asctime)s.%(msecs)03d][%(levelname)s][%(name)s] %(message)s",
                        datefmt='%H:%M:%S')
    logging.info('Set logging level to %s (%i)' % (logging.getLevelName(level), level))


def create_dataset(num_labels: int = 2, count: int = 10, width: float = 2.0, margin: float = 5.0,
                   **kwargs) -> util.narray:
    result = np.zeros((count * num_labels, 3))
    for i in range(num_labels):
        result[i * count:(i + 1) * count, 0] = np.array([i for _ in range(count)])
        result[i * count:(i + 1) * count, 1:] = width * np.random.random((count, 2)) + margin * i
    return result


def debug(cache: bool = False, verbose: int = 0, interactive: bool = False, **kwargs):
    set_logger(verbose)
    ftrain = kwargs.get('training_file', r'bin/debug_train.txt')
    ftest = kwargs.get('testing_file', r'bin/debug_test.txt')

    if ftrain is not None and os.path.exists(ftrain):
        logger.info("Loading training data at <%s>" % ftrain)
        train_data = np.genfromtxt(ftrain)
    else:
        train_data = create_dataset()
        if cache:
            np.savetxt(util.get(ftrain, 'bin/debug_train.txt'), train_data, newline='\n', fmt='%f')

    if ftest is not None and os.path.exists(ftest):
        logger.info("Loading testing data at <%s>" % ftest)
        test_data = np.genfromtxt(ftest)
    else:
        test_data = create_dataset()
        if cache:
            np.savetxt(util.get(ftest, 'bin/debug_test.txt'), test_data, newline='\n', fmt='%f')


def run(method, training_file, testing_file, rule="fixed", strategy="rest", **kwargs):
    set_logger(kwargs.get('verbose', 5))

    # Parse files
    logging.info('Loading files "%s" and "%s"' % (training_file, testing_file))
    train_data = np.genfromtxt(training_file)
    test_data = np.genfromtxt(testing_file)

    classf = None

    if method == "ada":
        adaboost.run(train_data, test_data)
        return

    if classf is None: return
    classf.test(test_data[:, 1:], test_data[:, 0])

def run2(method, training_file, testing_file, rule="fixed", strategy="rest", verbose: int = 0, cache: bool = False,
        **kwargs):
    set_logger(verbose)
    logging.info('Loading files <%s> and <%s>' % (training_file, testing_file))

    # Parse files
    train_data = np.genfromtxt(training_file)
    test_data = np.genfromtxt(testing_file)

    classf = None

    if method == "ada":
        cpath = 'bin/classf_cache/ada_%s_%s.pcl' % (method, os.path.basename(training_file).split('_')[0])

        if cache and os.path.exists(cpath):
            logging.info("Loading classifer at path <%s>" % (cpath))
            classf = adaboost.AdaBoost.load(cpath)
        else:
            if cache:
                logging.info("Coulnd't find classifier in cache, training a new one.")
            classf = adaboost.AdaBoost()
        classf.train(train_data[:, 1:], train_data[:, 0])

        if cache:
            logging.info("Caching Perceptron to <%s>" % (cpath))
            classf.save(cpath)
    elif method == "multi":
        cpath = 'bin/classf_cache/mptron_%s_%s.picl' % (method, os.path.basename(training_file).split('_')[0])

        if cache and os.path.exists(cpath):
            logging.info("Loading classifer at path <%s>" % (cpath))
            classf = perceptron.MulticlassPerceptron.load(cpath)
        else:
            if cache:
                logging.info("Coulnd't find classifier in cache, training a new one.")
            classf = perceptron.MulticlassPerceptron(perceptron.TrainingRule(rule),
                                                     strategy=perceptron.Strategy(strategy))

        classf.train(train_data[:, 1:], train_data[:, 0])

        if cache:
            logging.info("Caching Perceptron to <%s>" % (cpath))
            classf.save(cpath)
    elif method == "single":
        cpath = 'bin/classf_cache/ptron_%s_%s.picl' % (method, os.path.basename(training_file).split('_')[0])

        if cache and os.path.exists(cpath):
            logging.info("Loading classifer at path <%s>" % (cpath))
            classf = perceptron.Perceptron.load(cpath)
        else:
            if cache: logging.info("Coulnd't find classifier in cache, training a new one.")
            classf = perceptron.Perceptron(perceptron.TrainingRule(rule))

        classf.train(train_data[:, 1:], train_data[:, 0])

        if cache:
            logging.info("Caching Perceptron to <%s>" % (cpath))
            classf.save(cpath)

    if classf is None: return
    classf.test(test_data[:, 1:], test_data[:, 0])
