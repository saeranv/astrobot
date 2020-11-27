import os
import sys
import numpy as np

parent_dir = os.path.abspath(os.path.join(os.getcwd(), '..', '..', 'astrobot'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import utils
fd, pp = utils.fd, utils.pp

SIM_BREAK_NUM = 1000


def logit(x):
    return np.log(x / (1 - x))


def inv_logit(x):
    return 1 / (1 + np.exp(-x))


def draw_samples(mu, sigma, biden_states=None, trump_states=None, states=None,
                 upper_biden=None, lower_biden=None, print_acceptance=False,
                 target_nsim=1000):
    """Draw samples.

    Args:
        mu: mean for mvn distribution.
        sigma: covariance matrix

    TODO: finish description from Drew.
    """
    # TODO: mu is globally defined. Fix.
    sim = np.zeros(len(mu))

    n = 0
    while sim.shape[0] < target_nsim:
        # random sample from posterior distribution and reject based on constraints
        n += 1

        # make multivariate normal distribution w/ 100,000 samples
        mvn = np.random.multivariate_normal(mu, sigma, 1e5)
        proposals = inv_logit(mvn)

        if n == SIM_BREAK_NUM:
            break

    print(n)
    assert False
