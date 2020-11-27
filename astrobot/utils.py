"""
Helpful tricks in jupyter

# reload modules
%load_ext autoreload
%autoreload 2

# Disable warnings
import warnings
warnings.filterwarnings('ignore')

# Set precision
pd.set_option('precision', 2)
np.set_printoptions(precision=2)
"""
import os
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pprint import pprint


def pp(x, *args):
    pprint(x) if not args else print(x, *args)


def fd(module, key=None):
    """ To efficiently search modules in osm"""
    def hfd(m, k): return k.lower() in m.lower()
    if key is None:
        return [m for m in dir(module)][::-1]
    else:
        return [m for m in dir(module) if hfd(m, key)][::-1]
