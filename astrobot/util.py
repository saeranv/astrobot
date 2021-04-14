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
from pprint import pprint

# Package to astrobot/ root folder
ROOT_DIR = os.path.abspath(os.path.join(__file__, '..', '..'))

# path to this package from experiments folder
PACKAGE_FPATH_FOR_EXPERIMENTS = os.path.abspath(
    os.path.join(os.getcwd(), '..', '..'))

# path to epw from experiments folder
EPW_FPATH_FOR_EXPERIMENTS = os.path.abspath(
    os.path.join('..', '..', 'resources', 'epw', 'philadelphia', 'philadelphia.epw'))


def pp(x, *args):
    pprint(x) if not args else print(x, *args)


def fd(module, key=None):
    """ To efficiently search modules."""
    def hfd(m, k): return k.lower() in m.lower()
    if key is None:
        return [m for m in dir(module)][::-1]
    else:
        return [m for m in dir(module) if hfd(m, key)][::-1]
