import os
import sys
import json

from astrobot.utils import *
from astrobot.geomdataframe import GeomDataFrame
from astrobot.r import R
from astrobot import bem_utils
from astrobot import geom_utils
from astrobot import mtx_utils

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as stats
import itertools

import ladybug_geometry.geometry3d as geom3
import ladybug_geometry.geometry2d as geom2
from ladybug_geometry_polyskel import polyskel

# python -m experiments.facade_data.scratch


if __name__ == '__main__':
    print('hello')
