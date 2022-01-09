"""Distributions from empirical datasets"""
import numpy as np
import pandas as pd
from typing import Sequence, Callable
from pprint import pprint
from functools import reduce


def pp(x, *args):
    pprint(x) if not args else (pprint(x), pp(*args))


class Pmf1(object):
    """One-dimensional Probability Mass Function (pmf) for discrete data."""

    def __init__(self, qs:np.ndarray, bins:int, ps:np.ndarray=None,
                 bin_stat_fx:Callable=None, id:np.ndarray=None) -> None:
        """Initialize pmf properties."""

        # Required args
        self.qs_arr = _seq2arr(qs)
        self.bins = bins
        self._bin_stat = np.mean if bin_stat_fx is None else bin_stat_fx

        # Optional args
        data_num = self.qs_arr.shape[0]
        self.ps_arr = np.ones(data_num) / data_num \
                      if ps is None else _seq2arr(ps)
        self._id_arr = np.arange(data_num, dtype=int) if id is None \
                       else np.array(id, dtype=int)
        self.N = np.sum(self.ps_arr)  # total var dataset size

        # Cached args
        self._bin_edges = None
        self._bin_idx = None
        self._df = None

        # Formatting
        self.prec = 2

    def __repr__(self) -> str:
        return "Pmf1: bins={}, N={}.".format(self.bins, round(self.N, 2))

    @property
    def bin_idx(self):
        """Nested list of qs indices per bin.

        qs = [0, 1, 3, 1, 2]
        dist = Pmf1(qs, ps, bins=3)
        dist.bin_idx: [[0], [1, 3], [2, 4]]
        """
        if self._bin_idx is None:
            self._bin_idx = make_bin_idx(self.qs_arr, self.bin_edges)
        return self._bin_idx

    @property
    def bin_edges(self):
        """Array of bin_edges.

        qs = [0, 1, 3, 1, 2]
        dist = Pmf1(qs, ps, bins=3)
        dist.bin_edges: [0, 1, 2, 3]
        """
        if self._bin_edges is None:
            self._bin_edges = make_bin_edges(
                self.qs_arr, bins=self.bins,
                bin_range=(self.qs_arr.min(), self.qs_arr.max()))
        return self._bin_edges

    @property
    def df(self):
        if self._df is None:
            self._df = _make_pmf1(self.qs_arr, self.ps_arr, self.bin_idx,
                self.bin_edges, self._bin_stat)
        return self._df


def _seq2arr(seq:Sequence) -> np.ndarray:
    """Cast to np.ndarray else throw exception."""
    seq = seq if isinstance(seq, np.ndarray) else np.array(seq)
    assert isinstance(seq, np.ndarray), "Input sequence must be " \
        "np.ndarray, got {}.".format(seq)
    return seq


def _make_pmf1(qs:np.ndarray, ps:np.ndarray, bin_idx:Sequence,
               bin_edges:np.ndarray, bin_stat_fx:Callable) -> np.ndarray:
    """Pmf1 object from empirical data."""

    N = np.sum(ps)  # total var dataset size
    bins = len(bin_idx)

    bin_ps = [np.sum(ps[bi]) for bi in bin_idx]
    bin_mu = [bin_stat_fx(bin_edges[[i-1, i]])
              for i in range(1, bins + 1)]

    # Create initial dataframe
    bin_df = pd.DataFrame(
        {"bins": range(bins),
         "ps": np.array(bin_ps) / N, # normalize by dataset size
         "qs": np.array(bin_mu)})
    bin_df.set_index("bins", inplace=True)
    return bin_df


def make_bin_edges(vals:np.ndarray, bins:int, bin_range:Sequence[float]) -> np.ndarray:
    """Create bin_edges."""
    # Get bin_edges
    min_edge, max_edge = bin_range
    bin_inc = (max_edge - min_edge) / bins
    return min_edge + np.array([bin_inc * i for i in range(bins + 1)])


def make_bin_idx(qs:np.ndarray, bin_edges:np.ndarray, eps=1e-10) -> Sequence:
    """Creates bin_idx, a list of arrays from qs. Arrays are of unequal length."""

    bins = len(bin_edges) - 1
    bin_edges[-1] += eps  # to force last datapoint in

    # Filter by bin_edges
    _in_bin_fx = lambda v, lo, hi: np.where((v >= lo) & (v < hi))
    bin_idx = [_in_bin_fx(qs, bin_edges[i-1], bin_edges[i])
               for i in range(1, bins + 1)]

    # Clean datatype
    return [bi[0].astype(int).tolist() for bi in bin_idx]


def prec(pmf_df: pd.DataFrame, precision:int=2) -> pd.DataFrame:
    """Mutates precisions in Pmf for pretty printing."""
    pmf_df_ = pmf_df.copy()
    pmf_df_.loc[:, "ps"] = np.round(pmf_df_["ps"].values, precision)
    return pmf_df_


# TODO move this to viz
import matplotlib.pyplot as plt

def plt_hist(pmf, **kwargs):
    """Plot histogram"""
    if "ax" in kwargs:
        ax = kwargs.pop("ax")
    else:
        _, ax = plt.subplots(1,1)

    ax.bar(pmf.df.qs, height=pmf.df.ps, **kwargs)

    return ax