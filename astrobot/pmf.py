"""Distributions from empirical datasets"""
import numpy as np
import pandas as pd
from typing import Set, Sequence


class Pmf1(object):
    """One-dimensional Probability Mass Function (pmf) for discrete data."""
    # TODO: add bld_idx
    def __init__(self, qs:np.ndarray, ps:np.ndarray, id:np.ndarray, bins:int) -> None:
        """Initialize pmf properties."""
        self.qs_arr = qs if isinstance(qs, np.ndarray) else np.array(qs)
        self.ps_arr = ps if isinstance(ps, np.ndarray) else np.array(ps)
        self.id_arr= np.array(id, dtype=int)
        self.N = np.sum(self.ps_arr)  # total var dataset size
        self.bins = bins
        self._bin_edges = None
        self._bin_mu = None
        self._bin_idx = None
        self._df = None
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
            bin_data = make_bin_data(
                self.qs_arr, bins=self.bins, bin_range=(self.qs_arr.min(), self.qs_arr.max()))
            self._bin_idx, self._bin_mu, self._bin_edges = bin_data
        return self._bin_idx

    @property
    def bin_mu(self):
        """Array of mean qs of bin.

        qs = [0, 1, 3, 1, 2]
        dist = Pmf1(qs, ps, bins=3)
        dist.bin_mu: [0.5, 1.5, 2.5]
        """
        if self._bin_mu is None:
            bin_data = make_bin_data(
                self.qs_arr, bins=self.bins, bin_range=(self.qs_arr.min(), self.qs_arr.max()))
            self._bin_idx, self._bin_mu, self._bin_edges = bin_data
        return self._bin_mu

    @property
    def bin_edges(self):
        """Array of bin_edges.

        qs = [0, 1, 3, 1, 2]
        dist = Pmf1(qs, ps, bins=3)
        dist.bin_edges: [0, 1, 2, 3]
        """
        if self._bin_edges is None:
            bin_data = make_bin_data(
                self.qs_arr, bins=self.bins, bin_range=(self.qs_arr.min(), self.qs_arr.max()))
            self._bin_idx, self._bin_mu, self._bin_edges = bin_data
        return self._bin_edges

    @property
    def df(self):
        if self._df is None:
            self._df = _make_pmf1(self.qs_arr, self.ps_arr, self.bin_idx, self.bin_mu)
        return self._df

def _make_pmf1(qs:np.ndarray, ps:np.ndarray, bin_idx:Sequence, bin_mu:np.ndarray) -> np.ndarray:
    """Pmf1 object from empirical data."""

    N = np.sum(ps)  # total var dataset size
    bins = len(bin_idx)
    binps_arr = [np.sum(ps[bi]) for bi in bin_idx]

    # Create initial dataframe
    bin_df = pd.DataFrame(
        {"bins": range(bins),
         "ps": np.array(binps_arr) / N, # normalize by dataset size
         "qs": bin_mu})

    return bin_df

def make_bin_data(vals:np.ndarray, bins:int, bin_range:Sequence[float], eps=1e-10) -> Sequence[float]:
    """Creates bins from vals."""
    # Get bin_edges
    min_edge, max_edge = bin_range
    bin_inc = (max_edge - min_edge) / bins
    bin_edges = min_edge + np.array([bin_inc * i for i in range(bins + 1)])

    # Get bin_mu
    bin_mu = np.array([np.mean(bin_edges[[i-1, i]])
                       for i in range(1, bins + 1)])

    # Get binnumber
    bin_edges[-1] += eps  # to force last datapoint in
    _in_bin_fx = lambda v, lo, hi: np.where((v >= lo) & (v < hi))[0]
    bin_idx = [_in_bin_fx(vals, bin_edges[i-1], bin_edges[i]).tolist()
               for i in range(1, bins + 1)]

    return bin_idx, bin_mu, bin_edges

def prec(pmf_df, precision:int=2) -> pd.DataFrame:
    """Mutates precisions in Pmf for pretty printing."""
    pmf_df_ = pmf_df.copy()
    pmf_df_.loc[:, "ps"] = np.round(pmf_df_["ps"].values, precision)
    return pmf_df_


import matplotlib.pyplot as plt

def plt_hist(pmf, **kwargs):
    """Plot histogram"""
    if "ax" in kwargs:
        ax = kwargs.pop("ax")
    else:
        _, ax = plt.subplots(1,1)

    ax.bar(pmf.df.qs, height=pmf.df.ps, **kwargs)

    return ax