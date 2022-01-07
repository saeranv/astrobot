"""Distributions from empirical datasets"""
import numpy as np
import pandas as pd
from scipy.stats._binned_statistic import binned_statistic, BinnedStatisticddResult
from typing import List, Set


class Pmf1(object):
    """One-dimensional Probability Mass Function (pmf) for discrete data."""
    # TODO: add bld_idx
    def __init__(self, qs:np.ndarray, ps:np.ndarray, bins:int) -> None:
        """Initialize pmf properties."""
        self._qs = qs.to_numpy() if isinstance(qs, pd.Series) else qs
        self._ps = ps.to_numpy() if isinstance(ps, pd.Series) else ps
        self.N = np.sum(self._ps)  # total var dataset size
        self.bins = bins
        # TODO: calc bin_edge manually and use make_bin_arr
        self._binstat = binned_statistic(
            self._qs, self._qs, bins=self.bins, statistic="mean",
            range=(self._qs.min(), self._qs.max()))
        self._binnumber = self._binstat.binnumber - 1
        self._df = None
        self.prec = 2


    def __repr__(self) -> str:
        return "Pmf1: bins={}, N={}.".format(self.bins, round(self.N, 2))

    @property
    def df(self):
        if self._df is None:
            _df = _make_pmf1(self._qs, self._ps, self._binstat)
            self._df = _df

        return self._df

# TODO: rewrite this with make_bin_arr
def _make_pmf1(qs:np.ndarray, ps:np.ndarray, binstat: BinnedStatisticddResult) -> np.ndarray:
    """Pmf1 object from empirical data."""

    N = np.sum(ps)  # total var dataset size
    bins = binstat.bin_edges.shape[0] - 1

    # Create initial dataframe
    _df = pd.DataFrame(
        {"bins": binstat.binnumber - 1,
         "ps": ps / N})  # normalize by dataset size

    # Create bin_df before grouping since not all datapoints are in
    # bins, so we need to explicitly create bins that will have 0 prob.
    bin_df = pd.DataFrame({"ps": np.zeros(bins)})
    bin_idx = np.unique(binstat.binnumber - 1) # existing bins
    # Group by bins to create df
    bin_df.loc[bin_idx, :] = _df.groupby("bins").sum()

    # Add statistic
    bin_edges = binstat.bin_edges
    bin_df["qs"] = bin_edges[:-1] + np.diff(bin_edges) / 2.0

    return bin_df

def make_bin_arr(val:np.ndarray, binnumber:np.ndarray) -> List[np.ndarray]:
    """Creates nested list of value array binned to pmf."""
    binnumber_ = binnumber[:]  # make copy
    return [val[np.where(b == binnumber_)] for b in range(binnumber_.max() + 1)]

def make_bin_set(val:np.ndarray, binnumber:np.ndarray) -> List[Set]:
    """Creates nested list of value set binned to pmf."""
    return [set(b) for b in make_bin_arr(val, binnumber)]

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